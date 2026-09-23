#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#define API __declspec(dllexport)

/* Semi-implicit Ito Euler: diffusion at old X, V first, then X.
   Raw upcrossings retain the minimum since the previous upcrossing so that
   Schmitt thresholds can be applied after measuring the complete-run RMS. */
API void sweep(int nc,int ns,int burn,double dt,const double *noise,
 const double *rr,const double *dd,const double *ww,int cap,
 double *ev,double *dep,double *vel,int *counts,double *stats,
 int snapstride,int n_snap,double *snap,int thin,double *trace,int n_trace) {
 for(int j=0;j<nc;j++) {
  double x=1,v=0,mn=1,ss=0,sy=0,sv=0,base_x=1,base_v=0,mean_x=0,mean_v=0;
  double om=ww[j],delta=dd[j],r=rr[j],a=1-om*om,b=.05*om;
  double den=a*a+b*b,sn=0,cs=1,sd=sin(om*dt),cd=cos(om*dt);
  double qre[3]={0},qim[3]={0},phase0=0,prev=atan2(-v,x),unw=prev;
  int k=0,kp=0,kt=0;
  for(int i=0;i<ns;i++) {
   double old=x;
   if(r<0){ /* Negative r is a test-mode flag, not a physical parameter. */
    base_v+=(-base_x-.05*base_v)*dt+sqrt(.05)*noise[i];base_x+=base_v*dt;
    mean_v+=(-mean_x-.05*mean_v+delta*sn)*dt;mean_x+=mean_v*dt;
    x=base_x+mean_x;v=base_v+mean_v;
   }else{
    v+=(-x-.05*v+delta*sn)*dt+sqrt(.05*(1+2*r*x*x))*noise[i];
    x+=v*dt;
   }
   double z=sn*cd+cs*sd;cs=cs*cd-sn*sd;sn=z;
   if(i%100000==99999){sn=sin(om*(i+1)*dt);cs=cos(om*(i+1)*dt);}
   if(i<burn)continue;
   if(i==burn){mn=fmin(old,x);prev=atan2(-v,x);unw=prev;phase0=prev;}
   mn=fmin(mn,x);
   if(old<0 && x>=0) {
    if(k<cap){ev[j*cap+k]=(i+(-old)/(x-old))*dt;dep[j*cap+k]=mn;vel[j*cap+k]=v;}
    k++;mn=x;
   }
   ss+=x*x;sv+=v*v;
   double m=delta*(a*sn-b*cs)/den;sy+=(x-m)*(x-m);
   if((i-burn)%thin==0) {
    double ph=atan2(-v,x),inc=remainder(ph-prev,2*M_PI);unw+=inc;prev=ph;
    for(int n=1;n<=3;n++){double p=n*ph-om*(i+1)*dt;qre[n-1]+=cos(p);qim[n-1]+=sin(p);}
    if(trace && kt<n_trace){trace[(j*n_trace+kt)*3]=x;trace[(j*n_trace+kt)*3+1]=v;trace[(j*n_trace+kt)*3+2]=unw;}
    kt++;
   }
   if((i-burn)%snapstride==0 && kp<n_snap){snap[(j*n_snap+kp)*3]=x;snap[(j*n_snap+kp)*3+1]=v;snap[(j*n_snap+kp)*3+2]=(i+1)*dt;kp++;}
  }
  counts[j]=k;stats[j*8]=sqrt(ss/(ns-burn));stats[j*8+1]=sy/(ns-burn);
  stats[j*8+2]=sv/(ns-burn);stats[j*8+3]=(unw-phase0)/((ns-burn)*dt);
  for(int n=0;n<3;n++)stats[j*8+4+n]=hypot(qre[n],qim[n])/kt;
  stats[j*8+7]=ss/(ns-burn);
 }
}

API void superposition(int ns,double dt,const double *noise,double delta,double om,double *out) {
 double x=1,v=0,y=1,w=0,m=0,u=0,err=0,econt=0;
 double a=1-om*om,b=.05*om,den=a*a+b*b;
 /* all components begin with matching initial data; compare continuous mean
    only after burn-in, when its deterministic transient has decayed. */
 for(int i=0;i<ns;i++){
  double f=delta*sin(om*i*dt);
  v+=(-x-.05*v+f)*dt+sqrt(.05)*noise[i];x+=v*dt;
  w+=(-y-.05*w)*dt+sqrt(.05)*noise[i];y+=w*dt;
  u+=(-m-.05*u+f)*dt;m+=u*dt;
  err=fmax(err,fmax(fabs(x-y-m),fabs(v-w-u)));
  if(i*dt>1000){double t=(i+1)*dt;double mc=delta*(a*sin(om*t)-b*cos(om*t))/den;econt=fmax(econt,fabs(x-y-mc));}
 }
 out[0]=err;out[1]=econt;
}

API void perturb(int nt,int ns,double dt,const double *noise,const double *initial,
 double r,double delta,double om,double alpha,int nk,const int *at,double *out) {
 for(int j=0;j<nt;j++) {
  double x=initial[3*j],v=initial[3*j+1],t0=initial[3*j+2];
  double xp=x*cos(alpha)+v*sin(alpha),vp=v*cos(alpha)-x*sin(alpha);
  double d0=hypot(xp-x,vp-v),sn=sin(om*t0),cs=cos(om*t0),sd=sin(om*dt),cd=cos(om*dt);
  int k=0;
  for(int i=0;i<=ns;i++) {
   if(k<nk && i==at[k]) {
    double *o=out+(j*nk+k)*4;
    o[0]=hypot(xp-x,vp-v)/d0;
    o[1]=fabs(remainder(atan2(-vp,xp)-atan2(-v,x),2*M_PI));
    o[2]=remainder(2*atan2(-vp,xp)-om*(t0+i*dt),2*M_PI);
    o[3]=remainder(2*atan2(-v,x)-om*(t0+i*dt),2*M_PI);k++;
   }
   if(i==ns)break;
   double z=noise[j*ns+i],f=delta*sn;
   v+=(-x-.05*v+f)*dt+sqrt(.05*(1+2*r*x*x))*z;x+=v*dt;
   vp+=(-xp-.05*vp+f)*dt+sqrt(.05*(1+2*r*xp*xp))*z;xp+=vp*dt;
   double s=sn*cd+cs*sd;cs=cs*cd-sn*sd;sn=s;
  }
 }
}
