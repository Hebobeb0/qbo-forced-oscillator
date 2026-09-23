import json
import numpy as np
from scipy.linalg import expm
from run_experiments import ROOT,run
def main():
 par=np.array([[0,2,7/3],[.5,4,2.4]]);dt=.0025;z=np.random.default_rng(42).normal(0,np.sqrt(dt),800)
 f=run(par,42,dt,T=2,burn=0,trace=True,noise=z);x=np.ones(2);v=np.zeros(2);trace=[]
 for i in range(800):
  v+=(-x-.05*v+par[:,1]*np.sin(par[:,2]*i*dt))*dt+np.sqrt(.05*(1+2*par[:,0]*x*x))*z[i];x+=v*dt
  if i%20==0:trace.append(np.column_stack([x.copy(),v.copy()]))
 a=np.array(trace).transpose(1,0,2);err=float(np.max(abs(a-f['trace'][:,:,:2])));assert err<1e-10
 mx=0.;M=np.array([[0.,1.],[-1.,-.05]])
 for p in (ROOT/'runs').glob('restore_0_*.npz'):
  f=np.load(p);initial=f['initial'][:,:2];vv=f['values'];times=f['times']
  for ia,alpha in enumerate([.2,-.2]):
   R=np.array([[np.cos(alpha),np.sin(alpha)],[-np.sin(alpha),np.cos(alpha)]]);diff=initial@(R-np.eye(2)).T
   for k,t in enumerate(times):
    pred=np.linalg.norm(diff@expm(M*t).T,axis=1)/np.linalg.norm(diff,axis=1);mx=max(mx,float(np.max(abs(pred-vv[ia,:,k,0]))))
 assert mx<.002
 results=dict(independent_numpy_max_state_error=err,r0_restoration_max_normalized_separation_error=mx)
 (ROOT.parent/'outputs'/'forcing_followup'/'numerical_validation.json').write_text(json.dumps(results,indent=2));print(results)
if __name__=='__main__':main()
