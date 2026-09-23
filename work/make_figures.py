from pathlib import Path
import csv,sys
import numpy as np
from scipy.ndimage import gaussian_filter1d
from PIL import Image,ImageDraw,ImageFont
from analyse_experiments import accepted
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent/'outputs'/'forcing_followup'
COL={0:'#64748b',1:'#2563eb',2:'#d97706',4:'#dc2626'}
def font(n,bold=False):return ImageFont.truetype('C:/Windows/Fonts/'+('segoeuib.ttf' if bold else 'segoeui.ttf'),n)
def rows(name):
 with (OUT/name).open() as f:return [{k:(v if k=='rule' else float(v)) for k,v in a.items()} for a in csv.DictReader(f)]
class Panel:
 def __init__(self,draw,rect,xlim,ylim,title,xlabel,ylabel,xticks,yticks,log=False):
  self.d=draw;self.rect=rect;self.xlim=xlim;self.ylim=ylim;self.log=log
  l,t,r,b=rect;draw.text((l,t-68),title,font=font(25,True),fill='#0f172a')
  for x in xticks:
   px,_=self.xy(x,ylim[0]);draw.line((px,t,px,b),fill='#e2e8f0',width=1);draw.text((px,b+8),f'{x:g}',font=font(18),fill='#475569',anchor='mt')
  for y in yticks:
   _,py=self.xy(xlim[0],y);draw.line((l,py,r,py),fill='#e2e8f0',width=1);draw.text((l-10,py),f'{y:g}',font=font(18),fill='#475569',anchor='rm')
  draw.line((l,t,l,b,r,b),fill='#64748b',width=2)
  draw.text(((l+r)/2,b+42),xlabel,font=font(20),fill='#334155',anchor='mt')
  draw.text((l,t-18),ylabel,font=font(17),fill='#475569',anchor='lb')
 def xy(self,x,y):
  l,t,r,b=self.rect;a,z=self.ylim
  if self.log:y=np.log10(max(y,1e-15));a=np.log10(a);z=np.log10(z)
  return (l+(x-self.xlim[0])/(self.xlim[1]-self.xlim[0])*(r-l),b-(y-a)/(z-a)*(b-t))
 def line(self,x,y,color,width=3,dash=False):
  pts=[self.xy(a,min(self.ylim[1],max(self.ylim[0],b))) for a,b in zip(x,y) if self.xlim[0]<=a<=self.xlim[1]]
  for i in range(len(pts)-1):
   if not dash or i%4<2:self.d.line((*pts[i],*pts[i+1]),fill=color,width=width)
 def hline(self,y,color='#94a3b8'):
  l,t,r,b=self.rect;_,py=self.xy(self.xlim[0],y)
  for x in range(l,r,14):self.d.line((x,py,min(r,x+7),py),fill=color,width=2)
 def legend(self,items,x=None,y=None):
  l,t,r,b=self.rect;x=l+15 if x is None else x;y=t+12 if y is None else y
  for label,color,dashed in items:
   self.d.rectangle((x-5,y-3,x+260,y+26),fill='white')
   for z in range(0,36,12):self.d.line((x+z,y+12,x+z+(7 if dashed else 12),y+12),fill=color,width=3)
   self.d.text((x+46,y),label,font=font(18),fill='#334155');y+=28

def main():
 im=Image.new('RGB',(1660,1220),'white');dr=ImageDraw.Draw(im)
 dr.text((75,32),'Additive forcing: branches, crossing phases and recovery',font=font(37,True),fill='#0f172a')
 dr.text((75,88),'Fresh simulations | ε = 0.05, σ = 1 | integration step 0.0025 | 8 grid seeds, 16 focal seeds',font=font(23),fill='#475569')
 p=Panel(dr,(100,205,785,565),(1.9,2.6),(1.6,2.7),'A  Cycle-distribution branch','Forcing angular frequency ω','Mode of N = interval / drive period',[1.9,2,2.2,2.4,2.6],[1.6,1.8,2,2.2,2.4,2.6])
 p.hline(2);branches=rows('branch_positions.csv')
 for r in [0,.5]:
  for delta in [1,2,4]:
   a=[v for v in branches if v['r']==r and v['delta']==delta and v['rule']=='hysteresis'];p.line([v['omega'] for v in a],[v['mode_N'] for v in a],COL[delta],3,r==0)
 p.legend([('δ = 1',COL[1],False),('δ = 2',COL[2],False),('δ = 4',COL[4],False)])
 dr.text((405,535),'solid r = 0.5; dashed r = 0',font=font(18),fill='#475569')
 files=[]
 for path in sorted((ROOT/'runs').glob('grid_*.npz'))+sorted((ROOT/'runs').glob('focal_*.npz')):
  with np.load(path) as f:files.append({k:f[k] for k in ['params','ev','dep','counts','stats']})
 events={}
 for r in [0,.5]:
  for delta in [0,2,4]:
   ts=[]
   for f in files:
    j=np.where((f['params'][:,0]==r)&(f['params'][:,1]==delta)&np.isclose(f['params'][:,2],7/3))[0][0];ts.append(accepted(f,j))
   events[r,delta]=ts
 p=Panel(dr,(930,205,1585,565),(.5,3.5),(0,3.2),'B  Cycle intervals at ω = 7/3','N = interval / drive period','Probability density (hysteresis, r = 0.5)',[.5,1,1.5,2,2.5,3,3.5],[0,1,2,3])
 for delta in [0,2,4]:
  n=np.concatenate([np.diff(t)*7/3/(2*np.pi) for t in events[.5,delta]]);bins=np.arange(0,6.00001,.025);h,_=np.histogram(n,bins,density=True);p.line((bins[1:]+bins[:-1])/2,gaussian_filter1d(h,1),COL[delta])
 p.legend([(f'δ = {x}',COL[x],False) for x in [0,2,4]],x=1300)
 p=Panel(dr,(100,770,785,1120),(0,1),(0,4.5),'C  Preferred drive phases of crossings','Drive phase at crossing / 2π','Probability density (hysteresis, r = 0.5)',[0,.25,.5,.75,1],[0,1,2,3,4])
 p.hline(1)
 for delta in [0,2,4]:
  ph=np.concatenate([t*7/3/(2*np.pi)%1 for t in events[.5,delta]]);h,e=np.histogram(ph,np.linspace(0,1,49),density=True);p.line((e[1:]+e[:-1])/2,gaussian_filter1d(h,.7,mode='wrap'),COL[delta])
 p.legend([(f'δ = {x}',COL[x],False) for x in [0,2,4]])
 p=Panel(dr,(930,770,1585,1120),(0,160),(.001,1.1),'D  Recovery with matched future noise','Time after ±0.2 rad phase perturbation','Median state separation / initial separation',[0,40,80,120,160],[.001,.01,.1,1],log=True)
 rr=rows('restoration.csv')
 for delta in [0,2,4]:
  a=[v for v in rr if v['r']==.5 and v['delta']==delta];p.line([v['time'] for v in a],[v['median_distance'] for v in a],COL[delta])
 tx=np.linspace(0,160,161);p.line(tx,np.exp(-.025*tx),'#111827',2,True)
 p.legend([('r = 0: damping envelope','#111827',True)]+[(f'r = 0.5, δ = {x}',COL[x],False) for x in [0,2,4]],x=950,y=985)
 im.save(OUT/'findings.png')
 # Gating map, split by r to expose the linear control.
 im=Image.new('RGB',(1660,820),'white');dr=ImageDraw.Draw(im)
 dr.text((75,28),'Crossing phase and subsequent interval: δ = 4, ω = 7/3',font=font(37,True),fill='#0f172a')
 dr.text((75,86),'Hysteresis crossings | same fixed colour scale | pooled over 16 independent simulation seeds',font=font(23),fill='#475569')
 for ir,r in enumerate([0,.5]):
  rect=(100+ir*820,205,765+ir*820,665)
  p=Panel(dr,rect,(0,1),(0,4),f'r = {r:g}','Drive phase at starting crossing / 2π','Next interval N',[0,.25,.5,.75,1],[0,1,2,3,4])
  ts=events[r,4];x=np.concatenate([t[:-1]*7/3/(2*np.pi)%1 for t in ts]);y=np.concatenate([np.diff(t)*7/3/(2*np.pi) for t in ts]);h,xe,ye=np.histogram2d(x,y,[np.linspace(0,1,41),np.linspace(0,4,81)]);h=h/h.sum()
  # Bin probability mapped logarithmically, 0 to 0.01; data are not interpolated.
  for i in range(h.shape[0]):
   for j in range(h.shape[1]):
    q=min(1,np.log1p(h[i,j]/.0001)/np.log1p(.01/.0001));color=tuple(int(255*(1-q)+c*q) for c in (30,64,175));a,b=p.xy(xe[i],ye[j+1]);c,d=p.xy(xe[i+1],ye[j]);dr.rectangle((a,b,c,d),fill=color)
  for n in [1,2,3]:p.hline(n,'#dc2626')
 dr.text((100,754),'White = no events; dark blue = ≥1% of events per bin. Red guides mark integer drive-period intervals.',font=font(22),fill='#475569')
 im.save(OUT/'crossing_phase_intervals.png')
 print('Figures saved')
if __name__=='__main__':main()
