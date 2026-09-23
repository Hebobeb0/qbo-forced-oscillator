import csv
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
import conclusion_figures as base
from conclusion_figures import Panel,text,ci,Drawing,Group,Rect,HexColor,ROOT,DATA,MODEL,MUTED
from analyse_experiments import accepted

OUT=ROOT.parent/'outputs'/'conclusion_graphs_1_to_3';base.OUT=OUT
DELTAS=[0,1,1.5,2,2.5,3]
COL={0:'#77818d',1:'#7040a0',1.5:'#2563c5',2:'#008e9b',2.5:'#6d902c',3:'#d47b17'}
def read(name):
 with (OUT/name).open() as f:return [{k:v if k=='rule' else float(v) for k,v in a.items()} for a in csv.DictReader(f)]
def legend(p):p.legend([(f'δ = {d:g}',COL[d],d==0,x) for d,x in zip(DELTAS,[20,108,196,284,372,460])])
def main():
 # Existing event records plus freshly simulated delta=2.5 events, all at omega=7/3.
 events={d:[] for d in DELTAS}
 with np.load(DATA/'events.npz') as zz:z={k:zz[k] for k in zz.files}
 par=z['parameters'];off=z['offsets']
 for j,(r,d,w,s) in enumerate(par):
  if r!=.5 or d not in events or not np.isclose(w,7/3):continue
  sl=slice(off[j],off[j+1]);f=dict(counts=np.array([off[j+1]-off[j]]),ev=z['time'][sl][None,:],dep=z['minimum_since_previous'][sl][None,:],stats=z['statistics'][j:j+1]);events[d].append(accepted(f,0))
 for p in sorted((ROOT/'runs').glob('add25_*.npz')):
  with np.load(p) as zz:f={k:zz[k] for k in ['params','counts','ev','dep','stats']}
  for j,(r,d,w) in enumerate(f['params']):
   if r==.5 and np.isclose(w,7/3):events[d].append(accepted(f,j))
 phase={};density={};phase_rows=[];density_rows=[]
 for d in DELTAS:
  assert len(events[d])==16
  hh=[];nn=[]
  for tt in events[d]:
   h,e=np.histogram((tt*(7/3)/(2*np.pi))%1,np.linspace(0,1,49),density=True);hh.append(gaussian_filter1d(h,.7,mode='wrap'))
   h,b=np.histogram(np.diff(tt)*(7/3)/(2*np.pi),np.linspace(0,6,241),density=True);nn.append(gaussian_filter1d(h,1.))
  x=(e[1:]+e[:-1])/2;m,lo,hi=ci(np.array(hh));phase[d]=(x,m,lo,hi)
  phase_rows.extend(dict(delta=d,drive_phase_cycles=a,density=v,ci_low=l,ci_high=h,seeds=16) for a,v,l,h in zip(x,m,lo,hi))
  x=(b[1:]+b[:-1])/2;m,lo,hi=ci(np.array(nn));density[d]=(x,m,lo,hi)
  density_rows.extend(dict(delta=d,N=a,density=v,ci_low=l,ci_high=h,seeds=16) for a,v,l,h in zip(x,m,lo,hi))
 base.write('crossing_phase_density.csv',phase_rows);base.write('interval_density.csv',density_rows)
 a=Panel('A  Crossing phase: δ = 1–3','Five forcing amplitudes, with an unforced reference.',(0,1),(0,3.8),[0,.25,.5,.75,1],[0,1,2,3],'Forcing phase at crossing / 2π','Probability density','r = 0.5; ω = 7/3. Shading: pointwise 95% CI, 16 seeds.')
 for d in DELTAS:
  x,m,lo,hi=phase[d];a.ribbon(x,lo,hi,COL[d]);a.line(x,m,COL[d],[4,3] if d==0 else None)
 a.horizontal(1);legend(a);A=a.finish();base.save(A,'01_crossing_phase')
 f=[a for a in read('focal_summary.csv') if a['r']==.5 and a['rule']=='hysteresis' and a['delta'] in DELTAS];f.sort(key=lambda a:a['delta'])
 b=Panel('B  Interval probabilities: δ = 1–3','Measurements at half-unit intervals between δ = 1 and 3.',(0,3),(0,40),[0,1,1.5,2,2.5,3],[0,10,20,30,40],'Forcing amplitude δ','Fraction of intervals (%)','r = 0.5; ω = 7/3. Bands: |N − n| < 0.1; 95% CI, 16 seeds.')
 cat=[('integer_fraction','Any integer','#374151'),('band1','N ≈ 1','#d5781d'),('band2','N ≈ 2','#2476ad'),('band3','N ≈ 3','#9269a7')];weights=[]
 for key,label,c in cat:
  x=np.array([v['delta'] for v in f]);m=np.array([v[key] for v in f])*100;lo=np.array([v[key+'_low'] for v in f])*100;hi=np.array([v[key+'_high'] for v in f])*100
  b.line(x,m,c,width=2.2 if key=='integer_fraction' else 1.5,marker=True);b.errors(x,lo,hi,c)
  weights.extend(dict(delta=d,band=label,percent=a,ci_low=l,ci_high=h) for d,a,l,h in zip(x,m,lo,hi))
 b.legend([(label,c,False,pos) for (_,label,c),pos in zip(cat,[35,178,302,426])]);B=b.finish();base.save(B,'02_integer_interval_probabilities');base.write('interval_band_probabilities.csv',weights)
 c=Panel('C  Linear-control comparison: δ = 1–3','The new δ = 2.5 point uses simulated branch data.',(0,3),(0,1.12),[0,1,1.5,2,2.5,3],[0,.25,.5,.75,1],'Forcing amplitude δ','Slope of the N ≈ 2 branch, dN/dω','8 seeds; 95% seed-bootstrap CI. Fit range: ω = 1.9–2.6.')
 slopes=[v for v in read('branch_slopes.csv') if v['rule']=='hysteresis' and v['delta'] in DELTAS]
 for r in [0,.5]:
  rs=sorted([v for v in slopes if v['r']==r],key=lambda a:a['delta']);x=[v['delta'] for v in rs];y=[v['slope'] for v in rs]
  c.line(x,y,MODEL[r],[5,3] if r==0 else None,marker=True);c.errors(x,[v['bootstrap_low'] for v in rs],[v['bootstrap_high'] for v in rs],MODEL[r])
 c.horizontal(1,'Unforced reference');c.horizontal(0,'Flat branch');c.legend([('r = 0: exact superposition',MODEL[0],True,60),('r = 0.5',MODEL[.5],False,355)]);C=c.finish();base.save(C,'03_linear_control')
 paths={d:[] for d in DELTAS};drift_rows=[]
 for p in sorted((ROOT/'runs').glob('focal_*.npz'))+sorted((ROOT/'runs').glob('add25_focal_*.npz')):
  with np.load(p) as z:par=z['params'];tr=z['trace']
  t=np.arange(tr.shape[1])*.05;xx=t*(7/3)/(2*np.pi);ids=np.where(xx<=300)[0][::20];x=xx[ids]
  for j,(r,d,w) in enumerate(par):
   if r==.5 and d in paths:
    phi=tr[j,:,2];paths[d].append((2*(phi[ids]-phi[0])-(7/3)*t[ids])/(2*np.pi))
 d=Panel('D  Relative phase across forcing amplitudes','Zero net drift alone is not a sufficient test of locking.',(0,300),(-60,125),[0,100,200,300],[-50,0,50,100],'Elapsed forcing periods','Change in (2φ − ωt) / 2π  [turns]','r = 0.5; ω = 7/3. 8 seeds; shaded pointwise 95% CI; after burn-in.')
 for delta in DELTAS:
  assert len(paths[delta])==8;m,lo,hi=ci(np.array(paths[delta]));assert lo.min()>-60 and hi.max()<125
  d.ribbon(x,lo,hi,COL[delta]);d.line(x,m,COL[delta],[4,3] if delta==0 else None)
  drift_rows.extend(dict(delta=delta,elapsed_drive_periods=a,phase_change_turns=v,ci_low=l,ci_high=h,seeds=8) for a,v,l,h in zip(x,m,lo,hi))
 d.horizontal(0,'Zero net drift');legend(d);D=d.finish();base.save(D,'04_relative_phase_drift');base.write('relative_phase_drift.csv',drift_rows)
 master=Drawing(1200,1080);master.add(Rect(0,0,1200,1080,fillColor=HexColor('#ffffff'),strokeColor=None))
 text(master,45,1040,'Additive forcing: a closer look at δ = 1–3',27,True)
 text(master,45,1010,'δ = 1, 1.5, 2, 2.5 and 3, with δ = 0 retained as the unforced reference.',15,color=MUTED)
 text(master,45,987,'ε = 0.05, σ = 1, integration step = 0.0025. Hysteresis threshold = 0.25 RMS(X).',10,color=MUTED)
 for pic,x0,y0 in [(A,45,520),(B,630,520),(C,45,45),(D,630,45)]:g=Group();g.add(pic);g.translate(x0,y0);master.add(g)
 text(master,45,20,'The δ = 2.5 data come from new simulations. N = cycle interval / forcing period.',9,color=MUTED);base.save(master,'conclusion_summary')
 e=Panel('E  Cycle distributions: δ = 1–3','All five forcing amplitudes are shown on identical axes.',(.5,3.5),(0,2.8),[.5,1,1.5,2,2.5,3,3.5],[0,1,2],'N = cycle interval / forcing period','Probability density','r = 0.5; ω = 7/3. Shading: pointwise 95% CI, 16 seeds.')
 for d in DELTAS:
  xx,m,lo,hi=density[d];sel=(xx>=.5)&(xx<=3.5);e.ribbon(xx[sel],lo[sel],hi[sel],COL[d]);e.line(xx[sel],m[sel],COL[d],[4,3] if d==0 else None)
 for n in [1,2,3]:e.vertical(n)
 legend(e);base.save(e.finish(),'05_interval_distributions')
 print('All revised graphs saved.')
if __name__=='__main__':main()
