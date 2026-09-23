import csv,json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from run_experiments import ROOT,run
from analyse_experiments import accepted,metrics,ci
OUT=ROOT.parent/'outputs'/'conclusion_graphs_1_to_3';OUT.mkdir(exist_ok=True,parents=True)
OLD=ROOT.parent/'outputs'/'forcing_followup'
def load(path):
 with np.load(path) as f:return {k:f[k] for k in f.files}
def write(name,rows):
 with (OUT/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def read(name):
 with (OLD/name).open() as f:return [{k:v if k=='rule' else float(v) for k,v in a.items()} for a in csv.DictReader(f)]
def main():
 ww=np.unique(np.r_[np.round(np.arange(1.9,2.60001,.025),8),7/3,1+np.sqrt(2)])
 par=np.array([(r,2.5,w) for r in [0,.5] for w in ww])
 for seed in range(101,109):
  dest=ROOT/'runs'/f'add25_grid_{seed}.npz'
  if not dest.exists():np.savez_compressed(dest,**run(par,seed))
  print('New amplitude grid',seed,flush=True)
 for seed in range(201,209):
  dest=ROOT/'runs'/f'add25_focal_{seed}.npz'
  if not dest.exists():np.savez_compressed(dest,**run(np.array([[0,2.5,7/3],[.5,2.5,7/3]]),seed,trace=True))
 grids=[load(ROOT/'runs'/f'add25_grid_{s}.npz') for s in range(101,109)]
 focal=[load(ROOT/'runs'/f'add25_focal_{s}.npz') for s in range(201,209)]
 summary=[a for a in read('focal_summary.csv') if a['delta']<=3]
 keys=['mean_N','median_N','integer_fraction','band1','band2','band3','event_R1','event_R2','interval_R','next2_given2','Q2','phase_drift2','residual_variance']
 for r in [0,.5]:
  for rule in ['raw','hysteresis']:
   vals=[]
   for f in grids+focal:
    j=np.where((f['params'][:,0]==r)&np.isclose(f['params'][:,2],7/3))[0][0]
    mm=metrics(accepted(f,j,rule),7/3);mm.update(Q2=f['stats'][j,5],phase_drift2=2*f['stats'][j,3]-7/3,residual_variance=f['stats'][j,1]);vals.append(mm)
   row=dict(r=r,delta=2.5,rule=rule,n_seeds=16,total_cycles=sum(a['cycles'] for a in vals))
   for key in keys:row[key],row[key+'_low'],row[key+'_high']=ci([a[key] for a in vals])
   summary.append(row)
 write('focal_summary.csv',sorted(summary,key=lambda a:(a['r'],a['rule'],a['delta'])))
 slopes=[a for a in read('branch_slopes.csv') if a['delta']<=3]
 edges=np.arange(0,6.00001,.025);centres=(edges[1:]+edges[:-1])/2;rng=np.random.default_rng(99225)
 def branch(h,ids,bw=1.):
  last=None;x=[];y=[]
  for j in ids:
   sm=gaussian_filter1d(h[j].astype(float),bw);pk,_=find_peaks(sm,prominence=.05*sm.max());pk=pk[(centres[pk]>1.6)&(centres[pk]<2.6)]
   if not len(pk):continue
   k=pk[np.argmax(sm[pk])] if last is None else pk[np.argmin(abs(centres[pk]-last))]
   last=centres[k];x.append(par[j,2]);y.append(last)
  return np.polyfit(x,y,1)[0]
 for rule in ['raw','hysteresis']:
  hh=np.array([[np.histogram(np.diff(accepted(f,j,rule))*par[j,2]/(2*np.pi),edges)[0] for j in range(len(par))] for f in grids])
  for r in [0,.5]:
   ids=np.where(par[:,0]==r)[0];h=hh.sum(axis=0);bs=[branch(hh[rng.integers(0,8,8)].sum(axis=0),ids) for _ in range(300)];lo,hi=np.quantile(bs,[.025,.975])
   slopes.append(dict(rule=rule,r=r,delta=2.5,slope=branch(h,ids),bootstrap_low=lo,bootstrap_high=hi,slope_bw_015=branch(h,ids,.6),slope_bw_025=branch(h,ids),slope_bw_040=branch(h,ids,1.6)))
 write('branch_slopes.csv',sorted(slopes,key=lambda a:(a['rule'],a['r'],a['delta'])))
 checks=[]
 for f in grids:
  ids=np.where(par[:,0]==0)[0];pp=par[ids].copy();pp[:,0]=-1;b=run(pp,int(f['seed']))
  for jj,j in enumerate(ids):
   for rule in ['raw','hysteresis']:
    t=accepted(f,j,rule);u=accepted(b,jj,rule);assert len(t)==len(u);err=np.max(abs(t-u));assert err<1e-7
    checks.append(dict(seed=int(f['seed']),delta=2.5,omega=par[j,2],rule=rule,crossings=len(t),max_event_time_error=err))
 write('delta_2p5_superposition_checks.csv',checks)
 (OUT/'new_simulations.json').write_text(json.dumps(dict(delta=2.5,grid_runs=496,focal_runs=16,grid_seeds=list(range(101,109)),focal_seeds=list(range(201,209)),dt=.0025,T=4000,burn=1000,epsilon=.05,sigma=1,maximum_superposition_event_error=max(a['max_event_time_error'] for a in checks)),indent=2))
 print('Amplitude extension and validation complete.',flush=True)
if __name__=='__main__':main()
