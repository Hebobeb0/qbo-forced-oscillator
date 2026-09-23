from run_experiments import ROOT,run
from analyse_experiments import accepted,writecsv
import numpy as np
def main():
 rows=[]
 for seed in range(101,109):
  with np.load(ROOT/'runs'/f'grid_{seed}.npz') as f:a={k:f[k] for k in f.files}
  ids=np.where(a['params'][:,0]==0)[0];par=a['params'][ids].copy();par[:,0]=-1
  b=run(par,seed)
  for j,k in enumerate(ids):
   for rule in ['raw','hysteresis']:
    t=accepted(a,k,rule);u=accepted(b,j,rule)
    assert len(t)==len(u),(seed,j,rule,len(t),len(u))
    err=np.max(abs(t-u));assert err<1e-7
    rows.append(dict(seed=seed,delta=par[j,1],omega=par[j,2],rule=rule,crossings=len(t),max_event_time_error=err,rms_error=abs(a['stats'][k,0]-b['stats'][j,0])))
  print('Full-grid superposition verified',seed,flush=True)
 writecsv('superposition_event_checks.csv',rows)
 print('Maximum event-time difference',max(r['max_event_time_error'] for r in rows),flush=True)
if __name__=='__main__':main()
