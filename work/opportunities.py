import numpy as np
from analyse_experiments import ROOT,OUT,accepted,writecsv,ci
def main():
 rows=[]
 for p in sorted((ROOT/'runs').glob('grid_*.npz'))+sorted((ROOT/'runs').glob('focal_*.npz')):
  with np.load(p) as f:a={k:f[k] for k in ['params','seed','counts','ev','dep','stats']}
  for r in [0,.5]:
   for d in [0,2,4]:
    j=np.where((a['params'][:,0]==r)&(a['params'][:,1]==d)&np.isclose(a['params'][:,2],7/3))[0][0]
    tt=accepted(a,j);period=2*np.pi/(7/3);lo=int(np.ceil(1000/period));hi=int(np.floor(4000/period))
    ix=np.floor(tt/period).astype(int);ix=ix[(ix>=lo)&(ix<hi)];count=np.bincount(ix-lo,minlength=hi-lo)
    n=np.diff(tt)/period;k=np.rint(n).astype(int);near2=abs(n-2)<.1;run=0;longest=0
    for good in near2:
     run=run+1 if good else 0;longest=max(longest,run)
    rows.append(dict(r=r,delta=d,seed=int(a['seed']),complete_drive_periods=hi-lo,empty_fraction=np.mean(count==0),one_fraction=np.mean(count==1),multiple_fraction=np.mean(count>=2),longest_near2_sequence=longest))
 writecsv('drive_period_opportunities.csv',rows)
 for r in [.5]:
  for d in [0,2,4]:
   a=[x for x in rows if x['r']==r and x['delta']==d]
   print(r,d,{k:ci([x[k] for x in a]) for k in ['empty_fraction','one_fraction','multiple_fraction']},'longest',max(x['longest_near2_sequence'] for x in a))
if __name__=='__main__':main()
