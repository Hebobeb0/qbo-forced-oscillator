import json,csv,zipfile,sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import t as student_t
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'outputs'/'forcing_followup';OUT.mkdir(parents=True,exist_ok=True)

def writecsv(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def accepted(f,j,rule='hysteresis',factor=.25):
    k=int(f['counts'][j]);tt=f['ev'][j,:k];dep=f['dep'][j,:k]
    if rule=='raw':return tt
    threshold=factor*float(f['stats'][j,0]);out=[];mn=0.
    for t,d in zip(tt,dep):
        mn=min(mn,d)
        if mn<=-threshold:out.append(t);mn=0.
    return np.array(out)

def metrics(tt,w):
    n=np.diff(tt)*w/(2*np.pi);theta=tt*w
    near2=np.abs(n-2)<.1;near=np.abs(n-np.rint(n))<.1
    return dict(cycles=len(n),mean_N=n.mean(),median_N=np.median(n),integer_fraction=near.mean(),
      band1=np.mean(np.abs(n-1)<.1),band2=near2.mean(),band3=np.mean(np.abs(n-3)<.1),
      event_R1=abs(np.mean(np.exp(1j*theta))),event_R2=abs(np.mean(np.exp(2j*theta))),
      interval_R=abs(np.mean(np.exp(2j*np.pi*n))),
      next2_given2=np.sum(near2[:-1]&near2[1:])/max(1,np.sum(near2[:-1])),
      event_Omega=2*np.pi/np.mean(np.diff(tt)))

def ci(values):
    a=np.asarray(values);m=a.mean();hw=student_t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a))
    return [float(m),float(m-hw),float(m+hw)]

def main():
    def load(p):
      with np.load(p) as f:return {k:f[k] for k in f.files}
    gs=[load(p) for p in sorted((ROOT/'runs').glob('grid_*.npz'))]
    fs=[load(p) for p in sorted((ROOT/'runs').glob('focal_*.npz'))]
    assert len(gs)==8 and len(fs)==8
    params=gs[0]['params'];rows=[];hists={};bin_edges=np.arange(0,6.00001,.025);centres=(bin_edges[1:]+bin_edges[:-1])/2
    for rule in ['raw','hysteresis']:
      hh=[]
      for f in gs:
        hist=[]
        for j,(r,d,w) in enumerate(params):
          tt=accepted(f,j,rule);mm=metrics(tt,w)
          rows.append(dict(r=r,delta=d,omega=w,seed=int(f['seed']),rule=rule,**mm,Q1=f['stats'][j,4],Q2=f['stats'][j,5],Q3=f['stats'][j,6],phase_drift2=2*f['stats'][j,3]-w))
          hist.append(np.histogram(np.diff(tt)*w/(2*np.pi),bin_edges)[0])
        hh.append(hist)
      hists[rule]=np.array(hh)
    writecsv('grid_metrics.csv',rows)
    # Pool seed histograms for each branch fit; seed bootstrap repeats peak extraction.
    fitrows=[];modedata=[];rng=np.random.default_rng(992)
    def branch(h,indices,bw=1.):
      modes=[];xx=[];last=None
      for j in indices:
        sm=gaussian_filter1d(h[j].astype(float),bw);peaks,_=find_peaks(sm,prominence=.05*sm.max())
        candidates=peaks[(centres[peaks]>1.6)&(centres[peaks]<2.6)]
        if not len(candidates):continue
        # Highest peak first; thereafter nearest to previous branch position.
        pk=candidates[np.argmax(sm[candidates])] if last is None else candidates[np.argmin(abs(centres[candidates]-last))]
        last=centres[pk];modes.append(last);xx.append(params[j,2])
      return np.polyfit(xx,modes,1)[0] if len(xx)>5 else np.nan,np.array(xx),np.array(modes)
    for rule,hh in hists.items():
      for r in [0,.5]:
       for d in [0,1,1.5,2,3,4]:
        idx=np.where((params[:,0]==r)&(params[:,1]==d))[0]
        s,x,y=branch(hh.sum(axis=0),idx)
        bs=[]
        for _ in range(300):
          b=hh[rng.integers(0,len(gs),len(gs))].sum(axis=0);bs.append(branch(b,idx)[0])
        lo,hi=np.nanquantile(bs,[.025,.975]);sens=[branch(hh.sum(axis=0),idx,b)[0] for b in [.6,1,1.6]]
        fitrows.append(dict(rule=rule,r=r,delta=d,slope=s,bootstrap_low=lo,bootstrap_high=hi,slope_bw_015=sens[0],slope_bw_025=sens[1],slope_bw_040=sens[2]))
        for xx,yy in zip(x,y):modedata.append(dict(rule=rule,r=r,delta=d,omega=xx,mode_N=yy))
    writecsv('branch_slopes.csv',fitrows);writecsv('branch_positions.csv',modedata)
    # Independent focal runs supplement the focal cells of the grid: 16 seeds.
    focalrows=[];events={};joint={}
    for r in [0,.5]:
     for d in [0,1,1.5,2,3,4]:
      for rule in ['raw','hysteresis']:
       ee=[]
       for f in gs+fs:
        j=np.where((f['params'][:,0]==r)&(f['params'][:,1]==d)&np.isclose(f['params'][:,2],7/3))[0][0]
        tt=accepted(f,j,rule);mm=metrics(tt,7/3);ee.append(tt)
        focalrows.append(dict(r=r,delta=d,rule=rule,seed=int(f['seed']),**mm,Q2=f['stats'][j,5],phase_drift2=2*f['stats'][j,3]-7/3,residual_variance=f['stats'][j,1]))
       key=f'{r}_{d}_{rule}';events[key]=ee
    writecsv('focal_metrics.csv',focalrows)
    summary=[]
    for r in [0,.5]:
     for d in [0,1,1.5,2,3,4]:
      for rule in ['raw','hysteresis']:
       rs=[a for a in focalrows if a['r']==r and a['delta']==d and a['rule']==rule]
       row=dict(r=r,delta=d,rule=rule,n_seeds=len(rs),total_cycles=sum(a['cycles'] for a in rs))
       for key in ['mean_N','median_N','integer_fraction','band1','band2','band3','event_R1','event_R2','interval_R','next2_given2','Q2','phase_drift2','residual_variance']:
        m,lo,hi=ci([a[key] for a in rs]);row[key]=m;row[key+'_low']=lo;row[key+'_high']=hi
       summary.append(row)
    writecsv('focal_summary.csv',summary)
    # Threshold sensitivity on same focal paths; each seed remains one replicate.
    thresh=[]
    for f in fs:
     for j,(r,d,w) in enumerate(f['params']):
      for fac in [.1,.25,.5]:
       mm=metrics(accepted(f,j,factor=fac),w);thresh.append(dict(r=r,delta=d,seed=int(f['seed']),factor=fac,**mm))
    writecsv('threshold_sensitivity.csv',thresh)
    # Nested Brownian-path convergence, also check sampled .2 crossing definitions.
    conv=[]
    for p in sorted((ROOT/'runs').glob('convergence_*.npz')):
     f=np.load(p)
     for j,(r,d,w) in enumerate(f['params']):
      for rule in ['raw','hysteresis']:
       conv.append(dict(r=r,delta=d,seed=int(f['seed']),dt=float(f['dt']),rule=rule,**metrics(accepted(f,j,rule),w)))
    writecsv('timestep_checks.csv',conv)
    sample=[]
    for f in fs:
     for j,(r,d,w) in enumerate(f['params']):
      for step in [1,4]:
       x=f['trace'][j,::step,0];dt=.05*step;t=float(f['burn'])+float(f['dt'])+np.arange(len(x))*dt
       ix=np.where((x[:-1]<0)&(x[1:]>=0))[0];tt=t[ix]-x[ix]*dt/(x[ix+1]-x[ix])
       deps=np.array([x[max(0,ix[k-1]+1 if k else 0):i+2].min() for k,i in enumerate(ix)])
       fake=dict(counts=np.array([len(tt)]),ev=tt[None,:],dep=deps[None,:],stats=f['stats'][j:j+1])
       for rule in ['raw','hysteresis']:
        sample.append(dict(r=r,delta=d,seed=int(f['seed']),sample_dt=dt,rule=rule,**metrics(accepted(fake,0,rule),w)))
    writecsv('sampling_checks.csv',sample)
    restore=[]
    for p in sorted((ROOT/'runs').glob('restore_*.npz')):
     f=np.load(p);v=f['values'];times=f['times'];r=float(f['r']);d=float(f['delta'])
     # +/- perturbations paired within each initial state; 15 initial states per original seed.
     for k,tt in enumerate(times):
      dist=v[:,:,k,0].ravel();phase=v[:,:,k,1].ravel()
      row=dict(r=r,delta=d,time=tt,median_distance=np.median(dist),distance_q10=np.quantile(dist,.1),distance_q90=np.quantile(dist,.9),median_phase_error=np.median(phase),Q2_perturbed=abs(np.mean(np.exp(1j*v[:,:,k,2]))),Q2_reference=abs(np.mean(np.exp(1j*v[:,:,k,3]))))
      restore.append(row)
    writecsv('restoration.csv',restore)
    writecsv('superposition_checks.csv',json.loads((ROOT/'superposition.json').read_text()))
    np.savez_compressed(ROOT/'analysis_arrays.npz',params=params,centres=centres,raw=hists['raw'],hysteresis=hists['hysteresis'])
    print('FOCAL HYSTERESIS SUMMARY',flush=True)
    for row in summary:
     if row['rule']=='hysteresis' and row['delta'] in [0,2,4]:print({k:round(row[k],4) for k in ['r','delta','mean_N','median_N','integer_fraction','band1','band2','band3','event_R1','Q2','phase_drift2']},flush=True)
    print('BRANCH SLOPES',flush=True)
    for row in fitrows:
     if row['rule']=='hysteresis':print(row,flush=True)
    print('RESTORATION at time 80',flush=True)
    for row in restore:
     if row['time']==80:print(row,flush=True)

if __name__=='__main__':main()
