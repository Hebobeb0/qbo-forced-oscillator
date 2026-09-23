import os,ctypes,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
os.add_dll_directory('C:/msys64/ucrt64/bin')
lib=ctypes.CDLL(str(ROOT/'forcing.dll'))
P=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
I=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
lib.sweep.argtypes=[ctypes.c_int]*3+[ctypes.c_double,P,P,P,P,ctypes.c_int,P,P,P,I,P,ctypes.c_int,ctypes.c_int,P,ctypes.c_int,ctypes.c_void_p,ctypes.c_int]
lib.superposition.argtypes=[ctypes.c_int,ctypes.c_double,P,ctypes.c_double,ctypes.c_double,P]
lib.perturb.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_double,P,P,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int,I,P]

def run(params,seed,dt=.0025,T=4000,burn=1000,trace=False,noise=None):
    params=np.asarray(params,float);nc=len(params);ns=round(T/dt);ib=round(burn/dt);cap=5000
    if noise is None:noise=np.random.default_rng(seed).normal(0,np.sqrt(dt),ns)
    rr,dd,ww=[np.ascontiguousarray(params[:,i]) for i in range(3)]
    ev=np.zeros((nc,cap));dep=ev.copy();vel=ev.copy();counts=np.zeros(nc,np.int32);stats=np.zeros((nc,8))
    stride=round(50/dt);n_snap=(ns-ib-1)//stride+1;snap=np.zeros((nc,n_snap,3))
    thin=round(.05/dt);n_trace=(ns-ib-1)//thin+1
    tr=np.zeros((nc,n_trace,3)) if trace else None
    lib.sweep(nc,ns,ib,dt,noise,rr,dd,ww,cap,ev,dep,vel,counts,stats,stride,n_snap,snap,thin,None if tr is None else tr.ctypes.data,n_trace)
    assert counts.max()<cap and np.isfinite(stats).all()
    return dict(params=params,seed=seed,dt=dt,T=T,burn=burn,ev=ev,dep=dep,vel=vel,counts=counts,stats=stats,snap=snap,**({} if tr is None else {'trace':tr}))

def main():
    (ROOT/'runs').mkdir(exist_ok=True)
    ww=np.unique(np.r_[np.round(np.arange(1.9,2.60001,.025),8),7/3,1+np.sqrt(2)])
    params=np.array([(r,d,w) for r in [0,.5] for d in [0,1,1.5,2,3,4] for w in ww])
    for seed in range(101,109):
        dest=ROOT/'runs'/f'grid_{seed}.npz'
        if dest.exists():continue
        start=time.time();data=run(params,seed);np.savez_compressed(dest,**data)
        print(f'Grid seed {seed}: {len(params)} trajectories; {data["counts"].sum()} crossings; {time.time()-start:.1f}s',flush=True)
    # Independent replication at focal frequency, with saved short-interval trajectories.
    focal=np.array([(r,d,7/3) for r in [0,.5] for d in [0,1,1.5,2,3,4]])
    for seed in range(201,209):
        dest=ROOT/'runs'/f'focal_{seed}.npz'
        if dest.exists():continue
        data=run(focal,seed,trace=True);np.savez_compressed(dest,**data)
        print(f'Focal seed {seed} complete',flush=True)
    # Same-Brownian-path time-step refinement; keep Brownian increments nested.
    checks=np.array([(r,d,7/3) for r in [0,.5] for d in [0,2,4]])
    for seed in range(301,305):
        fine_dt=.00125;noise=np.random.default_rng(seed).normal(0,np.sqrt(fine_dt),round(4000/fine_dt))
        for factor in [1,2,4]:
            dest=ROOT/'runs'/f'convergence_{seed}_{factor}.npz'
            if dest.exists():continue
            z=np.ascontiguousarray(noise.reshape(-1,factor).sum(axis=1));data=run(checks,seed,dt=fine_dt*factor,noise=z)
            np.savez_compressed(dest,**data)
        print(f'Nested step-size check {seed} complete',flush=True)
    # Direct versus unforced + deterministic numerical solution.
    rows=[]
    for dt in [.005,.0025,.00125]:
        noise=np.random.default_rng(991).normal(0,np.sqrt(dt),round(1400/dt))
        for w in [1.9,7/3,2.6]:
            for d in [1,2,4]:
                out=np.zeros(2);lib.superposition(len(noise),dt,noise,d,w,out)
                rows.append(dict(dt=dt,omega=w,delta=d,max_state_error=out[0],max_continuous_mean_error=out[1]))
    (ROOT/'superposition.json').write_text(json.dumps(rows,indent=2))
    # Initial conditions sampled from eight independent focal runs.
    fs=[np.load(ROOT/'runs'/f'focal_{s}.npz') for s in range(201,209)]
    times=np.array([0,1,2,5,10,20,40,80,120,160.]);dt=.0025;ns=round(times[-1]/dt);at=np.rint(times/dt).astype(np.int32)
    for r in [0,.5]:
        for d in [0,1.5,2,4]:
            dest=ROOT/'runs'/f'restore_{r}_{d}.npz'
            if dest.exists():continue
            j=np.where((focal[:,0]==r)&(focal[:,1]==d))[0][0]
            initial=np.concatenate([f['snap'][j,::4] for f in fs]) # 120 initial states
            nt=len(initial);noise=np.random.default_rng(879).normal(0,np.sqrt(dt),(nt,ns))
            vals=[]
            for alpha in [.2,-.2]:
                out=np.zeros((nt,len(times),4));lib.perturb(nt,ns,dt,noise,np.ascontiguousarray(initial),r,d,7/3,alpha,len(at),at,out);vals.append(out)
            np.savez_compressed(dest,values=np.array(vals),times=times,initial=initial,r=r,delta=d)
            print(f'Restoration r={r}, delta={d}: {2*nt} perturbations complete',flush=True)
    print('ALL EXPERIMENTS COMPLETE',flush=True)

if __name__=='__main__':main()
