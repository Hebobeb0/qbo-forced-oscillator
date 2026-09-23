import json,zipfile,hashlib,platform,importlib.metadata as md
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent/'outputs'/'forcing_followup'

def main():
    # Compact ragged raw crossing records, in run/parameter order.
    meta=[];times=[];depth=[];velocity=[];offset=[0];stat=[]
    for p in sorted((ROOT/'runs').glob('grid_*.npz'))+sorted((ROOT/'runs').glob('focal_*.npz')):
      with np.load(p) as f:
        a={k:f[k] for k in ['params','seed','counts','ev','dep','vel','stats']}
      for j,(r,d,w) in enumerate(a['params']):
        k=int(a['counts'][j]);meta.append([r,d,w,int(a['seed'])]);stat.append(a['stats'][j]);times.append(a['ev'][j,:k]);depth.append(a['dep'][j,:k]);velocity.append(a['vel'][j,:k]);offset.append(offset[-1]+k)
    np.savez_compressed(OUT/'events.npz',parameters=np.array(meta),offsets=np.array(offset,dtype=np.int64),time=np.concatenate(times),minimum_since_previous=np.concatenate(depth),crossing_velocity=np.concatenate(velocity),statistics=np.array(stat))
    manifest=dict(model='dX=V dt; dV=-(X+epsilon V)dt + sqrt(epsilon*(sigma^2+2*r*X^2))dB + delta*sin(omega*t)dt',epsilon=.05,sigma=1,dt=.0025,T=4000,burn=1000,grid_seeds=list(range(101,109)),focal_seeds=list(range(201,209)),grid_runs=2976,focal_runs=96,raw_crossings=offset[-1],parameters_columns=['r','delta','omega','seed'],statistics_columns=['rms_X','residual_variance','mean_V2','mean_geometric_phase_rate','Q1','Q2','Q3','mean_X2'],python=platform.python_version(),packages={p:md.version(p) for p in ['numpy','scipy','pillow']},source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'forcing.c',ROOT/'run_experiments.py',ROOT/'analyse_experiments.py']})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    with zipfile.ZipFile(OUT/'reproduction.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
      for name in ['forcing.c','run_experiments.py','analyse_experiments.py','validate_superposition.py','make_figures.py','opportunities.py','package_results.py','verify_numerics.py']:
        z.write(ROOT/name,'work/'+name)
      for p in sorted(OUT.iterdir()):
        if p.suffix!='.zip':z.write(p,'outputs/forcing_followup/'+p.name)
      for p in sorted((ROOT/'runs').glob('restore_*.npz')):z.write(p,'work/runs/'+p.name)
      z.write(ROOT/'superposition.json','work/superposition.json')
    print(json.dumps(dict(raw_crossings=offset[-1],runs=len(meta),archive_MB=(OUT/'reproduction.zip').stat().st_size/1048576)))
if __name__=='__main__':main()
