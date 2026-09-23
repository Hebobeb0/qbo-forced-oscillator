from pathlib import Path
import csv,json,sys,zipfile
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import t as student_t
from reportlab.graphics.shapes import Drawing,Group,String,Line,Polygon,Circle,Rect
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderSVG
from reportlab.lib.colors import HexColor,Color
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'outputs'/'forcing_followup'
OUT=ROOT.parent/'outputs'/'conclusion_graphs';OUT.mkdir(exist_ok=True,parents=True)
sys.path.insert(0,str(ROOT))
from analyse_experiments import accepted
pdfmetrics.registerFont(TTFont('SegoeUI','C:/Windows/Fonts/segoeui.ttf'))
pdfmetrics.registerFont(TTFont('SegoeUI-Bold','C:/Windows/Fonts/segoeuib.ttf'))
INK='#172536';MUTED='#526174';GRID='#dce3eb'
C={0:'#677586',2:'#2476ad',4:'#d5781d'}
MODEL={0:'#374151',.5:'#008577'}

def read(name):
 with (DATA/name).open() as f:return [{k:v if k=='rule' else float(v) for k,v in a.items()} for a in csv.DictReader(f)]
def write(name,rows):
 with (OUT/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def text(d,x,y,s,size=10,bold=False,color=INK,anchor='start'):
 d.add(String(x,y,s,fontName='SegoeUI-Bold' if bold else 'SegoeUI',fontSize=size,fillColor=HexColor(color),textAnchor=anchor))
def pale(c,f=.87):
 x=HexColor(c);return Color(1-(1-x.red)*(1-f),1-(1-x.green)*(1-f),1-(1-x.blue)*(1-f))
def ci(a):
 m=a.mean(axis=0);h=student_t.ppf(.975,len(a)-1)*a.std(axis=0,ddof=1)/np.sqrt(len(a));return m,m-h,m+h

class Panel:
 def __init__(self,title,subtitle,xlim,ylim,xsteps,ysteps,xlabel,ylabel,note):
  self.d=Drawing(550,440);self.x=55;self.y=100;self.w=470;self.h=260;self.xlim=xlim;self.ylim=ylim
  self.d.add(Rect(0,0,550,440,fillColor=HexColor('#ffffff'),strokeColor=None))
  text(self.d,0,418,title,16,True);text(self.d,0,396,subtitle,10,color=MUTED)
  text(self.d,self.x,371,ylabel,10,color=MUTED)
  text(self.d,self.x+self.w/2,30,xlabel,11,anchor='middle')
  text(self.d,0,7,note,9,color=MUTED)
  self.lp=LinePlot();p=self.lp;p.x=self.x;p.y=self.y;p.width=self.w;p.height=self.h
  p.xValueAxis.valueMin=xlim[0];p.xValueAxis.valueMax=xlim[1];p.xValueAxis.valueSteps=xsteps
  p.yValueAxis.valueMin=ylim[0];p.yValueAxis.valueMax=ylim[1];p.yValueAxis.valueSteps=ysteps
  for ax in [p.xValueAxis,p.yValueAxis]:
   ax.labels.fontName='SegoeUI';ax.labels.fontSize=9;ax.labels.fillColor=HexColor(MUTED)
   ax.strokeColor=HexColor('#a6b2c0');ax.strokeWidth=.6;ax.gridStrokeColor=HexColor(GRID);ax.gridStrokeWidth=.4;ax.visibleGrid=1
  p.xValueAxis.labels.dy=-8;p.yValueAxis.labels.dx=-7
  p.xValueAxis.tickDown=3;p.xValueAxis.tickUp=0;p.yValueAxis.tickLeft=3;p.yValueAxis.tickRight=0
  p.xValueAxis.labelTextFormat='%g';p.yValueAxis.labelTextFormat='%g'
  self.series=[];self.styles=[];self.guides=[];self.bars=[];self.markers=[]
 def xy(self,a,b):return self.x+(a-self.xlim[0])/(self.xlim[1]-self.xlim[0])*self.w,self.y+(b-self.ylim[0])/(self.ylim[1]-self.ylim[0])*self.h
 def ribbon(self,x,lo,hi,c):
  lo=np.maximum(lo,self.ylim[0]);hi=np.minimum(hi,self.ylim[1]);pts=[]
  for a,b in zip(x,lo):pts.extend(self.xy(a,b))
  for a,b in zip(x[::-1],hi[::-1]):pts.extend(self.xy(a,b))
  self.d.add(Polygon(pts,fillColor=pale(c),strokeColor=None))
 def line(self,x,y,c,dash=None,width=1.8,marker=False):
  self.series.append([(float(a),float(b)) for a,b in zip(x,y)]);self.styles.append((c,dash,width))
  if marker:
   for a,b in zip(x,y):self.markers.append((a,b,c,dash is not None))
 def errors(self,x,lo,hi,c):
  for a,b,z in zip(x,lo,hi):self.bars.append((a,max(b,self.ylim[0]),min(z,self.ylim[1]),c))
 def horizontal(self,v,label=None):self.guides.append(('h',v,label))
 def vertical(self,v):self.guides.append(('v',v,None))
 def legend(self,items):
  # Items specify label, colour, dash, horizontal start. All outside the data area.
  for label,c,dash,x in items:
   ln=Line(x,62,x+24,62,strokeColor=HexColor(c),strokeWidth=1.8)
   if dash:ln.strokeDashArray=[5,3]
   self.d.add(ln);text(self.d,x+30,59,label,10,color=MUTED)
 def finish(self):
  p=self.lp;p.data=self.series
  for i,(c,dash,width) in enumerate(self.styles):
   p.lines[i].strokeColor=HexColor(c);p.lines[i].strokeWidth=width
   if dash:p.lines[i].strokeDashArray=dash
  self.d.add(p)
  for mode,v,label in self.guides:
   if mode=='h':a,b=self.xy(self.xlim[0],v);e,f=self.xy(self.xlim[1],v)
   else:a,b=self.xy(v,self.ylim[0]);e,f=self.xy(v,self.ylim[1])
   self.d.add(Line(a,b,e,f,strokeColor=HexColor('#8b97a6'),strokeWidth=.8,strokeDashArray=[3,3]))
   if label:text(self.d,e-5,f+5,label,9,color=MUTED,anchor='end')
  for x,lo,hi,c in self.bars:
   a,b=self.xy(x,lo);_,z=self.xy(x,hi)
   self.d.add(Line(a,b,a,z,strokeColor=HexColor(c),strokeWidth=.9))
   for yy in [b,z]:self.d.add(Line(a-2.5,yy,a+2.5,yy,strokeColor=HexColor(c),strokeWidth=.9))
  for x,y,c,open_ in self.markers:
   a,b=self.xy(x,y)
   self.d.add(Circle(a,b,2.8,fillColor=HexColor('#ffffff') if open_ else HexColor(c),strokeColor=HexColor(c),strokeWidth=1.2))
  return self.d

def save(d,name):
 if name!='conclusion_summary':
  padded=Drawing(590,480);padded.add(Rect(0,0,590,480,fillColor=HexColor('#ffffff'),strokeColor=None));g=Group();g.add(d);g.translate(20,20);padded.add(g);d=padded
 renderSVG.drawToFile(d,str(OUT/(name+'.svg')))
 p=OUT/(name+'.svg');s=p.read_text(encoding='utf-8');s=s.replace('SegoeUI-Bold','Segoe UI Semibold').replace('SegoeUI','Segoe UI');s=s.replace('<title>...</title>','<title>'+name.replace('_',' ')+'</title>');s=s.replace('<desc>...</desc>','<desc>Computed simulation results with confidence intervals across independent seeds. See captions.md for definitions and scope.</desc>');p.write_text(s,encoding='utf-8')

def main():
 f=read('focal_summary.csv');f=[a for a in f if a['r']==.5 and a['rule']=='hysteresis']
 # Build seed-normalized event density estimates from compact raw events.
 z=np.load(DATA/'events.npz');par=z['parameters'];off=z['offsets'];phase_rows=[];density_rows=[];phase={};density={}
 for delta in [0,2,4]:
  hh=[];nn=[]
  for j,(r,d,w,seed) in enumerate(par):
   if r!=.5 or d!=delta or not np.isclose(w,7/3):continue
   sl=slice(off[j],off[j+1]);a={'counts':np.array([off[j+1]-off[j]]),'ev':z['time'][sl][None,:],'dep':z['minimum_since_previous'][sl][None,:],'stats':z['statistics'][j:j+1]}
   tt=accepted(a,0);ph=(tt*w/(2*np.pi))%1;n=np.diff(tt)*w/(2*np.pi)
   h,e=np.histogram(ph,bins=np.linspace(0,1,49),density=True);hh.append(gaussian_filter1d(h,.7,mode='wrap'))
   h,b=np.histogram(n,bins=np.linspace(0,6,241),density=True);nn.append(gaussian_filter1d(h,1.))
  assert len(hh)==16
  x=(e[1:]+e[:-1])/2;m,lo,hi=ci(np.array(hh));phase[delta]=(x,m,lo,hi)
  for a,b,c,d in zip(x,m,lo,hi):phase_rows.append(dict(delta=delta,drive_phase_cycles=a,density=b,ci_low=c,ci_high=d,seeds=16))
  edges=np.linspace(0,6,241);x=(edges[1:]+edges[:-1])/2
  m,lo,hi=ci(np.array(nn));density[delta]=(x,m,lo,hi)
  for a,b,c,d in zip(x,m,lo,hi):density_rows.append(dict(delta=delta,N=a,density=b,ci_low=c,ci_high=d,seeds=16))
 write('crossing_phase_density.csv',phase_rows);write('interval_density.csv',density_rows)
 a=Panel('A  Crossings prefer a forcing phase','Unforced crossings are approximately uniform.',(0,1),(0,4.3),[0,.25,.5,.75,1],[0,1,2,3,4],'Forcing phase at crossing / 2π','Probability density','r = 0.5; ω = 7/3. Shading: pointwise 95% CI, 16 seeds.')
 for delta in [0,2,4]:
  x,m,lo,hi=phase[delta];a.ribbon(x,lo,hi,C[delta]);a.line(x,m,C[delta])
 a.horizontal(1);a.legend([('δ = 0',C[0],False,75),('δ = 2',C[2],False,225),('δ = 4',C[4],False,375)])
 A=a.finish();save(A,'01_crossing_phase')
 b=Panel('B  Near-integer intervals become more common','Stronger forcing increasingly favours one-period intervals.',(0,4),(0,45),[0,1,2,3,4],[0,10,20,30,40],'Forcing amplitude δ','Fraction of intervals (%)','r = 0.5; ω = 7/3. Bands: |N − n| < 0.1; 95% CI, 16 seeds.')
 weights=[];cat=[('integer_fraction','Any integer','#374151'),('band1','N ≈ 1','#d5781d'),('band2','N ≈ 2','#2476ad'),('band3','N ≈ 3','#9269a7')]
 for key,label,color in cat:
  x=np.array([v['delta'] for v in f]);m=np.array([v[key] for v in f])*100;lo=np.array([v[key+'_low'] for v in f])*100;hi=np.array([v[key+'_high'] for v in f])*100
  b.line(x,m,color,width=2.2 if key=='integer_fraction' else 1.5,marker=True);b.errors(x,lo,hi,color)
  for aa,bb,cc,dd in zip(x,m,lo,hi):weights.append(dict(delta=aa,band=label,percent=bb,ci_low=cc,ci_high=dd))
 b.legend([(label,col,False,pos) for (_,label,col),pos in zip(cat,[35,178,302,426])]);B=b.finish();save(B,'02_integer_interval_probabilities');write('interval_band_probabilities.csv',weights)
 c=Panel('C  The linear control shows the same flattening','State-dependent diffusion is not required for the branch.',(0,4),(0,1.12),[0,1,2,3,4],[0,.25,.5,.75,1],'Forcing amplitude δ','Slope of the N ≈ 2 branch, dN/dω','8 seeds; 95% seed-bootstrap CI. Fit range: ω = 1.9–2.6.')
 slopes=[a for a in read('branch_slopes.csv') if a['rule']=='hysteresis'];write('branch_slopes.csv',slopes)
 for r in [0,.5]:
  rs=[v for v in slopes if v['r']==r];x=np.array([v['delta'] for v in rs]);y=np.array([v['slope'] for v in rs]);lo=np.array([v['bootstrap_low'] for v in rs]);hi=np.array([v['bootstrap_high'] for v in rs])
  c.line(x,y,MODEL[r],[5,3] if r==0 else None,marker=True);c.errors(x,lo,hi,MODEL[r])
 c.horizontal(1,'Unforced reference');c.horizontal(0,'Flat branch');c.legend([('r = 0: exact superposition',MODEL[0],True,60),('r = 0.5',MODEL[.5],False,355)])
 Cfig=c.finish();save(Cfig,'03_linear_control')
 # No trajectory selection: use all eight independent saved focal paths.
 phasepaths={d:[] for d in [0,2,4]};phase_drift_rows=[]
 for p in sorted((ROOT/'runs').glob('focal_*.npz')):
  with np.load(p) as z:
   par=z['params'];tr=z['trace']
   for delta in [0,2,4]:
    j=np.where((par[:,0]==.5)&(par[:,1]==delta))[0][0]
    t=np.arange(len(tr[j]))*.05;xx=t*(7/3)/(2*np.pi);ids=np.where(xx<=300)[0][::20];x=xx[ids]
    phi=tr[j,:,2];change=(2*(phi-phi[0])-(7/3)*t)/(2*np.pi);phasepaths[delta].append(change[ids])
 d=Panel('D  The 2:1 relative phase keeps drifting','Sustained two-period locking is not established here.',(0,300),(-60,170),[0,100,200,300],[-50,0,50,100,150],'Elapsed forcing periods','Change in (2φ − ωt) / 2π  [turns]','r = 0.5; ω = 7/3. 8 seeds; shaded pointwise 95% CI; after burn-in.')
 for delta in [0,2,4]:
  m,lo,hi=ci(np.array(phasepaths[delta]));assert lo.min()>-60 and hi.max()<170
  d.ribbon(x,lo,hi,C[delta]);d.line(x,m,C[delta])
  for aa,bb,cc,dd in zip(x,m,lo,hi):phase_drift_rows.append(dict(delta=delta,elapsed_drive_periods=aa,phase_change_turns=bb,ci_low=cc,ci_high=dd,seeds=8))
 d.horizontal(0,'Zero net drift');d.legend([('δ = 0',C[0],False,75),('δ = 2',C[2],False,225),('δ = 4',C[4],False,375)])
 D=d.finish();save(D,'04_relative_phase_drift');write('relative_phase_drift.csv',phase_drift_rows)
 master=Drawing(1200,1080)
 master.add(Rect(0,0,1200,1080,fillColor=HexColor('#ffffff'),strokeColor=None))
 text(master,45,1040,'Additive forcing organizes cycle timing',27,True)
 text(master,45,1010,'Near-integer intervals increase; sustained two-period locking is not established.',15,color=MUTED)
 text(master,45,987,'Fresh simulations: ε = 0.05, σ = 1, integration step = 0.0025. Hysteresis threshold = 0.25 RMS(X).',10,color=MUTED)
 for pic,x0,y0 in [(A,45,520),(B,630,520),(Cfig,45,45),(D,630,45)]:
  g=Group();g.add(pic);g.translate(x0,y0);master.add(g)
 text(master,45,20,'N = cycle interval / forcing period. Confidence intervals reflect variation across independent simulation seeds.',9,color=MUTED)
 save(master,'conclusion_summary')
 # Distribution detail: avoid suggesting a delta-function concentration at integers.
 e=Panel('E  Intervals redistribute between broad modes','The distributions have finite width; the N ≈ 2 branch is offset.',(.5,3.5),(0,2.8),[.5,1,1.5,2,2.5,3,3.5],[0,1,2],'N = cycle interval / forcing period','Probability density','r = 0.5; ω = 7/3. Shading: pointwise 95% CI, 16 seeds.')
 for delta in [0,2,4]:
  xx,m,lo,hi=density[delta];sel=(xx>=.5)&(xx<=3.5);e.ribbon(xx[sel],lo[sel],hi[sel],C[delta]);e.line(xx[sel],m[sel],C[delta])
 for n in [1,2,3]:e.vertical(n)
 e.legend([('δ = 0',C[0],False,75),('δ = 2',C[2],False,225),('δ = 4',C[4],False,375)])
 save(e.finish(),'05_interval_distributions')
 print('Five panels and combined vector figure saved.')

if __name__=='__main__':main()
