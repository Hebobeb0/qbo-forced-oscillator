# ============================================================================
#  Configuration
# ============================================================================
# SCALE controls how much compute the whole notebook does.
#   'quick' :  ~15 minutes on a laptop.  Coarser grids, shorter runs.
#              Every qualitative conclusion is already visible at this scale.
#   'full'  :  reproduces the grids used in the report (hours in pure NumPy,
#              so budget an overnight run, or cut it to the figure you want).
SCALE = 'quick'

import numpy as np, pandas as pd, matplotlib, itertools, time
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import warnings; warnings.filterwarnings('ignore')
print('SCALE =', SCALE)

# ============================================================================
#  The integrator
# ============================================================================
# Symplectic (semi-implicit) Euler-Maruyama, vectorised over parameter configs:
#
#     V <- V + ( -(k(t) X + eps V) + F(t) ) dt + g(X) dW
#     X <- X + V_new dt
#
# V is updated FIRST and X is then advanced with the NEW V.  The ordering is not
# cosmetic: the fully explicit forward scheme multiplies the energy by (1+dt^2)
# each step, an artificial anti-damping which at eps=0.01, dt=0.005 exactly
# cancels the physical damping and manufactures a spurious marginal state.
#
# g(X) is evaluated at the OLD X, so the scheme is Ito.  Because g does not
# depend on V the Milstein correction vanishes identically and this is already
# strong order 1.
#
# Under parametric forcing the solution can grow without bound above threshold,
# so the state is rescaled by 1e-6 whenever h = (X^2+V^2)/2 exceeds 1e12 and the
# rescaling is accumulated into a separately carried log h.  At that amplitude
# the sigma^2 term in the diffusion is 12 orders of magnitude below 2rX^2, so
# the rescaled trajectory is a faithful continuation.

def _bc(v, n):
    return np.ascontiguousarray(np.broadcast_to(np.atleast_1d(np.asarray(v, float)), (n,)), float)

def integrate(eps, sigma, r, delta, omega, mode='parametric', tau=None,
              T=3000.0, dt=0.0025, thin=80, seed=0, rescale_at=1e12, theta0=None):
    """mode: 'parametric' | 'additive' | 'none' | 'full3d' | 'full3d_param'.

    Returns (t, X, LH, PH, A2) where LH is LOG h with any rescaling folded back
    in, and PH is the unwrapped phase atan2(V, X).  Every parameter except eps
    and sigma may be an array; all arrays are broadcast to a common length and
    run simultaneously with common random numbers."""
    n = max(np.size(x) for x in (1 if r is None else r, delta, omega,
                                 1 if tau is None else tau))
    delta, omega = _bc(delta, n), _bc(omega, n)
    r   = _bc(0.0 if r   is None else r,   n)
    tau = _bc(1.0 if tau is None else tau, n)
    N = int(round(T/dt)); nrec = N//thin + 1
    rng = np.random.default_rng(seed)
    sq, s2 = np.sqrt(dt), sigma*sigma

    th0 = rng.random()*2*np.pi if theta0 is None else float(theta0)
    a0 = np.sqrt(2.0)                                   # h0 = 1
    X = np.full(n, a0*np.sin(th0)); V = np.full(n, a0*np.cos(th0))
    ls = np.zeros(n)
    is3d = mode.startswith('full3d')
    A  = rng.standard_normal(n) if is3d else None      # OU starts stationary
    ad = np.exp(-dt/tau); as_ = np.sqrt(1.0 - ad*ad)            # exact OU update

    Xo = np.empty((nrec, n)); LH = np.empty((nrec, n))
    PH = np.empty((nrec, n)); A2 = np.empty((nrec, n))
    gn = np.sqrt(eps)*sigma
    k = 0
    for i in range(N+1):
        if i % thin == 0 and k < nrec:
            Xo[k] = X; LH[k] = np.log(0.5*(X*X + V*V)) + 2.0*ls
            PH[k] = np.arctan2(V, X); A2[k] = A*A if A is not None else 1.0
            k += 1
        if i == N: break
        sn = np.sin(omega*(i*dt))
        z  = rng.standard_normal()*sq          # common random numbers across configs
        if mode == 'full3d':            # random frequency, ADDITIVE forcing
            kk, drive, g = A*A, delta*sn, gn
        elif mode == 'full3d_param':    # random frequency, PARAMETRIC forcing
            kk, drive, g = A*A + delta*sn, 0.0, gn
        elif mode == 'parametric':
            kk, drive, g = 1.0 + delta*sn, 0.0, np.sqrt(eps*(s2 + 2.0*r*X*X))
        elif mode == 'additive':
            kk, drive, g = 1.0, delta*sn, np.sqrt(eps*(s2 + 2.0*r*X*X))
        else:
            kk, drive, g = 1.0, 0.0, np.sqrt(eps*(s2 + 2.0*r*X*X))
        V = V + (-(kk*X + eps*V) + drive)*dt + g*z
        X = X + V*dt
        if is3d:
            A = ad*A + as_*rng.standard_normal()
        if rescale_at:
            big = 0.5*(X*X + V*V) > rescale_at
            if big.any():
                X[big] *= 1e-6; V[big] *= 1e-6; ls[big] += np.log(1e6)
    return np.arange(nrec)*dt*thin, Xo, LH, np.unwrap(PH, axis=0), A2

# ============================================================================
#  Diagnostics
# ============================================================================

def ps_exact(h, sigma=1.0, r=0.5):
    """Exact stationary density of the unforced energy, p_s(h) = C (sigma^2+rh)^(-2/r).
    Normalisable for r < 2; finite mean for r < 1."""
    h = np.asarray(h, float)
    return (sigma**2 + r*h)**(-2.0/r) / ((sigma**2)**(1 - 2.0/r) / (2.0 - r))

def unit_signal(PH):
    """u = X/sqrt(X^2+V^2) = cos(phi).  Bounded, unaffected by rescaling, and it
    carries exactly the quantity locking is about."""
    return np.cos(PH)

def spectrum(t, y, fmin=0.10, fmax=3.60):
    n = y.shape[0]
    P = np.abs(np.fft.rfft((y - y.mean(axis=0))*np.hanning(n)[:, None], axis=0))**2
    f = np.fft.rfftfreq(n, d=t[1]-t[0])*2*np.pi
    k = (f > fmin) & (f < fmax)
    return f[k], P[k]

def peak_freq(t, y, **kw):
    f, P = spectrum(t, y, **kw)
    return f[P.argmax(axis=0)], P.max(axis=0)/(P.sum(axis=0) + 1e-300)

def peak_excluding_drive(t, y, omega, nharm=4, tol=0.03, fmin=0.30, fmax=8.0):
    """Largest spectral peak of y EXCLUDING bands around omega, 2omega, ...
    Needed for additive forcing: at large delta the trajectory IS the linear
    response, so an unfiltered peak (or a phase counter) reports the drive."""
    f, P = spectrum(t, y, fmin=fmin, fmax=fmax)
    omega = np.atleast_1d(omega); out = np.empty(y.shape[1])
    for j in range(y.shape[1]):
        keep = np.ones(f.size, bool)
        for m in range(1, nharm+1):
            keep &= ~((f > m*omega[j]*(1-tol)) & (f < m*omega[j]*(1+tol)))
        out[j] = f[keep][P[keep, j].argmax()]
    return out

def growth_rate(t, LH):
    """Lyapunov-type exponent: half the slope of log h against t."""
    tc = t - t.mean()
    return (tc @ LH)/(tc @ tc)/2

def rotation_rate(t, PH):
    return np.abs((PH[-1] - PH[0])/(t[-1] - t[0]))

def coherence(t, PH, omega, n):
    """n:1 phase synchronisation index R_n = |<exp i(phi - omega t/n)>|.
    R = 1 is a rigid phase relation, R = 0 free drift."""
    phi = -(PH - PH[0])
    return np.abs(np.mean(np.exp(1j*(phi - np.atleast_1d(omega)[None, :]*t[:, None]/n)), axis=0))

def cycle_periods(t, PH):
    """Times for the unwrapped phase to advance by 2*pi, pooled over columns.
    Uses the first-passage envelope and linear interpolation, so the period is
    not quantised by the recording grid."""
    out = []
    for j in range(PH.shape[1]):
        col = PH[:, j]; good = np.isfinite(col)
        if good.sum() < 4: continue
        ph = -(col[good] - col[good][0]); tt = t[good]
        run = np.maximum.accumulate(ph)
        if not np.isfinite(run[-1]) or run[-1] < 2*np.pi: continue
        lv = np.arange(2*np.pi, run[-1], 2*np.pi)
        if lv.size < 2: continue
        out.append(np.diff(np.interp(lv, run, tt)))
    return np.concatenate(out) if out else np.array([])

def bimodality(x):
    """Sarle's coefficient. > 5/9 is the uniform reference."""
    n = len(x)
    if n < 8: return np.nan
    s = x.std(ddof=1)
    if not s > 0: return np.nan
    z = (x - x.mean())/s
    g1 = (n/((n-1)*(n-2)))*np.sum(z**3)
    g2 = ((n*(n+1))/((n-1)*(n-2)*(n-3)))*np.sum(z**4) - 3*(n-1)**2/((n-2)*(n-3))
    return float((g1**2 + 1)/(g2 + 3*(n-1)**2/((n-2)*(n-3))))

def delta_c(omega, eps):
    """Mathieu threshold for the principal (2:1) tongue, from two-timing.
    Asymptotic: valid for small delta and small detuning."""
    return 2*np.sqrt((np.asarray(omega) - 2.0)**2 + eps**2)

# ============================================================================
#  Plot style
# ============================================================================
INK, MUTED, GRID, SURF = '#1f2328', '#5c6370', '#e6e6e3', '#fcfcfb'
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = '#2f5fbf', '#d1600a', '#2f9e6e', '#b3271e', '#7a5195', '#9aa0a6'
# sequential = one hue, light -> dark (magnitude)
SEQ = LinearSegmentedColormap.from_list('seq',
      ['#fbfcfe','#e2ebf8','#b9cfef','#89aee4','#5586d6','#2f5fbf','#1d3f88','#101f47'])
# diverging = two hues + neutral midpoint (polarity, e.g. signed growth rate)
DIV = LinearSegmentedColormap.from_list('div',
      ['#1d3f88','#5586d6','#b9cfef','#f2f2ef','#f2c9a0','#e08a3c','#b3271e'])
plt.rcParams.update({
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'legend.frameon': False, 'legend.fontsize': 8, 'figure.dpi': 110,
    'axes.spines.top': False, 'axes.spines.right': False})

def tidy(ax):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)

# ============================================================================
#  Grids  (set by SCALE)
# ============================================================================
if SCALE == 'full':
    CFG = dict(
        tongue_om=np.round(np.arange(0.40, 4.0001, 0.02), 4),
        tongue_dl=np.round(np.arange(0.00, 3.0001, 0.03), 4),
        tongue_T=1500., tongue_burn=500.,
        sweep_om=np.round(np.arange(0.40, 4.0001, 0.025), 4),
        sweep_dl=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0],
        sweep_r=[0.0, 0.5, 0.9], sweep_sg=[0.5, 1.0, 2.0],
        sweep_eps=[0.02, 0.05, 0.1, 0.2], sweep_seeds=[11, 22, 33, 44],
        sweep_T=3000., sweep_burn=1000.,
        map_om=np.round(np.arange(0.45, 4.001, 0.05), 4), map_T=4000., map_burn=1500.,
        line_T=24000., line_burn=4000., nperseg=4096,
        d3_om=np.round(np.arange(0.5, 4.001, 0.05), 4), d3_T=6000., d3_burn=2000.,
    )
else:
    CFG = dict(
        tongue_om=np.round(np.arange(0.40, 4.0001, 0.05), 4),
        tongue_dl=np.round(np.arange(0.00, 3.0001, 0.075), 4),
        tongue_T=800., tongue_burn=250.,
        sweep_om=np.round(np.arange(0.40, 4.0001, 0.10), 4),
        sweep_dl=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
        sweep_r=[0.0, 0.5], sweep_sg=[1.0],
        sweep_eps=[0.05, 0.1], sweep_seeds=[11, 22],
        sweep_T=1500., sweep_burn=500.,
        map_om=np.round(np.arange(0.45, 4.001, 0.05), 4), map_T=1500., map_burn=500.,
        line_T=8000., line_burn=2000., nperseg=2048,
        d3_om=np.round(np.arange(0.5, 4.001, 0.10), 4), d3_T=3000., d3_burn=1000.,
    )
DT, THIN = 0.0025, 80
print({k: (len(v) if hasattr(v, "__len__") else v) for k, v in CFG.items()})

# ============================================================================
#  Validation.  Run this before trusting anything below.
# ============================================================================
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# --- 1. the scheme, against an independent stiff ODE solver -----------------
# With sigma = 0 and r = 0 the diffusion vanishes and the equation is the
# deterministic damped Mathieu equation, which scipy can integrate to high
# accuracy.  The growth rate is the quantity the report depends on.
print('1. Growth rate vs scipy solve_ivp (deterministic Mathieu):')
print(f'   {"omega":>6s} {"delta":>6s} {"solve_ivp":>12s} {"symplectic EM":>14s} {"rel err":>10s}')
for om, dl in ((1.5, 1.5), (2.0, 1.5), (2.5, 2.0), (3.0, 1.5)):
    eps, th0 = 0.02, 0.7
    y0 = [np.sqrt(2)*np.sin(th0), np.sqrt(2)*np.cos(th0)]
    f = lambda t, y: [y[1], -((1 + dl*np.sin(om*t))*y[0] + eps*y[1])]
    s = solve_ivp(f, [0, 300], y0, rtol=1e-10, atol=1e-12, dense_output=True, max_step=0.05)
    tt = np.linspace(50, 300, 2000); Y = s.sol(tt)
    g_ref = np.polyfit(tt, np.log(0.5*(Y[0]**2 + Y[1]**2)), 1)[0]/2
    t, X, LH, PH, _ = integrate(eps, 0.0, 0.0, dl, om, mode='parametric',
                                T=300., dt=0.0025, thin=20, theta0=th0)
    m = t >= 50
    g_em = np.polyfit(t[m], LH[m, 0], 1)[0]/2
    print(f'   {om:6.2f} {dl:6.2f} {g_ref:12.6f} {g_em:14.6f} {abs(g_em-g_ref)/max(abs(g_ref),1e-9):10.1e}')

# --- 2. the stationary density, against the closed form --------------------
print('\n2. Unforced stationary energy (sigma=1, r=0.5):')
hg = np.linspace(0, 60, 20000)
cdf = np.cumsum(ps_exact(hg))*(hg[1]-hg[0])
med_th = np.interp(0.5, cdf, hg)
t, X, LH, PH, _ = integrate(0.05, 1.0, 0.5, 0.0, 2.0, mode='none', T=20000., seed=7)
h = np.exp(LH[t >= 2000, 0])
print(f'   median h  measured {np.median(h):.4f}   exact {med_th:.4f}   rel err {abs(np.median(h)-med_th)/med_th:.3f}')

# --- 3. the unforced period ------------------------------------------------
per = cycle_periods(t[t >= 2000], PH[t >= 2000])
print(f'\n3. Unforced cycle period  measured {np.median(per):.4f}   2*pi = {2*np.pi:.4f}')

# --- 4. rescaling fidelity -------------------------------------------------
t, X, LH, PH, _ = integrate(0.05, 1.0, 0.5, 4.0, 2.5, mode='parametric', T=3000., seed=5)
print(f'\n4. Rescaling: log h reaches {LH[-1,0]:.0f} nats = 1e{LH[-1,0]/np.log(10):.0f}, '
      f'linear in t (residual sd {np.std(LH[:,0]-np.polyval(np.polyfit(t,LH[:,0],1),t))/LH[-1,0]:.1e} of the range)')

# ============================================================================
#  The tongue maps.  One computation, three views.
# ============================================================================
# For every (omega, delta) cell we record three things:
#   G  amplitude growth rate  (is the cell unstable?)
#   F  peak response frequency, from the spectrum of u = cos(phi)
#   C  how concentrated that peak is (fraction of power in the strongest bin)
# G gives figure P1, F and F/omega give figure P11.

def tongue_scan(sigma, r, eps=0.02, rows_per_chunk=8):
    OM, DL = CFG['tongue_om'], CFG['tongue_dl']
    T, BURN = CFG['tongue_T'], CFG['tongue_burn']
    F = np.full((len(DL), len(OM)), np.nan); C = np.full_like(F, np.nan); G = np.full_like(F, np.nan)
    for lo in range(0, len(DL), rows_per_chunk):
        hi = min(lo + rows_per_chunk, len(DL))
        dv = np.repeat(DL[lo:hi], len(OM)); wv = np.tile(OM, hi - lo)
        t, X, LH, PH, _ = integrate(eps, sigma, np.full(dv.size, r), dv, wv,
                                    mode='parametric', T=T, dt=DT, thin=THIN, seed=11)
        m = t >= BURN
        fpk, conc = peak_freq(t[m], unit_signal(PH[m]))
        F[lo:hi] = fpk.reshape(hi-lo, -1); C[lo:hi] = conc.reshape(hi-lo, -1)
        G[lo:hi] = growth_rate(t[m], LH[m]).reshape(hi-lo, -1)
    return F, C, G

SETTINGS = [('det', 0.0, 0.0, 'Deterministic  ($\\sigma$=0, r=0)'),
            ('noisy', 1.0, 0.5, 'With noise  ($\\sigma$=1, r=0.5)'),
            ('noisy_hi', 1.0, 1.4, 'Strong multiplicative noise  ($\\sigma$=1, r=1.4)')]
TONGUES = {}
for tag, sg, rv, ttl in SETTINGS:
    t0 = time.time(); TONGUES[tag] = tongue_scan(sg, rv)
    print(f'{tag:9s} {len(CFG["tongue_dl"])*len(CFG["tongue_om"]):6d} runs in {time.time()-t0:5.1f}s')

# ============================================================================
#  P1  -  instability tongues, coloured by GROWTH RATE
# ============================================================================
OM, DL = CFG['tongue_om'], CFG['tongue_dl']
oe = np.r_[OM - 0.5*(OM[1]-OM[0]), OM[-1] + 0.5*(OM[1]-OM[0])]
de = np.r_[DL - 0.5*(DL[1]-DL[0]), DL[-1] + 0.5*(DL[1]-DL[0])]

fig, AX = plt.subplots(1, 3, figsize=(13.6, 4.3), sharey=True)
for ax, (tag, sg, rv, ttl) in zip(AX, SETTINGS):
    F, C, G = TONGUES[tag]
    vm = np.nanpercentile(np.abs(G), 99)
    im = ax.pcolormesh(oe, de, G, cmap=DIV, vmin=-vm, vmax=vm, shading='flat', rasterized=True)
    ax.contour(OM, DL, (G > 0.01).astype(float), levels=[0.5], colors=['#1f2328'], linewidths=1.0)
    w = np.linspace(OM[0], OM[-1], 400)
    ax.plot(w, delta_c(w, 0.02), lw=1.4, ls=(0, (5, 3)), color='#1f2328')
    ax.set(xlim=(OM[0], OM[-1]), ylim=(0, DL[-1]), xlabel='forcing frequency  $\\omega$')
    ax.set_title(ttl, loc='left'); ax.grid(False)
AX[0].set_ylabel('forcing strength  $\\delta$')
AX[0].annotate('principal (2:1) tongue\nresponse at $\\omega$/2', (2.0, 1.35), ha='center',
               fontsize=8.5, bbox=dict(fc=SURF, ec=GRID, pad=2))
for n, lab in ((2, 'n=2\n($\\omega$$\\approx$1)'), (3, 'n=3')):
    AX[0].annotate(lab, (2.0/n, 2.3 if n == 2 else 1.9), ha='center', fontsize=7.5,
                   bbox=dict(fc=SURF, ec='none', pad=1))
AX[2].text(2.15, 2.78, 'dashed: $\\delta_c = 2\\sqrt{(\\omega-2)^2+\\epsilon^2}$',
           fontsize=8, bbox=dict(fc=SURF, ec=GRID, pad=2))
cb = fig.colorbar(im, ax=AX, pad=0.012, fraction=0.018); cb.outline.set_visible(False)
cb.set_label('amplitude growth rate  $\\lambda$', fontsize=8.5)
fig.suptitle(f'P1 - Instability tongues of the parametrically forced oscillator  '
             f'($\\epsilon$=0.02, {len(DL)*len(OM):,} runs per panel)', x=0.006, ha='left', fontsize=11)
plt.show()

# ============================================================================
#  P11  -  the same grid, coloured by PEAK RESPONSE FREQUENCY
# ============================================================================
# Top    : the response frequency itself (a magnitude -> sequential ramp),
#          shown only where the cell is unstable.
# Bottom : which rational ratio rho = omega_resp/omega the cell sits on
#          (an identity -> categorical hues, validated all-pairs for CVD).
#
# The rule the bottom row reveals: in the n-th tongue (tip at omega = 2/n) the
# response sits at omega_0 = 1, i.e. rho = n/2.  n=1 -> 1/2, n=2 -> 1, n=3 -> 3/2.
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'      # categorical slots 1-3
CNONE, CSTABLE = '#43464b', '#ececeb'             # neutrals
TOL = 0.04

fig, AX = plt.subplots(2, 3, figsize=(14.2, 8.0), sharex=True, sharey=True)
fig.subplots_adjust(left=0.055, right=0.90, top=0.855, bottom=0.075, hspace=0.20, wspace=0.06)
for j, (tag, sg, rv, ttl) in enumerate(SETTINGS):
    F, C, G = TONGUES[tag]
    unst = G > 0.01; rho = F/OM[None, :]

    ax = AX[0, j]; ax.set_facecolor(CSTABLE); ax.grid(False)
    im = ax.pcolormesh(oe, de, np.ma.masked_where(~unst, F), cmap=SEQ,
                       vmin=0.30, vmax=1.70, shading='flat', rasterized=True)
    ax.contour(OM, DL, unst.astype(float), levels=[0.5], colors=[RED], linewidths=1.1)
    cs = ax.contour(OM, DL, np.where(unst, F, np.nan), levels=[0.5, 0.75, 1.0, 1.25, 1.5],
                    colors=['#ffffff'], linewidths=0.8, alpha=.85)
    ax.clabel(cs, fmt='%.2f', fontsize=6.5, inline=True)
    w = np.linspace(OM[0], OM[-1], 400)
    ax.plot(w, delta_c(w, 0.02), lw=1.3, ls=(0, (5, 3)), color=RED)
    ax.set_title(ttl, loc='left', fontsize=9.5); ax.set_ylim(0, DL[-1]); ax.set_xlim(OM[0], OM[-1])

    ax = AX[1, j]; ax.grid(False)
    lab = np.zeros_like(F); lab[unst] = 1
    for val, code in ((0.5, 2), (1.0, 3), (1.5, 4)):
        lab[unst & (np.abs(rho - val) < TOL)] = code
    ax.pcolormesh(oe, de, lab, cmap=ListedColormap([CSTABLE, CNONE, C1, C2, C3]),
                  norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5, 4.5], 5), shading='flat', rasterized=True)
    ax.set_xlabel('forcing frequency  $\\omega$')
    for n, c_ in ((1, C1), (2, C2), (3, C3)):
        for a in (AX[0, j], AX[1, j]):
            a.plot([2.0/n], [0], marker='v', ms=6, mfc=c_, mec='#ffffff', mew=.8, clip_on=False, zorder=6)
    st = [(lab == v).mean() for v in (2, 3, 4, 1)]
    ax.set_title('$\\rho$=1/2 %.0f%% | $\\rho$=1 %.0f%% | $\\rho$=3/2 %.1f%% | none %.0f%%'
                 % tuple(100*np.array(st)), loc='left', fontsize=8.2, color=MUTED)
AX[0, 0].set_ylabel('forcing strength  $\\delta$'); AX[1, 0].set_ylabel('forcing strength  $\\delta$')
# relief for the low-contrast categorical slot: direct labels on the map
AX[1, 0].annotate('$\\rho$ = 1/2\nn = 1\nresponse $\\omega$/2', (2.05, 1.55), color='#ffffff', fontsize=9, ha='center')
AX[1, 0].annotate('$\\rho$ = 1  (n = 2)', (1.30, 2.62), color=C2, fontsize=8, ha='left',
                  bbox=dict(fc=SURF, ec='none', pad=1.5))
AX[1, 0].annotate('$\\rho$ = 3/2\n(n = 3)', (0.42, 2.30), color='#0d7a53', fontsize=8, ha='left',
                  bbox=dict(fc=SURF, ec='none', pad=1.5))
cax = fig.add_axes([0.912, 0.482, 0.011, 0.365])
cb = fig.colorbar(im, cax=cax); cb.outline.set_visible(False)
cb.set_label('peak response frequency  $\\omega_{resp}$', fontsize=8.5)
cax.text(0.5, -0.075, 'grey = stable', transform=cax.transAxes, ha='center', fontsize=7.5, color=MUTED)
AX[1, 2].legend(handles=[Patch(fc=C1, label='$\\rho$ = 1/2   (n=1, response at $\\omega$/2)'),
                         Patch(fc=C2, label='$\\rho$ = 1   (n=2, response at $\\omega$)'),
                         Patch(fc=C3, label='$\\rho$ = 3/2 (n=3, response at 3$\\omega$/2)'),
                         Patch(fc=CNONE, label='unstable, no simple ratio'),
                         Patch(fc=CSTABLE, ec=GRID, label='stable (no growth)')],
                loc='upper right', fontsize=7.8, labelspacing=.35,
                facecolor=SURF, framealpha=.92, frameon=True, edgecolor=GRID)
fig.suptitle('P11 - The tongues coloured by RESPONSE FREQUENCY rather than growth rate.  '
             'Triangles mark the tongue tips at $\\omega$ = 2/n.', x=0.006, ha='left', fontsize=10.5)
plt.show()

# ============================================================================
#  P11 follow-up: is a plateau real?  Two tests that separate a lock from a
#  coincidence.  A lock is PINNED - rho must not move as delta changes.
# ============================================================================
print('Peak sharpness (fraction of power in the strongest bin) and')
print('drift of rho across delta, by candidate ratio:\n')
print(f'   {"panel":10s} {"ratio":10s} {"cells":>7s} {"sharpness":>10s} {"drift of rho":>13s}')
for tag, sg, rv, ttl in SETTINGS:
    F, C, G = TONGUES[tag]
    unst = G > 0.01; rho = F/OM[None, :]
    for val, lab in ((0.5, '1/2'), (1.0, '1'), (1.5, '3/2'), (1/3, '1/3'), (2/3, '2/3')):
        m = unst & (np.abs(rho - val) < TOL)
        if m.sum() < 3:
            print(f'   {tag:10s} {lab:10s} {m.sum():7d} {"-":>10s} {"-":>13s}'); continue
        drifts = []
        for j in range(len(OM)):
            if m[:, j].sum() < 3: continue
            k = np.where(unst[:, j])[0]
            if k.size < 8: continue
            drifts.append(rho[k, j].max() - rho[k, j].min())
        d = np.median(drifts) if drifts else np.nan
        print(f'   {tag:10s} {lab:10s} {m.sum():7d} {np.median(C[m]):10.3f} {d:13.4f}')
print('\nA genuine lock has sharpness ~0.5 and near-zero drift.  Candidates with')
print('sharpness ~0.04 are the argmax of a broad smear; ones whose rho drifts')
print('are a continuously varying response passing THROUGH the ratio, not sitting on it.')

# ============================================================================
#  The blanket sweep
# ============================================================================
# delta = 0 is IN the grid, so the null distribution comes from the same
# pipeline at the same run length - the only honest way to judge a coherence.

def run_sweep():
    OMs = CFG['sweep_om']; T, BURN = CFG['sweep_T'], CFG['sweep_burn']
    rows = []
    combos = list(itertools.product(CFG['sweep_eps'], CFG['sweep_sg'],
                                    CFG['sweep_r'], CFG['sweep_dl'], CFG['sweep_seeds']))
    for c, (eps, sg, rv, dl, seed) in enumerate(combos):
        t, X, LH, PH, _ = integrate(eps, sg, np.full(OMs.size, rv), np.full(OMs.size, dl),
                                    OMs, mode='parametric', T=T, dt=DT, thin=THIN, seed=seed)
        m = t >= BURN; tt = t[m]; ph = PH[m]
        gr = growth_rate(tt, LH[m])
        fr = np.abs((ph[-1] - ph[0])/(tt[-1] - tt[0]))
        c1 = coherence(tt, ph, OMs, 1.); c2 = coherence(tt, ph, OMs, 2.); c3 = coherence(tt, ph, OMs, 3.)
        for j, om in enumerate(OMs):
            per = cycle_periods(tt, ph[:, j:j+1])
            rows.append(dict(eps=eps, sigma=sg, r=rv, delta=dl, omega=om, seed=seed,
                             growth=gr[j], fresp=fr[j], coh1=c1[j], coh2=c2[j], coh3=c3[j],
                             per_med=np.median(per) if per.size >= 8 else np.nan,
                             per_bc=bimodality(per) if per.size >= 8 else np.nan,
                             ncyc=per.size))
        if (c+1) % max(1, len(combos)//6) == 0:
            print(f'   {c+1}/{len(combos)} cells', flush=True)
    d = pd.DataFrame(rows)
    d['rho'] = d.fresp/d.omega
    d['dc'] = delta_c(d.omega, d.eps)
    d['inside'] = d.delta > d.dc
    return d

t0 = time.time(); SW = run_sweep()
print(f'\n{len(SW):,} runs in {time.time()-t0:.0f}s   |  NaNs: {int(SW[["growth","coh2"]].isna().sum().sum())}')
SW.head()

# ============================================================================
#  P2  -  frequency locking, three measures that fail independently
# ============================================================================
# 1. TRACKING   regress the response on omega over a window fixed in advance.
#               A locked response gives slope 1/n; an unlocked one gives 0.
#               Fixing the window matters: selecting runs whose peak happens to
#               lie near a rising line biases the slope upward.
# 2. WINDING    rho = <dphi/dt>/omega.  A lock is a PLATEAU at a rational value
#               over a range of omega - not a crossing of one.  An unlocked
#               response gives rho = 1/omega, which crosses 1/2 at omega = 2
#               without being locked to it.
# 3. COHERENCE  R_n = |<exp i(phi - omega t/n)>|, against a null.
fig, ax = plt.subplots(1, 3, figsize=(13.4, 4.1))

DLS = [d for d in (0.0, 0.25, 1.0, 4.0) if d in CFG['sweep_dl']]
for dl, c in zip(DLS, (GREY, GREEN, BLUE, ORANGE)):
    g = SW[SW.delta == dl].groupby('omega').fresp.median()
    ax[0].plot(g.index, g.values, lw=1.6, color=c, label=f'$\\delta$={dl:g}')
w = np.linspace(CFG['sweep_om'][0], CFG['sweep_om'][-1], 200)
ax[0].plot(w, w/2, lw=1.6, ls=(0, (5, 3)), color=INK, label='$\\omega$/2')
ax[0].axhline(1.0, lw=1.0, ls=(0, (1, 2.5)), color=MUTED)
ax[0].set(xlabel='forcing frequency  $\\omega$', ylabel='response frequency  $\\langle d\\phi/dt\\rangle$',
          ylim=(0, 2.3)); ax[0].set_title('Response tracks $\\omega$/2 inside the tongue', loc='left')
ax[0].legend(ncol=2, loc='upper left')

def slope(g, lo=1.8, hi=2.2):
    o = g.omega.values; y = g.fresp.values; k = (o >= lo) & (o <= hi)
    return np.polyfit(o[k], y[k], 1)[0] if k.sum() >= 4 and np.isfinite(y[k]).all() else np.nan

cells = SW[SW.delta > 0].groupby(['eps', 'sigma', 'r', 'delta', 'seed'])
obs = cells.apply(slope).dropna().values
# The in-grid delta=0 control is DEGENERATE for this test: with no forcing omega
# does not enter the dynamics, so the slope is identically 0 with zero variance
# and cannot calibrate a threshold.  Permute the omega labels instead.
rng = np.random.default_rng(0); null = []
for _, g in cells:
    o = g.omega.values; y = rng.permutation(g.fresp.values); k = (o >= 1.8) & (o <= 2.2)
    if k.sum() >= 4 and np.isfinite(y[k]).all(): null.append(np.polyfit(o[k], y[k], 1)[0])
null = np.array(null)
bins = np.linspace(-0.9, 1.1, 60)
ax[1].hist(obs, bins=bins, color=BLUE, alpha=.85, label=f'forced (n={len(obs)})')
ax[1].hist(null, bins=bins, color=GREY, alpha=.7, label='permutation null')
ax[1].axvline(0.5, lw=1.6, ls=(0, (5, 3)), color=INK)
ax[1].text(0.5, ax[1].get_ylim()[1]*.97, ' 2:1 lock\n slope 1/2', va='top', fontsize=8, color=INK)
ax[1].set(xlabel='d(response)/d$\\omega$ over $\\omega\\in$[1.8, 2.2]', ylabel='parameter cells')
ax[1].set_title(f'slope $\\geq$ 0.45 in {100*np.mean(obs>=0.45):.0f}% of forced cells '
                f'vs {100*np.mean(null>=0.45):.1f}% of nulls', loc='left')
ax[1].legend(loc='upper left')

for dl, c in zip(DLS, (GREY, GREEN, BLUE, ORANGE)):
    g = SW[SW.delta == dl].groupby('omega').rho.median()
    ax[2].plot(g.index, g.values, lw=1.6, color=c, label=f'$\\delta$={dl:g}')
ax[2].axhline(0.5, lw=1.6, ls=(0, (5, 3)), color=INK)
ax[2].text(0.45, 0.53, '$\\rho$ = 1/2  (2:1 lock)', fontsize=8, color=INK)
ax[2].set(xlabel='forcing frequency  $\\omega$', ylabel='winding number  $\\rho$', ylim=(0, 2.3))
ax[2].set_title('A plateau forms at $\\rho$ = 1/2', loc='left'); ax[2].legend(ncol=2)
for a in ax: tidy(a)
fig.suptitle(f'P2 - Frequency locking under parametric forcing, {len(SW):,} runs',
             x=0.008, ha='left', fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()
print(f'forced slope: mean {obs.mean():.4f} median {np.median(obs):.4f} | null sd {null.std():.4f}')

# ============================================================================
#  P3  -  spectral maps.  Response frequency (y) against forcing frequency (x)
# ============================================================================
# Each column of a panel is ONE simulation: run at that omega, take the power
# spectrum of u = cos(phi), normalise the column to its own maximum.
#
# Reading it: pick an omega, run your eye up, find the dark pixel - that is the
# frequency the oscillator actually turns at.  A HORIZONTAL band means the
# response ignores the drive.  A band lying ON the orange omega/2 line means it
# is captured.  Note that every unlocked panel CROSSES the orange line at
# omega = 2 (where omega/2 = 1); the crossing means nothing, the stretch that
# follows the slope is the lock.
#
# White does not mean empty: each column is normalised to its own peak with a
# colour floor ~1/2500 of it, so a column whose power is all in one bin renders
# white with a single dark dot.  The emptiest-looking panels are the sharpest.
def spec_map(eps, sg, r_list, dl_list, seed=11):
    """All (delta, r) cells and all omega in ONE vectorised call."""
    OMs = CFG['map_om']; T, BURN = CFG['map_T'], CFG['map_burn']
    dl = np.repeat(dl_list, len(r_list)*len(OMs))
    rv = np.tile(np.repeat(r_list, len(OMs)), len(dl_list))
    wv = np.tile(OMs, len(dl_list)*len(r_list))
    t, X, LH, PH, _ = integrate(eps, sg, rv, dl, wv, mode='parametric',
                                T=T, dt=DT, thin=THIN, seed=seed)
    m = t >= BURN
    f, P = spectrum(t[m], unit_signal(PH[m]), fmin=0.0, fmax=3.0)
    nf = 220; edges = np.linspace(0, 3.0, nf+1)
    idx = np.clip(np.digitize(f, edges)-1, 0, nf-1)
    M = np.zeros((nf, P.shape[1])); np.add.at(M, idx, P)
    M /= (M.max(axis=0, keepdims=True) + 1e-300)
    return edges, M.reshape(nf, len(dl_list), len(r_list), len(OMs)), OMs

DL_S = [d for d in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)]
R_S = [0.0, 0.5, 0.9, 1.4]
t0 = time.time(); edges, MM, OMs = spec_map(0.05, 1.0, R_S, DL_S); print(f'{time.time()-t0:.0f}s')

fig, AX = plt.subplots(len(DL_S), len(R_S), figsize=(2.9*len(R_S), 2.0*len(DL_S)),
                       sharex=True, sharey=True)
for i, dl in enumerate(DL_S):
    for j, rv in enumerate(R_S):
        ax = AX[i, j]
        ax.pcolormesh(np.r_[OMs-0.025, OMs[-1]+0.025], edges, np.log10(np.maximum(MM[:, i, j], 1e-5)),
                      cmap=SEQ, vmin=-3.4, vmax=0, shading='flat', rasterized=True)
        w = np.linspace(OMs[0], OMs[-1], 200)
        ax.plot(w, w/2, lw=1.1, ls=(0, (4, 3)), color=ORANGE)
        ax.plot(w, w,   lw=1.1, ls=(0, (4, 3)), color=RED)
        ax.plot(w, w/3, lw=1.1, ls=(0, (2, 3)), color=GREEN)
        ax.axhline(1.0, lw=1.0, ls=(0, (1, 2)), color='#ffffff', alpha=.8)
        ax.set(xlim=(OMs[0], OMs[-1]), ylim=(0, 3.0)); ax.grid(False)
        ax.set_title(f'$\\delta$={dl:g}   r={rv:g}', fontsize=8.3, loc='left', pad=3)
        ax.set_xticks([0.5, 1, 2, 3, 4]); ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
        ax.label_outer()
for j in range(len(R_S)): AX[-1, j].set_xlabel('forcing frequency  $\\omega$')
for i in range(len(DL_S)): AX[i, 0].set_ylabel('response')
fig.suptitle('P3 - Response spectrum vs forcing frequency, $\\epsilon$=0.05, $\\sigma$=1.\n'
             'red $\\omega$ | orange $\\omega$/2 | green $\\omega$/3 | white $\\omega_0$=1.  '
             'Colour: log$_{10}$ power, each column normalised.', x=0.006, ha='left', fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.945]); plt.show()

# ============================================================================
#  P4  -  Welch-averaged spectra of cos(phi), linear axis, peaks marked
# ============================================================================
from scipy.signal import welch, find_peaks
OMS = [1.0, 1.5, 1.8, 2.0, 2.4, 3.0]
DLS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]

dl = np.repeat(DLS, len(OMS)); wv = np.tile(OMS, len(DLS))
t0 = time.time()
t, X, LH, PH, _ = integrate(0.05, 1.0, np.full(dl.size, 0.5), dl, wv, mode='parametric',
                            T=CFG['line_T'], dt=DT, thin=THIN, seed=11)
m = t >= CFG['line_burn']
f, P = welch(unit_signal(PH[m]), fs=1.0/(t[1]-t[0]), nperseg=CFG['nperseg'],
             noverlap=CFG['nperseg']//2, detrend='constant', axis=0)
wgrid = 2*np.pi*f; keep = (wgrid > 0.12) & (wgrid < 4.6)
wgrid, P = wgrid[keep], P[keep]
print(f'{time.time()-t0:.0f}s')

fig, AX = plt.subplots(len(DLS), len(OMS), figsize=(2.7*len(OMS), 1.8*len(DLS)),
                       sharex=True, sharey=True)
for i, d_ in enumerate(DLS):
    for j, om in enumerate(OMS):
        ax = AX[i, j]; y = P[:, i*len(OMS)+j]; y = y/y.max()
        for v, c in ((om, RED), (om/2, ORANGE), (om/3, GREEN), (1.0, MUTED)):
            if v < 4.6: ax.axvline(v, lw=1.1, ls=(0, (4, 3)), color=c, alpha=.85)
        ax.plot(wgrid, y, lw=1.4, color=BLUE); ax.fill_between(wgrid, 0, y, color=BLUE, alpha=.13)
        pk, _ = find_peaks(y, height=0.08, prominence=0.06, distance=6)
        for q in pk[:5]:
            ax.plot(wgrid[q], y[q], 'o', ms=4, mfc='#fff', mec=INK, mew=1.1, zorder=5)
            ax.annotate(f'{wgrid[q]:.2f}', (wgrid[q], y[q]), (0, 4), textcoords='offset points',
                        ha='center', fontsize=6.6, color=INK)
        ax.set(ylim=(0, 1.22), xlim=(0, 4.6)); ax.set_yticks([0, 0.5, 1.0])
        ax.set_title(f'$\\delta$={d_:g},  $\\omega$={om:g}', fontsize=8.2, loc='left', pad=3)
        tidy(ax); ax.label_outer()
for j in range(len(OMS)): AX[-1, j].set_xlabel('response frequency')
for i in range(len(DLS)): AX[i, 0].set_ylabel('power (norm.)')
fig.suptitle('P4 - Welch spectra of the phase signal cos $\\phi$, $\\epsilon$=0.05, $\\sigma$=1, r=0.5.\n'
             'Read the bottom row: the peak sits at $\\omega$/2 for every $\\omega$.',
             x=0.006, ha='left', fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.955]); plt.show()

# ============================================================================
#  P5  -  sample paths.  ONE response cycle per TWO drive cycles is the lock.
# ============================================================================
eps, sg, rv, om = 0.05, 1.0, 0.5, 2.5
CASES = [(0.0, '$\\delta$ = 0   (unforced)'),
         (0.25, f'$\\delta$ = 0.25   (below threshold, $\\delta_c \\approx$ {delta_c(om, eps):.2f})'),
         (1.5, '$\\delta$ = 1.5   (above threshold)'),
         (4.0, '$\\delta$ = 4   (deep inside the tongue)')]
fig, AX = plt.subplots(len(CASES), 2, figsize=(13.2, 2.35*len(CASES)),
                       gridspec_kw={'width_ratios': [2.4, 1]})
for i, (dl, lab) in enumerate(CASES):
    t, X, LH, PH, _ = integrate(eps, sg, [rv], [dl], [om], mode='parametric',
                                T=900., dt=DT, thin=20, seed=5)
    u = unit_signal(PH[:, 0]); m = (t >= 400) & (t <= 460)
    AX[i, 0].plot(t[m], u[m], lw=1.3, color=BLUE, label='X / |X|  (unit amplitude)')
    AX[i, 0].plot(t[m], 0.55*np.sin(om*t[m]), lw=1.1, color=RED, alpha=.75,
                  label=f'drive sin($\\omega$t), $\\omega$={om}')
    AX[i, 0].axhline(0, lw=.7, color=GRID); AX[i, 0].set(ylim=(-1.35, 1.35), xlim=(400, 460))
    AX[i, 0].set_title(lab, loc='left', fontsize=9)
    if i == 0: AX[i, 0].legend(ncol=2, loc='lower right')
    AX[i, 1].plot(t, 0.5*LH[:, 0]/np.log(10), lw=1.4, color=ORANGE)
    AX[i, 1].set_title(f'amplitude,  growth rate $\\lambda$ = {np.polyfit(t, LH[:,0], 1)[0]/2:+.4f}',
                       loc='left', fontsize=9)
    AX[i, 1].set_ylabel('log$_{10}$ amplitude')
    for a in AX[i]: tidy(a)
AX[-1, 0].set_xlabel('time  t'); AX[-1, 1].set_xlabel('time  t')
fig.suptitle(f'P5 - Sample paths at $\\omega$ = {om}.  A 2:1 lock means response period '
             f'4$\\pi$/$\\omega$ = {4*np.pi/om:.3f} against drive period 2$\\pi$/$\\omega$ = {2*np.pi/om:.3f}.',
             x=0.006, ha='left', fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.94]); plt.show()

# ============================================================================
#  P6  -  distributions.  Above threshold the stationary state is GONE.
# ============================================================================
eps, sg, rv = 0.05, 1.0, 0.5
fig, AX = plt.subplots(2, 3, figsize=(13.4, 7.2))

# (a) below threshold the exact stationary density survives.  At omega = 3,
#     delta_c = 2*sqrt(1+eps^2) = 2.00, so delta = 0, 0.5, 1, 1.5 are all below.
DLS = [0.0, 0.5, 1.0, 1.5]
t, X, LH, PH, _ = integrate(eps, sg, np.full(4, rv), DLS, np.full(4, 3.0),
                            mode='parametric', T=8000., dt=DT, thin=THIN, seed=3)
hh = np.exp(LH[t >= 2000])
edges = np.linspace(0, 12, 61); c = 0.5*(edges[1:] + edges[:-1])
for j, dl in enumerate(DLS):
    y, _ = np.histogram(hh[:, j], bins=edges, density=True)
    AX[0, 0].step(c, y, where='mid', lw=1.4, label=f'$\\delta$={dl:g}')
AX[0, 0].plot(c, ps_exact(c, sg, rv), lw=2.0, ls=(0, (5, 3)), color=INK, label='exact  $p_s(h)$')
AX[0, 0].set(xlabel='energy  h', ylabel='density', xlim=(0, 12))
AX[0, 0].set_title('Below threshold the stationary density survives ($\\omega$=3)', loc='left')
AX[0, 0].legend()

# (b) crossing the threshold destroys it.  At omega = 2, delta_c = 2*eps = 0.10.
DL2 = [0.02, 0.05, 0.10, 0.20, 0.50]
t, X, LH, PH, _ = integrate(eps, sg, np.full(5, rv), DL2, np.full(5, 2.0),
                            mode='parametric', T=3000., dt=DT, thin=THIN, seed=3)
for j, dl in enumerate(DL2):
    AX[0, 1].plot(t, 0.5*LH[:, j]/np.log(10), lw=1.2, label=f'$\\delta$={dl:g}')
AX[0, 1].axhline(0, lw=.8, color=GRID)
AX[0, 1].set(xlabel='time  t', ylabel='log$_{10}$ amplitude')
AX[0, 1].set_title('$\\delta_c$ = 2$\\epsilon$ = 0.10 at $\\omega$=2: above it h grows without bound', loc='left')
AX[0, 1].legend(ncol=2)

# (c) growth rate across the sweep, split at the analytic boundary
bins = np.linspace(-0.05, 0.9, 70)
AX[0, 2].hist(SW.growth[~SW.inside], bins=bins, color=GREY, alpha=.8,
              label=f'outside tongue (n={(~SW.inside).sum():,})')
AX[0, 2].hist(SW.growth[SW.inside], bins=bins, color=BLUE, alpha=.8,
              label=f'inside tongue (n={SW.inside.sum():,})')
AX[0, 2].axvline(0, lw=1.2, ls=(0, (4, 3)), color=INK)
AX[0, 2].set(xlabel='amplitude growth rate  $\\lambda$', ylabel='runs', yscale='log')
AX[0, 2].set_title('Growth rate separates at the Mathieu boundary', loc='left'); AX[0, 2].legend()

# (d) cycle periods collapse onto 4*pi/omega
om = 2.5
for dl, c_, lab in ((0.25, GREY, '$\\delta$=0.25 (outside)'), (1.5, BLUE, '$\\delta$=1.5 (inside)'),
                    (4.0, ORANGE, '$\\delta$=4 (inside)')):
    t, X, LH, PH, _ = integrate(eps, sg, np.full(8, rv), np.full(8, dl), np.full(8, om),
                                mode='parametric', T=6000., dt=DT, thin=THIN, seed=101)
    per = cycle_periods(t[t >= 1500], PH[t >= 1500])
    AX[1, 0].hist(per, bins=np.linspace(1.5, 10, 120), density=True, color=c_, alpha=.72, label=lab)
AX[1, 0].axvline(4*np.pi/om, lw=1.5, ls=(0, (5, 3)), color=RED)
AX[1, 0].text(4*np.pi/om+0.1, AX[1, 0].get_ylim()[1]*.9, '4$\\pi$/$\\omega$ = %.3f' % (4*np.pi/om),
              color=RED, fontsize=8)
AX[1, 0].axvline(2*np.pi, lw=1.5, ls=(0, (1, 2.5)), color=INK)
AX[1, 0].text(2*np.pi+0.1, AX[1, 0].get_ylim()[1]*.7, '2$\\pi$ (unforced)', color=INK, fontsize=8)
AX[1, 0].set(xlabel='cycle period', ylabel='density', xlim=(1.5, 10))
AX[1, 0].set_title(f'Cycle periods collapse onto 4$\\pi$/$\\omega$ ($\\omega$={om})', loc='left')
AX[1, 0].legend()

# (e) relative error of the median period against 4*pi/omega
q = SW[(SW.delta > 0) & SW.per_med.notna()]
rel = np.abs(q.per_med - 4*np.pi/q.omega)/(4*np.pi/q.omega)
bb = np.linspace(0, 1.0, 60)
AX[1, 1].hist(rel[~q.inside], bins=bb, color=GREY, alpha=.8, label='outside tongue')
AX[1, 1].hist(rel[q.inside], bins=bb, color=BLUE, alpha=.8, label='inside tongue')
AX[1, 1].set(xlabel='|median period $-$ 4$\\pi$/$\\omega$| / (4$\\pi$/$\\omega$)', ylabel='runs', yscale='log')
AX[1, 1].set_title('%.0f%% of inside-tongue runs within 1%% of 4$\\pi$/$\\omega$'
                   % (100*np.mean(rel[q.inside] < 0.01)), loc='left')
AX[1, 1].legend()

# (f) bimodality of the period distribution
bb = np.linspace(0, 1, 50)
AX[1, 2].hist(SW.per_bc[~SW.inside].dropna(), bins=bb, color=GREY, alpha=.8, label='outside tongue')
AX[1, 2].hist(SW.per_bc[SW.inside].dropna(), bins=bb, color=BLUE, alpha=.8, label='inside tongue')
AX[1, 2].axvline(5/9, lw=1.5, ls=(0, (5, 3)), color=RED)
AX[1, 2].set(xlabel="Sarle bimodality coefficient", ylabel='runs', yscale='log')
AX[1, 2].set_title('BC > 5/9 in %.0f%% inside vs %.0f%% outside'
                   % (100*np.mean(SW.per_bc[SW.inside].dropna() > 5/9),
                      100*np.mean(SW.per_bc[~SW.inside].dropna() > 5/9)), loc='left')
AX[1, 2].legend()
for a in AX.ravel(): tidy(a)
fig.suptitle('P6 - Distributions under parametric forcing', x=0.006, ha='left', fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.955]); plt.show()

# ============================================================================
#  P7  -  phase coherence and the locking threshold
# ============================================================================
SW['ratio'] = SW.delta/SW.dc
fig, AX = plt.subplots(1, 3, figsize=(13.4, 4.2))

sub = SW[(SW.eps == CFG['sweep_eps'][0]) & (SW.r == CFG['sweep_r'][-1])]
piv = sub.pivot_table(index='delta', columns='omega', values='coh2', aggfunc='mean')
om = piv.columns.values; dl = piv.index.values
step = om[1]-om[0]
im = AX[0].pcolormesh(np.r_[om-step/2, om[-1]+step/2], np.arange(len(dl)+1)-0.5,
                      piv.values, cmap=SEQ, vmin=0, vmax=1, shading='flat')
AX[0].set_yticks(range(len(dl))); AX[0].set_yticklabels([f'{v:g}' for v in dl])
AX[0].set(xlabel='forcing frequency  $\\omega$', ylabel='forcing strength  $\\delta$'); AX[0].grid(False)
AX[0].set_title('2:1 phase coherence  $R_2$', loc='left')
w = np.linspace(om[0], om[-1], 300)
AX[0].plot(w, np.interp(delta_c(w, CFG['sweep_eps'][0]), dl, np.arange(len(dl))),
           lw=1.8, ls=(0, (5, 3)), color=RED)
cb = fig.colorbar(im, ax=AX[0], pad=0.02); cb.outline.set_visible(False)

ins = SW[(SW.delta > 0) & (SW.ratio < 4)]
b = np.linspace(0, 4, 33); k = np.clip(np.digitize(ins.ratio, b)-1, 0, 31)
cb_ = 0.5*(b[1:]+b[:-1])
def band(col, q):
    return np.array([np.quantile(ins[col].values[k == i], q) if (k == i).sum() > 5 else np.nan
                     for i in range(32)])
AX[1].fill_between(cb_, band('coh2', .25), band('coh2', .75), color=BLUE, alpha=.2)
AX[1].plot(cb_, band('coh2', .5), lw=2.0, color=BLUE, label='$R_2$  (median, IQR)')
nullmax = SW.coh2[SW.delta == 0].max()
AX[1].axhline(nullmax, lw=1.4, ls=(0, (5, 3)), color=GREY)
AX[1].text(3.9, nullmax+0.02, f'largest $R_2$ over {(SW.delta==0).sum():,} unforced runs',
           ha='right', fontsize=7.5, color=MUTED)
AX[1].axvline(1.0, lw=1.4, ls=(0, (4, 3)), color=RED)
AX[1].text(1.05, 0.94, 'threshold', color=RED, fontsize=8)
AX[1].set(xlabel='$\\delta$ / $\\delta_c$', ylabel='2:1 phase coherence  $R_2$', ylim=(0, 1.0), xlim=(0, 4))
AX[1].set_title('Coherence turns on at the instability threshold', loc='left')
AX[1].legend(loc='lower right')

for lab, col, c_ in (('$R_1$ (1:1)', 'coh1', GREEN), ('$R_2$ (2:1)', 'coh2', BLUE),
                     ('$R_3$ (3:1)', 'coh3', ORANGE)):
    AX[2].plot(cb_, band(col, .5), lw=2.0, color=c_, label=lab)
AX[2].axvline(1.0, lw=1.4, ls=(0, (4, 3)), color=RED)
AX[2].set(xlabel='$\\delta$ / $\\delta_c$', ylabel='phase coherence (median)', ylim=(0, 1.0), xlim=(0, 4))
AX[2].set_title('Only the 2:1 index responds', loc='left'); AX[2].legend()
for a in AX: tidy(a)
fig.suptitle('P7 - Phase coherence and the locking threshold', x=0.006, ha='left', fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()
print(f'null max R2 {nullmax:.4f} | inside-tongue median R2 {SW.coh2[SW.inside].median():.4f} '
      f'| median R3 {SW.coh3[SW.inside].median():.4f}')

# ============================================================================
#  P8  -  the threshold and growth rate against Mathieu theory
# ============================================================================
fig, AX = plt.subplots(1, 3, figsize=(13.4, 4.2))

def boundary(G, thr=0.01):
    """Lowest delta at which the growth rate rises above thr AND STAYS above it.
    A bare G>0 test is unusable once noise is on: for a stationary run lambda
    fluctuates about zero and crosses it by chance, placing the boundary at 0."""
    out = np.full(len(OM), np.nan)
    for j in range(len(OM)):
        up = G[:, j] > thr
        keep = up & np.minimum.accumulate(up[::-1])[::-1]
        k = np.where(keep)[0]
        if k.size and k[0] > 0:
            i = k[0]; g0, g1 = G[i-1, j], G[i, j]
            out[j] = DL[i-1] + (DL[i]-DL[i-1])*(thr-g0)/(g1-g0)
        elif k.size: out[j] = DL[0]
    return out

th = delta_c(OM, 0.02)
for (tag, sg, rv, ttl), c_ in zip(SETTINGS, (INK, BLUE, ORANGE)):
    b = boundary(TONGUES[tag][2])
    AX[0].plot(OM, b, lw=1.7, color=c_, label=ttl.replace('$\\sigma$', 'σ'))
    k = (OM > 1.2) & (OM < 2.8)
    print(f'{tag:9s} median |measured - theory| over 1.2<w<2.8 = '
          f'{np.nanmedian(np.abs(b[k]-th[k])):.4f}   (delta grid step {DL[1]-DL[0]:.3f})')
AX[0].plot(OM, th, lw=2.0, ls=(0, (5, 3)), color=RED, label='2$\\sqrt{(\\omega-2)^2+\\epsilon^2}$')
AX[0].set(xlabel='forcing frequency  $\\omega$', ylabel='measured threshold  $\\delta_c$',
          xlim=(1.0, 3.2), ylim=(0, 2.6))
AX[0].set_title('Measured tongue boundary, $\\epsilon$=0.02', loc='left'); AX[0].legend(fontsize=7.5)

q = SW[(SW.delta > 0) & (SW.ratio < 3)]
b = np.linspace(0, 3, 31); kk = np.clip(np.digitize(q.ratio, b)-1, 0, 29)
p = np.array([np.mean(q.growth.values[kk == i] > 0.005) if (kk == i).sum() > 10 else np.nan
              for i in range(30)])
AX[1].plot(0.5*(b[1:]+b[:-1]), p, lw=2.0, color=BLUE)
AX[1].axvline(1.0, lw=1.5, ls=(0, (4, 3)), color=RED)
AX[1].text(1.04, 0.12, 'predicted threshold', color=RED, fontsize=8)
AX[1].set(xlabel='$\\delta$ / $\\delta_c$', ylabel='P(growth rate > 0.005)', ylim=(-0.02, 1.02), xlim=(0, 3))
AX[1].set_title('Instability switches on where theory says', loc='left')

qq = SW[(SW.delta > 0) & (np.abs(SW.omega - 2.0) < 1.0)]
D = (qq.omega - 2.0).values; dl_ = qq.delta.values
pred = -qq.eps.values/2 + np.sqrt(np.maximum((dl_/4)**2 - (D/2)**2, 0))
AX[2].scatter(pred, qq.growth.values, s=5, alpha=.25, color=BLUE, edgecolors='none')
AX[2].plot([-0.1, 1.1], [-0.1, 1.1], lw=1.6, ls=(0, (5, 3)), color=INK)
ok = pred > 0.02
cc = np.corrcoef(pred[ok], qq.growth.values[ok])[0, 1]
AX[2].set(xlabel='linear theory:  $-\\epsilon$/2 + $\\sqrt{(\\delta/4)^2-(\\Delta/2)^2}$',
          ylabel='measured growth rate  $\\lambda$', xlim=(-0.05, 1.0), ylim=(-0.05, 1.0))
AX[2].set_title(f'Measured vs predicted growth rate (r = {cc:.3f})', loc='left')
for a in AX: tidy(a)
fig.suptitle('P8 - The threshold and growth rate against Mathieu theory', x=0.006, ha='left', fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()
print(f'P(unstable) below 0.5*dc = {np.mean(q.growth[q.ratio<0.5]>0.005):.4f} ; '
      f'above 1.3*dc = {np.mean(q.growth[q.ratio>1.3]>0.005):.4f} ; growth correlation {cc:.4f}')

# ============================================================================
#  P9  -  the full 3D system, before the reduction
# ============================================================================
#   dX = V dt
#   dV = -( (A_t^2 + delta sin wt) X + eps V ) dt + sqrt(eps) sigma dB
#   dA = -(A/tau) dt + sqrt(2/tau) dW        (OU, stationary N(0,1))
#
# tau is the CORRELATION TIME of the random frequency, measured against the orbit
# period 2*pi.  The reduced model is the tau -> 0 limit, where averaging replaces
# A^2 by its mean 1 plus a noise term.  Since the reduction is an approximation,
# it is worth asking whether the lock is a property of the physical system or of
# the limit.
#
# NOTE the forcing here is PARAMETRIC - inside the stiffness, matching the rest
# of this notebook.  mode='full3d_param' uses  A_t^2 + delta sin(wt)  as the
# stiffness; mode='full3d' would instead ADD delta sin(wt), which is the
# companion (additive) notebook's model.
eps, sg = 0.05, 1.0
TAUS = [0.05, 0.2, 0.5, 1.0, 2.0, 5.0]
OM3 = CFG['d3_om']; DL3 = [0.0, 1.0, 4.0]

t0 = time.time()
tv = np.repeat(TAUS, len(DL3)*len(OM3))
dv = np.tile(np.repeat(DL3, len(OM3)), len(TAUS))
wv = np.tile(OM3, len(TAUS)*len(DL3))
t, X, LH, PH, A2 = integrate(eps, sg, None, dv, wv, mode='full3d_param', tau=tv,
                             T=CFG['d3_T'], dt=DT, thin=THIN, seed=11)
m = t >= CFG['d3_burn']; tt = t[m]; ph = PH[m]
FR = np.abs((ph[-1]-ph[0])/(tt[-1]-tt[0]))
R2 = coherence(tt, ph, wv, 2.)
GR = growth_rate(tt, LH[m])
print(f'{time.time()-t0:.0f}s   <A^2> = {A2.mean():.3f} (should be ~1)')
shp = (len(TAUS), len(DL3), len(OM3))
FR, R2, GR = FR.reshape(shp), R2.reshape(shp), GR.reshape(shp)

fig, AX = plt.subplots(3, len(TAUS), figsize=(2.35*len(TAUS), 6.6), sharex=True)
for j, tau in enumerate(TAUS):
    for i, (dl, c_) in enumerate(zip(DL3, (GREY, BLUE, ORANGE))):
        AX[0, j].plot(OM3, FR[j, i], lw=1.4, color=c_, label=f'$\\delta$={dl:g}')
        AX[1, j].plot(OM3, R2[j, i], lw=1.4, color=c_)
        AX[2, j].plot(OM3, GR[j, i], lw=1.4, color=c_)
    AX[0, j].plot(OM3, OM3/2, lw=1.4, ls=(0, (5, 3)), color=INK)
    AX[0, j].set(ylim=(0, 2.3)); AX[1, j].set(ylim=(0, 1.0))
    AX[0, j].set_title(f'$\\tau$ = {tau:g}', loc='left', fontsize=9)
    AX[2, j].axhline(0, lw=.8, color=GRID); AX[2, j].set_xlabel('forcing frequency  $\\omega$')
    for i in range(3):
        tidy(AX[i, j])
        if j: AX[i, j].set_yticklabels([])
AX[0, 0].set_ylabel('$\\langle d\\phi/dt\\rangle$  (dashed $\\omega$/2)', fontsize=8.5)
AX[1, 0].set_ylabel('2:1 coherence  $R_2$', fontsize=8.5)
AX[2, 0].set_ylabel('growth rate  $\\lambda$', fontsize=8.5)
AX[0, 0].legend(fontsize=7.5)
fig.suptitle('P9 - The full 3D system with parametric forcing.  $\\tau$ is the correlation time of the\n'
             'random frequency $A_t$; the reduced model is the $\\tau\\to$0 limit.', x=0.006, ha='left', fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.93]); plt.show()

k = (OM3 > 1.6) & (OM3 < 2.6)
print('\ntau     mean R2 at delta=0 / 1 / 4      mean |f - w/2|/(w/2) at delta=4')
for j, tau in enumerate(TAUS):
    e4 = np.mean(np.abs(FR[j, 2][k] - OM3[k]/2)/(OM3[k]/2))
    print(f'{tau:5.2f}   {R2[j,0][k].mean():.3f}  {R2[j,1][k].mean():.3f}  {R2[j,2][k].mean():.3f}'
          f'          {e4:.4f}')
print('\nFrequency entrainment survives at every tau; the RIGIDITY of the phase')
print('relation does not.  A rigid lock needs the frequency noise to be fast')
print('compared with the orbit period, i.e. tau << 2*pi.')