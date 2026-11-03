"""
Is the two-layer DIURNAL current in Penn Cove tidal, or a daily (S1) wind /
forcing signal? The 73 h band fits lump K1, O1, P1 and S1 together. Two years
separate S1 (24.00 h) from K1 (23.93 h) and P1 (24.07 h) (Rayleigh 365 d), so a
full-record utide fit with S1 added says how much of the daily current is
locked to the sun's clock rather than to the tide.

Daily non-tidal candidates that would land on S1: the sea breeze, and the
once-a-day seam in the atm00 wind forcing at 00 UTC ([[wb1-wind-data-inventory]]).

Fits (utide, 2024-2025 hourly, nodal on, OLS):
  currents  every sigma layer of the pc_cove box on two N-S sections, the
            mouth (last rho column west of pc_lp) and pc_cp; faces averaged
            to rho as in 20261008_tidal_ellipses.py
  zeta      box-mean (S1 elevation should be ~mm)
  wind      hourly along- and cross-cove wind (u_pc, v_pc, 20260806_wind_reduce.py)
Constituents: the 11 of 20261008 + S1.

Output: LO_output/DM_outs/20261027_pc_diurnal_S1/
  pc_diurnal_S1.png  section-mean amplitude vs depth (K1, O1, P1, S1, M2) and
                     phase vs depth (K1, S1; major axis oriented eastward)
  report.txt, profiles.csv

  python 20261027_pc_diurnal_S1.py
"""

import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(_v, '1')
import warnings
import multiprocessing as mp

import numpy as np
import pandas as pd
import xarray as xr
import utide
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun

warnings.filterwarnings('ignore')
GTX, JOB, DS0, DS1 = 'wb1_t0_xn11abbur00', 'pc_cove', '2024.01.01', '2025.12.31'
Ldir = Lfun.Lstart(gridname='wb1')
box_fn = Ldir['LOo'] / 'extract' / GTX / 'box' / ('%s_%s_%s.nc' % (JOB, DS0, DS1))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261027_pc_diurnal_S1'
Lfun.make_dir(out_dir)
sect_dir = Ldir['LOo'] / 'section_lines'
CONS = ['M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1', 'M4', 'MS4', 'M6', 'S1']
KW = dict(method='ols', conf_int='linear', constit=CONS, nodal=True, trend=False, verbose=False)
SHOW = ['M2', 'K1', 'O1', 'P1', 'S1']
COL = {'M2': '#0072B2', 'K1': '#D55E00', 'O1': '#E69F00', 'P1': '#CC79A7', 'S1': '#009E73'}
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
AX, AY = -0.9789, -0.2043

ds = xr.open_dataset(box_fn)
T = ds.ocean_time.values
lon, lat = ds.lon_rho.values, ds.lat_rho.values
mask = ds.mask_rho.values == 1
LON_CP = pd.read_pickle(sect_dir / 'pc_cp.p').x.mean()
LON_LP = pd.read_pickle(sect_dir / 'pc_lp.p').x.mean()
SECTS = [('mouth', int(np.flatnonzero(lon[0] < LON_LP)[-1])), ('pc_cp', int(np.argmin(np.abs(lon[0] - LON_CP))))]
NZ = ds.sizes['s_rho']
Cs, s_rho, hc = ds.Cs_r.values, ds.s_rho.values, float(ds.hc)

SER = [] # (section, j, k, u, v)
for nm, i in SECTS:
    jw = [j for j in np.flatnonzero(mask[:, i]) if 0 < j < mask.shape[0] - 1]
    u = np.nan_to_num(ds.u[:, :, :, i - 1:i + 1].values).mean(axis=-1) # (T, NZ, NR) at rho col i
    v = np.nan_to_num(ds.v[:, :, :, i].values) # (T, NZ, NR-1)
    for j in jw:
        vr = 0.5 * (v[:, :, j - 1] + v[:, :, j])
        for k in range(NZ):
            SER.append((nm, j, k, u[:, k, j], vr[:, k]))
    print('%s: col %d, %d cells' % (nm, i, len(jw)), flush=True)
ZETA = np.nanmean(ds.zeta.values[:, mask], axis=1)
ds.close()

def fit(n):
    nm, j, k, u, v = SER[n]
    c = utide.solve(T, u.astype(float), v.astype(float), lat=float(lat[j, 0]), **KW)
    out = {}
    for con in SHOW:
        q = list(c.name).index(con)
        th, g = c.theta[q], c.g[q]
        if th > 90: # orient the major axis eastward so phases are comparable
            th, g = th - 180, (g + 180) % 360
        out[con] = (c.Lsmaj[q], c.Lsmaj_ci[q], g)
    return nm, j, k, out

with mp.get_context('fork').Pool(8) as pool:
    R = pool.map(fit, range(len(SER)), chunksize=20)

rows = []
for nm, j, k, out in R:
    for con, (A, ci, g) in out.items():
        rows.append(dict(sect=nm, j=j, k=k, con=con, A=A, ci=ci, g=g))
P = pd.DataFrame(rows)
P.to_csv(out_dir / 'profiles.csv', index=False, float_format='%.5f')

report = []
cz = utide.solve(T, ZETA.astype(float), lat=48.23, **KW)
report.append('box-mean zeta amplitude [cm]: ' + ', '.join('%s %.1f' % (n, 100 * cz.A[list(cz.name).index(n)]) for n in SHOW))
W = pd.read_pickle(Ldir['LOo'] / 'DM_outs' / '20260806_wind' / 'wind_hourly_atm00_2024.01.01_2025.12.31.p')['W']
W = W.reindex(pd.DatetimeIndex(T)).interpolate(limit=6)
wa, wc = (W.u_pc * AX + W.v_pc * AY).values, (W.u_pc * AY - W.v_pc * AX).values
ok = np.isfinite(wa) & np.isfinite(wc)
for lab, x in [('along-cove (+ toward head)', wa), ('cross-cove (+ north)', wc)]:
    cw = utide.solve(T[ok], x[ok].astype(float), lat=48.23, **KW)
    report.append('wind %s amplitude [m/s]: ' % lab + ', '.join('%s %.2f' % (n, cw.A[list(cw.name).index(n)]) for n in SHOW) + '; S1 phase %.0f deg (Greenwich)' % cw.g[list(cw.name).index('S1')])

def sect_profile(nm):
    """section-mean amplitude (and amplitude-weighted circular phase) per layer"""
    Q = P[P.sect == nm]
    out = {}
    for con in SHOW:
        q = Q[Q.con == con]
        A = q.groupby('k').A.mean().values
        ci = q.groupby('k').ci.mean().values
        g = np.rad2deg(np.angle((q.A * np.exp(1j * np.deg2rad(q.g))).groupby(q.k).sum().values)) % 360
        out[con] = (A, ci, g)
    return out

fig, axs = plt.subplots(2, 2, figsize=(12, 10), layout='constrained')
for col, (nm, i) in enumerate(SECTS):
    jw = sorted(P[P.sect == nm].j.unique())
    h = np.array([float(xr.open_dataset(box_fn).h.values[j, i]) for j in jw])
    z = np.array([np.mean(h * (hc * s + c * h) / (hc + h)) for s, c in zip(s_rho, Cs)]) # section-mean layer depth at zeta = 0
    pr = sect_profile(nm)
    ax = axs[0, col]
    for con in SHOW:
        A, ci, g = pr[con]
        ax.fill_betweenx(z, 100 * (A - ci), 100 * (A + ci), color=COL[con], alpha=0.15, lw=0)
        ax.plot(100 * A, z, '-', color=COL[con], lw=2, label=con)
    ax.set_xlabel('section-mean semi-major axis [cm/s] (band = 95% CI)')
    ax.set_ylabel('depth at mean sea level [m]')
    ax.set_title('%s: amplitude by depth' % nm, fontsize=10, loc='left')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(**GRID)
    ax = axs[1, col]
    for con in ['K1', 'O1', 'S1']:
        A, ci, g = pr[con]
        ax.plot(g, z, 'o-', ms=3, color=COL[con], lw=2, label=con)
    ax.set_xlim(0, 360)
    ax.set_xticks(range(0, 361, 90))
    ax.set_xlabel('Greenwich phase of max EASTWARD (out) current [deg]')
    ax.set_ylabel('depth at mean sea level [m]')
    ax.set_title('%s: phase by depth (a 180 deg jump = layers flow opposite ways)' % nm, fontsize=10, loc='left')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(**GRID)
    for con in SHOW:
        A, ci, g = pr[con]
        report.append('%-6s %s: surface %.1f cm/s (phase %.0f), min %.1f at %.1f m, bed %.1f cm/s (phase %.0f)'
                      % (nm, con, 100 * A[-1], g[-1], 100 * A.min(), z[np.argmin(A)], 100 * A[0], g[0]))
for k, ax in enumerate(axs.flat):
    ax.text(0.02, 0.97, 'abcd'[k], transform=ax.transAxes, fontsize=13, fontweight='bold', va='top')
fig.suptitle('Diurnal currents with S1 (24.00 h, solar clock) separated from K1/O1/P1: 2-year utide fits, %s' % GTX, fontsize=11)
fig.savefig(out_dir / 'pc_diurnal_S1.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
(out_dir / 'report.txt').write_text('\n'.join(report) + '\n')
print('\n'.join(report))
