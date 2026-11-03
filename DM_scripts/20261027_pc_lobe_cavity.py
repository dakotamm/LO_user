"""
Is the inner Penn Cove counter-clockwise cell DRIVEN BY THE INFLOW passing its
open north side (a cavity flow), as opposed to tidal rectification (ruled out:
the cell's residual vorticity is the same at spring and neap, see
20261027_tidal_ellipses_windowed.py) or rotation (the cell is ~0.1 internal
Rossby radii, so no geostrophic / thermal-wind balance)?

Cavity prediction: the cell's strength follows the strength of the inflow that
drags it, at the same depth, with the cell LAGGING the inflow (spin-up), and
this holds after the shared seasonal cycle is removed.

FIELD: pc_cove box, u and v averaged over depth bands below the moving surface
(0-4, 4-10, 10-bed m, plus the depth mean), Godin-filtered -- the same banding
as 20261021_pc_eddy_center.py (a band exists in a cell only if its mean column
depth is >= band top + 2 m; faces next to cells without it are walls).

  LOBE   mean relative vorticity over interior psi points within -R m of the
         eddy centre (-122.708, 48.226; 20261021_pc_eddy_center.py), in units
         of f. A FIXED disk, so no detection threshold enters.
  INFLOW mean westward velocity (-u) on the u faces of the section at pc_cp
         (upstream/east of the lobe) that are on the INFLOW side of that band's
         2-year mean flow (mean u < 0). The disk and the section do not overlap.

Both hourly -> sampled daily at 12 UTC. Correlations raw and as anomalies from
a 31-day running mean (removes the shared seasonal cycle); lagged correlation
corr(LOBE(t + lag), INFLOW(t)), so a peak at lag > 0 means the lobe lags.
n_eff from the lag-1 autocorrelations (Bretherton et al. 1999).

Output: LO_output/DM_outs/20261027_pc_lobe_cavity/
  pc_lobe_cavity.png   per band: standardized series, scatter of anomalies,
                       lagged correlation
  pc_lobe_cavity_daily.csv, pc_lobe_cavity_hourly.csv (Godin-valid hours), report.txt
  cache: LO_output/extract/<gtx>/box/pc_cove_godin_bands_<ds0>_<ds1>.npz
         (Godin band velocities, reused on rerun)

  python 20261027_pc_lobe_cavity.py
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-job', default='pc_cove')
p.add_argument('-0', '--ds0', default='2024.01.01')
p.add_argument('-1', '--ds1', default='2025.12.31')
p.add_argument('-R', type=float, default=500, help='lobe disk radius [m]')
p.add_argument('-lon_c', type=float, default=-122.708)
p.add_argument('-lat_c', type=float, default=48.226)
p.add_argument('-sect', default='pc_cp', help='N-S section_lines line for the inflow')
p.add_argument('-chunk', type=int, default=720)
args = p.parse_args()
warnings.simplefilter('ignore')

Ldir = Lfun.Lstart(gridname='wb1')
box_dir = Ldir['LOo'] / 'extract' / args.gtx / 'box'
box_fn = box_dir / ('%s_%s_%s.nc' % (args.job, args.ds0, args.ds1))
cache = box_dir / ('%s_godin_bands_%s_%s.npz' % (args.job, args.ds0, args.ds1))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261027_pc_lobe_cavity'
Lfun.make_dir(out_dir)
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
C_LOBE, C_IN = '#0072B2', '#D55E00'

ds = xr.open_dataset(box_fn)
lon, lat = ds.lon_rho.values, ds.lat_rho.values
mask = ds.mask_rho.values.astype(bool)
pm, pn = ds.pm.values, ds.pn.values
f0 = float(np.mean(ds.f.values[mask]))
tt = pd.to_datetime(ds.ocean_time.values)
NT = len(tt)
NR, NC = mask.shape
Hm = ds.h.values + ds.zeta.mean('ocean_time').values
EDGES = [(0, 4), (4, 10), (10, 99)]
BANDS = {('z%g-%g' % e if e[1] < 99 else 'z%g-bed' % e[0]): e for e in EDGES}
BANDS['dm'] = None
MB = {b: (mask if e is None else mask & (Hm >= e[0] + 2.0)) for b, e in BANDS.items()}

def godin(A):
    B = A.reshape(NT, -1)
    G = np.full(B.shape, np.nan, np.float32)
    for m in range(B.shape[1]):
        if np.any(B[:, m]):
            G[:, m] = zfun.lowpass(B[:, m].astype(float), f='godin')
        else:
            G[:, m] = 0.0
    return G.reshape(A.shape)

if cache.is_file():
    Z = np.load(cache)
    UG = {b: Z['U_' + b] for b in BANDS}
    VG = {b: Z['V_' + b] for b in BANDS}
    print('read cache ' + str(cache))
else:
    U = {b: np.zeros((NT, NR, NC - 1), np.float32) for b in BANDS}
    V = {b: np.zeros((NT, NR - 1, NC), np.float32) for b in BANDS}
    for t0 in range(0, NT, args.chunk):
        sl = slice(t0, min(t0 + args.chunk, NT))
        zw = ds.z_w[sl].values
        u = np.nan_to_num(ds.u[sl].values)
        v = np.nan_to_num(ds.v[sl].values)
        dz = np.diff(zw, axis=1)
        dzu, dzv = 0.5 * (dz[..., :-1] + dz[..., 1:]), 0.5 * (dz[..., :-1, :] + dz[..., 1:, :])
        dsf = zw[:, -1:] - 0.5 * (zw[:, :-1] + zw[:, 1:]) # depth below the surface
        du_, dv_ = 0.5 * (dsf[..., :-1] + dsf[..., 1:]), 0.5 * (dsf[..., :-1, :] + dsf[..., 1:, :])
        for b, e in BANDS.items():
            if e is None:
                wu, wv = dzu, dzv
            else:
                wu = np.where((du_ >= e[0]) & (du_ < e[1]), dzu, 0.0)
                wv = np.where((dv_ >= e[0]) & (dv_ < e[1]), dzv, 0.0)
            U[b][sl] = np.nan_to_num((u * wu).sum(1) / wu.sum(1))
            V[b][sl] = np.nan_to_num((v * wv).sum(1) / wv.sum(1))
        print('  read %s to %s' % (tt[sl.start].date(), tt[sl.stop - 1].date()), flush=True)
    UG = {b: godin(U[b]) for b in BANDS}
    VG = {b: godin(V[b]) for b in BANDS}
    del U, V
    np.savez(cache, **{'U_' + b: UG[b] for b in BANDS}, **{'V_' + b: VG[b] for b in BANDS})
    print('wrote cache ' + str(cache))
ds.close()

# ------------------------------------------------------------ geometry ---
c4 = lambda a: 0.25 * (a[:-1, :-1] + a[:-1, 1:] + a[1:, :-1] + a[1:, 1:])
lon_p, lat_p = c4(lon), c4(lat)
pm_p, pn_p = c4(pm), c4(pn)
M_PER_DEG = 111.32e3
r_p = np.hypot((lon_p - args.lon_c) * M_PER_DEG * np.cos(np.deg2rad(args.lat_c)), (lat_p - args.lat_c) * M_PER_DEG)
lon_u = 0.5 * (lon[:, :-1] + lon[:, 1:])
LON_S = pd.read_pickle(Ldir['LOo'] / 'section_lines' / (args.sect + '.p')).x.mean()
iu = int(np.argmin(np.abs(lon_u[0] - LON_S))) # u-face column of the section
print('section %s at lon %.4f -> u column %d (lon %.4f); disk R %d m, east edge lon %.4f'
      % (args.sect, LON_S, iu, lon_u[0, iu], args.R, args.lon_c + args.R / (M_PER_DEG * np.cos(np.deg2rad(args.lat_c)))))

valid = np.isfinite(UG['dm'][:, NR // 2, iu]) # Godin-valid hours
th = tt[valid]
rows = {}
info = []
for b in BANDS:
    Ug = UG[b][valid] * (MB[b][:, :-1] & MB[b][:, 1:])[None] # walls next to cells without the band
    Vg = VG[b][valid] * (MB[b][:-1, :] & MB[b][1:, :])[None]
    W4 = MB[b][:-1, :-1] & MB[b][:-1, 1:] & MB[b][1:, :-1] & MB[b][1:, 1:]
    disk = W4 & (r_p <= args.R)
    zeta = (Vg[:, :, 1:] - Vg[:, :, :-1]) * pm_p - (Ug[:, 1:, :] - Ug[:, :-1, :]) * pn_p
    lobe = np.nanmean(zeta[:, disk], axis=1) / f0
    wet_u = MB[b][:, iu] & MB[b][:, iu + 1]
    inside = wet_u & (np.nanmean(Ug[:, :, iu], axis=0) < 0)
    inflow = -np.nanmean(Ug[:, inside, iu], axis=1)
    rows['lobe_' + b] = lobe
    rows['inflow_' + b] = inflow
    info.append('%-8s disk %2d psi pts, inflow faces rows %s of %d wet' % (b, disk.sum(), list(np.flatnonzero(inside)), wet_u.sum()))
H = pd.DataFrame(rows, index=th)
H.to_csv(out_dir / 'pc_lobe_cavity_hourly.csv', float_format='%.5f') # for 20261027_pc_lobe_timing.py
D = H[H.index.hour == 12]
td = D.index
D.to_csv(out_dir / 'pc_lobe_cavity_daily.csv', float_format='%.5f')
A = D - D.rolling(31, center=True, min_periods=15).mean()

def neff(x, y):
    x, y = x.dropna(), y.dropna()
    r1 = lambda s: s.autocorr(1)
    a, c = r1(x), r1(y)
    n = min(len(x), len(y))
    return n * (1 - a * c) / (1 + a * c)

LAGS = np.arange(-10, 11) # days
report = info[:]
fig, axs = plt.subplots(len(BANDS), 3, figsize=(16, 3.3 * len(BANDS)), layout='constrained',
                        gridspec_kw={'width_ratios': [2.6, 1, 1.2]})
for n, b in enumerate(BANDS):
    L, I = D['lobe_' + b], D['inflow_' + b]
    La, Ia = A['lobe_' + b], A['inflow_' + b]
    z = lambda s: (s - s.mean()) / s.std()
    ax = axs[n, 0]
    ax.plot(td, z(I), '-', color=C_IN, lw=1, label='inflow at %s (mean %.1f cm/s)' % (args.sect, 100 * I.mean()))
    ax.plot(td, z(L), '-', color=C_LOBE, lw=1, label='lobe vorticity (mean %+.2f f)' % L.mean())
    ax.set_ylabel('standardized')
    ax.set_title(b, fontsize=10, loc='left')
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.grid(**GRID)
    ax = axs[n, 1]
    ax.plot(100 * Ia, La, '.', ms=3, color='0.3')
    ra = La.corr(Ia)
    ax.set_xlabel('inflow anomaly [cm/s]')
    ax.set_ylabel('lobe vorticity anomaly [f]')
    ax.set_title('31-d anomalies, lag 0: r = %+.2f' % ra, fontsize=9)
    ax.grid(**GRID)
    ax = axs[n, 2]
    rr = [L.shift(-k).corr(I) for k in LAGS]
    rra = [La.shift(-k).corr(Ia) for k in LAGS]
    ax.plot(LAGS, rr, '-o', ms=3, color='0.6', label='raw')
    ax.plot(LAGS, rra, '-o', ms=3, color='k', label='31-d anomalies')
    ax.axvline(0, color='0.5', lw=0.8)
    ax.axhline(0, color='0.5', lw=0.8)
    ax.set_xlabel('lag [d]  (> 0: lobe lags inflow)')
    ax.set_ylabel('r')
    ax.legend(fontsize=8)
    ax.grid(**GRID)
    k0, ka = int(np.nanargmax(rr)), int(np.nanargmax(rra))
    ne, nea = neff(L, I), neff(La, Ia)
    report.append('%-8s raw r(0) %+.2f, max %+.2f at lag %+d d (n_eff %.0f) | anomalies r(0) %+.2f, max %+.2f at lag %+d d (n_eff %.0f) | lobe mean %+.2f f, inflow mean %.2f cm/s'
                  % (b, rr[10], rr[k0], LAGS[k0], ne, rra[10], rra[ka], LAGS[ka], nea, L.mean(), 100 * I.mean()))
for k, ax in enumerate(axs.flat):
    ax.text(0.01, 0.97, 'abcdefghijklmnop'[k], transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')
fig.suptitle('Inner-cove cell vs the inflow that passes it: lobe = mean vorticity within %d m of (%.3f, %.3f); inflow = westward u on the inflow side of %s; Godin, daily: %s'
             % (args.R, args.lon_c, args.lat_c, args.sect, args.gtx), fontsize=10)
fig.savefig(out_dir / 'pc_lobe_cavity.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
(out_dir / 'report.txt').write_text('\n'.join(report) + '\n')
print('\n'.join(report))
print('Saved to ' + str(out_dir))
