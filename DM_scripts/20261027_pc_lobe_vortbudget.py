"""
CIRCULATION BUDGET of the inner Penn Cove cell at fixed depths: what spins it?

Around a closed horizontal loop at fixed z the pressure-gradient and kinetic-
energy-gradient terms integrate to zero (Boussinesq, hydrostatic), so the
circulation G = loop integral of u.dl obeys

  dG/dt = S + A + V + R
    S = -f  loop(u_n dl)        net horizontal INFLOW across the loop at this
                                level = convergence = stretching of planetary
                                vorticity (f matters here even though the cell
                                is far too small for geostrophy: zeta ~ 0.5 f)
    A = -   loop(zeta u_n dl)   relative vorticity carried across the loop
                                (the passing inflow handing over its shear
                                vorticity shows up here; includes stretching
                                of relative vorticity)
    V = -   loop(w du_h/dz . dl)  vertical advection + tilting
    R = remainder               friction / mixing (vertical stress curl,
                                horizontal mixing) + discretisation error +
                                hourly-snapshot sampling error. ROMS budget
                                diagnostics were not saved, so R is NOT a clean
                                friction estimate.
u_n is the outward normal velocity (loop counter-clockwise). Every term is
divided by the loop area and given in f per day (vorticity tendency / f * 86400).

LOOP: the boundary of the psi cells within -R m of the eddy centre
(-122.708, 48.226) whose four rho corners are wet and deeper than |z| + 1 m.
On the C-grid that boundary runs rho point to rho point, so the tangential
velocity is exactly a u or v face and G equals the area integral of psi
vorticity (checked). Normal velocity, zeta and w on each edge are averages of
the neighbouring faces / psi / rho points. Velocities are interpolated to fixed
z from z_rho (w from z_w) every hour; masked faces are walls (0).

TIME: all terms hourly (so tidal correlations are kept), then Godin-filtered.
dG/dt is a centred hourly difference.

Output: LO_output/DM_outs/20261027_pc_lobe_vortbudget/
  pc_lobe_vortbudget_hourly.csv  raw hourly terms per level (cache: delete to recompute)
  pc_lobe_vortbudget.png         per level: 2-year mean budget, monthly means,
                                 correlation of each term with dG/dt (Godin)
  pc_lobe_vortbudget_split.png   V split pointwise into vertical advection
                                 -w dzeta/dz and tilting (dw/dy du/dz - dw/dx dv/dz),
                                 plus S checked against continuity (f dw/dz)
  pc_lobe_vortbudget_split_hourly.csv  cache for the split
  report.txt

  python 20261027_pc_lobe_vortbudget.py
"""

import argparse
import warnings

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
p.add_argument('-R', type=float, default=500)
p.add_argument('-lon_c', type=float, default=-122.708)
p.add_argument('-lat_c', type=float, default=48.226)
p.add_argument('-levels', default='-5,-7,-9,-11', help='fixed z [m]; lowest zeta in the run is -3.4 m')
p.add_argument('-chunk', type=int, default=720)
p.add_argument('-test', default=False, type=Lfun.boolean_string) # first 60 days, separate output folder
args = p.parse_args()
warnings.simplefilter('ignore')
ZL = [float(v) for v in args.levels.split(',')]
DZ = 1.0 # half-spacing for du/dz [m]

Ldir = Lfun.Lstart(gridname='wb1')
box_fn = Ldir['LOo'] / 'extract' / args.gtx / 'box' / ('%s_%s_%s.nc' % (args.job, args.ds0, args.ds1))
out_dir = Ldir['LOo'] / 'DM_outs' / ('20261027_pc_lobe_vortbudget' + ('_test' if args.test else ''))
Lfun.make_dir(out_dir)
hfn = out_dir / 'pc_lobe_vortbudget_hourly.csv'
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
TERMS = ['S', 'A', 'V', 'R']
TLAB = {'S': 'S  stretching of f (net inflow)', 'A': 'A  relative vorticity carried in', 'V': 'V  vertical advection + tilting',
        'R': 'R  remainder (friction, mixing, error)', 'dG': 'dG/dt'}
TCOL = {'S': '#0072B2', 'A': '#D55E00', 'V': '#009E73', 'R': '0.55', 'dG': 'k'}

ds = xr.open_dataset(box_fn)
lon, lat = ds.lon_rho.values, ds.lat_rho.values
mask = ds.mask_rho.values.astype(bool)
pm, pn, h = ds.pm.values, ds.pn.values, ds.h.values
f0 = float(np.mean(ds.f.values[mask]))
tt = pd.to_datetime(ds.ocean_time.values)
NT = 60 * 24 if args.test else len(tt)
tt = tt[:NT]
NR, NC = mask.shape
M_PER_DEG = 111.32e3
c4 = lambda a: 0.25 * (a[:-1, :-1] + a[:-1, 1:] + a[1:, :-1] + a[1:, 1:])
lon_p, lat_p = c4(lon), c4(lat)
r_p = np.hypot((lon_p - args.lon_c) * M_PER_DEG * np.cos(np.deg2rad(args.lat_c)), (lat_p - args.lat_c) * M_PER_DEG)
area_p = c4(1 / (pm * pn))
pm_u, pn_v = 0.5 * (pm[:, :-1] + pm[:, 1:]), 0.5 * (pn[:-1, :] + pn[1:, :])
pm_p, pn_p = c4(pm), c4(pn)

# ------------------------------------------------------------- the loops ---
def build_loop(z0):
    deep = mask & (h > abs(z0) + 1)
    cells = deep[:-1, :-1] & deep[:-1, 1:] & deep[1:, :-1] & deep[1:, 1:] & (r_p <= args.R)
    E = {} # ('u', j, i) or ('v', j, i) -> tangential sign (+1 east / north)
    for j, i in zip(*np.nonzero(cells)): # CCW around psi cell (j, i)
        for key, c in [(('u', j, i), 1), (('v', j, i + 1), 1), (('u', j + 1, i), -1), (('v', j, i), -1)]:
            E[key] = E.get(key, 0) + c
    E = {k: c for k, c in E.items() if c != 0}
    return cells, E

LOOPS = {z0: build_loop(z0) for z0 in ZL}
for z0, (cells, E) in LOOPS.items():
    print('z %5.1f: %d psi cells, %d boundary edges, area %.2f km2' % (z0, cells.sum(), len(E), area_p[cells].sum() / 1e6))

def interp_z(F, Z, z0):
    """F, Z (T, K, ...) with Z increasing in K -> F at z0, NaN outside."""
    kk = (Z <= z0).sum(axis=1) - 1
    bad = (kk < 0) | (kk >= Z.shape[1] - 1)
    kk = np.clip(kk, 0, Z.shape[1] - 2)
    take = lambda A, k: np.take_along_axis(A, k[:, None], axis=1)[:, 0]
    z1, z2 = take(Z, kk), take(Z, kk + 1)
    f1, f2 = take(F, kk), take(F, kk + 1)
    out = f1 + (f2 - f1) * (z0 - z1) / (z2 - z1)
    return np.where(bad, np.nan, out)

def terms_chunk(u, v, w, zr, zw, z0):
    """hourly G, S, A, V (all / area) for one level, one chunk."""
    cells, E = LOOPS[z0]
    zu, zv = 0.5 * (zr[..., :-1] + zr[..., 1:]), 0.5 * (zr[..., :-1, :] + zr[..., 1:, :])
    U = {dz: np.nan_to_num(interp_z(u, zu, z0 + dz)) for dz in [0, -DZ, DZ]} # walls / outside -> 0
    V_ = {dz: np.nan_to_num(interp_z(v, zv, z0 + dz)) for dz in [0, -DZ, DZ]}
    W = interp_z(w, zw, z0)
    u0, v0 = U[0], V_[0]
    zeta = (v0[:, :, 1:] - v0[:, :, :-1]) * pm_p - (u0[:, 1:, :] - u0[:, :-1, :]) * pn_p # (T, NR-1, NC-1)
    dudz, dvdz = (U[DZ] - U[-DZ]) / (2 * DZ), (V_[DZ] - V_[-DZ]) / (2 * DZ)
    T = u.shape[0]
    G, S, A, VV = (np.zeros(T) for _ in range(4))
    for (kind, j, i), c in E.items():
        if kind == 'u': # E-W edge on rho row j between cols i, i+1
            L = 1 / pm_u[j, i]
            ut = c * u0[:, j, i]
            vn = np.mean([v0[:, jj, ii] for jj in [j - 1, j] for ii in [i, i + 1] if 0 <= jj < NR - 1], axis=0)
            un = -c * vn
            ze = np.nanmean([zeta[:, jj, i] for jj in [j - 1, j] if 0 <= jj < NR - 1], axis=0)
            we = 0.5 * (W[:, j, i] + W[:, j, i + 1])
            dt_dz = c * dudz[:, j, i]
        else: # N-S edge on rho col i between rows j, j+1
            L = 1 / pn_v[j, i]
            ut = c * v0[:, j, i]
            ue = np.mean([u0[:, jj, ii] for jj in [j, j + 1] for ii in [i - 1, i] if 0 <= ii < NC - 1], axis=0)
            un = c * ue
            ze = np.nanmean([zeta[:, j, ii] for ii in [i - 1, i] if 0 <= ii < NC - 1], axis=0)
            we = 0.5 * (W[:, j, i] + W[:, j + 1, i])
            dt_dz = c * dvdz[:, j, i]
        G += ut * L
        S += -f0 * un * L
        A += -ze * un * L
        VV += -we * dt_dz * L
    Ar = area_p[cells].sum()
    Gz = np.nansum(zeta[:, cells] * area_p[cells], axis=1) # Stokes check
    return G / Ar, S / Ar, A / Ar, VV / Ar, Gz / Ar

def split_chunk(u, v, w, zr, zw, z0):
    """area means over the disk cells: VA = -w dzeta/dz, TL = dw/dy du/dz - dw/dx dv/dz
    (VA + TL = V, the pointwise form), Sw = f dw/dz (continuity check on S)."""
    cells, _ = LOOPS[z0]
    zu, zv = 0.5 * (zr[..., :-1] + zr[..., 1:]), 0.5 * (zr[..., :-1, :] + zr[..., 1:, :])
    U = {dz: np.nan_to_num(interp_z(u, zu, z0 + dz)) for dz in [0, -DZ, DZ]}
    V_ = {dz: np.nan_to_num(interp_z(v, zv, z0 + dz)) for dz in [0, -DZ, DZ]}
    W = {dz: interp_z(w, zw, z0 + dz) for dz in [0, -DZ, DZ]}
    zet = {dz: (V_[dz][:, :, 1:] - V_[dz][:, :, :-1]) * pm_p - (U[dz][:, 1:, :] - U[dz][:, :-1, :]) * pn_p for dz in [-DZ, DZ]}
    w0 = W[0]
    wp = 0.25 * (w0[:, :-1, :-1] + w0[:, :-1, 1:] + w0[:, 1:, :-1] + w0[:, 1:, 1:])
    VA = -wp * (zet[DZ] - zet[-DZ]) / (2 * DZ)
    dwdx = 0.5 * ((w0[:, :-1, 1:] - w0[:, :-1, :-1]) + (w0[:, 1:, 1:] - w0[:, 1:, :-1])) * pm_p
    dwdy = 0.5 * ((w0[:, 1:, :-1] - w0[:, :-1, :-1]) + (w0[:, 1:, 1:] - w0[:, :-1, 1:])) * pn_p
    dudz = (U[DZ] - U[-DZ]) / (2 * DZ)
    dvdz = (V_[DZ] - V_[-DZ]) / (2 * DZ)
    TL = dwdy * 0.5 * (dudz[:, :-1, :] + dudz[:, 1:, :]) - dwdx * 0.5 * (dvdz[:, :, :-1] + dvdz[:, :, 1:])
    dw = (W[DZ] - W[-DZ]) / (2 * DZ)
    Sw = f0 * 0.25 * (dw[:, :-1, :-1] + dw[:, :-1, 1:] + dw[:, 1:, :-1] + dw[:, 1:, 1:])
    a = area_p[cells]
    am = lambda X: np.nansum(X[:, cells] * a, axis=1) / a.sum()
    return am(VA), am(TL), am(Sw)

sfn = out_dir / 'pc_lobe_vortbudget_split_hourly.csv'
need_h, need_s = not hfn.is_file(), not sfn.is_file()
cols, scols = {}, {}
if need_h or need_s:
    for t0 in range(0, NT, args.chunk):
        sl = slice(t0, min(t0 + args.chunk, NT))
        u, v = ds.u[sl].values, ds.v[sl].values
        w, zr, zw = ds.w[sl].values, ds.z_rho[sl].values, ds.z_w[sl].values
        for z0 in ZL:
            if need_h:
                for k, x in zip(['G', 'S', 'A', 'V', 'Gzeta'], terms_chunk(u, v, w, zr, zw, z0)):
                    cols.setdefault('%s_%g' % (k, z0), []).append(x)
            if need_s:
                for k, x in zip(['VA', 'TL', 'Sw'], split_chunk(u, v, w, zr, zw, z0)):
                    scols.setdefault('%s_%g' % (k, z0), []).append(x)
        print('  %s to %s' % (tt[sl.start].date(), tt[sl.stop - 1].date()), flush=True)
if need_h:
    Hh = pd.DataFrame({k: np.concatenate(x) for k, x in cols.items()}, index=tt)
    Hh.to_csv(hfn, float_format='%.6e')
else:
    Hh = pd.read_csv(hfn, index_col=0, parse_dates=True)
    print('read ' + str(hfn))
if need_s:
    Hs = pd.DataFrame({k: np.concatenate(x) for k, x in scols.items()}, index=tt)
    Hs.to_csv(sfn, float_format='%.6e')
else:
    Hs = pd.read_csv(sfn, index_col=0, parse_dates=True)
    print('read ' + str(sfn))
ds.close()

# --------------------------------------------------------------- analysis ---
UNIT = 86400 / f0 # s-2 -> f per day
B = {}
report = ['circulation budget around a %d m disk at (%.3f, %.3f); terms in f/day; G in f' % (args.R, args.lon_c, args.lat_c)]
for z0 in ZL:
    k = lambda n: '%s_%g' % (n, z0)
    G = Hh[k('G')].values
    dG = np.gradient(G, 3600.0)
    d = {'G': G / f0, 'dG': dG * UNIT, 'S': Hh[k('S')].values * UNIT, 'A': Hh[k('A')].values * UNIT, 'V': Hh[k('V')].values * UNIT}
    d['R'] = d['dG'] - d['S'] - d['A'] - d['V']
    for n in ['VA', 'TL', 'Sw']:
        d[n] = Hs[k(n)].values * UNIT
    chk = np.nanmax(np.abs(Hh[k('G')] - Hh[k('Gzeta')])) / np.nanstd(Hh[k('G')])
    Dg = pd.DataFrame({n: zfun.lowpass(x.astype(float), f='godin') for n, x in d.items()}, index=Hh.index)
    B[z0] = Dg
    m = Dg.mean()
    rr = {n: Dg['dG'].corr(Dg[n]) for n in TERMS}
    report.append('z %5.1f: mean G %+.2f f | mean S %+.3f A %+.3f V %+.3f R %+.3f dG %+.4f | corr with dG/dt: S %+.2f A %+.2f V %+.2f R %+.2f | rms S %.3f A %.3f V %.3f R %.3f | Stokes check max|G-sum zeta A|/sd %.1e'
                  % (z0, m.G, m.S, m.A, m.V, m.R, m.dG, rr['S'], rr['A'], rr['V'], rr['R'], Dg.S.std(), Dg.A.std(), Dg.V.std(), Dg.R.std(), chk))

fig, axs = plt.subplots(len(ZL), 3, figsize=(16, 3.4 * len(ZL)), layout='constrained', gridspec_kw={'width_ratios': [1, 2.2, 1]})
for n, z0 in enumerate(ZL):
    Dg = B[z0]
    ax = axs[n, 0]
    names = TERMS + ['dG']
    vals = [Dg[t].mean() for t in names]
    ax.bar(range(len(names)), vals, color=[TCOL[t] for t in names], width=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(['S', 'A', 'V', 'R', 'dG/dt'])
    ax.axhline(0, color='0.3', lw=0.8)
    ax.set_ylabel('f per day')
    ax.set_title('z = %g m: 2-year mean (cell %+.2f f)' % (z0, Dg.G.mean()), fontsize=10, loc='left')
    ax.grid(axis='y', **GRID)
    ax = axs[n, 1]
    Mo = Dg.resample('MS').mean()
    for t in TERMS:
        ax.plot(Mo.index + pd.Timedelta(days=14), Mo[t], '-o', ms=3, color=TCOL[t], lw=2, label=TLAB[t])
    ax.axhline(0, color='0.3', lw=0.8)
    ax.set_ylabel('f per day')
    ax.set_title('monthly means', fontsize=10, loc='left')
    ax.grid(**GRID)
    if n == 0:
        ax.legend(fontsize=8, ncol=2, loc='upper left')
    ax = axs[n, 2]
    rr = [Dg['dG'].corr(Dg[t]) for t in TERMS]
    ax.bar(range(len(TERMS)), rr, color=[TCOL[t] for t in TERMS], width=0.7)
    ax.set_xticks(range(len(TERMS)))
    ax.set_xticklabels(TERMS)
    ax.set_ylim(-1, 1)
    ax.axhline(0, color='0.3', lw=0.8)
    ax.set_ylabel('r with dG/dt')
    ax.set_title('which term moves the cell', fontsize=10, loc='left')
    ax.grid(axis='y', **GRID)
for k, ax in enumerate(axs.flat):
    ax.text(0.01, 0.97, 'abcdefghijklmnop'[k], transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')
fig.suptitle('Circulation budget of the inner-cove cell (%d m disk at %.3f, %.3f), fixed depths, Godin-filtered: %s'
             % (args.R, args.lon_c, args.lat_c, args.gtx), fontsize=10)
fig.savefig(out_dir / 'pc_lobe_vortbudget.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
report.append('')
report.append('SPLIT of V (pointwise, disk area mean) and continuity check of S (Sw = f dw/dz):')
fig, axs = plt.subplots(len(ZL), 2, figsize=(14, 3.2 * len(ZL)), layout='constrained', gridspec_kw={'width_ratios': [1, 2.2]})
SCOL = {'S': '#0072B2', 'Sw': '#56B4E9', 'V': '#009E73', 'VA': '#CC79A7', 'TL': '#E69F00'}
SLAB = {'S': 'S loop', 'Sw': 'S from dw/dz', 'V': 'V loop', 'VA': 'vertical adv.', 'TL': 'tilting'}
for n, z0 in enumerate(ZL):
    Dg = B[z0]
    m = Dg.mean()
    report.append('z %5.1f: S loop %+.3f vs S from dw/dz %+.3f (r %.2f) | V loop %+.3f vs VA+TL %+.3f (r %.2f): vertical advection %+.3f, tilting %+.3f | monthly r(G, TL) %+.2f, r(G, VA) %+.2f'
                  % (z0, m.S, m.Sw, Dg.S.corr(Dg.Sw), m.V, m.VA + m.TL, Dg.V.corr(Dg.VA + Dg.TL), m.VA, m.TL,
                     Dg.resample('MS').mean().G.corr(Dg.resample('MS').mean().TL), Dg.resample('MS').mean().G.corr(Dg.resample('MS').mean().VA)))
    ax = axs[n, 0]
    names = ['S', 'Sw', 'V', 'VA', 'TL']
    ax.bar(range(len(names)), [m[t] for t in names], color=[SCOL[t] for t in names], width=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([SLAB[t] for t in names], rotation=30, ha='right', fontsize=8)
    ax.axhline(0, color='0.3', lw=0.8)
    ax.set_ylabel('f per day')
    ax.set_title('z = %g m: 2-year mean' % z0, fontsize=10, loc='left')
    ax.grid(axis='y', **GRID)
    ax = axs[n, 1]
    Mo = Dg.resample('MS').mean()
    for t in ['VA', 'TL']:
        ax.plot(Mo.index + pd.Timedelta(days=14), Mo[t], '-o', ms=3, color=SCOL[t], lw=2, label=SLAB[t])
    ax.axhline(0, color='0.3', lw=0.8)
    ax.set_ylabel('f per day')
    ax.set_title('monthly means (cell strength in title of the main figure)', fontsize=10, loc='left')
    ax.grid(**GRID)
    ax.legend(fontsize=8, loc='upper left')
for k, ax in enumerate(axs.flat):
    ax.text(0.01, 0.97, 'abcdefghijklmnop'[k], transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')
fig.suptitle('Vertical term split (vertical advection vs tilting) and continuity check on stretching: %s' % args.gtx, fontsize=10)
fig.savefig(out_dir / 'pc_lobe_vortbudget_split.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
(out_dir / 'report.txt').write_text('\n'.join(report) + '\n')
print('\n'.join(report))
