"""
Penn Cove residual circulation in PLAN VIEW (x-y), from the pc_cove box.

WHY. 20260818_pc_lateral_circulation.py showed the cross-channel plane does NOT
close -- rms|du/dx| / rms|dv/dy| ~ 1.0 at all three sections -- so a (y,z)
streamfunction is not a material circulation and its contours are not flow
paths. What the pc_lp section does show is a strong depth-mean LATERAL
CONVERGENCE (+2.45 cm/s northward on the south flank to -3.91 cm/s southward on
the north flank, crossing zero over the deep channel), which has to leave along
the cove axis. Together with the tef2 result that along-channel u enters on the
north and leaves on the south ([[pc-lp-mouth-points]]), that points to a
HORIZONTAL circulation. This script looks at it directly instead of inferring it.

WHAT IT DOES
Godin-filters u and v on their native faces, averages each to rho points ONLY
at the end (masked faces zeroed first -- that is the correct cell-centred value
at a wall), and maps the residual flow in three layers: surface, mid, bottom.
Layer means are thickness-weighted over the model's own sigma layers.

  - quiver of residual (u,v) per layer, over bathymetry
  - the depth-mean flow separately, since at pc_lp it is ~3x the sheared part
  - a horizontal streamfunction-like check: is the depth-mean flow rotational?
    Reported as the area-integrated relative vorticity of the depth-mean flow
    and its sign, plus the circulation around the cove perimeter.

SEASON. Everything is also produced for the stratified (Dec-Feb) and weak
(Aug-Sep) extremes, because [[pc-lateral-baroclinic]] found the cross-channel
signal peaks in December and troughs in September.

SIGN. u eastward, v northward. Positive vorticity is counter-clockwise.

run 20260819_pc_planview_circulation.py
"""
import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
p.add_argument('-job', default='pc_cove', type=str)
p.add_argument('-0', '--ds0', default='2024.01.01', type=str)
p.add_argument('-1', '--ds1', default='2025.12.31', type=str)
args = p.parse_args()

warnings.simplefilter('ignore')
Ldir = Lfun.Lstart(gridname='wb1')
box_fn = (Ldir['LOo'] / 'extract' / args.gtagex / 'box' /
          ('%s_%s_%s.nc' % (args.job, args.ds0, args.ds1)))
out_dir = Ldir['LOo'] / 'DM_outs' / '20260819_pc_planview_circulation'
Lfun.make_dir(out_dir)
CB = dict(blue='#0072B2', red='#CC0000', green='#009E73', orange='#D55E00',
          purple='#CC79A7')
FS = 13

ds = xr.open_dataset(box_fn)
lon = ds.lon_rho.values
lat = ds.lat_rho.values
mask = ds.mask_rho.values.astype(bool)
h = ds.h.values
pm, pn = ds.pm.values, ds.pn.values
tt = pd.to_datetime(ds.ocean_time.values)
NR, NC = mask.shape
print('box %s : %d times, %d x %d' % (box_fn.name, len(tt), NR, NC))


def godin_np(A):
    """Godin filter down axis 0, NaN-safe, for (t, ...) arrays."""
    sh = A.shape
    B = A.reshape(sh[0], -1)
    G = np.full(B.shape, np.nan)
    for m in range(B.shape[1]):
        c = B[:, m]
        if np.isfinite(c).all():
            G[:, m] = zfun.lowpass(c.astype(float), f='godin')
    return G.reshape(sh)


# ---- layer-mean velocities on native faces, then to rho
zw = ds.z_w.values
DZ = np.diff(zw, axis=1)
zr = 0.5 * (zw[:, :-1, :, :] + zw[:, 1:, :, :])
hh = np.where(mask, h, np.nan)[None, None, :, :]
frac = (zr + hh) / hh                                  # 0 at bed, 1 at surface
LAYERS = [('surface (top 25%)', frac >= 0.75),
          ('mid (25-75%)', (frac > 0.25) & (frac < 0.75)),
          ('bottom (lower 25%)', frac <= 0.25)]

u = np.nan_to_num(ds.u.values)                         # masked faces -> 0 (wall)
v = np.nan_to_num(ds.v.values)
DZu = 0.5 * (DZ[:, :, :, :-1] + DZ[:, :, :, 1:])
DZv = 0.5 * (DZ[:, :, :-1, :] + DZ[:, :, 1:, :])
fu = 0.5 * (frac[:, :, :, :-1] + frac[:, :, :, 1:])
fv = 0.5 * (frac[:, :, :-1, :] + frac[:, :, 1:, :])


def to_rho(Fu, Fv):
    """u,v on faces -> rho points. Zero outside, one-sided at the box edge."""
    ur = np.full((Fu.shape[0], NR, NC), np.nan)
    ur[:, :, 1:-1] = 0.5 * (Fu[:, :, :-1] + Fu[:, :, 1:])
    ur[:, :, 0] = Fu[:, :, 0]
    ur[:, :, -1] = Fu[:, :, -1]
    vr = np.full((Fv.shape[0], NR, NC), np.nan)
    vr[:, 1:-1, :] = 0.5 * (Fv[:, :-1, :] + Fv[:, 1:, :])
    vr[:, 0, :] = Fv[:, 0, :]
    vr[:, -1, :] = Fv[:, -1, :]
    return ur, vr


def layer_mean(sel_u, sel_v):
    wu = np.where(sel_u, DZu, 0.0)
    wv = np.where(sel_v, DZv, 0.0)
    Uu = (u * wu).sum(1) / np.maximum(wu.sum(1), 1e-9)
    Vv = (v * wv).sum(1) / np.maximum(wv.sum(1), 1e-9)
    return Uu, Vv


# Seasons are taken FROM THE DATA, not from habit. Monthly cove-mean depth-mean
# speed and stratification (2024-25): fastest/most stratified are Nov-Dec
# (2.16, 2.36 cm/s; strat 4.46, 5.86), slowest/least stratified are Aug-Sep-Oct
# (1.55, 1.24, 1.27; strat 2.14, 1.84, 1.64). An earlier draft used DJF vs
# Aug-Sep: DJF dilutes the December peak with middling Jan (1.84) and Feb
# (1.73), and Aug-Sep drops October, which is the LEAST stratified month of the
# year. corr(monthly speed, monthly strat) = 0.82 -- the link is seasonal;
# daily correlations are only ~0.1. June is a genuine outlier (2nd fastest at
# only moderate stratification), so the relation is not monotonic.
SEASONS = [('all', np.ones(len(tt), bool)),
           ('strong (Nov-Dec)', np.isin(tt.month, [11, 12])),
           ('weak (Aug-Oct)', np.isin(tt.month, [8, 9, 10]))]

store = {}
for lab, selz_u in LAYERS:
    sel_v_ = {'surface (top 25%)': fv >= 0.75,
              'mid (25-75%)': (fv > 0.25) & (fv < 0.75),
              'bottom (lower 25%)': fv <= 0.25}[lab]
    Uu, Vv = layer_mean(fu >= 0.75 if lab.startswith('surface') else
                        ((fu > 0.25) & (fu < 0.75)) if lab.startswith('mid') else fu <= 0.25,
                        sel_v_)
    UuG, VvG = godin_np(Uu), godin_np(Vv)
    for sname, smask in SEASONS:
        m = smask & np.isfinite(UuG[:, 0, 0])
        store[(lab, sname)] = to_rho(np.nanmean(UuG[m], 0)[None],
                                     np.nanmean(VvG[m], 0)[None])
    print('  layer %-20s done' % lab)

# depth-mean flow and its vorticity
Uu_dm = (u * DZu).sum(1) / DZu.sum(1)
Vv_dm = (v * DZv).sum(1) / DZv.sum(1)
UdG, VdG = godin_np(Uu_dm), godin_np(Vv_dm)
okt = np.isfinite(UdG[:, 0, 0])
rep = []
for sname, smask in SEASONS:
    m = smask & okt
    Ub, Vb = np.nanmean(UdG[m], 0), np.nanmean(VdG[m], 0)
    ur, vr = to_rho(Ub[None], Vb[None])
    ur, vr = ur[0], vr[0]
    # relative vorticity dv/dx - du/dy at interior rho points
    dvdx = np.full((NR, NC), np.nan)
    dudy = np.full((NR, NC), np.nan)
    dvdx[:, 1:-1] = (vr[:, 2:] - vr[:, :-2]) * pm[:, 1:-1] / 2
    dudy[1:-1, :] = (ur[2:, :] - ur[:-2, :]) * pn[1:-1, :] / 2
    zeta_v = dvdx - dudy
    ok = mask & np.isfinite(zeta_v)
    A = (1 / pm) * (1 / pn)
    circ = np.nansum((zeta_v * A)[ok])
    store[('depthmean', sname)] = (ur[None], vr[None], zeta_v)
    rep.append(dict(season=sname, mean_speed_cms=100 * np.nanmean(np.hypot(ur, vr)[mask]),
                    max_speed_cms=100 * np.nanmax(np.hypot(ur, vr)[mask]),
                    mean_vort_1e5=1e5 * np.nanmean(zeta_v[ok]),
                    circulation_m2s=circ / 1e3))
R = pd.DataFrame(rep)
txt = ['PENN COVE PLAN-VIEW RESIDUAL CIRCULATION -- %s, %s to %s'
       % (args.gtagex, args.ds0, args.ds1), '',
       'Depth-mean subtidal flow. circulation = area-integrated relative vorticity',
       '(1e3 m2/s); positive = counter-clockwise.', '',
       R.to_string(index=False, float_format=lambda v: '%.3f' % v)]
print('\n' + '\n'.join(txt))
(out_dir / 'report.txt').write_text('\n'.join(txt) + '\n')
R.to_csv(out_dir / 'planview_summary.csv', index=False)

# ------------------------------------------------------------------ figures
SKIP = 1
hv = np.where(mask, h, np.nan)


def basemap(ax):
    ax.pcolormesh(lon, lat, hv, cmap='Blues', vmin=0, vmax=np.nanmax(hv), shading='nearest',
                  alpha=0.85)
    ax.set_aspect(1 / np.cos(np.deg2rad(48.23)))
    ax.set_xticks([]); ax.set_yticks([])


fig, axs = plt.subplots(3, 3, figsize=(15.5, 11))
for r, (lab, _) in enumerate(LAYERS):
    for c, (sname, _) in enumerate(SEASONS):
        ax = axs[r, c]
        basemap(ax)
        ur, vr = store[(lab, sname)]
        ur, vr = ur[0], vr[0]
        sp = np.hypot(ur, vr)
        q = ax.quiver(lon[::SKIP, ::SKIP], lat[::SKIP, ::SKIP],
                      np.where(mask, ur, np.nan)[::SKIP, ::SKIP],
                      np.where(mask, vr, np.nan)[::SKIP, ::SKIP],
                      np.where(mask, 100 * sp, np.nan)[::SKIP, ::SKIP],
                      cmap='inferno_r', clim=(0, 6), scale=0.9, width=0.006)
        if r == 0:
            ax.set_title(sname, fontsize=FS)
        if c == 0:
            ax.set_ylabel(lab, fontsize=FS - 1)
        if r == 2 and c == 2:
            plt.colorbar(q, ax=axs[:, 2], label='speed (cm s$^{-1}$)', shrink=0.6)
fig.suptitle('Penn Cove residual circulation by layer -- %s' % args.gtagex, fontsize=FS + 2)
fig.savefig(out_dir / 'pc_planview_layers.png', dpi=200, transparent=True,
            bbox_inches='tight')
plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
for c, (sname, _) in enumerate(SEASONS):
    ax = axs[c]
    ur, vr, zv = store[('depthmean', sname)]
    ur, vr = ur[0], vr[0]
    lim = np.nanpercentile(np.abs(zv[mask]), 98)
    pc = ax.pcolormesh(lon, lat, np.where(mask, zv, np.nan), cmap='RdBu_r',
                       vmin=-lim, vmax=lim, shading='nearest')
    ax.quiver(lon, lat, np.where(mask, ur, np.nan), np.where(mask, vr, np.nan),
              scale=0.7, width=0.005, color='k')
    ax.set_aspect(1 / np.cos(np.deg2rad(48.23)))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('%s\ndepth-mean flow + relative vorticity' % sname, fontsize=FS)
    plt.colorbar(pc, ax=ax, label='$\\zeta$ (s$^{-1}$)')
fig.suptitle('Penn Cove depth-mean residual flow -- is the circulation horizontal?',
             fontsize=FS + 2)
fig.tight_layout()
fig.savefig(out_dir / 'pc_planview_depthmean.png', dpi=200, transparent=True)
plt.close(fig)
print('\nwrote %s' % out_dir)
