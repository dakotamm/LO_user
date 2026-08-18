"""
The LATERAL OVERTURNING CELL in Penn Cove, seen directly.

WHY THIS EXISTS
20260818_pc_lateral_baroclinic.py showed the mouth carries a depth-varying
lateral exchange (~390 m3/s, 35 % of the lateral exchange) that is controlled
by stratification and not at all by the tide -- a baroclinic lateral exchange.
But tef2 stores only the SECTION-NORMAL velocity, so that result is the lateral
structure of the ALONG-channel flow. The cross-channel (v) and vertical (w)
velocities that would close the cell were not in any extraction. This script
uses a box extraction that carries them.

    (on apogee)
    python extract_box.py -gtx wb1_t0_xn11abbur00 -ro 0 \
        -0 2024.01.01 -1 2025.12.31 -lt hourly -job pc_lat -uv_to_rho True

WHAT IT COMPUTES
pc_lp, pc_lj and pc_cp are lines of constant longitude, so at each of them the
cross-channel direction is exactly north-south and the lateral plane is (y, z)
with velocities (v, w). No axis rotation is needed and the geometry matches the
tef2 sections cell for cell. For a station at longitude x:

    v'(y,z) = v(y,z) - (1/h) int v dz          depth-mean removed at each y
    psi(y,z) = int_{-h}^{z} v' dz'             [m2 s-1]

psi vanishes at the bed and at the surface by construction, so its contours are
CLOSED streamlines and max|psi| is the strength of the lateral overturning.
Integrated along-channel it becomes a transport in m3 s-1. Everything is done
on the model's own sigma layers using DZ = diff(z_w), so no vertical
interpolation enters the streamfunction.

THE HONESTY CHECK, AND IT MATTERS
psi is a 2-D streamfunction, which is only meaningful if the lateral plane is
close to non-divergent -- i.e. if dv/dy + dw/dz ~ 0 and the along-channel
divergence du/dx is small. That is NOT guaranteed at a cove mouth. So the
script reconstructs w from lateral continuity,

    w_2D(y,z) = - int_{-h}^{z} dv/dy dz'

and correlates it against the model's own w. A high correlation means the cell
is genuinely a 2-D lateral overturning; a low one means along-channel
divergence is doing the work and psi should be read as a diagnostic only. The
result is reported, not assumed.

TIDAL vs RESIDUAL. The cell is a residual feature, so everything is Godin
filtered before psi is formed. Filtering v BEFORE the depth-mean removal and
before integration (not after) is deliberate -- see
[[lowpassed-transport-stokes]] for what averaging in the wrong order costs.

SIGN. v is northward, w is up. Positive psi is a CLOCKWISE cell viewed with
north to the right, i.e. northward flow near the surface over southward flow
near the bed.

run 20260818_pc_lateral_circulation.py -gtx wb1_r0_xn11b -0 2017.09.05 -1 2017.09.17
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
p.add_argument('-job', default='pc_lat', type=str)
p.add_argument('-0', '--ds0', default='2024.01.01', type=str)
p.add_argument('-1', '--ds1', default='2025.12.31', type=str)
args = p.parse_args()

warnings.simplefilter('ignore')
Ldir = Lfun.Lstart(gridname='wb1')
box_fn = (Ldir['LOo'] / 'extract' / args.gtagex / 'box' /
          ('%s_%s_%s.nc' % (args.job, args.ds0, args.ds1)))
out_dir = Ldir['LOo'] / 'DM_outs' / '20260818_pc_lateral_circulation'
Lfun.make_dir(out_dir)

CB = dict(blue='#0072B2', red='#CC0000', green='#009E73', orange='#D55E00',
          purple='#CC79A7', grey='#7f7f7f')
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
FS = 13
# the three tef2 lines, by the longitude each was snapped to on 2026-08-04
STATIONS = [('pc_lp (mouth)', -122.6534), ('pc_lj (mid)', -122.6936),
            ('pc_cp (Coupeville)', -122.7218)]


def godin_t(A):
    """Godin filter down axis 0 of an array of any shape."""
    sh = A.shape
    F = np.full(sh, np.nan)
    B = A.reshape(sh[0], -1)
    G = F.reshape(sh[0], -1)
    for m in range(B.shape[1]):
        col = B[:, m]
        if np.isfinite(col).all():
            G[:, m] = zfun.lowpass(col.astype(float), f='godin')
    return G.reshape(sh)


# ------------------------------------------------------------------ load
ds = xr.open_dataset(box_fn)
lon = ds.lon_rho.values[0, :]
lat = ds.lat_rho.values[:, 0]
mask = ds.mask_rho.values.astype(bool)
h = ds.h.values
pn = ds.pn.values                                   # 1/dy
tt = pd.to_datetime(ds.ocean_time.values)
print('box %s : %d times, %d x %d cells' % (box_fn.name, len(tt), *h.shape))

results = []
panels = []
for name, xlon in STATIONS:
    i = int(np.argmin(np.abs(lon - xlon)))
    wet = mask[:, i]
    if wet.sum() < 4:
        print('  %s : only %d wet cells, skipped' % (name, wet.sum()))
        continue
    jj = np.flatnonzero(wet)
    y = (lat[jj] - lat[jj].mean()) * 111.32                     # km, + = north
    dy = 1.0 / pn[jj, i]                                        # m
    # (t, z, y) slabs on the model's own layers
    v = ds.v.values[:, :, jj, i]
    w = ds.w.values[:, :, jj, i]                                # on s_w
    zw = ds.z_w.values[:, :, jj, i]
    DZ = np.diff(zw, axis=1)
    hh = h[jj, i]

    vG = godin_t(v)
    DZG = godin_t(DZ)
    wG = godin_t(0.5 * (w[:, :-1, :] + w[:, 1:, :]))             # to s_rho
    ok = np.isfinite(vG[:, 0, 0])
    vG, DZG, wG = vG[ok], DZG[ok], wG[ok]

    # ---- lateral overturning streamfunction
    vbar = (vG * DZG).sum(1) / DZG.sum(1)                        # (t, y)
    vp = vG - vbar[:, None, :]
    psi = np.cumsum(vp * DZG, axis=1)                            # (t, z, y), 0 at bed & surface
    psi_m = psi.mean(0)
    kmax = np.unravel_index(np.abs(psi_m).argmax(), psi_m.shape)
    strength = psi_m[kmax]
    # transport equivalent: psi has units m2/s; scale by the cove's along-channel cell size
    dx_cell = 1.0 / ds.pm.values[jj, i].mean()

    # ---- continuity check: w from lateral divergence vs the model's own w
    dvdy = np.gradient(vp, axis=2) / dy[None, None, :]
    w2d = -np.cumsum(dvdy * DZG, axis=1)
    a, b = w2d.ravel(), wG.ravel()
    g = np.isfinite(a) & np.isfinite(b)
    r_cont = np.corrcoef(a[g], b[g])[0, 1]
    amp = np.sqrt((a[g] ** 2).mean()) / np.sqrt((b[g] ** 2).mean())

    zm = (0.5 * (zw[ok][:, :-1, :] + zw[ok][:, 1:, :])).mean(0)
    results.append(dict(station=name, lon=lon[i], nwet=int(wet.sum()),
                        width_km=float(y.max() - y.min()), hmax=float(hh.max()),
                        psi_max=float(strength), z_core=float(zm[kmax]),
                        y_core=float(y[kmax[1]]), Q_equiv=float(strength * dx_cell),
                        r_continuity=float(r_cont), amp_ratio=float(amp),
                        vprime_rms=float(np.sqrt(np.nanmean(vp ** 2)))))
    panels.append((name, y, zm, psi_m, vp.mean(0)))

R = pd.DataFrame(results)
txt = ['PENN COVE LATERAL OVERTURNING -- %s, %s to %s' % (args.gtagex, args.ds0, args.ds1),
       '', 'psi_max is the subtidal lateral overturning streamfunction (m2/s);',
       'Q_equiv scales it by one along-channel cell (m3/s).',
       'r_continuity: model w vs w reconstructed from lateral divergence alone.', '',
       R.to_string(index=False, float_format=lambda v: '%.3f' % v)]
report = '\n'.join(txt)
print('\n' + report)
(out_dir / 'report.txt').write_text(report + '\n')
R.to_csv(out_dir / 'lateral_cells.csv', index=False)

# ------------------------------------------------------------------ figure
n = len(panels)
fig, axs = plt.subplots(2, n, figsize=(5.4 * n, 8.4), squeeze=False)
for c, (name, y, zm, psi_m, vpm) in enumerate(panels):
    Y = np.tile(y, (zm.shape[0], 1))
    ax = axs[0, c]
    lim = np.abs(psi_m).max()
    pc = ax.pcolormesh(Y, zm, psi_m, cmap='RdBu_r', vmin=-lim, vmax=lim, shading='gouraud')
    cs = ax.contour(Y, zm, psi_m, 9, colors='k', linewidths=0.7)
    ax.clabel(cs, fontsize=7, fmt='%.2f')
    plt.colorbar(pc, ax=ax, label=r'$\psi$ (m$^2$ s$^{-1}$)')
    ax.set_title('%s\nlateral overturning streamfunction' % name, fontsize=FS)
    ax.set_xlabel('distance north of section centre (km)')
    ax.set_ylabel('z (m)')

    ax = axs[1, c]
    lim = np.abs(100 * vpm).max()
    pc = ax.pcolormesh(Y, zm, 100 * vpm, cmap='PuOr_r', vmin=-lim, vmax=lim, shading='gouraud')
    ax.contour(Y, zm, vpm, [0], colors='k', linewidths=1.2)
    plt.colorbar(pc, ax=ax, label='cm s$^{-1}$')
    rr = R.iloc[c]
    ax.set_title(r"cross-channel $v'$  (r$_{cont}$ = %+.2f)" % rr.r_continuity, fontsize=FS)
    ax.set_xlabel('distance north of section centre (km)')
    ax.set_ylabel('z (m)')

fig.suptitle('Penn Cove lateral circulation -- %s  %s to %s'
             % (args.gtagex, args.ds0, args.ds1), fontsize=FS + 2)
fig.tight_layout()
fig.savefig(out_dir / 'pc_lateral_circulation.png', dpi=200, transparent=True)
plt.close(fig)
print('\nwrote %s' % out_dir)
