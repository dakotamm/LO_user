"""
The average flood and the average ebb across pc_lp, drawn as sections.

Distance across the mouth on x (north on the left, at the Coupeville-side
shore), depth on y, colour is the section-NORMAL velocity

    u(t,z,p) = q / (dd * DZ)                                      [m s-1]

negated so that BLUE is INTO Penn Cove and RED is out of it. Every hour of
2024-2025 is sorted by the sign of the section net transport -- flood is
qnet into the cove, ebb is qnet out -- and the two composites are the plain
means of u over those hours. Not peak flood and peak ebb: every flooding hour
counts the same as every other, so the panels answer "what does a flood look
like on average", not "what does the strongest flood look like".

The two are put on one symmetric colour scale so their magnitudes can be
compared by eye, and the u = 0 contour is drawn on both -- that line is where
the section stops flowing one way and starts flowing the other, which on flood
runs across the cove at depth rather than down the middle.

GEOMETRY IS REAL, NOT SIGMA. Cells are drawn at their record-mean thickness
built up from the bed out of DZ, so the panel is in metres and the sloping bed
(h = 15.9 m at the north end, 26.8 m in the middle, 20.0 m at the south end) is
the shape it actually is. Face width is a uniform dd = 200.5 m.

WHAT TO LOOK FOR. The flood is not a slab: the inflow is surface-intensified
and the deepest water can still be leaving. See
20260811_pc_lp_baroclinic_barotropic.py for why -- the barotropic tide at this
section is M2 and the vertical shear on top of it is a diurnal baroclinic mode
with a node ~10 m down. The ebb is closer to unidirectional. The flood/ebb
difference in lateral position is the other half of the story: the residual at
this mouth is a lateral exchange, in on the north side and out on the south
(see 20260806_pc_mouth_salinity_tides.py).

SIGN. Positive q at pc_lp runs minus-side -> plus-side = eastward = OUT of the
cove; corr(qnet, d(ssh)/dt) = -1.00 confirms flood is qnet < 0 in the stored
sign. Everything here is negated once, at load.

CAVEAT. u is the section-normal component only; the along-section component
never enters the tef2 extraction, so |u| is a lower bound on true speed.

Runs on the mac from the local extractions_avg.
run 20260811_pc_lp_flood_ebb_section.py
run 20260811_pc_lp_flood_ebb_section.py -sect pc_cp
"""
import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cmcrameri import cm as cmc

from lo_tools import Lfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
p.add_argument('-gctag', default='wb1_pc1', type=str)
p.add_argument('-sect', default='pc_lp', type=str)
p.add_argument('-0', '--ds0', default='2024.01.01', type=str)
p.add_argument('-1', '--ds1', default='2025.12.31', type=str)
args = p.parse_args()

warnings.simplefilter('ignore')
Ldir = Lfun.Lstart(gridname=args.gctag.split('_')[0])
tef2 = Ldir['LOo'] / 'extract' / args.gtagex / 'tef2'
ex_fn = tef2 / ('extractions_avg_%s_%s' % (args.ds0, args.ds1)) / (args.sect + '.nc')
st_fn = tef2 / ('structure_%s_%s_%s.nc' % (args.ds0, args.ds1, args.gctag))
out_dir = Ldir['LOo'] / 'DM_outs' / '20260811_pc_lp_flood_ebb'
Lfun.make_dir(out_dir)

FS = 14
BED = '#b9a894'

# ---------------------------------------------------------------------------
# load, split on the sign of the net transport
# ---------------------------------------------------------------------------
ds = xr.open_dataset(ex_fn)
tt = pd.to_datetime(ds.time.values)
dd, hh = ds.dd.values, ds.h.values
q = -ds.q.values                        # + = INTO the cove
DZ = ds.DZ.values
ds.close()
A = DZ * dd[None, None, :]
u = q / A
qnet = q.sum(axis=(1, 2))
flood, ebb = qnet > 0, qnet < 0

dstr = xr.open_dataset(st_fn)
lat = dstr['%s_lat' % args.sect].values
dstr.close()

U = {'flood': np.nanmean(u[flood], axis=0), 'ebb': np.nanmean(u[ebb], axis=0)}
DZm = {'flood': DZ[flood].mean(axis=0), 'ebb': DZ[ebb].mean(axis=0)}

print('%s: %d faces, %d hours -- %d flood (%.0f%%), %d ebb'
      % (args.sect, len(dd), len(tt), flood.sum(), 100 * flood.mean(), ebb.sum()))
print('mean |qnet|  flood %+.0f  ebb %+.0f m3/s'
      % (qnet[flood].mean(), qnet[ebb].mean()))
for k in ['flood', 'ebb']:
    print('  %-5s u range %+.3f to %+.3f m/s ; %.0f%% of the section area flows '
          'INTO the cove' % (k, U[k].min(), U[k].max(),
                             100 * (A.mean(axis=0) * (U[k] > 0)).sum() / A.mean(axis=0).sum()))

# ---------------------------------------------------------------------------
# section geometry: x edges from the cumulative face width, z edges from DZ
# ---------------------------------------------------------------------------
xe = np.concatenate([[0], np.cumsum(dd)]) / 1000.               # km, north -> south
xc = 0.5 * (xe[:-1] + xe[1:])


def z_edges(dzm):
    """Cell interfaces in metres below the surface, built up from the bed."""
    zw = np.zeros((dzm.shape[0] + 1, dzm.shape[1]))
    zw[1:, :] = np.cumsum(dzm, axis=0)
    return -(zw[-1:, :] - zw)          # 0 at the surface, -h at the bed


def on_common_grid(uu, dzm, zg):
    """Each face's profile interpolated onto one depth grid, masked below its
    own bed -- only so the u = 0 line can be contoured across faces. The
    colours are drawn from the untouched sigma cells, not from this."""
    G = np.full((len(zg), dzm.shape[1]), np.nan)
    zw = z_edges(dzm)
    for j in range(dzm.shape[1]):
        zc = 0.5 * (zw[:-1, j] + zw[1:, j])
        ok = zg >= zw[0, j]
        G[ok, j] = np.interp(zg[ok], zc, uu[:, j])
    return np.ma.masked_invalid(G)


VM = max(abs(U['flood']).max(), abs(U['ebb']).max())
VM = np.ceil(VM * 50) / 50.            # round up to a tidy 0.02
zg = np.linspace(-hh.max(), 0, 120)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
for ax, k in zip(axes, ['flood', 'ebb']):
    zw = z_edges(DZm[k])
    for j in range(len(dd)):
        ax.pcolormesh(xe[j:j + 2], zw[:, j], U[k][:, j:j + 1], cmap=cmc.vik_r,
                      vmin=-VM, vmax=VM, shading='flat', rasterized=True)
    ax.contour(xc, zg, on_common_grid(U[k], DZm[k], zg), levels=[0],
               colors='k', linewidths=1.4)
    # the bed, drawn at the bottom of the cells actually plotted so no white
    # seam opens up between the deepest cell and the fill
    hb = DZm[k].sum(axis=0)
    xs, zs = np.repeat(xe, 2)[1:-1], np.repeat(-hb, 2)
    ax.fill_between(xs, zs, -hb.max() - 3, color=BED, lw=0, zorder=5)
    ax.plot(xs, zs, color='k', lw=1.0, zorder=6)
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(-hh.max() - 3, 0)
    ax.set_xlabel('distance across the mouth [km]', fontsize=FS)
    ax.set_title('%s  (n = %d h, mean $Q_{net}$ = %+.0f m$^3$ s$^{-1}$)'
                 % (k, {'flood': flood, 'ebb': ebb}[k].sum(),
                    qnet[{'flood': flood, 'ebb': ebb}[k]].mean()),
                 fontsize=FS, loc='left')
    ax.grid(color='lightgray', linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=FS - 2)
axes[0].set_ylabel('depth below surface [m]', fontsize=FS)
axes[0].text(0.01, 1.10, 'NORTH', transform=axes[0].transAxes, fontsize=FS - 2,
             ha='left', color='#555555')
axes[1].text(0.99, 1.10, 'SOUTH', transform=axes[1].transAxes, fontsize=FS - 2,
             ha='right', color='#555555')

sm = plt.cm.ScalarMappable(cmap=cmc.vik_r,
                           norm=plt.Normalize(vmin=-VM, vmax=VM))
cb = fig.colorbar(sm, ax=axes, fraction=0.035, pad=0.02)
cb.set_label('section-normal $u$ [m s$^{-1}$],  blue = into Penn Cove',
             fontsize=FS - 1)
cb.ax.tick_params(labelsize=FS - 3)

fn_out = out_dir / ('flood_ebb_section_%s_%s.png' % (args.gtagex, args.sect))
fig.savefig(fn_out, dpi=400, bbox_inches='tight', transparent=True)
print('\nwrote ' + str(fn_out))
