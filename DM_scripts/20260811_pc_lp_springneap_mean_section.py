"""
The mean section across pc_lp during spring tides and during neap tides.

Two section views, no flood/ebb split: distance across the mouth on x (north on
the left), depth on y, colour is the section-NORMAL velocity

    u(t,z,p) = q / (dd * DZ)                                      [m s-1]

negated so that BLUE is INTO Penn Cove and RED is out of it, averaged over all
spring hours and over all neap hours. Because each average runs over floods and
ebbs together, the tidal part very nearly cancels and what is left is the
RESIDUAL circulation -- the pattern that actually exchanges water with Saratoga
Passage. It is an order of magnitude smaller than the flood and ebb composites
in 20260811_pc_lp_springneap_section.py, so this figure gets its own colour
scale; do not compare the two by eye.

The cancellation is not exact. Spring hours are not split exactly half flood
and half ebb, so a small spurious net remains; the section-mean velocity of
each panel is printed so it can be judged, and it is small next to the
structure being plotted.

HOW SPRING AND NEAP ARE DEFINED. Same as the flood/ebb version, and for the
same reason: the fortnightly signal is a few tenths of the seasonal one, so
binning on the raw envelope would just select seasons. Tidal strength is the
Godin-filtered |Qnet|, its 30-day running mean is removed, and spring / neap
are the upper / lower quartile of the anomaly.

GEOMETRY IS REAL, NOT SIGMA. Cells are drawn at the mean thickness they have
within each panel's own hours, built up from the bed out of DZ, so the axis is
in metres and the sloping bed (h = 15.9 m at the north end, 26.8 m in the
middle, 20.0 m at the south end) is the shape it actually is. Face width is a
uniform dd = 200.5 m. The u = 0 contour is drawn on both panels.

SIGN. Positive q at pc_lp runs minus-side -> plus-side = eastward = OUT of the
cove; corr(qnet, d(ssh)/dt) = -1.00 confirms flood is qnet < 0 in the stored
sign. Everything here is negated once, at load.

CAVEAT. u is the section-normal component only; the along-section component
never enters the tef2 extraction, so |u| is a lower bound on true speed.

Runs on the mac from the local extractions_avg.
run 20260811_pc_lp_springneap_mean_section.py
run 20260811_pc_lp_springneap_mean_section.py -sect pc_cp
run 20260811_pc_lp_springneap_mean_section.py -qq 0.15
"""
import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cmcrameri import cm as cmc

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
p.add_argument('-gctag', default='wb1_pc1', type=str)
p.add_argument('-sect', default='pc_lp', type=str)
p.add_argument('-0', '--ds0', default='2024.01.01', type=str)
p.add_argument('-1', '--ds1', default='2025.12.31', type=str)
p.add_argument('-qq', default=0.25, type=float,
               help='quantile defining spring/neap; 0.25 = outer quartiles')
args = p.parse_args()

warnings.simplefilter('ignore')
Ldir = Lfun.Lstart(gridname=args.gctag.split('_')[0])
tef2 = Ldir['LOo'] / 'extract' / args.gtagex / 'tef2'
ex_fn = tef2 / ('extractions_avg_%s_%s' % (args.ds0, args.ds1)) / (args.sect + '.nc')
out_dir = Ldir['LOo'] / 'DM_outs' / '20260811_pc_lp_flood_ebb'
Lfun.make_dir(out_dir)

FS = 14
BED = '#b9a894'

# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
ds = xr.open_dataset(ex_fn)
tt = pd.to_datetime(ds.time.values)
dd, hh = ds.dd.values, ds.h.values
q = -ds.q.values                        # + = INTO the cove
DZ = ds.DZ.values
ds.close()
u = q / (DZ * dd[None, None, :])
qnet = q.sum(axis=(1, 2))

# ---------------------------------------------------------------------------
# spring / neap on the 30-day anomaly of the Godin-filtered |Qnet|
# ---------------------------------------------------------------------------
env = pd.Series(zfun.lowpass(np.abs(qnet), f='godin'), index=tt)
anom = env - env.rolling(30 * 24, center=True, min_periods=200).mean()
hi, lo = anom.quantile(1 - args.qq), anom.quantile(args.qq)
MASK = {'spring': (anom >= hi).values, 'neap': (anom <= lo).values}

print('%s: %d faces, %d hours' % (args.sect, len(dd), len(tt)))
print('tidal envelope (Godin |Qnet|) mean %.0f m3/s, range %.0f to %.0f'
      % (env.mean(), env.min(), env.max()))
print('spring = 30-day anomaly >= %+.0f m3/s, neap <= %+.0f' % (hi, lo))

CELL = {}
print('\n%-7s %7s %8s %11s %10s %10s %10s'
      % ('tide', 'n [h]', '% flood', 'mean Qnet', 'u min', 'u max', 'area in'))
for tide, m in MASK.items():
    uu, dzm = np.nanmean(u[m], axis=0), DZ[m].mean(axis=0)
    a = dzm * dd[None, :]
    CELL[tide] = (uu, dzm, m)
    print('%-7s %7d %7.0f%% %11.0f %10.3f %10.3f %9.0f%%'
          % (tide, m.sum(), 100 * (qnet[m] > 0).mean(), qnet[m].mean(),
             uu.min(), uu.max(), 100 * (a * (uu > 0)).sum() / a.sum()))
    print('        section-mean u = %+.4f m/s (the residual net; small next to '
          'the %.3f m/s structure)' % ((a * uu).sum() / a.sum(), abs(uu).max()))

# ---------------------------------------------------------------------------
# section geometry
# ---------------------------------------------------------------------------
xe = np.concatenate([[0], np.cumsum(dd)]) / 1000.               # km, north -> south
xc = 0.5 * (xe[:-1] + xe[1:])
zg = np.linspace(-hh.max(), 0, 120)


def z_edges(dzm):
    """Cell interfaces in metres below the surface, built up from the bed."""
    zw = np.zeros((dzm.shape[0] + 1, dzm.shape[1]))
    zw[1:, :] = np.cumsum(dzm, axis=0)
    return -(zw[-1:, :] - zw)


def on_common_grid(uu, dzm):
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


VM = np.ceil(max(abs(v[0]).max() for v in CELL.values()) * 200) / 200.

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
for ax, tide in zip(axes, ['spring', 'neap']):
    uu, dzm, m = CELL[tide]
    zw = z_edges(dzm)
    for j in range(len(dd)):
        ax.pcolormesh(xe[j:j + 2], zw[:, j], uu[:, j:j + 1], cmap=cmc.vik_r,
                      vmin=-VM, vmax=VM, shading='flat', rasterized=True)
    ax.contour(xc, zg, on_common_grid(uu, dzm), levels=[0], colors='k',
               linewidths=1.4)
    # the bed, drawn at the bottom of the cells actually plotted so no white
    # seam opens up between the deepest cell and the fill
    hb = dzm.sum(axis=0)
    xs, zs = np.repeat(xe, 2)[1:-1], np.repeat(-hb, 2)
    ax.fill_between(xs, zs, -hb.max() - 3, color=BED, lw=0, zorder=5)
    ax.plot(xs, zs, color='k', lw=1.0, zorder=6)
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(-hh.max() - 3, 0)
    ax.set_xlabel('distance across the mouth [km]', fontsize=FS)
    ax.set_title('%s mean  (n = %d h)' % (tide, m.sum()), fontsize=FS,
                 loc='left')
    ax.grid(color='lightgray', linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=FS - 2)
axes[0].set_ylabel('depth below surface [m]', fontsize=FS)
axes[0].text(0.01, 1.10, 'NORTH', transform=axes[0].transAxes, fontsize=FS - 2,
             ha='left', color='#555555')
axes[1].text(0.99, 1.10, 'SOUTH', transform=axes[1].transAxes, fontsize=FS - 2,
             ha='right', color='#555555')

sm = plt.cm.ScalarMappable(cmap=cmc.vik_r, norm=plt.Normalize(vmin=-VM, vmax=VM))
cb = fig.colorbar(sm, ax=axes, fraction=0.035, pad=0.02)
cb.set_label('section-normal $u$ [m s$^{-1}$],  blue = into Penn Cove',
             fontsize=FS - 1)
cb.ax.tick_params(labelsize=FS - 3)

fn_out = out_dir / ('springneap_mean_%s_%s.png' % (args.gtagex, args.sect))
fig.savefig(fn_out, dpi=400, bbox_inches='tight', transparent=True)
print('\nwrote ' + str(fn_out))
