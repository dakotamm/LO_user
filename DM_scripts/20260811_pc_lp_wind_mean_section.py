"""
The mean section across pc_lp when the along-cove wind blows INTO the cove and
when it blows OUT of it.

Two section views, the wind analogue of 20260811_pc_lp_springneap_mean_section.py:
distance across the mouth on x (north on the left), depth on y, colour is the
section-NORMAL velocity

    u(t,z,p) = q / (dd * DZ)                                      [m s-1]

negated so that BLUE is INTO Penn Cove and RED is out of it, averaged over the
hours in each wind bin. Each bin holds floods and ebbs in almost equal numbers,
so the tidal part cancels and what is left is the residual circulation.

THE AXIS AND ITS SIGN. The along-cove axis is the line joining the pc_lp
centroid to the pc_cp centroid, taken from structure_*.nc -- the same
construction as 20260807_pc_alongchannel_wind.py and 20260811_pc_forcing_stack.py,
so "into the cove" means the same thing in all three. w_along is the region-mean
wind VECTOR projected onto that axis; positive is mouth -> head. Velocity and
not stress, for the reason set out in 20260811_pc_forcing_stack.py: tau_pc is
the mean of the per-cell stress MAGNITUDE, so projecting it mixes two different
averages. The cross-cove component is not used here, which sidesteps the known
mislabelling of cross-cove sign in reduce_wind_cove.py.

THE SPLIT IS ON AN ANOMALY, AND IT HAS TO BE. The along-cove wind has a large
seasonal cycle -- it blows out of the cove nearly all summer and into it nearly
all winter -- so splitting on the raw sign does not produce a wind composite,
it produces a season composite. On the raw Godin-filtered wind, 74 % of the
"into" hours fall in Nov-Feb and 86 % of the "out" hours in Apr-Aug. The
default therefore removes a 30-day running mean first and splits on the
quartiles of the anomaly, which spreads both bins across every month (5-12 %
each) while still separating the wind itself: mean total w_along is +0.7 m/s in
the "into" bin against -2.5 m/s in the "out" bin. Pass -mode raw to get the
season-confounded version deliberately; the month table is printed either way
so the confound is always visible.

LAG. Composited at zero lag by default. The Penn Cove velocity response to
along-cove wind is at 0 days -- it is the salinity GRADIENT that lags by 1-2
days -- so zero is the right choice for a velocity figure. -lag takes hours if
you want to check that.

Both panels share one symmetric colour scale, set by the larger of the two.

GEOMETRY IS REAL, NOT SIGMA. Cells are drawn at the mean thickness they have
within each panel's own hours, built up from the bed out of DZ, so the axis is
in metres and the sloping bed (h = 15.9 m at the north end, 26.8 m in the
middle, 20.0 m at the south end) is the shape it actually is. Face width is a
uniform dd = 200.5 m. The u = 0 contour is drawn on both panels.

SIGN AT THE SECTION. Positive q at pc_lp runs minus-side -> plus-side =
eastward = OUT of the cove; corr(qnet, d(ssh)/dt) = -1.00 confirms flood is
qnet < 0 in the stored sign. Everything here is negated once, at load.

CAVEAT. u is the section-normal component only; the along-section component
never enters the tef2 extraction, so |u| is a lower bound on true speed.

Runs on the mac from the local extractions_avg and the wind pickle.
run 20260811_pc_lp_wind_mean_section.py
run 20260811_pc_lp_wind_mean_section.py -mode raw
run 20260811_pc_lp_wind_mean_section.py -lag 24
"""
import argparse
import sys
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
p.add_argument('-mode', default='anom', choices=['anom', 'raw'],
               help='anom = split on the 30-day anomaly (default); '
                    'raw = split on the Godin wind itself, season-confounded')
p.add_argument('-qq', default=0.25, type=float,
               help='quantile defining the two bins; 0.25 = outer quartiles')
p.add_argument('-lag', default=0, type=int,
               help='hours the section lags the wind; 0 for velocity')
args = p.parse_args()

warnings.simplefilter('ignore')
Ldir = Lfun.Lstart(gridname=args.gctag.split('_')[0])
tef2 = Ldir['LOo'] / 'extract' / args.gtagex / 'tef2'
ex_fn = tef2 / ('extractions_avg_%s_%s' % (args.ds0, args.ds1)) / (args.sect + '.nc')
st_fn = tef2 / ('structure_%s_%s_%s.nc' % (args.ds0, args.ds1, args.gctag))
wind_fn = (Ldir['LOo'] / 'DM_outs' / '20260806_wind'
           / ('wind_hourly_atm00_%s_%s.p' % (args.ds0, args.ds1)))
out_dir = Ldir['LOo'] / 'DM_outs' / '20260811_pc_lp_flood_ebb'
Lfun.make_dir(out_dir)

FS = 14
BED = '#b9a894'
MOUTH, HEAD = 'pc_lp', 'pc_cp'

for fn in [ex_fn, st_fn, wind_fn]:
    if not fn.is_file():
        print('*** missing: %s' % fn)
        sys.exit(1)

# ---------------------------------------------------------------------------
# load the section
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
# the along-cove wind, projected onto the mouth -> head axis
# ---------------------------------------------------------------------------
dstr = xr.open_dataset(st_fn)
CEN = {sn: (float(np.mean(dstr['%s_lon' % sn].values)),
            float(np.mean(dstr['%s_lat' % sn].values))) for sn in [MOUTH, HEAD]}
dstr.close()
COS = np.cos(np.deg2rad(np.mean([c[1] for c in CEN.values()])))
ex = (CEN[HEAD][0] - CEN[MOUTH][0]) * COS * 111.32
ey = (CEN[HEAD][1] - CEN[MOUTH][1]) * 111.32
axl = np.hypot(ex, ey)
ex, ey = ex / axl, ey / axl
print('along-cove axis (%.3f, %.3f), %.0f deg true, mouth -> head, %.2f km'
      % (ex, ey, np.rad2deg(np.arctan2(ex, ey)) % 360, axl))

W = pd.read_pickle(wind_fn)['W']
wa = pd.Series(zfun.lowpass(W.u_pc.values * ex + W.v_pc.values * ey, f='godin'),
               index=W.index)                       # + = blowing INTO the cove
if args.lag:
    wa.index = wa.index + pd.Timedelta(hours=args.lag)
# the wind is on the hour, the extraction on the half hour
wa = wa.reindex(wa.index.union(tt)).interpolate('time').reindex(tt)

if args.mode == 'anom':
    split = wa - wa.rolling(30 * 24, center=True, min_periods=200).mean()
    lbl_split = '30-day anomaly of the Godin along-cove wind'
else:
    split = wa
    lbl_split = 'Godin along-cove wind (season-confounded)'
hi, lo = split.quantile(1 - args.qq), split.quantile(args.qq)
MASK = {'wind into the cove': (split >= hi).fillna(False).values,
        'wind out of the cove': (split <= lo).fillna(False).values}

print('\nsplit on the %s, lag %+d h' % (lbl_split, args.lag))
print('  Godin w_along: mean %+.2f m/s, blowing into the cove %.0f%% of hours'
      % (wa.mean(), 100 * (wa > 0).mean()))
print('  bins: >= %+.2f and <= %+.2f' % (hi, lo))

CELL = {}
print('\n%-21s %7s %8s %11s %10s %10s %9s'
      % ('bin', 'n [h]', '% flood', 'w_along', 'u min', 'u max', 'area in'))
for k, m in MASK.items():
    uu, dzm = np.nanmean(u[m], axis=0), DZ[m].mean(axis=0)
    a = dzm * dd[None, :]
    CELL[k] = (uu, dzm, m, (a * uu).sum() / a.sum())
    print('%-21s %7d %7.0f%% %+11.2f %10.3f %10.3f %8.0f%%'
          % (k, m.sum(), 100 * (qnet[m] > 0).mean(), wa.values[m].mean(),
             uu.min(), uu.max(), 100 * (a * (uu > 0)).sum() / a.sum()))
    print('%-21s   section-mean u = %+.4f m/s; months %s'
          % ('', CELL[k][3],
             np.round(100 * pd.Series(1, index=tt[m]).groupby(tt[m].month).size()
                      / m.sum()).astype(int).to_dict()))

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
for ax, k in zip(axes, list(MASK)):
    uu, dzm, m, _ = CELL[k]
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
    ax.set_title('%s  (n = %d h, $w_{along}$ = %+.2f m s$^{-1}$)'
                 % (k, m.sum(), wa.values[m].mean()), fontsize=FS, loc='left')
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

fn_out = out_dir / ('wind_mean_%s_%s_%s.png'
                    % (args.gtagex, args.sect, args.mode))
fig.savefig(fn_out, dpi=400, bbox_inches='tight', transparent=True)
print('\nwrote ' + str(fn_out))
