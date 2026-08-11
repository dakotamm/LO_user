"""
The mean section across pc_lp in the four combinations of spring/neap tide and
along-cove wind into/out of the cove.

A 2 x 2 of section views: rows are spring and neap, columns are wind into the
cove and wind out of it. Distance across the mouth on x (north on the left),
depth on y, colour is the section-NORMAL velocity

    u(t,z,p) = q / (dd * DZ)                                      [m s-1]

negated so that BLUE is INTO Penn Cove and RED is out of it, averaged over the
hours in each cell. The two one-variable versions are
20260811_pc_lp_springneap_mean_section.py and
20260811_pc_lp_wind_mean_section.py; this one crosses them, to see whether the
wind response looks the same at both ends of the fortnightly cycle.

BOTH SPLITS ARE ON ANOMALIES, AND BOTH HAVE TO BE. The fortnightly signal is a
few tenths of the seasonal one and the along-cove wind has a large seasonal
cycle of its own (out of the cove nearly all summer, into it nearly all
winter), so splitting either on its raw value would mostly sort hours by
season. Tidal strength is the Godin-filtered |Qnet| and wind is the Godin-
filtered along-cove velocity; each has its 30-day running mean removed, and
each is split at the quantiles set by -qtide and -qwind. The month composition
of every cell is printed so the residual confound is always visible.

THE TIDAL PART CANCELS ONLY IF THE CELL IS BALANCED. Each panel is a plain mean
over its hours, so the tidal signal cancels to the extent that the cell holds
equal numbers of flooding and ebbing hours. That is not guaranteed once two
conditions are imposed at once, so the flood fraction and the resulting
section-mean velocity are printed per cell -- read them before reading the
panels. A cell far from 50 % flood is showing a tidal remnant, not a residual.

Cell counts are the product of two quantile bins, so they are much smaller than
in the one-variable figures. -qtide and -qwind default to terciles rather than
quartiles for that reason; the counts are printed.

All four panels share one symmetric colour scale, set by the largest of the
four. It is not the scale of either one-variable figure.

THE WIND AXIS. The line joining the pc_lp centroid to the pc_cp centroid from
structure_*.nc, positive mouth -> head, the same construction as
20260807_pc_alongchannel_wind.py and 20260811_pc_forcing_stack.py. Velocity,
not stress, and the along component only -- see
20260811_pc_lp_wind_mean_section.py for why both of those matter. Composited at
zero lag, since the velocity response is at 0 days.

GEOMETRY IS REAL, NOT SIGMA. Cells are drawn at the mean thickness they have
within each panel's own hours, built up from the bed out of DZ. Face width is a
uniform dd = 200.5 m. The u = 0 contour is drawn on every panel.

SIGN AT THE SECTION. Positive q at pc_lp runs minus-side -> plus-side =
eastward = OUT of the cove; corr(qnet, d(ssh)/dt) = -1.00 confirms flood is
qnet < 0 in the stored sign. Everything here is negated once, at load.

CAVEAT. u is the section-normal component only; the along-section component
never enters the tef2 extraction, so |u| is a lower bound on true speed.

Runs on the mac from the local extractions_avg and the wind pickle.
run 20260811_pc_lp_springneap_wind_section.py
run 20260811_pc_lp_springneap_wind_section.py -qtide 0.25 -qwind 0.25
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
p.add_argument('-qtide', default=1 / 3., type=float,
               help='quantile defining spring/neap on the tidal anomaly')
p.add_argument('-qwind', default=1 / 3., type=float,
               help='quantile defining the wind bins on the wind anomaly')
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
TIDES, WINDS = ['spring', 'neap'], ['wind into the cove', 'wind out of the cove']

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


def anomaly(s):
    """30-day running mean removed, so the split is not a season split."""
    return s - s.rolling(30 * 24, center=True, min_periods=200).mean()


# ---------------------------------------------------------------------------
# the two splits
# ---------------------------------------------------------------------------
env = pd.Series(zfun.lowpass(np.abs(qnet), f='godin'), index=tt)
ta = anomaly(env)
t_hi, t_lo = ta.quantile(1 - args.qtide), ta.quantile(args.qtide)
TMASK = {'spring': (ta >= t_hi).fillna(False).values,
         'neap': (ta <= t_lo).fillna(False).values}

dstr = xr.open_dataset(st_fn)
CEN = {sn: (float(np.mean(dstr['%s_lon' % sn].values)),
            float(np.mean(dstr['%s_lat' % sn].values))) for sn in [MOUTH, HEAD]}
dstr.close()
COS = np.cos(np.deg2rad(np.mean([c[1] for c in CEN.values()])))
ex = (CEN[HEAD][0] - CEN[MOUTH][0]) * COS * 111.32
ey = (CEN[HEAD][1] - CEN[MOUTH][1]) * 111.32
axl = np.hypot(ex, ey)
ex, ey = ex / axl, ey / axl

W = pd.read_pickle(wind_fn)['W']
wa = pd.Series(zfun.lowpass(W.u_pc.values * ex + W.v_pc.values * ey, f='godin'),
               index=W.index)                       # + = blowing INTO the cove
if args.lag:
    wa.index = wa.index + pd.Timedelta(hours=args.lag)
wa = wa.reindex(wa.index.union(tt)).interpolate('time').reindex(tt)
waa = anomaly(wa)
w_hi, w_lo = waa.quantile(1 - args.qwind), waa.quantile(args.qwind)
WMASK = {WINDS[0]: (waa >= w_hi).fillna(False).values,
         WINDS[1]: (waa <= w_lo).fillna(False).values}

print('along-cove axis (%.3f, %.3f), %.0f deg true, mouth -> head, %.2f km'
      % (ex, ey, np.rad2deg(np.arctan2(ex, ey)) % 360, axl))
print('tidal envelope mean %.0f m3/s; spring anomaly >= %+.0f, neap <= %+.0f'
      % (env.mean(), t_hi, t_lo))
print('Godin w_along mean %+.2f m/s; into anomaly >= %+.2f, out <= %+.2f, lag %+d h'
      % (wa.mean(), w_hi, w_lo, args.lag))

CELL = {}
print('\n%-7s %-21s %6s %8s %9s %10s %10s %9s'
      % ('tide', 'wind', 'n [h]', '% flood', 'w_along', 'u min', 'u max', 'mean u'))
for tide in TIDES:
    for wind in WINDS:
        m = TMASK[tide] & WMASK[wind]
        uu, dzm = np.nanmean(u[m], axis=0), DZ[m].mean(axis=0)
        a = dzm * dd[None, :]
        CELL[(tide, wind)] = (uu, dzm, m)
        print('%-7s %-21s %6d %7.0f%% %+9.2f %10.3f %10.3f %+10.4f'
              % (tide, wind, m.sum(), 100 * (qnet[m] > 0).mean(),
                 wa.values[m].mean(), uu.min(), uu.max(),
                 (a * uu).sum() / a.sum()))
        print('%-29s months %s' % ('',
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

fig, axes = plt.subplots(2, 2, figsize=(14, 9.4), sharex=True, sharey=True)
for i, tide in enumerate(TIDES):
    for j, wind in enumerate(WINDS):
        ax = axes[i, j]
        uu, dzm, m = CELL[(tide, wind)]
        zw = z_edges(dzm)
        for jj in range(len(dd)):
            ax.pcolormesh(xe[jj:jj + 2], zw[:, jj], uu[:, jj:jj + 1],
                          cmap=cmc.vik_r, vmin=-VM, vmax=VM, shading='flat',
                          rasterized=True)
        ax.contour(xc, zg, on_common_grid(uu, dzm), levels=[0], colors='k',
                   linewidths=1.4)
        # the bed, drawn at the bottom of the cells actually plotted so no
        # white seam opens up between the deepest cell and the fill
        hb = dzm.sum(axis=0)
        xs, zs = np.repeat(xe, 2)[1:-1], np.repeat(-hb, 2)
        ax.fill_between(xs, zs, -hb.max() - 3, color=BED, lw=0, zorder=5)
        ax.plot(xs, zs, color='k', lw=1.0, zorder=6)
        ax.set_xlim(xe[0], xe[-1])
        ax.set_ylim(-hh.max() - 3, 0)
        ax.set_title('%s, %s\n(n = %d h, $w_{along}$ = %+.2f m s$^{-1}$, '
                     '%.0f%% flood)'
                     % (tide, wind, m.sum(), wa.values[m].mean(),
                        100 * (qnet[m] > 0).mean()), fontsize=FS - 1, loc='left')
        ax.grid(color='lightgray', linestyle='--', alpha=0.4)
        ax.tick_params(labelsize=FS - 2)
    axes[i, 0].set_ylabel('depth below surface [m]', fontsize=FS)
for j in range(2):
    axes[1, j].set_xlabel('distance across the mouth [km]', fontsize=FS)
axes[0, 0].text(0.01, 1.20, 'NORTH', transform=axes[0, 0].transAxes,
                fontsize=FS - 2, ha='left', color='#555555')
axes[0, 1].text(0.99, 1.20, 'SOUTH', transform=axes[0, 1].transAxes,
                fontsize=FS - 2, ha='right', color='#555555')

sm = plt.cm.ScalarMappable(cmap=cmc.vik_r, norm=plt.Normalize(vmin=-VM, vmax=VM))
cb = fig.colorbar(sm, ax=axes, fraction=0.035, pad=0.02)
cb.set_label('section-normal $u$ [m s$^{-1}$],  blue = into Penn Cove',
             fontsize=FS - 1)
cb.ax.tick_params(labelsize=FS - 3)

fn_out = out_dir / ('springneap_wind_%s_%s.png' % (args.gtagex, args.sect))
fig.savefig(fn_out, dpi=400, bbox_inches='tight', transparent=True)
print('\nwrote ' + str(fn_out))
