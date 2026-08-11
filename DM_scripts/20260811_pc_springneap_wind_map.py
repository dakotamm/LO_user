"""
Penn Cove in plan view in the four combinations of spring/neap tide and
along-cove wind into/out of the cove.

The map counterpart of 20260811_pc_lp_springneap_wind_section.py: same four
cells, same splits, same sign convention, but the whole cove instead of one
section. Rows are spring and neap, columns are wind into the cove and wind out
of it. Each panel is the depth-averaged velocity

    ubar = <QU2> / (dy * (h_u + <zeta>))                           [m s-1]

shaded by its ALONG-COVE (westward) component so that BLUE is INTO Penn Cove
and RED is out of it, with the full vector on top. The one-variable map
versions are 20260811_pc_flood_ebb_springneap_map.py and
20260811_pc_wind_map.py; this crosses the tide with the wind, to see whether
the wind does the same thing to the gyre at both ends of the fortnightly cycle.

BOTH SPLITS ARE ON ANOMALIES, AND BOTH HAVE TO BE. The fortnightly signal is a
few tenths of the seasonal one and the along-cove wind has a large seasonal
cycle of its own (out of the cove nearly all summer, into it nearly all
winter), so splitting either on its raw value would mostly sort hours by
season. Tidal strength is the Godin-filtered |Qnet| at the mouth and wind is
the Godin-filtered along-cove velocity; each has its 30-day running mean
removed and is split at -qtide / -qwind. The month composition of every cell is
printed so the residual confound stays visible.

THE TIDAL PART CANCELS ONLY IF THE CELL IS BALANCED. Each panel is a plain mean
over its hours, so the tide cancels only to the extent that the cell holds
equal numbers of flooding and ebbing hours -- which is not guaranteed once two
conditions are imposed at once. The flood fraction is printed per cell and put
in every panel title: read it before reading the panel. A cell far from 50%
flood is showing a tidal remnant, not a residual circulation.

Cell counts are the product of two quantile bins, so they are much smaller than
in the one-variable figures; -qtide and -qwind therefore default to terciles
rather than quartiles.

WHY THIS ONE OPENS THE SECTION EXTRACTION. The spring/neap split has to be the
same series the section figures use, so Qnet comes from the tef2 extraction
even though the cove box's mouth column is the same faces. That file is on the
ocean_avg clock (hourly means at :30) and the maps are on ocean_his (on the
hour), so it is interpolated across, and the pickle's own mouth transport is
compared against it as a check. 20260811_pc_wind_map.py skips the extraction
because it only needs the flood fraction, where the pickle is enough.

THE WIND AXIS. The line joining the pc_lp centroid to the pc_cp centroid from
structure_*.nc, positive mouth -> head, the same construction as
20260807_pc_alongchannel_wind.py and 20260811_pc_forcing_stack.py. Velocity,
not stress, and the along component only. Composited at zero lag, since the
velocity response is at 0 days.

All four panels share one symmetric colour scale, set by the largest of the
four and rounded to 0.005 m/s -- these are residual-scale panels, not the
0.02 m/s of the tidal composites. It is not the scale of either one-variable
figure.

Runs on the mac. Needs the hourly transports from 20260807_pc_turning_reduce.py,
the local extractions_avg and the wind pickle.
run 20260811_pc_springneap_wind_map.py
run 20260811_pc_springneap_wind_map.py -qtide 0.25 -qwind 0.25
"""
import argparse
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import ListedColormap
from cmcrameri import cm as cmc

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
p.add_argument('-gctag', default='wb1_pc1', type=str)
p.add_argument('-sect', default='pc_lp', type=str)
p.add_argument('-0', '--ds0', default='2024.01.01', type=str)
p.add_argument('-1', '--ds1', default='2025.12.31', type=str)
p.add_argument('-f', '--fn', default='turning_his_wb1_t0_xn11abbur00_'
                                     '2024.01.01_2025.12.31.p',
               help='pickle from 20260807_pc_turning_reduce.py')
p.add_argument('-qtide', default=1 / 3., type=float,
               help='quantile defining spring/neap on the tidal anomaly')
p.add_argument('-qwind', default=1 / 3., type=float,
               help='quantile defining the wind bins on the wind anomaly')
p.add_argument('-lag', default=0, type=int,
               help='hours the cove lags the wind; 0 for velocity')
p.add_argument('--quiver-step', default=1, type=int, dest='quiver_step')
p.add_argument('--vmax', type=float, help='colour limit for all four panels [m/s]')
args = p.parse_args()

warnings.simplefilter('ignore')
Ldir = Lfun.Lstart(gridname=args.gctag.split('_')[0])
tef2 = Ldir['LOo'] / 'extract' / args.gtagex / 'tef2'
ex_fn = tef2 / ('extractions_avg_%s_%s' % (args.ds0, args.ds1)) / (args.sect + '.nc')
st_fn = tef2 / ('structure_%s_%s_%s.nc' % (args.ds0, args.ds1, args.gctag))
wind_fn = (Ldir['LOo'] / 'DM_outs' / '20260806_wind'
           / ('wind_hourly_atm00_%s_%s.p' % (args.ds0, args.ds1)))
turn_fn = Ldir['LOo'] / 'DM_outs' / '20260807_pc_turning' / args.fn
out_dir = Ldir['LOo'] / 'DM_outs' / '20260811_pc_lp_flood_ebb'
Lfun.make_dir(out_dir)
for fn in [ex_fn, st_fn, wind_fn, turn_fn]:
    if not fn.is_file():
        print('*** missing: %s' % fn)
        sys.exit(1)

FS = 14
BED = '#b9a894'
CMAP = cmc.vik_r
MOUTH, HEAD = 'pc_lp', 'pc_cp'
TIDES = ['spring', 'neap']
WINDS = ['wind into the cove', 'wind out of the cove']

# ---------------------------------------------------------------------------
# the maps
# ---------------------------------------------------------------------------
D = pd.read_pickle(turn_fn)
th = pd.DatetimeIndex(D['time_h'])
h, cove, UM, VM_, lonr, latr = (D['h'], D['cove'], D['UM'], D['VM'],
                                D['lon_rho'], D['lat_rho'])
QU2h = np.where(UM[None], D['QU2_h'], np.nan)
QV2h = np.where(VM_[None], D['QV2_h'], np.nan)
zeta_h = D['zeta_h']
h_u = 0.5 * (h[:, :-1] + h[:, 1:])
h_v = 0.5 * (h[:-1, :] + h[1:, :])
im = int(list(D['iu_glob']).index(D['mouth_iu']))


def onto_his(s):
    """Any hourly series on the his clock. Interpolated, not nearest-matched:
    half an hour is 15 degrees of M2 phase, enough to put a slack-water hour in
    the wrong bin."""
    return s.reindex(s.index.union(th)).interpolate('time').reindex(th)


def anomaly(s):
    """30-day running mean removed, so the split is not a season split."""
    return s - s.rolling(30 * 24, center=True, min_periods=200).mean()


# ---------------------------------------------------------------------------
# split 1: spring / neap, on the section's own Qnet
# ---------------------------------------------------------------------------
ds = xr.open_dataset(ex_fn)
ts = pd.to_datetime(ds.time.values)
qnet_s = pd.Series((-ds.q.values).sum(axis=(1, 2)), index=ts)   # + = INTO cove
ds.close()
env = pd.Series(zfun.lowpass(np.abs(qnet_s.values), f='godin'), index=ts)
ta = onto_his(anomaly(env))
qn = onto_his(qnet_s)
t_hi, t_lo = ta.quantile(1 - args.qtide), ta.quantile(args.qtide)
TMASK = {'spring': (ta >= t_hi).fillna(False).values,
         'neap': (ta <= t_lo).fillna(False).values}

q_mouth = -np.nansum(QU2h[:, :, im], axis=1)          # the pickle's own version
print('tidal envelope mean %.0f m3/s; spring anomaly >= %+.0f, neap <= %+.0f'
      % (env.mean(), t_hi, t_lo))
print('Qnet: section vs the pickle mouth column, r = %+.3f, same sign on %.1f%% '
      'of hours' % (np.corrcoef(q_mouth, np.nan_to_num(qn.values))[0, 1],
                    100 * np.mean(np.sign(q_mouth) == np.sign(qn.values))))

# ---------------------------------------------------------------------------
# split 2: along-cove wind
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

W = pd.read_pickle(wind_fn)['W']
wa = pd.Series(zfun.lowpass(W.u_pc.values * ex + W.v_pc.values * ey, f='godin'),
               index=W.index)                       # + = blowing INTO the cove
if args.lag:
    wa.index = wa.index + pd.Timedelta(hours=args.lag)
wa = onto_his(wa)
waa = anomaly(wa)
w_hi, w_lo = waa.quantile(1 - args.qwind), waa.quantile(args.qwind)
WMASK = {WINDS[0]: (waa >= w_hi).fillna(False).values,
         WINDS[1]: (waa <= w_lo).fillna(False).values}
print('along-cove axis (%.3f, %.3f), %.0f deg true, mouth -> head, %.2f km'
      % (ex, ey, np.rad2deg(np.arctan2(ex, ey)) % 360, axl))
print('Godin w_along mean %+.2f m/s; into anomaly >= %+.2f, out <= %+.2f, '
      'lag %+d h' % (wa.mean(), w_hi, w_lo, args.lag))

# ---------------------------------------------------------------------------
# the four composites
# ---------------------------------------------------------------------------
AREA = np.where(cove, D['area'], np.nan)
CELL = {}
print('\n%-7s %-21s %6s %8s %9s %10s %10s %10s'
      % ('tide', 'wind', 'n [h]', '% flood', 'w_along', 'mean zeta', 'max |u|',
         'mouth Qex'))
for tide in TIDES:
    for wind in WINDS:
        m = TMASK[tide] & WMASK[wind]
        if m.sum() < 24:
            print('*** only %d hours in %s / %s -- loosen -qtide/-qwind'
                  % (m.sum(), tide, wind))
            sys.exit(1)
        zb = float(np.nanmean(zeta_h[m]))      # the cell's own mean sea level
        ub = np.nanmean(QU2h[m], axis=0) / (D['DYU'] * (h_u + zb))
        vb = np.nanmean(QV2h[m], axis=0) / (D['DXV'] * (h_v + zb))
        u_rho = np.full(h.shape, np.nan)
        v_rho = np.full(h.shape, np.nan)
        # a dry face carries no flow, so it enters as zero rather than as a NaN
        # that would eat the whole cell
        u_rho[:, 1:-1] = 0.5 * (np.nan_to_num(ub[:, :-1]) + np.nan_to_num(ub[:, 1:]))
        v_rho[1:-1, :] = 0.5 * (np.nan_to_num(vb[:-1, :]) + np.nan_to_num(vb[1:, :]))
        u_rho[~cove], v_rho[~cove] = np.nan, np.nan
        # the lateral exchange at the mouth: half the sum of |mean transport|
        # over the mouth faces, i.e. how much water the gyre turns over
        Qex = float(0.5 * np.nansum(np.abs(np.nanmean(QU2h[m], axis=0)[:, im])))
        CELL[(tide, wind)] = dict(u_in=-u_rho, u=u_rho, v=v_rho, m=m,
                                  w=wa.values[m].mean(), Qex=Qex,
                                  fl=100 * np.mean(q_mouth[m] > 0))
        print('%-7s %-21s %6d %7.0f%% %+9.2f %10.3f %10.3f %10.0f'
              % (tide, wind, m.sum(), CELL[(tide, wind)]['fl'],
                 wa.values[m].mean(), zb, np.nanmax(np.hypot(u_rho, v_rho)), Qex))
        print('%-29s months %s' % ('',
              np.round(100 * pd.Series(1, index=th[m]).groupby(th[m].month).size()
                       / m.sum()).astype(int).to_dict()))

# The question the figure exists for: is the wind response the same at both ends
# of the fortnightly cycle? Compared row by row rather than left to the eye.
print('\nwind response (into minus out of the cove), within each tide:')
for tide in TIDES:
    A, B = CELL[(tide, WINDS[0])], CELL[(tide, WINDS[1])]
    dif = A['u_in'] - B['u_in']
    print('  %-6s dw = %+.2f m/s -> mouth exchange %.0f to %.0f m3/s (%+.0f%%), '
          'max |du| %.4f m/s'
          % (tide, A['w'] - B['w'], B['Qex'], A['Qex'],
             100 * (A['Qex'] / B['Qex'] - 1), np.nanmax(np.abs(dif))))

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
VM = (args.vmax if args.vmax else
      np.ceil(max(np.nanmax(np.abs(C['u_in'])) for C in CELL.values()) * 200) / 200.)
qs = max(1, args.quiver_step)
S95 = float(np.nanpercentile(
    np.concatenate([np.hypot(C['u'], C['v'])[cove] for C in CELL.values()]), 95))
qscale = S95 * lonr.shape[1] / (1.2 * qs)
print('\ncolour scale +/- %.3f m/s, quiver key %.3f m/s' % (VM, S95))

DAR = 1 / np.cos(np.deg2rad(float(np.mean(latr))))
dlon = float(np.diff(lonr[0, :]).mean())
dlat = float(np.diff(latr[:, 0]).mean())
land = ~D['mask_rho']
SL = pd.read_pickle(Ldir['LOo'] / 'section_lines' / (MOUTH + '.p'))

fig, axes = plt.subplots(2, 2, figsize=(15, 9.2), sharex=True, sharey=True)
for i, tide in enumerate(TIDES):
    for j, wind in enumerate(WINDS):
        ax = axes[i, j]
        C = CELL[(tide, wind)]
        ax.pcolormesh(lonr, latr, np.ma.masked_where(~land, np.ones(land.shape)),
                      cmap=ListedColormap([BED]), shading='nearest', zorder=0)
        ax.pcolormesh(lonr, latr, C['u_in'], cmap=CMAP, vmin=-VM, vmax=VM,
                      shading='nearest', zorder=1, rasterized=True)
        Q = ax.quiver(lonr[::qs, ::qs], latr[::qs, ::qs],
                      C['u'][::qs, ::qs], C['v'][::qs, ::qs], scale=qscale,
                      scale_units='width', units='width', width=0.0030,
                      color='k', zorder=6)
        ax.plot(SL.x.astype(float), SL.y.astype(float), '-', color='k', lw=2.0,
                zorder=9, solid_capstyle='butt')
        # half a cell of margin: shading='nearest' centres each cell on its rho
        # point, so limits at the rho points cut the outer cells in half
        ax.set_xlim(lonr.min() - dlon / 2, lonr.max() + dlon / 2)
        ax.set_ylim(latr.min() - dlat / 2, latr.max() + dlat / 2)
        ax.set_aspect(DAR)
        # the flood fraction rides in the title: a cell far from 50% is showing
        # a tidal remnant rather than a residual
        ax.set_title('%s, %s\n(n = %d h, $w_{along}$ = %+.2f m s$^{-1}$, '
                     '%.0f%% flood)' % (tide, wind, C['m'].sum(), C['w'], C['fl']),
                     fontsize=FS - 1, loc='left')
        ax.tick_params(labelsize=FS - 3)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    axes[i, 0].set_ylabel('latitude', fontsize=FS)
for j in range(2):
    axes[1, j].set_xlabel('longitude', fontsize=FS)
# on the axes that OWNS Q -- coordinates='axes' resolves against Q's own axes
Q.axes.quiverkey(Q, 0.97, 0.06, round(S95, 3), '%.3f m s$^{-1}$' % S95,
                 labelpos='W', coordinates='axes',
                 fontproperties=dict(size=FS - 3))

sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=-VM, vmax=VM))
cb = fig.colorbar(sm, ax=axes, fraction=0.035, pad=0.02)
cb.set_label('depth-averaged along-cove $u$ [m s$^{-1}$]\nblue = into Penn Cove',
             fontsize=FS - 1)
cb.ax.tick_params(labelsize=FS - 3)

fn_out = out_dir / ('springneap_wind_map_%s.png' % args.gtagex)
fig.savefig(fn_out, dpi=400, bbox_inches='tight', transparent=True)
print('\nwrote ' + str(fn_out))
