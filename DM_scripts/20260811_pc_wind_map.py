"""
Penn Cove in plan view when the along-cove wind blows INTO the cove and when it
blows OUT of it.

The map counterpart of 20260811_pc_lp_wind_mean_section.py: same two bins, same
axis, same sign convention, but the whole cove instead of one section. Each
panel is the depth-averaged velocity

    ubar = <QU2> / (dy * (h_u + <zeta>))                           [m s-1]

shaded by its ALONG-COVE (westward) component so that BLUE is INTO Penn Cove
and RED is out of it, with the full vector on top. The section says how the
wind rearranges the exchange over depth at the mouth; this says what it does to
the gyre inside.

THESE ARE RESIDUAL-SCALE PANELS, unlike the flood/ebb maps. Each wind bin holds
floods and ebbs in near-equal numbers (the fraction is printed), so the
barotropic tide cancels out of the average and what is left is the residual
circulation -- about a tenth of the tidal velocity. The colour scale is
therefore rounded to 0.005 m/s rather than the 0.02 used for the tidal
composites, exactly as in the section pair.

THE AXIS AND ITS SIGN. The along-cove axis is the line joining the pc_lp
centroid to the pc_cp centroid, taken from structure_*.nc -- the same
construction as 20260807_pc_alongchannel_wind.py, 20260811_pc_forcing_stack.py
and the wind section, so "into the cove" means the same thing in all of them.
w_along is the region-mean wind VECTOR projected onto that axis, positive mouth
-> head. Velocity and not stress: tau_pc is the mean of the per-cell stress
MAGNITUDE, so projecting it mixes two different averages. The cross-cove
component is not used, which sidesteps the known mislabelling of cross-cove
sign in reduce_wind_cove.py.

THE SPLIT IS ON AN ANOMALY, AND IT HAS TO BE. The along-cove wind has a large
seasonal cycle -- out of the cove nearly all summer, into it nearly all winter
-- so splitting on the raw sign gives a season composite rather than a wind
composite. The default removes a 30-day running mean first and splits on the
quartiles of the anomaly; the month distribution of each bin is printed either
way so the confound stays visible. -mode raw gives the season-confounded
version deliberately.

LAG. Zero by default: the Penn Cove velocity response to along-cove wind is at
0 days -- it is the salinity GRADIENT that lags 1-2 days -- so zero is right
for a velocity figure. -lag takes hours.

WHERE THE FLOOD/EBB BALANCE COMES FROM. Not the tef2 extraction: the mouth
column of the cove box IS pc_lp, so the check is made on the pickle's own
hourly transport and this script never opens the half-gigabyte section file.

Runs on the mac. Needs the hourly transports from 20260807_pc_turning_reduce.py
and the wind pickle from 20260806_wind_reduce.py.
run 20260811_pc_wind_map.py
run 20260811_pc_wind_map.py -mode raw
run 20260811_pc_wind_map.py -lag 24
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
p.add_argument('-0', '--ds0', default='2024.01.01', type=str)
p.add_argument('-1', '--ds1', default='2025.12.31', type=str)
p.add_argument('-f', '--fn', default='turning_his_wb1_t0_xn11abbur00_'
                                     '2024.01.01_2025.12.31.p',
               help='pickle from 20260807_pc_turning_reduce.py')
p.add_argument('-mode', default='anom', choices=['anom', 'raw'],
               help='anom = split on the 30-day anomaly (default); '
                    'raw = split on the Godin wind itself, season-confounded')
p.add_argument('-qq', default=0.25, type=float,
               help='quantile defining the two bins; 0.25 = outer quartiles')
p.add_argument('-lag', default=0, type=int,
               help='hours the cove lags the wind; 0 for velocity')
p.add_argument('--quiver-step', default=1, type=int, dest='quiver_step')
p.add_argument('--vmax', type=float, help='colour limit for both panels [m/s]')
args = p.parse_args()

warnings.simplefilter('ignore')
Ldir = Lfun.Lstart(gridname=args.gctag.split('_')[0])
tef2 = Ldir['LOo'] / 'extract' / args.gtagex / 'tef2'
st_fn = tef2 / ('structure_%s_%s_%s.nc' % (args.ds0, args.ds1, args.gctag))
wind_fn = (Ldir['LOo'] / 'DM_outs' / '20260806_wind'
           / ('wind_hourly_atm00_%s_%s.p' % (args.ds0, args.ds1)))
turn_fn = Ldir['LOo'] / 'DM_outs' / '20260807_pc_turning' / args.fn
out_dir = Ldir['LOo'] / 'DM_outs' / '20260811_pc_lp_flood_ebb'
Lfun.make_dir(out_dir)
for fn in [st_fn, wind_fn, turn_fn]:
    if not fn.is_file():
        print('*** missing: %s' % fn)
        sys.exit(1)

FS = 14
BED = '#b9a894'
CMAP = cmc.vik_r
MOUTH, HEAD = 'pc_lp', 'pc_cp'

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

# the wind pickle and the history files are both on the hour, but interpolate
# rather than assume it -- a -lag that is not a whole hour would slide them
wa = wa.reindex(wa.index.union(th)).interpolate('time').reindex(th)

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

# The tide has to cancel inside each bin or these are not residual panels. The
# mouth column of the cove box IS pc_lp, so the check needs no extra file.
im = int(list(D['iu_glob']).index(D['mouth_iu']))
q_mouth = -np.nansum(QU2h[:, :, im], axis=1)                    # + = INTO cove

AREA = np.where(cove, D['area'], np.nan)
CELL = {}
print('\n%-21s %7s %8s %11s %10s %10s %9s'
      % ('bin', 'n [h]', '% flood', 'w_along', 'mean zeta', 'max |u|', 'area in'))
for k, m in MASK.items():
    if m.sum() == 0:
        print('*** no hours in %s' % k)
        sys.exit(1)
    zb = float(np.nanmean(zeta_h[m]))          # the bin's own mean sea level
    ub = np.nanmean(QU2h[m], axis=0) / (D['DYU'] * (h_u + zb))
    vb = np.nanmean(QV2h[m], axis=0) / (D['DXV'] * (h_v + zb))
    u_rho = np.full(h.shape, np.nan)
    v_rho = np.full(h.shape, np.nan)
    # a dry face carries no flow, so it enters as zero rather than as a NaN
    # that would eat the whole cell
    u_rho[:, 1:-1] = 0.5 * (np.nan_to_num(ub[:, :-1]) + np.nan_to_num(ub[:, 1:]))
    v_rho[1:-1, :] = 0.5 * (np.nan_to_num(vb[:-1, :]) + np.nan_to_num(vb[1:, :]))
    u_rho[~cove], v_rho[~cove] = np.nan, np.nan
    u_in = -u_rho                              # + = westward = INTO the cove
    CELL[k] = dict(u_in=u_in, u=u_rho, v=v_rho, m=m, w=wa.values[m].mean())
    print('%-21s %7d %7.0f%% %+11.2f %10.3f %10.3f %8.0f%%'
          % (k, m.sum(), 100 * (q_mouth[m] > 0).mean(), wa.values[m].mean(), zb,
             np.nanmax(np.hypot(u_rho, v_rho)),
             100 * np.nansum(AREA * (u_in > 0)) / np.nansum(AREA)))
    # the lateral exchange at the mouth: half the sum of |mean transport| over
    # the mouth faces, i.e. how much water the gyre is turning over, which is
    # the number the two panels are really being compared on
    CELL[k]['Q_ex'] = float(0.5 * np.nansum(
        np.abs(np.nanmean(QU2h[m], axis=0)[:, im])))
    print('%-21s   mouth exchange %.0f m3/s; months %s'
          % ('', CELL[k]['Q_ex'],
             np.round(100 * pd.Series(1, index=th[m]).groupby(th[m].month)
                      .size() / m.sum()).astype(int).to_dict()))

# The two panels are similar by eye, so the comparison is made numerically as
# well: the difference field, and what it does to the gyre at the mouth.
A, B = [CELL[k] for k in MASK]
dif = A['u_in'] - B['u_in']
print('\ninto minus out of the cove, for a %+.2f m/s change in w_along:'
      % (A['w'] - B['w']))
print('  mouth exchange %.0f -> %.0f m3/s (%+.0f%%)'
      % (B['Q_ex'], A['Q_ex'], 100 * (A['Q_ex'] / B['Q_ex'] - 1)))
print('  along-cove velocity differs by up to %.4f m/s (%.0f%% of the colour '
      'scale),\n  cove-mean |difference| %.4f m/s'
      % (np.nanmax(np.abs(dif)),
         100 * np.nanmax(np.abs(dif)) / max(np.nanmax(np.abs(A['u_in'])),
                                            np.nanmax(np.abs(B['u_in']))),
         np.nanmean(np.abs(dif[cove]))))

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
# rounded to 0.005, not the 0.02 of the tidal composites: the tide cancels
# inside each bin, so these panels are residual-scale
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

fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), sharex=True, sharey=True)
for ax, k in zip(axes, list(MASK)):
    C = CELL[k]
    ax.pcolormesh(lonr, latr, np.ma.masked_where(~land, np.ones(land.shape)),
                  cmap=ListedColormap([BED]), shading='nearest', zorder=0)
    ax.pcolormesh(lonr, latr, C['u_in'], cmap=CMAP, vmin=-VM, vmax=VM,
                  shading='nearest', zorder=1, rasterized=True)
    Q = ax.quiver(lonr[::qs, ::qs], latr[::qs, ::qs],
                  C['u'][::qs, ::qs], C['v'][::qs, ::qs], scale=qscale,
                  scale_units='width', units='width', width=0.0030, color='k',
                  zorder=6)
    ax.plot(SL.x.astype(float), SL.y.astype(float), '-', color='k', lw=2.0,
            zorder=9, solid_capstyle='butt')
    # half a cell of margin: shading='nearest' centres each cell on its rho
    # point, so limits at the rho points cut the outer cells in half
    ax.set_xlim(lonr.min() - dlon / 2, lonr.max() + dlon / 2)
    ax.set_ylim(latr.min() - dlat / 2, latr.max() + dlat / 2)
    ax.set_aspect(DAR)
    ax.set_xlabel('longitude', fontsize=FS)
    ax.set_title('%s  (n = %d h, $w_{along}$ = %+.2f m s$^{-1}$)'
                 % (k, C['m'].sum(), C['w']), fontsize=FS, loc='left')
    ax.tick_params(labelsize=FS - 3)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
axes[0].set_ylabel('latitude', fontsize=FS)
# on the axes that OWNS Q -- coordinates='axes' resolves against Q's own axes
Q.axes.quiverkey(Q, 0.97, 0.06, round(S95, 3), '%.3f m s$^{-1}$' % S95,
                 labelpos='W', coordinates='axes',
                 fontproperties=dict(size=FS - 3))

sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=-VM, vmax=VM))
cb = fig.colorbar(sm, ax=axes, fraction=0.035, pad=0.02)
cb.set_label('depth-averaged along-cove $u$ [m s$^{-1}$]\nblue = into Penn Cove',
             fontsize=FS - 1)
cb.ax.tick_params(labelsize=FS - 3)

fn_out = out_dir / ('wind_mean_map_%s_%s.png' % (args.gtagex, args.mode))
fig.savefig(fn_out, dpi=400, bbox_inches='tight', transparent=True)
print('\nwrote ' + str(fn_out))
