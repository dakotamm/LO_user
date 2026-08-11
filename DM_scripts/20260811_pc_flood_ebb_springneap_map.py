"""
The average flood and the average ebb OVER PENN COVE, all hours and by spring
and neap.

The map counterpart of the two pc_lp section figures, and it writes both of
them from one read of the transports:

  flood_ebb_map_*.png             1 x 2, all hours, the pair to
                                  20260811_pc_lp_flood_ebb_section.py
  flood_ebb_springneap_map_*.png  2 x 2, rows spring and neap, the pair to
                                  20260811_pc_lp_springneap_section.py

Same bins, same sign convention, same colour-scale logic as those, but the
whole cove in plan view instead of one section. Each figure is scaled by its
OWN panels, exactly as in the section pair: the spring flood is faster than the
all-hours flood, and one scale across both figures would flatten neap.

Every panel is the depth-averaged velocity,

    ubar = <QU2> / (dy * (h_u + <zeta>))                           [m s-1]

shaded by its ALONG-COVE (westward) component so that BLUE is INTO Penn Cove
and RED is out of it, exactly as in the section figures, with the full vector
drawn on top. The sections say how the flood is arranged over depth at the
mouth; these say where in the cove it goes.

BINNING IS IDENTICAL TO THE SECTION FIGURES, and deliberately comes from the
same file rather than being redefined here: flood/ebb is the sign of the pc_lp
net transport, and spring/neap is the upper/lower quartile of the 30-day
anomaly of the Godin-filtered |Qnet| (binning on the raw envelope would put
most "spring" hours in whichever season has the biggest tides). Those series
are on the tef2 clock -- ocean_avg, hourly means stamped at :30 -- and the maps
are on the ocean_his clock, on the hour, so the two are interpolated onto the
his times rather than assumed to line up. The pickle's own mouth transport is
computed as well, and the fraction of hours the two definitions of "flood"
agree on is printed: that number is the check that this figure is binned the
same way as the sections.

TRANSPORT VELOCITY, NOT MEAN VELOCITY, and note what that means in a tidal bin:
each panel is the bin's mean transport over the bin's mean area, with the area
built on the mean zeta OF THAT BIN. Flood and ebb sit at different mean sea
level (printed), so using one record-mean area for all four would put a few
percent of the difference between panels into the geometry instead of the flow.

WHY THE PANELS ARE NOT THE RESIDUAL. These are raw hourly composites, so what
dominates them is the barotropic tide -- ~10x the residual circulation in
20260811_pc_mean_circulation.py, which is what is left after flood and ebb
cancel. Flood and ebb are near mirror images by construction; the interest is
in where they are NOT, and in how that changes between spring and neap.

Runs on the mac. Needs the hourly transports from
20260807_pc_turning_reduce.py and the local extractions_avg.
run 20260811_pc_flood_ebb_springneap_map.py
run 20260811_pc_flood_ebb_springneap_map.py -qq 0.15
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
p.add_argument('-sect', default='pc_lp', type=str,
               help='section whose net transport defines flood/ebb and the '
                    'spring/neap envelope')
p.add_argument('-0', '--ds0', default='2024.01.01', type=str)
p.add_argument('-1', '--ds1', default='2025.12.31', type=str)
p.add_argument('-f', '--fn', default='turning_his_wb1_t0_xn11abbur00_'
                                     '2024.01.01_2025.12.31.p',
               help='pickle from 20260807_pc_turning_reduce.py')
p.add_argument('-qq', default=0.25, type=float,
               help='quantile defining spring/neap; 0.25 = outer quartiles')
p.add_argument('--quiver-step', default=1, type=int, dest='quiver_step')
p.add_argument('--vmax', type=float, help='colour limit for all four panels [m/s]')
args = p.parse_args()

warnings.simplefilter('ignore')
Ldir = Lfun.Lstart(gridname=args.gctag.split('_')[0])
tef2 = Ldir['LOo'] / 'extract' / args.gtagex / 'tef2'
ex_fn = tef2 / ('extractions_avg_%s_%s' % (args.ds0, args.ds1)) / (args.sect + '.nc')
turn_fn = Ldir['LOo'] / 'DM_outs' / '20260807_pc_turning' / args.fn
out_dir = Ldir['LOo'] / 'DM_outs' / '20260811_pc_lp_flood_ebb'
Lfun.make_dir(out_dir)
for fn in [ex_fn, turn_fn]:
    if not fn.is_file():
        print('*** missing: %s' % fn)
        sys.exit(1)

FS = 14
BED = '#b9a894'          # land here, the seabed in the section figures
CMAP = cmc.vik_r         # blue = into the cove, as in the sections

# ---------------------------------------------------------------------------
# the bins, from the section -- identical definitions to the section figures
# ---------------------------------------------------------------------------
ds = xr.open_dataset(ex_fn)
ts = pd.to_datetime(ds.time.values)
qnet_s = pd.Series((-ds.q.values).sum(axis=(1, 2)), index=ts)   # + = INTO cove
ds.close()

env = pd.Series(zfun.lowpass(np.abs(qnet_s.values), f='godin'), index=ts)
anom_s = env - env.rolling(30 * 24, center=True, min_periods=200).mean()
hi, lo = anom_s.quantile(1 - args.qq), anom_s.quantile(args.qq)
print('%s: %d hourly steps, envelope mean %.0f m3/s (%.0f to %.0f)'
      % (args.sect, len(ts), env.mean(), env.min(), env.max()))
print('spring = 30-day anomaly >= %+.0f m3/s, neap <= %+.0f' % (hi, lo))

# ---------------------------------------------------------------------------
# the maps, and the two clocks
# ---------------------------------------------------------------------------
D = pd.read_pickle(turn_fn)
INFO = D['info']
th = pd.DatetimeIndex(D['time_h'])
h, cove, UM, VM_, lonr, latr = (D['h'], D['cove'], D['UM'], D['VM'],
                                D['lon_rho'], D['lat_rho'])
QU2h = np.where(UM[None], D['QU2_h'], np.nan)
QV2h = np.where(VM_[None], D['QV2_h'], np.nan)
zeta_h = D['zeta_h']
h_u = 0.5 * (h[:, :-1] + h[:, 1:])
h_v = 0.5 * (h[:-1, :] + h[1:, :])


def onto_his(s):
    """A tef2 series (hourly means stamped at :30) on the his clock (on the
    hour). Interpolated, not nearest-matched: half an hour is 15 degrees of M2
    phase, which is enough to put a slack-water hour in the wrong bin."""
    return s.reindex(s.index.union(th)).interpolate('time').reindex(th)


qn = onto_his(qnet_s)
an = onto_his(anom_s)
print('clocks: tef2 %s .. %s, his %s .. %s; %d his hours outside the tef2 span'
      % (ts[0], ts[-1], th[0], th[-1], int(qn.isna().sum())))

flood, ebb = (qn > 0).values, (qn < 0).values
spring, neap = (an >= hi).values, (an <= lo).values

# The same flood/ebb call made from the pickle's own mouth column. It is the
# same faces as the section, but ocean_his rather than ocean_avg, so this is
# the check that the two figure families really are binned alike.
im = int(list(D['iu_glob']).index(D['mouth_iu']))
q_mouth = -np.nansum(QU2h[:, :, im], axis=1)                    # + = INTO cove
agree = float(np.mean(np.sign(q_mouth) == np.sign(qn.values)))
print('flood/ebb from the pickle mouth column agrees with the section on '
      '%.1f%% of hours\n  (r = %+.3f; the disagreement is slack water, where '
      'the half-hour offset flips the sign)'
      % (100 * agree, np.corrcoef(q_mouth, np.nan_to_num(qn.values))[0, 1]))

# ---------------------------------------------------------------------------
# the four composites
# ---------------------------------------------------------------------------
AREA = np.where(cove, D['area'], np.nan)


def composite(m, label):
    """Bin-mean depth-averaged velocity over the cove for the hours in m."""
    if m.sum() == 0:
        print('*** no hours in %s' % label)
        sys.exit(1)
    # the bin's own mean sea level goes into the bin's face areas
    zb = float(np.nanmean(zeta_h[m]))
    ub = np.nanmean(QU2h[m], axis=0) / (D['DYU'] * (h_u + zb))
    vb = np.nanmean(QV2h[m], axis=0) / (D['DXV'] * (h_v + zb))
    u_rho = np.full(h.shape, np.nan)
    v_rho = np.full(h.shape, np.nan)
    # a dry face carries no flow, so it enters as zero rather than as a NaN
    # that would eat the whole cell
    u_rho[:, 1:-1] = 0.5 * (np.nan_to_num(ub[:, :-1]) + np.nan_to_num(ub[:, 1:]))
    v_rho[1:-1, :] = 0.5 * (np.nan_to_num(vb[:-1, :]) + np.nan_to_num(vb[1:, :]))
    u_rho[~cove], v_rho[~cove] = np.nan, np.nan
    u_in = -u_rho                           # + = westward = INTO the cove
    f_in = 100 * np.nansum(AREA * (u_in > 0)) / np.nansum(AREA)
    print('%-18s %8d %12.0f %10.3f %10.3f %8.0f%%'
          % (label, m.sum(), qn[m].mean(), zb,
             np.nanmax(np.hypot(u_rho, v_rho)), f_in))
    return dict(u_in=u_in, u=u_rho, v=v_rho, m=m, zb=zb)


print('\n%-18s %8s %12s %10s %10s %9s'
      % ('bin', 'n [h]', 'mean Qnet', 'mean zeta', 'max |u|', 'area in'))
ALL = {k: composite(mk, k) for k, mk in [('flood', flood), ('ebb', ebb)]}
SN = {(tide, k): composite(mt & mk, '%s %s' % (tide, k))
      for tide, mt in [('spring', spring), ('neap', neap)]
      for k, mk in [('flood', flood), ('ebb', ebb)]}

# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
DAR = 1 / np.cos(np.deg2rad(float(np.mean(latr))))
dlon = float(np.diff(lonr[0, :]).mean())
dlat = float(np.diff(latr[:, 0]).mean())
land = ~D['mask_rho']
SL = pd.read_pickle(Ldir['LOo'] / 'section_lines' / (args.sect + '.p'))
qs = max(1, args.quiver_step)


def draw(ax, C, vm, qscale):
    ax.pcolormesh(lonr, latr, np.ma.masked_where(~land, np.ones(land.shape)),
                  cmap=ListedColormap([BED]), shading='nearest', zorder=0)
    ax.pcolormesh(lonr, latr, C['u_in'], cmap=CMAP, vmin=-vm, vmax=vm,
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
    ax.tick_params(labelsize=FS - 3)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    return Q


def scales(cells):
    """Colour limit and quiver scale from THIS figure's panels only -- the
    all-hours flood is slower than the spring flood, and one scale across both
    figures would flatten neap."""
    vm = (args.vmax if args.vmax else
          np.ceil(max(np.nanmax(np.abs(C['u_in'])) for C in cells) * 50) / 50.)
    s95 = float(np.nanpercentile(
        np.concatenate([np.hypot(C['u'], C['v'])[cove] for C in cells]), 95))
    return vm, s95, s95 * lonr.shape[1] / (1.2 * qs)


def finish(fig, axes, vm, Q, s95, fn):
    # on the axes that OWNS Q -- coordinates='axes' resolves against Q's own
    # axes, so keying it to any other panel puts the arrow outside that panel
    # and it is clipped away
    Q.axes.quiverkey(Q, 0.97, 0.06, round(s95, 2), '%.2f m s$^{-1}$' % s95,
                     labelpos='W', coordinates='axes',
                     fontproperties=dict(size=FS - 3))
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=-vm, vmax=vm))
    cb = fig.colorbar(sm, ax=axes, fraction=0.035, pad=0.02)
    # two lines, not one: rotated vertically, the one-line version is longer
    # than the colourbar on the 1 x 2 figure and gets clipped at both ends
    cb.set_label('depth-averaged along-cove $u$ [m s$^{-1}$]\n'
                 'blue = into Penn Cove', fontsize=FS - 1)
    cb.ax.tick_params(labelsize=FS - 3)
    fig.savefig(fn, dpi=400, bbox_inches='tight', transparent=True)
    print('wrote ' + str(fn))


def title(C, lbl):
    return ('%s  (n = %d h, mean $Q_{net}$ = %+.0f m$^3$ s$^{-1}$)'
            % (lbl, C['m'].sum(), qn[C['m']].mean()))


# --- all hours, 1 x 2
vm, s95, qsc = scales(list(ALL.values()))
print('\nall hours:   colour +/- %.2f m/s, quiver key %.3f m/s' % (vm, s95))
fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), sharex=True, sharey=True)
for ax, k in zip(axes, ['flood', 'ebb']):
    Q = draw(ax, ALL[k], vm, qsc)
    ax.set_title(title(ALL[k], k), fontsize=FS, loc='left')
    ax.set_xlabel('longitude', fontsize=FS)
axes[0].set_ylabel('latitude', fontsize=FS)
finish(fig, axes, vm, Q, s95,
       out_dir / ('flood_ebb_map_%s.png' % args.gtagex))

# --- spring / neap, 2 x 2
vm, s95, qsc = scales(list(SN.values()))
print('spring/neap: colour +/- %.2f m/s, quiver key %.3f m/s' % (vm, s95))
fig, axes = plt.subplots(2, 2, figsize=(15, 8.6), sharex=True, sharey=True)
for i, tide in enumerate(['spring', 'neap']):
    for j, k in enumerate(['flood', 'ebb']):
        C = SN[(tide, k)]
        Q = draw(axes[i, j], C, vm, qsc)
        axes[i, j].set_title(title(C, '%s %s' % (tide, k)), fontsize=FS,
                             loc='left')
    axes[i, 0].set_ylabel('latitude', fontsize=FS)
for j in range(2):
    axes[1, j].set_xlabel('longitude', fontsize=FS)
finish(fig, axes, vm, Q, s95,
       out_dir / ('flood_ebb_springneap_map_%s.png' % args.gtagex))
