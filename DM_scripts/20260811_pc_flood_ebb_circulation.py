"""
The average flood and the average ebb, map and mouth, in one figure.

Two rows, the same pair of panels in each, laid out exactly like
20260811_pc_mean_circulation.py so the tidal composites can be read against
the residual without re-learning the figure:

  row 1  FLOOD      left  MAP of the bin-mean depth-averaged velocity over the
  row 2  EBB              whole cove, shaded by the eastward component (red =
                          out of the cove) with the vectors on top.
                    right SECTION through pc_lp: the bin-mean velocity normal
                          to the mouth, face by face and level by level, same
                          sign convention, so a red column is water leaving.

This supersedes the 1 x 2 all-hours figures from
20260811_pc_lp_flood_ebb_section.py and the first figure of
20260811_pc_flood_ebb_springneap_map.py -- same bins, same numbers, one figure
instead of two that had to be held side by side. The 2 x 2 spring/neap map in
that script is untouched.

SIGN, ONCE. Positive q at pc_lp runs minus-side -> plus-side = eastward = OUT
of the cove, and the map's eastward component is out of the cove too, so
nothing is negated anywhere: red is water leaving in all four panels. Flood is
the set of hours with net transport INTO the cove, i.e. qnet < 0 in this sign.

THE BINS ARE THE SECTION'S. Flood/ebb is the sign of the pc_lp net transport,
and the map is binned on that same series rather than on its own mouth column,
so the two columns of a row are the same hours. The section is on the tef2
clock (ocean_avg, hourly means stamped at :30) and the map on the ocean_his
clock (on the hour), so the series is interpolated onto the his times -- half
an hour is 15 degrees of M2 phase, enough to put a slack hour in the wrong bin.
The pickle's own mouth-column call is made as well and the agreement printed;
that is the check that both columns really are binned alike.

TRANSPORT VELOCITY, NOT MEAN VELOCITY, in both panels and in each bin
separately,

    map      ubar = <QU2>_bin / (dy * (h_u + <zeta>_bin))
    section  u    = <q>_bin   / (dd * <DZ>_bin)

so each panel is its bin's mean transport over its bin's mean area. Flood and
ebb sit at different mean sea level (printed), and using one record-mean area
for both would put a few percent of the difference between the rows into the
geometry instead of the flow. It is also why the section's surface in row 1
sits a little higher than in row 2: that is the tide, drawn.

ONE COLOUR SCALE ACROSS ALL FOUR PANELS, and one colourbar, set by the
sections, which are the larger of the two fields. Flood and ebb are near mirror
images by construction -- the interest is in where they are NOT, and a shared
scale is what makes that visible. The map comes out paler than the section
because depth-averaging cancels part of the exchange within each column; both
saturation fractions are printed, and the arrows carry the map's own scale
through the quiver key.

WHAT THESE ARE NOT. Raw hourly composites, so what dominates them is the
barotropic tide, ~10x the residual circulation in
20260811_pc_mean_circulation.py, which is what is left after these two cancel.
And every flooding hour counts the same as every other, so this is "what does a
flood look like on average", not "what does the strongest flood look like".

Runs on the mac -- both inputs are local.
run 20260811_pc_flood_ebb_circulation.py
run 20260811_pc_flood_ebb_circulation.py -sect pc_cp
run 20260811_pc_flood_ebb_circulation.py --vmax 0.3
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
from cmocean import cm

from lo_tools import Lfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
p.add_argument('-gctag', default='wb1_pc1', type=str)
p.add_argument('-sect', default='pc_lp', type=str)
p.add_argument('-0', '--ds0', default='2024.01.01', type=str)
p.add_argument('-1', '--ds1', default='2025.12.31', type=str)
p.add_argument('-f', '--fn', default='turning_his_wb1_t0_xn11abbur00_'
                                     '2024.01.01_2025.12.31.p',
               help='pickle from 20260807_pc_turning_reduce.py -- the maps')
p.add_argument('--quiver-step', default=1, type=int, dest='quiver_step')
p.add_argument('--quiver-scale', type=float, dest='quiver_scale')
p.add_argument('--vmax', type=float,
               help='colour limit for ALL FOUR panels [m/s]; default the 99th '
                    'pctl of the sections, the larger of the two fields')
args = p.parse_args()
warnings.simplefilter('ignore', RuntimeWarning)     # all-NaN rows over land

Ldir = Lfun.Lstart(gridname=args.gctag.split('_')[0])
tef2 = Ldir['LOo'] / 'extract' / args.gtagex / 'tef2'
sect_fn = (tef2 / ('extractions_avg_%s_%s' % (args.ds0, args.ds1))
           / (args.sect + '.nc'))
turn_fn = Ldir['LOo'] / 'DM_outs' / '20260807_pc_turning' / args.fn
out_dir = Ldir['LOo'] / 'DM_outs' / '20260811_pc_lp_flood_ebb'
Lfun.make_dir(out_dir)
for fn in [sect_fn, turn_fn]:
    if not fn.is_file():
        print('*** missing: %s' % fn)
        sys.exit(1)

RED = '#e04256'
LAND = '0.85'
FS = 13
BINS = ['flood', 'ebb']

# ---------------------------------------------------------------------------
# a. the section, and the bins it defines
# ---------------------------------------------------------------------------
ds = xr.open_dataset(sect_fn)
ts = pd.DatetimeIndex(ds.time.values)
dd, hh = ds.dd.values, ds.h.values
q = ds.q.values                                   # (nt, z, p)  m3/s, + = OUT
DZ = ds.DZ.values                                 # (nt, z, p)  m
ds.close()
qnet_s = pd.Series(q.sum(axis=(1, 2)), index=ts)  # + = out of the cove
M_sect = {'flood': (qnet_s < 0).values, 'ebb': (qnet_s > 0).values}

print('%s\nsection %s: %d faces, %d levels, %d hourly steps, %s to %s'
      % (args.gtagex, args.sect, len(dd), q.shape[1], len(ts),
         ts[0].date(), ts[-1].date()))
print('  %d flood (%.0f%%), %d ebb; mean Qnet flood %+.0f, ebb %+.0f m3/s'
      % (M_sect['flood'].sum(), 100 * M_sect['flood'].mean(),
         M_sect['ebb'].sum(), qnet_s[M_sect['flood']].mean(),
         qnet_s[M_sect['ebb']].mean()))

SEC = {}
for k in BINS:
    m = M_sect[k]
    q_m = np.nanmean(q[m], axis=0)                            # (z, p)
    DZ_m = np.nanmean(DZ[m], axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        u_sect = q_m / (dd[None, :] * DZ_m)                   # m/s, + = out
    # z from the bed by integrating the same DZ the velocities were built on,
    # so the mesh lands exactly on h and the surface lands on the bin's zeta
    zw = np.concatenate([np.zeros_like(DZ_m[:1]), np.cumsum(DZ_m, axis=0)]) \
         - hh[None, :]
    A = dd[None, :] * DZ_m
    SEC[k] = dict(u=u_sect, q=q_m, zw=zw, qf=np.nansum(q_m, axis=0),
                  f_in=100 * np.nansum(A * (u_sect < 0)) / np.nansum(A),
                  zbar=float(np.nanmean(zw[-1])))

SL = pd.read_pickle(Ldir['LOo'] / 'section_lines' / (args.sect + '.p'))
SL = SL.assign(x=SL.x.astype(float), y=SL.y.astype(float))
print('  p runs from %.4f to %.4f degN, so p=0 is the %s end'
      % (SL.y.iloc[0], SL.y.iloc[-1],
         'NORTH' if SL.y.iloc[0] > SL.y.iloc[-1] else 'SOUTH'))
for k in BINS:
    S = SEC[k]
    print('  %-5s u %+.3f to %+.3f m/s; %.0f%% of the section area flows IN; '
          '<zeta> %+.3f m' % (k, np.nanmin(S['u']), np.nanmax(S['u']),
                              S['f_in'], S['zbar']))
    print('        per-face net transport (m3/s): '
          + '  '.join('%+.0f' % v for v in S['qf']))

# ---------------------------------------------------------------------------
# b. the maps, binned on the same series across the two clocks
# ---------------------------------------------------------------------------
D = pd.read_pickle(turn_fn)
if D['info'].get('stokes_missing'):
    print('** this pickle came from lowpassed.nc: the transports are missing '
          'the tidal Stokes term, so these are NOT transport velocities **')
th = pd.DatetimeIndex(D['time_h'])
h, cove, UM, VM_ = D['h'], D['cove'], D['UM'], D['VM']
lonr, latr = D['lon_rho'], D['lat_rho']
QU2h = np.where(UM[None], D['QU2_h'], np.nan)
QV2h = np.where(VM_[None], D['QV2_h'], np.nan)
zeta_h = D['zeta_h']
h_u = 0.5 * (h[:, :-1] + h[:, 1:])
h_v = 0.5 * (h[:-1, :] + h[1:, :])
AREA = np.where(cove, D['area'], np.nan)

# the tef2 series (hourly means stamped at :30) onto the his clock (on the
# hour). Interpolated, not nearest-matched: half an hour is 15 degrees of M2
# phase, which is enough to put a slack-water hour in the wrong bin.
qn = qnet_s.reindex(qnet_s.index.union(th)).interpolate('time').reindex(th)
M_map = {'flood': (qn < 0).values, 'ebb': (qn > 0).values}
print('\nmap: %d hourly fields, %s to %s; %d his hours outside the tef2 span'
      % (len(th), th[0].date(), th[-1].date(), int(qn.isna().sum())))

# the same flood/ebb call made from the pickle's own mouth column -- the same
# faces as the section, but ocean_his rather than ocean_avg
im = int(list(D['iu_glob']).index(D['mouth_iu']))
q_mouth = np.nansum(QU2h[:, :, im], axis=1)                     # + = out
print('  flood/ebb from the pickle mouth column agrees with the section on '
      '%.1f%% of hours\n  (r = %+.3f; the disagreement is slack water, where '
      'the half-hour offset flips the sign)'
      % (100 * np.mean(np.sign(q_mouth) == np.sign(qn.values)),
         np.corrcoef(q_mouth, np.nan_to_num(qn.values))[0, 1]))

MAP = {}
print('\n%-6s %8s %12s %11s %10s %9s'
      % ('bin', 'n [h]', 'mean Qnet', 'mean zeta', 'max |u|', 'area in'))
for k in BINS:
    m = M_map[k]
    zb = float(np.nanmean(zeta_h[m]))       # the bin's own mean sea level
    ub = np.nanmean(QU2h[m], axis=0) / (D['DYU'] * (h_u + zb))
    vb = np.nanmean(QV2h[m], axis=0) / (D['DXV'] * (h_v + zb))
    # faces onto the rho points they straddle. A dry face carries no flow, so
    # it enters as zero rather than as a NaN that would eat the whole cell.
    u_rho = np.full(h.shape, np.nan)
    v_rho = np.full(h.shape, np.nan)
    u_rho[:, 1:-1] = 0.5 * (np.nan_to_num(ub[:, :-1]) + np.nan_to_num(ub[:, 1:]))
    v_rho[1:-1, :] = 0.5 * (np.nan_to_num(vb[:-1, :]) + np.nan_to_num(vb[1:, :]))
    u_rho[~cove], v_rho[~cove] = np.nan, np.nan
    spd = np.hypot(u_rho, v_rho)
    MAP[k] = dict(u=u_rho, v=v_rho, spd=spd, zb=zb, n=int(m.sum()))
    print('%-6s %8d %12.0f %11.3f %10.3f %8.0f%%'
          % (k, m.sum(), qn[m].mean(), zb, np.nanmax(spd),
             100 * np.nansum(AREA * (u_rho < 0)) / np.nansum(AREA)))

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
plt.close('all')
# The map carries a fixed aspect (dar), so its panel has to be roughly the
# cove's own shape or it shrinks inside its gridspec cell and leaves a band of
# white on either side. The section is given the same height so each row reads
# as a pair.
DAR = 1 / np.cos(np.deg2rad(float(np.mean(latr))))
fig = plt.figure(figsize=(16, 9.2), layout='constrained')
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])
AXM = [fig.add_subplot(gs[i, 0]) for i in range(2)]
AXV = [fig.add_subplot(gs[i, 1]) for i in range(2)]

# One symmetric scale for all four panels, set by the sections -- the larger of
# the two fields, so nothing there is clipped and the maps are left honestly
# paler.
# rounded up to a tidy 0.01 rather than taken as a percentile: the ebb section
# is the fastest thing in the figure by some margin and a percentile limit
# clips exactly the cells the ebb panel exists to show
VMAX = (args.vmax if args.vmax else
        float(np.ceil(max(np.nanmax(np.abs(SEC[k]['u'])) for k in BINS) * 100)
              / 100.))
KW = dict(cmap=cm.balance, vmin=-VMAX, vmax=VMAX)
print('\nshared colour scale +/- %.4f m/s' % VMAX)
for k in BINS:
    for lbl, a in [('map', MAP[k]['u']), ('section', SEC[k]['u'])]:
        v = a[np.isfinite(a)]
        print('  %-5s %-7s |u| reaches %.1f%% of the scale; %.1f%% of cells '
              'saturate' % (k, lbl, 100 * np.max(np.abs(v)) / VMAX,
                            100 * np.mean(np.abs(v) > VMAX)))

# one quiver scale for both rows, so the flood and ebb arrows are comparable
qs = max(1, args.quiver_step)
spd95 = float(np.nanpercentile(
    np.concatenate([MAP[k]['spd'][cove] for k in BINS]), 95))
qscale = (args.quiver_scale if args.quiver_scale
          else spd95 * lonr.shape[1] / (1.2 * qs))

# half a cell of margin: shading='nearest' centres each cell on its rho point,
# so limits set to the rho points themselves cut the outer cells in half
land = ~D['mask_rho']
dlon = float(np.diff(lonr[0, :]).mean())
dlat = float(np.diff(latr[:, 0]).mean())

x_e = np.concatenate([[0.0], np.cumsum(dd)]) / 1000.0          # km across
xc = 0.5 * (x_e[:-1] + x_e[1:])
# one y range for both rows: down to the deepest face, up to the higher of the
# two mean sea levels, so the flood really does sit above the ebb rather than
# each row being renormalised to its own surface
zbot = float(-hh.max()) * 1.04
ztop = max([0.0] + [float(np.nanmax(SEC[kk]['zw'][-1])) for kk in BINS])

for i, k in enumerate(BINS):
    # --- map
    axm = AXM[i]
    axm.pcolormesh(lonr, latr, np.ma.masked_where(~land, np.ones(land.shape)),
                   cmap=ListedColormap([LAND]), shading='nearest', zorder=0)
    pcm = axm.pcolormesh(lonr, latr, MAP[k]['u'], shading='nearest', zorder=1,
                         rasterized=True, **KW)
    Q = axm.quiver(lonr[::qs, ::qs], latr[::qs, ::qs],
                   MAP[k]['u'][::qs, ::qs], MAP[k]['v'][::qs, ::qs],
                   scale=qscale, scale_units='width', units='width',
                   width=0.0026, color='k', zorder=6)
    axm.plot(SL.x, SL.y, '-', color=RED, lw=3.5, zorder=9,
             solid_capstyle='butt')
    axm.set_xlim(lonr.min() - dlon / 2, lonr.max() + dlon / 2)
    axm.set_ylim(latr.min() - dlat / 2, latr.max() + dlat / 2)
    axm.set_aspect(DAR)
    axm.set_ylabel('latitude', fontsize=FS)
    axm.tick_params(labelsize=FS - 2)
    # the only text on the panels: which row this is. Everything else -- the
    # hour counts, the transports, the sea levels -- goes to stdout.
    axm.set_title(k, fontsize=FS + 1, loc='left')

    # --- section
    axv = AXV[i]
    zw = SEC[k]['zw']
    # One quadmesh PER FACE. A single mesh spanning all faces interpolates the
    # sigma levels across columns of different depth and smears the water
    # column past the stepped bathymetry -- the shading ends up below the bed.
    for ip in range(len(dd)):
        Xp = np.tile([x_e[ip], x_e[ip + 1]], (zw.shape[0], 1))
        Yp = np.repeat(zw[:, ip:ip + 1], 2, axis=1)
        axv.pcolormesh(Xp, Yp, SEC[k]['u'][:, ip:ip + 1], shading='flat',
                       rasterized=True, **KW)
    # The zero line, on face centres -- the boundary between water leaving and
    # water arriving. It is interpolated ACROSS columns of different depth, so
    # near the bed it can wander below the shallower of the two; the seabed is
    # drawn on top of it (zorder) so it is cut off at the bathymetry.
    zc = 0.5 * (zw[:-1] + zw[1:])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        axv.contour(np.tile(xc, (zc.shape[0], 1)), zc, SEC[k]['u'],
                    levels=[0.0], colors='k', linewidths=1.2, zorder=3)
    bed = -np.concatenate([hh[:1], hh])
    axv.fill_between(x_e, bed, zbot, step='pre', color=LAND, lw=0, zorder=4)
    axv.plot(x_e, bed, '-', color='0.3', lw=1.0, drawstyle='steps-pre',
             zorder=5)
    axv.set_ylim(zbot, ztop)
    axv.set_xlim(x_e[0], x_e[-1])
    axv.set_ylabel('z [m]', fontsize=FS)
    axv.tick_params(labelsize=FS - 2)
    # framed in the same red as the line on the map, so the section and its
    # location read as one object
    for sp in axv.spines.values():
        sp.set_color(RED)
        sp.set_linewidth(2.0)
    axv.tick_params(color=RED)

# x labels on the bottom row only -- the columns share their axes by
# construction, and repeating the labels just crowds the middle of the figure
for ax in AXM[:-1] + AXV[:-1]:
    ax.tick_params(labelbottom=False)
AXM[-1].set_xlabel('longitude', fontsize=FS)
AXV[-1].set_xlabel('distance across %s [km]  (north on the left)' % args.sect,
                   fontsize=FS)
# the quiver key on the axes that OWNS Q -- coordinates='axes' resolves against
# Q's own axes, so keying it to another panel puts the arrow outside it
Q.axes.quiverkey(Q, 0.90, 0.05, round(spd95, 3), '%.3f m s$^{-1}$' % spd95,
                 labelpos='W', coordinates='axes',
                 fontproperties=dict(size=FS - 3))

cb = fig.colorbar(pcm, ax=AXM + AXV, pad=0.015, aspect=40, fraction=0.05)
cb.ax.tick_params(labelsize=FS - 2)
cb.set_label('bin-mean velocity [m s$^{-1}$]  (red = out of the cove)',
             fontsize=FS)

stem = 'flood_ebb_circulation_%s_%s' % (args.gtagex, args.sect)
fn_out = out_dir / (stem + '.png')
fig.savefig(fn_out, dpi=300, bbox_inches='tight', transparent=True)
print('\nwrote %s' % fn_out)

# the numbers behind the panels, so they can be replotted without the pickle
pd.DataFrame(dict(p=np.arange(len(dd)), h=hh, dd=dd,
                  **{'q_net_' + k: SEC[k]['qf'] for k in BINS},
                  **{'u_depth_avg_' + k: SEC[k]['qf']
                     / (dd * (hh + SEC[k]['zbar'])) for k in BINS})
             ).to_csv(out_dir / (stem + '_section_faces.csv'), index=False,
                      float_format='%.5f')
np.savez(out_dir / (stem + '_fields.npz'), lon_rho=lonr, lat_rho=latr,
         cove=cove, h=h, dd=dd, h_sect=hh,
         **{'u_rho_' + k: MAP[k]['u'] for k in BINS},
         **{'v_rho_' + k: MAP[k]['v'] for k in BINS},
         **{'u_sect_' + k: SEC[k]['u'] for k in BINS},
         **{'q_sect_' + k: SEC[k]['q'] for k in BINS},
         **{'zw_' + k: SEC[k]['zw'] for k in BINS})
print('wrote %s and the section face table' % (stem + '_fields.npz'))
plt.close('all')
