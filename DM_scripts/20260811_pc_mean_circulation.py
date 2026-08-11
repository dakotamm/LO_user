"""
The Penn Cove RESIDUAL circulation, side by side: the map and the mouth.

  left   MAP of the record-mean depth-averaged velocity over the whole cove.
         Shaded by the eastward component (red = out of the cove) with the
         vectors on top, so the lateral split -- water in along the north
         shore, out along the south -- reads off the colour and is confirmed
         by the arrows.

  right  SECTION through pc_lp: the record-mean velocity normal to the mouth,
         face by face and level by level. Same sign convention as the map, red
         = out of the cove, so a red column here is water leaving and a blue
         column is water arriving. This is where the map's lateral split is
         resolved against the vertical one -- if the cove ran a classical
         estuarine exchange the section would be blue at depth and red at the
         surface across its whole width instead.

No titles: the panels are the figure, and the run, the window and the
transports all go to stdout, where they can be pasted into a caption.

WHAT "MEAN VELOCITY" MEANS HERE, AND WHY IT IS NOT A MEAN OF u. Both panels are
TRANSPORT velocities: the mean transport divided by the mean area,

    map      ubar = <QU2> / (dy * (h_u + <zeta>))
    section  u    = <q>   / (dd * <DZ>)

not the time mean of an instantaneous velocity. The two differ by the Stokes
term -- the correlation between the tidal velocity and the tidal thickness --
which in Penn Cove is not small: a ~2 m range on a ~16 m column, with a tidal
velocity ~3x the residual. Averaging velocity would throw that away and the
panels would no longer add up to the transport that actually moves water in and
out of the cove. It also means the two panels are directly comparable to each
other and to the tef2 transports, which are built the same way.

FILTERING. Both sources are hourly and are Godin-filtered over the FULL record
and only then sampled daily at 12:00 and windowed, so a partial window is not
biased by the filter's own half-width. Over a multi-year mean the filter barely
matters, but doing it identically on both panels means any difference between
them is physics rather than processing. The map's daily fields arrive already
filtered from 20260807_pc_turning_reduce.py; the section is filtered here.

SOURCES, AND THE ONE MISMATCH BETWEEN THEM. The map comes from ocean_his
(instantaneous, hourly) via the turning pickle; the section comes from the
tef2 extraction, which is built on ocean_avg (hourly MEANS). They are the same
water and the same faces -- the mouth column of the cove box IS pc_lp -- but
not the same sampling, so the section transport and the mouth-column transport
agree to a few percent rather than exactly. Both are printed as a check.

ONE COLOUR SCALE ACROSS BOTH PANELS, and one colourbar. They are the same
quantity in the same units with the same sign, so a shared scale is what makes
them comparable, and the limit is set by the section, which is the larger of
the two. The consequence is worth stating rather than tuning away: the map
comes out pale, because depth-averaging cancels most of the exchange within
each column -- the cove moves ~5x more water than its depth-averaged residual
suggests. That contrast IS the result, so both saturation fractions are printed
and the arrows carry the map's own scale through the quiver key.

Runs on the mac -- both inputs are local.
run 20260811_pc_mean_circulation.py
run 20260811_pc_mean_circulation.py -0 2025.06.01 -1 2025.09.30
run 20260811_pc_mean_circulation.py --vmax 0.05
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
from scipy.ndimage import convolve1d
from cmocean import cm

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
p.add_argument('-f', '--fn', default='turning_his_wb1_t0_xn11abbur00_'
                                     '2024.01.01_2025.12.31.p',
               help='pickle from 20260807_pc_turning_reduce.py -- the map')
p.add_argument('-ex', '--extract', default='extractions_avg_2024.01.01_2025.12.31',
               help='tef2 extraction directory holding the section')
p.add_argument('-sect', default='pc_lp', type=str)
p.add_argument('-0', '--ds0', default='', type=str,
               help='YYYY.MM.DD; blank = the whole record')
p.add_argument('-1', '--ds1', default='', type=str)
p.add_argument('--quiver-step', default=1, type=int, dest='quiver_step')
p.add_argument('--quiver-scale', type=float, dest='quiver_scale')
p.add_argument('--vmax', type=float,
               help='colour limit for BOTH panels [m/s]; default the 99th pctl '
                    'of the section, which is the larger of the two fields')
args = p.parse_args()
warnings.simplefilter('ignore', RuntimeWarning)     # all-NaN rows over land

Ldir = Lfun.Lstart(gridname='wb1')
turn_fn = Ldir['LOo'] / 'DM_outs' / '20260807_pc_turning' / args.fn
sect_fn = (Ldir['LOo'] / 'extract' / args.gtagex / 'tef2' / args.extract
           / (args.sect + '.nc'))
out_dir = Ldir['LOo'] / 'DM_outs' / '20260811_pc_mean_circulation'
Lfun.make_dir(out_dir)
for fn in [turn_fn, sect_fn]:
    if not fn.is_file():
        print('*** missing: %s' % fn)
        sys.exit(1)

RED = '#e04256'
LAND = '0.85'
FS = 13
GODIN = zfun.godin_shape()
NPAD = len(GODIN) // 2

# ---------------------------------------------------------------------------
# a. the map: mean depth-averaged velocity over the cove
# ---------------------------------------------------------------------------
D = pd.read_pickle(turn_fn)
INFO = D['info']
if INFO.get('stokes_missing'):
    print('** this pickle came from lowpassed.nc: the transports are missing '
          'the tidal Stokes term, so these are NOT residual transport '
          'velocities **')

t_map = pd.DatetimeIndex(D['time'])
sel = np.ones(len(t_map), bool)
if args.ds0:
    sel &= t_map >= pd.Timestamp(args.ds0.replace('.', '-'))
if args.ds1:
    sel &= t_map <= pd.Timestamp(args.ds1.replace('.', '-')) + pd.Timedelta('1D')
if not sel.any():
    print('*** no daily samples between %s and %s' % (args.ds0, args.ds1))
    sys.exit(1)
t0, t1 = t_map[sel][0], t_map[sel][-1]
print('%s\nmap: %d of %d daily subtidal samples, %s to %s'
      % (INFO['gtx'], sel.sum(), len(t_map), t0.date(), t1.date()))

h, cove, UM, VM = D['h'], D['cove'], D['UM'], D['VM']
lonr, latr = D['lon_rho'], D['lat_rho']
# <zeta> over the record, not zero: the cove sits at a nonzero mean surface in
# this run, and it is ~2% of the depth, so it belongs in the face area.
zbar = float(D['T'].zeta[sel].mean())
h_u = 0.5 * (h[:, :-1] + h[:, 1:])
h_v = 0.5 * (h[:-1, :] + h[1:, :])
A_U = np.where(UM, D['DYU'] * (h_u + zbar), np.nan)
A_V = np.where(VM, D['DXV'] * (h_v + zbar), np.nan)

# <Q> / <A>: the residual transport velocity on each face
ubar = np.nanmean(np.where(UM[None], D['QU2'][sel], np.nan), axis=0) / A_U
vbar = np.nanmean(np.where(VM[None], D['QV2'][sel], np.nan), axis=0) / A_V

# faces onto the rho points they straddle. A dry face carries no flow, so it
# enters the average as zero rather than as a NaN that would eat the cell.
u_rho = np.full(h.shape, np.nan)
v_rho = np.full(h.shape, np.nan)
u_rho[:, 1:-1] = 0.5 * (np.nan_to_num(ubar[:, :-1]) + np.nan_to_num(ubar[:, 1:]))
v_rho[1:-1, :] = 0.5 * (np.nan_to_num(vbar[:-1, :]) + np.nan_to_num(vbar[1:, :]))
u_rho[~cove], v_rho[~cove] = np.nan, np.nan
spd = np.hypot(u_rho, v_rho)
print('  mean depth-averaged speed: median %.4f, 95th pctl %.4f, max %.4f m/s'
      % (np.nanmedian(spd), np.nanpercentile(spd, 95), np.nanmax(spd)))
print('  <zeta> = %+.3f m over the window, used in the face areas' % zbar)

# Sanity: the same mean taken from the raw hourly transports instead of the
# daily filtered ones. A record mean should barely care, and if it does the
# window is short enough that the tide has not averaged out of it.
QU2h = np.where(UM[None], D['QU2_h'], np.nan)
th = pd.DatetimeIndex(D['time_h'])
selh = (th >= t0 - pd.Timedelta('12h')) & (th <= t1 + pd.Timedelta('12h'))
u_h = np.nanmean(QU2h[selh], axis=0) / A_U
print('  filtered-daily vs raw-hourly mean: max |diff| %.5f m/s (%.1f%% of the '
      '95th pctl speed)'
      % (np.nanmax(np.abs(ubar - u_h)),
         100 * np.nanmax(np.abs(ubar - u_h)) / np.nanpercentile(spd, 95)))

# the mouth column of the cove box, for the cross-check against the section
im = int(list(D['iu_glob']).index(D['mouth_iu']))
Q_mouth = float(np.nansum(np.nanmean(np.where(UM[None], D['QU2'][sel], np.nan),
                                     axis=0)[:, im]))

# ---------------------------------------------------------------------------
# b. the section: mean normal velocity through pc_lp
# ---------------------------------------------------------------------------
ds = xr.open_dataset(sect_fn)
ts = pd.DatetimeIndex(ds.time.values)
dd, hh = ds.dd.values, ds.h.values
q = ds.q.values                                   # (nt, z, p)  m3/s, + = east
DZ = ds.DZ.values                                 # (nt, z, p)  m
ds.close()
print('\nsection %s: %d faces, %d levels, %d hourly steps'
      % (args.sect, len(dd), q.shape[1], len(ts)))


def godin_daily(a):
    """Godin along time over the FULL record, then the 12:00 sample of each day.

    Filtering first and windowing after is the point: window first and the
    filter blanks NPAD hours off each end of the window itself.
    """
    out = convolve1d(np.nan_to_num(a), GODIN, axis=0, mode='nearest')
    out[:NPAD] = np.nan
    out[-NPAD:] = np.nan
    return out


kd = np.where(ts.hour == 12)[0]
kd = kd[(kd >= NPAD) & (kd < len(ts) - NPAD)]
td = ts[kd]
ks = np.ones(len(kd), bool)
if args.ds0:
    ks &= td >= pd.Timestamp(args.ds0.replace('.', '-'))
if args.ds1:
    ks &= td <= pd.Timestamp(args.ds1.replace('.', '-')) + pd.Timedelta('1D')
if not ks.any():
    print('*** the section has no daily samples in the window')
    sys.exit(1)
print('  %d of %d daily subtidal samples, %s to %s'
      % (ks.sum(), len(kd), td[ks][0].date(), td[ks][-1].date()))

q_m = np.nanmean(godin_daily(q)[kd][ks], axis=0)          # (z, p)
DZ_m = np.nanmean(godin_daily(DZ)[kd][ks], axis=0)
with np.errstate(invalid='ignore', divide='ignore'):
    u_sect = q_m / (dd[None, :] * DZ_m)                   # m/s, + = out of cove

# z from the bed by integrating the same DZ the velocities were built on, so
# the mesh lands exactly on h
zw = np.concatenate([np.zeros_like(DZ_m[:1]), np.cumsum(DZ_m, axis=0)]) \
     - hh[None, :]

Q_sect = float(np.nansum(q_m))
Q_in = float(np.nansum(q_m[q_m < 0]))
Q_out = float(np.nansum(q_m[q_m > 0]))
print('  mean transport %+.1f m3/s net = %.1f in + %+.1f out; exchange %.1f m3/s'
      % (Q_sect, Q_in, Q_out, 0.5 * (abs(Q_in) + Q_out)))
print('  the cove-box mouth column gives %+.1f m3/s net -- his vs avg output, '
      'so a few\n  percent apart is expected, a sign flip is not' % Q_mouth)
print('  velocity: %.4f to %.4f m/s, |mean| %.4f'
      % (np.nanmin(u_sect), np.nanmax(u_sect), np.nanmean(np.abs(u_sect))))

# where the section changes sign, face by face -- the lateral split at the mouth
qf = np.nansum(q_m, axis=0)
SL = pd.read_pickle(Ldir['LOo'] / 'section_lines' / (args.sect + '.p'))
SL = SL.assign(x=SL.x.astype(float), y=SL.y.astype(float))
print('  p runs from %.4f to %.4f degN, so p=0 is the %s end'
      % (SL.y.iloc[0], SL.y.iloc[-1],
         'NORTH' if SL.y.iloc[0] > SL.y.iloc[-1] else 'SOUTH'))
print('  per-face net transport (m3/s):')
print('   ' + '  '.join('%+.0f' % v for v in qf))
print('  faces flowing IN (net): %s of %d -- a LATERAL split at the mouth is '
      'a contiguous\n  run of them at one end; a vertical one would leave this '
      'list empty or ragged' % (np.where(qf < 0)[0].tolist(), len(qf)))

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
plt.close('all')
# The map carries a fixed aspect (dar), so its panel has to be roughly the
# cove's own shape or it shrinks inside its gridspec cell and leaves a band of
# white on either side. The cove box is ~1.5 wide for 1 tall at this latitude,
# and the section is given the same height so the two read as a pair.
DAR = 1 / np.cos(np.deg2rad(float(np.mean(latr))))
fig = plt.figure(figsize=(16, 4.6), layout='constrained')
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1])
axm = fig.add_subplot(gs[0])
axv = fig.add_subplot(gs[1])

# One symmetric scale for both panels, set by the section -- the larger of the
# two fields, so nothing there is clipped and the map is left honestly pale.
VMAX = args.vmax if args.vmax else float(np.nanpercentile(np.abs(u_sect), 99))
KW = dict(cmap=cm.balance, vmin=-VMAX, vmax=VMAX)
print('\nshared colour scale +/- %.4f m/s' % VMAX)
for lbl, a in [('map', u_rho), ('section', u_sect)]:
    v = a[np.isfinite(a)]
    print('  %-7s |u| reaches %.1f%% of the scale; %.1f%% of cells saturate'
          % (lbl, 100 * np.max(np.abs(v)) / VMAX, 100 * np.mean(np.abs(v) > VMAX)))

# --- map
land = (~D['mask_rho'])
axm.pcolormesh(lonr, latr, np.ma.masked_where(~land, np.ones(land.shape)),
               cmap=ListedColormap([LAND]), shading='nearest', zorder=0)
pcm = axm.pcolormesh(lonr, latr, u_rho, shading='nearest', zorder=1, **KW)

qs = max(1, args.quiver_step)
spd95 = float(np.nanpercentile(spd, 95))
qscale = (args.quiver_scale if args.quiver_scale
          else spd95 * lonr.shape[1] / (1.2 * qs))
Q = axm.quiver(lonr[::qs, ::qs], latr[::qs, ::qs],
               u_rho[::qs, ::qs], v_rho[::qs, ::qs],
               scale=qscale, scale_units='width', units='width',
               width=0.0026, color='k', zorder=6)
axm.quiverkey(Q, 0.90, 0.05, round(spd95, 3), '%.3f m s$^{-1}$' % spd95,
              labelpos='W', coordinates='axes', fontproperties=dict(size=FS - 3))

axm.plot(SL.x, SL.y, '-', color=RED, lw=3.5, zorder=9, solid_capstyle='butt')
# half a cell of margin: shading='nearest' centres each cell on its rho point,
# so limits set to the rho points themselves cut the outer cells in half
dlon = float(np.diff(lonr[0, :]).mean())
dlat = float(np.diff(latr[:, 0]).mean())
axm.set_xlim(lonr.min() - dlon / 2, lonr.max() + dlon / 2)
axm.set_ylim(latr.min() - dlat / 2, latr.max() + dlat / 2)
axm.set_aspect(DAR)
axm.set_xlabel('longitude', fontsize=FS)
axm.set_ylabel('latitude', fontsize=FS)
axm.tick_params(labelsize=FS - 2)

# --- section
x_e = np.concatenate([[0.0], np.cumsum(dd)]) / 1000.0          # km across
# One quadmesh PER FACE. A single mesh spanning all faces interpolates the
# sigma levels across columns of different depth and smears the water column
# past the stepped bathymetry -- the shading ends up below the bed.
for ip in range(len(dd)):
    Xp = np.tile([x_e[ip], x_e[ip + 1]], (zw.shape[0], 1))
    Yp = np.repeat(zw[:, ip:ip + 1], 2, axis=1)
    m = axv.pcolormesh(Xp, Yp, u_sect[:, ip:ip + 1], shading='flat', **KW)

# The zero line, on face centres -- the boundary between water leaving and
# water arriving. It is interpolated ACROSS columns of different depth, so near
# the bed it can wander below the shallower of the two; the seabed is drawn on
# top of it (zorder) so it is cut off at the bathymetry rather than drawn
# through it.
xc = 0.5 * (x_e[:-1] + x_e[1:])
zc = 0.5 * (zw[:-1] + zw[1:])
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    axv.contour(np.tile(xc, (zc.shape[0], 1)), zc, u_sect, levels=[0.0],
                colors='k', linewidths=1.2, zorder=3)

bed = -np.concatenate([hh[:1], hh])
zbot = float(bed.min()) * 1.04
axv.fill_between(x_e, bed, zbot, step='pre', color=LAND, lw=0, zorder=4)
axv.plot(x_e, bed, '-', color='0.3', lw=1.0, drawstyle='steps-pre', zorder=5)
axv.set_ylim(zbot, 0.0)
axv.set_xlim(x_e[0], x_e[-1])
axv.set_xlabel('distance across %s [km]  (north on the left)' % args.sect,
               fontsize=FS)
axv.set_ylabel('z [m]', fontsize=FS)
axv.tick_params(labelsize=FS - 2)
# framed in the same red as the line on the map, so the section and its
# location read as one object
for sp in axv.spines.values():
    sp.set_color(RED)
    sp.set_linewidth(2.0)
axv.tick_params(color=RED)

# one colourbar for the pair, spanning both panels
cb = fig.colorbar(pcm, ax=[axm, axv], pad=0.015, aspect=28, fraction=0.05)
cb.ax.tick_params(labelsize=FS - 2)
cb.set_label('mean velocity [m s$^{-1}$]  (red = out of the cove)', fontsize=FS)

stem = ('pc_mean_circulation_%s_%s_%s'
        % (args.gtagex, t0.strftime('%Y.%m.%d'), t1.strftime('%Y.%m.%d')))
fn_out = out_dir / (stem + '.png')
fig.savefig(fn_out, dpi=300, bbox_inches='tight', transparent=True)
print('\nwrote %s' % fn_out)

# the numbers behind both panels, so they can be replotted without the pickle
pd.DataFrame(dict(p=np.arange(len(dd)), h=hh, dd=dd, q_net=qf,
                  u_depth_avg=qf / (dd * (hh + zbar)))
             ).to_csv(out_dir / (stem + '_section_faces.csv'), index=False,
                      float_format='%.5f')
np.savez(out_dir / (stem + '_fields.npz'), lon_rho=lonr, lat_rho=latr,
         u_rho=u_rho, v_rho=v_rho, cove=cove, h=h,
         u_sect=u_sect, q_sect=q_m, zw=zw, dd=dd, h_sect=hh)
print('wrote %s and the section face table' % (stem + '_fields.npz'))
plt.close('all')
