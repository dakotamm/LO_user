"""
Movie of the TIDAL flow in Penn Cove over one tidal day, reconstructed from the
2-year harmonic fits of 20261008_tidal_ellipses.py (11 constituents at every
cell, depth-averaged and on every sigma layer). Tide only: no residual, no
wind, no S1 (the daily wind-driven current, 20261027_pc_diurnal_S1.py) --
though the K1/P1 currents carry some seasonally modulated sea breeze.

Default day: 2025-06-25, a spring AND tropic day (both the semidiurnal and the
diurnal tide near their largest), so the diurnal inequality is strong. Contrast:
-day 2024.11.25, the weakest equatorial + neap day. Color and arrow scales are
FIXED (-smax_cm, -vsec_cm) so movies of different days compare directly; the
title labels the day from the band-fit tercile classification.

Reconstruction is utide's own: u + iv = sum F (ap e^{i(V+U)} + am e^{-i(V+U)}),
with ap, am from (Lsmaj, Lsmin, theta, g) and F, U, V from utide's ut_E for the
same constituents, reference time and nodal settings as the fits (checked
against the model's box-mean zeta over the movie window).

Frames every 30 min:
  top     box-mean sea level over the day, with the current time
  left    plan view: depth-averaged tidal current (arrows, one scale)
  right   the mouth section (last rho column west of pc_lp): along-cove
          velocity by depth and latitude, red = OUT (ebb), blue = IN (flood)

Output: LO_output/DM_outs/20261027_pc_tidal_cycle_movie/pc_tidal_cycle_<day>.mp4
        (+ a frame-grid PNG of 6 phases)

  python 20261027_pc_tidal_cycle_movie.py -day 2025.06.25
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import utide
from utide.harmonics import ut_E
from utide._time_conversion import _normalize_time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.path import Path as MplPath
import cmocean

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

warnings.filterwarnings('ignore')
p = argparse.ArgumentParser()
p.add_argument('-day', default='2025.06.25')
p.add_argument('-hours', type=float, default=25.0)
p.add_argument('-dt_min', type=int, default=30)
p.add_argument('-fps', type=int, default=6)
p.add_argument('-smax_cm', type=float, default=12.0) # fixed scales so different days compare
p.add_argument('-vsec_cm', type=float, default=22.0)
args = p.parse_args()

GTX, JOB, DS0, DS1 = 'wb1_t0_xn11abbur00', 'pc_cove', '2024.01.01', '2025.12.31'
Ldir = Lfun.Lstart(gridname='wb1')
F8 = xr.open_dataset(Ldir['LOo'] / 'extract' / GTX / 'tidal_ellipses' / ('%s_%s_%s.nc' % (JOB, DS0, DS1)))
box = xr.open_dataset(Ldir['LOo'] / 'extract' / GTX / 'box' / ('%s_%s_%s.nc' % (JOB, DS0, DS1)))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261027_pc_tidal_cycle_movie'
Lfun.make_dir(out_dir)
sect_dir = Ldir['LOo'] / 'section_lines'
CONS = [str(c) for c in F8.con.values]
OUT = np.array([0.9789, 0.2043]) # unit vector pointing OUT of the cove (head -> mouth)

# ---- basis functions from a template fit with the same constituents / reftime / nodal settings
T = box.ocean_time.values
mask = box.mask_rho.values == 1
ZB = np.nanmean(box.zeta.values[:, mask], axis=1)
tmpl = utide.solve(T, ZB.astype(float), lat=48.23, method='ols', conf_int='none', constit=CONS, nodal=True, trend=False, verbose=False)
iT = [list(tmpl.name).index(c) for c in CONS]
frq, lind = np.asarray(tmpl.aux.frq)[iT], np.asarray(tmpl.aux.lind)[iT]
o = tmpl.aux.opt
ngflgs = [o['nodsatlint'], o['nodsatnone'], o['gwchlint'], o['gwchnone']]
t0 = pd.Timestamp(args.day.replace('.', '-'))
TA = pd.date_range(t0, t0 + pd.Timedelta(hours=args.hours), freq='%dmin' % args.dt_min)
E = ut_E(_normalize_time(TA.values), tmpl.aux.reftime, frq, lind, 48.23, ngflgs, None) # (nt, ncon)

def recon_uv(L, Ln, th, g):
    """(ncon, ...) ellipse params -> (nt, ...) u, v"""
    r = np.pi / 180
    sh = L.shape[1:]
    L, Ln, th, g = [x.reshape(len(CONS), -1) for x in (L, Ln, th, g)]
    ap = 0.5 * (L + Ln) * np.exp(1j * (th - g) * r)
    am = 0.5 * (L - Ln) * np.exp(1j * (th + g) * r)
    w = E @ np.nan_to_num(ap) + np.conj(E) @ np.nan_to_num(am)
    bad = ~np.isfinite(L).all(0)
    w[:, bad] = np.nan
    return w.real.reshape((-1,) + sh), w.imag.reshape((-1,) + sh)

def recon_z(A, g):
    ap = 0.5 * A * np.exp(-1j * g * np.pi / 180)
    return (E @ ap + np.conj(E) @ np.conj(ap)).real

zA, zg = np.asarray(tmpl.A)[iT], np.asarray(tmpl.g)[iT]
ZR = recon_z(zA, zg) + tmpl.mean
# check against the model's box-mean zeta over the window
zm = pd.Series(ZB, index=pd.DatetimeIndex(T)).reindex(TA, method='nearest', tolerance=pd.Timedelta('1min'))
ok = zm.notna().values
print('reconstructed vs model box-mean zeta over the window: r = %.3f, rms diff %.1f cm (range %.1f m)'
      % (np.corrcoef(ZR[ok], zm.values[ok])[0, 1], 100 * np.sqrt(np.mean((ZR[ok] - zm.values[ok]) ** 2)), np.ptp(ZR)))

g4 = lambda P, vn: F8[P + '_' + vn].values
UB, VB = recon_uv(g4('bar', 'Lsmaj'), g4('bar', 'Lsmin'), g4('bar', 'theta'), g4('bar', 'g'))
LON_LP = pd.read_pickle(sect_dir / 'pc_lp.p').x.mean()
lon, lat = F8.lon_rho.values, F8.lat_rho.values
wb = pd.read_pickle(sect_dir / 'wb.p')
pth = MplPath(np.column_stack([wb.x.values, wb.y.values]))
show = mask & pth.contains_points(np.column_stack([lon.ravel(), lat.ravel()])).reshape(lon.shape)
im = int(np.flatnonzero(lon[0] < LON_LP)[-1])
jw = np.flatnonzero(show[:, im])
U3, V3 = recon_uv(g4('3d', 'Lsmaj')[:, :, jw, im], g4('3d', 'Lsmin')[:, :, jw, im], g4('3d', 'theta')[:, :, jw, im], g4('3d', 'g')[:, :, jw, im])
SEC = 100 * (U3 * OUT[0] + V3 * OUT[1]) # (nt, NZ, nj) cm/s, + = out
z0 = F8.z0_rho.values[:, jw, im]
yy = np.broadcast_to(lat[jw, im], z0.shape)
QOUT = np.nanmean(SEC, axis=(1, 2)) # section-mean out velocity, for the flood/ebb label
box.close()

# ---- figure
plon, plat = pfun.get_plon_plat(lon, lat)
aa = [plon.min(), plon.max(), plat.min(), plat.max()]
step = 2
jj, ii = np.meshgrid(np.arange(0, lon.shape[0], step), np.arange(0, lon.shape[1], step), indexing='ij')
sq = show[jj, ii] & np.isfinite(UB[0][jj, ii])
SPD = np.hypot(UB, VB)
smax = args.smax_cm / 100
vsec = args.vsec_cm
print('this day: 99.5th pct depth-avg speed %.1f cm/s, section |u| %.1f cm/s (fixed scales %g, %g)' % (100 * np.nanpercentile(SPD[:, show], 99.5), np.nanpercentile(np.abs(SEC), 99.5), args.smax_cm, args.vsec_cm))
# label the day from the band-fit classification (20261027_tidal_ellipses_windowed.py)
BF = xr.open_dataset(Ldir['LOo'] / 'extract' / GTX / 'tidal_ellipses' / ('%s_%s_%s_bands.nc' % (JOB, DS0, DS1)))
kd = int(np.argmin(np.abs(pd.DatetimeIndex(BF.time.values) - (t0 + pd.Timedelta(hours=12)))))
DAYLAB = ' + '.join([nm for nm in ['spring', 'neap', 'tropic', 'equatorial'] if int(BF['is_' + nm].values[kd]) == 1]) or 'mid-cycle'
BF.close()
tl = TA.tz_localize('UTC').tz_convert('America/Los_Angeles')

fig = plt.figure(figsize=(14, 8.2), layout='constrained')
gs = fig.add_gridspec(2, 2, height_ratios=[1, 2.6], width_ratios=[1.35, 1])
axz = fig.add_subplot(gs[0, :])
axm = fig.add_subplot(gs[1, 0])
axs = fig.add_subplot(gs[1, 1])

axz.plot(TA, ZR, '-', color='0.3', lw=2)
dot, = axz.plot([], [], 'o', ms=10, color='#D55E00', zorder=5)
axz.axhline(0, color='0.6', lw=0.8)
axz.set_ylabel('sea level [m]')
axz.set_xlim(TA[0], TA[-1])
axz.grid(color='lightgray', linestyle='--', alpha=0.5)
lab = axz.text(0.01, 0.92, '', transform=axz.transAxes, fontsize=12, fontweight='bold', va='top')

pc = axm.pcolormesh(plon, plat, np.where(show, 100 * SPD[0], np.nan), cmap=cmocean.cm.speed, vmin=0, vmax=100 * smax, shading='flat')
q = axm.quiver(lon[jj, ii][sq], lat[jj, ii][sq], UB[0][jj, ii][sq], VB[0][jj, ii][sq], scale=smax / 0.35, scale_units='inches', width=0.004, color='k', zorder=3)
axm.quiverkey(q, 0.12, 0.07, 0.05, '5 cm/s', labelpos='N', coordinates='axes', fontproperties={'size': 9})
pfun.add_coast(axm, color='gray', linewidth=0.6)
axm.axis(aa)
pfun.dar(axm)
axm.axvline(lon[0, im], color='k', ls=':', lw=1)
axm.set_xticks([])
axm.set_yticks([])
axm.set_title('depth-averaged tidal current (dotted = mouth section)', fontsize=10)
fig.colorbar(pc, ax=axm, shrink=0.6, label='speed [cm/s]')

cs = axs.pcolormesh(yy, z0, SEC[0], cmap=cmocean.cm.balance, vmin=-vsec, vmax=vsec, shading='nearest')
axs.set_xlabel('latitude across the mouth (south <-  -> north)')
axs.set_ylabel('depth at mean sea level [m]')
axs.set_title('mouth section: red = OUT (ebb), blue = IN (flood)', fontsize=10)
fig.colorbar(cs, ax=axs, shrink=0.6, label='along-cove velocity [cm/s]')
cont = [None]

def state(n):
    s = QOUT[n] / np.nanmax(np.abs(QOUT))
    if abs(s) < 0.15:
        return 'SLACK'
    return 'EBB (out)' if s > 0 else 'FLOOD (in)'

def draw(n):
    dot.set_data([TA[n]], [ZR[n]])
    lab.set_text('%s   %s   %s' % (TA[n].strftime('%Y-%m-%d %H:%M UTC'), tl[n].strftime('%H:%M %Z'), state(n)))
    pc.set_array(np.where(show, 100 * SPD[n], np.nan).ravel())
    q.set_UVC(UB[n][jj, ii][sq], VB[n][jj, ii][sq])
    cs.set_array(SEC[n].ravel())
    if cont[0] is not None:
        cont[0].remove()
    cont[0] = axs.contour(yy, z0, SEC[n], levels=[0], colors='k', linewidths=1)
    return dot, lab, pc, q, cs

fig.suptitle('Penn Cove tidal flow over one tidal day (%s, %s; tide only, reconstructed from 2-year fits): %s'
             % (args.day, DAYLAB, GTX), fontsize=11)
anim = animation.FuncAnimation(fig, draw, frames=len(TA), blit=False)
fn = out_dir / ('pc_tidal_cycle_%s.mp4' % args.day)
anim.save(fn, writer=animation.FFMpegWriter(fps=args.fps, extra_args=['-pix_fmt', 'yuv420p']), dpi=130)
print('Saved ' + str(fn))

# frame grid for a quick look: 6 phases
pick = np.linspace(0, len(TA) - 1, 7).astype(int)[:-1]
for n in pick:
    draw(n)
    fig.savefig(out_dir / ('frame_%s_%02d.png' % (args.day, n)), dpi=110, facecolor='white')
plt.close(fig)
print('frames:', [str(TA[n]) for n in pick])
