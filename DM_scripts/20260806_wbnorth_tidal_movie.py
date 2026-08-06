"""
Tidal-cycle movie of the wb_north surface field, with the Penn Cove mouth
timeseries running underneath it.

Layout (one figure, animated over hourly history files):
  left   -- map of surface salinity (or --var) over wb_north
  right  -- three stacked timeseries, sharing a clock with the map:
              1. surface + bottom salinity at the NORTH side of the cove mouth
              2. surface + bottom salinity at the SOUTH side of the cove mouth
              3. Penn Cove box-mean SSH (tidal phase)
            each carries a marker showing where the animation is.

The two mouth points are placed ON the pc_lp section (the cove mouth, see the
wb1_pc1 TEF collection) at --fracs along its length, then snapped to the nearest
wet rho cell. Putting them across the mouth rather than along the cove axis is
the point: the exchange at the mouth is laterally sheared, so the north and
south sides do not do the same thing over a tidal cycle, and top-vs-bottom at
each tells you whether that is the two-layer exchange or a lateral tilt.

Region-plot convention (DM 2026.08.04): the window is the rectangular bounding
box of the WHOLE wb_north polygon plus PAD_CELLS of margin, but only cells
inside `wb` are drawn -- otherwise Admiralty Inlet and the main Puget Sound
trench come along on the far side of Whidbey and take over the color scale.

Runs on apogee (needs the history files). Defaults to 2 days = 49 hourly frames,
about two diurnal / four semidiurnal cycles.

    python 20260806_wbnorth_tidal_movie.py
    python 20260806_wbnorth_tidal_movie.py --ds0 2025.07.15 --ds1 2025.07.16
    python 20260806_wbnorth_tidal_movie.py --var temp --region skagit_delta
    python 20260806_wbnorth_tidal_movie.py --fracs 0.15,0.85 --test
"""
import argparse
import multiprocessing as mp
import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.animation as animation
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MaxNLocator
from cmocean import cm

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun
from wb1_penncove_region import PENN_COVE_BOX

# ---- arguments -------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument('--gtx', default='wb1_t0_xn11abbur00')
p.add_argument('--ro', default=2, type=int)              # /dat2/dakotamm/LO_roms
p.add_argument('--ds0', default='2025.09.01')
p.add_argument('--ds1', default='2025.09.02')            # 2 days -> 49 hourly frames
p.add_argument('--lt', default='hourly0')                # clean hour-0 start on ds0
p.add_argument('--region', default='wb_north', help='polygon that sets the map window')
p.add_argument('--var', default='salt', help='surface field to animate')
p.add_argument('--sect', default='pc_lp', help='section the two points sit on')
p.add_argument('--fracs', default='0.25,0.75',
               help='fractional positions along --sect for the two points')
p.add_argument('--pts', default='',
               help='override: lon,lat;lon,lat (still snapped to nearest wet cell)')
p.add_argument('--vmin', type=float)                     # default: percentiles in window
p.add_argument('--vmax', type=float)
p.add_argument('--pad-cells', default=10, type=int)
p.add_argument('--fps', default=6, type=int)
p.add_argument('--nproc', default=min(8, os.cpu_count() or 1), type=int,
               help='parallel workers for reading history files (1 = serial)')
p.add_argument('--test', dest='test', action='store_true',
               help='save a single still of the first frame instead of the movie')
args = p.parse_args()

gridname, tag, ex_name = args.gtx.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
Ldir['roms_out'] = Ldir['roms_out' + str(args.ro)]

for label, dsx in [('--ds0', args.ds0), ('--ds1', args.ds1)]:
    try:
        datetime.strptime(dsx, '%Y.%m.%d')
    except ValueError:
        raise SystemExit('Invalid %s value %r -- use YYYY.MM.DD with a real '
                         'calendar day.' % (label, dsx))

out_dir = Ldir['LOo'] / 'DM_outs' / '20260806_wbnorth_tidal_movie'
Lfun.make_dir(out_dir)

fn_list = [fn for fn in Lfun.get_fn_list(args.lt, Ldir, args.ds0, args.ds1)
           if fn.is_file()]
if len(fn_list) == 0:
    raise SystemExit('No history files for %s %s-%s'
                     % (args.gtx, args.ds0, args.ds1))
print('%d hourly frames %s .. %s' % (len(fn_list), args.ds0, args.ds1))

# ---- grid, polygons, window ------------------------------------------------
dsg = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon = dsg.lon_rho.values
lat = dsg.lat_rho.values
mask_rho = dsg.mask_rho.values
h = dsg.h.values
dsg.close()
wet = mask_rho == 1
COS = np.cos(np.deg2rad(lat.mean()))
dx = float(np.diff(lon[0, :]).mean())
dy = float(np.diff(lat[:, 0]).mean())

sect_dir = Ldir['LOo'] / 'section_lines'


def load_sect(name):
    """A section_lines polygon/line, x and y forced to float.

    They pickle as object dtype, which silently turns np.diff/np.hypot on the
    coordinates into elementwise python-object math and raises.
    """
    fn = sect_dir / (name + '.p')
    if not fn.is_file():
        raise SystemExit('missing %s' % fn)
    df = pd.read_pickle(fn)
    return df.assign(x=df.x.astype(float), y=df.y.astype(float))


def poly_mask(df):
    pth = MplPath(np.column_stack([df.x.values, df.y.values]))
    return pth.contains_points(
        np.column_stack([lon.ravel(), lat.ravel()])).reshape(lon.shape)


reg = load_sect(args.region)
sect = load_sect(args.sect)
in_wb = poly_mask(load_sect('wb'))            # master clip, every wb1 region plot

# rectangular window around the WHOLE region polygon + a margin of cells
aa = [reg.x.min() - args.pad_cells * dx, reg.x.max() + args.pad_cells * dx,
      reg.y.min() - args.pad_cells * dy, reg.y.max() + args.pad_cells * dy]

# index bounds of that window, so the workers only ship back the subset
jj = np.where((lat[:, 0] >= aa[2]) & (lat[:, 0] <= aa[3]))[0]
ii = np.where((lon[0, :] >= aa[0]) & (lon[0, :] <= aa[1]))[0]
j0, j1 = int(jj[0]), int(jj[-1]) + 1
i0, i1 = int(ii[0]), int(ii[-1]) + 1
SUB = (slice(j0, j1), slice(i0, i1))
lon_s, lat_s = lon[SUB], lat[SUB]
plon_s, plat_s = pfun.get_plon_plat(lon_s, lat_s)
draw_s = (wet & in_wb)[SUB]                   # rectangular extent, wb-clipped content
print('window %s -> %d x %d cells, %d drawn'
      % (['%.4f' % v for v in aa], lat_s.shape[0], lat_s.shape[1], draw_s.sum()))

# Penn Cove SSH box (shared with the other Penn Cove plots)
box = PENN_COVE_BOX
in_box = ((lon >= box[0]) & (lon <= box[1]) &
          (lat >= box[2]) & (lat <= box[3]) & wet)
print('SSH box: %d wet cells' % in_box.sum())


# ---- the two points across the cove mouth ----------------------------------
def point_on_line(df, frac):
    """(lon, lat) at a fractional distance along the polyline."""
    x, y = df.x.values, df.y.values
    d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x) * COS, np.diff(y)))])
    return float(np.interp(frac * d[-1], d, x)), float(np.interp(frac * d[-1], d, y))


def nearest_wet(x0, y0):
    """(j, i) of the nearest wet rho cell -- the line ends sit on land."""
    dd = ((lon - x0) * COS) ** 2 + (lat - y0) ** 2
    dd = np.where(wet, dd, np.inf)
    return np.unravel_index(np.argmin(dd), dd.shape)


if args.pts:
    want = [tuple(float(v) for v in s.split(',')) for s in args.pts.split(';') if s]
else:
    want = [point_on_line(sect, float(f)) for f in args.fracs.split(',')]
if len(want) != 2:
    raise SystemExit('need exactly two points, got %d' % len(want))

PTS = []
for x0, y0 in want:
    j, i = nearest_wet(x0, y0)
    PTS.append(dict(j=int(j), i=int(i), lon=float(lon[j, i]),
                    lat=float(lat[j, i]), h=float(h[j, i])))
PTS.sort(key=lambda d: -d['lat'])             # north first
LABELS = ['north side of %s' % args.sect, 'south side of %s' % args.sect]
# colorblind-safe, and distinct from the haline map underneath
PCOLOR = ['#0072B2', '#D55E00']
for lab, P in zip(LABELS, PTS):
    print('%-24s lon %.4f lat %.4f  (j=%d i=%d)  h = %.1f m'
          % (lab, P['lon'], P['lat'], P['j'], P['i'], P['h']))

JI = [(P['j'], P['i']) for P in PTS]


# ---- read one history file -------------------------------------------------
def read_one(fn):
    """Surface field in the window, box-mean SSH, and top/bottom salt at PTS."""
    ds = xr.open_dataset(fn)
    if args.var not in ds.data_vars:
        ds.close()
        raise SystemExit('no variable %r in %s' % (args.var, fn))
    fld = ds[args.var][0, -1, SUB[0], SUB[1]].values.astype(np.float32)
    zeta = ds.zeta[0, :, :].values
    ssh = float(np.nanmean(np.where(in_box, zeta, np.nan)))
    srf, bot = [], []
    for j, i in JI:
        srf.append(float(ds.salt[0, -1, j, i].values))
        bot.append(float(ds.salt[0, 0, j, i].values))
    t_utc = pd.Timestamp(ds.ocean_time.values[0]).to_pydatetime()
    ds.close()
    return (fld, ssh, np.array(srf), np.array(bot),
            pfun.get_dt_local(t_utc).replace(tzinfo=None))   # naive local (PST)


nproc = max(1, min(args.nproc, len(fn_list)))
print('reading %d files on %d process(es)...' % (len(fn_list), nproc))
if nproc > 1:
    ctx = mp.get_context('fork')              # workers inherit the masks above
    with ctx.Pool(nproc) as pool:
        res = pool.map(read_one, fn_list)
else:
    res = [read_one(fn) for fn in fn_list]

FLD = np.stack([r[0] for r in res])                      # (nt, ny, nx)
SSH = np.array([r[1] for r in res])
SRF = np.stack([r[2] for r in res])                      # (nt, 2)
BOT = np.stack([r[3] for r in res])
TT = [r[4] for r in res]
FLD = np.where(draw_s[None, :, :], FLD, np.nan)          # land + outside-wb
print('SSH %.2f to %.2f m   (range %.2f m)'
      % (SSH.min(), SSH.max(), SSH.max() - SSH.min()))
for k, lab in enumerate(LABELS):
    print('%-24s surface %.2f-%.2f  bottom %.2f-%.2f g/kg'
          % (lab, SRF[:, k].min(), SRF[:, k].max(),
             BOT[:, k].min(), BOT[:, k].max()))

# color limits from the drawn cells only, fixed for the whole movie -- an
# auto-scaled frame would make the tide look like a color-table change
v = FLD[np.isfinite(FLD)]
vmin = args.vmin if args.vmin is not None else float(np.percentile(v, 1))
vmax = args.vmax if args.vmax is not None else float(np.percentile(v, 99))
CMAP = {'salt': cm.haline, 'temp': cm.thermal, 'oxygen': cm.oxy}.get(args.var, cm.haline)
UNITS = {'salt': 'g kg$^{-1}$', 'temp': '$^{\\circ}$C',
         'oxygen': 'mmol m$^{-3}$'}.get(args.var, '')
print('color limits %.2f to %.2f' % (vmin, vmax))

# ---- figure ----------------------------------------------------------------
# constrained layout, not manual margins: the map carries a fixed aspect (dar),
# so it shrinks inside its own gridspec cell and a hand-placed colorbar ends up
# stranded out in the whitespace.
plt.close('all')
fig = plt.figure(figsize=(15.5, 9), layout='constrained')
gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1])
axm = fig.add_subplot(gs[:, 0])
axa = fig.add_subplot(gs[0, 1])
axb = fig.add_subplot(gs[1, 1], sharex=axa, sharey=axa)
axs = fig.add_subplot(gs[2, 1], sharex=axa)

# --- map
cs = axm.pcolormesh(plon_s, plat_s, FLD[0], cmap=CMAP, vmin=vmin, vmax=vmax,
                    shading='flat', zorder=1)
fig.colorbar(cs, ax=axm, shrink=0.75, pad=0.01, aspect=35,
             label='surface %s [%s]' % (args.var, UNITS))
pfun.add_coast(axm, color='gray', linewidth=0.5)
axm.plot(reg.x, reg.y, '-', color='tab:purple', lw=1.8, zorder=6,
         label=args.region)
axm.plot(sect.x, sect.y, '-', color='magenta', lw=2.2, zorder=7, label=args.sect)
pfun.draw_box(axm, box, color='k', linewidth=1.0, linestyle='--')
for lab, P, c in zip(LABELS, PTS, PCOLOR):
    axm.plot(P['lon'], P['lat'], 'o', ms=10, color=c, markeredgecolor='k',
             markeredgewidth=1.0, zorder=10, label=lab)
axm.axis(aa)
pfun.dar(axm)
axm.xaxis.set_major_locator(MaxNLocator(nbins=4))
axm.tick_params(axis='x', labelrotation=45, labelsize=9)
axm.set_xlabel('Longitude')
axm.set_ylabel('Latitude')
axm.legend(loc='upper left', fontsize=8, framealpha=0.9)   # land, not water
ttl = axm.set_title('', fontsize=13)

# --- the two salinity panels
for ax, k, lab, c in zip([axa, axb], [0, 1], LABELS, PCOLOR):
    ax.plot(TT, SRF[:, k], '-', color=c, lw=1.8, label='surface')
    ax.plot(TT, BOT[:, k], '--', color=c, lw=1.8, label='bottom')
    ax.set_ylabel('salinity [g kg$^{-1}$]')
    ax.grid(color='lightgray', ls='--', alpha=0.6)
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.set_title('%s   (%.4f, %.4f, h = %.0f m)'
                 % (lab, PTS[k]['lon'], PTS[k]['lat'], PTS[k]['h']),
                 fontsize=10, color=c)
    plt.setp(ax.get_xticklabels(), visible=False)

# --- SSH
axs.plot(TT, SSH, '-', color='#3B0F70', lw=1.8)
axs.axhline(0, color='0.5', lw=0.8)
axs.set_ylabel('Penn Cove SSH [m]')
axs.set_title('tidal phase (box mean over the dashed box)', fontsize=10)
axs.grid(color='lightgray', ls='--', alpha=0.6)
axs.set_xlim(TT[0], TT[-1])
axs.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
axs.set_xlabel('local time (PST)')
for l in axs.get_xticklabels():
    l.set_rotation(30)
    l.set_horizontalalignment('right')

# moving markers
marks = []
for ax in [axa, axb, axs]:
    marks.append(ax.axvline(TT[0], color='k', lw=1.5, zorder=8))
dots = []
for ax, k, c in zip([axa, axb], [0, 1], PCOLOR):
    dots.append(ax.plot([TT[0]], [SRF[0, k]], 'o', ms=7, color=c,
                        markeredgecolor='k', zorder=9)[0])
    dots.append(ax.plot([TT[0]], [BOT[0, k]], 'o', ms=7, mfc='white',
                        markeredgecolor=c, zorder=9)[0])
dots.append(axs.plot([TT[0]], [SSH[0]], 'o', ms=8, color='r', zorder=9)[0])

fig.suptitle('%s -- surface %s over %s, with the %s mouth timeseries'
             % (args.gtx, args.var, args.region, args.sect), fontsize=13)


def update(fi):
    cs.set_array(FLD[fi].ravel())
    ttl.set_text('surface %s -- %s (PST)'
                 % (args.var, TT[fi].strftime('%Y-%m-%d %H:%M')))
    for m in marks:
        m.set_xdata([TT[fi], TT[fi]])
    for k in range(2):
        dots[2 * k].set_data([TT[fi]], [SRF[fi, k]])
        dots[2 * k + 1].set_data([TT[fi]], [BOT[fi, k]])
    dots[-1].set_data([TT[fi]], [SSH[fi]])
    return []


stem = ('20260806_%s_%s_%s_%s_%s'
        % (args.region, args.var, args.sect, args.ds0, args.ds1))
if args.test:
    update(0)
    fn_out = out_dir / (stem + '_frame0.png')
    fig.savefig(fn_out, dpi=150)
    print('TEST: saved %s' % fn_out)
else:
    anim = animation.FuncAnimation(fig, update, frames=len(TT),
                                   interval=1000 / args.fps, blit=False)
    fn_out = out_dir / (stem + '.mp4')
    anim.save(fn_out, writer=animation.FFMpegWriter(fps=args.fps, bitrate=3000))
    print('saved %s' % fn_out)
    update(0)
    fig.savefig(out_dir / (stem + '_frame0.png'), dpi=150)
    print('saved %s' % (out_dir / (stem + '_frame0.png')))
plt.close('all')
