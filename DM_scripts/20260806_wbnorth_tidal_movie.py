"""
Tidal-cycle movie of the wb_north surface field, with Penn Cove SSH beside it.

Layout (one figure, animated over hourly history files):
  right       -- map of surface salinity (or --var) over wb_north, with the
                 bounding box of the `pc` polygon drawn in red
  top left    -- SSH averaged over that same red box (tidal phase), carrying a
                 marker showing where the animation is
  lower left  -- deliberately blank, reserved for two more timeseries

The red box does double duty: it is what is drawn on the map AND what SSH is
averaged over, so there is only one rectangle in the figure and it means one
thing. (It replaced PENN_COVE_BOX from wb1_penncove_region.py, which was a
hand-tuned rectangle that did not match any drawn outline.)

Region-plot convention (DM 2026.08.04): the window is the rectangular bounding
box of the WHOLE wb_north polygon plus PAD_CELLS of margin, but only cells
inside `wb` are drawn -- otherwise Admiralty Inlet and the main Puget Sound
trench come along on the far side of Whidbey and take over the color scale.

Per-point top/bottom salinity at the cove mouth is no longer plotted here; that
question is answered properly, over two years, by
20260806_pc_mouth_salinity_tides.py.

Runs on apogee (needs the history files). Defaults to the first week of
September 2025 = 169 hourly frames, about 14 semidiurnal / 7 diurnal cycles,
which is long enough to watch the spring-neap envelope open and close.

    python 20260806_wbnorth_tidal_movie.py
    python 20260806_wbnorth_tidal_movie.py --ds0 2025.07.15 --ds1 2025.07.16
    python 20260806_wbnorth_tidal_movie.py --var temp --region skagit_delta
    python 20260806_wbnorth_tidal_movie.py --test
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
from matplotlib.colors import BoundaryNorm, Normalize, PowerNorm
from matplotlib.ticker import MaxNLocator
from cmocean import cm

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

# house style, same as the pc_mouth analysis figures. Grid on the timeseries,
# never on the map -- DM 2026.08.07.
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
RED = '#e04256'
BLUE = '#4565e8'

# ---- arguments -------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument('--gtx', default='wb1_t0_xn11abbur00')
p.add_argument('--ro', default=2, type=int)              # /dat2/dakotamm/LO_roms
p.add_argument('--ds0', default='2025.09.01')
p.add_argument('--ds1', default='2025.09.07')            # 1 week -> 169 hourly frames
p.add_argument('--lt', default='hourly0')                # clean hour-0 start on ds0
p.add_argument('--region', default='wb_north', help='polygon that sets the map window')
p.add_argument('--var', default='salt', help='surface field to animate')
p.add_argument('--pc-poly', default='pc', dest='pc_poly',
               help='polygon whose bounding box is drawn in red and used for SSH')
p.add_argument('--vmin', type=float)                     # default: percentiles in window
p.add_argument('--vmax', type=float)
p.add_argument('--pmin', default=0.5, type=float,
               help='percentile setting vmin -- lower it to reach further into '
                    'the fresh tail of the plume')
p.add_argument('--pmax', default=99.5, type=float)
p.add_argument('--norm', default='linear', choices=['linear', 'power', 'quantile'],
               help='power (default) stretches the fresh end; quantile gives '
                    'every colour band an equal number of cells; linear is the '
                    'plain scale')
p.add_argument('--gamma', default=0.5, type=float,
               help='power-norm exponent. <1 expands the FRESH end; 1 = linear')
p.add_argument('--levels', default='',
               help='explicit colour boundaries, e.g. 10,16,20,23,25,27,28,29,30')
p.add_argument('--pad-cells', default=10, type=int)
p.add_argument('--fps', default=6, type=int)
p.add_argument('--transparent', action='store_true',
               help='transparent background on the saved stills (off by '
                    'default; the movie is always opaque)')
p.add_argument('--dpi', default=150, type=int,
               help='render dpi for the movie frames -- 100 makes the labels '
                    'mushy once h264 has had a go at them')
p.add_argument('--vformat', default='mp4', choices=['mp4', 'prores', 'qtrle'],
               help='mp4 = h264 on white (small, plays anywhere); prores/qtrle '
                    '= .mov with a real alpha channel, for compositing')
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
dsg.close()
wet = mask_rho == 1
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
pc = load_sect(args.pc_poly)
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

# The red box is the bounding box of the pc polygon, and SSH is averaged over
# that same box -- one box on the map means one thing, rather than drawing the
# cove outline and quietly averaging over a different rectangle.
box = [float(pc.x.min()), float(pc.x.max()),
       float(pc.y.min()), float(pc.y.max())]
in_box = ((lon >= box[0]) & (lon <= box[1]) &
          (lat >= box[2]) & (lat <= box[3]) & wet)
print('%s bounding box %s -> %d wet cells for the SSH mean'
      % (args.pc_poly, ['%.4f' % v for v in box], in_box.sum()))


# ---- read one history file -------------------------------------------------
def read_one(fn):
    """Surface field in the window and the box-mean SSH."""
    ds = xr.open_dataset(fn)
    if args.var not in ds.data_vars:
        ds.close()
        raise SystemExit('no variable %r in %s' % (args.var, fn))
    fld = ds[args.var][0, -1, SUB[0], SUB[1]].values.astype(np.float32)
    zeta = ds.zeta[0, :, :].values
    ssh = float(np.nanmean(np.where(in_box, zeta, np.nan)))
    t_utc = pd.Timestamp(ds.ocean_time.values[0]).to_pydatetime()
    ds.close()
    return (fld, ssh,
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
TT = [r[2] for r in res]
FLD = np.where(draw_s[None, :, :], FLD, np.nan)          # land + outside-wb
print('SSH %.2f to %.2f m   (range %.2f m)'
      % (SSH.min(), SSH.max(), SSH.max() - SSH.min()))

# Colour limits from the drawn cells only, fixed for the whole movie -- an
# auto-scaled frame would make the tide look like a colour-table change.
#
# Salinity is capped at 27 by default rather than run to the 99.5th percentile.
# Whidbey Basin sits in a narrow salty band, so a full-range scale spends most
# of the colour table on water that never changes; capping saturates that and
# hands ~78% of the table to the sub-25 range where the Skagit plume lives.
# Everything above 27 goes one colour -- the colourbar is drawn with an arrow
# to say so. Per-variable, so --var temp / oxygen are unaffected.
DEFAULT_VMAX = {'salt': 27.0}
v = FLD[np.isfinite(FLD)]
vmin = args.vmin if args.vmin is not None else float(np.percentile(v, args.pmin))
if args.vmax is not None:
    vmax = args.vmax
elif args.var in DEFAULT_VMAX:
    vmax = DEFAULT_VMAX[args.var]
else:
    vmax = float(np.percentile(v, args.pmax))
CMAP = {'salt': cm.haline, 'temp': cm.thermal, 'oxygen': cm.oxy}.get(args.var, cm.haline)
UNITS = {'salt': 'g kg$^{-1}$', 'temp': '$^{\\circ}$C',
         'oxygen': 'mmol m$^{-3}$'}.get(args.var, '')

# Most of wb_north sits in a narrow salty range, so a LINEAR scale spends most
# of the colour table on water that never changes and squeezes the Skagit plume
# into the first few shades. Two ways out, both keeping the full data range:
#   power    -- continuous, gamma < 1 stretches the fresh end. Smooth, which
#               matters in a movie; discrete bands shimmer frame to frame.
#   quantile -- boundaries at data percentiles, so every colour band holds the
#               same number of cells. Maximum contrast everywhere, but the
#               colourbar is no longer linear in salinity and has to be read
#               off its tick labels.
NLEV = 12
if args.levels:
    lv = np.array(sorted(float(x) for x in args.levels.split(',')))
    norm = BoundaryNorm(lv, CMAP.N)
    print('colour levels (explicit): %s' % np.round(lv, 2).tolist())
elif args.norm == 'quantile':
    lv = np.unique(np.percentile(v[(v >= vmin) & (v <= vmax)],
                                 np.linspace(0, 100, NLEV + 1)))
    norm = BoundaryNorm(lv, CMAP.N)
    print('colour levels (quantile): %s' % np.round(lv, 2).tolist())
elif args.norm == 'power':
    norm = PowerNorm(gamma=args.gamma, vmin=vmin, vmax=vmax)
    print('colour scale: power norm, gamma %.2f, %.2f to %.2f'
          % (args.gamma, vmin, vmax))
else:
    norm = Normalize(vmin=vmin, vmax=vmax)
    print('colour scale: linear, %.2f to %.2f' % (vmin, vmax))
print('  data in window: min %.2f, %gth pctl %.2f, median %.2f, max %.2f'
      % (v.min(), args.pmin, vmin, np.median(v), v.max()))
f_hi = float(np.mean(v > vmax))
f_lo = float(np.mean(v < vmin))
print('  saturated: %.1f%% of cell-hours above vmax, %.1f%% below vmin'
      % (100 * f_hi, 100 * f_lo))
# arrow on the colourbar wherever data runs past the limit, so a saturated
# background never reads as a real value
CB_EXT = ('both' if (f_lo > 0 and f_hi > 0) else 'max' if f_hi > 0
          else 'min' if f_lo > 0 else 'neither')

# ---- figure ----------------------------------------------------------------
# constrained layout, not manual margins: the map carries a fixed aspect (dar),
# so it shrinks inside its own gridspec cell and a hand-placed colorbar ends up
# stranded out in the whitespace.
plt.close('all')
fig = plt.figure(figsize=(15.5, 9), layout='constrained')
gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.15])
axs = fig.add_subplot(gs[0, 0])                  # SSH, top left
axblank = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[2, 0])]
axm = fig.add_subplot(gs[:, 1])                  # map, right

# --- map
cs = axm.pcolormesh(plon_s, plat_s, FLD[0], cmap=CMAP, norm=norm,
                    shading='flat', zorder=1)
cb_kw = dict(shrink=0.75, pad=0.01, aspect=35,
             label='surface %s [%s]' % (args.var, UNITS))
if isinstance(norm, BoundaryNorm):
    cb = fig.colorbar(cs, ax=axm, **cb_kw)     # extend lives on the norm here
    cb.set_ticks(norm.boundaries)
    cb.ax.set_yticklabels(['%.1f' % b for b in norm.boundaries], fontsize=8)
else:
    cb = fig.colorbar(cs, ax=axm, extend=CB_EXT, **cb_kw)
pfun.add_coast(axm, color='gray', linewidth=0.5)
pfun.draw_box(axm, box, color=RED, linewidth=2.0)
axm.axis(aa)
pfun.dar(axm)
axm.xaxis.set_major_locator(MaxNLocator(nbins=4))
axm.tick_params(axis='x', labelrotation=45, labelsize=9)
axm.set_xlabel('Longitude')
axm.set_ylabel('Latitude')
ttl = axm.set_title('', fontsize=13)

# --- SSH
axs.plot(TT, SSH, '-', color=BLUE, lw=1.8)
axs.axhline(0, color='0.5', lw=0.8)
axs.set_ylabel('Penn Cove SSH [m]')
axs.set_title('tidal phase (mean over the red box)', fontsize=10, color=RED)
axs.grid(**GRID)
axs.set_xlim(TT[0], TT[-1])
axs.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
axs.set_xlabel('local time (PST)')
for l in axs.get_xticklabels():
    l.set_rotation(30)
    l.set_horizontalalignment('right')

# The lower two left cells are deliberately empty for now -- the axes are
# created so the SSH panel keeps its size and position, then switched off so
# they read as blank (transparent) rather than as two empty framed boxes.
for ax in axblank:
    ax.axis('off')

# moving markers
mark = axs.axvline(TT[0], color='k', lw=1.5, zorder=8)
dot = axs.plot([TT[0]], [SSH[0]], 'o', ms=8, color=RED, zorder=9)[0]

fig.suptitle('%s -- surface %s over %s'
             % (args.gtx, args.var, args.region), fontsize=13)

if args.transparent:
    fig.patch.set_alpha(0.0)
    for ax in [axm, axs]:
        ax.patch.set_alpha(0.0)


def update(fi):
    cs.set_array(FLD[fi].ravel())
    ttl.set_text('surface %s -- %s (PST)'
                 % (args.var, TT[fi].strftime('%Y-%m-%d %H:%M')))
    mark.set_xdata([TT[fi], TT[fi]])
    dot.set_data([TT[fi]], [SSH[fi]])
    return []


stem = ('20260806_%s_%s_%s_%s' % (args.region, args.var, args.ds0, args.ds1))
still_kw = dict(dpi=200, bbox_inches='tight', transparent=args.transparent)

# Video formats. h264 has no alpha channel, so mp4 is rendered on white -- a
# transparent figure piped to it just gets composited onto black. ProRes 4444
# and QuickTime RLE do carry alpha and are both NATIVE ffmpeg encoders, so they
# need nothing beyond a stock build.
#
# Do NOT reach for VP9/webm here: `-c:v libvpx-vp9 -pix_fmt yuva420p` encodes
# without error and the alpha is silently gone -- a decode round-trip comes
# back fully opaque. Verified 2026.08.07, ffmpeg 7.1.1.
#
# Text sharpness in the mp4 comes from two places, both handled here:
#   - render dpi (--dpi). At 100 dpi the tick labels are ~13 px tall and h264
#     has almost nothing to work with; 150 dpi gives it 20 px.
#   - constant-quality (-crf 18) instead of a fixed bitrate, so the encoder
#     spends bits on the thin glyph edges instead of holding a fixed rate.
# yuv420p is kept for player compatibility, and h264 needs even dimensions,
# hence the crop filter (same trick as make_vids.py).
EVEN = ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']
VFMT = {
    'mp4': dict(ext='.mp4',
                kw=dict(extra_args=EVEN + ['-crf', '18', '-pix_fmt', 'yuv420p']),
                save=dict(facecolor='white')),
    'prores': dict(ext='.mov',
                   kw=dict(codec='prores_ks',
                           extra_args=['-pix_fmt', 'yuva444p10le',
                                       '-profile:v', '4444', '-alpha_bits', '8']),
                   save=dict(transparent=True)),
    'qtrle': dict(ext='.mov',
                  kw=dict(codec='qtrle', extra_args=['-pix_fmt', 'argb']),
                  save=dict(transparent=True)),
}
VW = VFMT[args.vformat]['kw']
SAVE_KW = VFMT[args.vformat]['save']
if args.vformat != 'mp4' and not args.transparent:
    SAVE_KW = dict(facecolor='white')     # --no-transparent wins
if args.test:
    update(0)
    fn_out = out_dir / (stem + '_frame0.png')
    fig.savefig(fn_out, **still_kw)
    print('TEST: saved %s' % fn_out)
else:
    anim = animation.FuncAnimation(fig, update, frames=len(TT),
                                   interval=1000 / args.fps, blit=False)
    fn_out = out_dir / (stem + VFMT[args.vformat]['ext'])
    anim.save(fn_out, writer=animation.FFMpegWriter(fps=args.fps, **VW),
              savefig_kwargs=SAVE_KW, dpi=args.dpi)
    print('saved %s  (%s, %d dpi, %.1f MB)'
          % (fn_out, args.vformat, args.dpi, fn_out.stat().st_size / 1e6))
    update(0)
    fig.savefig(out_dir / (stem + '_frame0.png'), **still_kw)
    print('saved %s' % (out_dir / (stem + '_frame0.png')))
plt.close('all')
