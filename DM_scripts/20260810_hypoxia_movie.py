"""
Seasonal movie of BOTTOM hypoxia developing over Whidbey Basin, with the
hypoxic-area series it comes from running alongside it.

Layout (one figure, one frame per day) follows how many --series were asked
for:

  ONE (default, `pc`)   side by side -- the area series left, the map right,
                        with that polygon outlined in RED on the map so the
                        cells the curve counts are never in doubt.
  TWO (--series wb pc)  2x2 -- the first series over the second, map spanning
                        the right. The second is drawn as a PERCENT of its own
                        floor and boxed on the map, because 10 km2 means one
                        thing in a 620 km2 basin and quite another in a 12 km2
                        cove.

The series regions are INDEPENDENT of --region, which only sets the map window.
The default is deliberately a mismatch: the Penn Cove curve read against a map
of the whole northern basin, so the cove's hypoxia is seen in the context of
the water that ventilates it.

WINDOW. Same as 20260810_wbnorth_velocity_movie.py, and taken from it: the
--zoom-poly box extended east by --zoom of its own widths (the cove opens east
into Saratoga Passage), land filled grey, water that is not drawn left white,
--exclude cut out. The two movies can therefore be played side by side over the
same water. --zoom 0 falls back to the whole --region window.

WHY BOTTOM AREA. The bottom cell is what a benthic organism sits in and what a
grab survey or a bottom mooring reports, and it is the quantity whose seasonal
march is the thing being asked about. It is NOT the same as the hypoxic
footprint (any level in the column below the threshold) or the hypoxic volume;
20260806_hypoxia_reduce.py carries all three and its docstring says why they
separate. This script deliberately animates the one that has a map.

NESTED BANDS, NOT ONE LINE. A_bot(<5) >= A_bot(<3) >= A_bot(<2) by
construction, so the three curves stack without any arithmetic and the picture
reads as a single field draining downward: the 5 mg/L band fills first and from
farther away, then 3, then the 2 mg/L core appears in the deep holes. One
threshold alone would make the onset look like a step change, when what
actually happens is a slide.

SOURCE. lowpassed.nc -- one Godin-filtered field per day. For a May-to-November
question that is the right series: the tide is removed rather than aliased into
it, and one file per day makes ~214 frames of a season. Frames are whatever
lowpassed.nc files actually exist in the window, so a run with gaps animates
its gaps rather than failing.

The area series in the left panels is computed from the SAME files being shown
on the right, not read from the hypoxia_*.p reduction, so the marker cannot
drift away from the map. (It should agree with that pickle's A_bot_* columns to
roundoff; if it does not, one of the two masks is wrong.)

COLOUR SCALE. Discrete classes, not a continuous ramp, and the class edges ARE
the band thresholds -- so a colour on the map and a band in the left panels are
the same statement, and the boundary between two colours is a real line in the
water rather than a place where the eye happens to see one.

cmocean's cm.oxy is the obvious choice for this variable and is kept behind
--cmap oxy, but it is the wrong tool HERE: it spends its middle on a
low-contrast grey by design, and in Whidbey Basin nearly the whole sea floor
sits in that grey between 2 and 5 mg/L. The entire seasonal signal would be
rendered in the part of the colour table built to be unreadable.

Region-plot convention (DM 2026.08.04): with --zoom 0 the window is the
rectangular bounding box of the region polygon plus --pad-cells of margin, and
in either case only cells inside `wb` are drawn.

Runs wherever the lowpassed files are (apogee for the 2024-25 run; the 2017
wb1_r0_xn11b days on the Mac make a fine test bed).

    python 20260810_hypoxia_movie.py
    python 20260810_hypoxia_movie.py --test            # one still, fast
    python 20260810_hypoxia_movie.py --zoom 1.0        # further up Saratoga
    python 20260810_hypoxia_movie.py --series wb pc --zoom 0 --region wb \
        --exclude ''                                   # whole-basin version
    python 20260810_hypoxia_movie.py --stride 3        # every third day
    python 20260810_hypoxia_movie.py --gtx wb1_r0_xn11b --ds0 2017.08.01 \
        --ds1 2017.12.05                               # local test bed
"""
import argparse
import multiprocessing as mp
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.animation as animation
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage
from matplotlib.path import Path as MplPath
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from cmocean import cm

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

# house style, same as the pc_mouth analysis figures and the wb_north movies.
# Grid on the timeseries, never on the map -- DM 2026.08.07.
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
RED = '#e04256'
GREY = '0.45'
# Land fill and map furniture, taken from 20260807_grid_bathy_ppt.py (the wb1
# grid map) rather than the velocity movie's neutral '0.85' -- one warm land
# colour across the wb1 figures. Land is FILLED rather than left transparent: a
# see-through landmask reads as whatever is behind it, which kills the
# coastline. White still means "water not drawn" (outside wb, or inside
# --exclude); the warm fill is land.
LAND = '#e8e4dc'
COAST = 'k'

# DO classes, low to high: dark red = anoxic, red = hypoxic, warm = low-DO,
# cool = oxygenated. Warm and cool are mixed on purpose -- this is a
# classification, not a magnitude, and the warm/cool break lands on 5 mg/L,
# the low-DO line. One colour per class plus one for everything above the top
# edge.
LEVEL_COLORS = ['#4a0f16', '#b3202e', '#e8873c', '#f6c667', '#a9cfd4', '#5d9fc0']
OVER_COLOR = '#2f6b96'

DO_MMOL_TO_MGL = 32.0 / 1000.0        # mmol m-3 (uM) -> mg L-1

# ---- arguments -------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument('--gtx', default='wb1_t0_xn11abbur00')
p.add_argument('--ds0', default='2025.05.01')
p.add_argument('--ds1', default='2025.11.30')
p.add_argument('--stride', default=1, type=int,
               help='keep every Nth day (1 = every lowpassed file)')
p.add_argument('--region', default='wb_north',
               help='polygon the map window is taken from')
p.add_argument('--exclude', default='skagit_delta',
               help='comma-separated polygons to cut out of the map, and to '
                    'zoom past; empty string keeps the whole region')
p.add_argument('--zoom', default=0.6, type=float,
               help='window = the --zoom-poly box extended EAST by this many '
                    'of its widths, so 1.0 reaches about halfway up Saratoga '
                    'Passage. 0 or less falls back to the whole --region '
                    'window. Default matches the wbnorth velocity movie.')
p.add_argument('--zoom-poly', default='pc', dest='zoom_poly',
               help='polygon the --zoom window is built around')
p.add_argument('--aa', default='',
               help='explicit lon0,lon1,lat0,lat1 window; overrides --zoom')
p.add_argument('--series', default=['pc'], nargs='+',
               help='polygon(s) the area series are computed for, 1 or 2. One '
                    'gives a side-by-side figure and that polygon outlined on '
                    'the map; two stacks them, the second as a percent of its '
                    'own floor and boxed on the map. Need not be --region -- '
                    'the default is the basin window with the cove series.')
p.add_argument('--thresh', default=[2.0, 3.0, 5.0], type=float, nargs='+',
               help='mg/L bands, low to high; the lowest is the one contoured '
                    'on the map and named in the title')
p.add_argument('--cmap', default='levels', choices=['levels', 'oxy'],
               help='levels = discrete DO classes cut on --levels (default); '
                    'oxy = the continuous cmocean oxygen map on --vmin/--vmax')
p.add_argument('--levels', default=[0.0, 0.5, 2.0, 3.0, 5.0, 7.0, 10.0],
               type=float, nargs='+',
               help='mg/L class edges for --cmap levels; anything above the '
                    'last one is the over-colour')
p.add_argument('--vmin', default=0.0, type=float, help='--cmap oxy only')
p.add_argument('--vmax', default=10.0, type=float, help='--cmap oxy only')
p.add_argument('--no-contour', dest='contour', action='store_false',
               help='drop the threshold edge on the map')
p.add_argument('--pct', default='auto', choices=['auto', 'on', 'off'],
               help='units on the top (or only) series panel: auto = percent '
                    'of the floor when the map is zoomed to that one region, '
                    'km2 otherwise')
p.add_argument('--pad-cells', default=10, type=int)
p.add_argument('--fps', default=8, type=int)
p.add_argument('--dpi', default=150, type=int,
               help='render dpi for the movie frames -- 100 makes the labels '
                    'mushy once h264 has had a go at them')
p.add_argument('--transparent', action='store_true',
               help='transparent background on the saved stills (off by '
                    'default; the movie is always opaque)')
p.add_argument('--vformat', default='mp4', choices=['mp4', 'prores', 'qtrle'],
               help='mp4 = h264 on white (small, plays anywhere); prores/qtrle '
                    '= .mov with a real alpha channel, for compositing')
p.add_argument('--ro', default=None, type=int,
               help='force a roms_out key (2 = /dat2/dakotamm on apogee); '
                    'default searches roms_out2, roms_out1, roms_out')
p.add_argument('--nproc', default=min(8, os.cpu_count() or 1), type=int,
               help='parallel workers for reading lowpassed files (1 = serial)')
p.add_argument('--test', dest='test', action='store_true',
               help='save a single still of the PEAK frame instead of the movie')
args = p.parse_args()

gridname, tag, ex_name = args.gtx.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)

for label, dsx in [('--ds0', args.ds0), ('--ds1', args.ds1)]:
    try:
        datetime.strptime(dsx, '%Y.%m.%d')
    except ValueError:
        raise SystemExit('Invalid %s value %r -- use YYYY.MM.DD with a real '
                         'calendar day.' % (label, dsx))

THRESH = sorted(args.thresh)                 # low to high
T0 = THRESH[0]                               # the hypoxia line for this run
out_dir = Ldir['LOo'] / 'DM_outs' / '20260810_hypoxia_movie'
Lfun.make_dir(out_dir)

# ---- files -----------------------------------------------------------------
# Same search as 20260806_hypoxia_reduce.py rather than Lfun.get_fn_list: the
# run lives under whichever roms_out key is populated on this machine, and a
# missing day should be skipped, not crash the movie.
ROMS_OUT_KEYS = (['roms_out%d' % args.ro] if args.ro is not None
                 else ['roms_out2', 'roms_out1', 'roms_out'])


def find_files():
    """{'YYYY.MM.DD': path to lowpassed.nc}; the first roms_out key wins."""
    days = {}
    for key in ROMS_OUT_KEYS:
        base = Ldir.get(key)
        if base is None or str(base).endswith('BLANK'):
            continue
        base = Path(base) / args.gtx
        if not base.is_dir():
            continue
        for run_dir in sorted(base.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith('f'):
                continue
            ds_ = run_dir.name[1:]
            if not (args.ds0 <= ds_ <= args.ds1) or ds_ in days:
                continue
            fn = run_dir / 'lowpassed.nc'
            if fn.is_file():
                days[ds_] = fn
    return dict(sorted(days.items()))


FN = find_files()
if not FN:
    raise SystemExit(
        'no lowpassed.nc for %s between %s and %s under\n  %s'
        % (args.gtx, args.ds0, args.ds1,
           '\n  '.join(str(Ldir.get(k)) for k in ROMS_OUT_KEYS)))
fn_list = list(FN.values())[::max(1, args.stride)]
print('%d daily frames %s .. %s (stride %d, %d files found)'
      % (len(fn_list), args.ds0, args.ds1, args.stride, len(FN)))

# ---- grid, polygons, window ------------------------------------------------
dsg = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon = dsg.lon_rho.values
lat = dsg.lat_rho.values
wet = dsg.mask_rho.values == 1
h = dsg.h.values
cell_area = (1 / dsg.pm.values) * (1 / dsg.pn.values)      # m2, cell by cell
dsg.close()
dx = float(np.diff(lon[0, :]).mean())
dy = float(np.diff(lat[:, 0]).mean())
XY = np.column_stack([lon.ravel(), lat.ravel()])

sect_dir = Ldir['LOo'] / 'section_lines'


def load_sect(name):
    """A section_lines polygon, x and y forced to float.

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
    return pth.contains_points(XY).reshape(lon.shape)


reg = load_sect(args.region)
in_wb = poly_mask(load_sect('wb'))            # master clip, every wb1 region plot

# The series regions. These are the POLYGONS intersected with water, not the wb
# clip and not the map window -- the area series has to mean the same thing as
# the pickle's A_bot_*, which is per polygon. They are deliberately NOT tied to
# --region: the map can show the basin a cove sits in while the curve counts
# only the cove.
SER = list(dict.fromkeys(args.series))        # de-dupe, keep order
if not 1 <= len(SER) <= 2:
    raise SystemExit('--series takes 1 or 2 polygons, got %d: %s'
                     % (len(SER), SER))
SER0 = SER[0]
SINGLE = len(SER) == 1
RMASK = {nm: poly_mask(load_sect(nm)) & wet for nm in SER}
IDX = {nm: np.where(RMASK[nm]) for nm in SER}          # index once, reuse daily
RAREA = {nm: float(cell_area[RMASK[nm]].sum()) for nm in SER}
for nm in SER:
    m = RMASK[nm]
    print('  region %-14s %6d cells, %8.1f km2, mean h %5.1f m, max h %5.1f m'
          % (nm, m.sum(), RAREA[nm] / 1e6, h[m].mean(), h[m].max()))

# ---- map window ------------------------------------------------------------
# Lifted from 20260810_wbnorth_velocity_movie.py so the two movies frame the
# same water and can be played side by side. Everything here is about what is
# DRAWN; the area series are per polygon and are untouched by it.
#
# Cells to draw: inside wb, minus anything in --exclude. The standing region
# convention is "extent from the region polygon, content clipped to wb"; this
# adds a subtraction, so the window is taken from the cells that SURVIVE rather
# than from the region outline -- otherwise dropping the Skagit delta would
# leave a big empty rectangle where it used to be.
keep = wet & in_wb
excl = [s for s in args.exclude.split(',') if s]
for nm in excl:
    keep &= ~poly_mask(load_sect(nm))
in_win = keep & poly_mask(reg)
if not in_win.any():
    raise SystemExit('nothing left after excluding %s from %s'
                     % (excl, args.region))
# The window comes from the LARGEST CONNECTED PIECE of what survives, not from
# its outright min/max. Cutting a polygon out of another one leaves stray cells
# behind -- with skagit_delta removed from wb_north exactly ONE cell survives up
# by Deception Pass, and letting it set the northern edge stretched the map to
# 144 rows instead of 84 for a single cell. Drawing still uses the full `keep`.
lab, nlab = ndimage.label(in_win)
if nlab > 1:
    sizes = ndimage.sum(in_win, lab, range(1, nlab + 1))
    main = lab == (1 + int(np.argmax(sizes)))
    print('  %d disconnected pieces; window taken from the largest (%d cells, '
          '%d stray)' % (nlab, main.sum(), in_win.sum() - main.sum()))
else:
    main = in_win
aa = [lon[main].min() - args.pad_cells * dx,
      lon[main].max() + args.pad_cells * dx,
      lat[main].min() - args.pad_cells * dy,
      lat[main].max() + args.pad_cells * dy]

# Penn Cove zoom, built from the polygon box rather than hard-coded degrees so
# it follows the polygon if it is redrawn. The cove opens EAST into Saratoga
# Passage, so the window is extended asymmetrically: --zoom widths to the east,
# and only a little west, where the cove already ends at its head.
if args.aa:
    aa = [float(s) for s in args.aa.split(',')]
    print('window: explicit --aa')
elif args.zoom > 0:
    zp = load_sect(args.zoom_poly)
    zw = float(zp.x.max() - zp.x.min())
    zh = float(zp.y.max() - zp.y.min())
    aa = [zp.x.min() - 0.10 * zw, zp.x.max() + args.zoom * zw,
          zp.y.min() - 0.60 * zh, zp.y.max() + 0.90 * zh]
    print('window: %s box + %.2f of its widths east (velocity-movie zoom)'
          % (args.zoom_poly, args.zoom))
if excl:
    print('excluding %s from %s -> %d cells kept (was %d)'
          % (', '.join(excl), args.region, in_win.sum(),
             (wet & in_wb & poly_mask(reg)).sum()))

jj = np.where((lat[:, 0] >= aa[2]) & (lat[:, 0] <= aa[3]))[0]
ii = np.where((lon[0, :] >= aa[0]) & (lon[0, :] <= aa[1]))[0]
SUB = (slice(int(jj[0]), int(jj[-1]) + 1), slice(int(ii[0]), int(ii[-1]) + 1))
lon_s, lat_s = lon[SUB], lat[SUB]
plon_s, plat_s = pfun.get_plon_plat(lon_s, lat_s)
draw_s = keep[SUB]                            # wb-clipped, minus --exclude
land_s = (~wet)[SUB]                          # true land, filled grey
print('window %s -> %d x %d cells, %d drawn'
      % (['%.4f' % v for v in aa], lat_s.shape[0], lat_s.shape[1], draw_s.sum()))

# the second series region gets a dotted box on the map; the single-series case
# gets its polygon outlined instead (see the map block)
if not SINGLE:
    p2 = load_sect(SER[1])
    box = [float(p2.x.min()), float(p2.x.max()),
           float(p2.y.min()), float(p2.y.max())]

# ---- read one lowpassed file -----------------------------------------------
def read_one(fn):
    """(time, bottom DO in the window, per-region series row).

    s_rho index 0 is the BOTTOM cell in ROMS, so oxygen[0] is the bed and
    oxygen[-1] the surface -- the same convention 20260806_hypoxia_reduce.py
    uses. Only that one level is pulled off disk.

    Land and fill values are set to NaN before anything is compared to a
    threshold, so a masked cell can never be counted as hypoxic (a fill of 0
    would otherwise be the most hypoxic water in the domain).
    """
    ds = xr.open_dataset(fn)
    if 'oxygen' not in ds.data_vars:
        ds.close()
        raise SystemExit('no oxygen in %s -- this movie needs the bio fields' % fn)
    tname = 'ocean_time' if 'ocean_time' in ds.variables else 'time'
    t = pd.Timestamp(np.atleast_1d(ds[tname].values)[-1])
    o = ds['oxygen']
    if 's_rho' in o.dims:
        o = o.isel(s_rho=0)
    bot = np.atleast_3d(o.values)[-1] if o.values.ndim == 3 else o.values
    ds.close()
    bot = np.where(wet & (bot > 0), bot * DO_MMOL_TO_MGL, np.nan)

    row = {}
    for nm in SER:
        i, j = IDX[nm]
        v, a_ = bot[i, j], cell_area[i, j]
        r = {'do_mean': float(np.nanmean(v)), 'do_min': float(np.nanmin(v))}
        for th in THRESH:
            r['A_%g' % th] = float(a_[v < th].sum())     # NaN < th is False
        row[nm] = r
    return t, bot[SUB].astype(np.float32), row


nproc = max(1, min(args.nproc, len(fn_list)))
print('reading %d files on %d process(es)...' % (len(fn_list), nproc))
if nproc > 1:
    ctx = mp.get_context('fork')              # workers inherit the masks above
    with ctx.Pool(nproc) as pool:
        res = pool.map(read_one, fn_list)
else:
    res = [read_one(fn) for fn in fn_list]

TT = pd.to_datetime([r[0] for r in res])
FLD = np.where(draw_s[None, :, :], np.stack([r[1] for r in res]), np.nan)
S = {nm: pd.DataFrame([r[2][nm] for r in res], index=TT) for nm in SER}

# ---- what the series actually does, before anything is drawn ---------------
print('\nbottom DO in the window: min %.2f, median %.2f, max %.2f mg/L'
      % (np.nanmin(FLD), np.nanmedian(FLD), np.nanmax(FLD)))
print('\n%-14s %9s   %s' % ('region', 'floor km2',
                            '  '.join('%18s' % ('peak A_bot < %g mg/L' % t)
                                      for t in THRESH)))
for nm in SER:
    d = S[nm]
    cells = []
    for th in THRESH:
        a = d['A_%g' % th] / 1e6
        cells.append('%7.2f km2 %3.0f%% ' % (a.max(), 100 * a.max() / (RAREA[nm] / 1e6)))
    print('%-14s %9.1f   %s' % (nm, RAREA[nm] / 1e6, ' '.join(cells)))
for nm in SER:
    a = S[nm]['A_%g' % T0]
    if a.max() <= 0:
        print('%-14s never goes below %g mg/L at the bed in this window -- the '
              'bands above %g are the ones carrying the signal' % (nm, T0, T0))
    else:
        print('%-14s <%g mg/L: first %s, peak %s, last %s'
              % (nm, T0, a[a > 0].index[0].date(), a.idxmax().date(),
                 a[a > 0].index[-1].date()))

# ---- figure ----------------------------------------------------------------
# constrained layout, not manual margins: the map carries a fixed aspect (dar),
# so it shrinks inside its own gridspec cell and a hand-placed colorbar ends up
# stranded out in the whitespace.
#
# Two shapes, picked by how many series were asked for:
#   --series pc      (default)  1x2 side by side: one series, one map. A single
#                               panel stacked over an empty half is just a tall
#                               thin map and a lot of white.
#   --series wb pc              2x2: basin series over cove series, map
#                               spanning the right -- the cove read against the
#                               basin it sits in.
plt.close('all')
if SINGLE:
    # the series gets the wider half: a seven-month record read sideways in a
    # tall narrow panel is unreadable, and a cove zoom is a short wide map that
    # does not need the room
    fig = plt.figure(figsize=(16, 6.0), layout='constrained')
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = None
    axm = fig.add_subplot(gs[0, 1])
else:
    fig = plt.figure(figsize=(17.5, 9), layout='constrained')
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.05])
    ax1 = fig.add_subplot(gs[0, 0])              # region area series
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)  # pc area series
    axm = fig.add_subplot(gs[:, 1])              # map

# Units on the top/only panel. A percent needs a denominator the reader already
# has in mind. With one series the panel and the outline on the map are the
# same object, so "% of that floor" reads straight off the figure; with two,
# the top one is the basin and km2 keeps it comparable to the cove's absolute
# area. --pct forces either way, and the percent panels carry km2 on the right.
PCT1 = (args.pct == 'on') or (args.pct == 'auto' and SINGLE)

# --- map
# Discrete classes by default. spacing='proportional' keeps the colourbar a
# ruler in mg/L -- 'uniform' would give the 0-0.5 anoxic sliver the same height
# as the 5-7 class and quietly exaggerate it.
LEVELS = sorted(args.levels)
if args.cmap == 'levels':
    if len(LEVELS) - 1 > len(LEVEL_COLORS):
        raise SystemExit('--levels asks for %d classes; only %d colours are '
                         'defined' % (len(LEVELS) - 1, len(LEVEL_COLORS)))
    cmap = ListedColormap(LEVEL_COLORS[:len(LEVELS) - 1])
    cmap.set_over(OVER_COLOR)
    norm = BoundaryNorm(LEVELS, cmap.N)
    cb_kw = dict(extend='max', spacing='proportional', ticks=LEVELS)
    lo, hi = LEVELS[0], LEVELS[-1]
else:
    cmap, norm = cm.oxy, Normalize(vmin=args.vmin, vmax=args.vmax)
    lo, hi = args.vmin, args.vmax
    cb_kw = dict(ticks=sorted(set([lo] + list(THRESH) + [hi])))
# land under everything else. Only true land (mask_rho == 0) is filled -- water
# that is simply not drawn (outside wb, or inside --exclude) stays white, so
# the two are never confused.
axm.pcolormesh(plon_s, plat_s,
               np.ma.masked_where(~land_s, np.ones(land_s.shape)),
               cmap=ListedColormap([LAND]), shading='flat', zorder=0)
cs = axm.pcolormesh(plon_s, plat_s, FLD[0], cmap=cmap, norm=norm,
                    shading='flat', zorder=1)
f_lo = float(np.nanmean(FLD < lo))
f_hi = float(np.nanmean(FLD > hi))
cb_kw.setdefault('extend', 'both' if (f_lo > 0 and f_hi > 0) else
                 'max' if f_hi > 0 else 'min' if f_lo > 0 else 'neither')
cb = fig.colorbar(cs, ax=axm, shrink=0.75, pad=0.01, aspect=35,
                  label='bottom-cell dissolved oxygen [mg L$^{-1}$]', **cb_kw)
print('colour scale %s, %.1f to %.1f mg/L; %.1f%% of cell-days above the top '
      'class, %.1f%% below the bottom' % (args.cmap, lo, hi,
                                          100 * f_hi, 100 * f_lo))

pfun.add_coast(axm, color=COAST, linewidth=0.8)
if SINGLE:
    # one series: draw its POLYGON, not a bounding box. The outline says
    # exactly which cells the curve counts, which matters most here -- the map
    # window is the whole basin and most of the hypoxic water on screen is NOT
    # in the series. A box would claim a rectangle of coast that isn't the
    # region.
    s0 = load_sect(SER0)
    axm.plot(np.append(s0.x.values, s0.x.values[0]),
             np.append(s0.y.values, s0.y.values[0]), '-',
             color=RED, lw=2.0, zorder=8)
else:
    pfun.draw_box(axm, box, color=GREY, linewidth=1.5, linestyle=':')
# Ticks and labels as on the wb1 grid map: evenly spaced round values, upright,
# and the axes named with their units. set_xticks/set_yticks re-autoscale, and a
# rounded tick outside the window then drags the view past its edge -- so the
# limits are pinned AFTERWARDS. Decimals follow the span: the grid map's one
# decimal collapses to a single repeated value across a 0.16 deg cove zoom.
DEC = 1 if (aa[1] - aa[0]) >= 1.0 else 2


def nice_ticks(v0, v1, n, dec):
    """n evenly spaced round values, both ends INSIDE [v0, v1].

    Rounding the endpoints outward (which plain linspace+round does) puts the
    first and last tick past the window, and pinning the limits then simply
    drops them -- three ticks on a four-tick axis. Rounding inward first keeps
    all n.
    """
    q = 10.0 ** dec
    return np.linspace(np.ceil(v0 * q) / q, np.floor(v1 * q) / q, n).round(dec)


axm.set_xticks(nice_ticks(aa[0], aa[1], 4, DEC))
axm.set_yticks(nice_ticks(aa[2], aa[3], 5, DEC))
axm.axis(aa)
axm.set_autoscale_on(False)
pfun.dar(axm)
axm.tick_params(length=6, labelrotation=0)
axm.set_xlabel('Longitude [$^{\\circ}$E]')
axm.set_ylabel('Latitude [$^{\\circ}$N]')
for s in axm.spines.values():
    s.set_visible(True)
ttl = axm.set_title('', fontsize=13)

# The hypoxic edge. Contoured rather than hatched so it does not compete with
# the colour it is drawn on; it is redrawn each frame because the patch moves.
#
# Skipped when the threshold is already a class edge (the default case): the
# colour boundary IS that line, and drawing it again just thickens it.
CT = {'obj': None}
EDGE = args.contour and not (args.cmap == 'levels' and
                             any(abs(L - T0) < 1e-9 for L in LEVELS))
print('threshold %g mg/L: %s' % (T0, 'contoured on the map' if EDGE else
                                 'already a colour-class edge, not contoured'))


def draw_edge(fi):
    if not EDGE:
        return
    if CT['obj'] is not None:
        CT['obj'].remove()
        CT['obj'] = None
    f = FLD[fi]
    if np.isfinite(f).any() and np.nanmin(f) < T0:
        CT['obj'] = axm.contour(lon_s, lat_s, f, levels=[T0], colors='k',
                                linewidths=1.2, zorder=5)


# --- the two series panels
# Bands are nested by construction (A(<5) >= A(<3) >= A(<2)), so they are
# simply drawn from the widest to the narrowest and each one covers the last.
#
# Each band takes the map's colour for the water it is counting: the "< 3"
# band is the colour of 2-3 mg/L water on the map, because that is the water
# the band adds to the one inside it. So the stack of bands in these panels is
# the map's colourbar, stood on end -- no second legend to learn.
def band_colour(th):
    """Map colour for the class immediately below threshold th."""
    if args.cmap != 'levels':
        # no classes to borrow from: fall back on the same ramp, darkest at
        # the lowest threshold
        return LEVEL_COLORS[min(THRESH.index(th), len(LEVEL_COLORS) - 1)]
    k = int(np.searchsorted(LEVELS, th, side='left')) - 1
    return LEVEL_COLORS[min(max(k, 0), len(LEVELS) - 2)]


def gapped(d):
    """Break the series wherever days are missing.

    fill_between joins whatever points it is given, so a run with a hole in it
    draws a straight ramp across the hole -- weeks of invented, monotonic
    hypoxia. A NaN row dropped into the middle of each gap makes the hole a
    hole. (The 2024-25 run is continuous and this is a no-op there; the 2017
    test bed is nothing but gaps.)
    """
    if len(d) < 3:
        return d
    dt = d.index.to_series().diff()
    med = dt.median()
    hole = np.where(dt.values[1:] > 1.5 * med)[0] + 1
    if len(hole) == 0:
        return d
    mid = pd.DatetimeIndex([d.index[i - 1] + (d.index[i] - d.index[i - 1]) / 2
                            for i in hole])
    return pd.concat([d, pd.DataFrame(np.nan, index=mid,
                                      columns=d.columns)]).sort_index()


def series_panel(ax, nm, pct):
    """Nested hypoxic-area bands for one region; returns the marker artists."""
    d = gapped(S[nm])
    sc = 100.0 / RAREA[nm] if pct else 1e-6           # % of floor, or km2
    for k, th in enumerate(reversed(THRESH)):         # widest band first
        y = d['A_%g' % th] * sc
        ax.fill_between(d.index, 0, y, color=band_colour(th), lw=0,
                        label='< %g mg L$^{-1}$' % th, zorder=2 + k)
    ax.set_ylabel('bottom area %s' % ('[% of floor]' if pct else '[km$^2$]'))
    ax.set_ylim(0, None if not pct else 100)
    ax.set_xlim(TT[0], TT[-1])
    ax.grid(**GRID)
    if pct:
        # km2 on the right, because a percent answers "how much of the cove"
        # and only an area answers "how much habitat" -- both get asked
        km2 = RAREA[nm] / 1e6
        sax = ax.secondary_yaxis('right', functions=(lambda p: p * km2 / 100,
                                                     lambda a: a * 100 / km2))
        sax.set_ylabel('[km$^2$]')
    # the marker rides the UNgapped series, so its index still lines up frame
    # for frame with the map
    y = S[nm]['A_%g' % T0] * sc
    mark = ax.axvline(TT[0], color='k', lw=1.5, zorder=20)
    dot = ax.plot([TT[0]], [y.iloc[0]], 'o', ms=7, color='k', zorder=21)[0]
    return mark, dot, y


def panel1_title(ax):
    ax.set_title('bottom hypoxic area, %s (%s of sea floor%s)'
                 % (SER0,
                    ('%.1f km$^2$' if RAREA[SER0] < 5e7 else '%.0f km$^2$')
                    % (RAREA[SER0] / 1e6),
                    ', outlined in red on the map' if SINGLE else ''),
                 fontsize=11, color=RED if SINGLE else GREY)


m1, d1, y1 = series_panel(ax1, SER0, pct=PCT1)
panel1_title(ax1)
ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)

if ax2 is not None:
    plt.setp(ax1.get_xticklabels(), visible=False)
    m2, d2, y2 = series_panel(ax2, SER[1], pct=True)
    ax2.set_title('%s, as a percent of its own floor (%.1f km$^2$, dotted box '
                  'on the map)' % (SER[1], RAREA[SER[1]] / 1e6),
                  fontsize=11, color=RED)
    for sp in ax2.spines.values():                    # tie panel to the box
        sp.set_color(RED)
        sp.set_linewidth(1.6)
    ax2.tick_params(color=RED)
    tail = ax2
else:
    m2 = d2 = None
    tail = ax1

tail.xaxis.set_major_locator(mdates.MonthLocator())
tail.xaxis.set_major_formatter(mdates.DateFormatter('%b %-d'))
tail.set_xlabel('%d' % TT[0].year if TT[0].year == TT[-1].year else '')
for l in tail.get_xticklabels():
    l.set_rotation(30)
    l.set_horizontalalignment('right')

if args.transparent:
    fig.patch.set_alpha(0.0)
    for ax in [a for a in (axm, ax1, ax2) if a is not None]:
        ax.patch.set_alpha(0.0)


def update(fi):
    cs.set_array(FLD[fi].ravel())
    draw_edge(fi)
    # the number quoted is the SERIES region, not the map window -- it has to
    # be the same quantity the moving dot is sitting on, and at the cove zoom
    # there is hypoxic water on screen that the curve does not count
    a = S[SER0]['A_%g' % T0].iloc[fi]
    ttl.set_text('bottom oxygen -- %s\n%s below %g mg L$^{-1}$: %.1f km$^2$ '
                 '(%.0f%% of the %s floor)'
                 % (TT[fi].strftime('%Y-%m-%d'), SER0, T0, a / 1e6,
                    100 * a / RAREA[SER0], SER0))
    m1.set_xdata([TT[fi], TT[fi]])
    d1.set_data([TT[fi]], [y1.iloc[fi]])
    if m2 is not None:
        m2.set_xdata([TT[fi], TT[fi]])
        d2.set_data([TT[fi]], [y2.iloc[fi]])
    return []


# the series regions go in the name, not the map window: two runs over the same
# window with different series are different figures and must not overwrite
# each other, while the same series at a different zoom is the same figure
stem = ('20260810_hypoxia_%s_%s%s_%s_%s'
        % (args.gtx, '-'.join(SER), '_zoom%g' % args.zoom if args.zoom > 0 else
           '_%s' % args.region, args.ds0, args.ds1))
still_kw = dict(dpi=200, bbox_inches='tight', transparent=args.transparent)

# Video formats. h264 has no alpha channel, so mp4 is rendered on white -- a
# transparent figure piped to it just gets composited onto black. ProRes 4444
# and QuickTime RLE do carry alpha and are both NATIVE ffmpeg encoders.
# Do NOT reach for VP9/webm: it encodes without error and the alpha is silently
# gone (verified 2026.08.07, ffmpeg 7.1.1).
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

# The still is taken at the PEAK of the primary band, not at frame 0: frame 0
# of a May-to-November window is a fully oxygenated basin, which tells you the
# figure renders but nothing about whether the movie works.
peak = int(np.argmax(S[SER0]['A_%g' % T0].values))
if args.test:
    update(peak)
    fn_out = out_dir / (stem + '_peak.png')
    fig.savefig(fn_out, **still_kw)
    print('TEST: saved %s  (frame %d, %s)'
          % (fn_out, peak, TT[peak].strftime('%Y-%m-%d')))
else:
    anim = animation.FuncAnimation(fig, update, frames=len(TT),
                                   interval=1000 / args.fps, blit=False)
    fn_out = out_dir / (stem + VFMT[args.vformat]['ext'])
    anim.save(fn_out, writer=animation.FFMpegWriter(fps=args.fps, **VW),
              savefig_kwargs=SAVE_KW, dpi=args.dpi)
    print('saved %s  (%s, %d frames at %d fps = %.0f s, %.1f MB)'
          % (fn_out, args.vformat, len(TT), args.fps, len(TT) / args.fps,
             fn_out.stat().st_size / 1e6))
    for lbl, fi in [('frame0', 0), ('peak', peak)]:
        update(fi)
        fig.savefig(out_dir / ('%s_%s.png' % (stem, lbl)), **still_kw)
        print('saved %s' % (out_dir / ('%s_%s.png' % (stem, lbl))))

# ---- the series on its own ---------------------------------------------------
# The same two panels without the map, because the seasonal figure is wanted in
# talks and papers where a movie cannot go, and re-deriving it from the CSV
# later would mean re-deciding the bands and the colours.
fig2 = plt.figure(figsize=(9, 4) if SINGLE else (9, 7), layout='constrained')
gs2 = fig2.add_gridspec(1 if SINGLE else 2, 1)
bx1 = fig2.add_subplot(gs2[0, 0])
bx2 = None if SINGLE else fig2.add_subplot(gs2[1, 0], sharex=bx1)
series_panel(bx1, SER0, pct=PCT1)
panel1_title(bx1)
bx1.legend(loc='upper left', fontsize=9, framealpha=0.9)
if bx2 is not None:
    plt.setp(bx1.get_xticklabels(), visible=False)
    series_panel(bx2, SER[1], pct=True)
    bx2.set_title('%s, as a percent of its own floor (%.1f km$^2$)'
                  % (SER[1], RAREA[SER[1]] / 1e6),
                  fontsize=11, color=RED)
    tail2 = bx2
else:
    tail2 = bx1
tail2.xaxis.set_major_locator(mdates.MonthLocator())
tail2.xaxis.set_major_formatter(mdates.DateFormatter('%b %-d'))
tail2.set_xlabel('%d' % TT[0].year if TT[0].year == TT[-1].year else '')
for l in tail2.get_xticklabels():
    l.set_rotation(30)
    l.set_horizontalalignment('right')
# the markers series_panel leaves behind belong to the movie, not to a still
for ax in [a for a in (bx1, bx2) if a is not None]:
    for ln in list(ax.lines):
        ln.remove()
fig2.savefig(out_dir / (stem + '_series.png'), dpi=200, bbox_inches='tight')
print('saved %s' % (out_dir / (stem + '_series.png')))

# the series behind the movie, so a number can be quoted without re-running it
csv_fn = out_dir / (stem + '_area.csv')
pd.concat({nm: S[nm] for nm in SER}, axis=1).to_csv(csv_fn)
print('saved %s' % csv_fn)
