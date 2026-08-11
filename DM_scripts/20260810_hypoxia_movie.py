"""
Seasonal movie of BOTTOM hypoxia developing over Whidbey Basin, with the
hypoxic-area series it comes from running alongside it.

Layout (one figure, one frame per day):
  right       -- map of BOTTOM-CELL oxygen, with the hypoxic (<2 mg/L by
                 default) edge drawn on top and the `pc` polygon boxed
  upper left  -- bottom hypoxic AREA over the map region, as nested filled
                 bands (<5, <3, <2 mg/L), carrying a marker for where the
                 animation is
  lower left  -- the same thing for Penn Cove, as a PERCENT of the cove floor,
                 because 10 km2 means one thing in a 620 km2 basin and quite
                 another in a 12 km2 cove

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

Region-plot convention (DM 2026.08.04): the window is the rectangular bounding
box of the WHOLE region polygon plus --pad-cells of margin, but only cells
inside `wb` are drawn.

Runs wherever the lowpassed files are (apogee for the 2024-25 run; the 2017
wb1_r0_xn11b days on the Mac make a fine test bed).

    python 20260810_hypoxia_movie.py
    python 20260810_hypoxia_movie.py --test           # one still, fast
    python 20260810_hypoxia_movie.py --region wb_north
    python 20260810_hypoxia_movie.py --stride 3       # every third day
    python 20260810_hypoxia_movie.py --gtx wb1_r0_xn11b --ds0 2017.08.01 \
        --ds1 2017.12.05                              # local test bed
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
from matplotlib.path import Path as MplPath
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.ticker import MaxNLocator
from cmocean import cm

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

# house style, same as the pc_mouth analysis figures and the wb_north movies.
# Grid on the timeseries, never on the map -- DM 2026.08.07.
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
RED = '#e04256'
GREY = '0.45'

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
p.add_argument('--region', default='wb',
               help='polygon that sets the map window and the top-left series')
p.add_argument('--pc-poly', default='pc', dest='pc_poly',
               help='polygon boxed on the map and used for the lower-left '
                    'series; same as --region drops that panel')
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
p.add_argument('--pad-cells', default=6, type=int)
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
pc = load_sect(args.pc_poly)
in_wb = poly_mask(load_sect('wb'))            # master clip, every wb1 region plot

# The two series regions. These are the POLYGONS intersected with water, not
# the wb clip and not the map window -- the area series has to mean the same
# thing as the pickle's A_bot_*, which is per polygon.
SER = [args.region] + ([args.pc_poly] if args.pc_poly != args.region else [])
RMASK = {nm: poly_mask(load_sect(nm)) & wet for nm in SER}
IDX = {nm: np.where(RMASK[nm]) for nm in SER}          # index once, reuse daily
RAREA = {nm: float(cell_area[RMASK[nm]].sum()) for nm in SER}
for nm in SER:
    m = RMASK[nm]
    print('  region %-14s %6d cells, %8.1f km2, mean h %5.1f m, max h %5.1f m'
          % (nm, m.sum(), RAREA[nm] / 1e6, h[m].mean(), h[m].max()))

# rectangular window around the WHOLE region polygon + a margin of cells
aa = [reg.x.min() - args.pad_cells * dx, reg.x.max() + args.pad_cells * dx,
      reg.y.min() - args.pad_cells * dy, reg.y.max() + args.pad_cells * dy]
jj = np.where((lat[:, 0] >= aa[2]) & (lat[:, 0] <= aa[3]))[0]
ii = np.where((lon[0, :] >= aa[0]) & (lon[0, :] <= aa[1]))[0]
SUB = (slice(int(jj[0]), int(jj[-1]) + 1), slice(int(ii[0]), int(ii[-1]) + 1))
lon_s, lat_s = lon[SUB], lat[SUB]
plon_s, plat_s = pfun.get_plon_plat(lon_s, lat_s)
draw_s = (wet & in_wb)[SUB]                   # rectangular extent, wb-clipped
print('window %s -> %d x %d cells, %d drawn'
      % (['%.4f' % v for v in aa], lat_s.shape[0], lat_s.shape[1], draw_s.sum()))

box = [float(pc.x.min()), float(pc.x.max()),
       float(pc.y.min()), float(pc.y.max())]

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
plt.close('all')
fig = plt.figure(figsize=(17.5, 9), layout='constrained')
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.05])
ax1 = fig.add_subplot(gs[0, 0])                  # region area series
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)      # pc area series
axm = fig.add_subplot(gs[:, 1])                  # map

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

pfun.add_coast(axm, color='gray', linewidth=0.5)
pfun.draw_box(axm, box, color=GREY, linewidth=1.5, linestyle=':')
axm.axis(aa)
pfun.dar(axm)
axm.xaxis.set_major_locator(MaxNLocator(nbins=4))
axm.tick_params(axis='x', labelrotation=45, labelsize=9)
axm.set_xlabel('Longitude')
axm.set_ylabel('Latitude')
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
    # the marker rides the UNgapped series, so its index still lines up frame
    # for frame with the map
    y = S[nm]['A_%g' % T0] * sc
    mark = ax.axvline(TT[0], color='k', lw=1.5, zorder=20)
    dot = ax.plot([TT[0]], [y.iloc[0]], 'o', ms=7, color='k', zorder=21)[0]
    return mark, dot, y


m1, d1, y1 = series_panel(ax1, args.region, pct=False)
ax1.set_title('bottom hypoxic area, %s (%.0f km$^2$ of sea floor)'
              % (args.region, RAREA[args.region] / 1e6), fontsize=11, color=GREY)
ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
plt.setp(ax1.get_xticklabels(), visible=False)

if args.pc_poly != args.region:
    m2, d2, y2 = series_panel(ax2, args.pc_poly, pct=True)
    ax2.set_title('%s, as a percent of its own floor (%.1f km$^2$, dotted box '
                  'on the map)' % (args.pc_poly, RAREA[args.pc_poly] / 1e6),
                  fontsize=11, color=RED)
    for sp in ax2.spines.values():                    # tie panel to the box
        sp.set_color(RED)
        sp.set_linewidth(1.6)
    ax2.tick_params(color=RED)
    tail = ax2
else:
    ax2.axis('off')
    m2 = d2 = None
    tail = ax1
    plt.setp(ax1.get_xticklabels(), visible=True)

tail.xaxis.set_major_locator(mdates.MonthLocator())
tail.xaxis.set_major_formatter(mdates.DateFormatter('%b %-d'))
tail.set_xlabel('%d' % TT[0].year if TT[0].year == TT[-1].year else '')
for l in tail.get_xticklabels():
    l.set_rotation(30)
    l.set_horizontalalignment('right')

if args.transparent:
    fig.patch.set_alpha(0.0)
    for ax in [axm, ax1, ax2]:
        ax.patch.set_alpha(0.0)


def update(fi):
    cs.set_array(FLD[fi].ravel())
    draw_edge(fi)
    a = S[args.region]['A_%g' % T0].iloc[fi]
    ttl.set_text('bottom oxygen -- %s\n%s below %g mg L$^{-1}$: %.1f km$^2$ '
                 '(%.0f%% of the %s floor)'
                 % (TT[fi].strftime('%Y-%m-%d'), args.region, T0, a / 1e6,
                    100 * a / RAREA[args.region], args.region))
    m1.set_xdata([TT[fi], TT[fi]])
    d1.set_data([TT[fi]], [y1.iloc[fi]])
    if m2 is not None:
        m2.set_xdata([TT[fi], TT[fi]])
        d2.set_data([TT[fi]], [y2.iloc[fi]])
    return []


stem = ('20260810_hypoxia_%s_%s_%s_%s'
        % (args.gtx, args.region, args.ds0, args.ds1))
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
peak = int(np.argmax(S[args.region]['A_%g' % T0].values))
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
fig2 = plt.figure(figsize=(9, 7), layout='constrained')
gs2 = fig2.add_gridspec(2, 1)
bx1 = fig2.add_subplot(gs2[0, 0])
bx2 = fig2.add_subplot(gs2[1, 0], sharex=bx1)
series_panel(bx1, args.region, pct=False)
bx1.set_title('bottom hypoxic area, %s (%.0f km$^2$ of sea floor)'
              % (args.region, RAREA[args.region] / 1e6), fontsize=11, color=GREY)
bx1.legend(loc='upper left', fontsize=9, framealpha=0.9)
plt.setp(bx1.get_xticklabels(), visible=False)
if args.pc_poly != args.region:
    series_panel(bx2, args.pc_poly, pct=True)
    bx2.set_title('%s, as a percent of its own floor (%.1f km$^2$)'
                  % (args.pc_poly, RAREA[args.pc_poly] / 1e6),
                  fontsize=11, color=RED)
    tail2 = bx2
else:
    bx2.axis('off')
    tail2 = bx1
    plt.setp(bx1.get_xticklabels(), visible=True)
tail2.xaxis.set_major_locator(mdates.MonthLocator())
tail2.xaxis.set_major_formatter(mdates.DateFormatter('%b %-d'))
tail2.set_xlabel('%d' % TT[0].year if TT[0].year == TT[-1].year else '')
for l in tail2.get_xticklabels():
    l.set_rotation(30)
    l.set_horizontalalignment('right')
# the markers series_panel leaves behind belong to the movie, not to a still
for ax in (bx1, bx2):
    for ln in list(ax.lines):
        ln.remove()
fig2.savefig(out_dir / (stem + '_series.png'), dpi=200, bbox_inches='tight')
print('saved %s' % (out_dir / (stem + '_series.png')))

# the series behind the movie, so a number can be quoted without re-running it
csv_fn = out_dir / (stem + '_area.csv')
pd.concat({nm: S[nm] for nm in SER}, axis=1).to_csv(csv_fn)
print('saved %s' % csv_fn)
