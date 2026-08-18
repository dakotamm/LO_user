"""
Spring/neap movie of the BOTTOM hypoxic CONTOUR, with the tidal exchange that
is supposed to be moving it.

The question this animates is not "when does hypoxia happen" -- that is
20260810_hypoxia_movie.py, over a whole season -- but "does the hypoxic edge
breathe with the fortnightly cycle". So the window is a handful of spring/neap
cycles rather than seven months, the hypoxic EDGE is the object being drawn,
and the forcing panel is Qprism.

  left, top     Qprism at --qsect: the tidal exchange, already Godin filtered
                and daily-subsampled by bulk_calc_avg.py. Springs and neaps are
                marked on it and the marks are carried down through the other
                two panels, so a feature in the area series can be read against
                the tide without moving your eye between axes.
  left, middle  bottom hypoxic area in --series, as nested bands, exactly as in
                20260810_hypoxia_movie.py.
  left, bottom  the SAME area and Qprism as 30-day centred rolling ANOMALIES,
                on twin axes. This is the panel the fortnightly signal actually
                lives in; see WHY ANOMALIES below.
  right         bottom-cell DO, with the hypoxic contour drawn on it and
                optionally trailing --trail days of faded previous contours.

WHY QPRISM AND NOT A SEA-LEVEL RANGE (DM 2026.08.11). At pc_cp the diurnal and
semidiurnal bands are nearly equal in sea-level amplitude but not in transport,
because transport scales with d(ssh)/dt: a sea-level envelope weights the two
bands about equally, Qprism weights the semidiurnal about twice as heavily. The
two bands have different fortnightly periods (14.77 d semidiurnal, the
spring-neap cycle proper, vs 13.66 d diurnal) and beat with a period of 180 d,
so an ssh envelope's extrema wander 2-3 d either side of the real spring-neap
cycle. Measured over 2024-25: qprism vs semidiurnal envelope r = +0.99, vs the
total |ssh'| envelope r = +0.51. Only Qprism is locked to the moon. Hence
DM_outs/20260806_tidal_phase/phase_daily.csv, which is built on sea-level
range, is deliberately NOT used to label springs and neaps here.

WHY ANOMALIES (DM 2026.08.06). Over a summer the seasonal drawdown is an order
of magnitude larger than the fortnightly wiggle, so a raw area-vs-Qprism
regression returns "no spring/neap effect" even when there is one. Both series
are therefore also carried as 30-day centred rolling anomalies -- about two
spring-neap cycles, long enough to strip the seasonal and synoptic background
and short enough to leave the fortnightly signal alone -- and the correlation
that gets printed is computed on those, at the lag that maximises it, with an
autocorrelation-corrected n_eff (lag-1, Bretherton et al. 1999). Daily values
are strongly autocorrelated and nominal n makes almost anything significant.

--pad-days exists for the same reason: a centred 30-day mean cannot be formed
in the first and last 15 days of the window, so the script READS 15 extra days
on each side, filters on the padded record, and animates only --ds0 to --ds1.
The padding is data, not extrapolation; set it to 0 to see the difference.

SOURCE. lowpassed.nc -- one Godin-filtered field per day, so the tide is
removed rather than aliased. That is what makes a fortnightly signal in this
movie interpretable: anything left moving IS subtidal. Frames are whatever
lowpassed.nc files exist in the window, so a run with gaps animates its gaps.
For the tidal cycle itself, see 20260818_hypoxia_tidal_movie.py, which is this
script's twin on hourly history files.

THE CONTOUR IS DRAWN even when the threshold is already a colour-class edge --
which it is by default. That is the opposite of what 20260810_hypoxia_movie.py
does, and it is deliberate: there the map is context for an area series, here
the edge is the subject, and a black line reads as an object that moves in a
way that a boundary between two fills does not.

Region-plot convention (DM 2026.08.04): with --zoom 0 the window is the
rectangular bounding box of the region polygon plus --pad-cells of margin, and
in either case only cells inside `wb` are drawn.

Runs wherever the lowpassed files and the tef2 bulk files are (apogee for the
2024-25 run). The 2017 wb1_r0_xn11b days on the Mac make a smoke test, using
that run's old pc0 section for Qprism.

    python 20260818_hypoxia_springneap_movie.py
    python 20260818_hypoxia_springneap_movie.py --test
    python 20260818_hypoxia_springneap_movie.py --trail 7
    python 20260818_hypoxia_springneap_movie.py --ds0 2025.08.01 --ds1 2025.09.30
    python 20260818_hypoxia_springneap_movie.py --series wb --zoom 0 --region wb_north
    python 20260818_hypoxia_springneap_movie.py --gtx wb1_r0_xn11b --ds0 2017.09.04 --ds1 2017.09.18 --qsect pc0 --pad-days 0 --test
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
from scipy.signal import find_peaks
from scipy.stats import t as student_t
from matplotlib.path import Path as MplPath
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from cmocean import cm

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

# house style, shared with the pc_mouth analysis figures and the other wb1
# movies. Grid on the timeseries, never on the map -- DM 2026.08.07.
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
RED = '#e04256'
BLUE = '#4565e8'
GREY = '0.45'
# Land fill and map furniture, from 20260807_grid_bathy_ppt.py (the wb1 grid
# map). Land is FILLED rather than left transparent: a see-through landmask
# reads as whatever is behind it, which kills the coastline. White still means
# "water not drawn" (outside wb, or inside --exclude); the warm fill is land.
LAND = '#e8e4dc'
COAST = 'k'

# DO classes, low to high: dark red = anoxic, red = hypoxic, warm = low-DO,
# cool = oxygenated. The class edges ARE the band thresholds, so a colour on
# the map and a band in the area panel are the same statement. See
# 20260810_hypoxia_movie.py for why cmocean's oxy is not the default: it spends
# its middle on a low-contrast grey and nearly the whole Whidbey sea floor
# sits in that grey between 2 and 5 mg/L.
LEVEL_COLORS = ['#4a0f16', '#b3202e', '#e8873c', '#f6c667', '#a9cfd4', '#5d9fc0']
OVER_COLOR = '#2f6b96'

DO_MMOL_TO_MGL = 32.0 / 1000.0        # mmol m-3 (uM) -> mg L-1

# ---- arguments -------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument('--gtx', default='wb1_t0_xn11abbur00')
# Default window: 2025.07.15 to 2025.10.15, which is ~6 spring/neap cycles and
# sits on the part of the season where the Penn Cove bed is actually hypoxic
# (see the 20260810 seasonal movie -- the < 2 mg/L band is continuously present
# from mid-July to late October). A fortnightly signal cannot be seen in a
# quantity that is pinned at zero.
p.add_argument('--ds0', default='2025.07.15')
p.add_argument('--ds1', default='2025.10.15')
p.add_argument('--pad-days', default=15, type=int, dest='pad_days',
               help='extra days read on EACH side, used only to condition the '
                    'centred rolling anomaly; not animated. 0 disables')
p.add_argument('--stride', default=1, type=int,
               help='keep every Nth day (1 = every lowpassed file)')
p.add_argument('--region', default='wb_north',
               help='polygon the map window is taken from')
p.add_argument('--exclude', default='skagit_delta',
               help='comma-separated polygons to cut out of the map, and to '
                    'zoom past; empty string keeps the whole region')
p.add_argument('--zoom', default=0.6, type=float,
               help='window = the --zoom-poly box extended EAST by this many '
                    'of its widths. 0 or less falls back to the whole --region '
                    'window. Default matches the wbnorth velocity movie.')
p.add_argument('--zoom-poly', default='pc', dest='zoom_poly',
               help='polygon the --zoom window is built around')
p.add_argument('--aa', default='',
               help='explicit lon0,lon1,lat0,lat1 window; overrides --zoom')
p.add_argument('--series', default='pc',
               help='polygon the hypoxic-area series is computed for. Need not '
                    'be --region: the default is the basin window with the '
                    'cove series, outlined in red on the map')
p.add_argument('--thresh', default=[2.0, 3.0, 5.0], type=float, nargs='+',
               help='mg/L bands, low to high; the lowest is the CONTOUR drawn '
                    'on the map and the one the statistics are computed on')
p.add_argument('--qsect', default='pc_lp',
               help='tef2 section supplying Qprism. pc_lp is the Penn Cove '
                    'mouth, i.e. the exchange that ventilates the cove')
p.add_argument('--trail', default=0, type=int,
               help='draw this many previous days of hypoxic contour, fading '
                    'with age. 7 is half a spring/neap cycle and makes the '
                    'breathing visible in a single still')
p.add_argument('--anom-window', default=30, type=int, dest='anom_window',
               help='centred rolling window [days] for the anomalies; 30 is '
                    'about two spring/neap cycles (DM 2026.08.06)')
p.add_argument('--max-lag', default=10, type=int, dest='max_lag',
               help='lags [days] searched in the area-vs-Qprism correlation')
p.add_argument('--cmap', default='levels', choices=['levels', 'oxy'],
               help='levels = discrete DO classes cut on --levels (default); '
                    'oxy = the continuous cmocean oxygen map on --vmin/--vmax')
p.add_argument('--levels', default=[0.0, 0.5, 2.0, 3.0, 5.0, 7.0, 10.0],
               type=float, nargs='+',
               help='mg/L class edges for --cmap levels; anything above the '
                    'last one is the over-colour')
p.add_argument('--vmin', default=0.0, type=float, help='--cmap oxy only')
p.add_argument('--vmax', default=10.0, type=float, help='--cmap oxy only')
p.add_argument('--pad-cells', default=10, type=int, dest='pad_cells')
p.add_argument('--fps', default=6, type=int)
p.add_argument('--dpi', default=150, type=int,
               help='render dpi for the movie frames -- 100 makes the labels '
                    'mushy once h264 has had a go at them')
p.add_argument('--no-transparent', dest='transparent', action='store_false',
               help='opaque white stills. The saved PNGs are transparent by '
                    'default (standing preference); the movie is always opaque '
                    'because h264 has no alpha channel')
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
SER = args.series
out_dir = Ldir['LOo'] / 'DM_outs' / '20260818_hypoxia_springneap_movie'
Lfun.make_dir(out_dir)

# The animated window, and the wider window that is READ. Only the animated one
# is ever plotted; the pad exists so the centred rolling mean is a real mean at
# both ends rather than a shortening one-sided average.
T_LO = pd.Timestamp(args.ds0.replace('.', '-'))
T_HI = pd.Timestamp(args.ds1.replace('.', '-')) + pd.Timedelta(days=1)
pad = max(0, args.pad_days)
RD0 = (T_LO - pd.Timedelta(days=pad)).strftime('%Y.%m.%d')
RD1 = (T_HI + pd.Timedelta(days=pad)).strftime('%Y.%m.%d')

# ---- files -----------------------------------------------------------------
# Same search as 20260806_hypoxia_reduce.py rather than Lfun.get_fn_list: the
# run lives under whichever roms_out key is populated on this machine, and a
# missing day should be skipped, not crash the movie.
ROMS_OUT_KEYS = (['roms_out%d' % args.ro] if args.ro is not None
                 else ['roms_out2', 'roms_out1', 'roms_out'])


def find_files(ds0, ds1):
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
            if not (ds0 <= ds_ <= ds1) or ds_ in days:
                continue
            fn = run_dir / 'lowpassed.nc'
            if fn.is_file():
                days[ds_] = fn
    return dict(sorted(days.items()))


FN = find_files(RD0, RD1)
if not FN:
    raise SystemExit(
        'no lowpassed.nc for %s between %s and %s under\n  %s'
        % (args.gtx, RD0, RD1,
           '\n  '.join(str(Ldir.get(k)) for k in ROMS_OUT_KEYS)))
fn_list = list(FN.values())[::max(1, args.stride)]
n_anim = sum(1 for d in list(FN.keys())[::max(1, args.stride)]
             if args.ds0 <= d <= args.ds1)
print('%d daily files %s .. %s (stride %d); %d of them inside %s .. %s will '
      'be animated' % (len(fn_list), RD0, RD1, args.stride, n_anim,
                       args.ds0, args.ds1))
if n_anim == 0:
    raise SystemExit('no lowpassed.nc inside the animated window itself')
# Cycles asked for, so a window that is too short to show what the script is
# about says so rather than quietly producing a one-cycle movie.
n_days = (T_HI - T_LO).days
print('window is %d days = %.1f spring/neap cycles (14.77 d)'
      % (n_days, n_days / 14.77))
if n_days < 30:
    print('  WARNING: under two fortnightly cycles -- the 30-day anomaly has '
          'almost nothing to remove and the correlation below is not '
          'meaningful. Widen --ds0/--ds1.')

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

# The series region: the POLYGON intersected with water, not the wb clip and
# not the map window -- the area series has to mean the same thing as the
# hypoxia_*.p reduction's A_bot_*, which is per polygon.
RMASK = poly_mask(load_sect(SER)) & wet
IDX = np.where(RMASK)                                  # index once, reuse daily
RAREA = float(cell_area[RMASK].sum())
print('series region %-12s %6d cells, %8.1f km2, mean h %5.1f m, max h %5.1f m'
      % (SER, RMASK.sum(), RAREA / 1e6, h[RMASK].mean(), h[RMASK].max()))

# ---- map window ------------------------------------------------------------
# Lifted from 20260810_hypoxia_movie.py so all the wb1 movies frame the same
# water and can be played side by side. Everything here is about what is DRAWN;
# the area series is per polygon and is untouched by it.
keep = wet & in_wb
excl = [s for s in args.exclude.split(',') if s]
for nm in excl:
    keep &= ~poly_mask(load_sect(nm))
in_win = keep & poly_mask(reg)
if not in_win.any():
    raise SystemExit('nothing left after excluding %s from %s'
                     % (excl, args.region))
# The window comes from the LARGEST CONNECTED PIECE of what survives, not from
# its outright min/max: cutting skagit_delta out of wb_north leaves exactly one
# stray cell up by Deception Pass, which on its own stretched the map from 84
# rows to 144. Drawing still uses the full `keep`.
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
# Passage, so the window is extended asymmetrically.
if args.aa:
    aa = [float(s) for s in args.aa.split(',')]
    print('window: explicit --aa')
elif args.zoom > 0:
    zp = load_sect(args.zoom_poly)
    zw = float(zp.x.max() - zp.x.min())
    zh = float(zp.y.max() - zp.y.min())
    aa = [zp.x.min() - 0.10 * zw, zp.x.max() + args.zoom * zw,
          zp.y.min() - 0.60 * zh, zp.y.max() + 0.90 * zh]
    print('window: %s box + %.2f of its widths east' % (args.zoom_poly, args.zoom))
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
land_s = (~wet)[SUB]                          # true land, filled warm grey
print('window %s -> %d x %d cells, %d drawn'
      % (['%.4f' % v for v in aa], lat_s.shape[0], lat_s.shape[1], draw_s.sum()))


# ---- read one lowpassed file -----------------------------------------------
def read_one(fn):
    """(time, bottom DO in the window, area row for the series region).

    s_rho index 0 is the BOTTOM cell in ROMS, so oxygen[0] is the bed -- the
    same convention 20260806_hypoxia_reduce.py uses. Only that level is pulled
    off disk.

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

    i, j = IDX
    v, a_ = bot[i, j], cell_area[i, j]
    row = {'do_mean': float(np.nanmean(v)), 'do_min': float(np.nanmin(v))}
    for th in THRESH:
        row['A_%g' % th] = float(a_[v < th].sum())      # NaN < th is False
    return t, bot[SUB].astype(np.float32), row


nproc = max(1, min(args.nproc, len(fn_list)))
print('reading %d files on %d process(es)...' % (len(fn_list), nproc))
if nproc > 1:
    ctx = mp.get_context('fork')              # workers inherit the masks above
    with ctx.Pool(nproc) as pool:
        res = pool.map(read_one, fn_list)
else:
    res = [read_one(fn) for fn in fn_list]

TT_ALL = pd.to_datetime([r[0] for r in res])
FLD = np.where(draw_s[None, :, :], np.stack([r[1] for r in res]), np.nan)
S = pd.DataFrame([r[2] for r in res], index=TT_ALL)

# which of the frames read are actually animated (the pad is filter-only)
INWIN = np.where((TT_ALL >= T_LO) & (TT_ALL <= T_HI))[0]
TT = TT_ALL[INWIN]
if len(INWIN) == 0:
    raise SystemExit('the padded read returned nothing inside %s .. %s'
                     % (args.ds0, args.ds1))
print('%d frames animated, %d read for filtering only'
      % (len(INWIN), len(TT_ALL) - len(INWIN)))

# ---- Qprism ----------------------------------------------------------------
# From bulk_calc_avg.py, which has already Godin filtered and daily-subsampled
# it, so it lands on the same noon stamps as lowpassed.nc. Any bulk directory
# holding the section is acceptable; the one with the largest overlap with the
# read window wins, because these runs accumulate several extraction ranges.
def load_qprism():
    tdir = Ldir['LOo'] / 'extract' / args.gtx / 'tef2'
    cand = sorted(tdir.glob('bulk_avg_*/%s.nc' % args.qsect)) + \
        sorted(tdir.glob('bulk_*/%s.nc' % args.qsect))
    cand = [c for c in cand if 'bulk_plots' not in c.parent.name]
    if not cand:
        print('no bulk*/%s.nc under %s -- the Qprism panel and the spring/neap '
              'marks are skipped. Run bulk_calc_avg.py for this run, or point '
              '--qsect at a section that has been processed.' % (args.qsect, tdir))
        return None
    best, best_n = None, -1
    for fn in cand:
        ds = xr.open_dataset(fn)
        if 'qprism' not in ds.data_vars:
            ds.close()
            continue
        q = pd.Series(ds.qprism.values, index=pd.to_datetime(ds.time.values))
        ds.close()
        n = int(((q.index >= TT_ALL[0]) & (q.index <= TT_ALL[-1])).sum())
        if n > best_n:
            best, best_n, best_fn = q, n, fn
    if best is None or best_n == 0:
        print('bulk files exist for %s but none covers %s .. %s -- Qprism panel '
              'skipped' % (args.qsect, TT_ALL[0].date(), TT_ALL[-1].date()))
        return None
    span = (TT_ALL[-1] - TT_ALL[0]).days + 1
    print('Qprism from %s (%d daily values inside the %d calendar days the '
          'read window spans)'
          % (best_fn.parent.name + '/' + best_fn.name, best_n, span))
    return best


QP_SRC = load_qprism()

# Two views of the same series, deliberately:
#   QD  a REGULAR daily axis spanning the read window, gaps interpolated. The
#       springs and neaps are found on this one, because find_peaks counts
#       SAMPLES, not days -- on a record with holes in it (the 2017 test bed is
#       nothing but holes) a 10-sample separation is not 10 days and the marks
#       land in the wrong places. The 2024-25 run is continuous and the two
#       axes are then identical.
#   QP  the same series on the movie's own frame times, which is what the
#       statistics and the title need. reindex+nearest within half a day rather
#       than a join: the bulk stamps are noon like the lowpassed ones, but a
#       run whose bulk extraction was built on a different subsample would
#       otherwise silently drop every row.
QD = None
QP = None
if QP_SRC is not None:
    day_ax = pd.date_range(QP_SRC.index.min(), QP_SRC.index.max(), freq='D')
    day_ax = day_ax[(day_ax >= TT_ALL[0] - pd.Timedelta('1D')) &
                    (day_ax <= TT_ALL[-1] + pd.Timedelta('1D'))]
    QD = QP_SRC.reindex(day_ax, method='nearest',
                        tolerance=pd.Timedelta('12h')).interpolate(
                            limit_direction='both')
    QP = QP_SRC.reindex(TT_ALL, method='nearest', tolerance=pd.Timedelta('12h'))
    if QP.notna().sum() == 0:
        print('Qprism did not align with the frame times -- panel skipped')
        QP = QD = None

# ---- springs and neaps -----------------------------------------------------
# Extrema of Qprism itself, not of a sea-level range (see the docstring). The
# minimum separation is 10 days: the fortnightly period is 14.77 d, and a
# smaller distance lets a two-day wobble on the shoulder of a spring register
# as its own peak.
SPR_T, NEP_T = [], []
if QD is not None and len(QD) > 12:
    q = QD.values
    ispr, _ = find_peaks(q, distance=10)
    inep, _ = find_peaks(-q, distance=10)
    print('%d springs, %d neaps in the read window; Qprism %.0f to %.0f m3/s '
          '(spring/neap ratio %.2f)'
          % (len(ispr), len(inep), np.nanmin(q), np.nanmax(q),
             (np.mean(q[ispr]) / np.mean(q[inep]))
             if len(ispr) and len(inep) else np.nan))
    SPR_ALL = list(QD.index[ispr])
    NEP_ALL = list(QD.index[inep])
    # only the marks inside the animated window get drawn; the rest exist so
    # the pad still knows where it sits in the cycle
    SPR_T = [t for t in SPR_ALL if T_LO <= t <= T_HI]
    NEP_T = [t for t in NEP_ALL if T_LO <= t <= T_HI]
    print('  inside the animated window: %d springs %s, %d neaps %s'
          % (len(SPR_T), [str(t.date()) for t in SPR_T],
             len(NEP_T), [str(t.date()) for t in NEP_T]))
else:
    SPR_ALL, NEP_ALL = [], []

# ---- anomalies and the correlation that is the point of the script ---------
# 30-day CENTRED rolling anomaly on both series (DM 2026.08.06). min_periods is
# half the window so the padded ends still produce something, but the whole
# reason --pad-days exists is that those ends are not as well conditioned as
# the interior; with the default pad the animated window is entirely interior.
W = args.anom_window
AFRAC = S['A_%g' % T0] / RAREA * 100.0                  # % of the region floor


def anom(x):
    return x - x.rolling(W, center=True, min_periods=max(3, W // 2)).mean()


A_AN = anom(AFRAC)
Q_AN = anom(QP) if QP is not None else None


def corr_neff(x, y):
    """Pearson r with an autocorrelation-corrected n (Bretherton 1999).

    Daily values are strongly autocorrelated, so the nominal n makes almost
    anything significant; n_eff = n (1 - r1x r1y) / (1 + r1x r1y).
    """
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 8:
        return np.nan, np.nan, np.nan, 0
    a, b = d.iloc[:, 0].values, d.iloc[:, 1].values
    r = float(np.corrcoef(a, b)[0, 1])
    r1a = float(np.corrcoef(a[:-1], a[1:])[0, 1])
    r1b = float(np.corrcoef(b[:-1], b[1:])[0, 1])
    n = len(d)
    neff = n * (1 - r1a * r1b) / (1 + r1a * r1b)
    neff = float(np.clip(neff, 3, n))
    ts = r * np.sqrt(max(neff - 2, 1) / max(1 - r ** 2, 1e-12))
    pv = float(2 * student_t.sf(abs(ts), max(neff - 2, 1)))
    return r, pv, neff, n


LAGS = pd.DataFrame()
if Q_AN is not None:
    print('\nhypoxic area (%s, < %g mg/L) vs Qprism (%s), by lag'
          % (SER, T0, args.qsect))
    print('  positive lag = the AREA follows Qprism by that many days')
    print('  %5s  %7s  %8s  %6s   %s' % ('lag', 'r_anom', 'p_anom', 'n_eff', 'r_raw'))
    rows = []
    for L in range(-args.max_lag, args.max_lag + 1):
        r, pv, neff, n = corr_neff(A_AN.shift(-L), Q_AN)
        rr, _, _, _ = corr_neff(AFRAC.shift(-L), QP)
        rows.append(dict(lag_days=L, r_anom=r, p_anom=pv, n_eff=neff, n=n,
                         r_raw=rr))
    LAGS = pd.DataFrame(rows).set_index('lag_days')
    for L, row in LAGS.iterrows():
        star = ' *' if row.p_anom < 0.05 else ''
        print('  %5d  %+7.3f  %8.4f  %6.1f   %+6.3f%s'
              % (L, row.r_anom, row.p_anom, row.n_eff, row.r_raw, star))
    kb = LAGS.r_anom.abs().idxmax()
    b = LAGS.loc[kb]
    print('strongest at lag %+d d: r = %+.3f, p = %.4f (n_eff %.0f); the same '
          'lag on the RAW series gives r = %+.3f'
          % (kb, b.r_anom, b.p_anom, b.n_eff, b.r_raw))
    print('  a NEGATIVE r means more tidal exchange goes with LESS hypoxic '
          'area, i.e. spring tides ventilate the bed; positive would mean the '
          'opposite and is worth a hard look before believing.')
    print('  raw-vs-anomaly disagreement is expected, not a bug: over a summer '
          'the seasonal drawdown is much larger than the fortnightly wiggle '
          'and buries it (DM 2026.08.06).')

# what the field itself does, before anything is drawn
print('\nbottom DO in the window: min %.2f, median %.2f, max %.2f mg/L'
      % (np.nanmin(FLD), np.nanmedian(FLD), np.nanmax(FLD)))
a_win = S['A_%g' % T0].loc[TT]
print('%s floor %.1f km2; A_bot < %g mg/L over the animated window: '
      'min %.2f, mean %.2f, max %.2f km2 (%.0f%% to %.0f%% of the floor)'
      % (SER, RAREA / 1e6, T0, a_win.min() / 1e6, a_win.mean() / 1e6,
         a_win.max() / 1e6, 100 * a_win.min() / RAREA, 100 * a_win.max() / RAREA))
if a_win.max() <= 0:
    print('  the bed never goes below %g mg/L here -- nothing will be '
          'contoured. Try a higher --thresh, or a window later in the summer.'
          % T0)

# ---- figure ----------------------------------------------------------------
# constrained layout, not manual margins: the map carries a fixed aspect (dar),
# so it shrinks inside its own gridspec cell and a hand-placed colorbar ends up
# stranded in the whitespace.
plt.close('all')
fig = plt.figure(figsize=(17.5, 10), layout='constrained')
gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.05])
axq = fig.add_subplot(gs[0, 0])                    # Qprism
axa = fig.add_subplot(gs[1, 0], sharex=axq)        # hypoxic area bands
axn = fig.add_subplot(gs[2, 0], sharex=axq)        # anomalies
axm = fig.add_subplot(gs[:, 1])                    # map

# --- map
LEVELS = sorted(args.levels)
if args.cmap == 'levels':
    if len(LEVELS) - 1 > len(LEVEL_COLORS):
        raise SystemExit('--levels asks for %d classes; only %d colours are '
                         'defined' % (len(LEVELS) - 1, len(LEVEL_COLORS)))
    cmap = ListedColormap(LEVEL_COLORS[:len(LEVELS) - 1])
    cmap.set_over(OVER_COLOR)
    norm = BoundaryNorm(LEVELS, cmap.N)
    # spacing='proportional' keeps the colourbar a ruler in mg/L; 'uniform'
    # would give the 0-0.5 anoxic sliver the same height as the 5-7 class.
    cb_kw = dict(extend='max', spacing='proportional', ticks=LEVELS)
    lo, hi = LEVELS[0], LEVELS[-1]
else:
    cmap, norm = cm.oxy, Normalize(vmin=args.vmin, vmax=args.vmax)
    lo, hi = args.vmin, args.vmax
    cb_kw = dict(ticks=sorted(set([lo] + list(THRESH) + [hi])))
# land under everything else. Only true land (mask_rho == 0) is filled -- water
# that is simply not drawn (outside wb, or inside --exclude) stays white.
axm.pcolormesh(plon_s, plat_s,
               np.ma.masked_where(~land_s, np.ones(land_s.shape)),
               cmap=ListedColormap([LAND]), shading='flat', zorder=0)
cs = axm.pcolormesh(plon_s, plat_s, FLD[INWIN[0]], cmap=cmap, norm=norm,
                    shading='flat', zorder=1)
f_lo = float(np.nanmean(FLD < lo))
f_hi = float(np.nanmean(FLD > hi))
cb_kw.setdefault('extend', 'both' if (f_lo > 0 and f_hi > 0) else
                 'max' if f_hi > 0 else 'min' if f_lo > 0 else 'neither')
fig.colorbar(cs, ax=axm, shrink=0.75, pad=0.01, aspect=35,
             label='bottom-cell dissolved oxygen [mg L$^{-1}$]', **cb_kw)

pfun.add_coast(axm, color=COAST, linewidth=0.8)
# the series polygon, not a bounding box: the outline says exactly which cells
# the curve counts, and at this zoom there is hypoxic water on screen that the
# curve does not count.
s0 = load_sect(SER)
axm.plot(np.append(s0.x.values, s0.x.values[0]),
         np.append(s0.y.values, s0.y.values[0]), '-',
         color=RED, lw=2.0, zorder=8)

DEC = 1 if (aa[1] - aa[0]) >= 1.0 else 2


def nice_ticks(v0, v1, n, dec):
    """n evenly spaced round values, both ends INSIDE [v0, v1].

    Rounding the endpoints outward (which plain linspace+round does) puts the
    first and last tick past the window, and pinning the limits then drops
    them -- three ticks on a four-tick axis.
    """
    q_ = 10.0 ** dec
    return np.linspace(np.ceil(v0 * q_) / q_, np.floor(v1 * q_) / q_, n).round(dec)


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

# --- the hypoxic contour, with an optional fading trail
# Drawn ALWAYS, even though T0 is a colour-class edge by default: here the edge
# is the subject of the movie, not decoration on an area series, and a black
# line reads as an object that moves in a way a boundary between two fills does
# not. The trail is the same line at earlier days, faded by age -- half a
# spring/neap cycle of trail (--trail 7) shows the breathing in one still.
TRAIL = {'objs': []}
CLW = 1.6


def clear_trail():
    for c in TRAIL['objs']:
        if c is not None:
            c.remove()
    TRAIL['objs'] = []


def draw_edge(fi):
    f = FLD[fi]
    new = None
    if np.isfinite(f).any() and np.nanmin(f) < T0:
        new = axm.contour(lon_s, lat_s, f, levels=[T0], colors='k',
                          linewidths=CLW, zorder=5)
    TRAIL['objs'].append(new)
    while len(TRAIL['objs']) > max(0, args.trail) + 1:
        old = TRAIL['objs'].pop(0)
        if old is not None:
            old.remove()
    n = len(TRAIL['objs'])
    for k, c in enumerate(TRAIL['objs']):
        if c is None:
            continue
        age = n - 1 - k                      # 0 = the current day
        if age == 0:
            c.set_alpha(1.0)
            c.set_linewidth(CLW)
        else:
            c.set_alpha(max(0.10, 0.55 * (1 - age / (args.trail + 1.0))))
            c.set_linewidth(0.9)


# --- spring/neap marks, carried down every left panel
def mark_springneap(ax, label=False):
    for t_ in SPR_T:
        ax.axvline(t_, color=BLUE, lw=1.0, ls='-', alpha=0.55, zorder=1)
    for t_ in NEP_T:
        ax.axvline(t_, color=GREY, lw=1.0, ls=':', alpha=0.8, zorder=1)
    if label:
        # inside the axes, not above it: the panel title sits on that line and
        # a spring landing under a word is unreadable
        for t_, s_ in [(t, 'S') for t in SPR_T] + [(t, 'N') for t in NEP_T]:
            ax.annotate(s_, xy=(t_, 0.97), xycoords=('data', 'axes fraction'),
                        ha='center', va='top', fontsize=10, fontweight='bold',
                        color=BLUE if s_ == 'S' else GREY)


# --- Qprism
if QD is not None:
    # the regular daily series, so the line is not broken by holes in the
    # lowpassed record that Qprism itself does not have
    axq.plot(QD.index, QD.values / 1e3, '-', color=BLUE, lw=1.8)
    axq.set_ylabel('Qprism [10$^3$ m$^3$ s$^{-1}$]')
    axq.set_title('tidal exchange at %s -- S = spring, N = neap (extrema of '
                  'Qprism itself, not of sea-level range)' % args.qsect,
                  fontsize=10, color=BLUE)
else:
    axq.set_ylabel('Qprism')
    axq.text(0.5, 0.5, 'no bulk extraction for %s' % args.qsect,
             transform=axq.transAxes, ha='center', va='center', color=GREY)
axq.grid(**GRID)
mark_springneap(axq, label=True)

# --- hypoxic area, nested bands
# A(<5) >= A(<3) >= A(<2) by construction, so the bands stack with no
# arithmetic: the field reads as one thing draining downward. Each band takes
# the MAP's colour for the water it adds, so this panel is the colourbar stood
# on end -- no second legend to learn.
def band_colour(th):
    if args.cmap != 'levels':
        return LEVEL_COLORS[min(THRESH.index(th), len(LEVEL_COLORS) - 1)]
    k = int(np.searchsorted(LEVELS, th, side='left')) - 1
    return LEVEL_COLORS[min(max(k, 0), len(LEVELS) - 2)]


def gapped(d):
    """Break the series wherever days are missing.

    fill_between joins whatever points it is given, so a run with a hole in it
    draws a straight ramp across the hole -- days of invented, monotonic
    hypoxia. A NaN row in the middle of each gap makes the hole a hole. (The
    2024-25 run is continuous and this is a no-op there; the 2017 test bed is
    nothing but gaps.)
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
    if isinstance(d, pd.Series):
        return pd.concat([d, pd.Series(np.nan, index=mid)]).sort_index()
    return pd.concat([d, pd.DataFrame(np.nan, index=mid,
                                      columns=d.columns)]).sort_index()


dg = gapped(S)
for k, th in enumerate(reversed(THRESH)):             # widest band first
    axa.fill_between(dg.index, 0, dg['A_%g' % th] / RAREA * 100.0,
                     color=band_colour(th), lw=0,
                     label='< %g mg L$^{-1}$' % th, zorder=2 + k)
axa.set_ylabel('bottom area [% of floor]')
axa.set_ylim(0, 100)
axa.grid(**GRID)
axa.legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=len(THRESH))
axa.set_title('bottom hypoxic area, %s (%.1f km$^2$ of sea floor, outlined in '
              'red on the map)' % (SER, RAREA / 1e6), fontsize=11, color=RED)
km2 = RAREA / 1e6
sax = axa.secondary_yaxis('right', functions=(lambda p_: p_ * km2 / 100,
                                              lambda a_: a_ * 100 / km2))
sax.set_ylabel('[km$^2$]')
mark_springneap(axa)

# --- anomalies: the panel the fortnightly signal is actually visible in
axn.axhline(0, color='0.5', lw=0.8)
axn.plot(gapped(A_AN).index, gapped(A_AN).values, '-', color=RED, lw=1.8,
         label='hypoxic area')
axn.set_ylabel('area anomaly [% of floor]', color=RED)
axn.tick_params(axis='y', colors=RED)
axn.grid(**GRID)
if Q_AN is not None:
    axn2 = axn.twinx()
    axn2.plot(Q_AN.index, Q_AN.values / 1e3, '-', color=BLUE, lw=1.5, alpha=0.85)
    axn2.set_ylabel('Qprism anomaly [10$^3$ m$^3$ s$^{-1}$]', color=BLUE)
    axn2.tick_params(axis='y', colors=BLUE)
    b = LAGS.loc[LAGS.r_anom.abs().idxmax()] if len(LAGS) else None
    axn.set_title('%d-day rolling anomalies -- area (red) vs Qprism (blue)%s'
                  % (W, ('; best r = %+.2f at lag %+d d, p = %.3f'
                         % (b.r_anom, int(LAGS.r_anom.abs().idxmax()), b.p_anom))
                     if b is not None else ''),
                  fontsize=10, color=GREY)
else:
    axn.set_title('%d-day rolling anomaly of the hypoxic area' % W,
                  fontsize=10, color=GREY)
mark_springneap(axn)

# one shared time axis: the animated window, NOT the padded read window
for ax in (axq, axa, axn):
    ax.set_xlim(TT[0], TT[-1])
plt.setp(axq.get_xticklabels(), visible=False)
plt.setp(axa.get_xticklabels(), visible=False)
axn.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=9))
axn.xaxis.set_major_formatter(mdates.DateFormatter('%b %-d'))
axn.set_xlabel('%d' % TT[0].year if TT[0].year == TT[-1].year else '')
for l in axn.get_xticklabels():
    l.set_rotation(30)
    l.set_horizontalalignment('right')

# moving markers, one per left panel, riding the UNgapped series so the index
# still lines up frame for frame with the map
MK = [ax.axvline(TT[0], color='k', lw=1.5, zorder=20) for ax in (axq, axa, axn)]
dot = axa.plot([TT[0]], [AFRAC.loc[TT[0]]], 'o', ms=7, color='k', zorder=21)[0]

if args.transparent:
    fig.patch.set_alpha(0.0)
    for ax in (axm, axq, axa, axn):
        ax.patch.set_alpha(0.0)


def update(k):
    """k indexes the ANIMATED frames; INWIN maps it back into the read arrays."""
    fi = INWIN[k]
    cs.set_array(FLD[fi].ravel())
    draw_edge(fi)
    a_ = S['A_%g' % T0].iloc[fi]
    q_ = QP.iloc[fi] if QP is not None else np.nan
    ttl.set_text('bottom oxygen -- %s\n%s below %g mg L$^{-1}$: %.1f km$^2$ '
                 '(%.0f%% of the %s floor)%s'
                 % (TT_ALL[fi].strftime('%Y-%m-%d'), SER, T0, a_ / 1e6,
                    100 * a_ / RAREA, SER,
                    '' if not np.isfinite(q_) else
                    '   |   Qprism %.1f x10$^3$ m$^3$ s$^{-1}$' % (q_ / 1e3)))
    for m in MK:
        m.set_xdata([TT_ALL[fi], TT_ALL[fi]])
    dot.set_data([TT_ALL[fi]], [AFRAC.iloc[fi]])
    return []


def render(k):
    """Frame k as the MOVIE would have it, trail included.

    Calling update(k) on its own leaves the trail empty, which is not what that
    frame looks like mid-movie, so the preceding --trail frames are replayed
    first. With --trail 0 this is just update(k).
    """
    clear_trail()
    for j in range(max(0, k - max(0, args.trail)), k + 1):
        update(j)


stem = ('20260818_hypoxia_springneap_%s_%s%s_%s_%s'
        % (args.gtx, SER, '_zoom%g' % args.zoom if args.zoom > 0 else
           '_%s' % args.region, args.ds0, args.ds1))
# transparent by default (standing preference): these stills get dropped onto
# slides whose background is not white.
still_kw = dict(dpi=200, bbox_inches='tight', transparent=args.transparent)

# Video formats. h264 has no alpha channel, so mp4 is rendered on white -- a
# transparent figure piped to it just gets composited onto black. ProRes 4444
# and QuickTime RLE do carry alpha and are both NATIVE ffmpeg encoders. Do NOT
# reach for VP9/webm: it encodes without error and the alpha is silently gone
# (verified 2026.08.07, ffmpeg 7.1.1).
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

# Stills are taken at the largest and smallest hypoxic area INSIDE the animated
# window: those are the two frames the movie exists to compare, and if the
# fortnightly signal is real they should sit near a neap and near a spring.
k_max = int(np.argmax(S['A_%g' % T0].iloc[INWIN].values))
k_min = int(np.argmin(S['A_%g' % T0].iloc[INWIN].values))
if args.test:
    render(k_max)
    fn_out = out_dir / (stem + '_peak.png')
    fig.savefig(fn_out, **still_kw)
    print('TEST: saved %s  (frame %d, %s)'
          % (fn_out, k_max, TT[k_max].strftime('%Y-%m-%d')))
else:
    anim = animation.FuncAnimation(fig, update, frames=len(TT),
                                   interval=1000 / args.fps, blit=False)
    fn_out = out_dir / (stem + VFMT[args.vformat]['ext'])
    anim.save(fn_out, writer=animation.FFMpegWriter(fps=args.fps, **VW),
              savefig_kwargs=SAVE_KW, dpi=args.dpi)
    print('saved %s  (%s, %d frames at %d fps = %.0f s, %.1f MB)'
          % (fn_out, args.vformat, len(TT), args.fps, len(TT) / args.fps,
             fn_out.stat().st_size / 1e6))
    for lbl, k in [('largest', k_max), ('smallest', k_min)]:
        render(k)
        fig.savefig(out_dir / ('%s_%s.png' % (stem, lbl)), **still_kw)
        print('saved %s (%s, %s)' % (out_dir / ('%s_%s.png' % (stem, lbl)),
                                     lbl, TT[k].strftime('%Y-%m-%d')))

# ---- the series behind the movie -------------------------------------------
csv_fn = out_dir / (stem + '_series.csv')
out = S.copy()
out['A_frac_pct'] = AFRAC
out['A_anom_pct'] = A_AN
if QP is not None:
    out['qprism'] = QP
    out['qprism_anom'] = Q_AN
out['animated'] = (out.index >= T_LO) & (out.index <= T_HI)
# labelled by DATE, because the extrema were found on the regular daily axis
# and its noon stamps need not be a frame time on a gappy record
out['springneap'] = ''
d_index = out.index.normalize()
for t_ in SPR_ALL:
    out.loc[d_index == t_.normalize(), 'springneap'] = 'spring'
for t_ in NEP_ALL:
    out.loc[d_index == t_.normalize(), 'springneap'] = 'neap'
out.to_csv(csv_fn)
print('saved %s' % csv_fn)
if len(LAGS):
    LAGS.to_csv(out_dir / (stem + '_lagcorr.csv'))
    print('saved %s' % (out_dir / (stem + '_lagcorr.csv')))
plt.close('all')
