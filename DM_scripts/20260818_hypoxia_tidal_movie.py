"""
Tidal-cycle movie of the BOTTOM hypoxic CONTOUR, on hourly history files.

The twin of 20260818_hypoxia_springneap_movie.py, one band up in frequency.
That script asks whether the hypoxic edge breathes with the fortnight and runs
on Godin-filtered daily fields; this one asks how far the edge is pushed back
and forth within a single tidal cycle, so it has to be hourly and instantaneous
-- a lowpassed field has had exactly this signal removed from it by design.

  left, top     sea surface height, the mean over the --ssh-poly polygon: the
                tidal phase, so every feature below can be read as flood or ebb.
  left, middle  bottom hypoxic area in --series, as nested bands, at hourly
                resolution. Its 25-hour running mean is drawn over it, so the
                tidal wiggle is visibly separated from the subtidal state --
                the wiggle is what this movie is about and the running mean is
                what the spring/neap movie would have shown for the same days.
  left, bottom  HYPOXIC FRONT POSITION: how far along --front-axis the hypoxic
                water reaches inside the series polygon. The area can stay
                nearly constant while the patch slides back and forth, so the
                front is the excursion the area series cannot show.
  right         bottom-cell DO, with the hypoxic contour drawn on it and
                --trail hours of faded previous contours behind it -- at one
                frame per hour a trail of 12 is roughly one semidiurnal cycle
                and draws the excursion as a swept band.

WHAT THE FRONT IS. For each frame the hypoxic cells (< the lowest --thresh) in
the series polygon are projected onto --front-axis and TWO positions are taken,
both in km from the far end of the polygon:

  leading edge   the farthest hypoxic cell. The intuitive measure, and the one
                 the panel title names -- but it saturates. Once the patch
                 reaches the end of the polygon it cannot move any farther, and
                 in Penn Cove in September the hypoxic water is pinned against
                 the mouth for days at a time, so the edge sits flat at the
                 polygon's own length and is measuring the polygon.
  centroid       the area-weighted mean position of the hypoxic cells. It
                 cannot saturate and it responds to the whole patch sliding,
                 which is what a tidal current does to it.

Both are plotted and both go in the CSV. Which one the statistics and the
stills use is decided from the data: if the leading edge sits within one grid
cell of the polygon's far end in more than 20% of frames, the centroid takes
over, and the script says so.

The axis is a compass direction, not a section-normal, so this is a projection
rather than a distance along a curved channel; for Penn Cove, which lies almost
exactly east-west, the two agree to within the grid spacing. --front-axis none
drops the panel for regions where no single direction is meaningful.

WHY ANOMALIES FOR THE STATISTICS (DM 2026.08.06). The tidal excursion rides on
a subtidal state that is itself moving, so the correlations printed below are
computed on the tidal band -- the series minus its own 25-hour centred running
mean -- for the area, the front, SSH, and d(ssh)/dt. On the raw series the
subtidal drift dominates and the tidal relationship comes out weaker than it
is. Both SSH and its time derivative are tested because they answer different
questions: correlation with SSH means the front tracks water LEVEL, while
correlation with d(ssh)/dt means it tracks the tidal CURRENT, which is what
actually advects a patch of low-oxygen water. Expect the current.

WHY THE WINDOW IS SHORT. Four days is about eight semidiurnal cycles, enough to
see the excursion repeat and to see the diurnal inequality in it, and it is 97
hourly files. A week is fine; a month of hourly files is a different kind of
job and the fortnightly question it would be asked to answer is already
answered, properly filtered, by the spring/neap script.

PICKING A WINDOW. 20260818_hypoxia_springneap_movie.py writes a *_series.csv
with a `springneap` column; feeding one of its spring dates and one of its neap
dates into --ds0 here gives two runs of this movie whose excursions can be
compared directly.

Region-plot convention (DM 2026.08.04): with --zoom 0 the window is the
rectangular bounding box of the region polygon plus --pad-cells of margin, and
in either case only cells inside `wb` are drawn.

Runs wherever the history files are (apogee for the 2024-25 run). The 2017
wb1_r0_xn11b days on the Mac make a smoke test.

    python 20260818_hypoxia_tidal_movie.py
    python 20260818_hypoxia_tidal_movie.py --test
    python 20260818_hypoxia_tidal_movie.py --trail 12
    python 20260818_hypoxia_tidal_movie.py --ds0 2025.08.10 --ds1 2025.08.13
    python 20260818_hypoxia_tidal_movie.py --series wb --zoom 0 --region wb_north --front-axis none
    python 20260818_hypoxia_tidal_movie.py --gtx wb1_r0_xn11b --ds0 2017.09.10 --ds1 2017.09.12 --test
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

# house style, shared with the other wb1 movies. Grid on the timeseries, never
# on the map -- DM 2026.08.07.
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
RED = '#e04256'
BLUE = '#4565e8'
GREY = '0.45'
LAND = '#e8e4dc'
COAST = 'k'

# DO classes, low to high: dark red = anoxic, red = hypoxic, warm = low-DO,
# cool = oxygenated. The class edges ARE the band thresholds, so a colour on
# the map and a band in the area panel are the same statement.
LEVEL_COLORS = ['#4a0f16', '#b3202e', '#e8873c', '#f6c667', '#a9cfd4', '#5d9fc0']
OVER_COLOR = '#2f6b96'

DO_MMOL_TO_MGL = 32.0 / 1000.0        # mmol m-3 (uM) -> mg L-1
KM_PER_DEG_LAT = 111.32

# ---- arguments -------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument('--gtx', default='wb1_t0_xn11abbur00')
# Early September 2025: the Penn Cove bed is solidly hypoxic then (the < 2 mg/L
# band is continuous from mid-July to late October in the seasonal movie), so
# there is a contour to watch. Four days = 97 hourly frames = ~8 semidiurnal
# cycles.
p.add_argument('--ds0', default='2025.09.01')
p.add_argument('--ds1', default='2025.09.04')
p.add_argument('--lt', default='hourly0',
               help='hourly0 starts on ocean_his_0001 of --ds0')
p.add_argument('--stride', default=1, type=int,
               help='keep every Nth hourly file (1 = every hour)')
p.add_argument('--region', default='wb_north',
               help='polygon the map window is taken from')
p.add_argument('--exclude', default='skagit_delta',
               help='comma-separated polygons to cut out of the map, and to '
                    'zoom past; empty string keeps the whole region')
p.add_argument('--zoom', default=0.6, type=float,
               help='window = the --zoom-poly box extended EAST by this many '
                    'of its widths. 0 or less falls back to the whole --region '
                    'window')
p.add_argument('--zoom-poly', default='pc', dest='zoom_poly',
               help='polygon the --zoom window is built around')
p.add_argument('--aa', default='',
               help='explicit lon0,lon1,lat0,lat1 window; overrides --zoom')
p.add_argument('--series', default='pc',
               help='polygon the hypoxic area and the front are computed for; '
                    'outlined in red on the map')
p.add_argument('--ssh-poly', default='pc', dest='ssh_poly',
               help='polygon SSH is averaged over for the tidal-phase panel')
p.add_argument('--thresh', default=[2.0, 3.0, 5.0], type=float, nargs='+',
               help='mg/L bands, low to high; the lowest is the CONTOUR drawn '
                    'on the map, the front, and the statistics')
p.add_argument('--front-axis', default='east', dest='front_axis',
               choices=['east', 'west', 'north', 'south', 'none'],
               help='direction the hypoxic front is measured along; none drops '
                    'the panel. east = toward the Penn Cove mouth')
p.add_argument('--trail', default=0, type=int,
               help='draw this many previous HOURS of hypoxic contour, fading '
                    'with age. 12 is about one semidiurnal cycle and sweeps '
                    'out the whole tidal excursion in a single still')
p.add_argument('--sub-window', default=25, type=int, dest='sub_window',
               help='centred running mean [hours] defining the subtidal state; '
                    '25 h removes both tidal bands')
p.add_argument('--max-lag', default=12, type=int, dest='max_lag',
               help='lags [hours] searched in the correlations')
p.add_argument('--cmap', default='levels', choices=['levels', 'oxy'])
p.add_argument('--levels', default=[0.0, 0.5, 2.0, 3.0, 5.0, 7.0, 10.0],
               type=float, nargs='+',
               help='mg/L class edges for --cmap levels')
p.add_argument('--vmin', default=0.0, type=float, help='--cmap oxy only')
p.add_argument('--vmax', default=10.0, type=float, help='--cmap oxy only')
p.add_argument('--pad-cells', default=10, type=int, dest='pad_cells')
p.add_argument('--fps', default=8, type=int)
p.add_argument('--dpi', default=150, type=int,
               help='render dpi for the movie frames -- 100 makes the labels '
                    'mushy once h264 has had a go at them')
p.add_argument('--no-transparent', dest='transparent', action='store_false',
               help='opaque white stills. The saved PNGs are transparent by '
                    'default (standing preference); the movie is always opaque '
                    'because h264 has no alpha channel')
p.add_argument('--vformat', default='mp4', choices=['mp4', 'prores', 'qtrle'])
p.add_argument('--ro', default=None, type=int,
               help='force a roms_out key (2 = /dat2/dakotamm on apogee); '
                    'default searches roms_out2, roms_out1, roms_out')
p.add_argument('--nproc', default=min(8, os.cpu_count() or 1), type=int,
               help='parallel workers for reading history files (1 = serial)')
p.add_argument('--test', dest='test', action='store_true',
               help='save a single still instead of the movie, at the frame '
                    'where the front is farthest along --front-axis')
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
FRONT = args.front_axis != 'none'
out_dir = Ldir['LOo'] / 'DM_outs' / '20260818_hypoxia_tidal_movie'
Lfun.make_dir(out_dir)

# ---- files -----------------------------------------------------------------
# Lfun.get_fn_list builds the hourly names, but which roms_out key holds the
# run differs machine to machine (roms_out2 = /dat2/dakotamm on apogee, plain
# roms_out on the Mac), so the key is chosen by looking for the run.
ROMS_OUT_KEYS = (['roms_out%d' % args.ro] if args.ro is not None
                 else ['roms_out2', 'roms_out1', 'roms_out'])
for key in ROMS_OUT_KEYS:
    base = Ldir.get(key)
    if base is None or str(base).endswith('BLANK'):
        continue
    if (Path(base) / args.gtx).is_dir():
        Ldir['roms_out'] = Path(base)
        print('roms_out: %s (%s)' % (Ldir['roms_out'], key))
        break
else:
    raise SystemExit('%s not found under any of %s'
                     % (args.gtx, ', '.join(ROMS_OUT_KEYS)))

# missing hours are skipped rather than fatal, the same way the lowpassed
# movies skip missing days
fn_list = [fn for fn in Lfun.get_fn_list(args.lt, Ldir, args.ds0, args.ds1)
           if fn.is_file()]
fn_list = fn_list[::max(1, args.stride)]
if not fn_list:
    raise SystemExit('no history files for %s between %s and %s under %s'
                     % (args.gtx, args.ds0, args.ds1, Ldir['roms_out']))
hrs = len(fn_list) * max(1, args.stride)
print('%d hourly frames %s .. %s (stride %d) = %.1f d = %.1f semidiurnal '
      'cycles' % (len(fn_list), args.ds0, args.ds1, args.stride, hrs / 24.0,
                  hrs / 12.42))
if hrs < 26:
    print('  WARNING: under one day of frames -- the 25-hour subtidal mean '
          'below is barely defined and the correlations are not meaningful.')

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

# the series region: the POLYGON intersected with water, independent of the map
# window, so the curve means the same thing whatever the map is showing
RMASK = poly_mask(load_sect(SER)) & wet
IDX = np.where(RMASK)
RAREA = float(cell_area[RMASK].sum())
print('series region %-12s %6d cells, %8.1f km2, mean h %5.1f m, max h %5.1f m'
      % (SER, RMASK.sum(), RAREA / 1e6, h[RMASK].mean(), h[RMASK].max()))

# SSH is averaged over a POLYGON, not over a bounding box (which is what
# 20260806_wbnorth_tidal_movie.py did): inside a cove the two differ by
# whatever coast the box swallows, and this way the SSH panel and the outline
# on the map are the same set of cells.
SMASK = poly_mask(load_sect(args.ssh_poly)) & wet
print('SSH averaged over %s: %d wet cells' % (args.ssh_poly, SMASK.sum()))

# The front axis, as a signed coordinate in km. Distances are measured from the
# far edge of the series polygon, so 0 is the head of the cove for the default
# east axis and the number grows as the hypoxic water advances toward the mouth.
if FRONT:
    lat0 = float(lat[RMASK].mean())
    km_lon = KM_PER_DEG_LAT * np.cos(np.deg2rad(lat0))
    if args.front_axis in ('east', 'west'):
        coord = lon * km_lon
    else:
        coord = lat * KM_PER_DEG_LAT
    if args.front_axis in ('west', 'south'):
        coord = -coord
    C_REG = coord[RMASK]
    C0 = float(C_REG.min())                   # the tail end of the region
    FRONT_MAX = float(C_REG.max()) - C0
    print('front axis %s: %s spans %.2f km along it (0 = the %s end)'
          % (args.front_axis, SER, FRONT_MAX,
             {'east': 'western', 'west': 'eastern',
              'north': 'southern', 'south': 'northern'}[args.front_axis]))
else:
    coord = C0 = FRONT_MAX = None

# ---- map window ------------------------------------------------------------
# Same construction as the other wb1 movies so they frame the same water and
# can be played side by side. Everything here is about what is DRAWN.
keep = wet & in_wb
excl = [s for s in args.exclude.split(',') if s]
for nm in excl:
    keep &= ~poly_mask(load_sect(nm))
in_win = keep & poly_mask(reg)
if not in_win.any():
    raise SystemExit('nothing left after excluding %s from %s'
                     % (excl, args.region))
# largest connected piece, not the outright min/max: cutting skagit_delta out
# of wb_north leaves one stray cell by Deception Pass that on its own stretched
# the map from 84 rows to 144
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

jj = np.where((lat[:, 0] >= aa[2]) & (lat[:, 0] <= aa[3]))[0]
ii = np.where((lon[0, :] >= aa[0]) & (lon[0, :] <= aa[1]))[0]
SUB = (slice(int(jj[0]), int(jj[-1]) + 1), slice(int(ii[0]), int(ii[-1]) + 1))
lon_s, lat_s = lon[SUB], lat[SUB]
plon_s, plat_s = pfun.get_plon_plat(lon_s, lat_s)
draw_s = keep[SUB]
land_s = (~wet)[SUB]
print('window %s -> %d x %d cells, %d drawn'
      % (['%.4f' % v for v in aa], lat_s.shape[0], lat_s.shape[1], draw_s.sum()))


# ---- read one history file -------------------------------------------------
def read_one(fn):
    """(local time, UTC time, bottom DO in the window, series row).

    s_rho index 0 is the BOTTOM cell in ROMS, so oxygen[0, 0] is the bed -- the
    same convention 20260806_hypoxia_reduce.py uses. Only that level is read.

    Land and fill values become NaN before anything is compared to a threshold,
    so a masked cell can never be counted as hypoxic (a fill of 0 would
    otherwise be the most hypoxic water in the domain).
    """
    ds = xr.open_dataset(fn)
    if 'oxygen' not in ds.data_vars:
        ds.close()
        raise SystemExit('no oxygen in %s -- this movie needs the bio fields' % fn)
    bot = ds['oxygen'][0, 0, :, :].values
    zeta = ds['zeta'][0, :, :].values
    t_utc = pd.Timestamp(ds.ocean_time.values[0]).to_pydatetime()
    ds.close()
    bot = np.where(wet & (bot > 0), bot * DO_MMOL_TO_MGL, np.nan)

    i, j = IDX
    v, a_ = bot[i, j], cell_area[i, j]
    row = {'ssh': float(np.nanmean(np.where(SMASK, zeta, np.nan))),
           'do_mean': float(np.nanmean(v)), 'do_min': float(np.nanmin(v))}
    for th in THRESH:
        row['A_%g' % th] = float(a_[v < th].sum())      # NaN < th is False
    if FRONT:
        hyp = v < T0
        # Two measures of where the hypoxic water IS along the axis, both in km
        # from the far end of the region, both NaN when there is none:
        #   front_km      the leading edge -- the farthest hypoxic cell. The
        #                 intuitive one, but it SATURATES: once the patch
        #                 reaches the end of the polygon it cannot move
        #                 farther, and in Penn Cove in September it is pinned
        #                 against the mouth for days at a time.
        #   front_cen_km  the area-weighted centroid of the hypoxic cells. It
        #                 cannot saturate and it responds to the whole patch
        #                 sliding, which is what the tidal current does to it.
        # Which one drives the statistics is decided below, on the actual
        # series, rather than assumed here.
        if hyp.any():
            c_, aa_ = coord[i, j][hyp], a_[hyp]
            row['front_km'] = float(c_.max() - C0)
            row['front_cen_km'] = float((c_ * aa_).sum() / aa_.sum() - C0)
        else:
            row['front_km'] = np.nan
            row['front_cen_km'] = np.nan
    # local for display, UTC kept so this can be matched against tef2 products
    return (pfun.get_dt_local(t_utc).replace(tzinfo=None), t_utc,
            bot[SUB].astype(np.float32), row)


nproc = max(1, min(args.nproc, len(fn_list)))
print('reading %d files on %d process(es)...' % (len(fn_list), nproc))
if nproc > 1:
    ctx = mp.get_context('fork')              # workers inherit the masks above
    with ctx.Pool(nproc) as pool:
        res = pool.map(read_one, fn_list)
else:
    res = [read_one(fn) for fn in fn_list]

TT = pd.to_datetime([r[0] for r in res])                 # local (PST), naive
TT_UTC = pd.to_datetime([r[1] for r in res])
FLD = np.where(draw_s[None, :, :], np.stack([r[2] for r in res]), np.nan)
S = pd.DataFrame([r[3] for r in res], index=TT)
AFRAC = S['A_%g' % T0] / RAREA * 100.0                   # % of the region floor

# ---- what moves, and with what --------------------------------------------
# The tidal band is the series minus its own centred 25-hour running mean. That
# is a running mean, not a Godin filter: over four days Godin's 71-hour span
# would blank most of the record. It leaks a little of the diurnal band, which
# is fine here -- everything it is used for is a within-cycle comparison.
W = args.sub_window


def subtidal(x):
    return x.rolling(W, center=True, min_periods=max(3, W // 2)).mean()


def tidal(x):
    return x - subtidal(x)


SSH = S['ssh']
# d(ssh)/dt as the current proxy: transport scales with the rate of change of
# sea level, not with sea level itself, and it is the current that advects a
# patch of low-oxygen water (DM 2026.08.11).
DSSH = pd.Series(np.gradient(SSH.values, 3600.0), index=TT) * 1e3   # mm/s

print('\nSSH over %s: %.2f to %.2f m (range %.2f m)'
      % (args.ssh_poly, SSH.min(), SSH.max(), SSH.max() - SSH.min()))
print('bottom DO in the window: min %.2f, median %.2f, max %.2f mg/L'
      % (np.nanmin(FLD), np.nanmedian(FLD), np.nanmax(FLD)))
print('%s floor %.1f km2; A_bot < %g mg/L: min %.2f, mean %.2f, max %.2f km2 '
      '(%.0f%% to %.0f%% of the floor)'
      % (SER, RAREA / 1e6, T0, S['A_%g' % T0].min() / 1e6,
         S['A_%g' % T0].mean() / 1e6, S['A_%g' % T0].max() / 1e6,
         100 * S['A_%g' % T0].min() / RAREA, 100 * S['A_%g' % T0].max() / RAREA))
if S['A_%g' % T0].max() <= 0:
    print('  the bed never goes below %g mg/L here -- nothing will be '
          'contoured and the front is undefined. Try a higher --thresh or a '
          'window in the hypoxic season.' % T0)

# the tidal swing itself, which is the quantity the movie exists to show
swing = tidal(AFRAC)
print('\ntidal band (series minus its %d-hour running mean):' % W)
print('  hypoxic area: +/- %.2f %% of floor (sd %.2f), peak-to-peak %.2f %% '
      '= %.2f km2' % (swing.abs().max(), swing.std(),
                      swing.max() - swing.min(),
                      (swing.max() - swing.min()) * RAREA / 1e8))
PRIMARY = 'front_km'
if FRONT:
    for col, nm in [('front_km', 'leading edge'), ('front_cen_km', 'centroid')]:
        fs = tidal(S[col])
        print('  front %-13s along %s: +/- %.2f km (sd %.2f), peak-to-peak '
              '%.2f km, of a %.2f km region'
              % (nm, args.front_axis, fs.abs().max(), fs.std(),
                 fs.max() - fs.min(), FRONT_MAX))
    if S['front_km'].isna().any():
        print('  %d of %d frames have no hypoxic water at all in %s -- both '
              'front measures are NaN there and the panel breaks'
              % (int(S['front_km'].isna().sum()), len(S), SER))
    # The leading edge is the intuitive measure but it saturates against the
    # far end of the polygon; when it spends much of the record within one grid
    # cell of that end it is measuring the polygon, not the water, and the
    # centroid takes over as the series the statistics and the stills use.
    cell_km = float(np.hypot(dx * km_lon, dy * KM_PER_DEG_LAT))
    pinned = float((S['front_km'] > FRONT_MAX - cell_km).mean())
    if pinned > 0.2:
        PRIMARY = 'front_cen_km'
        print('  the leading edge is within one cell (%.2f km) of the far end '
              'of %s in %.0f%% of frames -- it is SATURATED, so the centroid '
              'is used for the statistics and the stills below. Both are '
              'plotted and both are in the CSV.' % (cell_km, SER, 100 * pinned))
    else:
        print('  the leading edge is unsaturated (%.0f%% of frames within one '
              'cell of the far end of %s) and is the primary series'
              % (100 * pinned, SER))


def lag_table(y, name):
    """Correlation of a tidal-band series against SSH and d(ssh)/dt, by lag.

    Positive lag = y FOLLOWS the forcing by that many hours. n is the nominal
    count; no n_eff correction is applied here because within a few days there
    are only a handful of independent tidal cycles and the honest statement is
    the lag structure and the amplitude, not a p-value.
    """
    yt = tidal(y)
    rows = []
    for L in range(-args.max_lag, args.max_lag + 1):
        r = {}
        for lbl, x in [('ssh', tidal(SSH)), ('dssh_dt', tidal(DSSH))]:
            d = pd.concat([yt.shift(-L), x], axis=1).dropna()
            r[lbl] = (float(np.corrcoef(d.iloc[:, 0], d.iloc[:, 1])[0, 1])
                      if len(d) > 5 else np.nan)
        rows.append(dict(lag_hours=L, **r))
    tab = pd.DataFrame(rows).set_index('lag_hours')
    print('\n%s, tidal band, vs the tide (positive lag = %s follows):'
          % (name, name))
    if not tab.notna().any().any():
        print('  no correlation to report -- the series is constant or all '
              'NaN over this window')
        return tab
    for lbl in ['ssh', 'dssh_dt']:
        if tab[lbl].notna().any():
            kb = tab[lbl].abs().idxmax()
            print('  vs %-8s strongest r = %+.3f at lag %+d h'
                  % (lbl, tab[lbl].loc[kb], kb))
    return tab


FRONT_NAME = {'front_km': 'front (leading edge)',
              'front_cen_km': 'front (centroid)'}[PRIMARY]
LAG_A = lag_table(AFRAC, 'hypoxic area')
LAG_F = lag_table(S[PRIMARY], FRONT_NAME) if FRONT else pd.DataFrame()
print('\n  a stronger correlation with d(ssh)/dt than with ssh means the patch '
      'is being ADVECTED by the tidal current rather than tracking water '
      'level; that is the expected result and the reason both are printed.')

# ---- figure ----------------------------------------------------------------
# constrained layout, not manual margins: the map carries a fixed aspect (dar),
# so it shrinks inside its own gridspec cell and a hand-placed colorbar ends up
# stranded in the whitespace.
# The front panel is built only when there IS a front: an axis switched off
# still holds its row, and --front-axis none otherwise leaves a third of the
# figure empty.
plt.close('all')
nrow = 3 if FRONT else 2
fig = plt.figure(figsize=(17.5, 10 if FRONT else 8), layout='constrained')
gs = fig.add_gridspec(nrow, 2, width_ratios=[1, 1.05])
axs = fig.add_subplot(gs[0, 0])                    # SSH
axa = fig.add_subplot(gs[1, 0], sharex=axs)        # hypoxic area
axf = fig.add_subplot(gs[2, 0], sharex=axs) if FRONT else None
axm = fig.add_subplot(gs[:, 1])                    # map
LEFT = [a for a in (axs, axa, axf) if a is not None]

# --- map
LEVELS = sorted(args.levels)
if args.cmap == 'levels':
    if len(LEVELS) - 1 > len(LEVEL_COLORS):
        raise SystemExit('--levels asks for %d classes; only %d colours are '
                         'defined' % (len(LEVELS) - 1, len(LEVEL_COLORS)))
    cmap = ListedColormap(LEVEL_COLORS[:len(LEVELS) - 1])
    cmap.set_over(OVER_COLOR)
    norm = BoundaryNorm(LEVELS, cmap.N)
    # spacing='proportional' keeps the colourbar a ruler in mg/L
    cb_kw = dict(extend='max', spacing='proportional', ticks=LEVELS)
    lo, hi = LEVELS[0], LEVELS[-1]
else:
    cmap, norm = cm.oxy, Normalize(vmin=args.vmin, vmax=args.vmax)
    lo, hi = args.vmin, args.vmax
    cb_kw = dict(ticks=sorted(set([lo] + list(THRESH) + [hi])))
axm.pcolormesh(plon_s, plat_s,
               np.ma.masked_where(~land_s, np.ones(land_s.shape)),
               cmap=ListedColormap([LAND]), shading='flat', zorder=0)
cs = axm.pcolormesh(plon_s, plat_s, FLD[0], cmap=cmap, norm=norm,
                    shading='flat', zorder=1)
f_lo = float(np.nanmean(FLD < lo))
f_hi = float(np.nanmean(FLD > hi))
cb_kw.setdefault('extend', 'both' if (f_lo > 0 and f_hi > 0) else
                 'max' if f_hi > 0 else 'min' if f_lo > 0 else 'neither')
fig.colorbar(cs, ax=axm, shrink=0.75, pad=0.01, aspect=35,
             label='bottom-cell dissolved oxygen [mg L$^{-1}$]', **cb_kw)

pfun.add_coast(axm, color=COAST, linewidth=0.8)
s0 = load_sect(SER)
axm.plot(np.append(s0.x.values, s0.x.values[0]),
         np.append(s0.y.values, s0.y.values[0]), '-',
         color=RED, lw=2.0, zorder=8)

DEC = 1 if (aa[1] - aa[0]) >= 1.0 else 2


def nice_ticks(v0, v1, n, dec):
    """n evenly spaced round values, both ends INSIDE [v0, v1]."""
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
# Drawn even though T0 is a colour-class edge by default: here the edge is the
# subject, and a black line reads as an object that moves in a way a boundary
# between two fills does not. At one frame per hour, --trail 12 sweeps out a
# whole semidiurnal excursion in one still.
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
        age = n - 1 - k                      # 0 = the current hour
        if age == 0:
            c.set_alpha(1.0)
            c.set_linewidth(CLW)
        else:
            c.set_alpha(max(0.10, 0.55 * (1 - age / (args.trail + 1.0))))
            c.set_linewidth(0.9)


# --- SSH: the tidal phase every other panel is read against
axs.plot(TT, SSH.values, '-', color=BLUE, lw=1.8)
axs.axhline(0, color='0.5', lw=0.8)
axs.set_ylabel('SSH [m]')
axs.set_title('tidal phase -- sea surface height over %s (mean of %d cells)'
              % (args.ssh_poly, SMASK.sum()), fontsize=10, color=BLUE)
axs.grid(**GRID)

# --- hypoxic area, nested bands + the subtidal state it wiggles around
def band_colour(th):
    if args.cmap != 'levels':
        return LEVEL_COLORS[min(THRESH.index(th), len(LEVEL_COLORS) - 1)]
    k = int(np.searchsorted(LEVELS, th, side='left')) - 1
    return LEVEL_COLORS[min(max(k, 0), len(LEVELS) - 2)]


for k, th in enumerate(reversed(THRESH)):             # widest band first
    axa.fill_between(TT, 0, S['A_%g' % th] / RAREA * 100.0,
                     color=band_colour(th), lw=0,
                     label='< %g mg L$^{-1}$' % th, zorder=2 + k)
# the subtidal state, so the tidal wiggle is visibly a wiggle ON something
axa.plot(TT, subtidal(AFRAC).values, '-', color='k', lw=1.4, zorder=10,
         label='%d-h mean of < %g' % (W, T0))
axa.set_ylabel('bottom area [% of floor]')
axa.set_ylim(0, 100)
axa.grid(**GRID)
axa.legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=len(THRESH) + 1)
axa.set_title('bottom hypoxic area, %s (%.1f km$^2$ of sea floor, outlined in '
              'red on the map)' % (SER, RAREA / 1e6), fontsize=11, color=RED)
km2 = RAREA / 1e6
sax = axa.secondary_yaxis('right', functions=(lambda p_: p_ * km2 / 100,
                                              lambda a_: a_ * 100 / km2))
sax.set_ylabel('[km$^2$]')

# --- front position: the excursion the area series cannot show
if FRONT:
    # both measures, with the one the statistics use drawn solid. The dashed
    # one is kept on the same axis on purpose: seeing the edge flat while the
    # centroid swings is the clearest statement that the patch is sliding
    # inside a polygon it already fills.
    for col, sty, lab in [('front_km', '-', 'leading edge'),
                          ('front_cen_km', '--', 'centroid')]:
        axf.plot(TT, S[col].values, sty, color=RED, lw=1.8 if col == PRIMARY else 1.2,
                 alpha=1.0 if col == PRIMARY else 0.7, label=lab)
    axf.plot(TT, subtidal(S[PRIMARY]).values, '-', color='k', lw=1.2, alpha=0.8,
             label='%d-h mean' % W)
    axf.set_ylabel('front along %s [km]' % args.front_axis)
    # a little headroom above the polygon's own length: a saturated leading
    # edge sits exactly at FRONT_MAX and would be hidden under the top spine,
    # which is the one case the reader most needs to see
    axf.set_ylim(0, FRONT_MAX * 1.08)
    axf.legend(loc='lower left', fontsize=9, framealpha=0.9, ncol=3)
    kb = (LAG_F['dssh_dt'].abs().idxmax()
          if len(LAG_F) and LAG_F['dssh_dt'].notna().any() else None)
    # two lines: this axes is only as wide as the left column and a one-line
    # version of this title runs out over the map
    axf.set_title('hypoxic front: how far %s the < %g mg L$^{-1}$ water reaches '
                  'in %s\n%s'
                  % (args.front_axis, T0, SER,
                     ('%s, tidal band vs d(ssh)/dt: r = %+.2f at lag %+d h'
                      % (FRONT_NAME, LAG_F['dssh_dt'].loc[kb], kb))
                     if kb is not None else 'no tidal correlation to report'),
                  fontsize=10, color=RED)
    axf.grid(**GRID)

for ax in LEFT:
    ax.set_xlim(TT[0], TT[-1])
tail = LEFT[-1]
for ax in LEFT[:-1]:                      # only the bottom panel keeps its dates
    plt.setp(ax.get_xticklabels(), visible=False)
tail.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=9))
tail.xaxis.set_major_formatter(mdates.DateFormatter('%b %-d %H:%M'))
tail.set_xlabel('local time (PST)')
for l in tail.get_xticklabels():
    l.set_rotation(30)
    l.set_horizontalalignment('right')

# moving markers, one per left panel
MK = [ax.axvline(TT[0], color='k', lw=1.5, zorder=20) for ax in LEFT]
dot = axs.plot([TT[0]], [SSH.iloc[0]], 'o', ms=7, color='k', zorder=21)[0]

if args.transparent:
    fig.patch.set_alpha(0.0)
    for ax in [axm] + LEFT:
        ax.patch.set_alpha(0.0)


def update(fi):
    cs.set_array(FLD[fi].ravel())
    draw_edge(fi)
    a_ = S['A_%g' % T0].iloc[fi]
    ttl.set_text('bottom oxygen -- %s (PST)\n%s below %g mg L$^{-1}$: %.1f '
                 'km$^2$ (%.0f%% of the %s floor)   |   SSH %+.2f m%s'
                 % (TT[fi].strftime('%Y-%m-%d %H:%M'), SER, T0, a_ / 1e6,
                    100 * a_ / RAREA, SER, SSH.iloc[fi],
                    ('   |   front (%s) %.2f km'
                     % ('edge' if PRIMARY == 'front_km' else 'centroid',
                        S[PRIMARY].iloc[fi]))
                    if FRONT and np.isfinite(S[PRIMARY].iloc[fi]) else ''))
    for m in MK:
        m.set_xdata([TT[fi], TT[fi]])
    dot.set_data([TT[fi]], [SSH.iloc[fi]])
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


stem = ('20260818_hypoxia_tidal_%s_%s%s_%s_%s'
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

# Stills at the two ends of the excursion -- the frames the movie exists to
# compare. On the front when there is one, on the area otherwise; frame 0 of a
# tidal movie is an arbitrary phase and says nothing.
key = S[PRIMARY] if (FRONT and S[PRIMARY].notna().any()) else AFRAC
k_max, k_min = int(np.nanargmax(key.values)), int(np.nanargmin(key.values))
if args.test:
    render(k_max)
    fn_out = out_dir / (stem + '_farthest.png')
    fig.savefig(fn_out, **still_kw)
    print('\nTEST: saved %s  (frame %d, %s)'
          % (fn_out, k_max, TT[k_max].strftime('%Y-%m-%d %H:%M')))
else:
    anim = animation.FuncAnimation(fig, update, frames=len(TT),
                                   interval=1000 / args.fps, blit=False)
    fn_out = out_dir / (stem + VFMT[args.vformat]['ext'])
    anim.save(fn_out, writer=animation.FFMpegWriter(fps=args.fps, **VW),
              savefig_kwargs=SAVE_KW, dpi=args.dpi)
    print('\nsaved %s  (%s, %d frames at %d fps = %.0f s, %.1f MB)'
          % (fn_out, args.vformat, len(TT), args.fps, len(TT) / args.fps,
             fn_out.stat().st_size / 1e6))
    for lbl, k in [('farthest', k_max), ('nearest', k_min)]:
        render(k)
        fig.savefig(out_dir / ('%s_%s.png' % (stem, lbl)), **still_kw)
        print('saved %s (%s, %s)' % (out_dir / ('%s_%s.png' % (stem, lbl)),
                                     lbl, TT[k].strftime('%Y-%m-%d %H:%M')))

# ---- the series behind the movie -------------------------------------------
csv_fn = out_dir / (stem + '_series.csv')
out = S.copy()
out['time_utc'] = TT_UTC
out['A_frac_pct'] = AFRAC
out['A_tidal_pct'] = tidal(AFRAC)
out['dssh_dt_mm_s'] = DSSH
if FRONT:
    out['front_tidal_km'] = tidal(S['front_km'])
    out['front_cen_tidal_km'] = tidal(S['front_cen_km'])
out.to_csv(csv_fn)
print('saved %s' % csv_fn)
LAG_A.to_csv(out_dir / (stem + '_lagcorr_area.csv'))
print('saved %s' % (out_dir / (stem + '_lagcorr_area.csv')))
if FRONT:
    LAG_F.to_csv(out_dir / (stem + '_lagcorr_front.csv'))
    print('saved %s' % (out_dir / (stem + '_lagcorr_front.csv')))
plt.close('all')
