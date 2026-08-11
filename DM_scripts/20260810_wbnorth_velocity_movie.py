"""
Tidal-cycle movie of the wb_north DEPTH-AVERAGED CURRENT, with the pc_lp
velocity cross-section and Penn Cove SSH beside it.

Same figure format as 20260806_wbnorth_tidal_movie.py (the salinity version) --
it is a copy, deliberately, so that one can keep running unchanged.

Layout (one figure, animated over hourly history files):
  right       -- map of depth-averaged speed over wb_north with current arrows
                 over the top, the bounding box of the `pc` polygon as a dotted
                 grey box, and the pc_lp section picked out in red
  upper left  -- SSH averaged over that dotted box (tidal phase), carrying a
                 marker showing where the animation is
  lower left  -- the pc_lp cross-section (--sect-var, default u = the
                 section-normal velocity), framed in red to tie it to the line
                 on the map

The map is ubar/vbar, the model's own depth-averaged transport velocities, so
it is the BAROTROPIC (tidal) flow -- a single vector for the whole water
column. The two-layer exchange does not appear there by construction; that is
what the cross-section panel is for. The two panels are answering different
questions and are deliberately NOT on a shared colour scale: the map is a
speed (a magnitude, >= 0) and the section is a signed normal velocity.

The dotted box does double duty: it is what is drawn on the map AND what SSH is
averaged over, so there is only one rectangle in the figure and it means one
thing. (It replaced PENN_COVE_BOX from wb1_penncove_region.py, which was a
hand-tuned rectangle that did not match any drawn outline.)

Region-plot convention (DM 2026.08.04): the window is the rectangular bounding
box of the WHOLE wb_north polygon plus PAD_CELLS of margin, but only cells
inside `wb` are drawn -- otherwise Admiralty Inlet and the main Puget Sound
trench come along on the far side of Whidbey and take over the color scale.

Runs on apogee (needs the history files, plus extractions_avg for the section).
Defaults to the first week of September 2025 = 169 hourly frames.

    python 20260810_wbnorth_velocity_movie.py
    python 20260810_wbnorth_velocity_movie.py --test
    python 20260810_wbnorth_velocity_movie.py --quiver-step 5   # denser arrows
    python 20260810_wbnorth_velocity_movie.py --sect-var salt   # salt section
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
from matplotlib.colors import Normalize
from scipy import ndimage
from matplotlib.ticker import MaxNLocator
from cmocean import cm

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

# house style, same as the pc_mouth analysis figures. Grid on the timeseries,
# never on the map -- DM 2026.08.10.
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
RED = '#e04256'
BLUE = '#4565e8'
GREY = '0.45'

# ---- arguments -------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument('--gtx', default='wb1_t0_xn11abbur00')
p.add_argument('--ro', default=2, type=int)              # /dat2/dakotamm/LO_roms
p.add_argument('--ds0', default='2025.09.01')
p.add_argument('--ds1', default='2025.09.07')            # 1 week -> 169 hourly frames
p.add_argument('--lt', default='hourly0')                # clean hour-0 start on ds0
p.add_argument('--region', default='wb_north', help='polygon that sets the map window')
p.add_argument('--exclude', default='skagit_delta',
               help='comma-separated polygons to cut out of the map, and to '
                    'zoom past; empty string keeps the whole region')
p.add_argument('--zoom', default=1.0, type=float,
               help='window = the pc box extended EAST by this many pc-widths, '
                    'so 1.0 reaches about halfway up Saratoga Passage. 0 or '
                    'less falls back to the whole --region window')
p.add_argument('--aa', default='',
               help='explicit lon0,lon1,lat0,lat1 window; overrides --zoom')
p.add_argument('--quiver-step', default=4, type=int, dest='quiver_step',
               help='draw one arrow every N grid cells; arrow length scales '
                    'with it so density stays readable')
p.add_argument('--quiver-scale', type=float, dest='quiver_scale',
               help='data units per axes width; default ties a 95th-pctl '
                    'arrow to ~1.2 grid steps')
p.add_argument('--quiver-key', type=float, dest='quiver_key',
               help='reference arrow magnitude [m/s]; default 95th pctl speed')
p.add_argument('--pc-poly', default='pc', dest='pc_poly',
               help='polygon whose bounding box is drawn in red and used for SSH')
p.add_argument('--sect', default='pc_lp',
               help='tef2 section for the velocity cross-section panel')
p.add_argument('--sect-var', default='u', dest='sect_var',
               choices=['salt', 'temp', 'u'],
               help='what to show in the cross-section; u is the '
                    'section-normal velocity, q/(dd*DZ)')
p.add_argument('--no-vel', dest='vel', action='store_false',
               help='skip the cross-section panel')
p.add_argument('--vscale', type=float,
               help='symmetric colour limit for --sect-var u [m/s]')
p.add_argument('--sect-vmin', type=float, dest='sect_vmin',
               help='cross-section colour limits; default is the SECTION own '
                    'percentiles, which do not match the map scale')
p.add_argument('--sect-vmax', type=float, dest='sect_vmax')
p.add_argument('--vmin', type=float)                     # default: percentiles in window
p.add_argument('--vmax', type=float)
p.add_argument('--pmax', default=99.5, type=float)
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

out_dir = Ldir['LOo'] / 'DM_outs' / '20260810_wbnorth_velocity_movie'
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
sect_line = load_sect(args.sect)              # drawn on the map in BLUE
in_wb = poly_mask(load_sect('wb'))            # master clip, every wb1 region plot

# Cells to draw: inside wb, minus anything in --exclude. The standing region
# convention is "extent from the region polygon, content clipped to wb"; this
# adds a subtraction, so the window is taken from the cells that SURVIVE
# rather than from the region outline -- otherwise dropping the Skagit delta
# would leave a big empty rectangle where it used to be.
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
# behind -- with skagit_delta removed from wb_north exactly ONE cell survives
# up by Deception Pass, and letting it set the northern edge stretched the map
# to 144 rows instead of 84 for a single cell. Drawing still uses the full
# `keep`, so nothing inside the window is hidden; only the extent is robustified.
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

# Penn Cove zoom. Built from the pc box rather than hard-coded degrees so it
# follows the polygon if she redraws it. The cove opens EAST into Saratoga
# Passage, so the window is extended asymmetrically: --zoom pc-widths to the
# east, and only a little west, where the cove already ends at its head.
if args.aa:
    aa = [float(s) for s in args.aa.split(',')]
    print('window: explicit --aa')
elif args.zoom > 0:
    w = float(pc.x.max() - pc.x.min())
    h = float(pc.y.max() - pc.y.min())
    aa = [pc.x.min() - 0.10 * w, pc.x.max() + args.zoom * w,
          pc.y.min() - 0.60 * h, pc.y.max() + 0.90 * h]
    inzoom = (keep & (lon >= aa[0]) & (lon <= aa[1])
              & (lat >= aa[2]) & (lat <= aa[3]))
    print('zoom %.2f pc-widths east of the cove -> %d of %d kept cells (%.0f%%)'
          % (args.zoom, inzoom.sum(), keep.sum(),
             100 * inzoom.sum() / max(1, keep.sum())))
if excl:
    print('excluding %s from %s -> %d cells kept (was %d)'
          % (', '.join(excl), args.region, in_win.sum(),
             (wet & in_wb & poly_mask(reg)).sum()))

# index bounds of that window, so the workers only ship back the subset
jj = np.where((lat[:, 0] >= aa[2]) & (lat[:, 0] <= aa[3]))[0]
ii = np.where((lon[0, :] >= aa[0]) & (lon[0, :] <= aa[1]))[0]
j0, j1 = int(jj[0]), int(jj[-1]) + 1
i0, i1 = int(ii[0]), int(ii[-1]) + 1
SUB = (slice(j0, j1), slice(i0, i1))
lon_s, lat_s = lon[SUB], lat[SUB]
plon_s, plat_s = pfun.get_plon_plat(lon_s, lat_s)
draw_s = keep[SUB]                            # wb-clipped, minus --exclude
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
    """Depth-averaged current in the window, and the box-mean SSH.

    ubar and vbar live on the u and v faces, so they are averaged onto the rho
    points they straddle -- the same staggering LO assumes everywhere. The
    shaded field is the speed; the components are carried through for arrows.

    ubar/vbar are the model's own DEPTH-AVERAGED transport velocities, not a
    surface slice, so this is the barotropic (tidal) flow. The two-layer
    exchange is in the cross-section panel, not here.
    """
    ds = xr.open_dataset(fn)
    for vn in ('ubar', 'vbar'):
        if vn not in ds.data_vars:
            ds.close()
            raise SystemExit('no %s in %s -- this movie needs ubar/vbar' % (vn, fn))
    ub = ds.ubar[0, :, :].values                     # (eta_rho, xi_u)
    vb = ds.vbar[0, :, :].values                     # (eta_v,  xi_rho)
    ur = np.full(lon.shape, np.nan)
    vr = np.full(lon.shape, np.nan)
    ur[:, 1:-1] = 0.5 * (ub[:, :-1] + ub[:, 1:])
    ur[:, 0], ur[:, -1] = ub[:, 0], ub[:, -1]
    vr[1:-1, :] = 0.5 * (vb[:-1, :] + vb[1:, :])
    vr[0, :], vr[-1, :] = vb[0, :], vb[-1, :]
    uq = ur[SUB].astype(np.float32)
    vq = vr[SUB].astype(np.float32)
    fld = np.hypot(uq, vq)
    zeta = ds.zeta[0, :, :].values
    ssh = float(np.nanmean(np.where(in_box, zeta, np.nan)))
    t_utc = pd.Timestamp(ds.ocean_time.values[0]).to_pydatetime()
    ds.close()
    # local for display, UTC kept so the tef2 extraction can be matched on a
    # clock that has no daylight-saving step in it
    return (fld, ssh, pfun.get_dt_local(t_utc).replace(tzinfo=None), t_utc,
            uq, vq)


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
TT_UTC = pd.to_datetime([r[3] for r in res])
FLD = np.where(draw_s[None, :, :], FLD, np.nan)          # land + outside-wb
UQ = np.where(draw_s[None, :, :], np.stack([r[4] for r in res]), np.nan)
VQ = np.where(draw_s[None, :, :], np.stack([r[5] for r in res]), np.nan)
print('depth-averaged speed: median %.3f, 95th pctl %.3f, max %.3f m/s'
      % (np.nanmedian(FLD), np.nanpercentile(FLD, 95), np.nanmax(FLD)))
print('SSH %.2f to %.2f m   (range %.2f m)'
      % (SSH.min(), SSH.max(), SSH.max() - SSH.min()))


# ---- cross-section from the existing tef2 extraction -----------------------
def load_section_field():
    """--sect-var as (time, z, p) at --sect, on the movie's clock.

    No new extraction: extract_sections_avg.py already stored salt, q and DZ
    per face, so salinity comes straight out and velocity is q / (dd * DZ).
    Returns None if the extraction is not there.

    The tef2 files come from ocean_avg (hourly MEANS) while the movie frames
    are ocean_his (instantaneous), so the two clocks can sit up to half an hour
    apart. Frames are matched to the nearest extraction time and the worst
    offset is reported rather than assumed to be zero.
    """
    tdir = Ldir['LOo'] / 'extract' / args.gtx / 'tef2'
    cand = sorted(tdir.glob('extractions_avg_*/%s.nc' % args.sect))
    if not cand:
        print('no extractions_avg_*/%s.nc under %s -- skipping the velocity '
              'panel' % (args.sect, tdir))
        return None
    ds = xr.open_dataset(cand[0])
    te = pd.to_datetime(ds.time.values)
    k = te.get_indexer(TT_UTC, method='nearest')
    off = np.abs((te[k] - TT_UTC).total_seconds()) / 60.0
    print('cross-section %s from %s' % (args.sect_var, cand[0].name))
    print('  matched %d frames, worst clock offset %.0f min' % (len(k), off.max()))
    if off.max() > 65:
        print('  WARNING: the extraction does not cover this window well')
    DZ = ds.DZ.values[k]                     # (nt, z, p)  m
    dd = ds.dd.values                        # (p)         m
    hh = ds.h.values                         # (p)         m
    if args.sect_var == 'u':
        with np.errstate(invalid='ignore', divide='ignore'):
            fld = ds.q.values[k] / (dd[None, None, :] * DZ)
    elif args.sect_var in ds.data_vars:
        fld = ds[args.sect_var].values[k]
    else:
        print('  %r not in %s -- skipping the panel'
              % (args.sect_var, cand[0].name))
        ds.close()
        return None
    ds.close()
    # z from the bed by integrating the same DZ the cells were built on, so
    # depths are exactly consistent with the data
    zw = np.concatenate([np.zeros_like(DZ[:, :1, :]),
                         np.cumsum(DZ, axis=1)], axis=1) - hh[None, None, :]
    return dict(fld=fld, zw=zw, dd=dd, h=hh)


VEL = load_section_field() if args.vel else None

# Colour limits from the drawn cells only, fixed for the whole movie -- an
# auto-scaled frame would make the tide look like a colour-table change.
#
# Salinity is capped at 27 by default rather than run to the 99.5th percentile.
# Whidbey Basin sits in a narrow salty band, so a full-range scale spends most
# of the colour table on water that never changes; capping saturates that and
# hands ~78% of the table to the sub-25 range where the Skagit plume lives.
# Everything above 27 goes one colour -- the colourbar is drawn with an arrow
# to say so. Per-variable, so --var temp / oxygen are unaffected.
# Speed is a magnitude, so the scale starts at zero -- a percentile floor
# would make slack water look like a current. cm.speed is the cmocean map
# built for exactly this: perceptually uniform and light at zero.
v = FLD[np.isfinite(FLD)]
vmin = args.vmin if args.vmin is not None else 0.0
vmax = (args.vmax if args.vmax is not None
        else float(np.percentile(v, args.pmax)))
CMAP = cm.speed
UNITS = 'm s$^{-1}$'
norm = Normalize(vmin=vmin, vmax=vmax)
print('colour scale: linear, %.3f to %.3f m/s' % (vmin, vmax))
print('  data in window: min %.3f, median %.3f, %gth pctl %.3f, max %.3f'
      % (v.min(), np.median(v), args.pmax, vmax, v.max()))
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
fig = plt.figure(figsize=(17.5, 9), layout='constrained')
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.05])
axs = fig.add_subplot(gs[0, 0])                  # SSH, upper left
axv = fig.add_subplot(gs[1, 0])                  # cross-section, lower left
axm = fig.add_subplot(gs[:, 1])                  # map, right

# --- map
cs = axm.pcolormesh(plon_s, plat_s, FLD[0], cmap=CMAP, norm=norm,
                    shading='flat', zorder=1)
cb = fig.colorbar(cs, ax=axm, shrink=0.75, pad=0.01, aspect=35,
                  extend=CB_EXT,
                  label='depth-averaged speed [%s]' % UNITS)

# Arrows on a subsample of the rho grid. The scale is fixed from the whole
# record, NOT autoscaled from frame 0 -- frame 0 can land on slack water, and
# an arrow length that meant something different each frame would be useless.
qs = max(1, args.quiver_step)
spd95 = float(np.nanpercentile(FLD, 95))
# Arrow LENGTH is tied to arrow SPACING, so --quiver-step alone is enough to
# change density: a typical (95th pctl) arrow spans ~1.2 grid steps whatever
# the step is. With a fixed scale, halving the step doubles the overlap and
# the map turns into a solid mat of arrowheads.
nx_win = lon_s.shape[1]
qscale = (args.quiver_scale if args.quiver_scale
          else spd95 * nx_win / (1.2 * qs))
qkey = args.quiver_key if args.quiver_key else round(spd95, 2)
# white arrows with a black outline: cm.speed runs pale yellow at slack to
# near-black at full flood, and a single arrow colour is invisible at one end
# or the other. The outline carries them over the pale water, the white fill
# over the dark.
Q = axm.quiver(lon_s[::qs, ::qs], lat_s[::qs, ::qs],
               UQ[0][::qs, ::qs], VQ[0][::qs, ::qs],
               scale=qscale, scale_units='width', units='width',
               width=0.0026, color='w', edgecolor='k', linewidth=0.5,
               zorder=6)
axm.quiverkey(Q, 0.86, 0.035, qkey, '%.2f m s$^{-1}$' % qkey,
              labelpos='E', coordinates='axes', fontproperties=dict(size=9))
print('quiver: every %d cells, scale %.2f (95th pctl speed %.3f m/s)'
      % (qs, qscale, spd95))

pfun.add_coast(axm, color='gray', linewidth=0.5)
# Penn Cove: dotted grey box (this is also what SSH is averaged over), with
# the pc_lp side picked out in red -- drawn as the actual section line rather
# than the box edge, so the red shows the section's true extent
pfun.draw_box(axm, box, color=GREY, linewidth=1.5, linestyle=':')
if args.vel:
    axm.plot(sect_line.x, sect_line.y, '-', color=RED, lw=3.0, zorder=9,
             solid_capstyle='butt')
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
axs.set_title('tidal phase (mean over the dotted box)', fontsize=10, color=GREY)
axs.grid(**GRID)
axs.set_xlim(TT[0], TT[-1])
axs.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
axs.set_xlabel('local time (PST)')
for l in axs.get_xticklabels():
    l.set_rotation(30)
    l.set_horizontalalignment('right')

# --- velocity cross-section
csv_ = None
if VEL is not None:
    # x is distance across the section, built from the face widths themselves
    x_e = np.concatenate([[0.0], np.cumsum(VEL['dd'])]) / 1000.0     # km
    # The mesh is drawn on the TIME-MEAN sigma depths and only the colours are
    # updated per frame. Rebuilding the mesh every frame would track the
    # surface exactly, but zeta is a few percent of the 20 m depth here and a
    # mesh that breathes makes the velocity structure much harder to read.
    zw_m = VEL['zw'].mean(axis=0)                                    # (nz+1, np)
    uu = VEL['fld'][np.isfinite(VEL['fld'])]
    if args.sect_var == 'u':
        # diverging, symmetric about zero -- the sign IS the signal
        vs = args.vscale if args.vscale else float(np.percentile(np.abs(uu), 99))
        s0, s1, scmap = -vs, vs, cm.balance
        slab = 'u [m s$^{-1}$]'
        stitle = '%s velocity: red = out of the cove, blue = in' % args.sect
    else:
        # Same scale as the map when it is the same variable, so the two
        # panels are directly comparable. NOTE the consequence: the map is
        # capped to bring out the Skagit plume, and the cove is saltier than
        # that cap, so most of the section can sit at the top of the scale.
        # The saturation fraction is printed below -- if it is near 100% the
        # section is one flat colour and you want --sect-vmax (or a higher
        # --vmax on both).
        # No sharing with the map here: the map is a SPEED (a magnitude, >= 0)
        # and this is a scalar on a different range entirely, so one scale
        # across the two panels would be meaningless.
        s0 = (args.sect_vmin if args.sect_vmin is not None
              else float(np.percentile(uu, 0.5)))
        s1 = (args.sect_vmax if args.sect_vmax is not None
              else float(np.percentile(uu, 99.5)))
        scmap = {'salt': cm.haline, 'temp': cm.thermal}[args.sect_var]
        slab = '%s [%s]' % (args.sect_var,
                            'g kg$^{-1}$' if args.sect_var == 'salt'
                            else '$^{\\circ}$C')
        stitle = '%s %s cross-section' % (args.sect, args.sect_var)
    print('  cross-section colour limits %.3f to %.3f%s'
          % (s0, s1, '  (shared with the map)'
             if args.sect_var != 'u' and s0 == vmin and s1 == vmax else ''))
    f_sat = float(np.mean(uu > s1))
    print('  cross-section saturated: %.1f%% above vmax, %.1f%% below vmin'
          % (100 * f_sat, 100 * float(np.mean(uu < s0))))
    if f_sat > 0.9:
        print('  WARNING: %.0f%% of the section is at the top of the scale -- '
              'it will render as one flat colour. Raise --sect-vmax.'
              % (100 * f_sat))
    # One quadmesh PER FACE. A single mesh spanning all faces has to
    # interpolate the sigma levels across columns of different depth, which
    # smears the water column past the stepped bathymetry -- the shading ends
    # up below the bed. Drawn column by column, each face keeps its own levels
    # and the mesh lands exactly on h.
    csv_ = []
    for ip in range(zw_m.shape[1]):
        Xp = np.tile([x_e[ip], x_e[ip + 1]], (zw_m.shape[0], 1))
        Yp = np.repeat(zw_m[:, ip:ip + 1], 2, axis=1)
        csv_.append(axv.pcolormesh(Xp, Yp, VEL['fld'][0][:, ip:ip + 1],
                                   cmap=scmap, vmin=s0, vmax=s1,
                                   shading='flat'))
    cbv = fig.colorbar(csv_[0], ax=axv, pad=0.01, aspect=12, label=slab)
    axv.plot(x_e, -np.concatenate([VEL['h'][:1], VEL['h']]), '-',
             color='0.3', lw=1.0, drawstyle='steps-pre')
    axv.set_xlim(x_e[0], x_e[-1])
    axv.set_xlabel('distance across %s [km]  (north on the left)' % args.sect)
    axv.set_ylabel('z [m]')
    # for u the sign convention is not assumed -- q runs minus-side to
    # plus-side, which at pc_lp is eastward, i.e. OUT of Penn Cove
    axv.set_title(stitle, fontsize=10, color=RED)
    # frame this panel in the same red as the section line on the map, so the
    # section and its location read as one object
    for sp in axv.spines.values():
        sp.set_color(RED)
        sp.set_linewidth(2.0)
    axv.tick_params(color=RED)
else:
    axv.axis('off')

# moving markers
mark = axs.axvline(TT[0], color='k', lw=1.5, zorder=8)
dot = axs.plot([TT[0]], [SSH[0]], 'o', ms=8, color='k', zorder=9)[0]

if args.transparent:
    fig.patch.set_alpha(0.0)
    for ax in [axm, axs]:
        ax.patch.set_alpha(0.0)


def update(fi):
    cs.set_array(FLD[fi].ravel())
    ttl.set_text('depth-averaged current -- %s (PST)'
                 % TT[fi].strftime('%Y-%m-%d %H:%M'))
    Q.set_UVC(UQ[fi][::qs, ::qs], VQ[fi][::qs, ::qs])
    mark.set_xdata([TT[fi], TT[fi]])
    dot.set_data([TT[fi]], [SSH[fi]])
    if csv_ is not None:
        for ip, m in enumerate(csv_):
            m.set_array(VEL['fld'][fi][:, ip:ip + 1].ravel())
    return []


stem = ('20260810_%s_vel_%s_%s_%s'
        % (args.region, args.sect_var, args.ds0, args.ds1))
still_kw = dict(dpi=200, bbox_inches='tight', transparent=args.transparent)

# Video formats. h264 has no alpha channel, so mp4 is rendered on white -- a
# transparent figure piped to it just gets composited onto black. ProRes 4444
# and QuickTime RLE do carry alpha and are both NATIVE ffmpeg encoders, so they
# need nothing beyond a stock build.
#
# Do NOT reach for VP9/webm here: `-c:v libvpx-vp9 -pix_fmt yuva420p` encodes
# without error and the alpha is silently gone -- a decode round-trip comes
# back fully opaque. Verified 2026.08.10, ffmpeg 7.1.1.
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
