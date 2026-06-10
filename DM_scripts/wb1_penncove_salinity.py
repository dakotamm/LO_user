"""
Shared engine: zoomed Penn Cove salinity frames + movie (+ Penn Cove SSH
tidal-phase panel) for the wb1_t0_xn11abbur00 run.

Surface by default; pass --bottom for bottom salinity. Standalone (does not use
pan_plot) so we can set a custom map extent and a fixed color scale. Usually run
via the wrappers wb1_penncove_salinity_surface.sh / _bottom.sh, but callable
directly, e.g.:
    python wb1_penncove_salinity.py --smin 15 --smax 25
    python wb1_penncove_salinity.py --bottom --smin 15 --smax 25
    python wb1_penncove_salinity.py --lon0 -122.78 --lon1 -122.40 --lat0 48.15 --lat1 48.40
"""
import argparse
import subprocess
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.path import Path as MplPath
from cmocean import cm

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun
# Shared zoom box, masking polygons, and Penn Cove SSH box (edit there to keep
# all Penn Cove plots in sync). --keep-west disables the polygon masking here.
from wb1_penncove_region import EXCLUDE_POLYS, INCLUDE_POLYS, PENN_COVE_BOX, ZOOM

# ---- arguments / settings -------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument('--gtx',  default='wb1_t0_xn11abbur00')
p.add_argument('--ro',   default=2, type=int)          # /dat2/dakotamm/LO_roms
p.add_argument('--ds0',  default='2025.12.03')
p.add_argument('--ds1',  default='2025.12.06')
p.add_argument('--lt',   default='hourly0')            # clean hour-0 start on ds0
# zoom box (defaults from the shared region module)
p.add_argument('--lon0', default=ZOOM['lon0'], type=float)
p.add_argument('--lon1', default=ZOOM['lon1'], type=float)
p.add_argument('--lat0', default=ZOOM['lat0'], type=float)
p.add_argument('--lat1', default=ZOOM['lat1'], type=float)
# color limits: leave as None to auto-pick from the data in the box
p.add_argument('--smin', default=None, type=float)
p.add_argument('--smax', default=None, type=float)
p.add_argument('--no-movie', dest='movie', action='store_false')
p.add_argument('--keep-west', dest='exclude', action='store_false',
               help='do not mask the EXCLUDE_POLY region')
p.add_argument('--debug-polys', dest='debug_polys', action='store_true',
               help='overlay exclude (red) / include (blue) polygons + fine grid')
p.add_argument('--bottom', dest='bottom', action='store_true',
               help='plot bottom salinity (s-level 0) instead of surface')
args = p.parse_args()

SLEV = 0 if args.bottom else -1          # ROMS: 0 = bottom, -1 = surface
LABEL = 'Bottom' if args.bottom else 'Surface'
OUTTAG = 'penncove_salt_bottom' if args.bottom else 'penncove_salt_surface'

gridname, tag, ex_name = args.gtx.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
Ldir['roms_out'] = Ldir['roms_out' + str(args.ro)]
aa = [args.lon0, args.lon1, args.lat0, args.lat1]
excl_paths = [MplPath(p) for p in EXCLUDE_POLYS] if args.exclude else []
incl_paths = [MplPath(p) for p in INCLUDE_POLYS] if args.exclude else []

fn_list = Lfun.get_fn_list(args.lt, Ldir, args.ds0, args.ds1)
fn_list = [fn for fn in fn_list if fn.is_file()]
if len(fn_list) == 0:
    raise SystemExit('No history files found for %s %s-%s' % (args.gtx, args.ds0, args.ds1))
print('%d frames to plot' % len(fn_list))

# ---- helper: surface salt within the zoom box -----------------------------
def get_surf_salt(fn):
    ds = xr.open_dataset(fn)
    lon = ds.lon_rho.values
    lat = ds.lat_rho.values
    salt = ds.salt[0, SLEV, :, :].values          # ocean_time=0, SLEV s-level
    mask = ds.mask_rho.values
    salt = np.where(mask == 0, np.nan, salt)
    ds.close()
    if excl_paths:
        pts = np.column_stack([lon.ravel(), lat.ravel()])
        remove = np.zeros(lon.size, dtype=bool)
        for ep in excl_paths:
            remove |= ep.contains_points(pts)
        for ip in incl_paths:
            remove &= ~ip.contains_points(pts)
        salt = np.where(remove.reshape(lon.shape), np.nan, salt)
    inbox = (lon >= aa[0]) & (lon <= aa[1]) & (lat >= aa[2]) & (lat <= aa[3])
    return lon, lat, salt, inbox

# ---- fixed color limits: from data in the box unless user forced them ------
if args.smin is not None and args.smax is not None:
    smin, smax = args.smin, args.smax
else:
    # sample a handful of frames across the run for a robust, fixed range
    samp = fn_list[:: max(1, len(fn_list) // 8)]
    vals = []
    for fn in samp:
        _, _, salt, inbox = get_surf_salt(fn)
        vals.append(salt[inbox])
    vals = np.concatenate(vals)
    vals = vals[np.isfinite(vals)]
    smin = np.floor(np.nanpercentile(vals, 1))
    smax = np.ceil(np.nanpercentile(vals, 99))
print('salinity color range: %.1f to %.1f' % (smin, smax))

# ---- Penn Cove SSH timeseries (tidal phase) --------------------------------
pcb = PENN_COVE_BOX
ssh_t, ssh_v = [], []
for fn in fn_list:
    ds = xr.open_dataset(fn)
    lon = ds.lon_rho.values
    lat = ds.lat_rho.values
    zeta = ds.zeta[0, :, :].values
    mask = ds.mask_rho.values
    inbox = ((lon >= pcb[0]) & (lon <= pcb[1]) &
             (lat >= pcb[2]) & (lat <= pcb[3]) & (mask == 1))
    ssh_v.append(np.nanmean(np.where(inbox, zeta, np.nan)))
    t_utc = pd.Timestamp(ds.ocean_time.values[0]).to_pydatetime()
    ssh_t.append(pfun.get_dt_local(t_utc).replace(tzinfo=None))  # naive local (PST)
    ds.close()
ssh_v = np.array(ssh_v)
print('Penn Cove SSH range: %.2f to %.2f m' % (np.nanmin(ssh_v), np.nanmax(ssh_v)))

# ---- output dir ------------------------------------------------------------
outdir = Ldir['LOo'] / 'plots' / ('%s_%s' % (OUTTAG, args.gtx))
Lfun.make_dir(outdir, clean=True)

# ---- plot every frame ------------------------------------------------------
for ii, fn in enumerate(fn_list):
    lon, lat, salt, _ = get_surf_salt(fn)
    plon, plat = pfun.get_plon_plat(lon, lat)
    pfun.start_plot(fs=14, figsize=(9, 11))
    fig = plt.figure()
    gs = fig.add_gridspec(5, 1, hspace=0.35)
    ax = fig.add_subplot(gs[0:4, 0])    # map
    axt = fig.add_subplot(gs[4, 0])     # SSH timeseries
    cs = ax.pcolormesh(plon, plat, salt, cmap=cm.haline, vmin=smin, vmax=smax)
    fig.colorbar(cs, ax=ax)
    pfun.add_coast(ax)
    ax.axis(aa)
    pfun.dar(ax)
    pfun.add_info(ax, fn)
    if args.debug_polys:
        for poly in EXCLUDE_POLYS:
            xs = [v[0] for v in poly] + [poly[0][0]]
            ys = [v[1] for v in poly] + [poly[0][1]]
            ax.plot(xs, ys, '-r', lw=1.2)
        for poly in INCLUDE_POLYS:
            xs = [v[0] for v in poly] + [poly[0][0]]
            ys = [v[1] for v in poly] + [poly[0][1]]
            ax.plot(xs, ys, '-b', lw=1.2)
        ax.grid(True, lw=0.3, alpha=0.5)
    ax.set_title('%s Salinity $(g\\ kg^{-1})$' % LABEL)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    # SSH (tidal phase) panel
    axt.plot(ssh_t, ssh_v, '-', color='tab:blue', lw=1)
    axt.plot(ssh_t[ii], ssh_v[ii], 'o', color='red', ms=8, zorder=5)
    axt.axhline(0, color='gray', lw=0.5)
    axt.set_xlim(ssh_t[0], ssh_t[-1])
    axt.set_ylabel('Penn Cove\nSSH (m)')
    axt.set_title('Tidal phase', fontsize=11)
    axt.grid(True, lw=0.3, alpha=0.5)
    axt.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    for lab in axt.get_xticklabels():
        lab.set_rotation(30)
        lab.set_horizontalalignment('right')
    fig.tight_layout()
    fig.savefig(outdir / ('plot_%04d.png' % (ii + 1)), dpi=100)
    plt.close(fig)
    pfun.end_plot()

print('Saved %d frames to %s' % (len(fn_list), outdir))

# ---- movie -----------------------------------------------------------------
if args.movie:
    cmd = ['ffmpeg', '-y', '-r', '8', '-i', str(outdir / 'plot_%04d.png'),
           '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '25',
           str(outdir / 'movie.mp4')]
    subprocess.run(cmd)
    print('Movie: %s' % (outdir / 'movie.mp4'))
