"""
Zoomed surface-salinity frames + movie for the wb1_t0_xn11abbur00 run.

Standalone (does not use pan_plot) so we can set a custom map extent and a
fixed, data-appropriate salinity color scale. Defaults target the Penn Cove /
Saratoga Passage region; override on the command line as needed.

Run on apogee inside loenv, e.g.:
    python plot_wb1_salt_zoom.py
    python plot_wb1_salt_zoom.py --lon0 -122.78 --lon1 -122.45 --lat0 48.15 --lat1 48.40
    python plot_wb1_salt_zoom.py --smin 20 --smax 30   # force the color range
"""
import argparse
import subprocess
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from cmocean import cm

# Cells inside any EXCLUDE polygon are removed before plotting AND before the
# color scale is computed -- UNLESS they also fall inside an INCLUDE polygon,
# which wins (used to add specific cells back). Vertices are (lon, lat).
# Disable all of this with --keep-west.
EXCLUDE_POLYS = [
    # western channel west of Whidbey
    [(-122.79, 48.41), (-122.61, 48.41), (-122.68, 48.33),
     (-122.72, 48.27), (-122.71, 48.20), (-122.62, 48.155), (-122.79, 48.145)],
    # bottom yellow strip (red circle)
    [(-122.71, 48.165), (-122.59, 48.165), (-122.59, 48.143), (-122.71, 48.143)],
    # western tip of Penn Cove (blue circle) -- remove explicitly
    [(-122.775, 48.188), (-122.708, 48.188), (-122.708, 48.232), (-122.775, 48.232)],
]
INCLUDE_POLYS = [
    # (none) -- western tip of Penn Cove removed again
]

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

# ---- arguments / settings -------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument('--gtx',  default='wb1_t0_xn11abbur00')
p.add_argument('--ro',   default=2, type=int)          # /dat2/dakotamm/LO_roms
p.add_argument('--ds0',  default='2025.12.03')
p.add_argument('--ds1',  default='2025.12.06')
p.add_argument('--lt',   default='hourly0')            # clean hour-0 start on ds0
# zoom box (Penn Cove / Saratoga Passage); edit to taste
p.add_argument('--lon0', default=-122.78, type=float)
p.add_argument('--lon1', default=-122.45, type=float)
p.add_argument('--lat0', default=48.15,  type=float)
p.add_argument('--lat1', default=48.40,  type=float)
# color limits: leave as None to auto-pick from the data in the box
p.add_argument('--smin', default=None, type=float)
p.add_argument('--smax', default=None, type=float)
p.add_argument('--no-movie', dest='movie', action='store_false')
p.add_argument('--keep-west', dest='exclude', action='store_false',
               help='do not mask the EXCLUDE_POLY region')
args = p.parse_args()

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
    salt = ds.salt[0, -1, :, :].values            # ocean_time=0, top s-level
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

# ---- output dir ------------------------------------------------------------
outdir = Ldir['LOo'] / 'plots' / ('saltzoom_%s' % args.gtx)
Lfun.make_dir(outdir, clean=True)

# ---- plot every frame ------------------------------------------------------
for ii, fn in enumerate(fn_list):
    lon, lat, salt, _ = get_surf_salt(fn)
    plon, plat = pfun.get_plon_plat(lon, lat)
    pfun.start_plot(fs=14, figsize=(9, 10))
    fig, ax = plt.subplots(1, 1)
    cs = ax.pcolormesh(plon, plat, salt, cmap=cm.haline, vmin=smin, vmax=smax)
    fig.colorbar(cs, ax=ax)
    pfun.add_coast(ax)
    ax.axis(aa)
    pfun.dar(ax)
    pfun.add_info(ax, fn)
    ax.set_title('Surface Salinity $(g\\ kg^{-1})$')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
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
