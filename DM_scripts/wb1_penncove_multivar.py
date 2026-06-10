"""
Three-panel zoomed Penn Cove movie for the wb1_t0_xn11abbur00 run:
    1. Surface salinity (g/kg)
    2. Bottom DO (mg/L)
    3. Hypoxic layer depth (m)  = thickness of water with DO < 2 mg/L
with a single Penn Cove SSH tidal-phase strip spanning the bottom of all three.

Same zoom box / exclude-include polygons / SSH box as wb1_penncove_salinity.py
(shared via wb1_penncove_region.py). Usually run via wb1_penncove_multivar.sh,
but callable directly, e.g.:
    python wb1_penncove_multivar.py --ds0 2025.09.01 --ds1 2025.09.03
    python wb1_penncove_multivar.py --do-min 0 --do-max 10   # force a DO range
"""
import argparse
import subprocess
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from cmocean import cm

from lo_tools import Lfun, zrfun
from lo_tools import plotting_functions as pfun
from wb1_penncove_region import (ZOOM, DO_MMOL_TO_MGL, HYPOXIC_MGL,
                                 mask_field, get_ssh_series)

# ---- arguments ------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument('--gtx', default='wb1_t0_xn11abbur00')
p.add_argument('--ro',  default=2, type=int)           # /dat2/dakotamm/LO_roms
p.add_argument('--ds0', default='2025.09.01')
p.add_argument('--ds1', default='2025.09.03')
p.add_argument('--lt',  default='hourly0')             # clean hour-0 start on ds0
p.add_argument('--lon0', default=ZOOM['lon0'], type=float)
p.add_argument('--lon1', default=ZOOM['lon1'], type=float)
p.add_argument('--lat0', default=ZOOM['lat0'], type=float)
p.add_argument('--lat1', default=ZOOM['lat1'], type=float)
# optional fixed color limits per field (default: auto from data in the box)
p.add_argument('--salt-min', type=float); p.add_argument('--salt-max', type=float)
p.add_argument('--do-min',   type=float); p.add_argument('--do-max',   type=float)
p.add_argument('--hyp-min',  type=float); p.add_argument('--hyp-max',  type=float)
p.add_argument('--no-movie', dest='movie', action='store_false')
args = p.parse_args()

gridname, tag, ex_name = args.gtx.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
Ldir['roms_out'] = Ldir['roms_out' + str(args.ro)]
aa = [args.lon0, args.lon1, args.lat0, args.lat1]

fn_list = Lfun.get_fn_list(args.lt, Ldir, args.ds0, args.ds1)
fn_list = [fn for fn in fn_list if fn.is_file()]
if len(fn_list) == 0:
    raise SystemExit('No history files for %s %s-%s' % (args.gtx, args.ds0, args.ds1))
print('%d frames to plot' % len(fn_list))

# vertical structure S is constant across the run; grab once
S = zrfun.get_basic_info(fn_list[0], only_S=True)


def get_fields(fn):
    ds = xr.open_dataset(fn)
    lon = ds.lon_rho.values
    lat = ds.lat_rho.values
    mask = ds.mask_rho.values
    if 'oxygen' not in ds.data_vars:
        ds.close()
        raise SystemExit('no oxygen variable in %s -- need a bgc run' % fn)
    salt_s = ds.salt[0, -1, :, :].values                       # surface
    do_b = ds.oxygen[0, 0, :, :].values * DO_MMOL_TO_MGL        # bottom, mg/L
    # hypoxic layer thickness: sum of layer dz where DO < threshold
    h = ds.h.values
    zeta = ds.zeta[0, :, :].values
    z_rho, z_w = zrfun.get_z(h, zeta, S)
    dz = np.diff(z_w, axis=0)                                   # (N, eta, xi), m
    oxy = ds.oxygen[0, :, :, :].values * DO_MMOL_TO_MGL         # (N, eta, xi)
    hyp = np.sum(dz * (oxy < HYPOXIC_MGL), axis=0)              # meters
    ds.close()
    salt_s = mask_field(salt_s, lon, lat, mask)
    do_b = mask_field(do_b, lon, lat, mask)
    hyp = mask_field(hyp, lon, lat, mask)
    inbox = (lon >= aa[0]) & (lon <= aa[1]) & (lat >= aa[2]) & (lat <= aa[3])
    return lon, lat, salt_s, do_b, hyp, inbox


# ---- color limits: auto from data in the box unless forced -----------------
def auto_lims(arrs, lo=1, hi=99, floor0=False):
    v = np.concatenate(arrs)
    v = v[np.isfinite(v)]
    vmin = 0.0 if floor0 else float(np.floor(np.nanpercentile(v, lo)))
    vmax = float(np.ceil(np.nanpercentile(v, hi)))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return (vmin, vmax)


samp = fn_list[:: max(1, len(fn_list) // 8)]
sS, sD, sH = [], [], []
for fn in samp:
    _, _, ss, dd, hh, inbox = get_fields(fn)
    sS.append(ss[inbox]); sD.append(dd[inbox]); sH.append(hh[inbox])

def forced(lo, hi):
    return (lo, hi) if (lo is not None and hi is not None) else None

salt_lims = forced(args.salt_min, args.salt_max) or auto_lims(sS)
do_lims   = forced(args.do_min,   args.do_max)   or auto_lims(sD)
hyp_lims  = forced(args.hyp_min,  args.hyp_max)  or auto_lims(sH, floor0=True)
print('color limits -> salt %s  DO %s  hyp %s' % (salt_lims, do_lims, hyp_lims))

ssh_t, ssh_v = get_ssh_series(fn_list)
print('Penn Cove SSH range: %.2f to %.2f m' % (np.nanmin(ssh_v), np.nanmax(ssh_v)))

# ---- output dir (date range in the name) -----------------------------------
outdir = Ldir['LOo'] / 'plots' / ('penncove_multivar_%s_%s_%s'
                                  % (args.ds0, args.ds1, args.gtx))
Lfun.make_dir(outdir, clean=True)

panels = [
    ('Surface Salinity $(g\\ kg^{-1})$', cm.haline, salt_lims),
    ('Bottom DO $(mg\\ L^{-1})$',        cm.oxy,    do_lims),
    ('Hypoxic layer depth (m)',          cm.deep,   hyp_lims),
]

# ---- plot every frame ------------------------------------------------------
for ii, fn in enumerate(fn_list):
    lon, lat, ss, dd, hh, _ = get_fields(fn)
    plon, plat = pfun.get_plon_plat(lon, lat)
    flds = [ss, dd, hh]
    pfun.start_plot(fs=13, figsize=(16, 9))
    fig = plt.figure()
    gs = fig.add_gridspec(5, 3, hspace=0.4, wspace=0.25)
    for jj, (title, cmap, lims) in enumerate(panels):
        ax = fig.add_subplot(gs[0:4, jj])
        cs = ax.pcolormesh(plon, plat, flds[jj], cmap=cmap,
                           vmin=lims[0], vmax=lims[1])
        fig.colorbar(cs, ax=ax)
        pfun.add_coast(ax)
        ax.axis(aa)
        pfun.dar(ax)
        ax.set_title(title)
        ax.set_xlabel('Longitude')
        if jj == 0:
            ax.set_ylabel('Latitude')
            pfun.add_info(ax, fn)
        else:
            ax.set_yticklabels([])
    # SSH (tidal phase) strip spanning all three panels
    axt = fig.add_subplot(gs[4, :])
    axt.plot(ssh_t, ssh_v, '-', color='tab:blue', lw=1)
    axt.plot(ssh_t[ii], ssh_v[ii], 'o', color='red', ms=8, zorder=5)
    axt.axhline(0, color='gray', lw=0.5)
    axt.set_xlim(ssh_t[0], ssh_t[-1])
    axt.set_ylabel('Penn Cove SSH (m)')
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
