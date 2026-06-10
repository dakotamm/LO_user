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
import pickle
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm
from cmocean import cm

from lo_tools import Lfun, zrfun
from lo_tools import plotting_functions as pfun
from wb1_penncove_region import (ZOOM, DO_MMOL_TO_MGL, HYPOXIC_MGL, LOWDO_MGL,
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
p.add_argument('--salt-min', default=15.0, type=float)
p.add_argument('--salt-max', default=25.0, type=float)
p.add_argument('--hyp-min',  type=float); p.add_argument('--hyp-max',  type=float)
p.add_argument('--low-min',  type=float); p.add_argument('--low-max',  type=float)
# obs station overlay (combined obsmod pickles, filtered to these sources)
p.add_argument('--obs-sources', default='ecology_nc,kc,kc_whidbeyBasin')
p.add_argument('--obs-otype', default='all')   # bottle, ctd, or all
p.add_argument('--no-obs', dest='obs', action='store_false')
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
    hyp = np.sum(dz * (oxy < HYPOXIC_MGL), axis=0)              # < 2 mg/L thickness, m
    low = np.sum(dz * (oxy < LOWDO_MGL), axis=0)                # < 5 mg/L thickness, m
    ds.close()
    salt_s = mask_field(salt_s, lon, lat, mask)
    do_b = mask_field(do_b, lon, lat, mask)
    hyp = mask_field(hyp, lon, lat, mask)
    low = mask_field(low, lon, lat, mask)
    inbox = (lon >= aa[0]) & (lon <= aa[1]) & (lat >= aa[2]) & (lat <= aa[3])
    return lon, lat, salt_s, do_b, hyp, low, inbox


# ---- obs stations with/without hypoxia in the movie window -----------------
def load_obs_stations(ds0, ds1, sources, otype='all'):
    """Stations from the combined obsmod pickles (given sources), within the
    ds0-ds1 window and the zoom box. A cast is hypoxic if any depth < 2 mg/L.
    Returns a DataFrame [station, lon, lat, hyp(bool), n_casts] or None."""
    hyp_uM = HYPOXIC_MGL / DO_MMOL_TO_MGL          # 2 mg/L = 62.5 uM
    t0 = datetime.strptime(ds0, '%Y.%m.%d')
    t1 = datetime.strptime(ds1, '%Y.%m.%d') + timedelta(days=1)   # inclusive end day
    year = t0.year
    otypes = ['bottle', 'ctd'] if otype == 'all' else [otype]
    recs = []
    for ot in otypes:
        fn = Ldir['LOo'] / 'obsmod' / ('combined_%s_%d_%s.p' % (ot, year, args.gtx))
        if not fn.is_file():
            print('  obs: missing %s' % fn.name); continue
        obs = pickle.load(open(fn, 'rb'))['obs']
        if len(obs) == 0 or 'DO (uM)' not in obs.columns:
            continue
        sub = obs[obs['source'].isin(sources)].dropna(subset=['DO (uM)']).copy()
        tt = pd.to_datetime(sub['time'], errors='coerce')
        if tt.dt.tz is not None:                 # obs times may be tz-aware
            tt = tt.dt.tz_localize(None)
        sub = sub[(tt >= t0) & (tt < t1)]
        if len(sub):
            recs.append(sub[['cid', 'lon', 'lat', 'name', 'DO (uM)']])
    if not recs:
        return None
    allobs = pd.concat(recs, ignore_index=True)
    cast = allobs.groupby('cid').agg(
        hyp=('DO (uM)', lambda x: bool(np.any(x < hyp_uM))),
        lon=('lon', 'mean'), lat=('lat', 'mean'), name=('name', 'first')).reset_index()
    cast['station'] = cast['name'].astype(str)
    stn = cast.groupby('station').agg(
        hyp=('hyp', 'max'), n_casts=('hyp', 'count'),
        lon=('lon', 'mean'), lat=('lat', 'mean')).reset_index()
    # keep stations inside the zoom box
    stn = stn[(stn.lon >= aa[0]) & (stn.lon <= aa[1]) &
              (stn.lat >= aa[2]) & (stn.lat <= aa[3])].reset_index(drop=True)
    return stn if len(stn) else None


def overlay_stations(ax, stn):
    if stn is None or len(stn) == 0:
        return
    no = stn[~stn['hyp']]
    yes = stn[stn['hyp']]
    if len(no):
        ax.scatter(no.lon, no.lat, marker='o', s=32, facecolors='white',
                   edgecolors='k', linewidths=1.0, zorder=6,
                   label='station, no hypoxic cast')
    if len(yes):
        ax.scatter(yes.lon, yes.lat, marker='o', s=48, facecolors='red',
                   edgecolors='k', linewidths=0.9, zorder=7,
                   label='station, hypoxic cast')


obs_stn = None
if args.obs:
    srcs = [s.strip() for s in args.obs_sources.split(',') if s.strip()]
    obs_stn = load_obs_stations(args.ds0, args.ds1, srcs, args.obs_otype)
    if obs_stn is None:
        print('obs: no casts from %s within %s..%s in the box'
              % (srcs, args.ds0, args.ds1))
    else:
        print('obs: %d stations in box, %d with a hypoxic cast (%s..%s)'
              % (len(obs_stn), int(obs_stn['hyp'].sum()), args.ds0, args.ds1))

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
sH, sL = [], []
for fn in samp:
    _, _, _, _, hh, ll, inbox = get_fields(fn)
    sH.append(hh[inbox]); sL.append(ll[inbox])

def forced(lo, hi):
    return (lo, hi) if (lo is not None and hi is not None) else None

# salinity locked to 15-25 (override with --salt-min/--salt-max); the two layer-
# depth panels auto-scale; bottom DO uses a fixed threshold-demarcating colorbar.
salt_lims = (args.salt_min, args.salt_max)
hyp_lims  = forced(args.hyp_min, args.hyp_max) or auto_lims(sH, floor0=True)
low_lims  = forced(args.low_min, args.low_max) or auto_lims(sL, floor0=True)
DO_BOUNDS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12]   # 2 and 5 mg/L are band edges
print('limits -> salt %s  hyp<2 %s  low<5 %s  (DO bounds %s)'
      % (salt_lims, hyp_lims, low_lims, DO_BOUNDS))

ssh_t, ssh_v = get_ssh_series(fn_list)
print('Penn Cove SSH range: %.2f to %.2f m' % (np.nanmin(ssh_v), np.nanmax(ssh_v)))

# ---- output dir (date range in the name) -----------------------------------
outdir = Ldir['LOo'] / 'plots' / ('penncove_multivar_%s_%s_%s'
                                  % (args.ds0, args.ds1, args.gtx))
Lfun.make_dir(outdir, clean=True)

do_norm = BoundaryNorm(DO_BOUNDS, cm.oxy.N)   # discrete bands; 2 & 5 are edges
panels = [
    dict(title='Surface Salinity $(g\\ kg^{-1})$', key='salt', cmap=cm.haline,
         vmin=salt_lims[0], vmax=salt_lims[1]),
    dict(title='Bottom DO $(mg\\ L^{-1})$', key='do', cmap=cm.oxy,
         norm=do_norm, ticks=DO_BOUNDS, contours=[HYPOXIC_MGL, LOWDO_MGL]),
    dict(title='DO < 2 mg/L layer depth (m)', key='hyp', cmap=cm.deep,
         vmin=hyp_lims[0], vmax=hyp_lims[1]),
    dict(title='DO < 5 mg/L layer depth (m)', key='low', cmap=cm.matter,
         vmin=low_lims[0], vmax=low_lims[1]),
]

# ---- plot every frame ------------------------------------------------------
for ii, fn in enumerate(fn_list):
    lon, lat, ss, dd, hh, ll, _ = get_fields(fn)
    plon, plat = pfun.get_plon_plat(lon, lat)
    fields = {'salt': ss, 'do': dd, 'hyp': hh, 'low': ll}
    pfun.start_plot(fs=12, figsize=(22, 9))
    fig = plt.figure()
    gs = fig.add_gridspec(5, 4, hspace=0.4, wspace=0.28)
    for jj, P in enumerate(panels):
        ax = fig.add_subplot(gs[0:4, jj])
        fld = fields[P['key']]
        if 'norm' in P:
            cs = ax.pcolormesh(plon, plat, fld, cmap=P['cmap'], norm=P['norm'])
        else:
            cs = ax.pcolormesh(plon, plat, fld, cmap=P['cmap'],
                               vmin=P['vmin'], vmax=P['vmax'])
        fig.colorbar(cs, ax=ax, ticks=P.get('ticks'))
        if P.get('contours'):
            cc = ax.contour(lon, lat, fld, levels=P['contours'],
                            colors=['red', 'gold'], linewidths=1.3, zorder=4)
            ax.clabel(cc, fmt='%g', fontsize=7)
        pfun.add_coast(ax)
        overlay_stations(ax, obs_stn)
        ax.axis(aa)
        pfun.dar(ax)
        ax.set_title(P['title'])
        ax.set_xlabel('Longitude')
        if jj == 0:
            ax.set_ylabel('Latitude')
            pfun.add_info(ax, fn)
            if obs_stn is not None and len(obs_stn):
                ax.legend(loc='lower left', fontsize=7, framealpha=0.7)
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
