"""
make_hypoxia_animations.py

Build per-layer animations (one mp4 per layer) for the merged hypoxia
events (event 12 = event_01 + event_02 unioned and deduped, plus
event 3) at the Penn Cove zoom used in the tide_phase workflow.

Each frame (one per day per layer) shows:
  Top-left:  velocity speed at the chosen layer (rho-grid pcolormesh +
             sparse quiver).
  Top-right: Okubo-Weiss field (diverging cmap; OW<0 rotation-dominated,
             OW>0 strain-dominated) with detected vortex features
             (read from the existing per-event ow_vortices_*.csv) drawn
             as circles colored by rotation sense.
  Bottom-left:  Penn Cove SSH (hourly + daily lowpass) over the full
                year with spring/neap shading and a marker at the
                current frame's date.
  Bottom-right: M1 bottom DO (full year) with a marker at the current
                frame's date.

The Penn Cove zoom matches the tide_phase workflow:
  ZOOM_BOUNDS  = (-122.755, -122.60, 48.205, 48.255)
  PLOT_EXCLUDE = (-122.755, -122.73, 48.205, 48.215)  (NaN-masked corner)

For each layer, frames from BOTH events (date-sorted, deduped on date)
are concatenated into a single mp4 via ffmpeg.

Designed to run on apogee where the raw lowpassed.nc files live:
    <ROMS_OUT>/<gtagex>/f<YYYY.MM.DD>/lowpassed.nc

Usage (on apogee):
    python make_hypoxia_animations.py -gtx wb1_r0_xn11b -mooring M1 \\
        -year 2017 -layers all -fps 4

Outputs:
    <LOo>/swirl/<gtagex>/hypoxia_events/_animations/<layer>/frame_*.png
    <LOo>/swirl/<gtagex>/hypoxia_events/_animations/anim_<layer>.mp4
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter

# Reuse the OW / velocity machinery from the existing detection script
# so the OW field and rho-grid interpolation match exactly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_swirl_roms import (  # noqa: E402
    get_velocity_2d,
    get_grid_spacing,
    compute_okubo_weiss,
)

try:
    from lo_tools import Lfun
    from lo_tools import plotting_functions as pfun
    _HAS_LO_TOOLS = True
except ImportError:
    _HAS_LO_TOOLS = False


# ---------------------------------------------------------------------------
# Penn Cove geometry (matches tide_phase/tide_phase_analysis.py)
# ---------------------------------------------------------------------------
ZOOM_BOUNDS = (-122.755, -122.60, 48.205, 48.255)
PLOT_EXCLUDE = [(-122.755, -122.73, 48.205, 48.215)]

# DO unit conversion (mmol/m^3 -> mg/L)
DO_UM_TO_MGL = 32.0 / 1000.0

# Layers to animate: (vel_type, s_level, layer_name)
LAYERS = [
    ('surface',     -1, 'surface'),
    ('depth_avg',   -1, 'depth_avg'),
    ('depth_level',  0, 'bottom'),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-gtx', '--gtagex', type=str, default='wb1_r0_xn11b')
    p.add_argument('-ro', '--roms_out_num', type=int, default=2)
    p.add_argument('-mooring', type=str, default='M1')
    p.add_argument('-job', type=str, default='pc0')
    p.add_argument('-year', type=int, default=2017)
    p.add_argument('-layers', type=str, default='all',
        help="Comma list of surface,depth_avg,bottom or 'all'.")
    p.add_argument('-events_csv', type=str, default=None)
    p.add_argument('-phase_file', type=str, default=None,
        help='UTide tide-phase NC (for spring/neap shading on SSH panel).')
    p.add_argument('-mooring_file', type=str, default=None)
    p.add_argument('-out_dir', type=str, default=None,
        help='Override output base dir.')
    p.add_argument('-fps', type=int, default=4)
    p.add_argument('-dpi', type=int, default=130)
    p.add_argument('-smooth_sigma', type=float, default=2.0,
        help='Gaussian sigma (grid cells) applied to OW before plotting.')
    p.add_argument('-quiver_stride', type=int, default=3)
    p.add_argument('-dry', action='store_true',
        help='Print plan and exit; do not render frames.')
    p.add_argument('-skip_existing', action='store_true',
        help='Skip frames whose PNG already exists.')
    return p.parse_args()


def select_layers(spec):
    if spec.strip().lower() == 'all':
        return LAYERS
    wanted = {s.strip() for s in spec.split(',')}
    chosen = [L for L in LAYERS if L[2] in wanted]
    if not chosen:
        raise ValueError(f'No valid layers in {spec!r}')
    return chosen


def fmt_date(ts):
    return pd.Timestamp(ts).strftime('%Y.%m.%d')


def apply_plot_mask(arr, lon, lat):
    """NaN-mask cells outside ZOOM_BOUNDS or inside any PLOT_EXCLUDE box."""
    out = arr.astype(float, copy=True)
    lo0, lo1, la0, la1 = ZOOM_BOUNDS
    outside = (lon < lo0) | (lon > lo1) | (lat < la0) | (lat > la1)
    out[outside] = np.nan
    for xo0, xo1, yo0, yo1 in PLOT_EXCLUDE:
        m = (lon >= xo0) & (lon <= xo1) & (lat >= yo0) & (lat <= yo1)
        out[m] = np.nan
    return out


def compute_layer_fields(lowpass_nc, dsg, vel_type, s_level, smooth_sigma):
    """Return (speed_masked, OW_masked, vx, vy, lon_rho, lat_rho, vel_title)."""
    with xr.open_dataset(lowpass_nc) as ds:
        vx, vy, vel_title = get_velocity_2d(ds, dsg, vel_type, s_level)
    dx_m, dy_m = get_grid_spacing(dsg)
    ny, nx = vx.shape
    lon_rho = dsg.lon_rho.values[:ny, :nx]
    lat_rho = dsg.lat_rho.values[:ny, :nx]
    mask_rho = dsg.mask_rho.values[:ny, :nx]
    OW, _ = compute_okubo_weiss(vx, vy, dx_m, dy_m)
    if smooth_sigma > 0:
        OW = gaussian_filter(OW, sigma=smooth_sigma)
    OW = OW.astype(float)
    OW[mask_rho == 0] = np.nan
    speed = np.sqrt(vx**2 + vy**2)
    speed[mask_rho == 0] = np.nan
    speed = apply_plot_mask(speed, lon_rho, lat_rho)
    OW = apply_plot_mask(OW, lon_rho, lat_rho)
    return speed, OW, vx, vy, lon_rho, lat_rho, vel_title


def find_lowpassed(date_str, Ldir, gtagex):
    for key in ['roms_out', 'roms_out2', 'roms_out1']:
        if key not in Ldir:
            continue
        cand = Ldir[key] / gtagex / f'f{date_str}' / 'lowpassed.nc'
        if cand.exists():
            return cand
    return None


def load_mooring_DO(mooring_file):
    """Return (times, bot_DO_mgL) for the full year."""
    with xr.open_dataset(mooring_file) as ds:
        times = pd.to_datetime(ds['ocean_time'].values)
        # bottom = s_rho index 0
        do_bot = ds['oxygen'].isel(s_rho=0).values * DO_UM_TO_MGL
    return times, do_bot


def load_phase(phase_file):
    """Return DataFrame indexed by time with zeta/zeta_pred/is_spring/is_neap."""
    with xr.open_dataset(phase_file) as ds:
        df = pd.DataFrame({
            'time': pd.to_datetime(ds['time'].values),
            'zeta': ds['zeta'].values,
            'zeta_pred': ds['zeta_pred'].values
                if 'zeta_pred' in ds else np.full(ds['zeta'].size, np.nan),
            'is_spring': ds['is_spring'].values.astype(bool)
                if 'is_spring' in ds else np.zeros(ds['zeta'].size, bool),
            'is_neap': ds['is_neap'].values.astype(bool)
                if 'is_neap' in ds else np.zeros(ds['zeta'].size, bool),
        })
    return df.set_index('time')


def load_event_vortices(event_dirs, layer):
    """Concat all ow_vortices_*.csv from the per-event/layer dirs."""
    frames = []
    for ev_dir in event_dirs:
        layer_dir = Path(ev_dir) / layer
        if not layer_dir.exists():
            continue
        for f in sorted(layer_dir.glob('ow_vortices_*.csv')):
            try:
                frames.append(pd.read_csv(f))
            except Exception as e:
                print(f'  WARN: could not read {f}: {e}')
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if 'date' in out.columns:
        out['date'] = pd.to_datetime(out['date'], format='%Y.%m.%d',
                                      errors='coerce')
    return out.drop_duplicates()


def shade_spring_neap(ax, phase_df, t0, t1, alpha=0.18):
    """Shade spring (crimson) and neap (royalblue) bands on a time axis."""
    if phase_df is None:
        return
    sub = phase_df.loc[(phase_df.index >= t0) & (phase_df.index <= t1)]
    if sub.empty:
        return
    times = sub.index.to_numpy()
    for col, color in [('is_spring', 'crimson'),
                       ('is_neap', 'royalblue')]:
        flag = sub[col].to_numpy()
        if not flag.any():
            continue
        # Find contiguous True runs
        change = np.diff(np.concatenate(([0], flag.astype(int), [0])))
        starts = np.where(change == 1)[0]
        ends = np.where(change == -1)[0]
        for s, e in zip(starts, ends):
            ax.axvspan(times[s], times[min(e, len(times) - 1)],
                       color=color, alpha=alpha, lw=0)


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------
def _draw_speed(ax, plon, plat, lon_rho, lat_rho, vx, vy, speed_plot,
                spd_max, layer_name, vel_title, quiver_stride, show_xlabel):
    cs = ax.pcolormesh(plon, plat, speed_plot, cmap='viridis',
                       shading='flat', vmin=0, vmax=spd_max)
    cb = plt.colorbar(cs, ax=ax, label='|u| [m s$^{-1}$]', shrink=0.85)
    cb.ax.tick_params(labelsize=8)
    st = quiver_stride
    spd_sub = speed_plot[::st, ::st]
    ax.quiver(lon_rho[::st, ::st], lat_rho[::st, ::st],
              np.where(np.isfinite(spd_sub), vx[::st, ::st], np.nan),
              np.where(np.isfinite(spd_sub), vy[::st, ::st], np.nan),
              scale=4.0, width=0.0025, color='k', alpha=0.7)
    ax.set_xlim(ZOOM_BOUNDS[0], ZOOM_BOUNDS[1])
    ax.set_ylim(ZOOM_BOUNDS[2], ZOOM_BOUNDS[3])
    pfun.dar(ax)
    ax.set_title(f'{vel_title} ({layer_name})', fontsize=10)
    ax.set_ylabel('lat', fontsize=9)
    if show_xlabel:
        ax.set_xlabel('lon', fontsize=9)
    else:
        ax.set_xticklabels([])
    ax.tick_params(labelsize=8)


def _draw_ow(ax, plon, plat, lon_rho, lat_rho, OW_plot, ow_lim,
             vortices_today, layer_name, show_xlabel):
    co = ax.pcolormesh(plon, plat, OW_plot, cmap='RdBu_r',
                       shading='flat', vmin=-ow_lim, vmax=ow_lim)
    cb = plt.colorbar(co, ax=ax, label='OW [s$^{-2}$]', shrink=0.85)
    cb.formatter.set_powerlimits((-2, 2)); cb.update_ticks()
    cb.ax.tick_params(labelsize=8)
    if not vortices_today.empty:
        lat0 = float(np.mean(lat_rho))
        m_per_deg_lat = 111_000.0
        m_per_deg_lon = 111_000.0 * np.cos(np.deg2rad(lat0))
        for _, v in vortices_today.iterrows():
            r_m = float(v.get('radius_m', np.nan))
            if not np.isfinite(r_m):
                continue
            rot = str(v.get('rotation', '')).upper()
            color = 'red' if rot == 'CW' else 'blue'
            r_deg = 0.5 * (r_m / m_per_deg_lat + r_m / m_per_deg_lon)
            ax.add_patch(Circle(
                (float(v['center_lon']), float(v['center_lat'])),
                radius=r_deg, fill=False, edgecolor=color, lw=1.6))
            ax.plot(float(v['center_lon']), float(v['center_lat']),
                    marker='+', color=color, ms=7, mew=1.3)
    ax.set_xlim(ZOOM_BOUNDS[0], ZOOM_BOUNDS[1])
    ax.set_ylim(ZOOM_BOUNDS[2], ZOOM_BOUNDS[3])
    pfun.dar(ax)
    n_feat = 0 if vortices_today.empty else len(vortices_today)
    ax.set_title(f'OW ({layer_name})  n={n_feat}  (red=CW, blue=CCW)',
                 fontsize=10)
    ax.set_ylabel('lat', fontsize=9)
    if show_xlabel:
        ax.set_xlabel('lon', fontsize=9)
    else:
        ax.set_xticklabels([])
    ax.tick_params(labelsize=8)


def render_frame(*, lowpass_nc, dsg, layers,
                 date_ts, vortices_by_layer, phase_df, do_times, do_vals,
                 ssh_t0, ssh_t1, do_t0, do_t1,
                 frame_path, vlims,
                 dpi=130, smooth_sigma=2.0, quiver_stride=3):
    """Render one frame with all layers stacked + 2 time series.

    layers : list of (vel_type, s_level, layer_name)
    vortices_by_layer : dict layer_name -> DataFrame of today's vortices.
    vlims : dict layer_name -> (spd_max, ow_lim)  (fixed across animation)
    """
    n_layers = len(layers)
    # Figure: n_layers rows of map pairs + 1 row of time series (2 cols)
    fig = plt.figure(figsize=(13, 3.6 * n_layers + 3.6))
    heights = [1.4] * n_layers + [1.0]
    gs = fig.add_gridspec(n_layers + 1, 2, height_ratios=heights,
                          hspace=0.40, wspace=0.22,
                          left=0.07, right=0.96, top=0.95, bottom=0.06)

    for i, (vel_type, s_level, layer_name) in enumerate(layers):
        speed_plot, OW_plot, vx, vy, lon_rho, lat_rho, vel_title = (
            compute_layer_fields(lowpass_nc, dsg, vel_type, s_level,
                                 smooth_sigma))
        plon, plat = pfun.get_plon_plat(lon_rho, lat_rho)
        spd_max, ow_lim = vlims[layer_name]
        show_xlabel = (i == n_layers - 1)
        ax_spd = fig.add_subplot(gs[i, 0])
        ax_ow = fig.add_subplot(gs[i, 1])
        _draw_speed(ax_spd, plon, plat, lon_rho, lat_rho, vx, vy,
                    speed_plot, spd_max, layer_name, vel_title,
                    quiver_stride, show_xlabel)
        _draw_ow(ax_ow, plon, plat, lon_rho, lat_rho, OW_plot, ow_lim,
                 vortices_by_layer.get(layer_name, pd.DataFrame()),
                 layer_name, show_xlabel)

    # --- Time-series row ---
    ax_ssh = fig.add_subplot(gs[n_layers, 0])
    ax_do = fig.add_subplot(gs[n_layers, 1])

    if phase_df is not None:
        sub = phase_df.loc[(phase_df.index >= ssh_t0) &
                           (phase_df.index <= ssh_t1)]
        shade_spring_neap(ax_ssh, phase_df, ssh_t0, ssh_t1)
        ax_ssh.plot(sub.index, sub['zeta'].values, color='0.35', lw=0.35,
                    alpha=0.45, label='hourly')
        ax_ssh.plot(sub.index,
                    pd.Series(sub['zeta'].values, index=sub.index)
                      .rolling('25h').mean().values,
                    color='k', lw=1.1, label='daily lowpass')
    t_mark = pd.Timestamp(date_ts) + pd.Timedelta(hours=12)
    ax_ssh.axvline(t_mark, color='firebrick', lw=1.3, ls='--')
    ax_ssh.set_xlim(ssh_t0, ssh_t1)
    ax_ssh.xaxis.set_major_locator(mdates.MonthLocator())
    ax_ssh.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax_ssh.set_ylabel('SSH [m]', fontsize=9)
    ax_ssh.set_title('Penn Cove SSH (crimson=spring, royal=neap)',
                     fontsize=10)
    ax_ssh.grid(alpha=0.3)
    ax_ssh.tick_params(labelsize=8)
    ax_ssh.legend(loc='upper right', fontsize=8)

    ax_do.plot(do_times, do_vals, color='steelblue', lw=1.0)
    ax_do.axhline(2.0, color='firebrick', ls=':', lw=0.9,
                  label='2 mg L$^{-1}$')
    idx = int(np.argmin(np.abs(np.asarray(do_times) - t_mark.to_numpy())))
    ax_do.plot(do_times[idx], do_vals[idx], marker='o', color='firebrick',
               ms=7, mec='k', mew=0.5)
    ax_do.axvline(t_mark, color='firebrick', lw=1.3, ls='--')
    ax_do.set_xlim(do_t0, do_t1)
    ax_do.xaxis.set_major_locator(mdates.MonthLocator())
    ax_do.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax_do.set_ylabel('M1 bottom DO [mg L$^{-1}$]', fontsize=9)
    ax_do.set_title('M1 bottom DO (2017)', fontsize=10)
    ax_do.grid(alpha=0.3)
    ax_do.tick_params(labelsize=8)
    ax_do.legend(loc='upper right', fontsize=8)

    fig.suptitle(date_ts.strftime('%Y-%m-%d'), fontsize=14, y=0.995)
    fig.savefig(frame_path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_event_groups(events_df):
    """
    Collapse overlapping events 1 and 2 into merged 'event 12' with the
    union of their windows; keep event 3 alone.

    Returns list of dicts: {label, eid_str, ds0, ds1, src_dirs[]}.
    """
    base = events_df.set_index('event_id')
    groups = []
    if 1 in base.index and 2 in base.index:
        r1, r2 = base.loc[1], base.loc[2]
        ds0 = min(pd.Timestamp(r1.lead_start), pd.Timestamp(r2.lead_start))
        ds1 = max(pd.Timestamp(r1.window_end), pd.Timestamp(r2.window_end))
        src = [
            f"event_01_{fmt_date(r1.lead_start)}_{fmt_date(r1.window_end)}",
            f"event_02_{fmt_date(r2.lead_start)}_{fmt_date(r2.window_end)}",
        ]
        groups.append(dict(label='event_12', eid_str='12',
                           ds0=ds0, ds1=ds1, src_dirs=src))
    for eid in sorted(base.index):
        if eid in (1, 2):
            continue
        r = base.loc[eid]
        ds0 = pd.Timestamp(r.lead_start)
        ds1 = pd.Timestamp(r.window_end)
        src = [f"event_{eid:02d}_{fmt_date(ds0)}_{fmt_date(ds1)}"]
        groups.append(dict(label=f'event_{eid:02d}', eid_str=str(eid),
                           ds0=ds0, ds1=ds1, src_dirs=src))
    return groups


def main():
    args = parse_args()
    if not _HAS_LO_TOOLS:
        raise RuntimeError('lo_tools is required (run on apogee).')

    gridname, tag, ex_name = args.gtagex.split('_')
    Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
    if args.roms_out_num > 0:
        Ldir['roms_out'] = Ldir['roms_out' + str(args.roms_out_num)]
    gtagex = args.gtagex

    # ---- Resolve paths ----
    base_out = (Path(args.out_dir) if args.out_dir
                else Ldir['LOo'] / 'swirl' / gtagex / 'hypoxia_events')
    anim_root = base_out / '_animations'
    anim_root.mkdir(parents=True, exist_ok=True)

    events_csv = (Path(args.events_csv) if args.events_csv
                  else Ldir['LOo'] / 'swirl' / gtagex
                  / f'hypoxia_events_{args.mooring}_{args.year}.csv')
    events = pd.read_csv(events_csv, parse_dates=[
        'event_start', 'event_end', 'lead_start', 'window_end'])
    groups = build_event_groups(events)

    moor_file = (Path(args.mooring_file) if args.mooring_file
                 else Ldir['LOo'] / 'extract' / gtagex / 'moor' / args.job
                 / f'{args.mooring}_{args.year}.01.02_'
                   f'{args.year}.12.30.nc')
    do_times, do_vals = load_mooring_DO(moor_file)

    phase_file = (Path(args.phase_file) if args.phase_file
                  else Ldir['LOo'] / 'tide_phase' / gtagex
                  / f'tide_phases_{args.year}.01.01_{args.year}.12.31'
                  / 'penn_cove.nc')
    phase_df = load_phase(phase_file) if phase_file.exists() else None
    if phase_df is None:
        print(f'WARNING: no tide-phase file at {phase_file}; '
              f'SSH panel will lack spring/neap shading.')

    # Grid file
    grid_file = Ldir['grid'] / 'grid.nc'
    dsg = xr.open_dataset(grid_file)

    layers = select_layers(args.layers)

    # Full-year axes limits
    ssh_t0 = pd.Timestamp(f'{args.year}-01-01')
    ssh_t1 = pd.Timestamp(f'{args.year}-12-31')
    do_t0, do_t1 = ssh_t0, ssh_t1

    print(f'gtagex      : {gtagex}')
    print(f'events CSV  : {events_csv}')
    print(f'groups      : {[g["label"] for g in groups]}')
    print(f'layers      : {[L[2] for L in layers]}')
    print(f'anim root   : {anim_root}')
    print(f'mooring NC  : {moor_file}')
    print(f'phase NC    : {phase_file}')
    print(f'grid NC     : {grid_file}')
    print()

    # Build combined date list (union of group windows, deduped)
    date_recs = []  # list of (date_ts, group)
    seen = set()
    for g in groups:
        d = g['ds0']
        while d <= g['ds1']:
            key = d.normalize()
            if key not in seen:
                date_recs.append((d.normalize(), g))
                seen.add(key)
            d += pd.Timedelta(days=1)
    date_recs.sort(key=lambda r: r[0])

    # Per-layer vortex CSVs grouped by group label
    vort_by_layer_group = {}  # vort_by_layer_group[layer_name][group_label] = df
    for vel_type, s_level, layer_name in layers:
        vort_by_layer_group[layer_name] = {}
        for g in groups:
            src_dirs = [base_out / s for s in g['src_dirs']]
            vort_by_layer_group[layer_name][g['label']] = \
                load_event_vortices(src_dirs, layer_name)

    frames_dir = anim_root / 'frames_all'
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f'--- combined animation: {len(date_recs)} frames, '
          f'{len(layers)} layers per frame ---')

    if args.dry:
        for i, (d, g) in enumerate(date_recs):
            print(f'  [{i:04d}] {d.date()}  ({g["label"]})')
        return

    # --- Per-layer prescan to fix colorbar limits across the whole animation ---
    vlims = {}
    for vel_type, s_level, layer_name in layers:
        print(f'  prescanning layer {layer_name} ...')
        spd_p98, ow_p98 = [], []
        for d, g in date_recs:
            ds_str = d.strftime('%Y.%m.%d')
            lp = find_lowpassed(ds_str, Ldir, gtagex)
            if lp is None:
                continue
            try:
                speed_m, OW_m, *_ = compute_layer_fields(
                    lp, dsg, vel_type, s_level, args.smooth_sigma)
            except Exception as e:
                print(f'    {ds_str} prescan failed: {e}')
                continue
            sp = speed_m[np.isfinite(speed_m)]
            ow = OW_m[np.isfinite(OW_m)]
            if sp.size:
                spd_p98.append(float(np.nanpercentile(sp, 98)))
            if ow.size:
                ow_p98.append(float(np.nanpercentile(np.abs(ow), 98)))
        spd_max = max(float(np.nanmax(spd_p98)) if spd_p98 else 0.5, 0.05)
        ow_lim = max(float(np.nanmax(ow_p98)) if ow_p98 else 1e-8, 1e-9)
        vlims[layer_name] = (spd_max, ow_lim)
        print(f'    {layer_name}: spd_max={spd_max:.3f} m/s   '
              f'|OW|_max={ow_lim:.2e} 1/s^2')

    # --- Render frames ---
    rendered = 0
    skipped = 0
    missing = 0
    for i, (d, g) in enumerate(date_recs):
        ds_str = d.strftime('%Y.%m.%d')
        frame_path = frames_dir / f'frame_{i:04d}.png'
        if args.skip_existing and frame_path.exists():
            skipped += 1
            continue
        lp = find_lowpassed(ds_str, Ldir, gtagex)
        if lp is None:
            print(f'  [{i:04d}] {ds_str}  MISSING lowpassed.nc -- skip')
            missing += 1
            continue

        vortices_by_layer = {}
        for _, _, layer_name in layers:
            v_all = vort_by_layer_group[layer_name][g['label']]
            if not v_all.empty and 'date' in v_all.columns:
                vortices_by_layer[layer_name] = \
                    v_all[v_all['date'] == d.normalize()]
            else:
                vortices_by_layer[layer_name] = pd.DataFrame()

        try:
            render_frame(
                lowpass_nc=lp, dsg=dsg, layers=layers,
                date_ts=d,
                vortices_by_layer=vortices_by_layer,
                phase_df=phase_df,
                do_times=do_times, do_vals=do_vals,
                ssh_t0=ssh_t0, ssh_t1=ssh_t1,
                do_t0=do_t0, do_t1=do_t1,
                frame_path=frame_path,
                vlims=vlims,
                dpi=args.dpi, smooth_sigma=args.smooth_sigma,
                quiver_stride=args.quiver_stride,
            )
            rendered += 1
            print(f'  [{i:04d}] {ds_str}  ({g["label"]})  ok')
        except Exception as e:
            print(f'  [{i:04d}] {ds_str}  FAILED: {e}')

    print(f'rendered={rendered}  skipped={skipped}  missing={missing}')

    # ---- ffmpeg into one mp4 ----
    if rendered + skipped == 0:
        print('no frames; skipping ffmpeg.')
    else:
        mp4_path = anim_root / 'anim_all_layers.mp4'
        if shutil.which('ffmpeg') is None:
            print('ffmpeg not on PATH; mp4 not built.')
        else:
            cmd = [
                'ffmpeg', '-y', '-r', str(args.fps),
                '-i', str(frames_dir / 'frame_%04d.png'),
                '-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2',
                '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23',
                str(mp4_path),
            ]
            print(' '.join(cmd))
            try:
                subprocess.run(cmd, check=True)
                print(f'wrote {mp4_path}')
            except subprocess.CalledProcessError as e:
                print(f'ffmpeg FAILED ({e.returncode})')

    dsg.close()
    print('done.')


if __name__ == '__main__':
    main()
