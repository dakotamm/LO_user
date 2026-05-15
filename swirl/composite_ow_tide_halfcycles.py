"""
Composite OW maps per individual tide half-cycle within each hypoxia event
window.

For each event in the events CSV, define the window as
    [event_start - leadup_days, event_end + postevent_days]
(default 7 lead, 1 recovery), pull every hourly ROMS `his` file in that
window, group snapshots by individual flood/ebb half-cycle using tide-phase
labels from compute_tide_phases.py, and produce one figure per layer
(surface, depth_avg, bottom) with one subplot per half-cycle showing the
mean OW field over that half-cycle. Each subplot is framed by spring/neap
status and labelled with the half-cycle's start-end time and snapshot count.

Prerequisite: run compute_tide_phases.py to generate
    LOo/tide_phase/<gtagex>/tide_phases_<ds0>_<ds1>/<label>.nc
covering the event windows. By default this script expects
ds0=YYYY.01.01, ds1=YYYY.12.31 and label='penn_cove' but both are
overridable.

Usage on apogee:
    conda activate loenv
    python composite_ow_tide_halfcycles.py \
        -gtx wb1_r0_xn11b -ro 2 -event_id 1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from lo_tools import Lfun

# Import helpers from sibling module (module guards argparse on import)
sys.path.insert(0, str(Path(__file__).parent))
from run_swirl_roms import (
    get_velocity_2d, subset_to_bbox, get_grid_spacing,
    detect_ow_features, find_date_dir, pfun,
)


PENN_COVE_BBOX = (-122.74, -122.625, 48.215, 48.245)

LAYERS = [
    ('surface',     None, 'surface'),
    ('depth_avg',   None, 'depth_avg'),
    ('depth_level',  0,   'bottom'),
]


def _bool(s):
    return str(s).lower() in ('true', '1', 'yes')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('-gtx', '--gtagex', type=str, default='wb1_r0_xn11b')
    p.add_argument('-ro', '--roms_out_num', type=int, default=2)
    p.add_argument('-mooring', type=str, default='M1')
    p.add_argument('-year', type=int, default=2017)
    p.add_argument('-events_csv', type=str, default=None)
    p.add_argument('-event_id', type=int, default=None,
                   help='If given, process only this event_id.')
    p.add_argument('-leadup_days', type=int, default=7)
    p.add_argument('-postevent_days', type=int, default=1)
    p.add_argument('-ftype', type=str, default='his',
                   choices=['his', 'avg'])
    p.add_argument('-layers', type=str, default='all',
                   help="Comma list from {surface,depth_avg,bottom} or 'all'.")
    p.add_argument('-min_cells', type=int, default=9)
    p.add_argument('-smooth', type=float, default=2.0)
    # Tide phase file (output of compute_tide_phases.py)
    p.add_argument('-tide_label', type=str, default='penn_cove')
    p.add_argument('-tide_ds0', type=str, default=None,
                   help='Phase NC range start (default YYYY.01.01).')
    p.add_argument('-tide_ds1', type=str, default=None,
                   help='Phase NC range end (default YYYY.12.31).')
    return p.parse_args()


def select_layers(spec):
    if spec.strip().lower() == 'all':
        return LAYERS
    wanted = {s.strip() for s in spec.split(',')}
    chosen = [L for L in LAYERS if L[2] in wanted]
    if not chosen:
        raise ValueError(f'No valid layers in {spec!r}')
    return chosen


def load_tide_phases(phase_fn):
    """Load the tide-phase NC and return a DataFrame indexed by time."""
    with xr.open_dataset(phase_fn) as dsp:
        t = pd.to_datetime(dsp['time'].values)
        is_flood = dsp['is_flood'].values.astype(bool)
        is_ebb = dsp['is_ebb'].values.astype(bool)
        is_spring = dsp['is_spring'].values.astype(bool)
        is_neap = dsp['is_neap'].values.astype(bool)
    return pd.DataFrame(dict(
        time=t, is_flood=is_flood, is_ebb=is_ebb,
        is_spring=is_spring, is_neap=is_neap,
    )).set_index('time').sort_index()


def find_halfcycles(phase_df, window_start, window_end):
    """
    Identify each contiguous flood or ebb half-cycle whose midpoint
    falls inside [window_start, window_end].

    Returns a list of dicts:
        dict(kind='flood'|'ebb', t_start, t_end, is_spring (majority))
    """
    sub = phase_df.loc[window_start:window_end].copy()
    if sub.empty:
        return []

    # Build a "state" column: 0=slack/other, 1=flood, -1=ebb
    state = np.zeros(len(sub), dtype=int)
    state[sub.is_flood.values] = 1
    state[sub.is_ebb.values] = -1
    sub['state'] = state

    # Run-length encode non-zero states
    times = sub.index.values
    runs = []
    i = 0
    n = len(state)
    while i < n:
        if state[i] == 0:
            i += 1
            continue
        s = state[i]
        j = i
        while j < n and state[j] == s:
            j += 1
        # half-cycle = indices [i, j)
        seg = sub.iloc[i:j]
        kind = 'flood' if s == 1 else 'ebb'
        is_spring_maj = bool(seg.is_spring.sum() >= (j - i) / 2.0)
        runs.append(dict(
            kind=kind,
            t_start=pd.Timestamp(times[i]),
            t_end=pd.Timestamp(times[j - 1]),
            is_spring=is_spring_maj,
        ))
        i = j
    return runs


def get_his_fn_list(date_str, Ldir, gtagex, ftype):
    """Return sorted list of hourly nc files for one date."""
    date_dir = find_date_dir(date_str, Ldir, gtagex)
    if date_dir is None or not date_dir.exists():
        return []
    return sorted(date_dir.glob(f'ocean_{ftype}_*.nc'))


def collect_snapshot_files(window_start, window_end, Ldir, gtagex, ftype):
    """Return list of (timestamp, Path) for every nc file inside window."""
    out = []
    d = pd.Timestamp(window_start).normalize()
    end = pd.Timestamp(window_end).normalize()
    while d <= end:
        ds = d.strftime('%Y.%m.%d')
        for fn in get_his_fn_list(ds, Ldir, gtagex, ftype):
            try:
                with xr.open_dataset(fn) as _ds:
                    t = pd.Timestamp(_ds.ocean_time.values[0])
            except Exception:
                continue
            if window_start <= t <= window_end:
                out.append((t, fn))
        d += pd.Timedelta(days=1)
    return out


def compute_mean_ow(files_in_cycle, dsg, bbox, vel_type, s_level,
                    smooth_sigma, min_cells, ow_thresh=None):
    """
    Open each file in `files_in_cycle`, compute the OW field on the
    Penn Cove subset, and return (mean_OW, dsg_sub, dx_m, dy_m,
    features_on_mean).
    """
    OW_sum = None
    OW_count = 0
    dsg_sub_out = None
    dx_m_out = dy_m_out = None
    for _, fn in files_in_cycle:
        try:
            with xr.open_dataset(fn) as ds:
                vx, vy, _ = get_velocity_2d(ds, dsg, vel_type, s_level)
        except Exception as e:
            print(f'    skip {fn.name}: {e}')
            continue
        if bbox is not None:
            vx, vy, dsg_sub, _, _ = subset_to_bbox(vx, vy, dsg, *bbox)
            dx_m, dy_m = get_grid_spacing(dsg_sub)
        else:
            dsg_sub = dsg
            dx_m, dy_m = get_grid_spacing(dsg)
        # Use detect_ow_features just to get the OW field; we'll redetect
        # on the composite. It computes OW and returns it.
        _, OW, _ = detect_ow_features(
            vx, vy, dsg_sub, dx_m, dy_m,
            ow_thresh=ow_thresh,
            min_cells=min_cells,
            smooth_sigma=smooth_sigma)
        if OW_sum is None:
            OW_sum = np.zeros_like(OW)
            dsg_sub_out = dsg_sub
            dx_m_out, dy_m_out = dx_m, dy_m
        OW_sum += OW
        OW_count += 1
    if OW_count == 0:
        return None
    OW_mean = OW_sum / OW_count
    return OW_mean, dsg_sub_out, dx_m_out, dy_m_out, OW_count


def detect_features_on_mean_ow(OW_mean, dsg_sub, dx_m, dy_m,
                               smooth_sigma, min_cells, ow_thresh=None):
    """Re-run feature detection on the composite OW field by treating it
    as a velocity-derived OW. We bypass the velocity step by patching
    detect_ow_features through a thin shim that uses the precomputed OW.

    Simplest approach: just threshold OW_mean directly.
    """
    from scipy.ndimage import label as nd_label
    mask_rho = dsg_sub.mask_rho.values
    ow = OW_mean.copy()
    ow[mask_rho == 0] = np.nan
    if ow_thresh is None:
        sigma = float(np.nanstd(ow))
        ow_thresh = -0.2 * sigma
    rot_mask = (ow < ow_thresh) & (mask_rho == 1)
    lab, nlab = nd_label(rot_mask)
    feats = []
    for k in range(1, nlab + 1):
        cells = (lab == k)
        n = int(cells.sum())
        if n < min_cells:
            continue
        ii, jj = np.where(cells)
        clon = float(np.nanmean(dsg_sub.lon_rho.values[ii, jj]))
        clat = float(np.nanmean(dsg_sub.lat_rho.values[ii, jj]))
        radius_m = float(np.sqrt(n * dx_m * dy_m / np.pi))
        feats.append(dict(
            center_lon=clon, center_lat=clat, radius_m=radius_m,
            n_cells=n, mean_ow=float(np.nanmean(ow[cells])),
        ))
    return feats, ow_thresh


def plot_composite_grid(halfcycles_with_data, layer_name, event_id,
                        window_start, window_end, out_path):
    """
    halfcycles_with_data : list of dicts with keys:
        kind, t_start, t_end, is_spring, OW_mean, dsg_sub, n_snaps, features
    """
    n = len(halfcycles_with_data)
    if n == 0:
        print(f'  No half-cycles for layer={layer_name}, skipping.')
        return
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 5.0 * nrows),
                             squeeze=False)

    # Global symmetric colour limit
    all_ow = np.concatenate([
        hc['OW_mean'][~np.isnan(hc['OW_mean'])].ravel()
        for hc in halfcycles_with_data
    ])
    vmax = float(np.nanpercentile(np.abs(all_ow), 95))

    for k, hc in enumerate(halfcycles_with_data):
        r, c = divmod(k, ncols)
        ax = axes[r, c]
        dsg_sub = hc['dsg_sub']
        ny, nx = hc['OW_mean'].shape
        lon = dsg_sub.lon_rho.values[:ny, :nx]
        lat = dsg_sub.lat_rho.values[:ny, :nx]
        mask = dsg_sub.mask_rho.values[:ny, :nx]
        ow_plot = hc['OW_mean'].copy()
        ow_plot[mask == 0] = np.nan
        plon, plat = pfun.get_plon_plat(lon, lat)
        cs = ax.pcolormesh(plon, plat, ow_plot, cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax, shading='flat')
        # Overlay feature footprints
        for feat in hc['features']:
            rdeg = feat['radius_m'] / 111000.0
            circ = mpatches.Circle((feat['center_lon'], feat['center_lat']),
                                   rdeg, fill=False,
                                   edgecolor='black', lw=1.5, ls='--')
            ax.add_patch(circ)
        pfun.dar(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        # Frame colour by spring/neap
        frame_color = 'crimson' if hc['is_spring'] else 'royalblue'
        for spine in ax.spines.values():
            spine.set_edgecolor(frame_color)
            spine.set_linewidth(2.5)
        sn_label = 'SPRING' if hc['is_spring'] else 'NEAP'
        kind_label = hc['kind'].upper()
        t0 = hc['t_start'].strftime('%m/%d %H:%M')
        t1 = hc['t_end'].strftime('%m/%d %H:%M')
        ax.set_title(f'{kind_label} | {sn_label}\n'
                     f'{t0} \u2192 {t1}  (n={hc["n_snaps"]})',
                     fontsize=9, color=frame_color)

    # Hide unused subplots
    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis('off')

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(cs, cax=cbar_ax, label='Mean Okubo-Weiss [1/s\u00b2]')

    fig.suptitle(
        f'Event {event_id:02d}  |  layer={layer_name}  |  '
        f'{window_start:%Y-%m-%d} \u2192 {window_end:%Y-%m-%d}  |  '
        f'half-cycle composites (red frame=spring, blue=neap)',
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.91, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Wrote {out_path}')


def main():
    args = parse_args()
    gridname, tag, ex_name = args.gtagex.split('_')
    Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
    if args.roms_out_num > 0:
        Ldir['roms_out'] = Ldir['roms_out' + str(args.roms_out_num)]

    # --- Events CSV ---
    if args.events_csv is not None:
        events_csv = Path(args.events_csv)
    else:
        events_csv = (Ldir['LOo'] / 'swirl' / args.gtagex
                      / f'hypoxia_events_{args.mooring}_{args.year}.csv')
    if not events_csv.exists():
        raise FileNotFoundError(f'Events CSV not found: {events_csv}')
    events = pd.read_csv(events_csv,
                         parse_dates=['event_start', 'event_end',
                                      'lead_start', 'window_end'])
    if args.event_id is not None:
        events = events[events.event_id == args.event_id]
        if events.empty:
            raise ValueError(f'event_id {args.event_id} not in CSV')

    # --- Tide-phase NC ---
    tide_ds0 = args.tide_ds0 or f'{args.year}.01.01'
    tide_ds1 = args.tide_ds1 or f'{args.year}.12.31'
    phase_fn = (Ldir['LOo'] / 'tide_phase' / args.gtagex
                / f'tide_phases_{tide_ds0}_{tide_ds1}'
                / f'{args.tide_label}.nc')
    if not phase_fn.exists():
        raise FileNotFoundError(
            f'Tide-phase NC not found: {phase_fn}\n'
            f'Run compute_tide_phases.py first '
            f'(-label {args.tide_label} -0 {tide_ds0} -1 {tide_ds1}).')
    print(f'Loading tide phases from {phase_fn}')
    phase_df = load_tide_phases(phase_fn)

    # --- Grid ---
    grid_file = Ldir['grid'] / 'grid.nc'
    dsg = xr.open_dataset(grid_file)
    print(f'Grid: {grid_file}')

    layers = select_layers(args.layers)
    base_out = Ldir['LOo'] / 'swirl' / args.gtagex / 'hypoxia_events'

    for _, row in events.iterrows():
        eid = int(row['event_id'])
        win_start = pd.Timestamp(row['event_start']) - \
            pd.Timedelta(days=args.leadup_days)
        win_end = pd.Timestamp(row['event_end']) + \
            pd.Timedelta(days=args.postevent_days)
        ds0_str = win_start.strftime('%Y.%m.%d')
        ds1_str = win_end.strftime('%Y.%m.%d')
        print(f'\n=== Event {eid}: {ds0_str} -> {ds1_str} ===')

        halfcycles = find_halfcycles(phase_df, win_start, win_end)
        print(f'  {len(halfcycles)} half-cycles in window.')
        if not halfcycles:
            continue

        snapshot_files = collect_snapshot_files(
            win_start, win_end, Ldir, args.gtagex, args.ftype)
        print(f'  {len(snapshot_files)} {args.ftype} snapshots in window.')

        event_dir = base_out / f'event_{eid:02d}_composites'
        event_dir.mkdir(parents=True, exist_ok=True)

        for vel_type, s_level, layer_name in layers:
            print(f'  --- layer={layer_name} ---')
            hc_data = []
            for hc in halfcycles:
                files_in = [(t, fn) for (t, fn) in snapshot_files
                            if hc['t_start'] <= t <= hc['t_end']]
                if not files_in:
                    print(f'    no files for {hc["kind"]} '
                          f'{hc["t_start"]} -> {hc["t_end"]}')
                    continue
                res = compute_mean_ow(
                    files_in, dsg, PENN_COVE_BBOX, vel_type, s_level,
                    args.smooth, args.min_cells)
                if res is None:
                    continue
                OW_mean, dsg_sub, dx_m, dy_m, n_snaps = res
                feats, ow_thresh = detect_features_on_mean_ow(
                    OW_mean, dsg_sub, dx_m, dy_m,
                    args.smooth, args.min_cells)
                hc_data.append(dict(
                    kind=hc['kind'],
                    t_start=hc['t_start'],
                    t_end=hc['t_end'],
                    is_spring=hc['is_spring'],
                    OW_mean=OW_mean,
                    dsg_sub=dsg_sub,
                    n_snaps=n_snaps,
                    features=feats,
                ))

            out_path = (event_dir
                        / f'ow_composite_halfcycles_{layer_name}_'
                          f'{ds0_str}_{ds1_str}.png')
            plot_composite_grid(hc_data, layer_name, eid,
                                win_start, win_end, out_path)


if __name__ == '__main__':
    main()
