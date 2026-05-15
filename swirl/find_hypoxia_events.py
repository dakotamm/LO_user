"""
Find bottom-hypoxia event windows from a Penn Cove mooring extraction.

Reads a single mooring NetCDF (default M1 in pc0/wb1_r0_xn11b), thresholds
bottom oxygen at 2 mg/L on a daily basis, and writes a CSV of contiguous
hypoxic events plus a 14-day lead-up window for each event. Also produces
a QC plot of the bottom DO time series with shaded events + lead-ups.

Usage:
    python find_hypoxia_events.py -gtx wb1_r0_xn11b -mooring M1 -year 2017

Outputs (under Ldir['LOo']/swirl/<gtx>/):
    hypoxia_events_<mooring>_<year>.csv
    plots/hypoxia_events_<mooring>_<year>.png

CSV columns:
    event_id, event_start, event_end, lead_start, n_event_days,
    n_window_days
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from lo_tools import Lfun


# --- Constants ---
DO_UM_TO_MGL = 32.0 / 1000.0     # multiply mmol/m^3 (=uM) -> mg/L
HYPOXIA_THRESHOLD_MGL = 2.0      # standard bottom-water hypoxia cutoff
LEADUP_DAYS = 14                 # window prefix before each event


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('-gtx', '--gtagex', type=str, default='wb1_r0_xn11b')
    p.add_argument('-mooring', type=str, default='M1',
                   help='Penn Cove mooring station id (M1, M3, M5).')
    p.add_argument('-job', type=str, default='pc0',
                   help='Mooring extract job name (subdir of moor/).')
    p.add_argument('-year', type=int, default=2017)
    p.add_argument('-ds0', type=str, default=None,
                   help='Override mooring file start date (YYYY.MM.DD).')
    p.add_argument('-ds1', type=str, default=None,
                   help='Override mooring file end date (YYYY.MM.DD).')
    p.add_argument('-thresh', type=float, default=HYPOXIA_THRESHOLD_MGL,
                   help='Bottom DO threshold [mg/L]. Default 2.0.')
    p.add_argument('-leadup_days', type=int, default=LEADUP_DAYS)
    p.add_argument('-out_dir', type=str, default=None,
                   help='Override output dir (default LOo/swirl/<gtx>).')
    return p.parse_args()


def load_bottom_do(moor_fn):
    """Return a pandas Series of bottom DO [mg/L] indexed by ocean_time."""
    with xr.open_dataset(moor_fn) as ds:
        # oxygen dims: (ocean_time, s_rho); s_rho index 0 = bottom
        oxygen_bottom_uM = ds.oxygen.values[:, 0]
        times = pd.to_datetime(ds.ocean_time.values)
    do_mgl = oxygen_bottom_uM * DO_UM_TO_MGL
    return pd.Series(do_mgl, index=times, name='bot_DO_mgL')


def daily_hypoxic_flags(do_series, thresh):
    """Resample to daily mean and return a bool Series (True = hypoxic)."""
    daily = do_series.resample('1D').mean()
    return daily, (daily < thresh)


def find_event_runs(daily_index, hypoxic_flags):
    """
    Run-length-encode contiguous True spans in `hypoxic_flags`.

    Returns a list of (event_start_ts, event_end_ts) inclusive on both ends,
    snapped to daily timestamps.
    """
    flags = hypoxic_flags.values.astype(bool)
    if not flags.any():
        return []
    # Pad with False on both sides to detect rising/falling edges
    padded = np.concatenate([[False], flags, [False]])
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0] - 1   # inclusive end index
    events = []
    for s, e in zip(starts, ends):
        events.append((daily_index[s], daily_index[e]))
    return events


def build_events_dataframe(events, leadup_days):
    rows = []
    for i, (es, ee) in enumerate(events):
        lead = es - pd.Timedelta(days=leadup_days)
        n_event = (ee - es).days + 1
        n_window = (ee - lead).days + 1
        rows.append(dict(
            event_id=i + 1,
            event_start=es,
            event_end=ee,
            lead_start=lead,
            n_event_days=n_event,
            n_window_days=n_window,
        ))
    return pd.DataFrame(rows)


def plot_qc(daily_do, events_df, thresh, out_path, title):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily_do.index, daily_do.values, color='k', lw=0.8,
            label='daily mean bottom DO')
    ax.axhline(thresh, color='red', lw=1, ls='--',
               label=f'{thresh:.1f} mg/L threshold')
    for _, row in events_df.iterrows():
        # Lead-up shaded lighter, event shaded darker
        ax.axvspan(row['lead_start'], row['event_start'],
                   color='goldenrod', alpha=0.15)
        ax.axvspan(row['event_start'], row['event_end'],
                   color='firebrick', alpha=0.30)
        ax.text(row['event_start'], ax.get_ylim()[1] * 0.95,
                f"#{row['event_id']}", color='firebrick',
                fontsize=8, ha='left', va='top')
    ax.set_ylabel('bottom DO [mg/L]')
    ax.set_ylim(0, max(8, np.nanmax(daily_do.values) * 1.05))
    ax.set_title(title)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.legend(loc='upper right', fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    gridname, tag, ex_name = args.gtagex.split('_')
    Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)

    # Resolve mooring filename
    if args.ds0 is not None and args.ds1 is not None:
        ds0_str, ds1_str = args.ds0, args.ds1
    else:
        ds0_str = f'{args.year}.01.02'
        ds1_str = f'{args.year}.12.30'
    moor_fn = (Ldir['LOo'] / 'extract' / args.gtagex / 'moor' / args.job
               / f'{args.mooring}_{ds0_str}_{ds1_str}.nc')
    if not moor_fn.exists():
        raise FileNotFoundError(f'Mooring file not found: {moor_fn}')
    print(f'Reading {moor_fn}')

    do_series = load_bottom_do(moor_fn)
    daily_do, hypoxic = daily_hypoxic_flags(do_series, args.thresh)
    events = find_event_runs(daily_do.index, hypoxic)
    events_df = build_events_dataframe(events, args.leadup_days)

    print(f'Found {len(events_df)} hypoxia events at {args.mooring} '
          f'(threshold {args.thresh} mg/L):')
    if len(events_df):
        print(events_df.to_string(index=False))

    # Resolve output dir
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Ldir['LOo'] / 'swirl' / args.gtagex
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f'hypoxia_events_{args.mooring}_{args.year}.csv'
    events_df.to_csv(csv_path, index=False)
    print(f'Wrote {csv_path}')

    plot_path = (out_dir / 'plots'
                 / f'hypoxia_events_{args.mooring}_{args.year}.png')
    plot_qc(daily_do, events_df, args.thresh, plot_path,
            title=(f'{args.gtagex} {args.mooring} bottom DO {args.year} '
                   f'(events shaded; {args.leadup_days}-d lead-up in tan)'))
    print(f'Wrote {plot_path}')


if __name__ == '__main__':
    main()
