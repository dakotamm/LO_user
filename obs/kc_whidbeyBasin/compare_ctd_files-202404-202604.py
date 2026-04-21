"""
Compare old and new KC Whidbey Basin CTD CSV files.

This script reports:
- Column/schema differences
- Date coverage differences
- Multiset row overlap on processing-relevant columns
- Value differences for shared CTD measurement keys
- A summary figure saved in this folder
"""

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lo_tools import Lfun

Ldir = Lfun.Lstart()

SOURCE = 'kc_whidbeyBasin'
OTYPE = 'ctd'
DATA_DIR = Ldir['data'] / 'obs' / SOURCE / OTYPE
OLD_FP = DATA_DIR / 'old' / 'Whidbey_Basin_CTD_Casts_April2024.csv'
NEW_FP = DATA_DIR / 'Whidbey_Basin_CTD_Casts_20260420.csv'
PLOT_FP = Path(__file__).parent / 'ctd_comparison.png'

COMPARE_COLS = [
    'Sample Date', 'Locator', 'Up Down', 'Depth (meters)',
    'Temperature (°C)', 'Salinity (PSU)', 'Dissolved Oxygen (mg/L)', 'Chlorophyll (µg/L)'
]

VALUE_KEY_COLS = ['Sample Date', 'Locator', 'Up Down', 'Depth (meters)']
VALUE_COLS = ['Temperature (°C)', 'Salinity (PSU)', 'Dissolved Oxygen (mg/L)', 'Chlorophyll (µg/L)']


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['Sample Date'] = pd.to_datetime(out['Sample Date'], errors='coerce', format='mixed')
    out['Sample Date'] = out['Sample Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    for c in ['Depth (meters)'] + VALUE_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce')

    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].fillna('').astype(str).str.strip()

    return out


def _row_counter(df: pd.DataFrame, cols: list[str]) -> Counter:
    # Canonicalize to strings so NaN/None/type formatting do not break equality checks.
    tmp = df[cols].copy()
    for c in cols:
        tmp[c] = tmp[c].astype(object).where(tmp[c].notna(), '__MISSING__').astype(str).str.strip()
    rows = map(tuple, tmp.itertuples(index=False, name=None))
    return Counter(rows)


def _overlap_stats(old_df: pd.DataFrame, new_df: pd.DataFrame, cols: list[str], label: str) -> None:
    old_ctr = _row_counter(old_df, cols)
    new_ctr = _row_counter(new_df, cols)

    overlap = 0
    for k, old_count in old_ctr.items():
        overlap += min(old_count, new_ctr.get(k, 0))

    old_total = len(old_df)
    new_total = len(new_df)
    print(f'\n[{label}]')
    print(f'  old rows: {old_total:,}')
    print(f'  new rows: {new_total:,}')
    print(f'  overlapping rows: {overlap:,}')
    print(f'  old-only rows: {old_total - overlap:,}')
    print(f'  new-only rows: {new_total - overlap:,}')


def _year_range(df: pd.DataFrame, label: str) -> None:
    t = pd.to_datetime(df['Sample Date'], errors='coerce')
    years = t.dt.year.dropna().astype(int)
    if len(years) == 0:
        print(f'{label}: no parseable dates')
    else:
        print(f'{label}: {years.min()}-{years.max()}')


def _compare_values_on_shared_keys(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    old_x = old_df[VALUE_KEY_COLS + VALUE_COLS].copy()
    new_x = new_df[VALUE_KEY_COLS + VALUE_COLS].copy()

    old_x = old_x.sort_values(VALUE_KEY_COLS).drop_duplicates(VALUE_KEY_COLS, keep='first')
    new_x = new_x.sort_values(VALUE_KEY_COLS).drop_duplicates(VALUE_KEY_COLS, keep='first')

    m = old_x.merge(new_x, on=VALUE_KEY_COLS, how='inner', suffixes=('_old', '_new'))
    print('\n[Value comparison on shared keys]')
    print(f'  shared keys: {len(m):,}')
    if len(m) == 0:
        return m

    for vn in VALUE_COLS:
        vo = m[f'{vn}_old']
        vnw = m[f'{vn}_new']
        good = vo.notna() & vnw.notna()
        if good.sum() == 0:
            print(f'  {vn}: no paired numeric values')
            continue
        equal = np.isclose(vo[good], vnw[good], rtol=1e-6, atol=1e-10)
        print(f'  {vn}: matched={equal.sum():,}, mismatched={(~equal).sum():,}, paired={good.sum():,}')

    # Show a small mismatch sample across all value columns.
    mismatch_mask = pd.Series(False, index=m.index)
    for vn in VALUE_COLS:
        vo = m[f'{vn}_old']
        vnw = m[f'{vn}_new']
        good = vo.notna() & vnw.notna()
        this_bad = pd.Series(False, index=m.index)
        this_bad.loc[good] = ~np.isclose(vo[good], vnw[good], rtol=1e-6, atol=1e-10)
        mismatch_mask |= this_bad

    mm_cols = VALUE_KEY_COLS + [f'{v}_old' for v in VALUE_COLS] + [f'{v}_new' for v in VALUE_COLS]
    mm = m.loc[mismatch_mask, mm_cols].head(10)
    if len(mm) > 0:
        print('\n  sample mismatches (first 10):')
        print(mm.to_string(index=False))

    return m


def _make_summary_figure(old_df: pd.DataFrame, new_df: pd.DataFrame, m_values: pd.DataFrame) -> None:
    old_down = old_df[old_df['Up Down'] == 'Down'].copy()
    new_down = new_df[new_df['Up Down'] == 'Down'].copy()

    old_t = pd.to_datetime(old_down['Sample Date'], errors='coerce')
    new_t = pd.to_datetime(new_down['Sample Date'], errors='coerce')
    old_year_counts = old_t.dt.year.value_counts().sort_index()
    new_year_counts = new_t.dt.year.value_counts().sort_index()

    old_loc = old_down['Locator'].value_counts()
    new_loc = new_down['Locator'].value_counts()
    top_locs = (old_loc.add(new_loc, fill_value=0).sort_values(ascending=False).head(10).index)
    old_top = old_loc.reindex(top_locs, fill_value=0)
    new_top = new_loc.reindex(top_locs, fill_value=0)

    # Pool all shared numeric values across CTD variables for summary panels.
    shared_old_parts = []
    shared_new_parts = []
    for vn in VALUE_COLS:
        v_old = pd.to_numeric(m_values.get(f'{vn}_old', pd.Series(dtype=float)), errors='coerce')
        v_new = pd.to_numeric(m_values.get(f'{vn}_new', pd.Series(dtype=float)), errors='coerce')
        good = v_old.notna() & v_new.notna()
        if good.sum() > 0:
            shared_old_parts.append(v_old[good].to_numpy())
            shared_new_parts.append(v_new[good].to_numpy())

    if len(shared_old_parts) > 0:
        shared_old = np.concatenate(shared_old_parts)
        shared_new = np.concatenate(shared_new_parts)
        shared_delta = shared_new - shared_old
    else:
        shared_old = np.array([])
        shared_new = np.array([])
        shared_delta = np.array([])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(old_year_counts.index, old_year_counts.values, marker='o', label='Old (April2024)')
    ax.plot(new_year_counts.index, new_year_counts.values, marker='o', label='New (20260420)')
    ax.set_title('Downcast Record Counts by Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Row count')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    x = np.arange(len(top_locs))
    w = 0.4
    ax.bar(x - w / 2, old_top.values, width=w, label='Old')
    ax.bar(x + w / 2, new_top.values, width=w, label='New')
    ax.set_title('Top Downcast Locations (Row Counts)')
    ax.set_xticks(x)
    ax.set_xticklabels(top_locs, rotation=45, ha='right')
    ax.set_ylabel('Row count')
    ax.legend()

    ax = axes[1, 0]
    if len(shared_delta) > 0:
        lo = np.nanpercentile(shared_delta, 1)
        hi = np.nanpercentile(shared_delta, 99)
        clipped = np.clip(shared_delta, lo, hi)
        ax.hist(clipped, bins=80)
        ax.axvline(0, color='k', linestyle='--', linewidth=1)
        ax.set_title('Shared Numeric Differences (New - Old), 1st-99th pct clipped')
        ax.set_xlabel('Delta (mixed units)')
        ax.set_ylabel('Count')
    else:
        ax.text(0.5, 0.5, 'No shared paired numeric values', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Shared Numeric Differences (New - Old)')

    ax = axes[1, 1]
    if len(shared_delta) > 0:
        xvals = shared_old
        yvals = shared_new
        if len(xvals) > 20000:
            idx = np.random.default_rng(42).choice(len(xvals), size=20000, replace=False)
            xvals = xvals[idx]
            yvals = yvals[idx]
        ax.scatter(xvals, yvals, s=4, alpha=0.2)
        lims = [np.nanmin([xvals.min(), yvals.min()]), np.nanmax([xvals.max(), yvals.max()])]
        ax.plot(lims, lims, 'k--', linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title('Shared Numeric Values: Old vs New (sampled)')
        ax.set_xlabel('Old values (mixed units)')
        ax.set_ylabel('New values (mixed units)')
    else:
        ax.text(0.5, 0.5, 'No shared paired numeric values', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Shared Numeric Values: Old vs New')

    fig.suptitle('KC Whidbey Basin CTD Comparison: April2024 vs 20260420', fontsize=14)
    fig.savefig(PLOT_FP, dpi=200)
    plt.close(fig)
    print(f'\nSaved comparison figure: {PLOT_FP}')


def main() -> None:
    print('Comparing KC Whidbey Basin CTD files:')
    print(f'  OLD: {OLD_FP}')
    print(f'  NEW: {NEW_FP}')

    old_raw = pd.read_csv(OLD_FP)
    new_raw = pd.read_csv(NEW_FP)

    print('\n[Schema]')
    old_cols = list(old_raw.columns)
    new_cols = list(new_raw.columns)
    print(f'  old columns: {len(old_cols)}')
    print(f'  new columns: {len(new_cols)}')

    old_only_cols = sorted(set(old_cols) - set(new_cols))
    new_only_cols = sorted(set(new_cols) - set(old_cols))
    print(f'  columns only in old: {old_only_cols if old_only_cols else "none"}')
    print(f'  columns only in new: {new_only_cols if new_only_cols else "none"}')

    missing_compare_old = [c for c in COMPARE_COLS if c not in old_raw.columns]
    missing_compare_new = [c for c in COMPARE_COLS if c not in new_raw.columns]
    if missing_compare_old or missing_compare_new:
        raise ValueError(
            f'Missing compare columns. old missing={missing_compare_old}, new missing={missing_compare_new}'
        )

    old_df = _normalize(old_raw[COMPARE_COLS])
    new_df = _normalize(new_raw[COMPARE_COLS])

    print('\n[Date range]')
    _year_range(old_df, '  old all rows')
    _year_range(new_df, '  new all rows')

    old_down = old_df[old_df['Up Down'] == 'Down'].copy()
    new_down = new_df[new_df['Up Down'] == 'Down'].copy()
    _year_range(old_down, '  old downcasts')
    _year_range(new_down, '  new downcasts')

    _overlap_stats(old_df, new_df, COMPARE_COLS, 'All rows (processing-relevant columns)')
    _overlap_stats(old_down, new_down, COMPARE_COLS, 'Downcasts only')

    m_values = _compare_values_on_shared_keys(old_df, new_df)
    _make_summary_figure(old_df, new_df, m_values)


if __name__ == '__main__':
    main()
