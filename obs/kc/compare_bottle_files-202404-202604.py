"""
Compare old and new King County bottle CSV files.

This script reports:
- Column/schema differences
- Date coverage differences (all rows and Marine Offshore)
- Multiset row overlap using processing-relevant columns
- Value changes for matching measurement keys
"""

from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DATA_DIR = Path('/Users/dakotamascarenas/LO_data/obs/kc/bottle')
OLD_FP = DATA_DIR / 'old' / 'Water_Quality_March2024.csv'
NEW_FP = DATA_DIR / 'Water_Quality_20260421.csv'
PLOT_FP = Path(__file__).parent / 'bottle_comparison.png'

# Columns most relevant to LO processing and robust record identity.
COMPARE_COLS = [
    'Sample ID', 'Profile ID', 'Collect DateTime', 'Depth (m)',
    'Site Type', 'Locator', 'Parameter', 'Value', 'Units',
    'Replicates', 'Replicate Of'
]

# Key columns to compare values on shared measurements.
VALUE_KEY_COLS = [
    'Sample ID', 'Profile ID', 'Collect DateTime', 'Depth (m)',
    'Locator', 'Parameter'
]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Normalize datetime representation so old/new formatting differences do not affect comparisons.
    out['Collect DateTime'] = pd.to_datetime(out['Collect DateTime'], errors='coerce')
    out['Collect DateTime'] = out['Collect DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].fillna('').astype(str).str.strip()

    return out


def _row_counter(df: pd.DataFrame, cols: list[str]) -> Counter:
    # Canonicalize to strings so NaN/None/type formatting do not break equality checks.
    tmp = df[cols].copy()
    for c in cols:
        tmp[c] = tmp[c].astype(object).where(tmp[c].notna(), '__MISSING__').astype(str).str.strip()
    row_tuples = map(tuple, tmp.itertuples(index=False, name=None))
    return Counter(row_tuples)


def _overlap_stats(old_df: pd.DataFrame, new_df: pd.DataFrame, cols: list[str], label: str) -> None:
    old_ctr = _row_counter(old_df, cols)
    new_ctr = _row_counter(new_df, cols)

    overlap = 0
    for k, old_count in old_ctr.items():
        overlap += min(old_count, new_ctr.get(k, 0))

    old_total = len(old_df)
    new_total = len(new_df)
    old_only = old_total - overlap
    new_only = new_total - overlap

    print(f'\n[{label}]')
    print(f'  old rows: {old_total:,}')
    print(f'  new rows: {new_total:,}')
    print(f'  overlapping rows: {overlap:,}')
    print(f'  old-only rows: {old_only:,}')
    print(f'  new-only rows: {new_only:,}')


def _year_range(df: pd.DataFrame, label: str) -> None:
    t = pd.to_datetime(df['Collect DateTime'], errors='coerce')
    years = t.dt.year.dropna().astype(int)
    if len(years) == 0:
        print(f'{label}: no parseable dates')
    else:
        print(f'{label}: {years.min()}-{years.max()}')


def _compare_values_on_shared_keys(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    old_x = old_df[VALUE_KEY_COLS + ['Value']].copy()
    new_x = new_df[VALUE_KEY_COLS + ['Value']].copy()

    # If duplicate keys exist, keep first after sorting for deterministic comparison.
    old_x = old_x.sort_values(VALUE_KEY_COLS).drop_duplicates(VALUE_KEY_COLS, keep='first')
    new_x = new_x.sort_values(VALUE_KEY_COLS).drop_duplicates(VALUE_KEY_COLS, keep='first')

    m = old_x.merge(new_x, on=VALUE_KEY_COLS, how='inner', suffixes=('_old', '_new'))
    if len(m) == 0:
        print('\n[Value comparison on shared keys]')
        print('  no shared measurement keys')
        return m

    # Numeric where possible; string compare fallback for non-numeric values.
    v_old_num = pd.to_numeric(m['Value_old'], errors='coerce')
    v_new_num = pd.to_numeric(m['Value_new'], errors='coerce')

    both_num = v_old_num.notna() & v_new_num.notna()
    num_equal = (v_old_num[both_num] == v_new_num[both_num]).sum()
    num_diff = both_num.sum() - num_equal

    either_non_num = ~both_num
    str_equal = (m.loc[either_non_num, 'Value_old'] == m.loc[either_non_num, 'Value_new']).sum()
    str_diff = either_non_num.sum() - str_equal

    print('\n[Value comparison on shared keys]')
    print(f'  shared keys: {len(m):,}')
    print(f'  numeric matches: {num_equal:,}')
    print(f'  numeric mismatches: {num_diff:,}')
    print(f'  non-numeric matches: {str_equal:,}')
    print(f'  non-numeric mismatches: {str_diff:,}')

    # Show a few mismatches for quick QA.
    mismatch_mask = pd.Series(False, index=m.index)
    mismatch_mask.loc[both_num] = (v_old_num[both_num] != v_new_num[both_num]).to_numpy()
    mismatch_mask.loc[either_non_num] = (
        m.loc[either_non_num, 'Value_old'] != m.loc[either_non_num, 'Value_new']
    ).to_numpy()

    mm = m.loc[mismatch_mask, VALUE_KEY_COLS + ['Value_old', 'Value_new']].head(10)
    if len(mm) > 0:
        print('\n  sample mismatches (first 10):')
        print(mm.to_string(index=False))

    return m


def _make_summary_figure(old_df: pd.DataFrame, new_df: pd.DataFrame, m_values: pd.DataFrame) -> None:
    # Keep visualization focused on the Marine Offshore subset used in processing.
    old_off = old_df[old_df['Site Type'] == 'Marine Offshore'].copy()
    new_off = new_df[new_df['Site Type'] == 'Marine Offshore'].copy()

    old_t = pd.to_datetime(old_off['Collect DateTime'], errors='coerce')
    new_t = pd.to_datetime(new_off['Collect DateTime'], errors='coerce')
    old_year_counts = old_t.dt.year.value_counts().sort_index()
    new_year_counts = new_t.dt.year.value_counts().sort_index()

    old_param = old_off['Parameter'].value_counts()
    new_param = new_off['Parameter'].value_counts()
    top_params = (old_param.add(new_param, fill_value=0).sort_values(ascending=False).head(12).index)
    old_top = old_param.reindex(top_params, fill_value=0)
    new_top = new_param.reindex(top_params, fill_value=0)

    # Numeric value comparison for shared keys.
    v_old_num = pd.to_numeric(m_values.get('Value_old', pd.Series(dtype=float)), errors='coerce')
    v_new_num = pd.to_numeric(m_values.get('Value_new', pd.Series(dtype=float)), errors='coerce')
    both_num = v_old_num.notna() & v_new_num.notna()
    v_old_num = v_old_num[both_num]
    v_new_num = v_new_num[both_num]
    v_delta = v_new_num - v_old_num

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # Panel 1: Marine Offshore yearly record counts.
    ax = axes[0, 0]
    ax.plot(old_year_counts.index, old_year_counts.values, marker='o', label='Old (Mar2024)')
    ax.plot(new_year_counts.index, new_year_counts.values, marker='o', label='New (20260421)')
    ax.set_title('Marine Offshore Record Counts by Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Row count')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: Top parameter counts.
    ax = axes[0, 1]
    x = np.arange(len(top_params))
    width = 0.4
    ax.bar(x - width / 2, old_top.values, width=width, label='Old')
    ax.bar(x + width / 2, new_top.values, width=width, label='New')
    ax.set_title('Top Marine Offshore Parameters (Row Counts)')
    ax.set_xticks(x)
    ax.set_xticklabels(top_params, rotation=50, ha='right')
    ax.set_ylabel('Row count')
    ax.legend()

    # Panel 3: Distribution of numeric value deltas (new-old).
    ax = axes[1, 0]
    if len(v_delta) > 0:
        # Clip extreme outliers for readable histogram while preserving central distribution.
        lo = np.nanpercentile(v_delta, 1)
        hi = np.nanpercentile(v_delta, 99)
        clipped = v_delta.clip(lo, hi)
        ax.hist(clipped, bins=80)
        ax.axvline(0, color='k', linestyle='--', linewidth=1)
        ax.set_title('Numeric Value Difference (New - Old), 1st-99th pct clipped')
        ax.set_xlabel('Delta')
        ax.set_ylabel('Count')
    else:
        ax.text(0.5, 0.5, 'No shared numeric values', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Numeric Value Difference (New - Old)')

    # Panel 4: Numeric shared value comparison (sampled scatter).
    ax = axes[1, 1]
    if len(v_old_num) > 0:
        nmax = 20000
        if len(v_old_num) > nmax:
            idx = np.random.default_rng(42).choice(len(v_old_num), size=nmax, replace=False)
            xvals = v_old_num.to_numpy()[idx]
            yvals = v_new_num.to_numpy()[idx]
        else:
            xvals = v_old_num.to_numpy()
            yvals = v_new_num.to_numpy()
        ax.scatter(xvals, yvals, s=4, alpha=0.2)
        lims = [np.nanmin([xvals.min(), yvals.min()]), np.nanmax([xvals.max(), yvals.max()])]
        ax.plot(lims, lims, 'k--', linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title('Shared Numeric Values: Old vs New (sampled)')
        ax.set_xlabel('Old value')
        ax.set_ylabel('New value')
    else:
        ax.text(0.5, 0.5, 'No shared numeric values', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Shared Numeric Values: Old vs New')

    fig.suptitle('KC Bottle Data Comparison: March2024 vs 20260421', fontsize=14)
    fig.savefig(PLOT_FP, dpi=200)
    plt.close(fig)
    print(f'\nSaved comparison figure: {PLOT_FP}')


def main() -> None:
    print('Comparing KC bottle files:')
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

    old_off = old_df[old_df['Site Type'] == 'Marine Offshore'].copy()
    new_off = new_df[new_df['Site Type'] == 'Marine Offshore'].copy()
    _year_range(old_off, '  old Marine Offshore')
    _year_range(new_off, '  new Marine Offshore')

    _overlap_stats(old_df, new_df, COMPARE_COLS, 'All rows (processing-relevant columns)')
    _overlap_stats(old_off, new_off, COMPARE_COLS, 'Marine Offshore only')

    # Re-load key/value subset from normalized frames for shared-key value comparison.
    old_kv = old_df[VALUE_KEY_COLS + ['Value']].copy()
    new_kv = new_df[VALUE_KEY_COLS + ['Value']].copy()
    m_values = _compare_values_on_shared_keys(old_kv, new_kv)
    _make_summary_figure(old_df, new_df, m_values)


if __name__ == '__main__':
    main()
