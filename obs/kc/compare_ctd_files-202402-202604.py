"""
Compare old and new King County CTD CSV files.

The KC CTD dataset is delivered as one CSV per Locator. This script aggregates
all per-Locator files in the old and new directories and reports:
- Locator/file coverage differences
- Column/schema differences
- Date coverage differences (all rows and Down casts)
- Multiset row overlap using processing-relevant columns
- Value changes for matching measurement columns on shared keys
"""

from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DATA_DIR = Path('/Users/dakotamascarenas/LO_data/obs/kc/ctd')
OLD_DIR = DATA_DIR / 'old'
NEW_DIR = DATA_DIR
PLOT_FP = Path(__file__).parent / 'ctd_comparison.png'

# Identity columns: uniquely identify a CTD sample row.
KEY_COLS = ['Locator', 'Sampledate', 'Depth', 'Updown']

# Measurement columns we want to compare values on (when present in both).
VALUE_COLS = [
    'Chlorophyll, Field (mg/m^3)',
    'Density, Field (Kg/m^3)',
    'Dissolved Oxygen, Field (mg/l ws=2)',
    'Sigma Density, Field (Kg/m^3)',
    'Light Transmission (%)',
    'Light Intensity (PAR), Field (umol/sm2)',
    'Surface Light Intensity (PAR), Field (umol/sm2)',
    'Salinity, Field (PSS)',
    'Sample Temperature, Field (deg C)',
    'Turbidity, Field (NTU)',
    'Nitrite + Nitrate Nitrogen, Field (mg/L)',
]


def _read_dir(d: Path) -> pd.DataFrame:
    fns = sorted(d.glob('ctd_extract_*.csv'))
    frames = []
    for f in fns:
        raw = pd.read_csv(f, encoding='cp1252', low_memory=False)
        if 'ï»¿Locator' in raw.columns:
            raw = raw.rename(columns={'ï»¿Locator': 'Locator'})
        raw['_source_file'] = f.name
        frames.append(raw)
    if not frames:
        raise FileNotFoundError(f'No CTD files in {d}')
    return pd.concat(frames, ignore_index=True)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['Sampledate'] = pd.to_datetime(out['Sampledate'], errors='coerce')
    out['Sampledate'] = out['Sampledate'].dt.strftime('%Y-%m-%d %H:%M:%S')
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].fillna('').astype(str).str.strip()
    return out


def _row_counter(df: pd.DataFrame, cols: list[str]) -> Counter:
    tmp = df[cols].copy()
    for c in cols:
        tmp[c] = tmp[c].astype(object).where(tmp[c].notna(), '__MISSING__').astype(str).str.strip()
    return Counter(map(tuple, tmp.itertuples(index=False, name=None)))


def _overlap_stats(old_df: pd.DataFrame, new_df: pd.DataFrame, cols: list[str], label: str) -> None:
    old_ctr = _row_counter(old_df, cols)
    new_ctr = _row_counter(new_df, cols)
    overlap = sum(min(c, new_ctr.get(k, 0)) for k, c in old_ctr.items())
    print(f'\n[{label}]')
    print(f'  old rows: {len(old_df):,}')
    print(f'  new rows: {len(new_df):,}')
    print(f'  overlapping rows: {overlap:,}')
    print(f'  old-only rows: {len(old_df) - overlap:,}')
    print(f'  new-only rows: {len(new_df) - overlap:,}')


def _year_range(df: pd.DataFrame, label: str) -> None:
    t = pd.to_datetime(df['Sampledate'], errors='coerce')
    years = t.dt.year.dropna().astype(int)
    if len(years) == 0:
        print(f'{label}: no parseable dates')
    else:
        print(f'{label}: {years.min()}-{years.max()}')


def _compare_values_on_shared_keys(old_df: pd.DataFrame, new_df: pd.DataFrame) -> dict:
    cols_present = [c for c in VALUE_COLS if c in old_df.columns and c in new_df.columns]
    old_x = old_df[KEY_COLS + cols_present].copy()
    new_x = new_df[KEY_COLS + cols_present].copy()

    old_x = old_x.sort_values(KEY_COLS).drop_duplicates(KEY_COLS, keep='first')
    new_x = new_x.sort_values(KEY_COLS).drop_duplicates(KEY_COLS, keep='first')

    m = old_x.merge(new_x, on=KEY_COLS, how='inner', suffixes=('_old', '_new'))
    print('\n[Value comparison on shared keys]')
    print(f'  shared keys: {len(m):,}')
    deltas = {}
    if len(m) == 0:
        return deltas
    for c in cols_present:
        v_old = pd.to_numeric(m[c + '_old'], errors='coerce')
        v_new = pd.to_numeric(m[c + '_new'], errors='coerce')
        both = v_old.notna() & v_new.notna()
        n_both = int(both.sum())
        if n_both == 0:
            print(f'  {c}: no shared numeric values')
            continue
        equal = int((v_old[both] == v_new[both]).sum())
        diff = n_both - equal
        d = (v_new[both] - v_old[both])
        deltas[c] = d
        print(
            f'  {c}: shared={n_both:,} equal={equal:,} diff={diff:,} '
            f'mean_delta={d.mean():.4g} max_abs_delta={d.abs().max():.4g}'
        )
    return deltas


def _make_summary_figure(
    old_df: pd.DataFrame, new_df: pd.DataFrame, deltas: dict
) -> None:
    old_t = pd.to_datetime(old_df['Sampledate'], errors='coerce')
    new_t = pd.to_datetime(new_df['Sampledate'], errors='coerce')
    old_year = old_t.dt.year.value_counts().sort_index()
    new_year = new_t.dt.year.value_counts().sort_index()

    old_loc = old_df['Locator'].value_counts()
    new_loc = new_df['Locator'].value_counts()
    locs = sorted(set(old_loc.index) | set(new_loc.index))
    old_loc = old_loc.reindex(locs, fill_value=0)
    new_loc = new_loc.reindex(locs, fill_value=0)

    # Pick a few key variables for value-delta comparison.
    focus_vars = [
        'Salinity, Field (PSS)',
        'Sample Temperature, Field (deg C)',
        'Dissolved Oxygen, Field (mg/l ws=2)',
        'Chlorophyll, Field (mg/m^3)',
    ]
    focus_vars = [v for v in focus_vars if v in deltas and len(deltas[v]) > 0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # Panel 1: yearly record counts.
    ax = axes[0, 0]
    ax.plot(old_year.index, old_year.values, marker='o', label='Old (Feb2024)')
    ax.plot(new_year.index, new_year.values, marker='o', label='New (Apr2026)')
    ax.set_title('CTD Record Counts by Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Row count')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: per-locator row counts.
    ax = axes[0, 1]
    x = np.arange(len(locs))
    width = 0.4
    ax.bar(x - width / 2, old_loc.values, width=width, label='Old')
    ax.bar(x + width / 2, new_loc.values, width=width, label='New')
    ax.set_title('Row Counts by Locator')
    ax.set_xticks(x)
    ax.set_xticklabels(locs, rotation=50, ha='right')
    ax.set_ylabel('Row count')
    ax.legend()

    # Panel 3: distribution of value deltas (new - old) for focus variables.
    ax = axes[1, 0]
    if focus_vars:
        data = []
        labels = []
        for v in focus_vars:
            d = deltas[v].to_numpy()
            d = d[np.isfinite(d)]
            if len(d) == 0:
                continue
            lo = np.nanpercentile(d, 1)
            hi = np.nanpercentile(d, 99)
            data.append(np.clip(d, lo, hi))
            labels.append(v.split(',')[0])
        if data:
            ax.boxplot(data, labels=labels, showfliers=False)
            ax.axhline(0, color='k', linestyle='--', linewidth=1)
            ax.set_title('Value Difference (New - Old), 1st-99th pct clipped')
            ax.set_ylabel('Delta')
            ax.tick_params(axis='x', rotation=20)
        else:
            ax.text(0.5, 0.5, 'No shared numeric deltas', ha='center', va='center',
                    transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 'No shared focus variables', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('Value Difference (New - Old)')

    # Panel 4: scatter old vs new for a single primary variable.
    ax = axes[1, 1]
    primary = next((v for v in focus_vars if v == 'Salinity, Field (PSS)'),
                   focus_vars[0] if focus_vars else None)
    if primary is not None:
        # Need the underlying paired values, not just the delta.
        cols_present = [c for c in VALUE_COLS if c in old_df.columns and c in new_df.columns]
        old_x = old_df[KEY_COLS + [primary]].sort_values(KEY_COLS).drop_duplicates(
            KEY_COLS, keep='first')
        new_x = new_df[KEY_COLS + [primary]].sort_values(KEY_COLS).drop_duplicates(
            KEY_COLS, keep='first')
        mm = old_x.merge(new_x, on=KEY_COLS, how='inner', suffixes=('_old', '_new'))
        v_old = pd.to_numeric(mm[primary + '_old'], errors='coerce')
        v_new = pd.to_numeric(mm[primary + '_new'], errors='coerce')
        both = v_old.notna() & v_new.notna()
        xv = v_old[both].to_numpy()
        yv = v_new[both].to_numpy()
        if len(xv) > 0:
            nmax = 20000
            if len(xv) > nmax:
                idx = np.random.default_rng(42).choice(len(xv), size=nmax, replace=False)
                xv, yv = xv[idx], yv[idx]
            ax.scatter(xv, yv, s=4, alpha=0.2)
            lims = [np.nanmin([xv.min(), yv.min()]), np.nanmax([xv.max(), yv.max()])]
            ax.plot(lims, lims, 'k--', linewidth=1)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_title(f'Shared values: Old vs New\n{primary}')
            ax.set_xlabel('Old')
            ax.set_ylabel('New')
        else:
            ax.text(0.5, 0.5, 'No shared numeric values', ha='center', va='center',
                    transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 'No focus variable available', ha='center', va='center',
                transform=ax.transAxes)

    fig.suptitle('KC CTD Data Comparison: Feb2024 vs Apr2026', fontsize=14)
    fig.savefig(PLOT_FP, dpi=200)
    plt.close(fig)
    print(f'\nSaved comparison figure: {PLOT_FP}')


def main() -> None:
    print('Comparing KC CTD files:')
    print(f'  OLD dir: {OLD_DIR}')
    print(f'  NEW dir: {NEW_DIR}')

    old_files = sorted(p.name for p in OLD_DIR.glob('ctd_extract_*.csv'))
    new_files = sorted(p.name for p in NEW_DIR.glob('ctd_extract_*.csv'))
    print('\n[Files]')
    print(f'  old file count: {len(old_files)}')
    print(f'  new file count: {len(new_files)}')

    def _loc_from_name(n: str) -> str:
        return n.split('_')[2] if len(n.split('_')) > 2 else n

    old_locs = {_loc_from_name(n) for n in old_files}
    new_locs = {_loc_from_name(n) for n in new_files}
    print(f'  locators only in old: {sorted(old_locs - new_locs) or "none"}')
    print(f'  locators only in new: {sorted(new_locs - old_locs) or "none"}')

    old_raw = _read_dir(OLD_DIR)
    new_raw = _read_dir(NEW_DIR)

    print('\n[Schema]')
    old_cols = [c for c in old_raw.columns if c != '_source_file']
    new_cols = [c for c in new_raw.columns if c != '_source_file']
    print(f'  old columns: {len(old_cols)}')
    print(f'  new columns: {len(new_cols)}')
    print(f'  columns only in old: {sorted(set(old_cols) - set(new_cols)) or "none"}')
    print(f'  columns only in new: {sorted(set(new_cols) - set(old_cols)) or "none"}')

    missing_old = [c for c in KEY_COLS if c not in old_raw.columns]
    missing_new = [c for c in KEY_COLS if c not in new_raw.columns]
    if missing_old or missing_new:
        raise ValueError(
            f'Missing key columns. old missing={missing_old}, new missing={missing_new}'
        )

    old_df = _normalize(old_raw)
    new_df = _normalize(new_raw)

    print('\n[Date range]')
    _year_range(old_df, '  old all rows')
    _year_range(new_df, '  new all rows')

    old_down = old_df[old_df['Updown'] == 'Down'].copy()
    new_down = new_df[new_df['Updown'] == 'Down'].copy()
    _year_range(old_down, '  old Down casts')
    _year_range(new_down, '  new Down casts')

    _overlap_stats(old_df, new_df, KEY_COLS, 'All rows (key columns)')
    _overlap_stats(old_down, new_down, KEY_COLS, 'Down casts only')

    deltas = _compare_values_on_shared_keys(old_down, new_down)
    _make_summary_figure(old_down, new_down, deltas)


if __name__ == '__main__':
    main()
