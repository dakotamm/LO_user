"""
Compare KC bottle dataset against kc_whidbeyBasin bottle dataset.

Goal:
- Check whether Whidbey records are represented in the full KC bottle data.

Outputs:
- Console summary (schema, date range, overlap stats)
- Figure saved in this folder
"""

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


KC_FP = Path('/Users/dakotamascarenas/LO_data/obs/kc/bottle/Water_Quality_20260421.csv')
WB_FP = Path('/Users/dakotamascarenas/LO_data/obs/kc_whidbeyBasin/bottle/Whidbey_Bottle_Data_20260421.csv')
PLOT_FP = Path(__file__).parent / 'bottle_kc_vs_whidbey_comparison.png'

COMPARE_COLS = [
    'Sample ID', 'Profile ID', 'Collect DateTime', 'Depth (m)',
    'Site Type', 'Locator', 'Parameter', 'Value', 'Units',
    'Replicates', 'Replicate Of'
]

VALUE_KEY_COLS = [
    'Sample ID', 'Profile ID', 'Collect DateTime', 'Depth (m)',
    'Locator', 'Parameter'
]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['Collect DateTime'] = pd.to_datetime(out['Collect DateTime'], errors='coerce', format='mixed')
    out['Collect DateTime'] = out['Collect DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    for c in ['Depth (m)', 'Value']:
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


def _subset_stats(kc_df: pd.DataFrame, wb_df: pd.DataFrame, cols: list[str], label: str) -> tuple[int, int, int]:
    kc_ctr = _row_counter(kc_df, cols)
    wb_ctr = _row_counter(wb_df, cols)

    overlap = 0
    for k, wb_count in wb_ctr.items():
        overlap += min(wb_count, kc_ctr.get(k, 0))

    wb_total = len(wb_df)
    kc_total = len(kc_df)
    wb_only = wb_total - overlap

    print(f'\n[{label}]')
    print(f'  KC rows: {kc_total:,}')
    print(f'  Whidbey rows: {wb_total:,}')
    print(f'  Whidbey rows found in KC: {overlap:,}')
    print(f'  Whidbey rows not found in KC: {wb_only:,}')
    if wb_total > 0:
        print(f'  Whidbey coverage in KC: {100 * overlap / wb_total:.2f}%')

    return kc_total, wb_total, overlap


def _key_coverage_stats(kc_df: pd.DataFrame, wb_df: pd.DataFrame) -> None:
    kc_ctr = _row_counter(kc_df, VALUE_KEY_COLS)
    wb_ctr = _row_counter(wb_df, VALUE_KEY_COLS)

    overlap = 0
    for k, wb_count in wb_ctr.items():
        overlap += min(wb_count, kc_ctr.get(k, 0))

    wb_total = len(wb_df)
    print('\n[Measurement-key coverage]')
    print(f'  Whidbey keys found in KC: {overlap:,}')
    print(f'  Whidbey keys not found in KC: {wb_total - overlap:,}')
    if wb_total > 0:
        print(f'  Whidbey key coverage in KC: {100 * overlap / wb_total:.2f}%')


def _year_range(df: pd.DataFrame, label: str) -> None:
    t = pd.to_datetime(df['Collect DateTime'], errors='coerce')
    years = t.dt.year.dropna().astype(int)
    if len(years) == 0:
        print(f'{label}: no parseable dates')
    else:
        print(f'{label}: {years.min()}-{years.max()}')


def _value_comparison(kc_df: pd.DataFrame, wb_df: pd.DataFrame) -> pd.DataFrame:
    kc_x = kc_df[VALUE_KEY_COLS + ['Value']].copy()
    wb_x = wb_df[VALUE_KEY_COLS + ['Value']].copy()

    kc_x = kc_x.sort_values(VALUE_KEY_COLS).drop_duplicates(VALUE_KEY_COLS, keep='first')
    wb_x = wb_x.sort_values(VALUE_KEY_COLS).drop_duplicates(VALUE_KEY_COLS, keep='first')

    m = wb_x.merge(kc_x, on=VALUE_KEY_COLS, how='inner', suffixes=('_wb', '_kc'))
    print('\n[Value comparison on shared keys]')
    print(f'  shared keys: {len(m):,}')

    if len(m) == 0:
        return m

    wb_num = pd.to_numeric(m['Value_wb'], errors='coerce')
    kc_num = pd.to_numeric(m['Value_kc'], errors='coerce')

    both_num = wb_num.notna() & kc_num.notna()
    num_equal = np.isclose(wb_num[both_num], kc_num[both_num], rtol=1e-6, atol=1e-10).sum()
    num_diff = both_num.sum() - num_equal

    either_non_num = ~both_num
    str_equal = (m.loc[either_non_num, 'Value_wb'].astype(str) == m.loc[either_non_num, 'Value_kc'].astype(str)).sum()
    str_diff = either_non_num.sum() - str_equal

    print(f'  numeric matches: {num_equal:,}')
    print(f'  numeric mismatches: {num_diff:,}')
    print(f'  non-numeric matches: {str_equal:,}')
    print(f'  non-numeric mismatches: {str_diff:,}')

    return m


def _make_figure(kc_df: pd.DataFrame, wb_df: pd.DataFrame, m_values: pd.DataFrame) -> None:
    kc_off = kc_df[kc_df['Site Type'] == 'Marine Offshore'].copy()
    wb_off = wb_df[wb_df['Site Type'] == 'Marine Offshore'].copy()

    kc_t = pd.to_datetime(kc_off['Collect DateTime'], errors='coerce')
    wb_t = pd.to_datetime(wb_off['Collect DateTime'], errors='coerce')
    kc_year = kc_t.dt.year.value_counts().sort_index()
    wb_year = wb_t.dt.year.value_counts().sort_index()

    kc_param = kc_off['Parameter'].value_counts()
    wb_param = wb_off['Parameter'].value_counts()
    top_params = kc_param.add(wb_param, fill_value=0).sort_values(ascending=False).head(12).index
    kc_top = kc_param.reindex(top_params, fill_value=0)
    wb_top = wb_param.reindex(top_params, fill_value=0)

    wb_loc = wb_off['Locator'].value_counts()
    kc_loc = kc_off['Locator'].value_counts()
    top_locs = wb_loc.head(12).index
    wb_loc_top = wb_loc.reindex(top_locs, fill_value=0)
    kc_loc_top = kc_loc.reindex(top_locs, fill_value=0)

    wb_num = pd.to_numeric(m_values.get('Value_wb', pd.Series(dtype=float)), errors='coerce')
    kc_num = pd.to_numeric(m_values.get('Value_kc', pd.Series(dtype=float)), errors='coerce')
    good = wb_num.notna() & kc_num.notna()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(kc_year.index, kc_year.values, marker='o', label='KC Marine Offshore')
    ax.plot(wb_year.index, wb_year.values, marker='o', label='Whidbey Marine Offshore')
    ax.set_title('Record Counts by Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Rows')
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    x = np.arange(len(top_params))
    w = 0.4
    ax.bar(x - w / 2, kc_top.values, width=w, label='KC')
    ax.bar(x + w / 2, wb_top.values, width=w, label='Whidbey')
    ax.set_title('Top Parameters (Marine Offshore)')
    ax.set_xticks(x)
    ax.set_xticklabels(top_params, rotation=50, ha='right')
    ax.set_ylabel('Rows')
    ax.legend()

    ax = axes[1, 0]
    x = np.arange(len(top_locs))
    ax.bar(x - w / 2, kc_loc_top.values, width=w, label='KC')
    ax.bar(x + w / 2, wb_loc_top.values, width=w, label='Whidbey')
    ax.set_title('Whidbey Locators: KC vs Whidbey Counts')
    ax.set_xticks(x)
    ax.set_xticklabels(top_locs, rotation=50, ha='right')
    ax.set_ylabel('Rows')
    ax.legend()

    ax = axes[1, 1]
    if good.sum() > 0:
        xvals = wb_num[good].to_numpy()
        yvals = kc_num[good].to_numpy()
        if len(xvals) > 20000:
            idx = np.random.default_rng(42).choice(len(xvals), size=20000, replace=False)
            xvals = xvals[idx]
            yvals = yvals[idx]
        ax.scatter(xvals, yvals, s=4, alpha=0.2)
        lims = [np.nanmin([xvals.min(), yvals.min()]), np.nanmax([xvals.max(), yvals.max()])]
        ax.plot(lims, lims, 'k--', linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title('Shared Numeric Values: Whidbey vs KC')
        ax.set_xlabel('Whidbey value')
        ax.set_ylabel('KC value')
    else:
        ax.text(0.5, 0.5, 'No shared numeric values', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Shared Numeric Values: Whidbey vs KC')

    fig.suptitle('KC vs kc_whidbeyBasin Bottle Comparison (20260421)')
    fig.savefig(PLOT_FP, dpi=200)
    plt.close(fig)
    print(f'\nSaved comparison figure: {PLOT_FP}')


def main() -> None:
    print('Comparing bottle files:')
    print(f'  KC: {KC_FP}')
    print(f'  Whidbey: {WB_FP}')

    kc_raw = pd.read_csv(KC_FP, low_memory=False)
    wb_raw = pd.read_csv(WB_FP, low_memory=False)

    print('\n[Schema]')
    print(f'  KC columns: {len(kc_raw.columns)}')
    print(f'  Whidbey columns: {len(wb_raw.columns)}')
    kc_only = sorted(set(kc_raw.columns) - set(wb_raw.columns))
    wb_only = sorted(set(wb_raw.columns) - set(kc_raw.columns))
    print(f'  columns only in KC: {kc_only if kc_only else "none"}')
    print(f'  columns only in Whidbey: {wb_only if wb_only else "none"}')

    missing_kc = [c for c in COMPARE_COLS if c not in kc_raw.columns]
    missing_wb = [c for c in COMPARE_COLS if c not in wb_raw.columns]
    if missing_kc or missing_wb:
        raise ValueError(f'Missing compare columns. KC missing={missing_kc}, Whidbey missing={missing_wb}')

    kc_df = _normalize(kc_raw[COMPARE_COLS])
    wb_df = _normalize(wb_raw[COMPARE_COLS])

    print('\n[Date range]')
    _year_range(kc_df, '  KC all rows')
    _year_range(wb_df, '  Whidbey all rows')

    kc_off = kc_df[kc_df['Site Type'] == 'Marine Offshore'].copy()
    wb_off = wb_df[wb_df['Site Type'] == 'Marine Offshore'].copy()
    _year_range(kc_off, '  KC Marine Offshore')
    _year_range(wb_off, '  Whidbey Marine Offshore')

    _subset_stats(kc_df, wb_df, COMPARE_COLS, 'All rows')
    _subset_stats(kc_off, wb_off, COMPARE_COLS, 'Marine Offshore only')
    _key_coverage_stats(kc_df, wb_df)

    m_values = _value_comparison(kc_df, wb_df)
    _make_figure(kc_df, wb_df, m_values)


if __name__ == '__main__':
    main()
