"""
20260609_penn_cove_bottom_DO_timeseries.py

Created 2026-06-09 by Dakota Mascarenas.

Extract daily minimum and spatially-averaged bottom dissolved oxygen within
Penn Cove from lowpassed.nc files for three wb1 model runs:
    wb1_t0_xn11ab
    wb1_t0_xn11abbur00
    wb1_t1_xn11abbur00

Penn Cove is defined as the connected body of wet cells WEST of the pc0 TEF
section (LO_output/extract/tef2/sections_wb1_pc0/pc0.p): pc0 cuts across the
cove and the coastline bounds the other three sides.

Designed to run on apogee where lowpassed.nc files live:
    /dat2/dakotamm/LO_roms/<gtagex>/f<YYYY.MM.DD>/lowpassed.nc  (checked first)
    /dat1/parker/LO_roms/<gtagex>/f<YYYY.MM.DD>/lowpassed.nc    (fallback)

Outputs (all under LO_output/DM_outs/20260609_penn_cove_bottom_DO/):
    bottom_DO_timeseries_<gtagex>.csv   (one per model: date, do_min_mgL, do_mean_mgL)
    penn_cove_bottom_DO_timeseries.png  (mean + min stacked, all 3 models)

Usage (on apogee):
    python 20260609_penn_cove_bottom_DO_timeseries.py
"""

import pickle
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

from scipy.ndimage import label

from lo_tools import Lfun

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS = [
    ('wb1', 't0', 'xn11ab'),
    ('wb1', 't0', 'xn11abbur00'),
    ('wb1', 't1', 'xn11abbur00'),
]

# roms_out keys to search in priority order (roms_out2 = /dat2/dakotamm,
# roms_out1 = /dat1/parker, roms_out = parent/LO_roms)
ROMS_OUT_KEYS = ['roms_out2', 'roms_out1', 'roms_out']

DO_MMOL_TO_MGL = 32.0 / 1000.0  # mmol m-3 (uM) -> mg L-1

OUT_NAME = '20260609_penn_cove_bottom_DO'

# Plot colors per model
MODEL_COLORS = {
    'wb1_t0_xn11ab':       'tab:blue',
    'wb1_t0_xn11abbur00':  'tab:orange',
    'wb1_t1_xn11abbur00':  'tab:green',
}


# ---------------------------------------------------------------------------
# Penn Cove mask
# ---------------------------------------------------------------------------

def build_penn_cove_mask(Ldir, seed_lon=-122.69, seed_lat=48.235):
    """
    Return a boolean 2D array (eta_rho, xi_rho) marking the wet grid cells of
    Penn Cove.

    The pc0 TEF section cuts across the cove; Penn Cove is the enclosed body of
    water WEST of pc0 (the coastline bounds the other three sides). We take wet
    cells west of pc0 and keep the connected component that contains a seed
    point inside the cove — this excludes Saratoga Passage.
    """
    pc0_fn = Ldir['LOo'] / 'extract' / 'tef2' / 'sections_wb1_pc0' / 'pc0.p'
    if not pc0_fn.exists():
        raise FileNotFoundError(
            f'pc0 section file not found: {pc0_fn}\n'
            'Make sure LO_output/extract/tef2/sections_wb1_pc0/pc0.p exists on this machine.'
        )

    pc0 = pickle.load(open(pc0_fn, 'rb'))
    sec_lon = float(pc0['x'].mean())

    grid_fn = Ldir['grid'] / 'grid.nc'
    with xr.open_dataset(grid_fn) as dsg:
        lon = dsg.lon_rho.values
        lat = dsg.lat_rho.values
        mask_rho = dsg.mask_rho.values.astype(bool)

    # Penn Cove = connected wet component west of pc0 containing the cove seed.
    wet_west = mask_rho & (lon < sec_lon)
    lab, _ = label(wet_west)
    j = np.unravel_index(
        np.argmin((lon - seed_lon) ** 2 + (lat - seed_lat) ** 2), lon.shape)
    seed_label = lab[j]
    if seed_label == 0:
        raise RuntimeError(
            'Penn Cove seed did not land on a wet cell west of pc0. '
            'Check seed_lon/seed_lat and that pc0.p matches the wb1 grid.'
        )
    pc_mask = (lab == seed_label)

    n = int(pc_mask.sum())
    print(f'Penn Cove mask: {n} wet cells  '
          f'lon [{lon[pc_mask].min():.3f}, {lon[pc_mask].max():.3f}]  '
          f'lat [{lat[pc_mask].min():.3f}, {lat[pc_mask].max():.3f}]')
    return pc_mask


# ---------------------------------------------------------------------------
# Process one model
# ---------------------------------------------------------------------------

def process_model(gridname, tag, ex_name, Ldir, pc_mask):
    """Return a DataFrame (date, do_min_mgL, do_mean_mgL) or None."""
    gtagex = f'{gridname}_{tag}_{ex_name}'
    print(f'\n=== {gtagex} ===')

    # Collect all available lowpassed.nc paths, preferring roms_out2 over
    # roms_out1 when both contain the same date.
    lp_by_date = {}
    for key in ROMS_OUT_KEYS:
        base_path = Ldir.get(key)
        if base_path is None or str(base_path).endswith('BLANK'):
            continue
        base = Path(base_path) / gtagex
        if not base.exists():
            continue
        for run_dir in sorted(base.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith('f'):
                continue
            lp = run_dir / 'lowpassed.nc'
            if lp.exists():
                date_str = run_dir.name[1:]  # strip leading 'f'
                if date_str not in lp_by_date:   # first hit wins (priority order)
                    lp_by_date[date_str] = lp

    if not lp_by_date:
        print(f'  No lowpassed.nc files found — skipping {gtagex}')
        return None

    print(f'  Found {len(lp_by_date)} lowpassed.nc files')

    records = []
    for date_str in sorted(lp_by_date):
        lp_fn = lp_by_date[date_str]
        try:
            with xr.open_dataset(lp_fn) as ds:
                if 'oxygen' not in ds.data_vars:
                    print(f'  {date_str}: no oxygen variable — skip')
                    continue
                # Take the last time step (lowpassed.nc typically has 1)
                t = pd.Timestamp(ds['ocean_time'].values[-1])
                # s_rho=0 is the bottom layer; shape (s_rho, eta_rho, xi_rho)
                oxy_bot = ds['oxygen'].values[-1, 0, :, :]  # bottom, mmol/m3

            bot_do_mgl = oxy_bot * DO_MMOL_TO_MGL
            pc_do = bot_do_mgl[pc_mask]
            # Drop land-fill NaN, exact zeros (ROMS land-fill artifact),
            # and unphysical negatives
            pc_do = pc_do[np.isfinite(pc_do) & (pc_do > 0)]

            if pc_do.size == 0:
                print(f'  {date_str}: no valid Penn Cove DO values — skip')
                continue

            records.append({
                'date': t.date(),
                'do_min_mgL': float(np.min(pc_do)),
                'do_mean_mgL': float(np.mean(pc_do)),
            })

        except Exception as e:
            print(f'  {date_str}: ERROR — {e}')

    if not records:
        print(f'  No valid records for {gtagex}')
        return None

    df = pd.DataFrame(records).sort_values('date').reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])

    print(f'  {len(df)} daily records  '
          f'min [{df["do_min_mgL"].min():.2f}, {df["do_min_mgL"].max():.2f}]  '
          f'mean [{df["do_mean_mgL"].min():.2f}, {df["do_mean_mgL"].max():.2f}] mg/L')
    return df


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(results, out_dir):
    """results: dict gtagex -> DataFrame. Stacked mean (top) + min (bottom)."""
    import matplotlib.pyplot as plt

    fig, (ax_mean, ax_min) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    for gtagex, df in results.items():
        c = MODEL_COLORS.get(gtagex, None)
        ax_mean.plot(df['date'], df['do_mean_mgL'], '-', color=c,
                     lw=1.5, label=gtagex)
        ax_min.plot(df['date'], df['do_min_mgL'], '-', color=c,
                    lw=1.5, label=gtagex)

    for ax in (ax_mean, ax_min):
        ax.axhline(2.0, color='gray', ls=':', lw=1, label='hypoxic (2 mg/L)')
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('DO (mg/L)')

    ax_mean.set_title('Penn Cove average bottom DO', fontweight='bold')
    ax_min.set_title('Penn Cove minimum bottom DO', fontweight='bold')
    ax_min.set_xlabel('Date')
    ax_mean.legend(loc='best', fontsize=9)

    fig.suptitle('Penn Cove bottom DO time series (lowpassed)',
                 fontweight='bold', fontsize=14, y=1.0)
    fig.tight_layout()

    fig_fn = out_dir / 'penn_cove_bottom_DO_timeseries.png'
    fig.savefig(fig_fn, dpi=150, bbox_inches='tight')
    print(f'\nSaved figure -> {fig_fn}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Lstart with any wb1 model — we only need grid and LOo paths
    Ldir = Lfun.Lstart(gridname='wb1', tag='t0', ex_name='xn11ab')

    # Headless backend on non-mac (apogee) machines
    if '_mac' not in Ldir['lo_env']:
        import matplotlib as mpl
        mpl.use('Agg')

    print(f'lo_env : {Ldir["lo_env"]}')
    print(f'LOo    : {Ldir["LOo"]}')
    for key in ROMS_OUT_KEYS:
        print(f'{key:12s}: {Ldir.get(key, "N/A")}')

    pc_mask = build_penn_cove_mask(Ldir)

    out_dir = Ldir['LOo'] / 'DM_outs' / OUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'\nOutput directory: {out_dir}')

    results = {}
    for gridname, tag, ex_name in MODELS:
        df = process_model(gridname, tag, ex_name, Ldir, pc_mask)
        if df is None:
            continue
        gtagex = f'{gridname}_{tag}_{ex_name}'
        csv_fn = out_dir / f'bottom_DO_timeseries_{gtagex}.csv'
        df.to_csv(csv_fn, index=False)
        print(f'  Saved CSV -> {csv_fn}')
        results[gtagex] = df

    if results:
        make_plot(results, out_dir)
    else:
        print('\nNo model produced data — no plot made.')

    print('\nDone.')


if __name__ == '__main__':
    main()
