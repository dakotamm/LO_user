"""
20260609_hypoxic_days_map.py

Created 2026-06-09 by Dakota Mascarenas.

Map the number of days each water column is hypoxic, for two wb1 model runs:
    wb1_t0_xn11abbur00
    wb1_t1_xn11abbur00

For every grid cell, a day counts as hypoxic if the MINIMUM dissolved oxygen
over the full water column (all s_rho levels) that day is below
HYPOXIA_THRESHOLD_MGL (default 2 mg/L). Counts are accumulated separately per
calendar year and plotted as a grid of lat/lon panels (rows = years, 2024 top
and 2025 bottom; columns = models), zoomed to the Penn Cove extent (the cove
west of the pc0 TEF section) with a shared colorbar (scaled from the Penn Cove
cells) and a coastline (pfun.add_coast). Model-years with no data are left blank.

Designed to run on apogee where lowpassed.nc files live:
    /dat2/dakotamm/LO_roms/<gtagex>/f<YYYY.MM.DD>/lowpassed.nc  (checked first)
    /dat1/parker/LO_roms/<gtagex>/f<YYYY.MM.DD>/lowpassed.nc    (fallback)

Outputs (all under LO_output/DM_outs/20260609_hypoxic_days_map/):
    hypoxic_days_<gtagex>_<year>.nc  (one per model-year: hypoxic_days + lon/lat)
    hypoxic_days_map.png             (rows = years, cols = models)

Usage (on apogee):
    python 20260609_hypoxic_days_map.py
"""

import pickle
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

from scipy.ndimage import label

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS = [
    ('wb1', 't0', 'xn11abbur00'),
    ('wb1', 't1', 'xn11abbur00'),
]

# roms_out keys to search in priority order (roms_out2 = /dat2/dakotamm,
# roms_out1 = /dat1/parker, roms_out = parent/LO_roms)
ROMS_OUT_KEYS = ['roms_out2', 'roms_out1', 'roms_out']

DO_MMOL_TO_MGL = 32.0 / 1000.0  # mmol m-3 (uM) -> mg L-1

HYPOXIA_THRESHOLD_MGL = 2.0     # a column is "hypoxic" if its min DO < this

YEARS = [2024, 2025]            # plot rows (2024 top, 2025 bottom)

OUT_NAME = '20260609_hypoxic_days_map'


# ---------------------------------------------------------------------------
# Find lowpassed.nc files for a model (priority: roms_out2 -> roms_out1 -> ...)
# ---------------------------------------------------------------------------

def find_lowpassed_files(gtagex, Ldir):
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
    return lp_by_date


# ---------------------------------------------------------------------------
# Penn Cove mask (east of the pc0 TEF section) — for zoom + color scaling
# ---------------------------------------------------------------------------

def penn_cove_mask(Ldir, lon, lat, mask_rho,
                   seed_lon=-122.69, seed_lat=48.235):
    """Boolean mask of Penn Cove.

    The pc0 TEF section cuts across the cove; Penn Cove is the enclosed body of
    water WEST of pc0 (the coastline bounds the other three sides). We take wet
    cells west of pc0 and keep the connected component that contains a seed
    point inside the cove — this excludes Saratoga Passage, which lies east of
    pc0 and beyond the cove's land rim to the west/south.
    """
    pc0_fn = Ldir['LOo'] / 'extract' / 'tef2' / 'sections_wb1_pc0' / 'pc0.p'
    if not pc0_fn.exists():
        raise FileNotFoundError(f'pc0 section file not found: {pc0_fn}')
    pc0 = pickle.load(open(pc0_fn, 'rb'))
    sec_lon = float(pc0['x'].mean())

    wet_west = mask_rho & (lon < sec_lon)
    lab, _ = label(wet_west)
    j = np.unravel_index(
        np.argmin((lon - seed_lon) ** 2 + (lat - seed_lat) ** 2), lon.shape)
    seed_label = lab[j]
    if seed_label == 0:
        raise RuntimeError(
            'Penn Cove seed did not land on a wet cell west of pc0.')
    return lab == seed_label


# ---------------------------------------------------------------------------
# Count hypoxic days per water column for one model
# ---------------------------------------------------------------------------

def count_hypoxic_days(gtagex, Ldir, mask_rho):
    """Return {year: (count_2d with NaN over land, n_days)} for years with data."""
    print(f'\n=== {gtagex} ===')
    lp_by_date = find_lowpassed_files(gtagex, Ldir)
    if not lp_by_date:
        print(f'  No lowpassed.nc files found — skipping {gtagex}')
        return {}
    print(f'  Found {len(lp_by_date)} lowpassed.nc files')

    counts = {}   # year -> running count array
    ndays = {}    # year -> running day count
    skipped_years = set()

    for date_str in sorted(lp_by_date):
        year = int(date_str[:4])
        if year not in YEARS:
            skipped_years.add(year)
            continue
        lp_fn = lp_by_date[date_str]
        try:
            with xr.open_dataset(lp_fn) as ds:
                if 'oxygen' not in ds.data_vars:
                    print(f'  {date_str}: no oxygen variable — skip')
                    continue
                # last time step; shape (s_rho, eta_rho, xi_rho)
                oxy = ds['oxygen'].values[-1, :, :, :]

            col = oxy * DO_MMOL_TO_MGL
            # Treat non-positive (land/fill) values as missing, then take the
            # water-column minimum DO at each cell.
            col[col <= 0] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN cols
                col_min = np.nanmin(col, axis=0)  # (eta_rho, xi_rho)

            # NaN < threshold -> False, so land/all-NaN columns never count.
            hypoxic = col_min < HYPOXIA_THRESHOLD_MGL
            if year not in counts:
                counts[year] = np.zeros(mask_rho.shape, dtype=float)
                ndays[year] = 0
            counts[year] += hypoxic
            ndays[year] += 1

        except Exception as e:
            print(f'  {date_str}: ERROR — {e}')

    if skipped_years:
        print(f'  Note: ignored data from {sorted(skipped_years)} '
              f'(not in YEARS={YEARS})')

    out = {}
    for year in sorted(counts):
        c = counts[year]
        # Blank out land so it plots as empty (coastline drawn on top).
        c[~mask_rho] = np.nan
        mx = np.nanmax(c)
        out[year] = (c, ndays[year])
        print(f'  {year}: {ndays[year]} days processed; '
              f'max hypoxic-day count = {int(mx) if np.isfinite(mx) else 0}')

    if not out:
        print(f'  No valid days for {gtagex}')
    return out


# ---------------------------------------------------------------------------
# Plot: three horizontal panels with shared colorbar + coastline
# ---------------------------------------------------------------------------

def make_plot(results, model_list, year_list, plon, plat, aa, out_dir,
              vmax_mask=None):
    """results: dict (gtagex, year) -> (count_2d, n_days).

    Grid: rows = year_list (top -> bottom), cols = model_list.
    aa : [lon0, lon1, lat0, lat1] axis limits (Penn Cove zoom).
    vmax_mask : optional bool array; when given, the shared color scale is set
        from cells inside it (so the zoom isn't washed out by hypoxia elsewhere
        in the domain). Model-years missing from results are drawn blank.
    """
    import matplotlib.pyplot as plt
    try:
        import cmocean
        cmap = cmocean.cm.matter
    except ImportError:
        cmap = plt.get_cmap('YlOrRd')
    cmap = cmap.copy()
    cmap.set_bad('lightgray')  # land / no data

    # Shared color scale, taken over the region of interest (Penn Cove cells)
    # rather than the whole domain so the zoomed panels use their full range.
    def region_max(c):
        vals = c[vmax_mask] if vmax_mask is not None else c
        m = np.nanmax(vals)
        return m if np.isfinite(m) else 0.0

    vmax = max((region_max(c) for (c, _) in results.values()), default=1.0)
    vmax = max(vmax, 1.0)

    nrow, ncol = len(year_list), len(model_list)
    # Size panels to the (zoomed) data aspect so the grid packs tightly.
    lon_span = aa[1] - aa[0]
    lat_span = aa[3] - aa[2]
    dar_aspect = 1.0 / np.cos(np.deg2rad(0.5 * (aa[2] + aa[3])))
    panel_hw = (lat_span * dar_aspect) / lon_span  # display height / width
    panel_w = 5.5
    figsize = (panel_w * ncol + 2.0, panel_w * panel_hw * nrow + 1.8)

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)

    cs = None
    for i, year in enumerate(year_list):
        for j, gtagex in enumerate(model_list):
            ax = axes[i, j]
            pfun.add_coast(ax)
            ax.axis(aa)
            pfun.dar(ax)
            if i == 0:
                ax.set_title(gtagex, fontsize=11)
            if i == nrow - 1:
                ax.set_xlabel('Longitude')
            if j == 0:
                ax.set_ylabel(f'{year}\nLatitude', fontsize=11)

            key = (gtagex, year)
            if key in results:
                count, n_days = results[key]
                cs = ax.pcolormesh(plon, plat, count, cmap=cmap,
                                   vmin=0, vmax=vmax)
                ax.text(0.03, 0.05, f'n = {n_days} days',
                        transform=ax.transAxes, fontsize=9,
                        va='bottom', ha='left',
                        bbox=dict(facecolor='white', alpha=0.6, lw=0))
            else:
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                        ha='center', va='center', color='0.5', fontsize=12)

    cb = fig.colorbar(cs, ax=list(axes.ravel()), shrink=0.9, pad=0.02)
    cb.set_label(f'Days with water-column DO < {HYPOXIA_THRESHOLD_MGL:g} mg/L')

    fig.suptitle('Penn Cove hypoxic days per water column (lowpassed)',
                 fontweight='bold', fontsize=14)

    fig_fn = out_dir / 'hypoxic_days_map.png'
    fig.savefig(fig_fn, dpi=150, bbox_inches='tight')
    print(f'\nSaved figure -> {fig_fn}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    Ldir = Lfun.Lstart(gridname='wb1', tag='t0', ex_name='xn11ab')

    # Headless backend on non-mac (apogee) machines
    if '_mac' not in Ldir['lo_env']:
        import matplotlib as mpl
        mpl.use('Agg')

    print(f'lo_env : {Ldir["lo_env"]}')
    print(f'LOo    : {Ldir["LOo"]}')
    for key in ROMS_OUT_KEYS:
        print(f'{key:12s}: {Ldir.get(key, "N/A")}')

    # Grid: mask + plotting coordinates
    grid_fn = Ldir['grid'] / 'grid.nc'
    with xr.open_dataset(grid_fn) as dsg:
        mask_rho = dsg.mask_rho.values.astype(bool)
        lon = dsg.lon_rho.values
        lat = dsg.lat_rho.values
    plon, plat = pfun.get_plon_plat(lon, lat)

    # Zoom view + color scaling focused on Penn Cove (east of pc0 section)
    pc_mask = penn_cove_mask(Ldir, lon, lat, mask_rho)
    margin = 0.01  # degrees of padding around the cove
    aa = [lon[pc_mask].min() - margin, lon[pc_mask].max() + margin,
          lat[pc_mask].min() - margin, lat[pc_mask].max() + margin]
    print(f'Penn Cove zoom: lon [{aa[0]:.3f}, {aa[1]:.3f}]  '
          f'lat [{aa[2]:.3f}, {aa[3]:.3f}]  ({int(pc_mask.sum())} cells)')

    out_dir = Ldir['LOo'] / 'DM_outs' / OUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'\nOutput directory: {out_dir}')
    print(f'Hypoxia threshold: {HYPOXIA_THRESHOLD_MGL:g} mg/L (water-column minimum)')

    model_list = [f'{g}_{t}_{e}' for (g, t, e) in MODELS]

    results = {}  # (gtagex, year) -> (count, n_days)
    for gtagex in model_list:
        year_dict = count_hypoxic_days(gtagex, Ldir, mask_rho)
        for year, (count, n_days) in year_dict.items():
            # Save each model-year count field as NetCDF for reuse.
            da = xr.DataArray(
                count,
                dims=('eta_rho', 'xi_rho'),
                coords={'lon_rho': (('eta_rho', 'xi_rho'), lon),
                        'lat_rho': (('eta_rho', 'xi_rho'), lat)},
                name='hypoxic_days',
                attrs={'long_name': 'days with water-column minimum DO below threshold',
                       'threshold_mgL': HYPOXIA_THRESHOLD_MGL,
                       'year': year,
                       'n_days': n_days,
                       'units': 'days'},
            )
            nc_fn = out_dir / f'hypoxic_days_{gtagex}_{year}.nc'
            da.to_netcdf(nc_fn)
            print(f'  Saved field -> {nc_fn}')
            results[(gtagex, year)] = (count, n_days)

    if results:
        make_plot(results, model_list, YEARS, plon, plat, aa, out_dir,
                  vmax_mask=pc_mask)
    else:
        print('\nNo model produced data — no plot made.')

    print('\nDone.')


if __name__ == '__main__':
    main()
