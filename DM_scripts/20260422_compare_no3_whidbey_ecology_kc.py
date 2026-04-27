"""
Compare 2024 nitrate (NO3) from:
  - ecology_nc bottles (Whidbey Basin stations only)
  - kc_whidbeyBasin bottles
  - kc_whidbeyBasin ctds

Whidbey Basin is defined here by the bounding box of the kc_whidbeyBasin
stations (extended slightly). Ecology stations falling inside that box are
included.

Outputs PNGs to LO_output/DM/whidbey_no3_compare_20260422/
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from lo_tools import plotting_functions as pfun

OBS = Path('/Users/dakotamascarenas/LO_output/obs')
OUT = Path('/Users/dakotamascarenas/LO_output/DM_outs/whidbey_no3_compare_20260422')
OUT.mkdir(parents=True, exist_ok=True)

YEAR = 2024


def load(src, sub):
    df = pickle.load(open(OBS / src / sub / f'{YEAR}.p', 'rb'))
    df = df[df['NO3 (uM)'].notna()].copy() if 'NO3 (uM)' in df.columns else df.iloc[0:0]
    return df


eco_b = load('ecology_nc', 'bottle')
kc_b = load('kc_whidbeyBasin', 'bottle')
kc_c = load('kc_whidbeyBasin', 'ctd')

# Use the wb1 model grid extent as the spatial filter so every station
# inside the model domain is captured (regardless of source).
grid_fn = '/Users/dakotamascarenas/LO_data/grids/wb1/grid.nc'
gds = xr.open_dataset(grid_fn)
glon = gds['lon_rho'].values
glat = gds['lat_rho'].values
gmask = gds['mask_rho'].values  # 1=ocean, 0=land
plon, plat = pfun.get_plon_plat(glon, glat)
lat_min, lat_max = float(glat.min()), float(glat.max())
lon_min, lon_max = float(glon.min()), float(glon.max())
print(f'wb1 grid bbox: lat [{lat_min:.3f},{lat_max:.3f}] '
      f'lon [{lon_min:.3f},{lon_max:.3f}]')


def in_box(df):
    return df[(df['lat'].between(lat_min, lat_max)) &
              (df['lon'].between(lon_min, lon_max))].copy()


eco_b_wb = in_box(eco_b)
kc_b = in_box(kc_b)
kc_c = in_box(kc_c)

# Exclude stations that fall in the wb1 domain bbox but are not actually
# inside Whidbey Basin (Admiralty Inlet / Main Basin sites).
EXCLUDE = {'PTH005', 'ADM001', 'ADM003'}
eco_b_wb = eco_b_wb[~eco_b_wb['name'].isin(EXCLUDE)].copy()
kc_b = kc_b[~kc_b['name'].isin(EXCLUDE)].copy()
kc_c = kc_c[~kc_c['name'].isin(EXCLUDE)].copy()
print(f'excluded (outside Whidbey Basin): {sorted(EXCLUDE)}')
print(f'ecology bottles in wb1 domain:    {len(eco_b_wb)} rows, '
      f'{eco_b_wb["name"].nunique()} stations: {sorted(eco_b_wb["name"].unique())}')
print(f'kc_whidbeyBasin bottles in domain: {len(kc_b)} rows, '
      f'{kc_b["name"].nunique()} stations: {sorted(kc_b["name"].unique())}')
print(f'kc_whidbeyBasin ctds in domain:    {len(kc_c)} rows, '
      f'{kc_c["name"].nunique()} stations: {sorted(kc_c["name"].unique())}')

# Tag and combine for unified plotting
eco_b_wb['source'] = 'ecology bottle'
kc_b['source'] = 'kc_whidbey bottle'
kc_c['source'] = 'kc_whidbey ctd'
all_df = pd.concat([eco_b_wb, kc_b, kc_c], ignore_index=True, sort=False)
all_df['time'] = pd.to_datetime(all_df['time'], utc=True).dt.tz_convert(None)

colors = {'ecology bottle': 'tab:blue',
          'kc_whidbey bottle': 'tab:orange',
          'kc_whidbey ctd': 'tab:green'}

# Shared marker style: ctd is faint gray background; bottle sources stand out
style = {
    'kc_whidbey ctd':    dict(color='lightgray', s=6,  alpha=0.35,
                              edgecolor='none', zorder=1),
    'ecology bottle':    dict(color=colors['ecology bottle'], s=40, alpha=0.9,
                              edgecolor='k', linewidth=0.5, zorder=4),
    'kc_whidbey bottle': dict(color=colors['kc_whidbey bottle'], s=55,
                              alpha=0.95, edgecolor='k', linewidth=0.6,
                              marker='D', zorder=5),
}
# Map-specific style: ctd stations are gray dots; kc bottle stations are
# drawn as orange rings encircling the ctd dot at the same location.
map_style = {
    'kc_whidbey ctd':    dict(color='gray', s=90, alpha=0.9,
                              edgecolor='k', linewidth=0.6, zorder=3),
    'ecology bottle':    dict(color=colors['ecology bottle'], s=110,
                              alpha=0.95, edgecolor='k', linewidth=0.6,
                              zorder=5),
    'kc_whidbey bottle': dict(facecolor='none',
                              edgecolor=colors['kc_whidbey bottle'],
                              s=260, linewidth=2.8, zorder=6),
}

# --- Plot 1: Station map ---

fig, ax = plt.subplots(figsize=(7, 9))
# wb1 grid: shade ocean cells lightly
ocean = np.where(gmask == 1, 1.0, np.nan)
ax.pcolormesh(plon, plat, ocean, cmap='Blues', vmin=0, vmax=2,
              alpha=0.25, shading='flat')
# wb1 grid outline
ax.plot([plon[0, 0], plon[0, -1], plon[-1, -1], plon[-1, 0], plon[0, 0]],
        [plat[0, 0], plat[0, -1], plat[-1, -1], plat[-1, 0], plat[0, 0]],
        '-', color='k', lw=1, label='wb1 grid')
# coastline
pfun.add_coast(ax)

for src in ['kc_whidbey ctd', 'ecology bottle', 'kc_whidbey bottle']:
    grp = all_df[all_df['source'] == src]
    if len(grp) == 0:
        continue
    sta = grp.groupby('name')[['lon', 'lat']].mean()
    ax.scatter(sta['lon'], sta['lat'], label=src, **map_style[src])
    for nm, row in sta.iterrows():
        ax.text(row['lon'] + 0.005, row['lat'], nm, fontsize=7, zorder=8)
# Set extent to encompass both stations and grid
ax.set_xlim(min(lon_min, plon.min()) - 0.02, max(lon_max, plon.max()) + 0.02)
ax.set_ylim(min(lat_min, plat.min()) - 0.02, max(lat_max, plat.max()) + 0.02)
pfun.dar(ax)
ax.set_xlabel('lon'); ax.set_ylabel('lat')
ax.set_title(f'Whidbey Basin NO3 stations, {YEAR}')
ax.legend(loc='lower left', fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'map_stations.png', dpi=150)
plt.close(fig)

# --- Plot 2: NO3 vs depth (all data, by source) ---
fig, ax = plt.subplots(figsize=(7, 7))
for src in ['kc_whidbey ctd', 'ecology bottle', 'kc_whidbey bottle']:
    grp = all_df[all_df['source'] == src]
    ax.scatter(grp['NO3 (uM)'], grp['z'], label=f'{src} (n={len(grp)})',
               **style[src])
ax.set_xlabel('NO3 (uM)'); ax.set_ylabel('z (m)')
ax.set_title(f'NO3 vs depth, Whidbey Basin {YEAR}')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'no3_vs_depth.png', dpi=150)
plt.close(fig)

# --- Plot 3: surface (z > -5 m) NO3 time series ---
surf = all_df[all_df['z'] > -5].copy()
fig, ax = plt.subplots(figsize=(11, 5))
for src in ['kc_whidbey ctd', 'ecology bottle', 'kc_whidbey bottle']:
    grp = surf[surf['source'] == src]
    ax.scatter(grp['time'], grp['NO3 (uM)'], label=f'{src} (n={len(grp)})',
               **style[src])
ax.set_xlabel('time'); ax.set_ylabel('NO3 (uM)')
ax.set_title(f'Surface (z > -5 m) NO3, Whidbey Basin {YEAR}')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'no3_surface_timeseries.png', dpi=150)
plt.close(fig)

# --- Plot 4: profile-by-station for the 4 kc stations (where all 3 sources may exist) ---
kc_stations = sorted(kc_b['name'].unique())
fig, axs = plt.subplots(1, len(kc_stations), figsize=(4 * len(kc_stations), 6),
                        sharey=True)
if len(kc_stations) == 1:
    axs = [axs]
for ax, sta in zip(axs, kc_stations):
    sta_df = all_df[all_df['name'] == sta]
    for src in ['kc_whidbey ctd', 'ecology bottle', 'kc_whidbey bottle']:
        grp = sta_df[sta_df['source'] == src]
        if len(grp) == 0:
            continue
        ax.scatter(grp['NO3 (uM)'], grp['z'], label=f'{src} (n={len(grp)})',
                   **style[src])
    ax.set_title(sta); ax.set_xlabel('NO3 (uM)')
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
axs[0].set_ylabel('z (m)')
fig.suptitle(f'NO3 profiles at kc_whidbeyBasin bottle stations, {YEAR}')
fig.tight_layout()
fig.savefig(OUT / 'no3_profiles_kc_stations.png', dpi=150)
plt.close(fig)

print(f'\nWrote plots to {OUT}')
for p in sorted(OUT.glob('*.png')):
    print(' ', p.name)
