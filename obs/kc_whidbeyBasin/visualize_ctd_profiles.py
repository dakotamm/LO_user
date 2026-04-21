"""
Visualize CTD depth profiles to check data quality.

Shows temperature, salinity, dissolved oxygen, and chlorophyll profiles
colored by cast ID to compare different casts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from lo_tools import Lfun
Ldir = Lfun.Lstart()

# Load raw data
source = 'kc_whidbeyBasin'
otype = 'ctd'
in_dir0 = Ldir['data'] / 'obs' / source / otype

big_df_raw = pd.read_csv(in_dir0 / 'Whidbey_Basin_CTD_Casts_20260420.csv')
sta_df = pd.read_csv(in_dir0 / 'WLRD_Sites_March2024.csv')

# Merge station data
big_df = big_df_raw.merge(sta_df[['Locator','Latitude', 'Longitude']], on='Locator', how='left')

# Use only downcasts
big_df_use = big_df[big_df['Up Down'] == 'Down'].copy()

# Parse time
big_df_use['time'] = pd.to_datetime(big_df_use['Sample Date'], infer_datetime_format=True)

# Create cast IDs
big_df_use['unique_date_location'] = (big_df_use['Locator'] + 
                                      big_df_use['time'].dt.year.astype(str) + 
                                      big_df_use['time'].dt.month.astype(str) + 
                                      big_df_use['time'].dt.day.astype(str))
unique_casts = big_df_use['unique_date_location'].unique()
cast_id_map = {cast: i for i, cast in enumerate(unique_casts)}
big_df_use['cid'] = big_df_use['unique_date_location'].map(cast_id_map)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('CTD Depth Profiles - Whidbey Basin', fontsize=14, fontweight='bold')

# Color by cast ID
cmap = plt.cm.tab20
n_casts = len(unique_casts)

# Temperature vs Depth
ax = axes[0, 0]
for cid in big_df_use['cid'].unique():
    mask = big_df_use['cid'] == cid
    data = big_df_use[mask].dropna(subset=['Depth (meters)', 'Temperature (°C)'])
    if len(data) > 0:
        color = cmap(cid % len(cmap.colors))
        ax.plot(data['Temperature (°C)'], data['Depth (meters)'], 
                marker='o', markersize=3, label=f'Cast {int(cid)}', color=color, alpha=0.7)
ax.set_xlabel('Temperature (°C)', fontsize=10)
ax.set_ylabel('Depth (meters)', fontsize=10)
ax.invert_yaxis()
ax.grid(True, alpha=0.3)
ax.set_title('Temperature Profile')

# Salinity vs Depth
ax = axes[0, 1]
for cid in big_df_use['cid'].unique():
    mask = big_df_use['cid'] == cid
    data = big_df_use[mask].dropna(subset=['Depth (meters)', 'Salinity (PSU)'])
    if len(data) > 0:
        color = cmap(cid % len(cmap.colors))
        ax.plot(data['Salinity (PSU)'], data['Depth (meters)'], 
                marker='o', markersize=3, label=f'Cast {int(cid)}', color=color, alpha=0.7)
ax.set_xlabel('Salinity (PSU)', fontsize=10)
ax.set_ylabel('Depth (meters)', fontsize=10)
ax.invert_yaxis()
ax.grid(True, alpha=0.3)
ax.set_title('Salinity Profile')

# Dissolved Oxygen vs Depth
ax = axes[1, 0]
for cid in big_df_use['cid'].unique():
    mask = big_df_use['cid'] == cid
    data = big_df_use[mask].dropna(subset=['Depth (meters)', 'Dissolved Oxygen (mg/L)'])
    if len(data) > 0:
        color = cmap(cid % len(cmap.colors))
        ax.plot(data['Dissolved Oxygen (mg/L)'], data['Depth (meters)'], 
                marker='o', markersize=3, label=f'Cast {int(cid)}', color=color, alpha=0.7)
ax.set_xlabel('Dissolved Oxygen (mg/L)', fontsize=10)
ax.set_ylabel('Depth (meters)', fontsize=10)
ax.invert_yaxis()
ax.grid(True, alpha=0.3)
ax.set_title('Dissolved Oxygen Profile')

# Chlorophyll vs Depth
ax = axes[1, 1]
for cid in big_df_use['cid'].unique():
    mask = big_df_use['cid'] == cid
    data = big_df_use[mask].dropna(subset=['Depth (meters)', 'Chlorophyll (µg/L)'])
    if len(data) > 0:
        color = cmap(cid % len(cmap.colors))
        ax.plot(data['Chlorophyll (µg/L)'], data['Depth (meters)'], 
                marker='o', markersize=3, label=f'Cast {int(cid)}', color=color, alpha=0.7)
ax.set_xlabel('Chlorophyll (µg/L)', fontsize=10)
ax.set_ylabel('Depth (meters)', fontsize=10)
ax.invert_yaxis()
ax.grid(True, alpha=0.3)
ax.set_title('Chlorophyll Profile')

plt.tight_layout()
plt.savefig('ctd_profiles.png', dpi=150, bbox_inches='tight')
print(f"Saved profile plot to ctd_profiles.png")
plt.show()
