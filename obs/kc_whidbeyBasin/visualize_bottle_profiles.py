"""
Visualize bottle data depth profiles to check data quality.

Shows discrete sample measurements of temperature, salinity, dissolved oxygen,
and nutrients with depth, colored by cast to compare different sampling events.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from lo_tools import Lfun
Ldir = Lfun.Lstart()

# Load raw data
source = 'kc_whidbeyBasin'
otype = 'bottle'
in_dir0 = Ldir['data'] / 'obs' / source / otype

big_df_raw = pd.read_csv(in_dir0 / 'Whidbey_Bottle_Data_20260421.csv')
sta_df = pd.read_csv(Ldir['data'] / 'obs' / source / 'WLRD_Sites_20260421.csv')

# Merge station data
big_df = big_df_raw.merge(sta_df[['Locator','Latitude', 'Longitude']], on='Locator', how='left')

# Keep only marine offshore data
big_df_use = big_df[big_df['Site Type'] == 'Marine Offshore'].copy()

# Filter for key parameters
key_params = ['Temperature', 'Salinity', 'Dissolved Oxygen', 
              'Nitrite + Nitrate Nitrogen', 'Ammonia Nitrogen', 'Chlorophyll a']
big_df_use = big_df_use[big_df_use['Parameter'].isin(key_params)].copy()

# Convert Value to numeric
big_df_use['Value'] = pd.to_numeric(big_df_use['Value'], errors='coerce')

# Parse time and pivot to wide format
big_df_use['Collect DateTime'] = pd.to_datetime(big_df_use['Collect DateTime'])
big_df_pivot = big_df_use.pivot_table(index=['Profile ID', 'Collect DateTime', 'Depth (m)', 
                                             'Locator', 'Latitude', 'Longitude'],
                                      columns='Parameter', values='Value').reset_index()

# Create cast IDs
big_df_pivot['time'] = pd.to_datetime(big_df_pivot['Collect DateTime'])
big_df_pivot['unique_date_location'] = (big_df_pivot['Locator'] + 
                                        big_df_pivot['time'].dt.year.astype(str) + 
                                        big_df_pivot['time'].dt.month.astype(str) + 
                                        big_df_pivot['time'].dt.day.astype(str))
unique_casts = big_df_pivot['unique_date_location'].unique()
cast_id_map = {cast: i for i, cast in enumerate(unique_casts)}
big_df_pivot['cid'] = big_df_pivot['unique_date_location'].map(cast_id_map)
big_df_use = big_df_pivot

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Bottle Data Depth Profiles - Whidbey Basin', fontsize=14, fontweight='bold')

# Color by cast ID
cmap = plt.cm.tab20
n_casts = len(unique_casts)

# Temperature vs Depth
ax = axes[0, 0]
for cid in sorted(big_df_use['cid'].unique()):
    mask = big_df_use['cid'] == cid
    data = big_df_use[mask].dropna(subset=['Depth (m)', 'Temperature'])
    if len(data) > 0:
        color = cmap(cid % len(cmap.colors))
        ax.scatter(data['Temperature'], data['Depth (m)'], 
                  s=50, alpha=0.6, color=color, edgecolors='black', linewidth=0.5)

ax.set_xlabel('Temperature (°C)', fontsize=10)
ax.set_ylabel('Depth (m)', fontsize=10)
ax.invert_yaxis()
ax.grid(True, alpha=0.3)
ax.set_title('Temperature Profile')

# Salinity vs Depth
ax = axes[0, 1]
for cid in sorted(big_df_use['cid'].unique()):
    mask = big_df_use['cid'] == cid
    data = big_df_use[mask].dropna(subset=['Depth (m)', 'Salinity'])
    if len(data) > 0:
        color = cmap(cid % len(cmap.colors))
        ax.scatter(data['Salinity'], data['Depth (m)'], 
                  s=50, alpha=0.6, color=color, edgecolors='black', linewidth=0.5)

ax.set_xlabel('Salinity (PSS)', fontsize=10)
ax.set_ylabel('Depth (m)', fontsize=10)
ax.invert_yaxis()
ax.grid(True, alpha=0.3)
ax.set_title('Salinity Profile')

# Dissolved Oxygen vs Depth
ax = axes[1, 0]
for cid in sorted(big_df_use['cid'].unique()):
    mask = big_df_use['cid'] == cid
    data = big_df_use[mask].dropna(subset=['Depth (m)', 'Dissolved Oxygen'])
    if len(data) > 0:
        color = cmap(cid % len(cmap.colors))
        ax.scatter(data['Dissolved Oxygen'], data['Depth (m)'], 
                  s=50, alpha=0.6, color=color, edgecolors='black', linewidth=0.5)

ax.set_xlabel('Dissolved Oxygen (mg/L)', fontsize=10)
ax.set_ylabel('Depth (m)', fontsize=10)
ax.invert_yaxis()
ax.grid(True, alpha=0.3)
ax.set_title('Dissolved Oxygen Profile')

# Nitrite + Nitrate vs Depth
ax = axes[1, 1]
for cid in sorted(big_df_use['cid'].unique()):
    mask = big_df_use['cid'] == cid
    data = big_df_use[mask].dropna(subset=['Depth (m)', 'Nitrite + Nitrate Nitrogen'])
    if len(data) > 0:
        color = cmap(cid % len(cmap.colors))
        ax.scatter(data['Nitrite + Nitrate Nitrogen'], data['Depth (m)'], 
                  s=50, alpha=0.6, color=color, edgecolors='black', linewidth=0.5)

ax.set_xlabel('NO3 + NO2 (mg/L)', fontsize=10)
ax.set_ylabel('Depth (m)', fontsize=10)
ax.invert_yaxis()
ax.grid(True, alpha=0.3)
ax.set_title('Nitrite + Nitrate Profile')

plt.tight_layout()
plt.savefig('bottle_profiles.png', dpi=150, bbox_inches='tight')
print(f"Saved profile plot to bottle_profiles.png")
plt.show()
