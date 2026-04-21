"""
Visualize bottle data summary - spatial distribution and temporal coverage.

Shows where samples were collected, how many samples per location, data completeness,
and temporal coverage across the entire time series.
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

# Convert Value to numeric and parse time
big_df_use['Value'] = pd.to_numeric(big_df_use['Value'], errors='coerce')
big_df_use['time'] = pd.to_datetime(big_df_use['Collect DateTime'])

# Pivot to wide format
big_df_pivot = big_df_use.pivot_table(index=['Profile ID', 'Collect DateTime', 'Depth (m)', 
                                             'Locator', 'Latitude', 'Longitude'],
                                      columns='Parameter', values='Value').reset_index()
big_df_pivot['time'] = pd.to_datetime(big_df_pivot['Collect DateTime'])
big_df_use = big_df_pivot

# Create figure
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Bottle Data Summary - Whidbey Basin', fontsize=14, fontweight='bold')

# 1. Spatial distribution of sampling locations
ax1 = plt.subplot(2, 3, 1)
for locator in big_df_use['Locator'].unique():
    loc_data = big_df_use[big_df_use['Locator'] == locator]
    if 'Latitude' in loc_data.columns and 'Longitude' in loc_data.columns:
        lat = loc_data['Latitude'].iloc[0]
        lon = loc_data['Longitude'].iloc[0]
        n_samples = len(loc_data)
        size = min(300, 50 + n_samples/2)
        ax1.scatter(lon, lat, s=size, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax1.text(lon, lat, locator, fontsize=7, ha='center')

ax1.set_xlabel('Longitude', fontsize=10)
ax1.set_ylabel('Latitude', fontsize=10)
ax1.set_title('Sampling Locations\n(size = number of samples)')
ax1.grid(True, alpha=0.3)

# 2. Number of samples per location
ax2 = plt.subplot(2, 3, 2)
samples_per_loc = big_df_use.groupby('Locator').size().sort_values(ascending=False)
samples_per_loc.plot(kind='barh', ax=ax2, color='steelblue')
ax2.set_xlabel('Number of Samples', fontsize=10)
ax2.set_title('Samples per Location')
ax2.grid(True, alpha=0.3, axis='x')

# 3. Temporal coverage
ax3 = plt.subplot(2, 3, 3)
by_date = big_df_use.groupby(big_df_use['time'].dt.date).size()
ax3.plot(by_date.index, by_date.values, marker='o', markersize=4, color='steelblue')
ax3.set_xlabel('Date', fontsize=10)
ax3.set_ylabel('Number of Samples', fontsize=10)
ax3.set_title('Samples per Day')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3)

# 4. Data completeness for key parameters
ax4 = plt.subplot(2, 3, 4)
parameters = ['Temperature', 'Salinity', 'Dissolved Oxygen', 
              'Nitrite + Nitrate Nitrogen', 'Ammonia Nitrogen', 'Chlorophyll a']
completeness = []
for param in parameters:
    if param in big_df_use.columns:
        comp = big_df_use[param].notna().sum() / len(big_df_use) * 100
    else:
        comp = 0
    completeness.append(comp)
colors = ['green' if x > 80 else 'orange' if x > 50 else 'red' for x in completeness]
ax4.barh(parameters, completeness, color=colors)
ax4.set_xlabel('% Data Available', fontsize=10)
ax4.set_xlim([0, 105])
ax4.set_title('Data Completeness')
for i, v in enumerate(completeness):
    ax4.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
ax4.grid(True, alpha=0.3, axis='x')

# 5. Distribution of depths sampled
ax5 = plt.subplot(2, 3, 5)
big_df_use['Depth (m)'] = pd.to_numeric(big_df_use['Depth (m)'], errors='coerce')
big_df_use['Depth (m)'].dropna().hist(bins=30, ax=ax5, color='steelblue', edgecolor='black')
ax5.set_xlabel('Depth (m)', fontsize=10)
ax5.set_ylabel('Frequency', fontsize=10)
ax5.set_title('Depth Distribution')
ax5.grid(True, alpha=0.3, axis='y')

# 6. Summary statistics
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary_text = f"""
DATA SUMMARY
{'='*40}
Total records: {len(big_df_use):,}
Date range: {big_df_use['time'].min().date()} to {big_df_use['time'].max().date()}
Unique locations: {big_df_use['Locator'].nunique()}
Unique dates: {big_df_use['time'].dt.date.nunique()}
Unique profiles: {big_df_use['Profile ID'].nunique()}

VARIABLE STATS
Temperature (°C):
  Count: {big_df_use['Temperature'].notna().sum()}
  Mean: {big_df_use['Temperature'].mean():.2f}

Salinity (PSS):
  Count: {big_df_use['Salinity'].notna().sum()}
  Mean: {big_df_use['Salinity'].mean():.2f}

DO (mg/L):
  Count: {big_df_use['Dissolved Oxygen'].notna().sum()}
  Mean: {big_df_use['Dissolved Oxygen'].mean():.2f}

NO3 + NO2 (mg/L):
  Count: {big_df_use['Nitrite + Nitrate Nitrogen'].notna().sum()}
  Mean: {big_df_use['Nitrite + Nitrate Nitrogen'].mean():.2f}
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
         fontsize=9, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('bottle_summary.png', dpi=150, bbox_inches='tight')
print(f"Saved summary plot to bottle_summary.png")
plt.show()
