"""
Visualize CTD data summary - spatial distribution and temporal coverage.

Shows where casts were taken, how many samples per location, and
data completeness across time and variables.
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

# Parse time
big_df['time'] = pd.to_datetime(big_df['Sample Date'], infer_datetime_format=True)

# Create figure
fig = plt.figure(figsize=(16, 10))
fig.suptitle('CTD Data Summary - Whidbey Basin', fontsize=14, fontweight='bold')

# 1. Spatial distribution of sampling locations
ax1 = plt.subplot(2, 3, 1)
for locator in big_df['Locator'].unique():
    loc_data = big_df[big_df['Locator'] == locator]
    lat = loc_data['Latitude'].iloc[0]
    lon = loc_data['Longitude'].iloc[0]
    n_samples = len(loc_data)
    size = min(300, 50 + n_samples/2)
    ax1.scatter(lon, lat, s=size, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax1.text(lon, lat, locator, fontsize=7, ha='center')

ax1.set_xlabel('Longitude', fontsize=10)
ax1.set_ylabel('Latitude', fontsize=10)
ax1.set_title('Sampling Locations\n(size = number of casts)')
ax1.grid(True, alpha=0.3)

# 2. Number of samples per location
ax2 = plt.subplot(2, 3, 2)
samples_per_loc = big_df.groupby('Locator').size().sort_values(ascending=False)
samples_per_loc.plot(kind='barh', ax=ax2, color='steelblue')
ax2.set_xlabel('Number of Samples', fontsize=10)
ax2.set_title('Samples per Location')
ax2.grid(True, alpha=0.3, axis='x')

# 3. Temporal coverage
ax3 = plt.subplot(2, 3, 3)
by_date = big_df.groupby(big_df['time'].dt.date).size()
ax3.plot(by_date.index, by_date.values, marker='o', markersize=4, color='steelblue')
ax3.set_xlabel('Date', fontsize=10)
ax3.set_ylabel('Number of Samples', fontsize=10)
ax3.set_title('Samples per Day')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3)

# 4. Data completeness for key variables
ax4 = plt.subplot(2, 3, 4)
variables = ['Temperature (°C)', 'Salinity (PSU)', 'Dissolved Oxygen (mg/L)', 
             'Chlorophyll (µg/L)', 'Nitrate + Nitrite (mg N/L)']
completeness = [big_df[var].notna().sum() / len(big_df) * 100 for var in variables]
colors = ['green' if x > 80 else 'orange' if x > 50 else 'red' for x in completeness]
ax4.barh(variables, completeness, color=colors)
ax4.set_xlabel('% Data Available', fontsize=10)
ax4.set_xlim([0, 105])
ax4.set_title('Data Completeness')
for i, v in enumerate(completeness):
    ax4.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
ax4.grid(True, alpha=0.3, axis='x')

# 5. Distribution of depths sampled
ax5 = plt.subplot(2, 3, 5)
big_df['Depth (meters)'].dropna().hist(bins=30, ax=ax5, color='steelblue', edgecolor='black')
ax5.set_xlabel('Depth (meters)', fontsize=10)
ax5.set_ylabel('Frequency', fontsize=10)
ax5.set_title('Depth Distribution')
ax5.grid(True, alpha=0.3, axis='y')

# 6. Summary statistics
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary_text = f"""
DATA SUMMARY
{'='*40}
Total records: {len(big_df):,}
Date range: {big_df['time'].min().date()} to {big_df['time'].max().date()}
Unique locations: {big_df['Locator'].nunique()}
Unique dates: {big_df['time'].dt.date.nunique()}

VARIABLE STATS
Temperature (°C):
  Mean: {big_df['Temperature (°C)'].mean():.2f}, Range: {big_df['Temperature (°C)'].min():.2f} - {big_df['Temperature (°C)'].max():.2f}

Salinity (PSU):
  Mean: {big_df['Salinity (PSU)'].mean():.2f}, Range: {big_df['Salinity (PSU)'].min():.2f} - {big_df['Salinity (PSU)'].max():.2f}

DO (mg/L):
  Mean: {big_df['Dissolved Oxygen (mg/L)'].mean():.2f}, Range: {big_df['Dissolved Oxygen (mg/L)'].min():.2f} - {big_df['Dissolved Oxygen (mg/L)'].max():.2f}

Chlorophyll (µg/L):
  Mean: {big_df['Chlorophyll (µg/L)'].mean():.2f}, Range: {big_df['Chlorophyll (µg/L)'].min():.2f} - {big_df['Chlorophyll (µg/L)'].max():.2f}
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
         fontsize=9, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('ctd_summary.png', dpi=150, bbox_inches='tight')
print(f"Saved summary plot to ctd_summary.png")
plt.show()
