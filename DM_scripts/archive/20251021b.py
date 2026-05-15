#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 14:47:27 2025

@author: dakotamascarenas
"""

import pandas as pd
from pathlib import Path
import re
import xarray as xr

import matplotlib.pyplot as plt


# %%


# Folder with Excel files
folder = Path('/Users/dakotamascarenas/LO_data/trapsD01/mohamedali_etal2020/point_sources/')

# List of facility names to match
facilities = [
    "Carlyon",
    "Boston Harbor",
    "Tamoshan",
    "Seashore Villa",
    "Shelton",
    "Rustlewood",
    "Hartstene",
    "Taylor Bay",
    "McNeil Is",
    "Chambers Creek",
    "Tacoma North",
    "Tacoma Central",
    "Gig Harbor",
    "Lakota",
    "Redondo",
    "Midway",
    "Vashon",
    "Miller Creek",
    "Salmon Creek",
    "Bremerton",
    "Port Orchard",
    "Manchester",
    "Bainbridge Kitsap Co 7",
    "South King",
    "Bainbridge Island City",
    "Messenger House",
    "West Point",
    "Central Kitsap",
    "Kitsap Co Kingston",
    "Brightwater",
    "Edmonds",
    "Lynnwood",
    "Alderwood",
    "Port Gamble",
    "Mukilteo",
    "Port Ludlow",
    "Alderbrook",
    "Snohomish",
    "Lake Stevens 001",
    "Lake Stevens 002",
    "Everett Snohomish",
    "Marysville",
    "OF100",
    "Langley",
    "Port Townsend",
    "Warm Beach Campground",
    "Stanwood",
    "Coupeville",
    "Penn Cove",
    "Oak Harbor RBC",
    "Oak Harbor Lagoon",
    "Skagit County 2 Big Lake",
    "Mt Vernon",
    "La Conner",
    "Anacortes",
    "Larrabee State Park",
    "Fisherman Bay",
    "Friday Harbor",
    "Eastsound Orcas Village",
    "Roche Harbor",
    "Rosario Utilities",
    "Eastsound Water District",
    "Bellingham",
    "Birch Bay",
    "Blaine",
    "Port Angeles",
    "Clallam Bay POTW",
    "Sekiu",
    "Clallam DOC"
]

dfs = []

for name in facilities:
    matches = list(folder.glob(f"[0-9][0-9][0-9]_{name}*.xlsx"))
    if not matches:
        print(f"⚠️ No file found for: {name}")
        continue

    file = matches[0]

    # Extract numeric prefix (e.g. 501)
    code_match = re.match(r"(\d+)_", file.name)
    code = code_match.group(1) if code_match else None

    # Read Excel, skipping the first row so the second row is the header
    df = pd.read_excel(file, header=1)

    # Clean up column names
    df.columns = (
        df.columns.astype(str)
        .str.replace(r"\n", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Add identifying columns
    df["Facility"] = name
    df["Code"] = code

    dfs.append(df)

# Combine all DataFrames
combined_df_m = pd.concat(dfs, ignore_index=True)


# %%

fn2 = '/Users/dakotamascarenas/LO_data/trapsD01/processed_data/wwtp_data_wasielewski_etal_2024.nc'
ds2 = xr.open_dataset(fn2)

# %%

# === 2. Identify variables that depend on both 'source' and 'date' ===
data_vars = [v for v in ds2.data_vars if {'source', 'date'} <= set(ds2[v].dims)]

# === 3. Convert to a tidy wide dataframe first ===
df_wide = ds2[data_vars + ['ID', 'name', 'lon', 'lat']].to_dataframe().reset_index()

# === 4. Melt to long form ===
df_long = df_wide.melt(
    id_vars=['date', 'name'],
    value_vars=['flow', 'temp', 'NH4', 'NO3'],
    var_name='variable',
    value_name='value'
)

df2 = df_long.sort_values(['name', 'date', 'variable']).reset_index(drop=True)

# %%

df_w = df2

# %%

df_m = combined_df_m

# %%

# Read your rename mapping file
rename_df = pd.read_excel('/Users/dakotamascarenas/LO_data/trapsD01/wwtp_names_overlapping_only_DM.xlsx')

# Ensure columns are named clearly
rename_df.columns = rename_df.columns.str.strip().str.lower()

# Convert to dictionary: {'old_name': 'new_name'}
rename_dict = dict(zip(rename_df['mohamedali et al., 2020'], rename_df['wasielewski et al., 2024']))

# Apply to your existing combined_df
df_m['Facility'] = df_m['Facility'].replace(rename_dict)

print("✅ Facility names updated!")
print(df_m[['Code', 'Facility']].drop_duplicates().head())

# %%

df_m = df_m[df_m['Year'] <= 2004]

# %%

df_m = df_m.rename(columns={'Flow, cms': 'flow', 'Temp (C)': 'temp', 'NH4 (mg/L)': 'NH4', 'NO3+NO2 (mg/L)': 'NO3', 'Date': 'date', 'Facility':'name'})

# %%

df_m = df_m.melt(
    id_vars=['date', 'name'],
    value_vars=['flow', 'temp', 'NH4', 'NO3'],
    var_name='variable',
    value_name='value'
)

df_m = df_m.sort_values(['name', 'date', 'variable']).reset_index(drop=True)

# %%

df = pd.concat([df_m, df_w])

# %%

df['year'] = df['date'].dt.year

df['month'] = df['date'].dt.month

# %%


df_sum = df.groupby(['year', 'month', 'variable']).mean(numeric_only=True).reset_index()

df_sum['year_month'] = pd.to_datetime(
    df_sum['year'].astype(str) + '-' + df_sum['month'].astype(str) + '-01'
)

# %%

for var in df['variable'].unique():
    
    fig, ax = plt.subplots()
    
    plot_df = df_sum[df_sum['variable'] == var]
    
    ax.plot(plot_df['year_month'], plot_df['value'])
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/wwtp_' + var +'.png',
                dpi=500, transparent=False, bbox_inches='tight')

        
