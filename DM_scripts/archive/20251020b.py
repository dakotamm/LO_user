#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:28:39 2025

@author: dakotamascarenas
"""

from lo_tools import Lfun, zfun, zrfun
from lo_tools import plotting_functions as pfun
import matplotlib.pyplot as plt
import matplotlib.path as mpth
import xarray as xr
import numpy as np
import pandas as pd
import datetime

from warnings import filterwarnings
filterwarnings('ignore') # skip some warning messages

import seaborn as sns

import scipy.stats as stats

import D_functions as dfun

import pickle

import math

from scipy.interpolate import interp1d

import gsw

import matplotlib.path as mpth

import matplotlib.patches as patches

import cmocean

# %%

fn0 = '/Users/dakotamascarenas/LO_data/trapsD01/processed_data/wwtp_data_mohamedali_etal_2020.nc'
ds0 = xr.open_dataset(fn0)

fn1 = '/Users/dakotamascarenas/LO_data/trapsD01/processed_data/river_data_mohamedali_etal_2020.nc'
ds1 = xr.open_dataset(fn1)

fn2 = '/Users/dakotamascarenas/LO_data/trapsD01/processed_data/wwtp_data_wasielewski_etal_2024.nc'
ds2 = xr.open_dataset(fn2)

# %%


# === 2. Identify variables that depend on both 'source' and 'date' ===
data_vars = [v for v in ds0.data_vars if {'source', 'date'} <= set(ds0[v].dims)]

# === 3. Convert to a tidy wide dataframe first ===
df_wide = ds0[data_vars + ['ID', 'name', 'lon', 'lat']].to_dataframe().reset_index()

# === 4. Melt to long form ===
df_long = df_wide.melt(
    id_vars=['source', 'date', 'ID', 'name', 'lon', 'lat'],
    value_vars=data_vars,
    var_name='variable',
    value_name='value'
)

# === 5. Optional: sort for readability ===
df0 = df_long.sort_values(['source', 'date', 'variable']).reset_index(drop=True)

# %%

# === 2. Identify variables that depend on both 'source' and 'date' ===
data_vars = [v for v in ds1.data_vars if {'source', 'date'} <= set(ds1[v].dims)]

# === 3. Convert to a tidy wide dataframe first ===
df_wide = ds1[data_vars + ['ID', 'name', 'lon', 'lat']].to_dataframe().reset_index()

# === 4. Melt to long form ===
df_long = df_wide.melt(
    id_vars=['source', 'date', 'ID', 'name', 'lon', 'lat'],
    value_vars=data_vars,
    var_name='variable',
    value_name='value'
)

df1 = df_long.sort_values(['source', 'date', 'variable']).reset_index(drop=True)

# %%

# === 2. Identify variables that depend on both 'source' and 'date' ===
data_vars = [v for v in ds2.data_vars if {'source', 'date'} <= set(ds2[v].dims)]

# === 3. Convert to a tidy wide dataframe first ===
df_wide = ds2[data_vars + ['ID', 'name', 'lon', 'lat']].to_dataframe().reset_index()

# === 4. Melt to long form ===
df_long = df_wide.melt(
    id_vars=['source', 'date', 'ID', 'name', 'lon', 'lat'],
    value_vars=data_vars,
    var_name='variable',
    value_name='value'
)

df2 = df_long.sort_values(['source', 'date', 'variable']).reset_index(drop=True)

# %%

df_long = df2

# %%

from pathlib import Path

# === 1. Set save directory ===
save_dir = Path("/Users/dakotamascarenas/Desktop/pltz")
save_dir.mkdir(parents=True, exist_ok=True)  # create if not exists

# === 2. Define your site list ===
sites = [
    "CARLYON BEACH STP",
    "BOSTON HARBOR STP",
    "TAMOSHAN STP",
    "SEASHORE VILLA STP",
    "SHELTON STP",
    "RUSTLEWOOD STP",
    "HARTSTENE POINTE STP",
    "TAYLOR BAY STP",
    "McNeil Island Special Commitment Center WWTP",
    "CHAMBERS CREEK STP",
    "TACOMA NORTH NO 3",
    "TACOMA CENTRAL NO 1",
    "GIG HARBOR STP",
    "LAKOTA WWTP",
    "REDONDO WWTP",
    "MIDWAY SEWER DISTRICT WWTP",
    "King County Vashon WWTP",
    "MILLER CREEK WWTP",
    "SALMON CREEK WWTP",
    "BREMERTON STP",
    "PORT ORCHARD WWTP",
    "Kitsap County Manchester WWTP",
    "Kitsap County Sewer District #7 Water Reclamation Facility",
    "King County South WWTP",
    "BAINBRIDGE ISLAND WWTP",
    "MESSENGER HOUSE CARE CENTER WWTP",
    "King County West Point WWTP",
    "Kitsap County Central Kitsap WWTP",
    "Kitsap County Kingston WWTP",
    "King County Brightwater WWTP",
    "EDMONDS STP",
    "LYNNWOOD STP",
    "ALDERWOOD STP",
    "Port Gamble WWTP",
    "MUKILTEO WATER AND WASTEWATER DISTRICT WWTP",
    "Port Ludlow Wastewater Treatment Plant",
    "ALDERBROOK RESORT & SPA",
    "SNOHOMISH STP",
    "LAKE STEVENS SEWER DISTRICT",
    "Lake Stevens Sewer District WWTP",
    "Everett Water Pollution Control Facility",
    "MARYSVILLE STP",
    "Everett Water Pollution Control Facility",
    "LANGLEY STP",
    "PORT TOWNSEND STP",
    "WARM BEACH CAMPGROUND WWTP",
    "STANWOOD STP",
    "COUPEVILLE STP",
    "PENN COVE WWTP",
    "OAK HARBOR STP",
    "SKAGIT COUNTY SEWER DIST 2 BIG LAKE WWTP",
    "MT VERNON WWTP",
    "LA CONNER STP",
    "ANACORTES WWTP",
    "WA PARKS LARRABEE WWTP",
    "FISHERMAN BAY STP",
    "FRIDAY HARBOR STP",
    "EASTSOUND ORCAS VILLAGE WWTP",
    "ROCHE HARBOR RESORT WWTP",
    "ROSARIO WWTP",
    "EASTSOUND WATER DISTRICT WWTP",
    "BELLINGHAM STP",
    "BIRCH BAY STP",
    "Lighthouse Point Water Reclamation Facility",
    "PORT ANGELES STP",
    "CLALLAM BAY STP",
    "SEKIU STP",
    "CLALLAM BAY CORRECTION CENTER STP"
]

# === 3. Choose variable to plot ===
variable = "flow"  # <-- change this to any variable in your dataset (e.g. "temp", "NO3", etc.)

# === 4. Loop through each site and plot ===
for site in sites:
    df_site = df_long.query("variable == @variable and name == @site")
    if df_site.empty:
        print(f"⚠️ No data for site: {site}")
        continue

    plt.figure(figsize=(10, 5))
    plt.plot(df_site["date"], df_site["value"], lw=1.5, color="steelblue")
    plt.title(f"{variable} — {site}")
    plt.xlabel("Date")
    plt.ylabel(variable)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # clean filename
    fname = f"{site.replace(' ', '_').replace('/', '-')}_{variable}.png"
    save_path = save_dir / fname
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✅ Saved plot: {save_path}")

print("\n🎉 All plots saved successfully!")

# %%

# === 3. Choose variable to plot ===
variable = "NO3"  # <-- change this to any variable in your dataset (e.g. "temp", "NO3", etc.)

# === 4. Loop through each site and plot ===
for site in sites:
    df_site = df_long.query("variable == @variable and name == @site")
    if df_site.empty:
        print(f"⚠️ No data for site: {site}")
        continue

    plt.figure(figsize=(10, 5))
    plt.plot(df_site["date"], df_site["value"], lw=1.5, color="steelblue")
    plt.title(f"{variable} — {site}")
    plt.xlabel("Date")
    plt.ylabel(variable)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # clean filename
    fname = f"{site.replace(' ', '_').replace('/', '-')}_{variable}.png"
    save_path = save_dir / fname
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✅ Saved plot: {save_path}")

print("\n🎉 All plots saved successfully!")

# %%

# === 3. Choose variable to plot ===
variable = "temp"  # <-- change this to any variable in your dataset (e.g. "temp", "NO3", etc.)

# === 4. Loop through each site and plot ===
for site in sites:
    df_site = df_long.query("variable == @variable and name == @site")
    if df_site.empty:
        print(f"⚠️ No data for site: {site}")
        continue

    plt.figure(figsize=(10, 5))
    plt.plot(df_site["date"], df_site["value"], lw=1.5, color="steelblue")
    plt.title(f"{variable} — {site}")
    plt.xlabel("Date")
    plt.ylabel(variable)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # clean filename
    fname = f"{site.replace(' ', '_').replace('/', '-')}_{variable}.png"
    save_path = save_dir / fname
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✅ Saved plot: {save_path}")

print("\n🎉 All plots saved successfully!")

# %%

# === 3. Choose variable to plot ===
variable = "NH4"  # <-- change this to any variable in your dataset (e.g. "temp", "NO3", etc.)

# === 4. Loop through each site and plot ===
for site in sites:
    df_site = df_long.query("variable == @variable and name == @site")
    if df_site.empty:
        print(f"⚠️ No data for site: {site}")
        continue

    plt.figure(figsize=(10, 5))
    plt.plot(df_site["date"], df_site["value"], lw=1.5, color="steelblue")
    plt.title(f"{variable} — {site}")
    plt.xlabel("Date")
    plt.ylabel(variable)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # clean filename
    fname = f"{site.replace(' ', '_').replace('/', '-')}_{variable}.png"
    save_path = save_dir / fname
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✅ Saved plot: {save_path}")

print("\n🎉 All plots saved successfully!")




