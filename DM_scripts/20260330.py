#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 2026

@author: dakotamascarenas

Tidal excursion at the Penn Cove mouth (pc0 section) for full year 2017.

Approach: compute section-averaged velocity from hourly extraction data,
integrate to get Lagrangian particle displacement, then extract tidal
excursion (peak-to-peak displacement per tidal cycle).
"""

from lo_tools import Lfun, zfun
from lo_tools import plotting_functions as pfun

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy.signal import argrelextrema

# %% Setup

gridname = 'wb1'
tag = 'r0'
ex_name = 'xn11b'

Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)

ds0 = '2017.01.01'
ds1 = '2017.12.31'

out_dir0 = Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'tef2'

# %% Load hourly extraction data at pc0 section

ext_dir = out_dir0 / ('extractions_avg_' + ds0 + '_' + ds1)
ext_fn = ext_dir / 'pc0.nc'

ds_ext = xr.open_dataset(ext_fn)

q = ds_ext.q.values       # (time, z, p) volume flux [m3/s]
dd = ds_ext.dd.values      # (p,) section point width [m]
h = ds_ext.h.values        # (p,) depth at section points [m]
zeta = ds_ext.zeta.values  # (time, p) sea surface height [m]

# time coordinate
ot = ds_ext.time.values
time_hours = ot  # raw hourly times

NT, NZ, NP = q.shape

# Compute DZ (vertical cell thickness) from h, zeta, and S-coordinates
# DZ is needed to get cross-sectional area. If DZ is in the dataset, use it.
if 'DZ' in ds_ext.data_vars:
    DZ = ds_ext.DZ.values  # (time, z, p)
else:
    # Reconstruct from S-coordinates
    from lo_tools import zrfun
    G, S, T = zrfun.get_basic_info(Ldir['grid'] / 'grid.nc')
    # Build DZ for each time step
    DZ = np.zeros((NT, NZ, NP))
    for tt in range(NT):
        zw = zrfun.get_z(h.reshape(1, NP), zeta[tt, :].reshape(1, NP), S, only_w=True)
        DZ[tt, :, :] = np.diff(zw, axis=0)

ds_ext.close()

# %% Compute section-averaged velocity

# qnet(t) = total volume flux through section [m3/s]
qnet = np.nansum(q.reshape(NT, -1), axis=1)

# A(t) = time-varying cross-sectional area [m2]
# dd is (p,), DZ is (time, z, p) -> dd[None, None, :] * DZ gives (time, z, p)
A = np.nansum(dd[np.newaxis, np.newaxis, :] * DZ, axis=(1, 2))

# Section-averaged velocity [m/s]
# Positive = into Penn Cove (consistent with TEF sign convention from pm in extraction)
v_mean = qnet / A

# %% Integrate velocity to get particle displacement

dt = 3600.0  # hourly data [s]

# Cumulative displacement [m]
x_displacement = np.cumsum(v_mean * dt)
x_displacement = np.insert(x_displacement, 0, 0.0)[:-1]  # start at 0

# Remove subtidal drift with Godin lowpass filter
x_lp = zfun.lowpass(x_displacement, f='godin', nanpad=True)

# Tidal component of displacement
x_tidal = x_displacement - x_lp

# Also lowpass the velocity for plotting
v_lp = zfun.lowpass(v_mean, f='godin', nanpad=True)

# %% Extract tidal excursion (peak-to-peak per tidal cycle)

# Find local maxima and minima of x_tidal
# Use order ~6 hours (6 points at hourly resolution) to avoid noise peaks
order = 6

# Mask out NaN regions (from Godin filter edges)
valid = ~np.isnan(x_tidal)
idx_valid = np.where(valid)[0]

x_tidal_valid = x_tidal[valid]

imax = argrelextrema(x_tidal_valid, np.greater, order=order)[0]
imin = argrelextrema(x_tidal_valid, np.less, order=order)[0]

# Merge and sort extrema
extrema_idx = np.sort(np.concatenate([imax, imin]))
extrema_vals = x_tidal_valid[extrema_idx]

# Tidal excursion = absolute difference between successive extrema
L_e_raw = np.abs(np.diff(extrema_vals))
# Time is midpoint between successive extrema (map back to full array indices)
extrema_full_idx = idx_valid[extrema_idx]
L_e_times_idx = ((extrema_full_idx[:-1] + extrema_full_idx[1:]) / 2).astype(int)

# Build a full-length array for L_e (NaN where no data), then lowpass
L_e_full = np.full(NT, np.nan)
L_e_full[L_e_times_idx] = L_e_raw

# Interpolate to fill gaps for filtering
from scipy.interpolate import interp1d
valid_Le = ~np.isnan(L_e_full)
if np.sum(valid_Le) > 2:
    f_interp = interp1d(np.where(valid_Le)[0], L_e_full[valid_Le],
                         kind='linear', bounds_error=False, fill_value=np.nan)
    L_e_interp = f_interp(np.arange(NT))
else:
    L_e_interp = L_e_full.copy()

# Godin lowpass for smooth daily series
L_e_lp = zfun.lowpass(L_e_interp, f='godin', nanpad=True)

# %% Load qprism from bulk output for comparison

bulk_dir = out_dir0 / ('bulk_avg_' + ds0 + '_' + ds1)
bulk_fn = bulk_dir / 'pc0.nc'

ds_bulk = xr.open_dataset(bulk_fn)
qprism = ds_bulk.qprism.values  # [m3/s]
qprism_time = ds_bulk.time.values
ds_bulk.close()

# Estimate excursion from qprism for comparison:
# L_e_qprism ~ (qprism * T_tide/pi) / A_mean
# where T_tide ~ 12.42 hrs (M2), and we use sinusoidal assumption
T_M2 = 12.42 * 3600  # M2 tidal period [s]
A_mean = np.nanmean(A)
L_e_qprism = (qprism * T_M2 / np.pi) / A_mean  # [m]

# %% Convert times for plotting

time_plot = ot  # already datetime64 from xarray

# %% Plot

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# --- Panel 1: Section-averaged velocity ---
ax = axes[0]
ax.plot(time_plot, v_mean, color='tab:blue', alpha=0.3, linewidth=0.5, label='hourly')
ax.plot(time_plot, v_lp, color='tab:blue', linewidth=1.5, label='tidally averaged')
ax.axhline(0, color='k', linewidth=0.5)
ax.set_ylabel('Velocity [m/s]')
ax.set_title('Penn Cove Mouth (pc0) — Section-Averaged Velocity')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# --- Panel 2: Tidal displacement ---
ax = axes[1]
ax.plot(time_plot, x_tidal / 1000, color='tab:orange', linewidth=0.5)
ax.axhline(0, color='k', linewidth=0.5)
ax.set_ylabel('Tidal Displacement [km]')
ax.set_title('Lagrangian Particle Displacement (tidal component)')
ax.grid(True, alpha=0.3)

# --- Panel 3: Tidal excursion ---
ax = axes[2]
ax.plot(time_plot, L_e_lp / 1000, color='tab:red', linewidth=2,
        label='velocity integration (Godin avg)')
ax.plot(qprism_time, L_e_qprism / 1000, color='tab:green', linewidth=1.5,
        linestyle='--', label=r'$Q_{prism} \cdot T_{M2} / (\pi \cdot A)$')
ax.set_ylabel('Tidal Excursion [km]')
ax.set_xlabel('Date (2017)')
ax.set_title('Tidal Excursion — Peak-to-Peak')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Format x-axis
axes[2].xaxis.set_major_locator(mdates.MonthLocator())
axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%b'))

plt.tight_layout()
plt.savefig(Ldir['LOo'] / 'plots' / 'tidal_excursion_pc0_2017.png', dpi=200)
plt.show()

# %% Print summary statistics

print('\n=== Tidal Excursion Summary (pc0, 2017) ===')
print(f'Mean cross-sectional area: {A_mean:.0f} m2')
print(f'Mean section-averaged velocity amplitude: {np.nanstd(v_mean):.3f} m/s')
valid_Le_lp = L_e_lp[~np.isnan(L_e_lp)]
if len(valid_Le_lp) > 0:
    print(f'Mean tidal excursion (velocity integration): {np.nanmean(valid_Le_lp)/1000:.2f} km')
    print(f'Max tidal excursion (spring): {np.nanmax(valid_Le_lp)/1000:.2f} km')
    print(f'Min tidal excursion (neap): {np.nanmin(valid_Le_lp)/1000:.2f} km')
print(f'Mean tidal excursion (qprism estimate): {np.nanmean(L_e_qprism)/1000:.2f} km')
