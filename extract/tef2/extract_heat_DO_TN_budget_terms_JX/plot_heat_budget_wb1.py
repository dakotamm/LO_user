"""
Plot the heat budget for Penn Cove (wb1_pc0), combining:
  - Transport terms (d_dt, ocn, riv, surf) from tracer_budget_avg.py pickle
  - Heat flux component breakdown from get_heat_air_sea_wb1.py NetCDF

All terms are daily (Godin-filtered) in units of [degC m3 s-1].

Usage:
  python plot_heat_budget_wb1.py
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from lo_tools import Lfun, zfun
from lo_tools import plotting_functions as pfun
from pathlib import Path
from netCDF4 import Dataset

# ============ USER SETTINGS ============
gtagex = 'wb1_r0_xn11ab'
ds0 = '2017.01.01'
ds1 = '2017.01.21'
sect_gctag = 'wb1_pc0'
which_vol = 'Penn Cove'
# =======================================

Ldir = Lfun.Lstart()
vol_str = which_vol.replace(' ', '_')
date_str = '_' + ds0 + '_' + ds1
tef2_dir = Ldir['LOo'] / 'extract' / gtagex / 'tef2'

# ============================================================
# 1. Load transport budget from tracer_budget_avg.py
# ============================================================
budget_dir = tef2_dir / ('Budgets_avg_' + date_str)
B_dict = pd.read_pickle(budget_dir / (vol_str + '.p'))
temp_df = B_dict['temp']  # columns: net, d_dt, riv, ocn, surf, err
# Units: [degC m3 s-1]

# ============================================================
# 2. Load heat flux component breakdown (hourly)
# ============================================================
heat_fn = tef2_dir / ('heat_air_sea_' + ds0 + '_' + ds1 + '_' + sect_gctag + '.nc')
nc = Dataset(str(heat_fn))

bgc_time_ns = nc['time'][:]
bgc_time = pd.to_datetime(bgc_time_ns, unit='ns')

# Hourly domain-integrated fluxes [W = J/s]
shflux_h = np.array(nc['shflux_sum'][:])
latent_h = np.array(nc['latent_sum'][:])
sensible_h = np.array(nc['sensible_sum'][:])
lwrad_h = np.array(nc['lwrad_sum'][:])
swrad_h = np.array(nc['swrad_sum'][:])
nc.close()

# Convert from W to degC m3 s-1 (same as tracer_budget_avg.py)
rho = 1025  # kg/m3
Cp = 3985   # J/(kg degK)
shflux_h /= (rho * Cp)
latent_h /= (rho * Cp)
sensible_h /= (rho * Cp)
lwrad_h /= (rho * Cp)
swrad_h /= (rho * Cp)

# ============================================================
# 3. Godin-filter to daily
# ============================================================
pad = 36

def godin_to_daily(vec_h):
    filtered = zfun.lowpass(vec_h, f='godin')
    return filtered[pad:-pad+1:24]

shflux_d = godin_to_daily(shflux_h)
latent_d = godin_to_daily(latent_h)
sensible_d = godin_to_daily(sensible_h)
lwrad_d = godin_to_daily(lwrad_h)
swrad_d = godin_to_daily(swrad_h)

bgc_time_d = bgc_time[pad:-pad+1:24]

# ============================================================
# 4. Align to budget time index
# ============================================================
heat_series = {}
for name, data in [('shflux', shflux_d), ('latent', latent_d),
                    ('sensible', sensible_d), ('lwrad', lwrad_d), ('swrad', swrad_d)]:
    s = pd.Series(data, index=bgc_time_d)
    heat_series[name] = s.reindex(temp_df.index, method='nearest', tolerance=pd.Timedelta('12h'))

# ============================================================
# 5. Plot
# ============================================================
pfun.start_plot(figsize=(14, 10))

fig, axes = plt.subplots(3, 1, sharex=True)

# --- Panel 1: Full heat budget ---
ax = axes[0]
ax.plot(temp_df.index, temp_df['d_dt'], 'k-', linewidth=1.5, label='d(T·V)/dt')
ax.plot(temp_df.index, temp_df['ocn'], 'b-', linewidth=1.2, label='ocean transport')
ax.plot(temp_df.index, temp_df['riv'], 'c-', linewidth=1.2, label='river')
ax.plot(temp_df.index, temp_df['surf'], 'r-', linewidth=1.2, label='surface (shflux)')
ax.plot(temp_df.index, temp_df['err'], 'r--', linewidth=1, label='error (residual)')
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_ylabel('$^\\circ$C m$^3$ s$^{-1}$')
ax.set_title('Heat Budget: ' + which_vol + ' (' + gtagex + ', ' + ds0 + ' to ' + ds1 + ')')
ax.legend(loc='best', fontsize=8, ncol=3)
ax.grid(True, alpha=0.3)

# --- Panel 2: Surface heat flux component breakdown ---
ax = axes[1]
ax.plot(temp_df.index, heat_series['swrad'], 'orange', linewidth=1.2, label='shortwave')
ax.plot(temp_df.index, heat_series['lwrad'], 'purple', linewidth=1.2, label='longwave')
ax.plot(temp_df.index, heat_series['latent'], 'b-', linewidth=1.2, label='latent')
ax.plot(temp_df.index, heat_series['sensible'], 'g-', linewidth=1.2, label='sensible')
ax.plot(temp_df.index, heat_series['shflux'], 'k--', linewidth=1, label='net (shflux)')
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_ylabel('$^\\circ$C m$^3$ s$^{-1}$')
ax.set_title('Surface Heat Flux Components')
ax.legend(loc='best', fontsize=8, ncol=3)
ax.grid(True, alpha=0.3)

# --- Panel 3: Budget balance check ---
ax = axes[2]
ax.plot(temp_df.index, temp_df['surf'], 'r-', linewidth=1.2, label='surf (from budget)')
ax.plot(temp_df.index, heat_series['shflux'], 'b--', linewidth=1.2, label='shflux (from extraction)')
ax.plot(temp_df.index, temp_df['surf'] - heat_series['shflux'], 'k-', linewidth=1, label='difference')
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_ylabel('$^\\circ$C m$^3$ s$^{-1}$')
ax.set_title('Surface Term Cross-Check')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.savefig(str(tef2_dir / ('heat_budget_' + ds0 + '_' + ds1 + '_' + sect_gctag + '.png')),
            dpi=200, bbox_inches='tight')
plt.show()
pfun.end_plot()

print('Done.')
