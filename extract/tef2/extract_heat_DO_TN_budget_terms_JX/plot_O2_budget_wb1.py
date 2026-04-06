"""
Plot the full O2 budget for Penn Cove (wb1_pc0), combining:
  - Transport terms (d_dt, ocn, riv) from tracer_budget_avg.py pickle
  - BGC + air-sea terms from get_DO_bgc_air_sea_wb1.py NetCDF

All terms are daily (Godin-filtered) in units of [mmol O2 s-1].

Budget equation:
  d_dt = ocn + riv + bio + airsea + error
where:
  bio = production - nitrification - remineralization - SOD

Usage (on apogee or wherever the output lives):
  python plot_O2_budget_wb1.py
"""

import numpy as np
import pandas as pd
import pickle
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
oxy_df = B_dict['oxygen']  # columns: net, d_dt, riv, ocn, surf, err
# Units: [mmol O2 s-1] (= uM * m3 / s)
# Index: daily at noon, NaN at first and last entries for d_dt/ocn

budget_time = oxy_df.index.values  # daily datetime index

# ============================================================
# 2. Load BGC + air-sea terms (hourly) from get_DO_bgc_air_sea_wb1.py
# ============================================================
bgc_fn = tef2_dir / ('O2_bgc_' + ds0 + '_' + ds1 + '_' + sect_gctag + '.nc')
nc = Dataset(str(bgc_fn))

# Time: ocean_time stored as datetime64 nanoseconds
bgc_time_ns = nc['time'][:]
bgc_time = pd.to_datetime(bgc_time_ns, unit='ns')

# Hourly domain-integrated terms [mmol O2/hr]
pro_h = np.array(nc['Oxy_pro_sum'][:])         # photosynthesis (source)
nitri_h = np.array(nc['Oxy_nitri_sum'][:])      # nitrification (sink)
remi_h = np.array(nc['Oxy_remi_sum'][:])        # remineralization (sink)
sed_h = np.array(nc['Oxy_sed_sum2'][:])         # SOD method 2 (sink) [mmol/hr]
airsea_h = np.array(nc['Oxy_air_flux_sum'][:])  # air-sea exchange [mmol/hr]
nc.close()

# Convert from mmol/hr to mmol/s
pro_h /= 3600
nitri_h /= 3600
remi_h /= 3600
sed_h /= 3600
airsea_h /= 3600

# Net bio source/sink [mmol O2 s-1]
# Positive = O2 gain; nitrification, remineralization, SOD are sinks
bio_h = pro_h - nitri_h - remi_h - sed_h

# ============================================================
# 3. Godin-filter hourly BGC terms to daily
#    (same method as tracer_budget_avg.py)
# ============================================================
pad = 36

def godin_to_daily(vec_h):
    """Godin-filter hourly data and subsample to daily."""
    filtered = zfun.lowpass(vec_h, f='godin')
    return filtered[pad:-pad+1:24]

pro_d = godin_to_daily(pro_h)
nitri_d = godin_to_daily(nitri_h)
remi_d = godin_to_daily(remi_h)
sed_d = godin_to_daily(sed_h)
airsea_d = godin_to_daily(airsea_h)
bio_d = godin_to_daily(bio_h)

# Daily times from the BGC hourly timestamps
bgc_time_d = bgc_time[pad:-pad+1:24]

# ============================================================
# 4. Align BGC daily values to budget DataFrame time index
#    Budget index has NaN at first/last; valid data is [1:-1].
#    Match by nearest date.
# ============================================================
bgc_series = {}
for name, data in [('pro', pro_d), ('nitri', nitri_d), ('remi', remi_d),
                    ('sed', sed_d), ('airsea', airsea_d), ('bio', bio_d)]:
    s = pd.Series(data, index=bgc_time_d)
    # Reindex to budget time, using nearest match within 12 hours
    bgc_series[name] = s.reindex(oxy_df.index, method='nearest', tolerance=pd.Timedelta('12h'))

# ============================================================
# 5. Compute updated error (residual with BGC terms included)
# ============================================================
# Original: err_old = d_dt - riv - ocn - 0
# Now:      err_new = d_dt - riv - ocn - bio - airsea
err_new = oxy_df['d_dt'] - oxy_df['riv'] - oxy_df['ocn'] - bgc_series['bio'] - bgc_series['airsea']

# ============================================================
# 6. Plot
# ============================================================
pfun.start_plot(figsize=(14, 10))

fig, axes = plt.subplots(3, 1, sharex=True)

# --- Panel 1: Full budget terms ---
ax = axes[0]
ax.plot(oxy_df.index, oxy_df['d_dt'], 'k-', linewidth=1.5, label='d(O2·V)/dt')
ax.plot(oxy_df.index, oxy_df['ocn'], 'b-', linewidth=1.2, label='ocean transport')
ax.plot(oxy_df.index, oxy_df['riv'], 'c-', linewidth=1.2, label='river')
ax.plot(oxy_df.index, bgc_series['bio'], 'g-', linewidth=1.2, label='net bio')
ax.plot(oxy_df.index, bgc_series['airsea'], 'm-', linewidth=1.2, label='air-sea')
ax.plot(oxy_df.index, err_new, 'r--', linewidth=1, label='error (residual)')
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_ylabel('mmol O2 s$^{-1}$')
ax.set_title('O2 Budget: ' + which_vol + ' (' + gtagex + ', ' + ds0 + ' to ' + ds1 + ')')
ax.legend(loc='best', fontsize=8, ncol=3)
ax.grid(True, alpha=0.3)

# --- Panel 2: BGC terms breakdown ---
ax = axes[1]
ax.plot(oxy_df.index, bgc_series['pro'], 'g-', linewidth=1.2, label='production')
ax.plot(oxy_df.index, -bgc_series['nitri'], 'orange', linewidth=1.2, label='-nitrification')
ax.plot(oxy_df.index, -bgc_series['remi'], 'brown', linewidth=1.2, label='-remineralization')
ax.plot(oxy_df.index, -bgc_series['sed'], 'r-', linewidth=1.2, label='-SOD')
ax.plot(oxy_df.index, bgc_series['bio'], 'k--', linewidth=1, label='net bio')
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_ylabel('mmol O2 s$^{-1}$')
ax.set_title('Biological O2 Sources and Sinks')
ax.legend(loc='best', fontsize=8, ncol=3)
ax.grid(True, alpha=0.3)

# --- Panel 3: Error comparison (old vs new) ---
ax = axes[2]
ax.plot(oxy_df.index, oxy_df['err'], 'r-', linewidth=1.2, label='error (no bio/airsea)')
ax.plot(oxy_df.index, err_new, 'b-', linewidth=1.2, label='error (with bio/airsea)')
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_ylabel('mmol O2 s$^{-1}$')
ax.set_title('Budget Residual: Before vs After Adding BGC + Air-Sea Terms')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.savefig(str(tef2_dir / ('O2_budget_' + ds0 + '_' + ds1 + '_' + sect_gctag + '.png')),
            dpi=200, bbox_inches='tight')
plt.show()
pfun.end_plot()

print('Done.')
