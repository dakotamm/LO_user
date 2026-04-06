"""
Plot the total nitrogen (TN) budget for Penn Cove (wb1_pc0), combining:
  - Transport terms from tracer_budget_avg.py pickle (if individual N tracers present)
  - Sediment terms (denitrification, detritus loss) from get_TN_sediment_wb1.py NetCDF

TN = NO3 + NH4 + phytoplankton + zooplankton + SdetritusN + LdetritusN

If tracer_budget_avg was run with all tracers (not just salt/temp/oxygen),
this script will sum the individual N-tracer budgets to get TN transport.
Otherwise, it computes d(TN*vol)/dt from the TN extraction and shows
sediment terms only.

Usage:
  python plot_TN_budget_wb1.py
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

# N species that make up TN
N_species = ['NO3', 'NH4', 'phytoplankton', 'zooplankton', 'SdetritusN', 'LdetritusN']

# ============================================================
# 1. Try to load transport budget from tracer_budget_avg.py
# ============================================================
budget_dir = tef2_dir / ('Budgets_avg_' + date_str)
B_dict = pd.read_pickle(budget_dir / (vol_str + '.p'))

# Check if all N species are in the budget
have_all_N = all(vn in B_dict for vn in N_species)

if have_all_N:
    print('Found all N species in budget pickle — computing TN transport.')
    # Sum individual N-tracer budgets to get TN budget
    TN_df = B_dict[N_species[0]].copy() * 0  # zero DataFrame with same structure
    for vn in N_species:
        TN_df += B_dict[vn]
    has_transport = True
else:
    print('Budget pickle does not contain all N species.')
    print(f'  Available: {[k for k in B_dict.keys()]}')
    print('  Will show sediment terms and d(TN*vol)/dt from extraction only.')
    # Use volume budget for time index
    TN_df = B_dict['volume'].copy() * 0
    has_transport = False

budget_time = TN_df.index.values

# ============================================================
# 2. Load sediment TN terms (hourly)
# ============================================================
tn_fn = tef2_dir / ('TN_sediment_' + ds0 + '_' + ds1 + '_' + sect_gctag + '.nc')
nc = Dataset(str(tn_fn))

bgc_time_ns = nc['time'][:]
bgc_time = pd.to_datetime(bgc_time_ns, unit='ns')

TN_vol_h = np.array(nc['TN_vol_sum'][:])            # mmol N
denitri_h = np.array(nc['denitri_flux_sum'][:])      # mmol N/hr
NH4_gain_h = np.array(nc['NH4_gain_flux_sum'][:])    # mmol N/hr
det_loss_h = np.array(nc['detritus_loss_sum'][:])    # mmol N/hr
nc.close()

# Convert from mmol/hr to mmol/s
denitri_h /= 3600
NH4_gain_h /= 3600
det_loss_h /= 3600

# Net sediment N loss = detritus settling - NH4 returned to water column + denitrification
# (denitrification removes N entirely; detritus_loss - NH4_gain = net burial/loss)
net_sed_loss_h = det_loss_h - NH4_gain_h + denitri_h

# d(TN*vol)/dt from hourly data [mmol N / s]
dTNdt_h = (TN_vol_h[2:] - TN_vol_h[:-2]) / (2 * 3600)
dTNdt_h_full = np.nan * np.zeros(len(TN_vol_h))
dTNdt_h_full[1:-1] = dTNdt_h

# ============================================================
# 3. Godin-filter to daily
# ============================================================
pad = 36

def godin_to_daily(vec_h):
    filtered = zfun.lowpass(vec_h, f='godin')
    return filtered[pad:-pad+1:24]

denitri_d = godin_to_daily(denitri_h)
NH4_gain_d = godin_to_daily(NH4_gain_h)
det_loss_d = godin_to_daily(det_loss_h)
net_sed_d = godin_to_daily(net_sed_loss_h)
dTNdt_d = godin_to_daily(dTNdt_h_full)

bgc_time_d = bgc_time[pad:-pad+1:24]

# ============================================================
# 4. Align to budget time index
# ============================================================
tn_series = {}
for name, data in [('denitri', denitri_d), ('NH4_gain', NH4_gain_d),
                    ('det_loss', det_loss_d), ('net_sed', net_sed_d),
                    ('dTNdt', dTNdt_d)]:
    s = pd.Series(data, index=bgc_time_d)
    tn_series[name] = s.reindex(TN_df.index, method='nearest', tolerance=pd.Timedelta('12h'))

# ============================================================
# 5. Plot
# ============================================================
pfun.start_plot(figsize=(14, 10))

if has_transport:
    fig, axes = plt.subplots(3, 1, sharex=True)

    # --- Panel 1: Full TN budget ---
    ax = axes[0]
    ax.plot(TN_df.index, TN_df['d_dt'], 'k-', linewidth=1.5, label='d(TN·V)/dt')
    ax.plot(TN_df.index, TN_df['ocn'], 'b-', linewidth=1.2, label='ocean transport')
    ax.plot(TN_df.index, TN_df['riv'], 'c-', linewidth=1.2, label='river')
    ax.plot(TN_df.index, -tn_series['net_sed'], 'r-', linewidth=1.2, label='-net sediment loss')
    err = TN_df['d_dt'] - TN_df['ocn'] - TN_df['riv'] + tn_series['net_sed']
    ax.plot(TN_df.index, err, 'r--', linewidth=1, label='error (residual)')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_ylabel('mmol N s$^{-1}$')
    ax.set_title('TN Budget: ' + which_vol + ' (' + gtagex + ', ' + ds0 + ' to ' + ds1 + ')')
    ax.legend(loc='best', fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Sediment terms breakdown ---
    ax = axes[1]
    ax.plot(TN_df.index, -tn_series['det_loss'], 'brown', linewidth=1.2, label='-detritus settling')
    ax.plot(TN_df.index, tn_series['NH4_gain'], 'g-', linewidth=1.2, label='NH4 recycled')
    ax.plot(TN_df.index, -tn_series['denitri'], 'r-', linewidth=1.2, label='-denitrification')
    ax.plot(TN_df.index, -tn_series['net_sed'], 'k--', linewidth=1, label='-net sediment loss')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_ylabel('mmol N s$^{-1}$')
    ax.set_title('Sediment N Sources and Sinks')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Error ---
    ax = axes[2]
    ax.plot(TN_df.index, TN_df['err'], 'r-', linewidth=1.2, label='error (no sediment)')
    ax.plot(TN_df.index, err, 'b-', linewidth=1.2, label='error (with sediment)')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_ylabel('mmol N s$^{-1}$')
    ax.set_title('Budget Residual: Before vs After Adding Sediment Terms')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

else:
    fig, axes = plt.subplots(2, 1, sharex=True)

    # --- Panel 1: d(TN*vol)/dt and sediment terms ---
    ax = axes[0]
    ax.plot(TN_df.index, tn_series['dTNdt'], 'k-', linewidth=1.5, label='d(TN·V)/dt')
    ax.plot(TN_df.index, -tn_series['net_sed'], 'r-', linewidth=1.2, label='-net sediment loss')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_ylabel('mmol N s$^{-1}$')
    ax.set_title('TN Budget (no transport): ' + which_vol + ' (' + gtagex + ', ' + ds0 + ' to ' + ds1 + ')')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Sediment breakdown ---
    ax = axes[1]
    ax.plot(TN_df.index, -tn_series['det_loss'], 'brown', linewidth=1.2, label='-detritus settling')
    ax.plot(TN_df.index, tn_series['NH4_gain'], 'g-', linewidth=1.2, label='NH4 recycled')
    ax.plot(TN_df.index, -tn_series['denitri'], 'r-', linewidth=1.2, label='-denitrification')
    ax.plot(TN_df.index, -tn_series['net_sed'], 'k--', linewidth=1, label='-net sediment loss')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_ylabel('mmol N s$^{-1}$')
    ax.set_title('Sediment N Sources and Sinks')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.savefig(str(tef2_dir / ('TN_budget_' + ds0 + '_' + ds1 + '_' + sect_gctag + '.png')),
            dpi=200, bbox_inches='tight')
plt.show()
pfun.end_plot()

print('Done.')
