"""
Diagnostic: compare raw hourly d_dt (from history files) with raw hourly qnet
(from avg Huon/Hvom section extraction) BEFORE any Godin filtering.

This checks whether ROMS volume conservation holds at the hourly level.
If it does, the budget error is introduced by the TEF pipeline (Godin, bulk_calc, etc.).
If it doesn't, the error is in the section/segment correspondence.

To run:
run budget_diagnostic -gtx cas7_trapsV00_meV00 -ctag c0 -riv trapsV00 -0 2017.01.01 -1 2017.01.22

"""

from lo_tools import Lfun, zfun
from lo_tools import plotting_functions as pfun
import tef_fun

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from lo_tools import extract_argfun as exfun
Ldir = exfun.intro()

sect_gctag = Ldir['gridname'] + '_' + Ldir['collection_tag']
riv_gctag = Ldir['gridname'] + '_' + Ldir['riv']
date_str = '_' + Ldir['ds0'] + '_' + Ldir['ds1']

# get budget_functions
pth = Ldir['LO'] / 'extract' / 'tef2'
upth = Ldir['LOu'] / 'extract' / 'tef2'
if (upth / 'budget_functions.py').is_file():
    bfun = Lfun.module_from_file('budget_functions', upth / 'budget_functions.py')
else:
    bfun = Lfun.module_from_file('budget_functions', pth / 'budget_functions.py')

which_vol = 'Penn Cove'

dir0 = Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'tef2'

# ===== 1. Get HOURLY qnet from the avg-based section extraction (BEFORE bulk_calc) =====
# This is the raw hourly total transport through the section from Huon/Hvom
in_dir = dir0 / ('extractions_avg_' + Ldir['ds0'] + '_' + Ldir['ds1'])

sntup_list, sect_base_list, outer_sns_list = bfun.get_sntup_list(sect_gctag, which_vol)

print('Section tuples:', sntup_list)

# Load section NetCDF and compute hourly qnet
ocn_hourly_list = []
time_avg = None
for tup in sntup_list:
    sn = tup[0]
    sgn = tup[1]
    ds_sect = xr.open_dataset(in_dir / (sn + '.nc'))
    q = ds_sect['q'].values  # (time, z, p) — volume flux from Huon/Hvom
    # qnet for each hour = sum over all z and p
    qnet_h = np.nansum(q.reshape(q.shape[0], -1), axis=1) * sgn
    ocn_hourly_list.append(qnet_h)
    if time_avg is None:
        time_avg = ds_sect['time'].values
    ds_sect.close()

ocn_hourly = np.sum(ocn_hourly_list, axis=0)
print(f'ocn_hourly shape: {ocn_hourly.shape}')
print(f'ocn_hourly time range: {time_avg[0]} to {time_avg[-1]}')

# ===== 2. Get HOURLY volume from history-based segment extraction =====
seg_ds_fn = dir0 / ('segments' + date_str + '_' + sect_gctag + '_' + Ldir['riv'] + '.nc')
seg_ds = xr.open_dataset(seg_ds_fn)

# get segment info
dir2 = Ldir['LOo'] / 'extract' / 'tef2'
seg_info_dict_fn = dir2 / ('seg_info_dict_' + sect_gctag + '_' + Ldir['riv'] + '.p')
seg_info_dict = pd.read_pickle(seg_info_dict_fn)

sect_df_fn = dir2 / ('sect_df_' + sect_gctag + '.p')
sect_df = pd.read_pickle(sect_df_fn)
sn_list = list(sect_df.sn)

# find valid segments for this volume
sns_list = []
for snb in sect_base_list:
    for sn in sn_list:
        if snb in sn:
            for pm in ['_p','_m']:
                sns = sn + pm
                if (sns not in outer_sns_list) and (sns not in sns_list):
                    sns_list.append(sns)

good_seg_key_list = []
for sk in seg_info_dict.keys():
    this_sns_list = seg_info_dict[sk]['sns_list']
    check_list = [item for item in this_sns_list if item in sns_list]
    if len(check_list) >= 1:
        good_seg_key_list.append(sk)

print(f'\nSegments in volume: {good_seg_key_list}')
print(f'seg_ds time steps: {len(seg_ds.time)}')

# compute total volume at each hourly snapshot
vol_h = np.zeros(len(seg_ds.time))
for sk in good_seg_key_list:
    this_ds = seg_ds.sel(seg=sk)
    vol_h += this_ds.volume.values

time_his = seg_ds.time.values
print(f'History time range: {time_his[0]} to {time_his[-1]}')

# Forward difference: dV/dt
d_dt_hourly = (vol_h[1:] - vol_h[:-1]) / 3600  # m3/s

seg_ds.close()

# ===== 3. Print sizes and time alignment =====
print(f'\n--- Time alignment check ---')
print(f'd_dt_hourly (from history forward diff): {len(d_dt_hourly)} values')
print(f'ocn_hourly (from avg sections): {len(ocn_hourly)} values')
print(f'History time[0]: {time_his[0]}')
print(f'History time[1]: {time_his[1]}')
print(f'Avg time[0]: {time_avg[0]}')
print(f'Avg time[1]: {time_avg[1]}')

# ===== 4. Align and compare =====
# Truncate to same length
N = min(len(d_dt_hourly), len(ocn_hourly))
d_dt_h = d_dt_hourly[:N]
ocn_h = ocn_hourly[:N]
err_h = d_dt_h - ocn_h

print(f'\n--- Raw hourly budget (first {N} hours) ---')
print(f'd_dt  mean: {np.nanmean(d_dt_h):.4f},  std: {np.nanstd(d_dt_h):.4f}')
print(f'ocn   mean: {np.nanmean(ocn_h):.4f},  std: {np.nanstd(ocn_h):.4f}')
print(f'err   mean: {np.nanmean(err_h):.4f},  std: {np.nanstd(err_h):.4f}')
print(f'|err|/|ocn|: {np.nanmean(np.abs(err_h))/np.nanmean(np.abs(ocn_h)):.4f}')

# ===== 5. Plot =====
pfun.start_plot(figsize=(14,10))

fig, axes = plt.subplots(3, 1, sharex=True)

# Use avg time axis for plotting (truncated)
t = time_avg[:N]

ax = axes[0]
ax.plot(t, d_dt_h, 'b-', alpha=0.5, label='d_dt (history)')
ax.plot(t, ocn_h, 'g-', alpha=0.5, label='ocn (avg Huon/Hvom)')
ax.legend()
ax.set_ylabel('m3/s')
ax.set_title('Raw hourly: volume budget for ' + which_vol)
ax.grid(True)

ax = axes[1]
ax.plot(t, err_h, 'purple', alpha=0.7, label='err = d_dt - ocn (hourly)')
ax.axhline(0, color='k', linewidth=0.5)
ax.legend()
ax.set_ylabel('m3/s')
ax.set_title('Hourly error (before Godin filter)')
ax.grid(True)

ax = axes[2]
# also show Godin-filtered comparison
pad = 36
if N > 2*pad + 24:
    d_dt_lp = zfun.lowpass(d_dt_h, f='godin')[pad:-pad+1:24]
    ocn_lp = zfun.lowpass(ocn_h, f='godin')[pad:-pad+1:24]
    err_lp = d_dt_lp - ocn_lp
    t_lp = t[pad:-pad+1:24]
    ax.plot(t_lp, d_dt_lp, 'b-o', label='d_dt (Godin)')
    ax.plot(t_lp, ocn_lp, 'g-o', label='ocn (Godin)')
    ax.plot(t_lp, err_lp, 'purple', marker='s', label='err (Godin)')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.legend()
    ax.set_title('After Godin filter (daily)')
else:
    ax.set_title('(Not enough data for Godin filter)')
ax.set_ylabel('m3/s')
ax.grid(True)

plt.tight_layout()
plt.show()
pfun.end_plot()
