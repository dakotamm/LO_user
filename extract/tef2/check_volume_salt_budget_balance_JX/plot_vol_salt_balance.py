"""
Check volume and salt balance
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from lo_tools import zrfun, zfun
import pandas as pd
import pickle
import get_two_layer # this was copied from /LO/extract/tef2/tef_fun.py
import scipy
from time import time
from datetime import datetime
from pathlib import Path
from lo_tools import plotting_functions as pfun

# adapting code from JX LO_user starting 20260305


# exchange flow
# at jdf1
in_dir=Path('/Users/dakotamascarenas/LO_output/extract/wb1_r0_xn11b/tef2/bulk_2017.09.01_2017.09.30/pc0.nc') # DM - I think this is bulk output
bulk = xr.open_dataset(in_dir)
tef_df, vn_list, vec_list = get_two_layer.get_two_layer(bulk)

# %%
salt_p_jdf1 = tef_df['salt_p'] # 
salt_m_jdf1 = tef_df['salt_m'] # 
Q_p_jdf1 = tef_df['q_p'] # Qout
Q_m_jdf1 = tef_df['q_m'] # Qin
sf_p_jdf1 = salt_p_jdf1 * Q_p_jdf1 # salt flux
sf_m_jdf1 = salt_m_jdf1 * Q_m_jdf1

#%%-----------------------------
# # at sog6
# in_dir=Path('./data/sog6_2017.nc')
# bulk = xr.open_dataset(in_dir)
# tef_df, vn_list, vec_list = get_two_layer.get_two_layer(bulk)
# salt_p_sog6 = tef_df['salt_p']
# salt_m_sog6 = tef_df['salt_m']
# Q_p_sog6 = tef_df['q_p'] # Qout
# Q_m_sog6 = tef_df['q_m'] # Qin

# sf_p_sog6 = salt_p_sog6 * Q_p_sog6 # salt flux
# sf_m_sog6 = salt_m_sog6 * Q_m_sog6

# add jdf1 and sog6 together
# Qin  = Q_m_jdf1 + Q_m_sog6
# Qout = Q_p_jdf1 + Q_p_sog6
# Fin  = sf_m_jdf1 + sf_m_sog6
# Fout = sf_p_jdf1 + sf_p_sog6
# %%
Qin = Q_m_jdf1
Qout = Q_p_jdf1
Fin = sf_m_jdf1
Fout = sf_p_jdf1

ot = bulk['time']

#%% river discharge
seg_name = '/Users/dakotamascarenas/LO_output/extract/tef2/seg_info_dict_wb1_pc0_riv00.p'
seg_df = pd.read_pickle(seg_name)
ji_list = seg_df['pc0_m']['ji_list']
jj = [x[0] for x in ji_list]
ii = [x[1] for x in ji_list]

# all rivers inside the segment - Salish
riv_list = seg_df['pc0_m']['riv_list']

# all river names in LO domain
fn = '/Users/dakotamascarenas/LO_output/forcing/wb1/f2017.07.04/riv00/rivers.nc'
ds = xr.open_dataset(fn)
river_name_all = ds.river_name.values

fn_riv = '/Users/dakotamascarenas/LO_output/pre/river1/wb1_riv00/Data_roms/extraction_2017.09.01_2017.09.30.nc'
ds_riv = xr.open_dataset(fn_riv)
t_riv = ds_riv.time.values # daily
Q_riv = ds_riv.transport.values

Q_riv_tmp = dict() # rivers in a specific domain
for riv in riv_list:
    ix = np.where(riv==river_name_all)[0]
    Q_riv_tmp[riv] = Q_riv[:, ix[0]]
    
Q_riv_sum = np.zeros(Q_riv.shape[0])      
for riv in riv_list:
    Q_riv_sum += Q_riv_tmp[riv]

#get 2017 river discharge
t_riv_2017 = t_riv[1:-1]
Q_riv_2017 = Q_riv_sum[1:-1]

#%% check volume balance
dir0 = Path('/Users/dakotamascarenas/LO_output/extract/wb1_r0_xn11b/tef2/'); #hourly volume, S*V, and EminusP
fn_list = sorted(dir0.glob('vol_SV_hrly_2017.09.01_2017.09.30_wb1_r0_xn11b.p'))
t1=[] 
vol_hrly = []
salt_vol_sum_hrly = []
surf_s_flux = []

for fn in fn_list:
    tmp = pickle.load(open(fn, 'rb'))
    t1 += tmp['t']
    vol_hrly += tmp['vol_hrly']
    salt_vol_sum_hrly += tmp['salt_vol_sum_hrly']
    surf_s_flux += tmp['surf_s_flux']

vol_hrly = np.array(vol_hrly)
dVdt = np.diff(vol_hrly)/3600
#dVdt_noon = dVdt[36:-34:24]
dVdt_lp = zfun.lowpass(dVdt, f='godin')[36:-30:24]
vol_lp  = zfun.lowpass(vol_hrly, f='godin')[36:-34:24]
t1 = np.array(t1)#[:,0]
t1_noon = t1[36:-34:24][:,0]

error = -Qin-Qout + Q_riv_2017 - dVdt_lp #dVdt_noon #dVdt_lp

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(211)
plt.plot(-Qin-Qout+Q_riv_2017, c='k', lw=2, label='Q$_{in}$+Q$_{out}$+Q$_r$')
plt.plot(t1_noon, dVdt_lp, c='r', label='storage', lw=1)
plt.plot(t1_noon, error, 'gray', lw=2, label='Error')
plt.legend(ncol=3, loc='upper center')
plt.ylabel('m$^3$ s$^{-1}$', fontsize=12)
#plt.xticks(fontsize=12); plt.yticks(fontsize=12)
#plt.ylim(-100,100)

#plt.plot(t1[:-1,0], dVdt)

plt.plot(t1[:,0], vol_hrly- vol_hrly.mean())


plt.plot(t1_noon, -Qin-Qout, color = 'm')
#plt.xlim(datetime(2017,1,1), datetime(2018,1,1))
plt.grid(which='major')
plt.title('Volume budget')

#%% check salt balance
S_vol = np.array(salt_vol_sum_hrly)  # sum(S*V)
dSvol_dt = np.diff(S_vol)/3600  # d(SV)/dt
dSvol_dt_lp = zfun.lowpass(dSvol_dt, f='godin')[36:-34:24]

surf_s_flux_lp = zfun.lowpass(np.array(surf_s_flux), f='godin')[36:-34:24]  # EminusP
# negative: upward flux, freshening (net precipitation)
# positive: downward flux, salting (net evaporation)

error = -Fin-Fout - dSvol_dt_lp
ax = fig.add_subplot(212)
plt.plot(-Fin-Fout+surf_s_flux_lp, c='k', lw=2, label='F$_{in}$+F$_{out}$+EminusP')
plt.plot(t1_noon, dSvol_dt_lp, c='r', label='storage', lw=1)
plt.plot(t1_noon, error, 'gray', lw=2, label='Error')
plt.legend(ncol=3, loc='upper center')
plt.ylabel('g kg$^{-1}$ m$^3$ s$^{-1}$', fontsize=12)
#plt.xticks(fontsize=12); plt.yticks(fontsize=12)

#plt.xlim(datetime(2017,1,1), datetime(2018,1,1))
plt.grid(which='major')
plt.title('Salt budget')

#plt.savefig('Vol_salt_budget_balance_2017',dpi=300, bbox_inches='tight')
