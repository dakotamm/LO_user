"""
Calculate air-sea heat flux components for the Penn Cove (wb1_pc0) domain
using wb1 ROMS output.

Adapted from Jilian Xiong's get_TNTempVol_Denitri_AirSeaHeat_1.py
(cas7_t0_x4b / sog6_m domain).

Key changes from original:
  - Grid/paths: wb1_r0_xn11ab via Ldir, output to LOo/extract/.../tef2/
  - Uses hourly avg files (ocean_avg_NNNN.nc) via get_avg_fn_list
  - Domain: Penn Cove segments from seg_info_dict_wb1_pc0_riv00.p

Budget terms computed (all integrated over Penn Cove surface area):
  - shflux_sum:   net surface heat flux [W = J/s]
  - latent_sum:   latent heat flux [W]
  - sensible_sum: sensible heat flux [W]
  - lwrad_sum:    longwave radiation flux [W]
  - swrad_sum:    shortwave radiation flux [W]
  - temp_vol_sum: temperature × volume [degC m3]

Note: tracer_budget_avg.py already computes the temp budget (d_dt, ocn, riv)
and includes shflux as the surface term. This script provides the breakdown
of shflux into its components (latent, sensible, lwrad, swrad).

Run on apogee:
  python get_heat_air_sea_wb1.py
"""

import xarray as xr
import numpy as np
from lo_tools import zfun, zrfun, Lfun
from datetime import datetime, timedelta
import pickle, sys
import pandas as pd
from pathlib import Path
from time import time as Time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tef2_avg_fun import get_avg_fn_list

tt0 = Time()

# ============ USER SETTINGS ============
gtagex = 'wb1_r0_xn11ab'
ds0 = '2017.01.01'
ds1 = '2017.01.21'
roms_out_key = 'roms_out2'
sect_gctag = 'wb1_pc0'
riv_tag = 'riv00'
# =======================================

Ldir = Lfun.Lstart()
Ldir['roms_out'] = Ldir[roms_out_key]
Ldir['gtagex'] = gtagex
Ldir['ds0'] = ds0

in_dir = Ldir['roms_out'] / Ldir['gtagex']

out_dir = Ldir['LOo'] / 'extract' / gtagex / 'tef2'
out_dir.mkdir(parents=True, exist_ok=True)
out_fn = out_dir / ('heat_air_sea_' + ds0 + '_' + ds1 + '_' + sect_gctag + '.nc')
print('Output will be saved to: ' + str(out_fn))

fn_list_all = get_avg_fn_list(Ldir, ds0, ds1)
print(f'Number of avg files: {len(fn_list_all)}')

G, S, T = zrfun.get_basic_info(fn_list_all[0])

fn0 = xr.open_dataset(fn_list_all[0])
dx = 1 / fn0.pm.values
dy = 1 / fn0.pn.values
area = dx * dy
NX, NY = dx.shape
fn0.close()

# ============ DOMAIN DEFINITION ============
seg_info_dict_fn = Ldir['LOo'] / 'extract' / 'tef2' / ('seg_info_dict_' + sect_gctag + '_' + riv_tag + '.p')
seg_info_dict = pd.read_pickle(seg_info_dict_fn)

sect_df_fn = Ldir['LOo'] / 'extract' / 'tef2' / ('sect_df_' + sect_gctag + '.p')
sect_df = pd.read_pickle(sect_df_fn)
sn_list = list(sect_df.sn)

upth = Ldir['LOu'] / 'extract' / 'tef2'
bfun = Lfun.module_from_file('budget_functions', upth / 'budget_functions.py')
sntup_list, sect_base_list, outer_sns_list = bfun.get_sntup_list(sect_gctag, 'Penn Cove')

sns_list = []
for snb in sect_base_list:
    for sn in sn_list:
        if snb in sn:
            for pm in ['_p', '_m']:
                sns = sn + pm
                if (sns not in outer_sns_list) and (sns not in sns_list):
                    sns_list.append(sns)

good_seg_key_list = []
for sk in seg_info_dict.keys():
    this_sns_list = seg_info_dict[sk]['sns_list']
    check_list = [item for item in this_sns_list if item in sns_list]
    if len(check_list) >= 1:
        good_seg_key_list.append(sk)

print('Penn Cove segments: ' + str(good_seg_key_list))

jj_all = []
ii_all = []
for sk in good_seg_key_list:
    ji_list = seg_info_dict[sk]['ji_list']
    for ji in ji_list:
        jj_all.append(ji[0])
        ii_all.append(ji[1])
jj = jj_all
ii = ii_all

inDomain = np.zeros([NX, NY])
inDomain[jj, ii] = 1
print(f'Penn Cove domain: {len(jj)} grid cells')

# ============ INITIALIZATION ============
shflux_sum = []
latent_sum = []
sensible_sum = []
lwrad_sum = []
swrad_sum = []
temp_vol_sum = []
t = []

shflux_spatial = np.zeros((24 * 32, NX, NY))
latent_spatial = np.zeros((24 * 32, NX, NY))
sensible_spatial = np.zeros((24 * 32, NX, NY))
lwrad_spatial = np.zeros((24 * 32, NX, NY))
swrad_spatial = np.zeros((24 * 32, NX, NY))

# ============ TIME LOOP ============
cnt = 0
N = S['N']
Nfiles = len(fn_list_all)

for fi, fn in enumerate(fn_list_all):
    if fi % 24 == 0:
        print(f'Processing file {fi+1}/{Nfiles}: {fn.name} ({fn.parent.name})')
        sys.stdout.flush()

    ds = xr.open_dataset(fn)

    h = ds.h.values
    zeta = ds.zeta.values.squeeze()
    z_w = zrfun.get_z(h, zeta, S, only_rho=False, only_w=True)
    vol = np.diff(z_w, axis=0) * area

    # Temperature × volume
    temp = ds.temp.values.squeeze()
    temp_vol_sum.append(np.nansum(temp[:, jj, ii] * vol[:, jj, ii]))

    # Air-sea heat flux components [W/m2] → integrate over area → [W]
    shflux = ds.shflux.values.squeeze()
    latent = ds.latent.values.squeeze()
    sensible = ds.sensible.values.squeeze()
    lwrad = ds.lwrad.values.squeeze()
    swrad = ds.swrad.values.squeeze()

    shflux_sum.append(np.nansum(area[jj, ii] * shflux[jj, ii]))
    latent_sum.append(np.nansum(area[jj, ii] * latent[jj, ii]))
    sensible_sum.append(np.nansum(area[jj, ii] * sensible[jj, ii]))
    lwrad_sum.append(np.nansum(area[jj, ii] * lwrad[jj, ii]))
    swrad_sum.append(np.nansum(area[jj, ii] * swrad[jj, ii]))

    # Spatial fields
    shflux_spatial[cnt, :, :] = shflux * inDomain
    latent_spatial[cnt, :, :] = latent * inDomain
    sensible_spatial[cnt, :, :] = sensible * inDomain
    lwrad_spatial[cnt, :, :] = lwrad * inDomain
    swrad_spatial[cnt, :, :] = swrad * inDomain

    cnt += 1
    t.append(ds.ocean_time.values)
    ds.close()

print('Processing complete. Total time steps: %d' % cnt)
print('Total time = %0.1f sec' % (Time() - tt0))

# ============ SAVE TO NETCDF ============
from netCDF4 import Dataset

nc = Dataset(str(out_fn), 'w')
time_dim = nc.createDimension('time', len(t))
eta_rho = nc.createDimension('eta_rho', NX)
xi_rho = nc.createDimension('xi_rho', NY)

nc.description = 'Air-sea heat flux components for Penn Cove (wb1_pc0)'
nc.source = 'Adapted from Jilian Xiong get_TNTempVol_Denitri_AirSeaHeat_1.py'
nc.gtagex = gtagex
nc.domain = sect_gctag
nc.date_range = ds0 + ' to ' + ds1
nc.segments = str(good_seg_key_list)

times = nc.createVariable('time', 'f8', ('time',))
times.units = 'seconds*1e9 since 1970-01-01 00:00:00'
times[:] = t

vars_1d = {
    'shflux_sum':   ('W', shflux_sum,   'net surface heat flux'),
    'latent_sum':   ('W', latent_sum,    'latent heat flux'),
    'sensible_sum': ('W', sensible_sum,  'sensible heat flux'),
    'lwrad_sum':    ('W', lwrad_sum,     'longwave radiation flux'),
    'swrad_sum':    ('W', swrad_sum,     'shortwave radiation flux'),
    'temp_vol_sum': ('degC m3', temp_vol_sum, 'temperature * volume'),
}
for vn, (units, data, desc) in vars_1d.items():
    v = nc.createVariable(vn, 'f4', ('time',), compression='zlib', complevel=9)
    v.units = units
    v.long_name = desc
    v[:] = data

vars_2d = {
    'shflux_spatial':   ('W/m2', shflux_spatial,   'net surface heat flux (spatial)'),
    'latent_spatial':   ('W/m2', latent_spatial,    'latent heat flux (spatial)'),
    'sensible_spatial': ('W/m2', sensible_spatial,  'sensible heat flux (spatial)'),
    'lwrad_spatial':    ('W/m2', lwrad_spatial,     'longwave radiation flux (spatial)'),
    'swrad_spatial':    ('W/m2', swrad_spatial,     'shortwave radiation flux (spatial)'),
}
for vn, (units, data, desc) in vars_2d.items():
    v = nc.createVariable(vn, 'f4', ('time', 'eta_rho', 'xi_rho'), compression='zlib', complevel=9)
    v.units = units
    v.long_name = desc
    v[:] = data[0:len(t), :, :]

nc.close()
print('Saved: ' + str(out_fn))
