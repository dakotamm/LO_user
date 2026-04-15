"""
Calculate sediment nitrogen loss (denitrification + detritus burial) for the
Penn Cove (wb1_pc0) domain using wb1 ROMS output.

Adapted from Jilian Xiong's get_TNTempVol_Denitri_AirSeaHeat_1.py
(cas7_t0_x4b / sog6_m domain).

Key changes from original:
  - Grid/paths: wb1_r0_xn11ab via Ldir, output to LOo/extract/.../tef2/
  - Uses hourly avg files (ocean_avg_NNNN.nc) via get_avg_fn_list
  - Domain: Penn Cove segments from seg_info_dict_wb1_pc0_riv00.p

Budget terms computed (all integrated over Penn Cove domain):
  - TN_vol_sum:          TN × volume [mmol N] (TN = NO3+NH4+Phy+Zoo+SDetN+LDetN)
  - denitri_flux_sum:    denitrification loss of NO3 [mmol N/hr]
  - NH4_gain_flux_sum:   remineralization gain of NH4 from sediment (50% burial accounted for) [mmol N/hr]
  - detritus_loss_sum:   particulate N settling flux into sediment [mmol N/hr]

The TN budget closes as:
  d(TN*vol)/dt = ocn_transport + riv - denitri - detritus_loss + NH4_gain
  (where detritus_loss - NH4_gain = net sediment N removal)

Note: tracer_budget_avg.py does NOT compute a TN budget directly — it handles
individual tracers (salt, temp, oxygen). TN requires summing all N pools.

Run on apogee:
  python get_TN_sediment_wb1.py
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
out_fn = out_dir / ('TN_sediment_' + ds0 + '_' + ds1 + '_' + sect_gctag + '.nc')
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

# ============ BIOLOGY PARAMETERS ============
Ws_L = 80.0   # Large detritus sinking velocity [m/day]
Ws_S = 8.0    # Small detritus sinking velocity [m/day]
rOxNH4 = 106 / 16  # O2:N ratio for remineralization
burials = 50  # Burial fraction [%] — only (1-burial/100) is remineralized

# ============ INITIALIZATION ============
TN_vol_sum = []
denitri_flux_sum = []
NH4_gain_flux_sum = []
detritus_loss_sum = []
t = []

denitri_spatial = np.zeros((24 * 32, NX, NY))
NH4_gain_spatial = np.zeros((24 * 32, NX, NY))
detritus_loss_spatial = np.zeros((24 * 32, NX, NY))

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
    dz = np.diff(z_w, axis=0)
    vol = dz * area

    # Read all N pools
    Phyt = ds.phytoplankton.values.squeeze()
    Zoop = ds.zooplankton.values.squeeze()
    NO3 = ds.NO3.values.squeeze()
    NH4 = ds.NH4.values.squeeze()
    LDetN = ds.LdetritusN.values.squeeze()
    SDetN = ds.SdetritusN.values.squeeze()
    Oxy = ds.oxygen.values.squeeze()

    # TN = sum of all nitrogen pools
    TN = Phyt + Zoop + NO3 + NH4 + LDetN + SDetN
    TN_vol_sum.append(np.nansum(TN[:, jj, ii] * vol[:, jj, ii]))

    # ---------- SEDIMENT PROCESSES ----------
    # Bottom layer values
    LDetN_bot = LDetN[0, :, :]
    SDetN_bot = SDetN[0, :, :]
    Oxy_bot = Oxy[0, :, :]
    NO3_bot = NO3[0, :, :]

    NO3loss = 1.2 * (1 / 24) / dz[0, :, :]  # mmol N/m3/hr (cap from fennel.h)

    # --- Small detritus ---
    # Account for burial: only (1-burial/100) fraction is remineralized in sediment
    FC_S = (SDetN_bot * Ws_S) / 24  # settling flux [mmol N/m2/hr]
    cff1_S = (SDetN_bot * (1 - burials / 100) * Ws_S) / 24 / dz[0, :, :] * 1  # concentration change [mmol N/m3]
    denitri_S = np.zeros([NX, NY])
    NH4_gain_S = np.zeros([NX, NY])
    for i in range(NX):
        for j in range(NY):
            if cff1_S[i, j] * rOxNH4 > Oxy_bot[i, j]:
                # O2 insufficient: denitrification
                denitri_S[i, j] = min(NO3_bot[i, j], cff1_S[i, j])
            else:
                # O2 sufficient: remineralization to NH4
                NH4_gain_S[i, j] = cff1_S[i, j]
                if cff1_S[i, j] > NO3loss[i, j]:
                    denitri_S[i, j] = min(NO3_bot[i, j], NO3loss[i, j])
    denitri_flux_S = denitri_S * area * dz[0, :, :]
    NH4_gain_flux_S = NH4_gain_S * area * dz[0, :, :]

    # --- Large detritus (after small detritus has modified bottom O2 and NO3) ---
    # Account for burial: only (1-burial/100) fraction is remineralized in sediment
    Oxy_bot2 = Oxy_bot - NH4_gain_S * rOxNH4
    NO3_bot2 = NO3_bot - denitri_S
    FC_L = (LDetN_bot * Ws_L) / 24
    cff1_L = (LDetN_bot * (1 - burials / 100) * Ws_L) / 24 / dz[0, :, :] * 1
    denitri_L = np.zeros([NX, NY])
    NH4_gain_L = np.zeros([NX, NY])
    for i in range(NX):
        for j in range(NY):
            if cff1_L[i, j] * rOxNH4 > Oxy_bot2[i, j]:
                denitri_L[i, j] = min(NO3_bot2[i, j], cff1_L[i, j])
            else:
                NH4_gain_L[i, j] = cff1_L[i, j]
                if cff1_L[i, j] > NO3loss[i, j]:
                    denitri_L[i, j] = min(NO3_bot2[i, j], NO3loss[i, j])
    denitri_flux_L = denitri_L * area * dz[0, :, :]
    NH4_gain_flux_L = NH4_gain_L * area * dz[0, :, :]

    # Totals
    denitri_flux = denitri_flux_S + denitri_flux_L
    NH4_gain_flux = NH4_gain_flux_S + NH4_gain_flux_L

    denitri_flux_sum.append(np.nansum(denitri_flux[jj, ii]))
    NH4_gain_flux_sum.append(np.nansum(NH4_gain_flux[jj, ii]))
    detritus_loss_sum.append(np.nansum((FC_L[jj, ii] + FC_S[jj, ii]) * area[jj, ii]))

    # Spatial fields
    denitri_spatial[cnt, :, :] = denitri_flux * inDomain
    NH4_gain_spatial[cnt, :, :] = NH4_gain_flux * inDomain
    detritus_loss_spatial[cnt, :, :] = (FC_L + FC_S) * area * inDomain

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

nc.description = 'TN sediment budget terms for Penn Cove (wb1_pc0)'
nc.source = 'Adapted from Jilian Xiong get_TNTempVol_Denitri_AirSeaHeat_1.py'
nc.gtagex = gtagex
nc.domain = sect_gctag
nc.date_range = ds0 + ' to ' + ds1
nc.segments = str(good_seg_key_list)

times = nc.createVariable('time', 'f8', ('time',))
times.units = 'seconds*1e9 since 1970-01-01 00:00:00'
times[:] = t

vars_1d = {
    'TN_vol_sum':         ('mmol N',     TN_vol_sum,         'total nitrogen * volume'),
    'denitri_flux_sum':   ('mmol N/hr',  denitri_flux_sum,   'denitrification loss of NO3'),
    'NH4_gain_flux_sum':  ('mmol N/hr',  NH4_gain_flux_sum,  'sediment remineralization gain of NH4'),
    'detritus_loss_sum':  ('mmol N/hr',  detritus_loss_sum,  'particulate N settling flux into sediment'),
}
for vn, (units, data, desc) in vars_1d.items():
    v = nc.createVariable(vn, 'f4', ('time',), compression='zlib', complevel=9)
    v.units = units
    v.long_name = desc
    v[:] = data

vars_2d = {
    'denitri_spatial':       ('mmol N/hr', denitri_spatial,       'denitrification (spatial)'),
    'NH4_gain_spatial':      ('mmol N/hr', NH4_gain_spatial,      'sediment NH4 gain (spatial)'),
    'detritus_loss_spatial': ('mmol N/hr', detritus_loss_spatial, 'particulate N settling (spatial)'),
}
for vn, (units, data, desc) in vars_2d.items():
    v = nc.createVariable(vn, 'f4', ('time', 'eta_rho', 'xi_rho'), compression='zlib', complevel=9)
    v.units = units
    v.long_name = desc
    v[:] = data[0:len(t), :, :]

nc.close()
print('Saved: ' + str(out_fn))
