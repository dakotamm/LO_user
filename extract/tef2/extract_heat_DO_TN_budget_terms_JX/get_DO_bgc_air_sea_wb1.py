"""
Calculate O2 production and consumption from BGC processes, and air-sea oxygen exchange,
for the Penn Cove (wb1_pc0) domain using wb1 ROMS output.

Adapted from Jilian Xiong's get_DO_bgc_air_sea_1.py (cas7_t0_x4b / sog6_m domain).

Key changes from original:
  - Grid/paths: wb1_r0_xn11ab via Ldir, output to LOo/extract/.../tef2/
  - Uses hourly avg files (ocean_avg_NNNN.nc) via get_avg_fn_list,
    consistent with the tef2 avg pipeline (extract_segments_avg.py, etc.)
  - Domain: Penn Cove segments from seg_info_dict_wb1_pc0_riv00.p
    (uses same segment selection logic as tracer_budget_avg.py)
  - AttSW: uniform 0.05 m-1 (matches wb1 bio_Fennel.in; original used 0.15 inside Salish Sea)
  - Vp: temperature-dependent Eppley curve Vp = Vp0 * 1.066^temp (matches ROMS fennel.h;
    original hardcoded Vp = 1.7)

Budget terms computed (all integrated over Penn Cove volume):
  - Oxy_pro_sum:      O2 production from photosynthesis [mmol O2/hr]
  - Oxy_nitri_sum:    O2 consumption by nitrification [mmol O2/hr]
  - Oxy_remi_sum:     O2 consumption by water column remineralization [mmol O2/hr]
  - Oxy_sed_sum:      SOD method 1 (Parker's benthic_flux.py approach) [mmol O2/day]
  - Oxy_sed_sum2:     SOD method 2 (fennel.h O2-limited approach) [mmol O2/hr]
  - Oxy_vol_sum:      DO × volume [mmol O2]
  - Oxy_air_flux_sum: Air-sea O2 exchange [mmol O2/hr]

Run on apogee:
  python get_DO_bgc_air_sea_wb1.py

"""

import xarray as xr
import numpy as np
from lo_tools import zfun, zrfun, Lfun
from datetime import datetime, timedelta
import pickle, sys
import pandas as pd
from pathlib import Path
from time import time as Time

# Import the avg file list utility (same as used by extract_segments_avg.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # add extract/tef2 to path
from tef2_avg_fun import get_avg_fn_list

tt0 = Time()

# ============ USER SETTINGS ============

gtagex = 'wb1_r0_xn11ab'  # grid-tag-experiment
ds0 = '2017.01.01'        # start date
ds1 = '2017.01.21'        # end date

# Which roms_out key to use (set in get_lo_info.py per machine)
# On apogee: roms_out2 = '/dat1/dakotamm/LO_roms'
roms_out_key = 'roms_out2'

# Segment info for Penn Cove domain
sect_gctag = 'wb1_pc0'
riv_tag = 'riv00'

# =======================================

# Set up paths using LO tools
Ldir = Lfun.Lstart()
Ldir['roms_out'] = Ldir[roms_out_key]
Ldir['gtagex'] = gtagex
Ldir['ds0'] = ds0

in_dir = Ldir['roms_out'] / Ldir['gtagex']

# Output path: LOo/extract/gtagex/tef2/
out_dir = Ldir['LOo'] / 'extract' / gtagex / 'tef2'
out_dir.mkdir(parents=True, exist_ok=True)
out_fn = out_dir / ('O2_bgc_' + ds0 + '_' + ds1 + '_' + sect_gctag + '.nc')
print('Output will be saved to: ' + str(out_fn))

# Build the list of hourly avg files (ocean_avg_0001.nc through ocean_avg_0024.nc per day)
fn_list_all = get_avg_fn_list(Ldir, ds0, ds1)
print(f'Number of avg files: {len(fn_list_all)}')
print(f'First: {fn_list_all[0]}')
print(f'Last: {fn_list_all[-1]}')

# Get grid info from first avg file
G, S, T = zrfun.get_basic_info(fn_list_all[0])

fn0 = xr.open_dataset(fn_list_all[0])
dx = 1 / fn0.pm.values
dy = 1 / fn0.pn.values
area = dx * dy
NX, NY = dx.shape
fn0.close()

# ============ DOMAIN DEFINITION ============
# Use Penn Cove segments from seg_info_dict, with the same segment selection
# logic as tracer_budget_avg.py (budget_functions.py -> get_sntup_list)

# Load segment info
seg_info_dict_fn = Ldir['LOo'] / 'extract' / 'tef2' / ('seg_info_dict_' + sect_gctag + '_' + riv_tag + '.p')
seg_info_dict = pd.read_pickle(seg_info_dict_fn)

# Load section dataframe to find all valid section names
sect_df_fn = Ldir['LOo'] / 'extract' / 'tef2' / ('sect_df_' + sect_gctag + '.p')
sect_df = pd.read_pickle(sect_df_fn)
sn_list = list(sect_df.sn)

# Get budget_functions to determine which segments are in the Penn Cove volume
# (replicate the logic from tracer_budget_avg.py)
upth = Ldir['LOu'] / 'extract' / 'tef2'
bfun = Lfun.module_from_file('budget_functions', upth / 'budget_functions.py')
sntup_list, sect_base_list, outer_sns_list = bfun.get_sntup_list(sect_gctag, 'Penn Cove')

# Find valid segment-name-signs (sns)
sns_list = []
for snb in sect_base_list:
    for sn in sn_list:
        if snb in sn:
            for pm in ['_p', '_m']:
                sns = sn + pm
                if (sns not in outer_sns_list) and (sns not in sns_list):
                    sns_list.append(sns)

# Find which segment keys belong to Penn Cove volume
good_seg_key_list = []
for sk in seg_info_dict.keys():
    this_sns_list = seg_info_dict[sk]['sns_list']
    check_list = [item for item in this_sns_list if item in sns_list]
    if len(check_list) >= 1:
        good_seg_key_list.append(sk)

print('Penn Cove segments: ' + str(good_seg_key_list))

# Combine all (j,i) indices from the Penn Cove segments
jj_all = []
ii_all = []
for sk in good_seg_key_list:
    ji_list = seg_info_dict[sk]['ji_list']
    for ji in ji_list:
        jj_all.append(ji[0])
        ii_all.append(ji[1])
jj = jj_all
ii = ii_all

# Build inDomain mask for spatial output
inDomain = np.zeros([NX, NY])
inDomain[jj, ii] = 1
print(f'Penn Cove domain: {len(jj)} grid cells')

# ============ LIGHT ATTENUATION ============
# wb1 uses uniform AttSW = 0.05 m-1 (from bio_Fennel_BLANK.in)
# Jilian's cas7 code used 0.15 inside Salish Sea — NOT applicable to wb1
AttSW = 0.05  # [1/m], uniform for wb1

# ============ BIOLOGY PARAMETERS ============
# All from wb1 bio_Fennel_BLANK.in (confirmed identical to Jilian's values)
AttChl = 0.012   # [(mg_Chl m-3)-1 m-1]
PhyIS = 0.07     # initial slope of P-I curve [1/(Watts m-2 day)]
Vp0 = 1.0        # Eppley base parameter (ROMS uses Vp = Vp0 * 1.066^temp)
rOxNO3 = 138/16  # O2:N for new production
rOxNH4 = 106/16  # O2:N for regenerated production
K_NO3 = 10.0     # [1/(millimole_N m-3)]
K_NH4 = 10.0     # [1/(millimole_N m-3)]
SDeRRN = 0.1     # Small detritus remineralization rate [1/day]
LDeRRN = 0.1     # Large detritus remineralization rate [1/day]
NitriR = 0.05    # Nitrification rate [1/day]
Ws_L = 80.0      # Large detritus sinking velocity [m/day]
Ws_S = 8.0       # Small detritus sinking velocity [m/day]
PARfrac = 0.43   # Fraction of shortwave that is PAR

# ============ AIR-SEA FLUX PARAMETERS ============
dtdays = 3600 * (1.0 / 86400) / 1  # time step in days (1 hr history file, BioIter=1)

# Transfer velocity coefficient (Wanninkhof 1992)
cff2_air = dtdays * 0.31 * 24 / 100

# Schmidt number coefficients (Wanninkhof 1992)
A_O2 = 1953.4;   B_O2 = 128.0;  C_O2 = 3.9918
D_O2 = 0.050091; E_O2 = 0.0

# O2 saturation coefficients (Garcia and Gordon / L&O 1992)
OA0 = 2.00907
OA1 = 3.22014;    OA2 = 4.05010;    OA3 = 4.94457
OA4 = -0.256847;  OA5 = 3.88767;    OB0 = -0.00624523
OB1 = -0.00737614; OB2 = -0.0103410; OB3 = -0.00817083
OC0 = -0.000000488682

# ============ INITIALIZATION ============
# Domain-integrated time series
Oxy_sed_sum = []       # SOD method 1
Oxy_sed_sum2 = []      # SOD method 2
Oxy_pro_sum = []       # photosynthesis production
Oxy_nitri_sum = []     # nitrification consumption
Oxy_remi_sum = []      # remineralization consumption
Oxy_vol_sum = []       # DO * volume
Oxy_air_flux_sum = []  # air-sea exchange
t = []

# Spatial fields (vertically summed, saved for each hourly time step)
Oxy_pro_spatial = np.zeros((24 * 32, NX, NY))
Oxy_nitri_spatial = np.zeros((24 * 32, NX, NY))
Oxy_remi_spatial = np.zeros((24 * 32, NX, NY))
Oxy_sed_spatial = np.zeros((24 * 32, NX, NY))
Oxy_sed_spatial2 = np.zeros((24 * 32, NX, NY))
Oxy_air_flux_spatial = np.zeros((24 * 32, NX, NY))
diff_O2_spatial = np.zeros((24 * 32, NX, NY))

# ============ TIME LOOP ============
cnt = 0
N = S['N']  # number of vertical levels (30 for wb1)
Nfiles = len(fn_list_all)

for fi, fn in enumerate(fn_list_all):
    if fi % 24 == 0:
        print(f'Processing file {fi+1}/{Nfiles}: {fn.name} ({fn.parent.name})')
        sys.stdout.flush()

    ds = xr.open_dataset(fn)
    swrad = ds.swrad.values.squeeze()       # W/m2
    chl = ds.chlorophyll.values.squeeze()    # mg Chl/m3
    zeta = ds.zeta.values.squeeze()
    h = ds.h.values
    temp = ds.temp.values.squeeze()          # degC (3D, for Eppley curve)
    zw = zrfun.get_z(h, zeta, S, only_w=True)
    dz = np.diff(zw, axis=0)
    salt = ds.salt.values.squeeze()
    NH4 = ds.NH4.values.squeeze()
    NO3 = ds.NO3.values.squeeze()
    phy = ds.phytoplankton.values.squeeze()
    SDeN = ds.SdetritusN.values.squeeze()
    LDeN = ds.LdetritusN.values.squeeze()
    Oxy = ds.oxygen.values.squeeze()

    # PAR at the surface
    Att = np.zeros(salt.shape)
    nk, ni, nj = Att.shape
    PAR = np.zeros([nk + 1, ni, nj])
    PAR[-1, :, :] = PARfrac * swrad

    # Initialize biological rate arrays
    Oxy_pro = np.zeros(Att.shape)     # O2 production
    Oxy_nitri = np.zeros(Att.shape)   # O2 consumption by nitrification
    Oxy_remi = np.zeros(Att.shape)    # O2 consumption by remineralization
    Oxy_sed = np.zeros([ni, nj])      # SOD

    z_w = zrfun.get_z(h, zeta, S, only_rho=False, only_w=True)
    vol = np.diff(z_w, axis=0) * area  # grid cell volume [m3]

    # DO * volume for the domain
    Oxy_vol_sum.append(np.nansum(Oxy[:, jj, ii] * vol[:, jj, ii]))

    # ---------- PHOTOSYNTHESIS & NITRIFICATION ----------
    if np.nanmin(PARfrac * swrad) > 0:  # daytime: photosynthesis occurs
        for k in np.arange(N - 1, -1, -1):  # surface to bottom
            # Light attenuation
            Att[k, :, :] = (AttSW + AttChl * chl[k, :, :] - 0.0065 * (salt[k, :, :] - 32)) \
                            * (z_w[k + 1, :, :] - z_w[k, :, :])
            # PAR averaged at cell center
            PAR[k, :, :] = PAR[k + 1, :, :] * (1 - np.exp(-Att[k, :, :])) / Att[k, :, :]

            # O2 production from photosynthesis
            fac1 = PAR[k, :, :] * PhyIS
            # Temperature-dependent Eppley growth (matches ROMS fennel.h)
            Vp = Vp0 * (1.066 ** temp[k, :, :])
            Epp = Vp / np.sqrt(Vp * Vp + fac1 * fac1)
            t_PPmax = Epp * fac1

            cff1 = NH4[k, :, :] * K_NH4; cff1[cff1 < 0] = 0
            cff2 = NO3[k, :, :] * K_NO3; cff2[cff2 < 0] = 0
            inhNH4 = 1.0 / (1.0 + cff1)

            # New production (NO3-based)
            cff4 = dtdays * t_PPmax * K_NO3 * inhNH4 / (1.0 + cff2 + 2.0 * np.sqrt(cff2)) * phy[k, :, :]
            # Regenerated production (NH4-based)
            cff5 = dtdays * t_PPmax * K_NH4 / (1.0 + cff1 + 2.0 * np.sqrt(cff1)) * phy[k, :, :]

            Oxy_pro[k, :, :] = NO3[k, :, :] * cff4 * rOxNO3 + NH4[k, :, :] * cff5 * rOxNH4

            # O2 consumption by nitrification (light-inhibited)
            fac2 = Oxy[k, :, :].copy(); fac2[fac2 < 0] = 0
            fac3 = fac2 / (3 + fac2); fac3[fac3 < 0] = 0
            Oxy_nitri[k, :, :] = 2.0 * NH4[k, :, :] * dtdays * NitriR * fac3

            # Update PAR to bottom of grid cell for next layer
            PAR[k, :, :] = PAR[k + 1, :, :] * np.exp(-Att[k, :, :])

    else:  # nighttime: no photosynthesis, nitrification at max rate
        for k in np.arange(N - 1, -1, -1):
            Oxy_nitri[k, :, :] = 2.0 * NH4[k, :, :] * dtdays * NitriR

    # ---------- REMINERALIZATION ----------
    for k in np.arange(0, N):
        Oxy_remi[k, :, :] = (SDeN[k, :, :] * dtdays * SDeRRN + LDeN[k, :, :] * dtdays * LDeRRN) * rOxNH4

    # Convert from mmol O2/m3/hr to mmol O2/hr
    Oxy_pro = Oxy_pro * vol
    Oxy_nitri = Oxy_nitri * vol
    Oxy_remi = Oxy_remi * vol

    # Sum over Penn Cove domain
    Oxy_pro_sum.append(np.nansum(Oxy_pro[:, jj, ii]))
    Oxy_nitri_sum.append(np.nansum(Oxy_nitri[:, jj, ii]))
    Oxy_remi_sum.append(np.nansum(Oxy_remi[:, jj, ii]))

    # Spatial fields (vertically summed, masked to domain)
    Oxy_pro_spatial[cnt, :, :] = np.nansum(Oxy_pro * inDomain, axis=0)
    Oxy_nitri_spatial[cnt, :, :] = np.nansum(Oxy_nitri * inDomain, axis=0)
    Oxy_remi_spatial[cnt, :, :] = np.nansum(Oxy_remi * inDomain, axis=0)

    # ---------- SEDIMENT SOD: METHOD 1 (Parker's approach) ----------
    F_Det = LDeN[0, :, :] * Ws_L + SDeN[0, :, :] * Ws_S  # mmol N/m2/d
    F_NO3 = F_Det.copy()
    F_NO3[F_Det > 1.2] = 1.2  # cap at 1.2 mmol N/m2/d
    F_NH4 = F_Det - F_NO3
    F_NH4[F_NH4 < 0] = 0

    Oxy_sed_sum.append(np.nansum(F_NH4[jj, ii] * rOxNH4 * area[jj, ii]))
    Oxy_sed_spatial[cnt, :, :] = F_NH4 * inDomain * rOxNH4 * area

    # ---------- SEDIMENT SOD: METHOD 2 (fennel.h O2-limited) ----------
    # Large detritus decomposition in sediment
    cff1_L = (LDeN[0, :, :] * Ws_L) / 24 / dz[0, :, :] * 1  # mmol/m3
    NH4_gain_L = np.zeros([NX, NY])
    for i in range(NX):
        for j in range(NY):
            if cff1_L[i, j] * rOxNH4 <= Oxy[0, :, :][i, j]:
                NH4_gain_L[i, j] = cff1_L[i, j]
    NH4_gain_flux_L = NH4_gain_L * area * dz[0, :, :]  # mmol/hr

    # Small detritus decomposition in sediment
    cff1_S = (SDeN[0, :, :] * Ws_S) / 24 / dz[0, :, :] * 1  # mmol/m3
    NH4_gain_S = np.zeros([NX, NY])
    for i in range(NX):
        for j in range(NY):
            if cff1_S[i, j] * rOxNH4 <= Oxy[0, :, :][i, j]:
                NH4_gain_S[i, j] = cff1_S[i, j]
    NH4_gain_flux_S = NH4_gain_S * area * dz[0, :, :]  # mmol/hr

    NH4_gain_flux = NH4_gain_flux_L + NH4_gain_flux_S
    Oxy_sed_sum2.append(np.nansum(NH4_gain_flux[jj, ii] * rOxNH4))
    Oxy_sed_spatial2[cnt, :, :] = NH4_gain_flux * inDomain * rOxNH4

    # ---------- AIR-SEA O2 FLUX ----------
    Uwind = ds.Uwind.values.squeeze()
    Vwind = ds.Vwind.values.squeeze()
    temp_surf = ds.temp.values[0, -1, :, :]     # surface temperature
    salt_surf = ds.salt.values[0, -1, :, :]     # surface salinity
    Oxy_surf = ds.oxygen.values[0, -1, :, :]    # surface O2

    # Wind speed squared
    u10squ = Uwind * Uwind + Vwind * Vwind

    # Schmidt number
    SchmidtN_Ox = A_O2 - temp_surf * (B_O2 - temp_surf * (C_O2 - temp_surf * (D_O2 - temp_surf * E_O2)))

    # Transfer velocity [m]
    cff3 = cff2_air * u10squ * np.sqrt(660.0 / SchmidtN_Ox)

    # O2 saturation concentration [mmol/m3]
    TS = np.log((298.15 - temp_surf) / (273.15 + temp_surf))
    AA = (OA0 + TS * (OA1 + TS * (OA2 + TS * (OA3 + TS * (OA4 + TS * OA5))))
          + salt_surf * (OB0 + TS * (OB1 + TS * (OB2 + TS * OB3)))
          + OC0 * salt_surf * salt_surf)
    O2satu = 1000.0 / 22.3916 * np.exp(AA)  # mmol/m3

    # Air-sea O2 flux [mmol O2/hr]
    Oxy_air_flux_sum.append(np.nansum(cff3[jj, ii] * (O2satu[jj, ii] - Oxy_surf[jj, ii]) * area[jj, ii]))
    Oxy_air_flux_spatial[cnt, :, :] = cff3 * (O2satu - Oxy_surf) * inDomain
    diff_O2_spatial[cnt, :, :] = (O2satu - Oxy_surf) * inDomain

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
s_rho = nc.createDimension('s_rho', N)

# Global attributes
nc.description = 'O2 budget BGC and air-sea terms for Penn Cove (wb1_pc0)'
nc.source = 'Adapted from Jilian Xiong get_DO_bgc_air_sea_1.py'
nc.gtagex = gtagex
nc.domain = sect_gctag
nc.date_range = ds0 + ' to ' + ds1
nc.segments = str(good_seg_key_list)
nc.AttSW = str(AttSW) + ' m-1 (uniform for wb1)'
nc.Vp_formulation = 'Vp = Vp0 * 1.066^temp (Vp0=' + str(Vp0) + ', temperature-dependent Eppley curve)'

# Time
times = nc.createVariable('time', 'f8', ('time',))
times.units = 'seconds*1e9 since 1970-01-01 00:00:00'
times[:] = t

# Domain-integrated time series
vars_1d = {
    'Oxy_pro_sum':      ('mmol O2/hr',  Oxy_pro_sum,      'O2 production from photosynthesis'),
    'Oxy_nitri_sum':    ('mmol O2/hr',  Oxy_nitri_sum,    'O2 consumption by nitrification'),
    'Oxy_remi_sum':     ('mmol O2/hr',  Oxy_remi_sum,     'O2 consumption by remineralization'),
    'Oxy_sed_sum':      ('mmol O2/day', Oxy_sed_sum,      'SOD method 1 (Parker benthic flux)'),
    'Oxy_sed_sum2':     ('mmol O2/hr',  Oxy_sed_sum2,     'SOD method 2 (fennel.h O2-limited)'),
    'Oxy_vol_sum':      ('mmol O2',     Oxy_vol_sum,      'DO * volume'),
    'Oxy_air_flux_sum': ('mmol O2/hr',  Oxy_air_flux_sum, 'Air-sea O2 exchange'),
}
for vn, (units, data, desc) in vars_1d.items():
    v = nc.createVariable(vn, 'f4', ('time',), compression='zlib', complevel=9)
    v.units = units
    v.long_name = desc
    v[:] = data

# Spatial fields (vertically summed)
vars_2d = {
    'Oxy_pro_spatial':      ('mmol O2/hr',    Oxy_pro_spatial,      'O2 production (vertically summed)'),
    'Oxy_nitri_spatial':    ('mmol O2/hr',    Oxy_nitri_spatial,    'O2 consumption by nitrification (vertically summed)'),
    'Oxy_remi_spatial':     ('mmol O2/hr',    Oxy_remi_spatial,     'O2 consumption by remineralization (vertically summed)'),
    'Oxy_sed_spatial':      ('mmol O2/day',   Oxy_sed_spatial,      'SOD method 1 (spatial)'),
    'Oxy_sed_spatial2':     ('mmol O2/hr',    Oxy_sed_spatial2,     'SOD method 2 (spatial)'),
    'Oxy_air_flux_spatial': ('mmol O2/m2/hr', Oxy_air_flux_spatial, 'Air-sea O2 flux (spatial)'),
    'diff_O2_spatial':      ('mmol O2/m3',    diff_O2_spatial,      'O2 saturation deficit (spatial)'),
}
for vn, (units, data, desc) in vars_2d.items():
    v = nc.createVariable(vn, 'f4', ('time', 'eta_rho', 'xi_rho'), compression='zlib', complevel=9)
    v.units = units
    v.long_name = desc
    v[:] = data[0:len(t), :, :]

nc.close()
print('Saved: ' + str(out_fn))
