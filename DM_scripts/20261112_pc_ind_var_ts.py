"""
Penn Cove environmental variable time series, built step by step, each saved as it is made.

All from box extractions of wb1_t0_xn11abbur00 ocean_his (2024-2025, hourly
instantaneous snapshots at hh:00, -lt hourly0). Cove only in STEP 1: rho columns
xi_rho <= 30; column 31 is Saratoga Passage.

STEP 1: hourly along- and cross-cove wind stress [Pa], from the pc_cove_mix box.
    sustr, svstr: the surface stress ROMS applied, on u / v faces.
    Cove faces: u faces with water on both sides and xi_u < 30 (xi_u 30 is the
    pc_lp mouth face); v faces with water on both sides in rho columns <= 30.
    Mean over cove faces, then projected on the cove axis (ax, ay) = (-0.9789, -0.2043),
    mouth -> head:
        tau_al = sustr*ax + svstr*ay    + toward the head (up-cove)
        tau_cr = sustr*ay - svstr*ax    + toward the north shore (right-hand normal)

Saved to LO_output/DM_outs/20261112_pc_ind_var_ts/ind_var_ts.p, a DataFrame on the
hourly time index (tau_al, tau_cr, tau_ew, tau_ns, drho_<sect>); each step adds its
columns and saves again.

run 20261112_pc_ind_var_ts.py
"""
import pickle

import gsw
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
AX, AY = -0.9789, -0.2043

gridname, tag, ex_name = GTAGEX.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
box_dir = Ldir['LOo'] / 'extract' / GTAGEX / 'box'
mix_fn = box_dir / ('pc_cove_mix_' + DS0 + '_' + DS1 + '.nc')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261112_pc_ind_var_ts'
Lfun.make_dir(out_dir)
out_fn = out_dir / 'ind_var_ts.p'

dm = xr.open_dataset(mix_fn)

T = dm.ocean_time.values
mask = dm.mask_rho.values == 1

mu = mask[:, :-1] & mask[:, 1:]
mu[:, 30:] = False
mv = mask[:-1, :] & mask[1:, :]
mv[:, 31:] = False

taux = dm.sustr.values[:, mu].mean(axis=1)
tauy = dm.svstr.values[:, mv].mean(axis=1)

tau_al = taux * AX + tauy * AY
tau_cr = taux * AY - tauy * AX

print('%d hours, %s to %s' % (len(T), str(T[0])[:16], str(T[-1])[:16]))
print('%d u faces, %d v faces' % (mu.sum(), mv.sum()))
print('tau_al mean %+.4f, std %.4f Pa; up-cove %.0f%% of hours' % (tau_al.mean(), tau_al.std(), 100 * np.mean(tau_al > 0)))
print('tau_cr mean %+.4f, std %.4f Pa; northward %.0f%% of hours' % (tau_cr.mean(), tau_cr.std(), 100 * np.mean(tau_cr > 0)))

df = pd.DataFrame(index=pd.DatetimeIndex(T, name='time'))
df['tau_al'] = tau_al
df['tau_cr'] = tau_cr
df.to_pickle(out_fn)
print('saved %s (%d hours, %d columns)' % (out_fn, len(df), df.shape[1]))

"""
STEP 2: hourly east-west and north-south wind stress [Pa] over upper Saratoga
Passage, from the sp_stress box (-surf True).
    tau_ew  sustr, + toward the east
    tau_ns  svstr, + toward the north
    wb1 is a plaid lon/lat grid (no angle), so sustr is east and svstr is north.
    Saratoga Passage = the tef2 segment pc_lp_p (between pc_lp, sp_mid and
    skagit_sp, 2602 cells). Its (j, i) cells are placed in the box by the box's
    corner index on the full grid. Faces with segment cells on both sides only,
    the same rule as the cove in STEP 1.
"""
sp_fn = box_dir / ('sp_stress_surf_' + DS0 + '_' + DS1 + '.nc')
seg_fn = Ldir['LOo'] / 'extract' / 'tef2' / 'seg_info_dict_wb1_pc1_trapsN00.p'

ds = xr.open_dataset(sp_fn)
assert (ds.ocean_time.values == T).all()

G = xr.open_dataset(Ldir['grid'] / 'grid.nc')
i0 = np.argmin(np.abs(G.lon_rho.values[0, :] - ds.lon_rho.values[0, 0]))
j0 = np.argmin(np.abs(G.lat_rho.values[:, 0] - ds.lat_rho.values[0, 0]))

ji = np.array(pickle.load(open(seg_fn, 'rb'))['pc_lp_p']['ji_list'])
sp = np.zeros(ds.mask_rho.shape, dtype=bool)
sp[ji[:, 0] - j0, ji[:, 1] - i0] = True
assert (ds.mask_rho.values[sp] == 1).all()

su = sp[:, :-1] & sp[:, 1:]
sv = sp[:-1, :] & sp[1:, :]

tau_ew = ds.sustr.values[:, su].mean(axis=1)
tau_ns = ds.svstr.values[:, sv].mean(axis=1)

print('Saratoga: %d cells, %d u faces, %d v faces' % (sp.sum(), su.sum(), sv.sum()))
print('tau_ew mean %+.4f, std %.4f Pa; eastward %.0f%% of hours' % (tau_ew.mean(), tau_ew.std(), 100 * np.mean(tau_ew > 0)))
print('tau_ns mean %+.4f, std %.4f Pa; northward %.0f%% of hours' % (tau_ns.mean(), tau_ns.std(), 100 * np.mean(tau_ns > 0)))

df['tau_ew'] = tau_ew
df['tau_ns'] = tau_ns
df.to_pickle(out_fn)
print('saved %s (%d hours, %d columns)' % (out_fn, len(df), df.shape[1]))

"""
STEP 3: hourly top-bottom density difference at the three Penn Cove cross sections, from the
pc_cove box (salt, temp at rho points, the same 17545 times as STEPS 1-2).
    Each section is the box rho column nearest its section line in
    LO_output/section_lines/<sect>.p (the lines run due north-south), wet cells
    between the line's two end latitudes, at the bottom (s_rho 0) and top
    (s_rho -1) levels.
    drho[sn]    top-bottom density difference = sigma0(bottom) - sigma0(top) [kg/m3] in each cell, then the mean
                across the section's cells (+ = stable). All cells are one grid
                cell (~200 m) wide.
    sigma0 is TEOS-10: absolute salinity from practical salinity (at sea pressure
    0 dbar and one Penn Cove lon/lat; the real ~20 dbar changes sigma0 by
    << 0.001 kg/m3), conservative temperature from ROMS potential temperature.
"""
SECTS = ['pc_cp', 'pc_lj', 'pc_lp']
LON, LAT = -122.69, 48.23
cove_fn = box_dir / ('pc_cove_' + DS0 + '_' + DS1 + '.nc')

def sigma0(s, t):
    SA = gsw.SA_from_SP(s, 0, LON, LAT)
    return gsw.sigma0(SA, gsw.CT_from_pt(SA, t))

dc = xr.open_dataset(cove_fn)
assert (dc.ocean_time.values == T).all()
lon_r = dc.lon_rho.values
lat_r = dc.lat_rho.values

drho = {}
for sn in SECTS:
    line = pd.read_pickle(Ldir['LOo'] / 'section_lines' / (sn + '.p'))
    ii = np.argmin(np.abs(lon_r[0, :] - line.x.values[0]))
    jj = np.nonzero(mask[:, ii] & (lat_r[:, ii] >= line.y.min()) & (lat_r[:, ii] <= line.y.max()))[0]
    s = dc.salt.isel(s_rho=[0, -1], eta_rho=jj, xi_rho=ii).values         # (time, bot/top, cell)
    t = dc.temp.isel(s_rho=[0, -1], eta_rho=jj, xi_rho=ii).values
    drho[sn] = (sigma0(s[:, 0], t[:, 0]) - sigma0(s[:, 1], t[:, 1])).mean(axis=1)
    print('%s: xi_rho %d, %d cells; drho mean %.2f, std %.2f, min %.2f, max %.2f kg/m3'
          % (sn, ii, len(jj), drho[sn].mean(), drho[sn].std(), drho[sn].min(), drho[sn].max()))

for sn in SECTS:
    df['drho_' + sn] = drho[sn]
df.to_pickle(out_fn)
print('saved %s (%d hours, %d columns)' % (out_fn, len(df), df.shape[1]))

"""
STEP 4: hourly air temperature over Penn Cove [degC], from the pc_cove_air box
(-surf True, same footprint as pc_cove).
    Tair    surface air temperature ROMS read (atm00), mean over cove water
            cells (rho columns xi_rho <= 30, as in STEP 1).
"""
air_fn = box_dir / ('pc_cove_air_surf_' + DS0 + '_' + DS1 + '.nc')

da = xr.open_dataset(air_fn)
assert (da.ocean_time.values == T).all()

cove = mask.copy()
cove[:, 31:] = False

Tair = da.Tair.values[:, cove].mean(axis=1)

print('Tair: %d cove cells; mean %.1f, min %.1f, max %.1f degC' % (cove.sum(), Tair.mean(), Tair.min(), Tair.max()))
