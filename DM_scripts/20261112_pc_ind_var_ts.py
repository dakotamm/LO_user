"""
Penn Cove environmental variable time series, built step by step, each saved as it is made.

All from box extractions of wb1_t0_xn11abbur00 ocean_his (2024-2025, hourly
instantaneous snapshots at hh:00, -lt hourly0), everything on rho points:
    u-face and v-face fields go onto rho points as in 20261112_pc_utide.py:
    average the two faces on either side of each rho cell, with masked (land)
    faces set to 0 first, the no-normal-flow wall value. The box edge ring has
    only one face and stays NaN.
    wb1 is a plaid lon/lat grid (no angle), so u is east, v is north.
    Penn Cove = wet rho cells in rho columns xi_rho <= 30. Column 31 is in the box
    only to carry the pc_lp u-faces (xi_u 30); its wet cells are Saratoga Passage.

STEP 1: hourly along- and cross-cove wind stress [Pa], from the pc_cove_mix box.
    sustr, svstr: the surface stress ROMS applied, onto rho points, mean over the
    Penn Cove cells, then projected on the cove axis (ax, ay) = (-0.9789, -0.2043),
    mouth -> head:
        tau_al = sustr*ax + svstr*ay    + toward the head (up-cove)
        tau_cr = sustr*ay - svstr*ax    + toward the north shore (right-hand normal)

Saved to LO_output/DM_outs/20261112_pc_ind_var_ts/ind_var_ts.p, a DataFrame on the
hourly time index (tau_al, tau_cr, tau_ew, tau_ns, drho_<sect>, Tair, D2, D1,
eta_tide, Q_tide); each
step adds its columns and saves again.

run 20261112_pc_ind_var_ts.py
"""
import pickle

import gsw
import numpy as np
import pandas as pd
import utide
import xarray as xr
from scipy.signal import hilbert

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

def to_rho(uf, vf):
    """(..., NR, NC-1) u-faces, (..., NR-1, NC) v-faces -> (..., NR, NC) rho."""
    uf = np.nan_to_num(uf, nan=0.0)
    vf = np.nan_to_num(vf, nan=0.0)
    NR, NC = vf.shape[-2] + 1, uf.shape[-1] + 1
    ur = np.full(uf.shape[:-2] + (NR, NC), np.nan, dtype=np.float32)
    vr = np.full(vf.shape[:-2] + (NR, NC), np.nan, dtype=np.float32)
    ur[..., 1:-1] = 0.5 * (uf[..., :-1] + uf[..., 1:])
    vr[..., 1:-1, :] = 0.5 * (vf[..., :-1, :] + vf[..., 1:, :])
    return ur, vr

dm = xr.open_dataset(mix_fn)

T = dm.ocean_time.values
mask = dm.mask_rho.values == 1
cove = mask.copy()
cove[:, 31:] = False

sustr_r, svstr_r = to_rho(dm.sustr.values, dm.svstr.values)
taux = sustr_r[:, cove].mean(axis=1)
tauy = svstr_r[:, cove].mean(axis=1)

tau_al = taux * AX + tauy * AY
tau_cr = taux * AY - tauy * AX

print('%d hours, %s to %s' % (len(T), str(T[0])[:16], str(T[-1])[:16]))
print('%d cove cells' % cove.sum())
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
    onto rho points, mean over the Saratoga Passage cells.
    Saratoga Passage = the tef2 segment pc_lp_p (between pc_lp, sp_mid and
    skagit_sp, 2602 cells). Its (j, i) cells are placed in the box by the box's
    corner index on the full grid; the box has 2 cells of margin, so none is on
    the NaN edge ring.
"""
sp_fn = box_dir / ('sp_stress_surf_' + DS0 + '_' + DS1 + '.nc')
seg_fn = Ldir['LOo'] / 'extract' / 'tef2' / 'seg_info_dict_wb1_pc1_trapsN00.p'

if not sp_fn.exists():
    print('STEP 2 skipped: %s not extracted yet' % sp_fn.name)
else:
    ds = xr.open_dataset(sp_fn)
    assert (ds.ocean_time.values == T).all()

    G = xr.open_dataset(Ldir['grid'] / 'grid.nc')
    i0 = np.argmin(np.abs(G.lon_rho.values[0, :] - ds.lon_rho.values[0, 0]))
    j0 = np.argmin(np.abs(G.lat_rho.values[:, 0] - ds.lat_rho.values[0, 0]))

    ji = np.array(pickle.load(open(seg_fn, 'rb'))['pc_lp_p']['ji_list'])
    sp = np.zeros(ds.mask_rho.shape, dtype=bool)
    sp[ji[:, 0] - j0, ji[:, 1] - i0] = True
    assert (ds.mask_rho.values[sp] == 1).all()

    sustr_r, svstr_r = to_rho(ds.sustr.values, ds.svstr.values)
    tau_ew = sustr_r[:, sp].mean(axis=1)
    tau_ns = svstr_r[:, sp].mean(axis=1)

    print('Saratoga: %d cells' % sp.sum())
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
    Tair    surface air temperature ROMS read (atm00, already on rho points),
            mean over the Penn Cove cells.
"""
air_fn = box_dir / ('pc_cove_air_surf_' + DS0 + '_' + DS1 + '.nc')

if not air_fn.exists():
    print('STEP 4 skipped: %s not extracted yet' % air_fn.name)
else:
    da = xr.open_dataset(air_fn)
    assert (da.ocean_time.values == T).all()

    Tair = da.Tair.values[:, cove].mean(axis=1)

    print('Tair: %d cove cells; mean %.1f, min %.1f, max %.1f degC' % (cove.sum(), Tair.mean(), Tair.min(), Tair.max()))

    df['Tair'] = Tair
    df.to_pickle(out_fn)
    print('saved %s (%d hours, %d columns)' % (out_fn, len(df), df.shape[1]))

"""
STEP 5: hourly spring/neap and tropic/equatorial variation [m3/s], as the envelopes
of the semidiurnal and diurnal tidal transport through the Penn Cove mouth.
    Q       transport through pc_lp, from the pc_cove box: u on rho points in the
            pc_lp rho column (the same cells as drho in STEP 3; the to_rho rule,
            applied to just the two u-face columns either side, since the full 3D
            u box is 6 GB), times dz (from z_w) and dy (1/pn), summed over the
            column. + = east, out of the cove.
            The cove floods and ebbs together, so the mouth serves all three sections.
    One utide fit to Q (OLS, nodal corrections, no trend; M2 S2 N2 K2 K1 O1 P1 Q1
    M4 MS4 M6 S1), then each band is reconstructed and its envelope taken as
    |hilbert(band)|, with the record mean (which reconstruct adds back) removed first:
    D2      spring/neap: envelope of M2 + S2 + N2 + K2 (the tidal prism signal; the
            sea-level envelope wanders 2-3 d off spring/neap because it weights the
            diurnal band as much as the semidiurnal)
    D1      tropic/equatorial: envelope of K1 + O1 + P1 + Q1. S1 is left out: in
            Penn Cove it is mostly the daily wind cycle, not tide.
"""
CONS = ['M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1', 'M4', 'MS4', 'M6', 'S1']
BANDS = {'D2': ['M2', 'S2', 'N2', 'K2'], 'D1': ['K1', 'O1', 'P1', 'Q1']}

line = pd.read_pickle(Ldir['LOo'] / 'section_lines' / 'pc_lp.p')
ii = np.argmin(np.abs(lon_r[0, :] - line.x.values[0]))
jj = np.nonzero(mask[:, ii] & (lat_r[:, ii] >= line.y.min()) & (lat_r[:, ii] <= line.y.max()))[0]

uf = np.nan_to_num(dc.u.isel(eta_u=jj, xi_u=[ii - 1, ii]).values)     # (time, s_rho, cell, west/east face)
u = 0.5 * (uf[..., 0] + uf[..., 1])                                     # (time, s_rho, cell) at rho
dz = np.diff(dc.z_w.isel(eta_rho=jj, xi_rho=ii).values, axis=1)        # (time, s_rho, cell)
dy = 1 / dc.pn.values[jj, ii]                                           # (cell)
Q = (u * dz * dy).sum(axis=(1, 2))

coef = utide.solve(T, Q, lat=LAT, method='ols', conf_int='linear', constit=CONS, nodal=True, trend=False, verbose=False)

env = {}
for band, cons in BANDS.items():
    x = utide.reconstruct(T, coef, constit=cons, verbose=False).h - coef['mean']     # reconstruct adds the mean back
    env[band] = np.abs(hilbert(x))

print('Q at pc_lp (xi_rho %d, %d cells): mean %+.1f, std %.0f m3/s' % (ii, len(jj), Q.mean(), Q.std()))
for band in BANDS:
    print('%s envelope: mean %.0f, min %.0f, max %.0f m3/s' % (band, env[band].mean(), env[band].min(), env[band].max()))

df['D2'] = env['D2']
df['D1'] = env['D1']
df.to_pickle(out_fn)
print('saved %s (%d hours, %d columns)' % (out_fn, len(df), df.shape[1]))

"""
STEP 6: hourly tidal sea level and tidal transport at the Penn Cove mouth, the
astronomical tide only (harmonic predictions, so no wind or subtidal signal).
Together they place each hour in the tidal cycle: Penn Cove is a standing wave,
so Q leads eta by a quarter cycle (high and low water at Q ~ 0, peak flood and
ebb near mid-tide).
    eta_tide  [m]     zeta mean over the pc_lp rho column (the STEP 5 cells), one
                      utide fit with the STEP 5 settings
    Q_tide    [m3/s]  from the STEP 5 fit to Q; + = east, out of the cove (ebb)
    Both are reconstructed from every constituent except S1 (the daily wind
    band), with the record mean removed: anomalies about zero.
"""
TIDE = [c for c in CONS if c != 'S1']

zeta = dc.zeta.isel(eta_rho=jj, xi_rho=ii).values.mean(axis=1)
coef_eta = utide.solve(T, zeta, lat=LAT, method='ols', conf_int='linear', constit=CONS, nodal=True, trend=False, verbose=False)

eta_tide = utide.reconstruct(T, coef_eta, constit=TIDE, verbose=False).h - coef_eta['mean']
Q_tide = utide.reconstruct(T, coef, constit=TIDE, verbose=False).h - coef['mean']

for vn, x, raw in [('eta_tide', eta_tide, zeta), ('Q_tide', Q_tide, Q)]:
    print('%s: std %.3g, min %.3g, max %.3g; explains %.1f%% of the raw variance'
          % (vn, x.std(), x.min(), x.max(), 100 * (1 - np.var(raw - raw.mean() - x) / np.var(raw))))

df['eta_tide'] = eta_tide
df['Q_tide'] = Q_tide
df.to_pickle(out_fn)
print('saved %s (%d hours, %d columns)' % (out_fn, len(df), df.shape[1]))
