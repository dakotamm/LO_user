"""
Penn Cove response variable time series, built step by step, each saved as it is made.

All from box extractions of wb1_t0_xn11abbur00 ocean_his (2024-2025, hourly
instantaneous snapshots at hh:00, -lt hourly0), the same 17545 times as
20261112_pc_ind_var_ts.py.

STEP 1: hourly maximum velocity [m/s] at the three Penn Cove cross sections, from the
pc_cove box, with u on rho points.
    u onto rho points as in 20261112_pc_utide.py: average the two u faces on
    either side of each rho cell (xi_u ii-1 and ii for rho column ii), land faces set
    to 0 first (the no-normal-flow wall value). wb1 is a plaid lon/lat grid (no
    angle), so u is east. v is not needed: it runs along these sections.
    Each section is the box rho column nearest its section line in
    LO_output/section_lines/<sect>.p, wet cells between the line's two end
    latitudes (the same cells as drho in 20261112_pc_ind_var_ts.py), all 30 s
    levels. The lines run due north-south, so u is the section-normal velocity.
    umax[sn]    max |u| over the section's cells and levels in each hour.
    u + is east = out of the cove; |u| drops the sign, so umax is flood or ebb.
    uin[sn]     max of -u (westward, into the cove), + = speed of the fastest inflow
    uout[sn]    max of u (eastward, out of the cove), + = speed of the fastest outflow
                Either is negative in an hour when the whole section flows the other way.
    Saved to LO_output/DM_outs/20261112_pc_res_var_ts/res_var_ts.p, a DataFrame on
    the hourly time index with columns <var>_<sect> (e.g. uin_pc_lp); each later
    step adds its columns and saves again.

run 20261112_pc_res_var_ts.py
"""
import gsw
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SECTS = ['pc_cp', 'pc_lj', 'pc_lp']

gridname, tag, ex_name = GTAGEX.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
box_dir = Ldir['LOo'] / 'extract' / GTAGEX / 'box'
cove_fn = box_dir / ('pc_cove_' + DS0 + '_' + DS1 + '.nc')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261112_pc_res_var_ts'
Lfun.make_dir(out_dir)
out_fn = out_dir / 'res_var_ts.p'

dc = xr.open_dataset(cove_fn)

T = dc.ocean_time.values
mask = dc.mask_rho.values == 1
lon_r = dc.lon_rho.values
lat_r = dc.lat_rho.values

umax, uin, uout = {}, {}, {}
for sn in SECTS:
    line = pd.read_pickle(Ldir['LOo'] / 'section_lines' / (sn + '.p'))
    ii = np.argmin(np.abs(lon_r[0, :] - line.x.values[0]))
    jj = np.nonzero(mask[:, ii] & (lat_r[:, ii] >= line.y.min()) & (lat_r[:, ii] <= line.y.max()))[0]
    uf = np.nan_to_num(dc.u.isel(eta_u=jj, xi_u=[ii - 1, ii]).values)    # (time, s_rho, cell, west/east face)
    u = 0.5 * (uf[..., 0] + uf[..., 1])                                    # (time, s_rho, cell) at rho
    umax[sn] = np.abs(u).max(axis=(1, 2))
    uin[sn] = (-u).max(axis=(1, 2))
    uout[sn] = u.max(axis=(1, 2))
    print('%s: xi_rho %d, %d cells' % (sn, ii, len(jj)))
    for vn, x in [('umax', umax[sn]), ('uin', uin[sn]), ('uout', uout[sn])]:
        print('    %-4s mean %.2f, std %.2f, min %+.2f, max %.2f m/s; < 0 in %.1f%% of hours'
              % (vn, x.mean(), x.std(), x.min(), x.max(), 100 * np.mean(x < 0)))

df = pd.DataFrame(index=pd.DatetimeIndex(T, name='time'))
for sn in SECTS:
    df['umax_' + sn] = umax[sn]
    df['uin_' + sn] = uin[sn]
    df['uout_' + sn] = uout[sn]
df.to_pickle(out_fn)
print('saved %s (%d hours, %d columns)' % (out_fn, len(df), df.shape[1]))

"""
STEP 2: hourly (tidal, unfiltered) inflow and outflow transport Q_in, Q_out [m3/s] at the
three Penn Cove sections, from the tef2 section extractions
(tef2/extractions_avg_<dates>/<sect>.nc).
    q is the hour-mean transport through each section element (z, p), from ROMS
    ocean_avg Huon/Hvom, so it is stamped hh:30 (17544 hours), not hh:00 like STEP 1.
    Hour means close the cove volume budget; his snapshots u*dA do not (sub-hourly
    motion aliases into them).
    Q_in  = sum of the elements flowing into the cove (+)
    Q_out = sum of the elements flowing out of the cove (-)
    Q_in + Q_out = Q_net, the whole-section transport.
    Raw section + is east, out of the cove, so q is flipped. Every element counts by
    its own sign, so this is the gross in/out transport, not TEF (which sorts by
    salinity and lets in- and outflow at the same salinity cancel).
    Saved to its own DataFrame on the hh:30 index:
    LO_output/DM_outs/20261112_pc_res_var_ts/res_var_ts_avg.p, columns Qin_<sect>,
    Qout_<sect>, Qnet_<sect>.
"""
ext_dir = Ldir['LOo'] / 'extract' / GTAGEX / 'tef2' / ('extractions_avg_' + DS0 + '_' + DS1)
out_fn_avg = out_dir / 'res_var_ts_avg.p'

da = pd.DataFrame()
for sn in SECTS:
    de = xr.open_dataset(ext_dir / (sn + '.nc'))
    q = -de.q.values                                                       # (time, z, p), + into the cove
    if da.empty:
        da = pd.DataFrame(index=pd.DatetimeIndex(de.time.values, name='time'))
    else:
        assert (da.index == de.time.values).all()
    da['Qin_' + sn] = np.where(q > 0, q, 0).sum(axis=(1, 2))
    da['Qout_' + sn] = np.where(q < 0, q, 0).sum(axis=(1, 2))
    da['Qnet_' + sn] = q.sum(axis=(1, 2))
    print('%s: %d elements' % (sn, q.shape[1] * q.shape[2]))
    for vn in ['Qin', 'Qout', 'Qnet']:
        x = da[vn + '_' + sn]
        print('    %-4s mean %+8.1f, std %6.1f, min %+8.1f, max %+8.1f m3/s' % (vn, x.mean(), x.std(), x.min(), x.max()))
da.to_pickle(out_fn_avg)
print('saved %s (%d hours, %d columns)' % (out_fn_avg, len(da), da.shape[1]))

"""
STEP 3: hourly whole-cove top-bottom density difference split into its salinity and
temperature parts [kg/m3], from the pc_cove box (salt, temp at rho points, the same
17545 hh:00 times as STEP 1).
    Cove = wet rho cells in columns xi_rho <= 30 (column 31 is Saratoga Passage).
    In each cell, step the bottom water (s_rho 0) to the top water (s_rho -1) one
    property at a time:
        drho_S = sigma0(s_bot, t_top) - sigma0(s_top, t_top)    salinity, at top temperature
        drho_T = sigma0(s_bot, t_bot) - sigma0(s_bot, t_top)    temperature, at bottom salinity
    drho_S + drho_T = sigma0(bottom) - sigma0(top) exactly (+ = stable). The order
    matters only through the nonlinearity of the equation of state, which is small over
    Penn Cove's range. Then the area-weighted mean over the cove cells, dA = 1/(pm*pn).
        drhoS_pc, drhoT_pc    whole Penn Cove
    Daily surface heating shows up in drhoT_pc; drhoS_pc is the salinity layering.
    sigma0 as in 20261112_pc_ind_var_ts.py: TEOS-10, absolute salinity from practical
    salinity at 0 dbar and one Penn Cove lon/lat, conservative temperature from ROMS
    potential temperature.
    Added to res_var_ts.p and saved again.
"""
LON, LAT = -122.69, 48.23

def sigma0(s, t):
    SA = gsw.SA_from_SP(s, 0, LON, LAT)
    return gsw.sigma0(SA, gsw.CT_from_pt(SA, t))

cove = mask.copy()
cove[:, 31:] = False
jc, ic = np.nonzero(cove)
dA = 1 / (dc.pm.values[jc, ic] * dc.pn.values[jc, ic])

s = dc.salt.isel(s_rho=[0, -1]).values[:, :, jc, ic]                    # (time, bot/top, cell)
t = dc.temp.isel(s_rho=[0, -1]).values[:, :, jc, ic]
sig_bot = sigma0(s[:, 0], t[:, 0])                                       # (time, cell)
sig_top = sigma0(s[:, 1], t[:, 1])
drhoS_cell = sigma0(s[:, 0], t[:, 1]) - sig_top
drhoT_cell = sig_bot - sigma0(s[:, 0], t[:, 1])
drhoS_pc = (drhoS_cell * dA).sum(axis=1) / dA.sum()
drhoT_pc = (drhoT_cell * dA).sum(axis=1) / dA.sum()

print('cove: %d cells, %.2f km2' % (len(jc), dA.sum() / 1e6))
for vn, x in [('drhoS_pc', drhoS_pc), ('drhoT_pc', drhoT_pc)]:
    print('%s mean %+.2f, std %.2f, min %+.2f, max %+.2f kg/m3; < 0 in %.1f%% of hours'
          % (vn, x.mean(), x.std(), x.min(), x.max(), 100 * np.mean(x < 0)))

df['drhoS_pc'] = drhoS_pc
df['drhoT_pc'] = drhoT_pc
df.to_pickle(out_fn)
print('saved %s (%d hours, %d columns)' % (out_fn, len(df), df.shape[1]))

"""
STEP 4: hourly along-cove density gradient, head minus mouth [kg/m3 per km], at the top
and bottom levels.
    Head  = inner cove, rho columns xi_rho <= 15 (the tef2 segment pc_cp_m, west of
            pc_cp; 165 cells, mean h 13 m)
    Mouth = outer cove, rho columns 24-30 (segment pc_lp_m, between pc_lj and pc_lp;
            77 cells, mean h 22 m)
    Area-weighted sigma0 over each region at the top level (s_rho -1) and at the bottom
    level (s_rho 0), head minus mouth, divided by the distance L between the two
    regions' area centroids (3.8 km):
        drhodx_top, drhodx_bot    + = denser toward the head
    The bottom level is a terrain-following level, so the two regions' bottom values sit
    at different depths (~13 vs ~22 m).
    With the vertical difference this gives the isopycnal tilt:
    slope ~ -(drho/dx) / (drho/dz).
    Added to res_var_ts.p and saved again.
"""
head = ic <= 15
mouth = (ic >= 24) & (ic <= 30)

def amean(x, r):
    return (x[:, r] * dA[r]).sum(axis=1) / dA[r].sum()

lon_c, lat_c = lon_r[jc, ic], lat_r[jc, ic]
dx = (lon_c[head].mean() - lon_c[mouth].mean()) * 111.32 * np.cos(np.deg2rad(LAT))
dy = (lat_c[head].mean() - lat_c[mouth].mean()) * 111.32
L = np.hypot(dx, dy)                                                      # km

drhodx_top = (amean(sig_top, head) - amean(sig_top, mouth)) / L
drhodx_bot = (amean(sig_bot, head) - amean(sig_bot, mouth)) / L

print('head %d cells, mouth %d cells; L = %.2f km' % (head.sum(), mouth.sum(), L))
for vn, x in [('drhodx_top', drhodx_top), ('drhodx_bot', drhodx_bot)]:
    print('%s mean %+.3f, std %.3f, min %+.3f, max %+.3f kg/m3/km; > 0 in %.0f%% of hours'
          % (vn, x.mean(), x.std(), x.min(), x.max(), 100 * np.mean(x > 0)))

df['drhodx_top'] = drhodx_top
df['drhodx_bot'] = drhodx_bot
df.to_pickle(out_fn)
print('saved %s (%d hours, %d columns)' % (out_fn, len(df), df.shape[1]))

"""
STEP 5: hourly whole-cove potential energy anomaly phi [J/m3] (Simpson et al. 1981),
same cells, times and weighting as STEP 3, now using all 30 levels.
    In each cell, with D = water depth, sigma_bar = depth mean of sigma0:
        phi = (g/D) * sum over levels of (sigma_bar - sigma0) * z_rho * dz
    the energy per volume needed to mix the column completely (0 = mixed, + = stable).
    dz from z_w, z_rho and z_w from the box (they move with zeta). The 1000 in
    rho = 1000 + sigma0 drops out, and so does the z origin.
        phi_pc      area-weighted cove mean
    g = 9.81. Read in blocks of 1000 hours to keep memory down.
    Added to res_var_ts.p and saved again.
"""
G = 9.81
NB = 1000
phi_pc = np.full(len(T), np.nan)
for t0 in range(0, len(T), NB):
    tt = slice(t0, t0 + NB)
    sb = dc.salt.isel(ocean_time=tt).values[:, :, jc, ic]                 # (time, s_rho, cell)
    tb = dc.temp.isel(ocean_time=tt).values[:, :, jc, ic]
    zr = dc.z_rho.isel(ocean_time=tt).values[:, :, jc, ic]
    zw = dc.z_w.isel(ocean_time=tt).values[:, :, jc, ic]                  # (time, s_w, cell)
    sg = sigma0(sb, tb)
    dz = np.diff(zw, axis=1)
    D = dz.sum(axis=1)                                                     # (time, cell)
    sg_bar = (sg * dz).sum(axis=1) / D
    phi_cell = (G / D) * ((sg_bar[:, None, :] - sg) * zr * dz).sum(axis=1)
    phi_pc[tt] = (phi_cell * dA).sum(axis=1) / dA.sum()
    print('  phi block %d-%d' % (t0, min(t0 + NB, len(T))), flush=True)

print('phi_pc mean %.1f, std %.1f, min %+.1f, max %.1f J/m3'
      % (phi_pc.mean(), phi_pc.std(), phi_pc.min(), phi_pc.max()))

df['phi_pc'] = phi_pc
df.to_pickle(out_fn)
print('saved %s (%d hours, %d columns)' % (out_fn, len(df), df.shape[1]))
