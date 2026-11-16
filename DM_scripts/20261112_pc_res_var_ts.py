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
from scipy import ndimage

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
and bottom levels and depth-averaged.
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
    Depth-averaged: in each cell the dz-weighted mean sigma0 over all 30 levels (dz
    from the box z_w, which moves with zeta), then the same area-weighted region means,
    head minus mouth, / L:
        drhodx_bar                + = denser toward the head
    Read in blocks of NB hours to keep memory down.
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

NB = 1000
sig_bar = np.full((len(T), len(jc)), np.nan)                             # (time, cell) depth-mean sigma0
for t0 in range(0, len(T), NB):
    tt = slice(t0, t0 + NB)
    sb = dc.salt.isel(ocean_time=tt).values[:, :, jc, ic]                 # (time, s_rho, cell)
    tb = dc.temp.isel(ocean_time=tt).values[:, :, jc, ic]
    dz = np.diff(dc.z_w.isel(ocean_time=tt).values[:, :, jc, ic], axis=1)
    sig_bar[tt] = (sigma0(sb, tb) * dz).sum(axis=1) / dz.sum(axis=1)
drhodx_bar = (amean(sig_bar, head) - amean(sig_bar, mouth)) / L

print('head %d cells, mouth %d cells; L = %.2f km' % (head.sum(), mouth.sum(), L))
for vn, x in [('drhodx_top', drhodx_top), ('drhodx_bot', drhodx_bot), ('drhodx_bar', drhodx_bar)]:
    print('%s mean %+.3f, std %.3f, min %+.3f, max %+.3f kg/m3/km; > 0 in %.0f%% of hours'
          % (vn, x.mean(), x.std(), x.min(), x.max(), 100 * np.mean(x > 0)))

df['drhodx_top'] = drhodx_top
df['drhodx_bot'] = drhodx_bot
df['drhodx_bar'] = drhodx_bar
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
    g = 9.81. Read in blocks of NB hours, as in STEP 4.
    Added to res_var_ts.p and saved again.
"""
G = 9.81
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

"""
STEP 6: hourly shape of the flow across the three Penn Cove sections, from the same tef2
extractions as STEP 2 (hour-mean Huon q, DZ, face width dd; hh:30). + = into the cove.
    Each section is a grid of elements (z, p): 30 s levels (0 = bottom) by the faces
    (p runs north -> south, all 200 m wide). Element area dA = dd*DZ, u = q/dA.
    Position of each element, so sections of different size compare:
        y      0 at the south end of the section, 1 at the north end
        sig    -1 at the bed, 0 at the surface, (z - zeta)/D in each column
    The s levels are terrain-following, so a "level" is a fixed fraction of the local
    depth, not a fixed depth.

    1 Usurf, Ubot   mean u over the top / bottom row (sum q / sum dA)       [m/s]
                    the direction of the surface and bottom flow
    2 Unorth, Usouth  mean u over the northmost / southmost face column   [m/s]
                    (sum q / sum dA, i.e. that column's depth mean)
                    the direction of the flow at the north and south edges
    3 ain           share of section area flowing in (out = 1 - ain)
    4 Hin, Hout     layered view of thickness: the width-integrated profile Qz(z) =
                    sum over p of q; share of section area in levels where Qz > 0 / < 0
    5 Win, Wout     lateral view of width: the depth-integrated transport Qp(p) = sum
                    over z of q; share of section width in faces where Qp > 0 / < 0
    6 zin, zout     |q|-weighted mean sig of the inflowing / outflowing elements
      yin, yout     |q|-weighted mean y of the same
                    zin - zout > 0: inflow above outflow; yin - yout > 0: inflow north
                    of outflow. Only means what it says when 9 counts 1 patch of each.
    7 rex           min(Qin, |Qout|) / max(Qin, |Qout|), element sums as in STEP 2
                    0 = all one way, 1 = pure exchange
    8 flay          share of the counterflow variance that is layered:
                    u - ubar = lateral (each column's depth mean - ubar)
                             + layered (u - its column's depth mean)
                    the two are orthogonal (layered averages to 0 down each column), so
                    their dA-weighted variances add; flay = var(layered)/var(u - ubar),
                    and the lateral share is 1 - flay. Same split as
                    20261110_pc_column_u0u1u2.py.
    9 nlay          layers in Qz(z): runs of one sign, after dropping runs that carry
                    < 10% of that sign's total and merging what is left
      nlat          lateral bands in Qp(p), the same way
      npin, npout   connected inflow / outflow patches in the (z, p) grid (side-sharing
                    elements; scipy.ndimage.label) carrying >= 10% of Qin / |Qout|
                    "tubes": catches an inflow core inside outflow that 4-5 miss
    Columns <metric>_<sect>, added to res_var_ts_avg.p and saved again.
"""
FRAC = 0.1

def n_runs(x):
    """Runs of one sign in 1-D x, dropping runs with < FRAC of their sign's total, then merging."""
    runs = np.split(x, np.flatnonzero(np.diff(np.sign(x))) + 1)
    tot = {1: x[x > 0].sum(), -1: -x[x < 0].sum()}
    keep = [np.sign(r[0]) for r in runs if r[0] != 0 and abs(r.sum()) >= FRAC * tot[np.sign(r[0])]]
    return 1 + np.count_nonzero(np.diff(keep)) if keep else 0

def n_patches(qq):
    """Connected patches of qq > 0 carrying >= FRAC of the total positive transport."""
    lab, n = ndimage.label(qq > 0)
    if n == 0:
        return 0
    tr = ndimage.sum(qq, lab, index=np.arange(1, n + 1))
    return np.count_nonzero(tr >= FRAC * tr.sum())

for sn in SECTS:
    de = xr.open_dataset(ext_dir / (sn + '.nc'))
    q = -de.q.values                                                       # (time, z, p), + into the cove
    dz = de.DZ.values
    dd = de.dd.values
    NT, NZ, NP = q.shape
    dA = dz * dd[None, None, :]
    u = q / dA
    y = 1 - (np.arange(NP) + 0.5) / NP                                     # faces run north -> south
    sig = (np.cumsum(dz, axis=1) - dz / 2) / dz.sum(axis=1)[:, None, :] - 1
    qin = np.where(q > 0, q, 0)
    qout = np.where(q < 0, -q, 0)
    A = dA.sum(axis=(1, 2))

    # 1 surface and bottom direction
    Usurf = q[:, -1, :].sum(axis=1) / dA[:, -1, :].sum(axis=1)
    Ubot = q[:, 0, :].sum(axis=1) / dA[:, 0, :].sum(axis=1)

    # 2 north and south edge direction
    Unorth = q[:, :, 0].sum(axis=1) / dA[:, :, 0].sum(axis=1)               # p = 0 is the north end
    Usouth = q[:, :, -1].sum(axis=1) / dA[:, :, -1].sum(axis=1)

    # 3 inflow area share
    ain = (dA * (q > 0)).sum(axis=(1, 2)) / A

    # 4 layered thickness, 5 lateral width
    Qz = q.sum(axis=2)                                                     # (time, z)
    Az = dA.sum(axis=2)
    Hin = (Az * (Qz > 0)).sum(axis=1) / A
    Hout = (Az * (Qz < 0)).sum(axis=1) / A
    Qp = q.sum(axis=1)                                                     # (time, p)
    Ap = dA.sum(axis=1)
    Win = (dd * (Qp > 0)).sum(axis=1) / dd.sum()
    Wout = (dd * (Qp < 0)).sum(axis=1) / dd.sum()

    # 6 where the inflow and outflow sit
    with np.errstate(invalid='ignore'):
        zin = (qin * sig).sum(axis=(1, 2)) / qin.sum(axis=(1, 2))
        zout = (qout * sig).sum(axis=(1, 2)) / qout.sum(axis=(1, 2))
        yin = (qin * y).sum(axis=(1, 2)) / qin.sum(axis=(1, 2))
        yout = (qout * y).sum(axis=(1, 2)) / qout.sum(axis=(1, 2))

    # 7 exchange ratio
    Qi, Qo = qin.sum(axis=(1, 2)), qout.sum(axis=(1, 2))
    rex = np.minimum(Qi, Qo) / np.maximum(Qi, Qo)

    # 8 layered share of the counterflow variance
    ubar = q.sum(axis=(1, 2)) / A
    ucol = Qp / Ap                                                         # column depth means
    v_tot = (dA * (u - ubar[:, None, None])**2).sum(axis=(1, 2)) / A
    v_lat = (Ap * (ucol - ubar[:, None])**2).sum(axis=1) / A
    v_lay = (dA * (u - ucol[:, None, :])**2).sum(axis=(1, 2)) / A
    flay = v_lay / v_tot

    # 9 counts
    nlay = np.array([n_runs(Qz[k]) for k in range(NT)])
    nlat = np.array([n_runs(Qp[k]) for k in range(NT)])
    npin = np.array([n_patches(q[k]) for k in range(NT)])
    npout = np.array([n_patches(-q[k]) for k in range(NT)])

    M = {'Usurf': Usurf, 'Ubot': Ubot, 'Unorth': Unorth, 'Usouth': Usouth, 'ain': ain, 'Hin': Hin, 'Hout': Hout, 'Win': Win,
         'Wout': Wout, 'zin': zin, 'zout': zout, 'yin': yin, 'yout': yout, 'rex': rex,
         'flay': flay, 'nlay': nlay, 'nlat': nlat, 'npin': npin, 'npout': npout}
    for vn in M:
        da[vn + '_' + sn] = M[vn]

    print('%s: %d levels x %d faces; max |v_lat + v_lay - v_tot| / v_tot %.1e'
          % (sn, NZ, NP, np.nanmax(np.abs(v_lat + v_lay - v_tot) / v_tot)))
    for vn in ['Usurf', 'Ubot', 'Unorth', 'Usouth', 'ain', 'Hin', 'Win', 'zin', 'zout', 'yin', 'yout', 'rex', 'flay']:
        x = M[vn]
        print('    %-5s mean %+.3f, std %.3f, min %+.3f, max %+.3f' % (vn, np.nanmean(x), np.nanstd(x), np.nanmin(x), np.nanmax(x)))
    for vn in ['nlay', 'nlat', 'npin', 'npout']:
        c = np.bincount(M[vn], minlength=5)
        print('    %-5s %% of hours with 0/1/2/3/4+: %s' % (vn, ' '.join('%.0f' % v for v in 100 * np.r_[c[:4], c[4:].sum()] / NT)))

da.to_pickle(out_fn_avg)
print('saved %s (%d hours, %d columns)' % (out_fn_avg, len(da), da.shape[1]))

"""
STEP 7: hourly back-bay eddy rotation, from the pc_cove box (u, v, z_w; the same 17545
hh:00 times as STEP 1).
    The ring is found here, from the 2-year mean depth-mean flow (ubar, vbar onto rho):
    every back-bay wet cell (rho columns xi_rho <= 15, west of pc_cp) is tried as a
    centre, with outer radii R = 400-1200 m every 50 m and inner radius R/2. A ring
    counts only if it is closed: >= 95% of the annulus area is water (box edge cells
    excluded). The eddy is the ring with the fastest mean counter-clockwise speed
    (the radius of maximum swirl). Its cells are then used for every hour.
    u, v onto rho as in STEP 1 (two faces either side, land faces 0). In each ring cell
    and s level, the azimuthal velocity about the centre:
        u_t = (-y*u + x*v) / r                    + = counter-clockwise
    with x, y the cell's east and north distance from the centre.
        Ut_ed     ring-mean u_t at each level, then the dz-weighted depth mean   [m/s]
                  how fast the ring turns
        fccw_ed   fraction of ring cells whose depth-mean u_t > 0
                  1 = the whole ring turns counter-clockwise (a closed eddy),
                  0.5 = no turning about the centre
        fcol_ed   fraction of the water column (dz-weighted s levels) whose ring-mean
                  u_t > 0; how much of the column is turning counter-clockwise
    The s levels are terrain-following, so a level is a fixed fraction of the local
    depth.
    Added to res_var_ts.p and saved again.
"""
interior = np.zeros_like(mask)
interior[1:-1, 1:-1] = True
dA_all = 1 / (dc.pm.values * dc.pn.values)
NR, NC = mask.shape
ubm = np.nan_to_num(dc.ubar.values).mean(axis=0)
vbm = np.nan_to_num(dc.vbar.values).mean(axis=0)
urm = np.full((NR, NC), np.nan)
vrm = np.full((NR, NC), np.nan)
urm[:, 1:-1] = 0.5 * (ubm[:, :-1] + ubm[:, 1:])
vrm[1:-1, :] = 0.5 * (vbm[:-1, :] + vbm[1:, :])

def ring_xy(jc, ic):
    """East / north distance [m] and radius of every rho cell from cell (jc, ic)."""
    X = (lon_r - lon_r[jc, ic]) * 111.32e3 * np.cos(np.deg2rad(LAT))
    Y = (lat_r - lat_r[jc, ic]) * 111.32e3
    return X, Y, np.hypot(X, Y)

best = (-np.inf, None, None)
back = mask & interior & (np.arange(NC)[None, :] <= 15)
for jc, ic in zip(*np.nonzero(back)):
    X, Y, R = ring_xy(jc, ic)
    with np.errstate(invalid='ignore', divide='ignore'):
        utm = (-Y * urm + X * vrm) / R
    for Rm in range(400, 1201, 50):
        rg = mask & interior & (R >= Rm / 2) & (R <= Rm)
        if dA_all[rg].sum() < 0.95 * np.pi * 0.75 * Rm**2:
            continue
        if utm[rg].mean() > best[0]:
            best = (utm[rg].mean(), (jc, ic), Rm)

Ut_mean, (jc_ed, ic_ed), R_ed = best
X, Y, R = ring_xy(jc_ed, ic_ed)
ring = mask & interior & (R >= R_ed / 2) & (R <= R_ed)
jr, ir = np.nonzero(ring)
print('eddy: centre %.4f %.4f, R %d m, 2-yr mean depth-mean ring speed %.2f cm/s'
      % (lon_r[jc_ed, ic_ed], lat_r[jc_ed, ic_ed], R_ed, 100 * Ut_mean))
j0, i0 = jr.min(), ir.min()

uf = np.nan_to_num(dc.u.isel(eta_u=slice(j0, jr.max() + 1), xi_u=slice(i0 - 1, ir.max() + 1)).values)
vf = np.nan_to_num(dc.v.isel(eta_v=slice(j0 - 1, jr.max() + 1), xi_v=slice(i0, ir.max() + 1)).values)
zw = dc.z_w.isel(eta_rho=slice(j0, jr.max() + 1), xi_rho=slice(i0, ir.max() + 1)).values
ur = 0.5 * (uf[:, :, jr - j0, ir - i0] + uf[:, :, jr - j0, ir - i0 + 1])     # (time, s_rho, cell)
vr = 0.5 * (vf[:, :, jr - j0, ir - i0] + vf[:, :, jr - j0 + 1, ir - i0])
dz = np.diff(zw[:, :, jr - j0, ir - i0], axis=1)
ut = (-Y[jr, ir] * ur + X[jr, ir] * vr) / R[jr, ir]

def eddy_metrics(ut, dz):
    """(time, s_rho, cell) u_t and dz -> Ut_ed, fccw_ed, fcol_ed (time,)"""
    Uring = ut.mean(axis=2)                                                # (time, s_rho)
    dzr = dz.mean(axis=2)
    Ut = (Uring * dzr).sum(axis=1) / dzr.sum(axis=1)
    ucol = (ut * dz).sum(axis=1) / dz.sum(axis=1)                          # (time, cell)
    fccw = (ucol > 0).mean(axis=1)
    fcol = ((Uring > 0) * dzr).sum(axis=1) / dzr.sum(axis=1)
    return Ut, fccw, fcol

ED = {}
ED['Ut_ed'], ED['fccw_ed'], ED['fcol_ed'] = eddy_metrics(ut, dz)

print('ring: %d cells, r %.0f-%.0f m' % (len(jr), R[ring].min(), R[ring].max()))
for vn, x in ED.items():
    print('%-10s mean %+.3f, std %.3f, min %+.3f, max %+.3f; > 0.5 in %.0f%%, >= 0.75 in %.0f%% of hours'
          % (vn, np.nanmean(x), np.nanstd(x), np.nanmin(x), np.nanmax(x),
             100 * np.nanmean(x[np.isfinite(x)] > 0.5), 100 * np.nanmean(x[np.isfinite(x)] >= 0.75)))
    df[vn] = x
df.to_pickle(out_fn)
print('saved %s (%d hours, %d columns)' % (out_fn, len(df), df.shape[1]))

"""
STEP 8: hourly along-cove velocity in the bottom half of the water column, in the north
and south halves of the back bay [m/s], from the pc_cove box (u, v, z_w, hh:00).
    Back bay = cove cells west of pc_cp (rho columns xi_rho <= 15; 165 cells). Each rho
    column is split at the midpoint of its wet latitudes: north half 81 cells (mean h
    13.7 m), south half 84 cells (12.7 m).
    Bottom half = the s levels whose centre is below half the local depth, from
    (z_rho - zeta)/D; on these terrain-following levels that fraction does not change
    with time, so each cell keeps the same 10-12 levels throughout.
    u, v onto rho as in STEP 1 (land faces 0); along-cove component on the cove axis of
    20261112_pc_ind_var_ts.py, (ax, ay) = (-0.9789, -0.2043), mouth -> head:
        u_al = u*ax + v*ay                       + = toward the head (into the cove)
    Volume-weighted mean over each half's bottom-half cells (weight dz*dA):
        UbotN_bb, UbotS_bb
    Read in blocks of NB hours.
"""
AX, AY = -0.9789, -0.2043
NR8, NC8 = mask.shape
back = cove & (np.arange(NC8)[None, :] <= 15)
north = np.zeros_like(back)
for i in range(NC8):
    j = np.nonzero(back[:, i])[0]
    if len(j) > 0:
        north[j[lat_r[j, i] > 0.5 * (lat_r[j, i].min() + lat_r[j, i].max())], i] = True
jb, ib = np.nonzero(back)
isN = north[jb, ib]
assert jb.min() >= 1 and ib.min() >= 1
dAb = 1 / (dc.pm.values[jb, ib] * dc.pn.values[jb, ib])

zr0 = dc.z_rho.isel(ocean_time=0).values[:, jb, ib]
zw0 = dc.z_w.isel(ocean_time=0).values[:, jb, ib]
BH = (zr0 - zw0[-1]) / (zw0[-1] - zw0[0]) < -0.5                          # (s_rho, cell) bottom-half levels

NB = 1000
num = {'N': np.full(len(T), np.nan), 'S': np.full(len(T), np.nan)}
den = {'N': np.full(len(T), np.nan), 'S': np.full(len(T), np.nan)}
for t0 in range(0, len(T), NB):
    tt = slice(t0, t0 + NB)
    U = np.nan_to_num(dc.u.isel(ocean_time=tt).values)
    V = np.nan_to_num(dc.v.isel(ocean_time=tt).values)
    dz = np.diff(dc.z_w.isel(ocean_time=tt).values[:, :, jb, ib], axis=1)  # (time, s_rho, cell)
    ur = 0.5 * (U[:, :, jb, ib - 1] + U[:, :, jb, ib])
    vr = 0.5 * (V[:, :, jb - 1, ib] + V[:, :, jb, ib])
    w = dz * dAb * BH                                                      # volume weight, bottom half only
    for hn, r in [('N', isN), ('S', ~isN)]:
        num[hn][tt] = ((ur * AX + vr * AY) * w)[:, :, r].sum(axis=(1, 2))
        den[hn][tt] = w[:, :, r].sum(axis=(1, 2))
    print('  back-bay block %d-%d' % (t0, min(t0 + NB, len(T))), flush=True)

print('back bay: %d cells, north %d, south %d; bottom-half levels per cell %d-%d'
      % (len(jb), isN.sum(), (~isN).sum(), BH.sum(axis=0).min(), BH.sum(axis=0).max()))
for hn in ['N', 'S']:
    x = num[hn] / den[hn]
    df['Ubot%s_bb' % hn] = x
    print('Ubot%s_bb mean %+.4f, std %.4f, min %+.3f, max %+.3f m/s; toward the head in %.0f%% of hours'
          % (hn, x.mean(), x.std(), x.min(), x.max(), 100 * np.mean(x > 0)))
df.to_pickle(out_fn)
print('saved %s (%d hours, %d columns)' % (out_fn, len(df), df.shape[1]))
