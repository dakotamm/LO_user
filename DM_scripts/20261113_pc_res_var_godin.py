"""
Penn Cove response variable time series, Godin filtered: the same STEPS 1-8 as
20261112_pc_res_var_ts.py, computed from Godin-filtered model fields instead of the
hourly fields. So each variable describes the residual (subtidal) flow and density
field, e.g. the fastest residual inflow, not a tidal average of the hourly fastest inflow.

Sources as in 20261112_pc_res_var_ts.py: box extractions of wb1_t0_xn11abbur00 ocean_his
(2024-2025, hourly snapshots at hh:00) and the tef2 section extractions (hour-mean
ocean_avg, hh:30).

Filtering: LO's Godin filter (zfun.lowpass f='godin', 24-24-25 h), one hourly value per
hour kept, NaN for the first and last 35 hours.
    Velocity is filtered as a transport: u_res = Godin(u*dz) / Godin(dz) (and q, the
    tef2 element transport, is filtered directly), so the residual includes the
    correlation of tidal velocity with tidal layer thickness.
    Density is filtered the same way where it is averaged over depth:
    sigma_res = Godin(sigma0*dz) / Godin(dz). Single-level tracers (top, bottom) are
    filtered directly.
    Box snapshots alias sub-hourly motion into the subtidal band (worst for section
    sums, ~6x at pc_lp); the tef2 hour means do not.

Saved to LO_output/DM_outs/20261113_pc_res_var_godin/:
    res_var_godin.p       hh:00 index (STEPS 1, 3, 4, 5, 7, 8)
    res_var_godin_avg.p   hh:30 index (STEPS 2, 6)
Column names as in 20261112_pc_res_var_ts.py.

STEP 1: residual maximum velocity [m/s] at the three Penn Cove cross sections.
    Sections, cells and u at rho as in 20261112 STEP 1 (box rho column nearest each
    section line, two u faces averaged, land faces 0). dz from the box z_w.
    u_res = Godin(u*dz) / Godin(dz) in each cell and level, then
        umax    max |u_res|;  uin  max of -u_res (into the cove);  uout  max of u_res

run 20261113_pc_res_var_godin.py
"""
import gsw
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

from lo_tools import Lfun, zfun

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SECTS = ['pc_cp', 'pc_lj', 'pc_lp']

gridname, tag, ex_name = GTAGEX.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
box_dir = Ldir['LOo'] / 'extract' / GTAGEX / 'box'
cove_fn = box_dir / ('pc_cove_' + DS0 + '_' + DS1 + '.nc')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261113_pc_res_var_godin'
Lfun.make_dir(out_dir)
out_fn = out_dir / 'res_var_godin.p'

def godin(x):
    return zfun.lowpass(np.asarray(x, dtype=float), f='godin')

def stats(vn, x, units):
    print('    %-10s mean %+.3f, std %.3f, min %+.3f, max %+.3f %s'
          % (vn, np.nanmean(x), np.nanstd(x), np.nanmin(x), np.nanmax(x), units))

dc = xr.open_dataset(cove_fn)

T = dc.ocean_time.values
mask = dc.mask_rho.values == 1
lon_r = dc.lon_rho.values
lat_r = dc.lat_rho.values

df = pd.DataFrame(index=pd.DatetimeIndex(T, name='time'))
for sn in SECTS:
    line = pd.read_pickle(Ldir['LOo'] / 'section_lines' / (sn + '.p'))
    ii = np.argmin(np.abs(lon_r[0, :] - line.x.values[0]))
    jj = np.nonzero(mask[:, ii] & (lat_r[:, ii] >= line.y.min()) & (lat_r[:, ii] <= line.y.max()))[0]
    uf = np.nan_to_num(dc.u.isel(eta_u=jj, xi_u=[ii - 1, ii]).values)    # (time, s_rho, cell, west/east face)
    u = 0.5 * (uf[..., 0] + uf[..., 1])                                    # (time, s_rho, cell) at rho
    dz = np.diff(dc.z_w.isel(eta_rho=jj, xi_rho=ii).values, axis=1)
    u = godin(u * dz) / godin(dz)
    df['umax_' + sn] = np.abs(u).max(axis=(1, 2))
    df['uin_' + sn] = (-u).max(axis=(1, 2))
    df['uout_' + sn] = u.max(axis=(1, 2))
    print('%s: xi_rho %d, %d cells' % (sn, ii, len(jj)))
    for vn in ['umax', 'uin', 'uout']:
        stats(vn, df[vn + '_' + sn], 'm/s')
df.to_pickle(out_fn)
print('saved %s (%d hours, %d columns)' % (out_fn, len(df), df.shape[1]))

"""
STEP 2: residual inflow and outflow transport Q_in, Q_out [m3/s] at the three sections,
from the tef2 section extractions (hour-mean Huon q per element, hh:30).
    q_res = Godin(q) in each element (z, p), + into the cove, then
        Q_in = sum of q_res > 0,  Q_out = sum of q_res < 0,  Q_net = Q_in + Q_out
    The Eulerian residual in/out transport: every element counts by the sign of its
    own residual, so it is not TEF.
"""
ext_dir = Ldir['LOo'] / 'extract' / GTAGEX / 'tef2' / ('extractions_avg_' + DS0 + '_' + DS1)
out_fn_avg = out_dir / 'res_var_godin_avg.p'

da = pd.DataFrame()
for sn in SECTS:
    de = xr.open_dataset(ext_dir / (sn + '.nc'))
    q = godin(-de.q.values)                                                # (time, z, p), + into the cove
    if da.empty:
        da = pd.DataFrame(index=pd.DatetimeIndex(de.time.values, name='time'))
    else:
        assert (da.index == de.time.values).all()
    bad = np.isnan(q).any(axis=(1, 2))
    da['Qin_' + sn] = np.where(bad, np.nan, np.where(q > 0, q, 0).sum(axis=(1, 2)))
    da['Qout_' + sn] = np.where(bad, np.nan, np.where(q < 0, q, 0).sum(axis=(1, 2)))
    da['Qnet_' + sn] = q.sum(axis=(1, 2))
    print('%s: %d elements' % (sn, q.shape[1] * q.shape[2]))
    for vn in ['Qin', 'Qout', 'Qnet']:
        stats(vn, da[vn + '_' + sn], 'm3/s')
da.to_pickle(out_fn_avg)
print('saved %s (%d hours, %d columns)' % (out_fn_avg, len(da), da.shape[1]))

"""
STEP 3: residual whole-cove top-bottom density difference split into its salinity and
temperature parts [kg/m3].
    Cove cells, sigma0, the salinity/temperature split and the area weighting as in
    20261112 STEP 3, from Godin(salt) and Godin(temp) at the bottom (s_rho 0) and top
    (s_rho -1) levels.
        drhoS_pc, drhoT_pc
"""
LON, LAT = -122.69, 48.23

def sigma0(s, t):
    SA = gsw.SA_from_SP(s, 0, LON, LAT)
    return gsw.sigma0(SA, gsw.CT_from_pt(SA, t))

cove = mask.copy()
cove[:, 31:] = False
jc, ic = np.nonzero(cove)
dA = 1 / (dc.pm.values[jc, ic] * dc.pn.values[jc, ic])

def cmean(x):
    """Area-weighted mean over the cove cells of a (time, cell) array."""
    return (x * dA).sum(axis=1) / dA.sum()

s = godin(dc.salt.isel(s_rho=[0, -1]).values[:, :, jc, ic])             # (time, bot/top, cell)
t = godin(dc.temp.isel(s_rho=[0, -1]).values[:, :, jc, ic])
sig_bot = sigma0(s[:, 0], t[:, 0])                                       # (time, cell)
sig_top = sigma0(s[:, 1], t[:, 1])
df['drhoS_pc'] = cmean(sigma0(s[:, 0], t[:, 1]) - sig_top)
df['drhoT_pc'] = cmean(sig_bot - sigma0(s[:, 0], t[:, 1]))

print('cove: %d cells, %.2f km2' % (len(jc), dA.sum() / 1e6))
for vn in ['drhoS_pc', 'drhoT_pc']:
    stats(vn, df[vn], 'kg/m3')
df.to_pickle(out_fn)
print('saved %s (%d hours, %d columns)' % (out_fn, len(df), df.shape[1]))

"""
STEP 4: residual along-cove density gradient, head minus mouth [kg/m3 per km], at the top
and bottom levels and depth-averaged.
    Head, mouth, L and area weighting as in 20261112 STEP 4.
        drhodx_top, drhodx_bot    from the STEP 3 residual sigma0 at the top / bottom level
        drhodx_bar                depth mean in each cell = Godin(sum sigma0*dz) / Godin(sum dz)
    All 30 levels of salt, temp, z_rho, z_w are read here in blocks of NB hours and kept
    (float32) for STEP 5.
"""
head = ic <= 15
mouth = (ic >= 24) & (ic <= 30)

def amean(x, r):
    return (x[:, r] * dA[r]).sum(axis=1) / dA[r].sum()

lon_c, lat_c = lon_r[jc, ic], lat_r[jc, ic]
dx = (lon_c[head].mean() - lon_c[mouth].mean()) * 111.32 * np.cos(np.deg2rad(LAT))
dy = (lat_c[head].mean() - lat_c[mouth].mean()) * 111.32
L = np.hypot(dx, dy)                                                      # km

NT, NZ, NC3 = len(T), dc.sizes['s_rho'], len(jc)
SG = np.full((NT, NZ, NC3), np.nan, dtype=np.float32)                    # sigma0
DZ = np.full((NT, NZ, NC3), np.nan, dtype=np.float32)
ZR = np.full((NT, NZ, NC3), np.nan, dtype=np.float32)
NB = 1000
for t0 in range(0, NT, NB):
    tt = slice(t0, t0 + NB)
    sb = dc.salt.isel(ocean_time=tt).values[:, :, jc, ic]                 # (time, s_rho, cell)
    tb = dc.temp.isel(ocean_time=tt).values[:, :, jc, ic]
    SG[tt] = sigma0(sb, tb)
    DZ[tt] = np.diff(dc.z_w.isel(ocean_time=tt).values[:, :, jc, ic], axis=1)
    ZR[tt] = dc.z_rho.isel(ocean_time=tt).values[:, :, jc, ic]
    print('  read block %d-%d' % (t0, min(t0 + NB, NT)), flush=True)

sig_bar = godin((SG * DZ).sum(axis=1)) / godin(DZ.sum(axis=1))            # (time, cell)

df['drhodx_top'] = (amean(sig_top, head) - amean(sig_top, mouth)) / L
df['drhodx_bot'] = (amean(sig_bot, head) - amean(sig_bot, mouth)) / L
df['drhodx_bar'] = (amean(sig_bar, head) - amean(sig_bar, mouth)) / L

print('head %d cells, mouth %d cells; L = %.2f km' % (head.sum(), mouth.sum(), L))
for vn in ['drhodx_top', 'drhodx_bot', 'drhodx_bar']:
    stats(vn, df[vn], 'kg/m3/km')
df.to_pickle(out_fn)
print('saved %s (%d hours, %d columns)' % (out_fn, len(df), df.shape[1]))

"""
STEP 5: residual whole-cove potential energy anomaly phi [J/m3] (Simpson et al. 1981).
    phi as in 20261112 STEP 5, from the residual column in each cell:
        sigma_res = Godin(sigma0*dz) / Godin(dz),  dz_res = Godin(dz),  z_res = Godin(z_rho)
        phi = (g/D) * sum of (sigma_bar - sigma_res) * z_res * dz_res,  D = sum dz_res
        phi_pc      area-weighted cove mean
"""
G = 9.81
dz_res = godin(DZ)
sg_res = godin(SG * DZ) / dz_res
z_res = godin(ZR)
del SG, ZR

D = dz_res.sum(axis=1)                                                     # (time, cell)
sg_bar = (sg_res * dz_res).sum(axis=1) / D
phi_cell = (G / D) * ((sg_bar[:, None, :] - sg_res) * z_res * dz_res).sum(axis=1)
df['phi_pc'] = cmean(phi_cell)
del dz_res, sg_res, z_res, DZ

stats('phi_pc', df['phi_pc'], 'J/m3')
df.to_pickle(out_fn)
print('saved %s (%d hours, %d columns)' % (out_fn, len(df), df.shape[1]))

"""
STEP 6: residual shape of the flow across the three sections, from the tef2 extractions
(hh:30).
    The 20261112 STEP 6 metrics 1-9 (definitions there), from the residual elements:
        q_res = Godin(q),  DZ_res = Godin(DZ),  dA = dd*DZ_res,  u_res = q_res/dA
    and sig from DZ_res. Hours inside the filter's NaN ends are NaN for every metric.
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
    q = godin(-de.q.values)                                                # (time, z, p), + into the cove
    dz = godin(de.DZ.values)
    dd = de.dd.values
    NTa, NZ, NP = q.shape
    ok = np.isfinite(q).all(axis=(1, 2))
    dA_e = dz * dd[None, None, :]
    u = q / dA_e
    y = 1 - (np.arange(NP) + 0.5) / NP                                     # faces run north -> south
    sig = (np.cumsum(dz, axis=1) - dz / 2) / dz.sum(axis=1)[:, None, :] - 1
    qin = np.where(q > 0, q, 0)
    qout = np.where(q < 0, -q, 0)
    A = dA_e.sum(axis=(1, 2))

    M = {}
    # 1 surface and bottom direction
    M['Usurf'] = q[:, -1, :].sum(axis=1) / dA_e[:, -1, :].sum(axis=1)
    M['Ubot'] = q[:, 0, :].sum(axis=1) / dA_e[:, 0, :].sum(axis=1)
    # 2 north and south edge direction
    M['Unorth'] = q[:, :, 0].sum(axis=1) / dA_e[:, :, 0].sum(axis=1)
    M['Usouth'] = q[:, :, -1].sum(axis=1) / dA_e[:, :, -1].sum(axis=1)
    # 3 inflow area share
    M['ain'] = (dA_e * (q > 0)).sum(axis=(1, 2)) / A
    # 4 layered thickness, 5 lateral width
    Qz = q.sum(axis=2)
    Az = dA_e.sum(axis=2)
    M['Hin'] = (Az * (Qz > 0)).sum(axis=1) / A
    M['Hout'] = (Az * (Qz < 0)).sum(axis=1) / A
    Qp = q.sum(axis=1)
    Ap = dA_e.sum(axis=1)
    M['Win'] = (dd * (Qp > 0)).sum(axis=1) / dd.sum()
    M['Wout'] = (dd * (Qp < 0)).sum(axis=1) / dd.sum()
    # 6 where the inflow and outflow sit
    with np.errstate(invalid='ignore'):
        M['zin'] = (qin * sig).sum(axis=(1, 2)) / qin.sum(axis=(1, 2))
        M['zout'] = (qout * sig).sum(axis=(1, 2)) / qout.sum(axis=(1, 2))
        M['yin'] = (qin * y).sum(axis=(1, 2)) / qin.sum(axis=(1, 2))
        M['yout'] = (qout * y).sum(axis=(1, 2)) / qout.sum(axis=(1, 2))
    # 7 exchange ratio
    Qi, Qo = qin.sum(axis=(1, 2)), qout.sum(axis=(1, 2))
    with np.errstate(invalid='ignore'):                                    # 0/0 in the NaN ends
        M['rex'] = np.minimum(Qi, Qo) / np.maximum(Qi, Qo)
    # 8 layered share of the counterflow variance
    ubar = q.sum(axis=(1, 2)) / A
    ucol = Qp / Ap
    v_tot = (dA_e * (u - ubar[:, None, None])**2).sum(axis=(1, 2)) / A
    v_lay = (dA_e * (u - ucol[:, None, :])**2).sum(axis=(1, 2)) / A
    M['flay'] = v_lay / v_tot
    # 9 counts
    for vn in ['nlay', 'nlat', 'npin', 'npout']:
        M[vn] = np.full(NTa, np.nan)
    for k in np.flatnonzero(ok):
        M['nlay'][k] = n_runs(Qz[k])
        M['nlat'][k] = n_runs(Qp[k])
        M['npin'][k] = n_patches(q[k])
        M['npout'][k] = n_patches(-q[k])

    print('%s: %d levels x %d faces' % (sn, NZ, NP))
    for vn in M:
        M[vn] = np.where(ok, M[vn], np.nan)
        da[vn + '_' + sn] = M[vn]
        if vn in ['nlay', 'nlat', 'npin', 'npout']:
            c = np.bincount(M[vn][ok].astype(int), minlength=5)
            print('    %-10s %% of hours with 0/1/2/3/4+: %s' % (vn, ' '.join('%.0f' % v for v in 100 * np.r_[c[:4], c[4:].sum()] / ok.sum())))
        else:
            stats(vn, M[vn], '')

da.to_pickle(out_fn_avg)
print('saved %s (%d hours, %d columns)' % (out_fn_avg, len(da), da.shape[1]))

"""
STEP 7: residual back-bay eddy rotation.
    Ring found as in 20261112 STEP 7 (2-year mean depth-mean flow; fastest closed ring
    over back-bay centres and R = 400-1200 m). In each ring cell and level the hourly
    azimuthal velocity u_t (+ = counter-clockwise), then
        u_t_res = Godin(u_t*dz) / Godin(dz),  dz_res = Godin(dz)
        Ut_ed, fccw_ed, fcol_ed    as in 20261112 STEP 7, from u_t_res and dz_res
"""
interior = np.zeros_like(mask)
interior[1:-1, 1:-1] = True
dA_all = 1 / (dc.pm.values * dc.pn.values)
NR, NCg = mask.shape
ubm = np.nan_to_num(dc.ubar.values).mean(axis=0)
vbm = np.nan_to_num(dc.vbar.values).mean(axis=0)
urm = np.full((NR, NCg), np.nan)
vrm = np.full((NR, NCg), np.nan)
urm[:, 1:-1] = 0.5 * (ubm[:, :-1] + ubm[:, 1:])
vrm[1:-1, :] = 0.5 * (vbm[:-1, :] + vbm[1:, :])

def ring_xy(j, i):
    """East / north distance [m] and radius of every rho cell from cell (j, i)."""
    X = (lon_r - lon_r[j, i]) * 111.32e3 * np.cos(np.deg2rad(LAT))
    Y = (lat_r - lat_r[j, i]) * 111.32e3
    return X, Y, np.hypot(X, Y)

best = (-np.inf, None, None)
back = mask & interior & (np.arange(NCg)[None, :] <= 15)
for j, i in zip(*np.nonzero(back)):
    X, Y, R = ring_xy(j, i)
    with np.errstate(invalid='ignore', divide='ignore'):
        utm = (-Y * urm + X * vrm) / R
    for Rm in range(400, 1201, 50):
        rg = mask & interior & (R >= Rm / 2) & (R <= Rm)
        if dA_all[rg].sum() < 0.95 * np.pi * 0.75 * Rm**2:
            continue
        if utm[rg].mean() > best[0]:
            best = (utm[rg].mean(), (j, i), Rm)

Ut_mean, (jc_ed, ic_ed), R_ed = best
X, Y, R = ring_xy(jc_ed, ic_ed)
ring = mask & interior & (R >= R_ed / 2) & (R <= R_ed)
jr, ir = np.nonzero(ring)
j0, i0 = jr.min(), ir.min()
print('eddy: centre %.4f %.4f, R %d m, %d ring cells' % (lon_r[jc_ed, ic_ed], lat_r[jc_ed, ic_ed], R_ed, len(jr)))

uf = np.nan_to_num(dc.u.isel(eta_u=slice(j0, jr.max() + 1), xi_u=slice(i0 - 1, ir.max() + 1)).values)
vf = np.nan_to_num(dc.v.isel(eta_v=slice(j0 - 1, jr.max() + 1), xi_v=slice(i0, ir.max() + 1)).values)
zw = dc.z_w.isel(eta_rho=slice(j0, jr.max() + 1), xi_rho=slice(i0, ir.max() + 1)).values
ur = 0.5 * (uf[:, :, jr - j0, ir - i0] + uf[:, :, jr - j0, ir - i0 + 1])     # (time, s_rho, cell)
vr = 0.5 * (vf[:, :, jr - j0, ir - i0] + vf[:, :, jr - j0 + 1, ir - i0])
dz = np.diff(zw[:, :, jr - j0, ir - i0], axis=1)
ut = (-Y[jr, ir] * ur + X[jr, ir] * vr) / R[jr, ir]

dz_res = godin(dz)
ut_res = godin(ut * dz) / dz_res

Uring = ut_res.mean(axis=2)                                                # (time, s_rho)
dzr = dz_res.mean(axis=2)
ucol = (ut_res * dz_res).sum(axis=1) / dz_res.sum(axis=1)                  # (time, cell)
ok = np.isfinite(ut_res).all(axis=(1, 2))
df['Ut_ed'] = (Uring * dzr).sum(axis=1) / dzr.sum(axis=1)
df['fccw_ed'] = np.where(ok, (ucol > 0).mean(axis=1), np.nan)
df['fcol_ed'] = np.where(ok, ((Uring > 0) * dzr).sum(axis=1) / dzr.sum(axis=1), np.nan)

for vn in ['Ut_ed', 'fccw_ed', 'fcol_ed']:
    stats(vn, df[vn], '')
df.to_pickle(out_fn)
print('saved %s (%d hours, %d columns)' % (out_fn, len(df), df.shape[1]))

"""
STEP 8: residual along-cove velocity in the bottom half of the water column, in the north
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
        UbotN_bb, UbotS_bb    Godin(sum u_al*dz*dA) / Godin(sum dz*dA)
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
    df['Ubot%s_bb' % hn] = godin(num[hn]) / godin(den[hn])
    stats('Ubot%s_bb' % hn, df['Ubot%s_bb' % hn], 'm/s')
df.to_pickle(out_fn)
print('saved %s (%d hours, %d columns)' % (out_fn, len(df), df.shape[1]))
