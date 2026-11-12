"""
Tidal and subtidal metric TIME SERIES at Penn Cove, wb1_t0_xn11abbur00, 2024-2025,
for 20261110_pc_metric_ts_plot.py. Signs: + = INTO the cove (west, toward the head).

SECTIONS pc_cp, pc_lj, pc_lp, from the pc_cove box: hourly ocean_his SNAPSHOTS (hh:00),
the same instants as the curtain and the rotation ring. Faces as in
20261110_pc_column_metrics.py (box xi_u 15 / 23 / 30, rows where both rho cells are wet);
u section-normal; element area dA = dz * dy, dz from z_w averaged across the face;
q = u dA. NB snapshot section sums alias sub-hourly motion into the subtidal band
([[snapshot-aliasing-budgets]]); chosen anyway, for one consistent time base.

  TIDAL (hourly)
    Q       = sum q                                   section transport [m3/s]
    Qprism  = 1/2 Godin|Q - Godin(Q)|                 LO bulk_calc definition, kept hourly
    umax    = max |u| over the elements;  u95 = area-weighted 95th percentile of |u|
    fopp    = area share with sign(u) != sign(Q)      counterflow against the section's net
    Usurf   = width-mean u of the top element         sign = surface direction
    dint    = depth of the surface/deep switch [m below the surface], from the width-mean
              profile on depth below the moving surface; runs of one sign with |U| >= 1 cm/s,
              >= 2 m thick, same-sign neighbours merged; NaN unless exactly two runs
              (rules and code copied from 20261110_pc_column_metrics_plot.py / 20261028)
  SUBTIDAL (Godin, hourly; first/last 35 h NaN)
    per element u_res = Godin(q) / Godin(dA), so the element residuals sum to Godin(Q)
    umax_res, fsea_res (area share with u_res < 0, i.e. flowing SEAWARD), Usurf_res,
    dint_res (same rules on the residual profile, depths from Godin(dz))

CURTAIN (vertical section): centre-line along-cove u of 20261110_pc_cove_layer_maps.py
(curtain_hourly.nc, his snapshots, hh:00, 30 columns head -> mouth). Element area
= dz * dx of the column (1/pm at the centre-line cell).
    fopp_cur      tidal: area share with sign(ual) != sign(Q_mouth), Q_mouth = Q at pc_lp
    fsea_cur_res  subtidal: area share with Godin(ual) < 0 (seaward); dz from Godin(zw)

BACK-COVE ROTATION (subtidal, his snapshots from the pc_cove box)
    Ring of wet rho cells at 0.5 r_eq .. r_eq about the FIXED median centre of the z4-10 eddy
    in its present hours (20261021_pc_eddy_center, pc_cp; the same ring rule, centre held fixed
    so the ring is one set of cells for the whole record). Per cell and s-level, azimuthal
    u_t (+ = counter-clockwise) is Godin filtered; ring mean per s-level.
    rot_U     depth-mean ring u_t  [m/s]
    rot_frac  share of the water column (dz-weighted s-levels) with ring-mean u_t > 0
    Uring     (time, s_rho) ring-mean u_t profile, saved for later
    Plus U_rot (moving-centre ring speed) and 'present' per band z0-4 / z4-10 / z10-bed,
    copied from the 20261021 csv files.

OUTPUT LO_output/DM_outs/20261110_pc_metric_ts/pc_metric_ts.nc
       dims time (hh:00), sect, band, s_rho

run 20261110_pc_metric_ts.py
"""
from time import time as clock

import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun, zfun

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SECTS = ['pc_cp', 'pc_lj', 'pc_lp']
SIGN = -1                                  # box u + is east = out of the cove
SECT_XI_U = {'pc_cp': 15, 'pc_lj': 23, 'pc_lp': 30}
UMIN, HMIN, DZG = 0.01, 2.0, 0.5           # run rules of 20261028_pc_mouth_interface.py
BANDS = ['z0-4', 'z4-10', 'z10-bed']
M_PER_DEG, LAT0 = 111.32e3, 48.23
BLK = 744

Ldir = Lfun.Lstart(gridname='wb1')
ext = Ldir['LOo'] / 'extract' / GTAGEX
dmo = Ldir['LOo'] / 'DM_outs'
out_dir = dmo / '20261110_pc_metric_ts'
Lfun.make_dir(out_dir)
lp = lambda a: zfun.lowpass(np.asarray(a, float), f='godin')
tt0 = clock()


# ------------------------------------------------------------------ helpers
def wpercentile(x, w, p):
    """Weighted percentile p (0-1) along axis 1 of x (t, n) with weights w (t, n)."""
    i = np.argsort(x, axis=1)
    xs, ws = np.take_along_axis(x, i, axis=1), np.take_along_axis(w, i, axis=1)
    cw = np.cumsum(ws, axis=1) / ws.sum(axis=1, keepdims=True)
    j = np.argmax(cw >= p, axis=1)
    return xs[np.arange(len(x)), j]


def on_depths(x, zr, zw, dy, DG):
    """Width-mean profile of x on depths DG below the moving surface (copied from
    20261110_pc_column_metrics.py). x, zr (t, k, p); zw (t, k+1, p); dy (p,). Returns (t, nd)."""
    d = (zw[:, -1:, :] - zr)[:, ::-1, :]
    xr_ = x[:, ::-1, :]
    K = d.shape[1]
    i1 = np.clip((d[:, :, None, :] < DG[None, None, :, None]).sum(axis=1), 1, K - 1)
    i0 = i1 - 1
    d0, d1 = np.take_along_axis(d, i0, axis=1), np.take_along_axis(d, i1, axis=1)
    x0, x1 = np.take_along_axis(xr_, i0, axis=1), np.take_along_axis(xr_, i1, axis=1)
    f = np.clip((DG[None, :, None] - d0) / (d1 - d0), 0, 1)
    xp = x0 + f * (x1 - x0)
    ok = DG[None, :, None] <= (zw[:, -1, :] - zw[:, 0, :])[:, None, :] - 0.25
    wgt = ok * dy[None, None, :]
    with np.errstate(invalid='ignore'):
        return (np.where(ok, xp, 0) * wgt).sum(axis=2) / wgt.sum(axis=2)


def runs_of(U):
    """[[sign, top_index, bottom_index], ...] after the 1 cm/s / 2 m rules (20261028)"""
    sg = np.where(np.isfinite(U) & (np.abs(U) >= UMIN), np.sign(U), 0)
    rr = []
    for n, s_ in enumerate(sg):
        if s_ == 0:
            continue
        if rr and rr[-1][0] == s_ and rr[-1][2] == n - 1:
            rr[-1][2] = n
        else:
            rr.append([s_, n, n])
    rr = [r for r in rr if (r[2] - r[1] + 1) * DZG >= HMIN]
    merged = []
    for r in rr:
        if merged and merged[-1][0] == r[0]:
            merged[-1][2] = r[2]
        else:
            merged.append(list(r))
    return merged


def divide(U, DG):
    """depth of the zero crossing between the two runs; NaN unless exactly two runs"""
    r = runs_of(U)
    if len(r) != 2:
        return np.nan
    a, b = r[0][2], r[1][1]
    seg, dd = U[a:b + 1], DG[a:b + 1]
    k = np.flatnonzero(np.sign(seg[:-1]) != np.sign(seg[1:]))
    if len(k):
        k = k[0]
        return dd[k] + (dd[k + 1] - dd[k]) * seg[k] / (seg[k] - seg[k + 1])
    return 0.5 * (dd[0] + dd[-1])


def section_metrics(u, dA, dy, zw, zr, ref):
    """umax, u95, share against ref (sign array (t,) or None = seaward share), Usurf, divide"""
    nt = len(u)
    au = np.abs(u).reshape(nt, -1)
    ok = np.isfinite(au).all(axis=1)
    umax, u95 = np.full(nt, np.nan), np.full(nt, np.nan)
    umax[ok] = au[ok].max(axis=1)
    u95[ok] = wpercentile(au[ok], dA.reshape(nt, -1)[ok], 0.95)
    bad = (u < 0) if ref is None else (np.sign(u) != ref[:, None, None])
    frac = (bad * dA).sum(axis=(1, 2)) / dA.sum(axis=(1, 2))
    W = dy / dy.sum()
    usurf = (u[:, -1, :] * W).sum(axis=1)
    DG = np.arange(0.25, np.nanmax(zw[:, -1] - zw[:, 0]) + 1.0, DZG)
    prof = on_depths(u, zr, zw, dy, DG)
    dint = np.array([divide(prof[t], DG) if ok[t] else np.nan for t in range(nt)])
    frac[~ok], usurf[~ok] = np.nan, np.nan
    return umax, u95, frac, usurf, dint


# ------------------------------------------------------------------ geometry: section faces and the rotation ring (pc_cove box)
box = xr.open_dataset(ext / 'box' / ('pc_cove_%s_%s.nc' % (DS0, DS1)))
T = box.ocean_time.values
NT, NZ = len(T), box.sizes['s_rho']
pm, pn, mask = box.pm.values, box.pn.values, box.mask_rho.values == 1
lon, lat = box.lon_rho.values, box.lat_rho.values
F = {}
for sn, XI_U in SECT_XI_U.items():
    IRM, IRP = XI_U, XI_U + 1
    ETA = np.flatnonzero(mask[:, IRM] & mask[:, IRP])[::-1]                     # north to south
    F[sn] = dict(XI_U=XI_U, IRM=IRM, IRP=IRP, ETA=ETA, dy=(1 / pn[ETA][:, [IRM, IRP]]).mean(axis=-1))
    print('%s: xi_u %d, %d faces' % (sn, XI_U, len(ETA)))
ec = dmo / '20261021_pc_eddy_center' / 'pc_cp'
E = {b: pd.read_csv(ec / ('pc_eddy_center_%s.csv' % b), parse_dates=['time']).set_index('time') for b in BANDS}
pr = E['z4-10'][E['z4-10'].present]
LONC, LATC, REQ = pr.lon_c.median(), pr.lat_c.median(), pr.r_eq.median()
X = (lon - LONC) * M_PER_DEG * np.cos(np.deg2rad(LAT0))
Y = (lat - LATC) * M_PER_DEG
R = np.hypot(X, Y)
DX = float(np.median(1 / pm[mask]))
interior = np.zeros_like(mask)
interior[1:-1, 1:-1] = True
ring = mask & interior & (R >= max(0.5 * REQ, DX)) & (R <= max(REQ, 2 * DX))
jr, ir = np.nonzero(ring)
print('ring: centre %.4f %.4f, r_eq %.0f m, %d cells' % (LONC, LATC, REQ, len(jr)))

# ------------------------------------------------------------------ one pass through the box, a month of hours at a time
U_S = {sn: np.full((NT, NZ, len(F[sn]['ETA'])), np.nan) for sn in SECTS}
ZW_S = {sn: np.full((NT, NZ + 1, len(F[sn]['ETA'])), np.nan) for sn in SECTS}
UT = np.full((NT, NZ, len(jr)), np.nan)
DZR = np.full((NT, NZ, len(jr)), np.nan)
for t0 in range(0, NT, BLK):
    tt = slice(t0, min(t0 + BLK, NT))
    U = box.u.isel(ocean_time=tt).values.astype(float)
    V = np.nan_to_num(box.v.isel(ocean_time=tt).values.astype(float))
    ZW = box.z_w.isel(ocean_time=tt).values
    for sn, c in F.items():
        U_S[sn][tt] = SIGN * U[:, :, c['ETA'], c['XI_U']]
        ZW_S[sn][tt] = ZW[:, :, c['ETA']][:, :, :, [c['IRM'], c['IRP']]].mean(axis=-1)
    U = np.nan_to_num(U)                                                        # walls = 0
    ur = 0.5 * (U[:, :, jr, ir - 1] + U[:, :, jr, ir])                          # u faces either side
    vr = 0.5 * (V[:, :, jr - 1, ir] + V[:, :, jr, ir])
    UT[tt] = (-Y[jr, ir] * ur + X[jr, ir] * vr) / R[jr, ir]                     # + counter-clockwise
    DZR[tt] = np.diff(ZW[:, :, jr, ir], axis=1)
    print('  hours %d-%d of %d, %.0f s' % (tt.start, tt.stop, NT, clock() - tt0))
box.close()

# ------------------------------------------------------------------ sections
S = {}
for sn in SECTS:
    u, zw, dy = U_S[sn], ZW_S[sn], F[sn]['dy']
    dz = np.diff(zw, axis=1)
    dA = dz * dy[None, None, :]
    q = u * dA
    Q = q.sum(axis=(1, 2))
    o = dict(Q=Q, Qprism=0.5 * lp(np.abs(Q - lp(Q))))
    zr = 0.5 * (zw[:, 1:] + zw[:, :-1])
    o['umax'], o['u95'], o['fopp'], o['Usurf'], o['dint'] = section_metrics(u, dA, dy, zw, zr, np.sign(Q))
    # subtidal: element residuals that sum to Godin(Q)
    qL, dzL = lp(q), lp(dz)
    dAL = dzL * dy[None, None, :]
    zwL = np.concatenate([lp(zw[:, :1]), lp(zw[:, :1]) + np.cumsum(dzL, axis=1)], axis=1)
    zwL = np.where(np.isfinite(zwL), zwL, zw)                                   # NaN pad -> raw (rows dropped below)
    zrL = 0.5 * (zwL[:, 1:] + zwL[:, :-1])
    with np.errstate(invalid='ignore', divide='ignore'):
        r = section_metrics(qL / dAL, np.nan_to_num(dAL), dy, zwL, zrL, None)
    o['umax_res'], o['u95_res'], o['fsea_res'], o['Usurf_res'], o['dint_res'] = r
    S[sn] = o
    print('%s: Qprism %.0f m3/s, fopp %.2f, two-layer %.0f%% tidal / %.0f%% residual hours, %.0f s'
          % (sn, np.nanmean(o['Qprism']), np.nanmean(o['fopp']), 100 * np.isfinite(o['dint']).mean(),
             100 * np.isfinite(o['dint_res']).mean(), clock() - tt0))

# ------------------------------------------------------------------ curtain (vertical section)
C = xr.open_dataset(dmo / '20261110_pc_cove_layer_maps' / 'curtain_hourly.nc')
assert (C.time.values == T).all()
ual, zwc = C.ual.values.astype(float), C.zw.values.astype(float)           # (t, k, col), (t, k+1, col)
C.close()
Qm = S['pc_lp']['Q']
# centre-line dx per column: 1/pm averaged over the column's wet cove cells (plaid grid: pm ~ f(xi))
dxc = np.array([np.mean(1 / pm[mask[:, c], c]) for c in range(1, 31)])
dzc = np.diff(zwc, axis=1)
ok = np.isfinite(ual)
aw0 = np.where(ok, dzc * dxc[None, None, :], 0)
fopp_cur = ((np.sign(ual) != np.sign(Qm)[:, None, None]) * aw0).sum(axis=(1, 2)) / aw0.sum(axis=(1, 2))
uL, awL = lp(ual), lp(dzc) * dxc[None, None, :]
okL = np.isfinite(uL)
fsea_cur_res = ((uL < 0) * np.where(okL, awL, 0)).sum(axis=(1, 2)) / np.where(okL, awL, 0).sum(axis=(1, 2))
fsea_cur_res[~okL.any(axis=(1, 2))] = np.nan
print('curtain: fopp %.2f, residual seaward share %.2f, %.0f s' % (np.nanmean(fopp_cur), np.nanmean(fsea_cur_res), clock() - tt0))

# ------------------------------------------------------------------ back-cove rotation
Uring = lp(UT).mean(axis=2)                                                     # (t, k)
dzr = lp(DZR).mean(axis=2)
rot_U = (Uring * dzr).sum(axis=1) / dzr.sum(axis=1)
rot_frac = ((Uring > 0) * dzr).sum(axis=1) / dzr.sum(axis=1)
rot_U[~np.isfinite(Uring).all(axis=1)] = np.nan
rot_frac[~np.isfinite(Uring).all(axis=1)] = np.nan
print('rotation: mean U %.1f cm/s, mean ccw share %.2f' % (100 * np.nanmean(rot_U), np.nanmean(rot_frac)))
Eb = {k: np.stack([E[b][k].reindex(pd.DatetimeIndex(T)).values.astype(float) for b in BANDS], axis=1) for k in ['U_rot', 'present']}

# ------------------------------------------------------------------ save
UNITS = dict(Q='m3/s', Qprism='m3/s', umax='m/s', u95='m/s', fopp='1', Usurf='m/s', dint='m',
             umax_res='m/s', u95_res='m/s', fsea_res='1', Usurf_res='m/s', dint_res='m')
out = xr.Dataset({k: (('time', 'sect'), np.stack([S[s][k] for s in SECTS], axis=1), {'units': UNITS[k]}) for k in UNITS},
                 coords={'time': T, 'sect': SECTS, 'band': BANDS, 's_rho': np.arange(NZ)})
out['fopp_cur'] = (('time',), fopp_cur, {'units': '1', 'long_name': 'curtain area share against Q_mouth'})
out['fsea_cur_res'] = (('time',), fsea_cur_res, {'units': '1', 'long_name': 'curtain area share with residual flow seaward'})
out['rot_U'] = (('time',), rot_U, {'units': 'm/s', 'long_name': 'ring depth-mean residual azimuthal u, + ccw'})
out['rot_frac'] = (('time',), rot_frac, {'units': '1', 'long_name': 'share of the water column turning ccw'})
out['Uring'] = (('time', 's_rho'), Uring.astype(np.float32), {'units': 'm/s', 'long_name': 'ring-mean residual azimuthal u per s-level (0 = bottom)'})
out['U_rot_band'] = (('time', 'band'), Eb['U_rot'], {'units': 'm/s', 'long_name': '20261021 moving-centre ring speed'})
out['present_band'] = (('time', 'band'), Eb['present'], {'long_name': '20261021 eddy present flag'})
out.attrs = {'sign': 'u, Q + into the cove (west); hourly ocean_his snapshots', 'ring': 'centre %.5f %.5f, r %.0f-%.0f m, %d cells'
             % (LONC, LATC, max(0.5 * REQ, DX), max(REQ, 2 * DX), len(jr)),
             'note': 'see the docstring of 20261110_pc_metric_ts.py'}
out.to_netcdf(out_dir / 'pc_metric_ts.nc')
print('saved %s, %.0f s' % (out_dir / 'pc_metric_ts.nc', clock() - tt0))
