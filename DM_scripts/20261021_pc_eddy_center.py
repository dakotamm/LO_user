"""
Where is the inner Penn Cove eddy centre, hour by hour, WHEN IS THERE AN EDDY,
and how far does it drift? Feeds 20261021_pc_eddy_winding.py, which counts
particle winding about this (moving) centre during present hours only.

FIELD: from the pc_cove box (hourly, native C-grid), u and v are averaged over
a band of the water column (thickness-weighted, fractional height above the
bed) on their own faces, then Godin-filtered. Masked faces are zero (the wall
value), as in 20260819_pc_planview_circulation.py. Bands:
  surf  top 25%     mid  25-75%     bot  lower 25%
  lower lower 75% (mid + bot, where the Nov-Dec eddy is)     dm  whole column
  z<a>-<b>  fixed depth bands, a to b m below the moving surface (-zedges,
            default 0-4, 4-10, 10-bed; inner cove median depth 12.7 m, max 21).
            A depth band exists only in rho cells whose mean column depth is
            >= a + -zmin; elsewhere it is land for that band (faces zeroed, as
            walls), so its interior psi points, search region and ring are its own.

VORTICITY at psi points straight from the faces (dv/dx - du/dy), only where
all four surrounding rho cells are wet, so wall shear never enters. Okubo-Weiss
OW = sn^2 + ss^2 - zeta^2 (sn = du/dx - dv/dy from rho, averaged to psi; ss =
dv/dx + du/dy at psi).

SEARCH REGION: interior psi points west of the -east section line (pc_cp =
inner cove, pc_lp = whole cove), optionally south of -lat1 (none by default).

CENTRE each hour:
  core       psi points in the search region with sense*zeta > 0 AND
             OW < -ow_k * sd(OW) (Isern-Fontanet et al. 2003; sd over the
             search region and the whole record, per band), split into
             connected patches (4-connectivity); the patch with the largest
             circulation is THE core
  x_c, y_c   that patch's centroid weighted by sense*zeta (m; x from the pc_lp
             mouth, y from 48.23 N, same local frame as the pcmap looping scripts)
  x_pk, y_pk the psi point of maximum sense*zeta in that patch
  circ       sum(sense*zeta * area) over that patch [m2/s];  circ_all  all patches
  r_eq       sqrt(patch area / pi) [m]
  T_turn_h   solid-body turnover 4 pi / mean core zeta = 4 pi area / circ [h]

PRESENCE (decided from the Eulerian field only, never from particles, so the
particle test stays non-circular):
  ring       wet rho cells at distance [max(0.5 r_eq, dx), max(r_eq, 2 dx)]
             from the centre; u_t = sense * tangential velocity (+ = with the
             eddy). ring_frac = share of ring cells with u_t > 0; quad_ok = all
             four quadrants (by angle) have ring cells and mean u_t > 0
             (closed circulation; cf. Nencioli et al. 2010). U_rot = ring-mean u_t.
  present_raw  core exists AND r_eq >= -req_min AND ring_frac >= -ring_thr
               AND quad_ok
  EVENTS     runs of present_raw hours, broken where the centre jumps more than
             r_eq in one hour; kept if the run lasts >= -n_turn turnovers
             (median T_turn_h over the run). present = hour in a kept event.
  Per event: start, end, dur_h, T_turn_h, n_turns, centre, r_eq, circ, U_rot,
  c_drift (median 24-h centre displacement / 24 h) and U_rot / c_drift (> 1:
  rotation outruns translation, so the eddy can carry water; Chelton et al. 2011).

SENSITIVITY (bands -sweep_band, comma list): ring_thr 0.6/0.75/0.9 x ow_k 0/0.2/0.5 x
n_turn 1/2 -> % hours present (all, Nov-Dec, Aug-Oct), events, median length.

Output: LO_output/DM_outs/20261021_pc_eddy_center/<east>[_lat<lat1>]/
  pc_eddy_center_<band>.csv  hourly (Godin-valid hours), incl. present flags
  pc_eddy_events_<band>.csv  kept events
  pc_eddy_sensitivity.csv    the sweep
  pc_eddy_center_map.png     time-mean zeta per band + daily centres of present hours
  pc_eddy_center_series.png  first -sweep_band: centre offset, circ, r_eq, ring_frac; present shaded
  report.txt

  python 20261021_pc_eddy_center.py               (inner cove, west of pc_cp)
  python 20261021_pc_eddy_center.py -east pc_lp    (whole cove)
  -cache <file.npz>  save / reuse the Godin-filtered band velocities (~0.5 GB)
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import ndimage

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-job', default='pc_cove')
p.add_argument('-0', '--ds0', default='2024.01.01')
p.add_argument('-1', '--ds1', default='2025.12.31')
p.add_argument('-east', default='pc_cp', help='section line (N-S) bounding the search region on the east')
p.add_argument('-lat1', type=float, default=None, help='optional: also south of this lat')
p.add_argument('-sense', type=int, default=1, help='+1 counterclockwise eddy, -1 clockwise')
p.add_argument('-ow_k', type=float, default=0.2, help='core: OW < -ow_k * sd(OW)')
p.add_argument('-ring_thr', type=float, default=0.75, help='min share of ring cells turning with the eddy')
p.add_argument('-req_min', type=float, default=400, help='min core radius [m] (~2 dx)')
p.add_argument('-n_turn', type=float, default=1, help='min event length in turnovers')
p.add_argument('-zedges', default='0,4,10,99', help='depth-band edges below the surface [m]')
p.add_argument('-zmin', type=float, default=2.0, help='min water in a depth band for a cell to have it [m]')
p.add_argument('-sweep_band', default='lower,z4-10,z10-bed', help='comma list')
p.add_argument('-chunk', type=int, default=720, help='hours read per chunk')
p.add_argument('-cache', default=None, help='npz of Godin-filtered band velocities (written if absent)')
args = p.parse_args()
SWB = args.sweep_band.split(',')

warnings.simplefilter('ignore')
Ldir = Lfun.Lstart(gridname='wb1')
box_fn = Ldir['LOo'] / 'extract' / args.gtx / 'box' / ('%s_%s_%s.nc' % (args.job, args.ds0, args.ds1))
tag = args.east + ('' if args.lat1 is None else '_lat%.4f' % args.lat1)
out_dir = Ldir['LOo'] / 'DM_outs' / '20261021_pc_eddy_center' / tag
Lfun.make_dir(out_dir)
M_PER_DEG = 111.32e3
LAT0 = 48.23
LON_MOUTH = pd.read_pickle(Ldir['LOo'] / 'section_lines' / 'pc_lp.p').x.mean()
LON1 = pd.read_pickle(Ldir['LOo'] / 'section_lines' / (args.east + '.p')).x.mean()   # pc_cp, pc_lp are N-S lines
LAT1 = np.inf if args.lat1 is None else args.lat1

ds = xr.open_dataset(box_fn)
lon, lat = ds.lon_rho.values, ds.lat_rho.values
mask = ds.mask_rho.values.astype(bool)
pm, pn = ds.pm.values, ds.pn.values
tt = pd.to_datetime(ds.ocean_time.values)
NT = len(tt)
NR, NC = mask.shape
print('box %s: %d hours, %d x %d rho' % (box_fn.name, NT, NR, NC))

BANDS = {'surf': lambda f, d: f >= 0.75, 'mid': lambda f, d: (f > 0.25) & (f < 0.75),
         'bot': lambda f, d: f <= 0.25, 'lower': lambda f, d: f < 0.75, 'dm': lambda f, d: f >= 0}
Hm = ds.h.values + ds.zeta.mean('ocean_time').values                      # mean column depth
MB = {b: mask for b in BANDS}                                              # rho cells that have the band
ze = [float(v) for v in args.zedges.split(',')]
for a, z in zip(ze[:-1], ze[1:]):
    b = 'z%g-%g' % (a, z) if z < 99 else 'z%g-bed' % a
    BANDS[b] = lambda f, d, a=a, z=z: (d >= a) & (d < z)
    MB[b] = mask & (Hm >= a + args.zmin)


def godin(A):
    B = A.reshape(NT, -1)
    G = np.full(B.shape, np.nan, np.float32)
    for m in range(B.shape[1]):
        G[:, m] = zfun.lowpass(B[:, m].astype(float), f='godin')
    return G.reshape(A.shape)


# --------------------------- band-mean face velocities, Godin-filtered ---
if args.cache and Path(args.cache).is_file() and all('U_' + b in np.load(args.cache).files for b in BANDS):
    Z = np.load(args.cache)
    UG = {b: Z['U_' + b] for b in BANDS}
    VG = {b: Z['V_' + b] for b in BANDS}
    print('read cache ' + args.cache)
else:
    U = {b: np.zeros((NT, NR, NC - 1), np.float32) for b in BANDS}
    V = {b: np.zeros((NT, NR - 1, NC), np.float32) for b in BANDS}
    for t0 in range(0, NT, args.chunk):
        sl = slice(t0, min(t0 + args.chunk, NT))
        zw = ds.z_w[sl].values
        u = np.nan_to_num(ds.u[sl].values)
        v = np.nan_to_num(ds.v[sl].values)
        dz = np.diff(zw, axis=1)
        frac = (0.5 * (zw[:, :-1] + zw[:, 1:]) - zw[:, :1]) / (zw[:, -1:] - zw[:, :1])
        dzu, dzv = 0.5 * (dz[..., :-1] + dz[..., 1:]), 0.5 * (dz[..., :-1, :] + dz[..., 1:, :])
        fu, fv = 0.5 * (frac[..., :-1] + frac[..., 1:]), 0.5 * (frac[..., :-1, :] + frac[..., 1:, :])
        dsf = zw[:, -1:] - 0.5 * (zw[:, :-1] + zw[:, 1:])                    # depth below the surface
        du_, dv_ = 0.5 * (dsf[..., :-1] + dsf[..., 1:]), 0.5 * (dsf[..., :-1, :] + dsf[..., 1:, :])
        for b, sel in BANDS.items():
            wu, wv = np.where(sel(fu, du_), dzu, 0.0), np.where(sel(fv, dv_), dzv, 0.0)
            U[b][sl] = np.nan_to_num((u * wu).sum(1) / wu.sum(1))
            V[b][sl] = np.nan_to_num((v * wv).sum(1) / wv.sum(1))
        print('  read %s to %s' % (tt[sl.start].date(), tt[sl.stop - 1].date()), flush=True)
    UG = {b: godin(U[b]) for b in BANDS}
    VG = {b: godin(V[b]) for b in BANDS}
    del U, V
    if args.cache:
        np.savez(args.cache, **{'U_' + b: UG[b] for b in BANDS}, **{'V_' + b: VG[b] for b in BANDS})
        print('wrote cache ' + args.cache)
ds.close()

# ------------------------------------------------------- geometry ---
wet4 = mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]       # (NR-1, NC-1)
c4 = lambda a: 0.25 * (a[:-1, :-1] + a[:-1, 1:] + a[1:, :-1] + a[1:, 1:])
lon_p, lat_p = c4(lon), c4(lat)
pm_p, pn_p = c4(pm), c4(pn)
area_p = 1 / (pm_p * pn_p)
to_x = lambda lo: (lo - LON_MOUTH) * M_PER_DEG * np.cos(np.deg2rad(LAT0))
to_y = lambda la: (la - LAT0) * M_PER_DEG
xp, yp = to_x(lon_p), to_y(lat_p)
xr_, yr_ = to_x(lon), to_y(lat)                                            # rho
search = wet4 & (lon_p < LON1) & (lat_p < LAT1)
DX = float(np.median(1 / pm[mask]))
W4, SR = {}, {}
for b in BANDS:                         # zero faces next to cells without the band (walls), per-band geometry
    UG[b] = UG[b] * (MB[b][:, :-1] & MB[b][:, 1:])[None]
    VG[b] = VG[b] * (MB[b][:-1, :] & MB[b][1:, :])[None]
    W4[b] = MB[b][:-1, :-1] & MB[b][:-1, 1:] & MB[b][1:, :-1] & MB[b][1:, 1:]
    SR[b] = W4[b] & (lon_p < LON1) & (lat_p < LAT1)
    print('  band %-8s %3d rho cells, %3d search psi points' % (b, MB[b].sum(), SR[b].sum()))
print('search region %s: %d interior psi points (of %d interior in the box); dx %.0f m'
      % (tag, search.sum(), wet4.sum(), DX))


def vort_ow(Ug, Vg):
    """zeta and OW at psi points, (T, NR-1, NC-1); NaN where not interior."""
    dvdx = (Vg[:, :, 1:] - Vg[:, :, :-1]) * pm_p
    dudy = (Ug[:, 1:, :] - Ug[:, :-1, :]) * pn_p
    zeta, ss = dvdx - dudy, dvdx + dudy
    sn = np.full((Ug.shape[0], NR, NC), np.nan, np.float32)            # at rho
    sn[:, 1:-1, 1:-1] = ((Ug[:, 1:-1, 1:] - Ug[:, 1:-1, :-1]) * pm[1:-1, 1:-1]
                         - (Vg[:, 1:, 1:-1] - Vg[:, :-1, 1:-1]) * pn[1:-1, 1:-1])
    sn_p = 0.25 * (sn[:, :-1, :-1] + sn[:, :-1, 1:] + sn[:, 1:, :-1] + sn[:, 1:, 1:])
    ow = sn_p ** 2 + ss ** 2 - zeta ** 2
    bad = ~wet4[None]
    return np.where(bad, np.nan, zeta), np.where(bad, np.nan, ow)


def to_rho(Ug, Vg):
    """Face velocities -> interior rho points (walls are zero faces); NaN at the box edge."""
    ur = np.full((Ug.shape[0], NR, NC), np.nan, np.float32)
    vr = np.full((Ug.shape[0], NR, NC), np.nan, np.float32)
    ur[:, :, 1:-1] = 0.5 * (Ug[:, :, :-1] + Ug[:, :, 1:])
    vr[:, 1:-1, :] = 0.5 * (Vg[:, :-1, :] + Vg[:, 1:, :])
    return ur, vr


def centres(zeta, ow, thr, tt_ok):
    """Largest-circulation core patch each hour -> hourly table."""
    core = search[None] & (args.sense * zeta > 0) & (ow < thr)
    wz = np.where(core, args.sense * zeta, 0.0)
    circ_all = (wz * area_p).reshape(wz.shape[0], -1).sum(1)
    w = np.zeros_like(wz)
    npatch = np.zeros(wz.shape[0], int)
    for k in range(wz.shape[0]):
        L, n = ndimage.label(core[k])
        npatch[k] = n
        if n == 0:
            continue
        cp = ndimage.sum(wz[k] * area_p, L, index=np.arange(1, n + 1))
        w[k] = np.where(L == np.argmax(cp) + 1, wz[k], 0.0)
    W = w.reshape(w.shape[0], -1)
    sw = W.sum(1)
    has = sw > 0
    cen = lambda f: np.where(has, (W * f.ravel()).sum(1) / np.where(has, sw, 1), np.nan)
    kpk = W.argmax(1)
    circ = (W * area_p.ravel()).sum(1)
    area = ((W > 0) * area_p.ravel()).sum(1)
    return pd.DataFrame(dict(time=tt_ok, lon_c=cen(lon_p), lat_c=cen(lat_p), x_c=cen(xp), y_c=cen(yp),
                             x_pk=np.where(has, xp.ravel()[kpk], np.nan),
                             y_pk=np.where(has, yp.ravel()[kpk], np.nan),
                             zeta_pk=np.where(has, W.max(1), np.nan), circ=circ, circ_all=circ_all,
                             n_patch=npatch, r_eq=np.sqrt(area / np.pi),
                             T_turn_h=np.where(has, 4 * np.pi * area / np.where(has, circ, 1) / 3600, np.nan)))


def ring_test(C, ur, vr):
    """Closed-circulation test on a ring of wet rho cells around each hour's centre."""
    ok = np.isfinite(ur[0]) & np.isfinite(vr[0]) & bmask
    X, Y = xr_[ok], yr_[ok]
    U, V = ur[:, ok], vr[:, ok]
    nt = len(C)
    frac, qok, urot, nring = np.full(nt, np.nan), np.zeros(nt, bool), np.full(nt, np.nan), np.zeros(nt, int)
    for k, (xc, yc, re) in enumerate(zip(C.x_c.values, C.y_c.values, C.r_eq.values)):
        if not np.isfinite(xc):
            continue
        dx, dy = X - xc, Y - yc
        r = np.hypot(dx, dy)
        s = (r >= max(0.5 * re, DX)) & (r <= max(re, 2 * DX))
        if s.sum() < 4:
            continue
        ut = args.sense * (-dy[s] * U[k, s] + dx[s] * V[k, s]) / r[s]
        q = np.clip(((np.arctan2(dy[s], dx[s]) + np.pi) // (np.pi / 2)).astype(int), 0, 3)
        frac[k], urot[k], nring[k] = (ut > 0).mean(), ut.mean(), s.sum()
        qok[k] = all(((q == m).any() and ut[q == m].mean() > 0) for m in range(4))
    return frac, qok, urot, nring


def events(C, ring_thr, n_turn):
    """present_raw -> runs (broken by centre jumps > r_eq) -> kept if >= n_turn turnovers."""
    raw = (np.isfinite(C.x_c) & (C.r_eq >= args.req_min) & (C.ring_frac >= ring_thr) & C.quad_ok).values
    jump = np.r_[False, np.hypot(np.diff(C.x_c.values), np.diff(C.y_c.values)) > C.r_eq.values[1:]]
    present = np.zeros(len(C), bool)
    eid = np.full(len(C), -1)
    rows = []
    k = 0
    while k < len(C):
        if not raw[k]:
            k += 1
            continue
        e = k + 1
        while e < len(C) and raw[e] and not jump[e]:
            e += 1
        R = C.iloc[k:e]
        Tt = np.nanmedian(R.T_turn_h)
        dur = e - k
        if dur >= n_turn * Tt:
            present[k:e] = True
            eid[k:e] = len(rows)
            s24 = np.hypot(R.x_c.diff(24), R.y_c.diff(24)).dropna()
            c = s24.median() / (24 * 3600) if len(s24) else np.nan
            rows.append(dict(start=R.time.iloc[0], end=R.time.iloc[-1], dur_h=dur, T_turn_h=Tt, n_turns=dur / Tt,
                             lon_c=R.lon_c.mean(), lat_c=R.lat_c.mean(), r_eq=R.r_eq.median(), circ=R.circ.median(),
                             U_rot_cms=100 * R.U_rot.median(), c_drift_cms=100 * c,
                             U_over_c=R.U_rot.median() / c if c > 0 else np.nan))
        k = e
    return raw, present, eid, pd.DataFrame(rows)


def summ(C, present, E):
    m = C.time.dt.month
    f = lambda sel: 100 * present[sel.values].mean()
    return dict(pct_all=f(m > 0), pct_NovDec=f(m.isin([11, 12])), pct_AugOct=f(m.isin([8, 9, 10])),
                n_events=len(E), med_dur_h=E.dur_h.median() if len(E) else np.nan,
                med_turns=E.n_turns.median() if len(E) else np.nan,
                med_U_over_c=E.U_over_c.median() if len(E) else np.nan)


# ------------------------------------------------------------- run ---
ZM, CEN = {}, {}
txt = ['PENN COVE EDDY CENTRE AND PRESENCE -- %s, %s to %s' % (args.gtx, args.ds0, args.ds1),
       'search region %s: lon < %.4f, lat < %s (%d interior psi points); sense %+d; largest-circulation patch'
       % (tag, LON1, 'none' if args.lat1 is None else '%.4f' % args.lat1, search.sum(), args.sense),
       'defaults: core OW < -%.2f sd; ring_frac >= %.2f + all 4 quadrants; r_eq >= %.0f m; events >= %g turnover(s)'
       % (args.ow_k, args.ring_thr, args.req_min, args.n_turn),
       'drift = distance of the hourly centroid from its mean over PRESENT hours; step24 = 24-h displacement', '']
sweep = []
for b in BANDS:
    wet4, search, bmask = W4[b], SR[b], MB[b]
    ok = np.isfinite(UG[b][:, 0, 0])
    zeta, ow = vort_ow(UG[b][ok], VG[b][ok])
    ur, vr = to_rho(UG[b][ok], VG[b][ok])
    ZM[b] = np.nanmean(zeta, 0)
    sd = float(np.nanstd(ow[:, search]))
    kset = sorted({args.ow_k} | ({0.0, 0.2, 0.5} if b in SWB else set()))
    for kk in kset:
        C = centres(zeta, ow, -kk * sd, tt[ok])
        C['ring_frac'], C['quad_ok'], C['U_rot'], C['n_ring'] = ring_test(C, ur, vr)
        if b in SWB:
            for rt in [0.6, 0.75, 0.9]:
                for nt_ in [1, 2]:
                    _, pr, _, E = events(C, rt, nt_)
                    sweep.append(dict(band=b, ow_k=kk, ring_thr=rt, n_turn=nt_, **summ(C, pr, E)))
        if kk == args.ow_k:
            raw, pr, eid, E = events(C, args.ring_thr, args.n_turn)
            C['present_raw'], C['present'], C['event'] = raw, pr, eid
            C.to_csv(out_dir / ('pc_eddy_center_%s.csv' % b), index=False, float_format='%.6g')
            E.to_csv(out_dir / ('pc_eddy_events_%s.csv' % b), index=False, float_format='%.5g')
            CEN[b] = C
            S = summ(C, pr, E)
            P = C[pr]
            dr = np.hypot(P.x_c - P.x_c.mean(), P.y_c - P.y_c.mean())
            s24 = pd.Series(np.where(pr, np.hypot(C.x_c.diff(24), C.y_c.diff(24)), np.nan)).dropna()
            txt.append('  %-6s sd(OW) %.2e | present %4.1f%% of hours (Nov-Dec %4.1f, Aug-Oct %4.1f); raw %4.1f%% | '
                       '%3d events, median %4.0f h = %.1f turnovers | centre (%.4f, %.4f) drift med %3.0f p90 %3.0f m, '
                       'step24 med %3.0f m | r_eq %3.0f m  U_rot %.1f cm/s  U/c med %.0f'
                       % (b, sd, S['pct_all'], S['pct_NovDec'], S['pct_AugOct'], 100 * raw.mean(), S['n_events'],
                          S['med_dur_h'], S['med_turns'], P.lon_c.mean(), P.lat_c.mean(), dr.median(),
                          dr.quantile(0.9), s24.median(), P.r_eq.median(), 100 * P.U_rot.median(),
                          S['med_U_over_c']))
            print(txt[-1], flush=True)
SW = pd.DataFrame(sweep)
SW.to_csv(out_dir / 'pc_eddy_sensitivity.csv', index=False, float_format='%.4g')
txt += ['', 'SENSITIVITY (%s): %% of hours present' % ', '.join(SWB),
        SW.to_string(index=False, float_format=lambda v: '%.2f' % v),
        '', 'dx is %.0f m. Tidal excursion in the lobe is ~100-200 m (20261008 ellipses).' % DX]
(out_dir / 'report.txt').write_text('\n'.join(txt) + '\n')
print('\n'.join(txt[-(len(SW) + 4):]))

# ------------------------------------------------------------- figures ---
cmap_m = plt.get_cmap('twilight')
fig, axs = plt.subplots(1, len(BANDS), figsize=(4.2 * len(BANDS), 4.2), layout='constrained')
for ax, b in zip(axs, BANDS):
    z = np.where(W4[b], ZM[b], np.nan)
    lim = np.nanpercentile(np.abs(z), 98)
    ax.pcolormesh(lon, lat, np.where(MB[b], 0.0, np.nan), cmap='Greys', vmin=-1, vmax=1, shading='nearest')
    ax.pcolormesh(lon_p, lat_p, z, cmap='RdBu_r', vmin=-lim, vmax=lim, shading='nearest')
    ax.axvline(LON1, color='k', ls='--', lw=0.8)
    if args.lat1 is not None:
        ax.plot([lon.min(), LON1], [LAT1, LAT1], 'k--', lw=0.8)
    C = CEN[b]
    D = C[C.present].set_index('time').resample('1D').mean(numeric_only=True).dropna(subset=['lon_c'])
    sc = ax.scatter(D.lon_c, D.lat_c, c=D.index.month, cmap=cmap_m, vmin=0.5, vmax=12.5, s=6, ec='none')
    ax.plot(C.lon_c[C.present].mean(), C.lat_c[C.present].mean(), 'k+', ms=12, mew=2)
    ax.set_aspect(1 / np.cos(np.deg2rad(LAT0)))
    ax.set_xlim(lon.min(), min(LON1 + 0.012, lon.max()))
    ax.set_title('%s: mean $\\zeta$ + daily centres\n(present %.0f%% of hours)' % (b, 100 * C.present.mean()),
                 fontsize=10)
    ax.tick_params(labelsize=7)
fig.colorbar(sc, ax=axs, label='month', ticks=range(1, 13), shrink=0.8)
fn = out_dir / 'pc_eddy_center_map.png'
fig.savefig(fn, dpi=200, transparent=True, bbox_inches='tight')
plt.close(fig)
print('saved ' + str(fn))

C = CEN[SWB[0]].set_index('time')
D = C.resample('1D').mean(numeric_only=True)
fig, axs = plt.subplots(4, 1, figsize=(13, 10), sharex=True, layout='constrained')
P = C[C.present]
axs[0].plot(C.index, C.x_c - P.x_c.mean(), lw=0.4, color='tab:blue', label='x (east)')
axs[0].plot(C.index, C.y_c - P.y_c.mean(), lw=0.4, color='tab:red', label='y (north)')
axs[0].set_ylabel('centre offset from\npresent-hour mean [m]')
axs[0].legend(loc='upper right', fontsize=8)
for ax, col, lab in [(axs[1], 'circ', 'circ [m$^2$/s]'), (axs[2], 'r_eq', 'r$_{eq}$ [m]'),
                     (axs[3], 'ring_frac', 'ring_frac')]:
    ax.plot(C.index, C[col], lw=0.4, color='k')
    ax.plot(D.index, D[col], lw=1.2, color='tab:orange')
    ax.set_ylabel(lab)
axs[2].axhline(args.req_min, color='tab:green', lw=1, ls='--')
axs[3].axhline(args.ring_thr, color='tab:green', lw=1, ls='--')
pr = C.present.values.astype(float)
for ax in axs:
    ax.fill_between(C.index, 0, 1, where=pr > 0, transform=ax.get_xaxis_transform(), color='tab:green',
                    alpha=0.15, lw=0)
    ax.grid(alpha=0.3)
fig.suptitle('Eddy centre and presence (%s, %s band, Godin-filtered), sense %+d; green = present (%.0f%% of hours)'
             % (tag, SWB[0], args.sense, 100 * C.present.mean()))
fn = out_dir / 'pc_eddy_center_series.png'
fig.savefig(fn, dpi=200, transparent=True, bbox_inches='tight')
plt.close(fig)
print('saved ' + str(fn))
