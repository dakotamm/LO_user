"""
TIME-VARYING TIDAL ELLIPSES: monthly, spring/neap, and tropic/equatorial
(diurnal inequality). Companion to 20261008_tidal_ellipses.py, which fits the
whole 2024-2025 record at once; this script asks how the ellipses CHANGE.

Same inputs as 20261008 (box extractions without -uv_to_rho, faces averaged to
rho the same way): -job pc_cove (zeta, ubar/vbar, all 30 layers) or
-job sp_head_tide (zeta, ubar/vbar, -surf and -bot layers).
REQUIRES the full-record fit from 20261008_tidal_ellipses.py for the same job
and dates, in LO_output/extract/<gtx>/tidal_ellipses/ (used for inference).

-mode monthly   utide on each calendar month. A month cannot separate P1 from
                K1 or K2 from S2 (both pairs beat at 182 d), and leaving them
                out is NOT harmless: P1 then aliases into K1 and K2 into S2
                with a semiannual beat, which would show up as a fake seasonal
                cycle in K1 and S2 (at a mid-cove cell, one month without
                inference gives K1 +25% and S2 -31%). So P1 and K2 are INFERRED
                cell by cell from the full-record fit: utide's `infer` with
                rotary amplitude ratios and phase offsets P1/K1 and K2/S2
                taken from the 20261008 output (self-check: refitting the full
                record with inference from itself reproduces it exactly).
                The other 9 constituents are fit freely; N2/M2 and Q1/O1 need
                661 h, so a 28-day February is just enough.
                The monthly time-mean velocity (Eulerian residual) is saved too.

-mode bands     short-window band fits for the fortnightly cycles. A 73 h
                window, stepped every 24 h, is fit by least squares with a mean,
                a linear trend and five tidal BANDS at nominal frequencies:
                  D1 1/24.84 h (between K1 and O1)   D2 M2   D3 M3   D4 M4   D6 M6
                A 3-day window cannot split M2/S2/N2 or K1/O1/P1 -- that is the
                point: D2 is the semidiurnal ellipse AS IT IS during that window
                (spring or neap), D1 the diurnal one (tropic or equatorial).
                Within-band frequency mismatch costs <3% of amplitude; the
                3-day window smooths the 14.8 d / 13.7 d envelopes by ~7%.
                The fit is linear in fixed frequencies, so one pseudo-inverse
                serves every window and cell (no utide, seconds per field).
                Note D1 cannot separate the diurnal sea breeze (S1) from K1/O1.

                Spring/neap and tropic/equatorial come from the BOX-MEAN zeta
                band amplitudes of the same fits: spring/neap on D2 (the
                semidiurnal band is equivalent to qprism, r = 0.99, see
                [[springneap-define-on-qprism]]); tropic/equatorial on D1. Each
                index is taken as an anomaly from its 31-day running mean
                ([[analyze-tides-on-anomalies]]) so the seasonal modulation of
                the range (semiannual, solstice/equinox) does not leak into the
                bins. Terciles: spring = top third of the D2 anomaly, neap =
                bottom third; tropic / equatorial the same on D1. Because the
                two fortnightly cycles beat at 180 d, the two binnings are
                nearly independent over two years (their correlation is printed).

OUTPUT (LO_output/extract/<gtx>/tidal_ellipses/)
  <job>_<ds0>_<ds1>_monthly.nc  dims (month, con, [s_rho,] eta_rho, xi_rho)
      same variable names as the 20261008 file (P_Lsmaj, P_Lsmin, P_theta,
      P_g, *_ci, zeta_A, zeta_g, P_umean, P_vmean), P in {bar, surf, bot, 3d}.
      P1 and K2 are inferred (attrs['inferred']).
  <job>_<ds0>_<ds1>_bands.nc
      time series (time = window centre): zidx_D2, zidx_D1 (box-mean zeta band
      amplitude), their anomalies, and the bin flags.
      daily fields for zeta and bar: zeta_A_d (time, band, eta, xi) and
      bar_Lsmaj_d / bar_Lsmin_d / bar_theta_d.
      composites, dims (bin, band, [s_rho,] eta, xi), bin in
      [all, spring, neap, tropic, equatorial]:
        P_Lsmaj, P_Lsmin   from the bin-mean rotary amplitudes
                           (Lsmaj = <A+> + <A->, Lsmin = <A+> - <A->)
        P_theta            axial mean [deg CCW from east, 0-180]
        P_lag              circular-mean phase of max current relative to LOCAL
                           high water of the same band [deg]: the current
                           peaks lag deg after HW, in the direction of the
                           major axis oriented EASTWARD (theta in -90..90, for
                           this phase only). In Penn Cove (E-W, mouth to the
                           east) eastward is ebb, so a standing wave gives
                           lag ~ +90 (max ebb a quarter cycle after HW).
        P_umean, P_vmean   bin-mean of the window-mean velocity
        zeta_A             bin-mean zeta band amplitude [m]

RUN (mac, pc_cove ~20 min monthly, ~1 min bands):
  python 20261027_tidal_ellipses_windowed.py -job pc_cove -mode bands -Nproc 8
  python 20261027_tidal_ellipses_windowed.py -job pc_cove -mode monthly -Nproc 8 > tidal_ellipses_windowed_pc_cove.log &
apogee (sp_head_tide boxes live there; copy the 20261008 sp_head_tide fit up first):
  python 20261027_tidal_ellipses_windowed.py -job sp_head_tide -mode both -Nproc 20 > tidal_ellipses_windowed_sp_head.log &
"""

import os
# one BLAS thread per worker: with the Pool on top, threaded BLAS makes utide ~3x slower
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(_v, '1')
import sys
import argparse
import multiprocessing as mp
import warnings
from time import time

import numpy as np
import pandas as pd
import xarray as xr
import utide
from utide.utilities import Bunch

from lo_tools import Lfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
parser.add_argument('-job', default='pc_cove', type=str)
parser.add_argument('-0', '--ds0', default='2024.01.01', type=str)
parser.add_argument('-1', '--ds1', default='2025.12.31', type=str)
parser.add_argument('-mode', default='both', type=str) # monthly, bands, both
parser.add_argument('-Nproc', default=8, type=int)
parser.add_argument('-whours', default=73, type=int) # band-fit window [h], odd
parser.add_argument('-do_3d', default=True, type=Lfun.boolean_string)
parser.add_argument('-test', default=False, type=Lfun.boolean_string) # 3 months, every 4th cell, 3 layers
args = parser.parse_args()
warnings.filterwarnings('ignore', category=RuntimeWarning) # all-NaN slices on land

CONS = ['M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1', 'M4', 'MS4', 'M6'] # same as 20261008
INF_I, INF_R = ['P1', 'K2'], ['K1', 'S2'] # inferred, reference
KW = dict(method='ols', conf_int='linear', constit=CONS, nodal=True, trend=False, verbose=False)
UV_OUT = ['Lsmaj', 'Lsmin', 'theta', 'g', 'Lsmaj_ci', 'Lsmin_ci', 'theta_ci', 'g_ci']
Z_OUT = ['A', 'g', 'A_ci', 'g_ci']
MIN_GOOD = 0.9
MIN_MONTH_H = 27 * 24

BANDS = ['D1', 'D2', 'D3', 'D4', 'D6']
BAND_PERIOD = {'D1': 24.84, 'D2': 12.4206, 'D3': 8.2804, 'D4': 6.2103, 'D6': 4.1402} # hours
BAND_KEEP = ['D1', 'D2', 'D4'] # saved; D3 and D6 only absorb variance
BINS = ['all', 'spring', 'neap', 'tropic', 'equatorial']

gridname, tag, ex_name = args.gtagex.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
box_dir = Ldir['LOo'] / 'extract' / args.gtagex / 'box'
ell_dir = Ldir['LOo'] / 'extract' / args.gtagex / 'tidal_ellipses'
Lfun.make_dir(ell_dir)
dstr = args.ds0 + '_' + args.ds1
tstr = '_test' if args.test else ''

found = {k: box_dir / (args.job + ('_' + k if k else '') + '_' + dstr + '.nc') for k in ['', 'surf', 'bot']}
found = {k: fn for k, fn in found.items() if fn.is_file()}
if len(found) == 0:
    print('*** No box file for %s %s in %s' % (args.job, dstr, box_dir))
    sys.exit()
full_fn = ell_dir / (args.job + '_' + dstr + '.nc')
if not full_fn.is_file():
    print('*** Need the full-record fit %s (run 20261008_tidal_ellipses.py, or copy it from the mac)' % full_fn)
    sys.exit()
box_fn = list(found.values())[0]
print('Reading ' + str(box_fn))
ds = xr.open_dataset(box_fn)
FULL = xr.open_dataset(full_fn)
T = ds.ocean_time.values
dt_h = np.diff(T).astype('timedelta64[s]').astype(float) / 3600
assert np.allclose(dt_h, 1.0), 'band fits assume an unbroken hourly record'
mask = ds.mask_rho.values == 1
assert np.array_equal(FULL.mask_rho.values == 1, mask)
lat = ds.lat_rho.values
NR, NC = mask.shape
wet = np.flatnonzero(mask)
if args.test:
    wet = wet[::4]
LAT = lat.ravel()[wet]
print('%d hours, %d x %d rho, %d wet cells' % (len(T), NR, NC, len(wet)))

def to_rho(u, v):
    """(NT, NR, NC-1) u and (NT, NR-1, NC) v on faces -> (NT, NR, NC) at rho."""
    u = np.nan_to_num(u, nan=0.0)
    v = np.nan_to_num(v, nan=0.0)
    ur = np.full((u.shape[0], NR, NC), np.nan, dtype=np.float32)
    vr = np.full((v.shape[0], NR, NC), np.nan, dtype=np.float32)
    ur[:, :, 1:-1] = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
    vr[:, 1:-1, :] = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    return ur, vr

# fields: (name, dataset, u var, v var, layer index or None, full-record prefix)
FIELDS = [('bar', ds, 'ubar', 'vbar', None)]
for k in ['surf', 'bot']:
    if k in found:
        dk = xr.open_dataset(found[k])
        assert np.array_equal(dk.ocean_time.values, T) and np.array_equal(dk.mask_rho.values, ds.mask_rho.values)
        FIELDS.append((k, dk, 'u', 'v', None))
HAS3D = args.do_3d and ('u' in ds) and (ds.u.ndim == 4)
LAYERS = []
if HAS3D:
    NZ = ds.sizes['s_rho']
    LAYERS = [0, NZ // 2, NZ - 1] if args.test else list(range(NZ))

def load_uv(P, dp, vu, vv, iz):
    if iz is None:
        return to_rho(dp[vu].values, dp[vv].values)
    return to_rho(dp[vu][:, iz].values, dp[vv][:, iz].values)

def full_slice(P, vn, iz):
    """Full-record fit (con, eta, xi) for field P (or layer iz of 3d)."""
    if iz is None:
        return FULL[P + '_' + vn].values
    return FULL['3d_' + vn].values[:, iz]

def cols(a):
    return a.reshape(a.shape[0], -1)[:, wet]

# ================================================================ monthly ===
per = pd.DatetimeIndex(T).to_period('M')
MONTHS = [m for m in per.unique() if (per == m).sum() >= MIN_MONTH_H]
if args.test:
    MONTHS = MONTHS[:3]
MIDX = [np.flatnonzero(per == m) for m in MONTHS]
ICON = {n: CONS.index(n) for n in CONS}

def rot(Lsmaj, Lsmin, theta, g):
    """utide ellipse -> rotary amplitudes and phases [deg]."""
    return (Lsmaj + Lsmin) / 2, (Lsmaj - Lsmin) / 2, theta - g, theta + g

# globals read by the workers (inherited through fork)
A = None
B = None
INF = None # (ncell, 8) uv: [rP1, rK2, rmP1, rmK2, oP1, oK2, omP1, omK2]; z: (ncell, 4)

def infer_uv(Fs):
    """Fs: dict vn -> (con, ncell) full-record fit -> inference table (ncell, 8)."""
    out = []
    for I, R in zip(INF_I, INF_R):
        Api, Ami, epi, emi = rot(*[Fs[vn][ICON[I]] for vn in ['Lsmaj', 'Lsmin', 'theta', 'g']])
        Apr, Amr, epr, emr = rot(*[Fs[vn][ICON[R]] for vn in ['Lsmaj', 'Lsmin', 'theta', 'g']])
        out.append((Api / Apr, Ami / Amr, epi - epr, -(emi - emr)))
    (rp1, rm1, op1, om1), (rp2, rm2, op2, om2) = out
    return np.column_stack([rp1, rp2, rm1, rm2, op1, op2, om1, om2])

def infer_z(Fs):
    rows = []
    for I, R in zip(INF_I, INF_R):
        rows.append((Fs['A'][ICON[I]] / Fs['A'][ICON[R]], Fs['g'][ICON[R]] - Fs['g'][ICON[I]]))
    (r1, o1), (r2, o2) = rows
    return np.column_stack([r1, r2, o1, o2])

def month_uv(k):
    res = np.full((len(MONTHS), len(UV_OUT), len(CONS)), np.nan)
    if not np.all(np.isfinite(INF[k])):
        return res
    inf = Bunch(inferred_names=INF_I, reference_names=INF_R, amp_ratios=list(INF[k, :4]), phase_offsets=list(INF[k, 4:]))
    for m, ii in enumerate(MIDX):
        a, b = A[ii, k], B[ii, k]
        if (np.isfinite(a) & np.isfinite(b)).mean() < MIN_GOOD:
            continue
        c = utide.solve(T[ii], a, b, lat=LAT[k], infer=inf, **KW)
        jj = [list(c.name).index(n) for n in CONS]
        res[m] = [np.asarray(c[vn])[jj] for vn in UV_OUT]
    return res

def month_z(k):
    res = np.full((len(MONTHS), len(Z_OUT), len(CONS)), np.nan)
    if not np.all(np.isfinite(INF[k])):
        return res
    inf = Bunch(inferred_names=INF_I, reference_names=INF_R, amp_ratios=list(INF[k, :2]), phase_offsets=list(INF[k, 2:]))
    for m, ii in enumerate(MIDX):
        a = A[ii, k]
        if np.isfinite(a).mean() < MIN_GOOD:
            continue
        c = utide.solve(T[ii], a, lat=LAT[k], infer=inf, **KW)
        jj = [list(c.name).index(n) for n in CONS]
        res[m] = [np.asarray(c[vn])[jj] for vn in Z_OUT]
    return res

def run_month(fun, a, b, inf):
    """-> (nmonth, nout, ncon, NR, NC)"""
    global A, B, INF
    A, B, INF = cols(a), (None if b is None else cols(b)), inf
    tt0 = time()
    with mp.get_context('fork').Pool(args.Nproc) as pool:
        res = pool.map(fun, range(len(wet)), chunksize=max(1, len(wet) // (20 * args.Nproc)))
    res = np.stack(res, axis=-1)
    full = np.full(res.shape[:3] + (NR * NC,), np.nan)
    full[..., wet] = res
    print('  %d cells x %d months in %0.1f s' % (len(wet), len(MONTHS), time() - tt0))
    sys.stdout.flush()
    return full.reshape(res.shape[:3] + (NR, NC))

def month_means(ur, vr):
    um = np.stack([np.nanmean(ur[ii], axis=0) for ii in MIDX])
    vm = np.stack([np.nanmean(vr[ii], axis=0) for ii in MIDX])
    return np.where(mask, um, np.nan), np.where(mask, vm, np.nan)

def do_monthly():
    print('\n=== MONTHLY: %d months, %s to %s' % (len(MONTHS), MONTHS[0], MONTHS[-1]))
    out = xr.Dataset(coords={'month': pd.PeriodIndex(MONTHS).to_timestamp().values, 'con': CONS,
        'lon_rho': (('eta_rho', 'xi_rho'), ds.lon_rho.values), 'lat_rho': (('eta_rho', 'xi_rho'), lat)})
    out['h'] = (('eta_rho', 'xi_rho'), ds.h.values)
    out['mask_rho'] = (('eta_rho', 'xi_rho'), ds.mask_rho.values)
    out['n_hours'] = (('month',), np.array([len(ii) for ii in MIDX]))
    dm = ('month', 'con', 'eta_rho', 'xi_rho')
    print('zeta')
    Fz = {vn: FULL['zeta_' + vn].values.reshape(len(CONS), -1)[:, wet] for vn in ['A', 'g']}
    res = run_month(month_z, ds.zeta.values, None, infer_z(Fz))
    for i, vn in enumerate(Z_OUT):
        out['zeta_' + vn] = (dm, res[:, i])
    for P, dp, vu, vv, _ in FIELDS:
        print(P)
        ur, vr = load_uv(P, dp, vu, vv, None)
        um, vm = month_means(ur, vr)
        out[P + '_umean'] = (('month', 'eta_rho', 'xi_rho'), um)
        out[P + '_vmean'] = (('month', 'eta_rho', 'xi_rho'), vm)
        Fs = {vn: full_slice(P, vn, None).reshape(len(CONS), -1)[:, wet] for vn in ['Lsmaj', 'Lsmin', 'theta', 'g']}
        res = run_month(month_uv, ur, vr, infer_uv(Fs))
        for i, vn in enumerate(UV_OUT):
            out[P + '_' + vn] = (dm, res[:, i])
        del ur, vr
    if HAS3D:
        r3 = np.full((len(MONTHS), len(UV_OUT), len(CONS), NZ, NR, NC), np.nan)
        m3 = np.full((2, len(MONTHS), NZ, NR, NC), np.nan)
        for iz in LAYERS:
            print('3d layer %d of %d' % (iz + 1, NZ))
            ur, vr = load_uv('3d', ds, 'u', 'v', iz)
            m3[0, :, iz], m3[1, :, iz] = month_means(ur, vr)
            Fs = {vn: full_slice('3d', vn, iz).reshape(len(CONS), -1)[:, wet] for vn in ['Lsmaj', 'Lsmin', 'theta', 'g']}
            r3[:, :, :, iz] = run_month(month_uv, ur, vr, infer_uv(Fs))
            del ur, vr
        d3 = ('month', 'con', 's_rho', 'eta_rho', 'xi_rho')
        for i, vn in enumerate(UV_OUT):
            out['3d_' + vn] = (d3, r3[:, i])
        out['3d_umean'] = (('month', 's_rho', 'eta_rho', 'xi_rho'), m3[0])
        out['3d_vmean'] = (('month', 's_rho', 'eta_rho', 'xi_rho'), m3[1])
        out['z0_rho'] = FULL['z0_rho']
        out = out.assign_coords(s_rho=ds.s_rho.values)
    set_units(out)
    out.attrs = {'source': ', '.join(str(fn) for fn in found.values()), 'full_record_fit': str(full_fn),
        'method': 'utide.solve per calendar month ' + str({k: v for k, v in KW.items() if k != 'constit'}),
        'inferred': 'P1 from K1, K2 from S2: rotary amplitude ratios + phase offsets from the full-record fit, per cell',
        'sign': 'Lsmin > 0 counterclockwise; theta deg CCW from east; g Greenwich phase deg'}
    fn = ell_dir / (args.job + '_' + dstr + '_monthly' + tstr + '.nc')
    out.to_netcdf(fn, encoding={vn: {'dtype': 'float32', 'zlib': True} for vn in out.data_vars if vn != 'n_hours'})
    print('Saved ' + str(fn))

def set_units(out):
    for vn in out.data_vars:
        if ('Lsm' in vn) or vn.endswith('mean'):
            out[vn].attrs['units'] = 'm s-1'
        elif vn.endswith(('theta', 'g', 'theta_ci', 'g_ci', 'lag')):
            out[vn].attrs['units'] = 'deg'
        elif vn.startswith('zeta_A') or vn.startswith('zidx'):
            out[vn].attrs['units'] = 'm'

# ================================================================== bands ===
HW = args.whours // 2
tau = np.arange(-HW, HW + 1, dtype=float) # hours from window centre
OM = np.array([2 * np.pi / BAND_PERIOD[b] for b in BANDS]) # rad/h
NB = len(BANDS)
# complex model for w = u + iv: [1, tau, e^{+i w tau} (NB), e^{-i w tau} (NB)]
Xc = np.column_stack([np.ones_like(tau), tau] + [np.exp(1j * o * tau) for o in OM] + [np.exp(-1j * o * tau) for o in OM])
Pc = np.linalg.pinv(Xc)
# real model for zeta: [1, tau, cos (NB), sin (NB)]
Xr = np.column_stack([np.ones_like(tau), tau] + [np.cos(o * tau) for o in OM] + [np.sin(o * tau) for o in OM])
Pr = np.linalg.pinv(Xr)
CEN = np.arange(HW, len(T) - HW, 24) # window centres, every 24 h
if args.test:
    CEN = CEN[:92]
TW = T[CEN]
KB = [BANDS.index(b) for b in BAND_KEEP]

def band_z(z):
    """z (NT, ncell) -> A, phi (nwin, NB, ncell): z = A cos(w tau - phi)."""
    Aa = np.full((len(CEN), NB, z.shape[1]), np.nan)
    Ph = np.full_like(Aa, np.nan)
    for n, c in enumerate(CEN):
        C = Pr @ z[c - HW:c + HW + 1]
        a, b = C[2:2 + NB], C[2 + NB:]
        Aa[n], Ph[n] = np.hypot(a, b), np.rad2deg(np.arctan2(b, a))
    return Aa, Ph

def band_uv(u, v):
    """-> Ap, Am, theta (0-180), g_east (phase of max along the eastward-
    oriented major axis), mean (complex) for each window."""
    w = u.astype(np.complex64) + 1j * v.astype(np.complex64)
    nw, nc = len(CEN), w.shape[1]
    Ap = np.full((nw, NB, nc), np.nan); Am = np.full_like(Ap, np.nan)
    th = np.full_like(Ap, np.nan); ge = np.full_like(Ap, np.nan)
    mu = np.full((nw, nc), np.nan, dtype=complex)
    for n, c in enumerate(CEN):
        C = Pc @ w[c - HW:c + HW + 1]
        Wp, Wm = C[2:2 + NB], C[2 + NB:]
        ep, em = np.angle(Wp, deg=True), np.angle(Wm, deg=True)
        t = 0.5 * (ep + em)
        g = 0.5 * (em - ep)
        t0 = (t + 90) % 180 - 90 # major axis pointing east
        g = (g + 180 * np.round((t0 - t) / 180)) % 360
        Ap[n], Am[n], th[n], ge[n], mu[n] = np.abs(Wp), np.abs(Wm), t0 % 180, g, C[0]
    return Ap, Am, th, ge, mu

def circ_mean(a, axis=0):
    r = np.deg2rad(a)
    return np.rad2deg(np.arctan2(np.nanmean(np.sin(r), axis=axis), np.nanmean(np.cos(r), axis=axis))) % 360

def axial_mean(a, axis=0):
    return (circ_mean(2 * a, axis=axis) / 2) % 180

def to_grid(x):
    """(..., ncell) -> (..., NR, NC)"""
    full = np.full(x.shape[:-1] + (NR * NC,), np.nan, dtype=np.float32)
    full[..., wet] = x
    return full.reshape(x.shape[:-1] + (NR, NC))

def do_bands():
    print('\n=== BANDS: %d windows of %d h, every 24 h' % (len(CEN), 2 * HW + 1))
    tt0 = time()
    zA, zP = band_z(cols(ds.zeta.values).astype(float))
    # box-mean band amplitude -> fortnightly indices, as anomalies from a 31-day running mean
    I = pd.DataFrame({'D2': np.nanmean(zA[:, BANDS.index('D2')], axis=1), 'D1': np.nanmean(zA[:, BANDS.index('D1')], axis=1)}, index=pd.DatetimeIndex(TW))
    Ia = I - I.rolling(31, center=True, min_periods=15).mean()
    q2, q1 = Ia.D2.quantile([1 / 3, 2 / 3]).values, Ia.D1.quantile([1 / 3, 2 / 3]).values
    FLAG = {'all': np.ones(len(CEN), bool), 'spring': (Ia.D2 >= q2[1]).values, 'neap': (Ia.D2 <= q2[0]).values,
        'tropic': (Ia.D1 >= q1[1]).values, 'equatorial': (Ia.D1 <= q1[0]).values}
    print('box-mean D2 zeta amp %.3f m (spring %.3f, neap %.3f); D1 %.3f m (tropic %.3f, equatorial %.3f)' % (
        I.D2.mean(), I.D2[FLAG['spring']].mean(), I.D2[FLAG['neap']].mean(), I.D1.mean(), I.D1[FLAG['tropic']].mean(), I.D1[FLAG['equatorial']].mean()))
    print('corr(D2 anomaly, D1 anomaly) = %+.2f; spring-and-tropic windows %d of %d spring' % (
        Ia.D2.corr(Ia.D1), (FLAG['spring'] & FLAG['tropic']).sum(), FLAG['spring'].sum()))
    out = xr.Dataset(coords={'time': TW, 'band': BAND_KEEP, 'bin': BINS,
        'lon_rho': (('eta_rho', 'xi_rho'), ds.lon_rho.values), 'lat_rho': (('eta_rho', 'xi_rho'), lat)})
    out['h'] = (('eta_rho', 'xi_rho'), ds.h.values)
    out['mask_rho'] = (('eta_rho', 'xi_rho'), ds.mask_rho.values)
    for b in ['D2', 'D1']:
        out['zidx_' + b] = (('time',), I[b].values)
        out['zidx_' + b + '_anom'] = (('time',), Ia[b].values)
    for k in BINS[1:]:
        out['is_' + k] = (('time',), FLAG[k].astype(np.int8))
    out['zeta_A_d'] = (('time', 'band', 'eta_rho', 'xi_rho'), to_grid(zA[:, KB]))
    out['zeta_A'] = (('bin', 'band', 'eta_rho', 'xi_rho'), to_grid(np.stack([np.nanmean(zA[FLAG[k]][:, KB], axis=0) for k in BINS])))

    def composites(Ap, Am, th, ge, mu):
        lag = (ge - zP) % 360 # local HW of the same band
        R = {vn: [] for vn in ['Lsmaj', 'Lsmin', 'theta', 'lag', 'umean', 'vmean']}
        for k in BINS:
            f = FLAG[k]
            ap, am = np.nanmean(Ap[f][:, KB], axis=0), np.nanmean(Am[f][:, KB], axis=0)
            R['Lsmaj'].append(ap + am)
            R['Lsmin'].append(ap - am)
            R['theta'].append(axial_mean(th[f][:, KB]))
            R['lag'].append(circ_mean(lag[f][:, KB]))
            m = np.nanmean(mu[f], axis=0)
            R['umean'].append(m.real)
            R['vmean'].append(m.imag)
        return {vn: to_grid(np.stack(R[vn])) for vn in R}

    db = ('bin', 'band', 'eta_rho', 'xi_rho')
    for P, dp, vu, vv, _ in FIELDS:
        ur, vr = load_uv(P, dp, vu, vv, None)
        Ap, Am, th, ge, mu = band_uv(cols(ur), cols(vr))
        del ur, vr
        for vn, x in composites(Ap, Am, th, ge, mu).items():
            out[P + '_' + vn] = (('bin', 'eta_rho', 'xi_rho') if vn.endswith('mean') else db, x)
        if P == 'bar':
            dd = ('time', 'band', 'eta_rho', 'xi_rho')
            out['bar_Lsmaj_d'] = (dd, to_grid((Ap + Am)[:, KB]))
            out['bar_Lsmin_d'] = (dd, to_grid((Ap - Am)[:, KB]))
            out['bar_theta_d'] = (dd, to_grid(th[:, KB]))
        print('  %s done (%0.1f s)' % (P, time() - tt0))
        sys.stdout.flush()
    if HAS3D:
        C3 = {vn: np.full((len(BINS), len(KB), NZ, NR, NC), np.nan, dtype=np.float32) for vn in ['Lsmaj', 'Lsmin', 'theta', 'lag']}
        M3 = {vn: np.full((len(BINS), NZ, NR, NC), np.nan, dtype=np.float32) for vn in ['umean', 'vmean']}
        for iz in LAYERS:
            ur, vr = load_uv('3d', ds, 'u', 'v', iz)
            R = composites(*band_uv(cols(ur), cols(vr)))
            del ur, vr
            for vn in C3:
                C3[vn][:, :, iz] = R[vn]
            for vn in M3:
                M3[vn][:, iz] = R[vn]
        print('  3d done (%0.1f s)' % (time() - tt0))
        for vn in C3:
            out['3d_' + vn] = (('bin', 'band', 's_rho', 'eta_rho', 'xi_rho'), C3[vn])
        for vn in M3:
            out['3d_' + vn] = (('bin', 's_rho', 'eta_rho', 'xi_rho'), M3[vn])
        out['z0_rho'] = FULL['z0_rho']
        out = out.assign_coords(s_rho=ds.s_rho.values)
    set_units(out)
    out.attrs = {'source': ', '.join(str(fn) for fn in found.values()),
        'method': '%d h windows every 24 h; LS fit of mean + trend + bands %s at periods %s h' % (2 * HW + 1, BANDS, [BAND_PERIOD[b] for b in BANDS]),
        'bins': 'terciles of box-mean zeta band amplitude anomaly (31-d running mean removed): spring/neap on D2, tropic/equatorial on D1',
        'sign': 'Lsmin > 0 counterclockwise; theta deg CCW from east (0-180); lag = phase of max current along the EASTWARD-oriented major axis minus local HW phase, same band [deg]'}
    fn = ell_dir / (args.job + '_' + dstr + '_bands' + tstr + '.nc')
    out.to_netcdf(fn, encoding={vn: {'dtype': 'float32', 'zlib': True} for vn in out.data_vars if not vn.startswith('is_')})
    print('Saved ' + str(fn))

if args.mode in ['bands', 'both']:
    do_bands()
if args.mode in ['monthly', 'both']:
    do_monthly()
