"""
Cross-channel salinity and velocity gradients in Penn Cove, and what the wind
does to them. The sibling of 20260807_pc_alongchannel_wind.py, built to be
read beside it -- same run, same sections, same event machinery, rotated 90
degrees.

y is the cross-channel coordinate, POSITIVE TOWARD THE NORTH SHORE, taken as
the unit vector (ay, -ax) where (ax, ay) is the along-channel vector the
along-channel script defines. Each section is a constant-longitude N-S line,
so y is very nearly latitude, and the script asserts corr(y, lat) = +1 rather
than trusting that.

  ***  SIGN WARNING  ***
  Both LO_user/extract/tef2/reduce_wind_cove.py (its `cove_axis` /`w_cross`
  attribute) and 20260806_wind_characterize.py say their cross-cove component
  is "positive toward the north shore". IT IS NOT -- it is positive toward the
  SOUTH shore. Both build cross as (-u*ay + v*ax) with (ax, ay) pointing
  mouth -> head, i.e. west; that is the LEFT-hand normal of a westward vector,
  which points south. Check it on a due-north wind (u=0, v=1): cross = ax,
  and ax is about -0.98. This script uses the right-hand normal so that
  positive really is northward, and it prints the check. Any earlier reading
  of cross_pc in daily_wind.csv has the sign backwards.

WHAT IS COMPUTED, per section, per hour

  ds/dy, du/dy   width-weighted least-squares slope of the per-face layer mean
                 against y, for the top 3 m, the bottom 3 m and the full
                 depth. Reported with the fraction of cross-channel variance
                 the straight line actually explains, because a slope is only
                 a summary of the lateral structure when that structure is
                 roughly linear -- and at the head it is not (see below).

  the h problem   a section is a channel cross-section: pc_lp runs 15.9 m at
                 the north end to 26.8 m in the middle. Depth is therefore
                 correlated with lateral position (corr(y, h) = -0.55 at
                 pc_lp), and a raw bottom-layer ds/dy is partly a statement
                 about which faces are deep, not about north versus south.
                 So every gradient is ALSO computed as the partial slope on y
                 from a regression on [1, y, h], which is the lateral gradient
                 at fixed depth, and both are carried. The surface layer is
                 nearly free of this and the two versions agree there; the
                 bottom layer is where they diverge.

  N - S          the difference between a DEPTH-MATCHED pair of faces, one
                 from each side of the mean transport sign change, chosen by
                 the rule already established in 20260806_pc_sections_series.py
                 (maximise |qbar_north| + |qbar_south| subject to a depth
                 tolerance and a minimum separation). This is the
                 confound-free version of the same question, and it is the
                 number to quote when the linear fit is poor.

WHY THE LINEAR FIT FAILS AT THE HEAD. Mean per-face transport across pc_lp and
pc_lj changes sign exactly once -- in on the north side, out on the south, the
lateral exchange these sections are known for. Across pc_cp it changes sign
THREE times (+26, -26, -73, -58, -22, +25, +74, +54 m3/s from north to south):
out at both shores, in through the middle-north. A straight line through that
is close to meaningless, and the R2 reported per section is what says so. Use
the N-S pair and the printed per-face profile at pc_cp, not the slope.

THE WIND CONNECTION THIS EXISTS TO TEST. The along-channel analysis found the
two event families are not mirror images: the "up-cove" family (the prevailing
south-easterlies, n=22) is mostly CROSS-cove -- 4.58 m/s cross against 1.98
along -- so it is a weak along-axis forcing that was being judged on the wrong
axis. Here it is judged on its own axis. Events are picked on the cross-cove
stress, and the lagged correlations are run against BOTH stress components, so
"which wind drives the lateral structure" is answered rather than assumed.
Ekman is a live possibility in the other direction too: a down-cove westerly
drives surface transport to its right, i.e. northward, so an along-cove wind
has every reason to show up in ds/dy.

Everything else follows the along-channel script: Godin lowpass, 30-day
rolling anomalies, events as peaks in the stress anomaly with a full window
inside the finite record, composites referenced to a lead-in day with +/- 2 SE
across events, and daily means with an autocorrelation-corrected n_eff for the
continuous test.

Runs on the mac from the local extractions_avg plus the reduced wind pickle.

run 20260807_pc_crosschannel_wind.py
run 20260807_pc_crosschannel_wind.py --hlay 4
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
from scipy.signal import find_peaks
from scipy import stats

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('--gtx', default='wb1_t0_xn11abbur00')
p.add_argument('--coll', default='wb1_pc1')
p.add_argument('--ds0', default='2024.01.01')
p.add_argument('--ds1', default='2025.12.31')
p.add_argument('--sects', default='pc_lp,pc_lj,pc_cp', help='mouth first')
p.add_argument('--hlay', type=float, default=3.0,
               help='thickness (m) of the surface and bottom layers')
p.add_argument('--anom_days', type=float, default=30.0)
p.add_argument('--minsep', type=float, default=4.0,
               help='minimum days between wind events')
p.add_argument('--pct', type=float, default=90.0,
               help='percentile of |cross-cove stress| that defines an event')
p.add_argument('--lead', type=float, default=3.0)
p.add_argument('--lag', type=float, default=5.0)
p.add_argument('--dhmax', type=float, default=0.5,
               help='initial depth tolerance (m) for the N/S pair')
p.add_argument('--minsep_face', type=float, default=0.4,
               help='minimum N-S separation of the pair, as a fraction of '
                    'section width')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
gctag = 'wb1_' + args.coll.split('_')[-1]
tef2 = Ldir['LOo'] / 'extract' / args.gtx / 'tef2'
in_dir = tef2 / ('extractions_avg_%s_%s' % (args.ds0, args.ds1))
wind_fn = (Ldir['LOo'] / 'DM_outs' / '20260806_wind'
           / ('wind_hourly_atm00_%s_%s.p' % (args.ds0, args.ds1)))
out_dir = Ldir['LOo'] / 'DM_outs' / '20260807_pc_crosschannel_wind'
Lfun.make_dir(out_dir)

SECTS = [s.strip() for s in args.sects.split(',')]
SLAB = {'pc_lp': 'mouth', 'pc_lj': 'mid-cove', 'pc_cp': 'head'}
CB = dict(blue='#0072B2', orange='#D55E00', green='#009E73', red='#CC0000',
          purple='#7B3294', yellow='#E69F00', pink='#CC79A7', grey='#7f7f7f')
SC = {'pc_lp': CB['blue'], 'pc_lj': CB['green'], 'pc_cp': CB['orange']}
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
O2_MMOL_TO_MGL = 32.0 / 1000.0


def godin(a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return zfun.lowpass(np.asarray(a, dtype=float), f='godin')


def anom(s):
    w = int(round(args.anom_days * 24))
    return s - s.rolling(w, center=True, min_periods=int(w * 0.6)).mean()


def neff_r(x, y):
    """corr with an autocorrelation-corrected n, and its two-sided p."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    n = len(x)
    if n < 20:
        return np.nan, np.nan, 0
    r = np.corrcoef(x, y)[0, 1]
    r1 = [np.corrcoef(v[:-1], v[1:])[0, 1] for v in (x, y)]
    f = (1 - r1[0] * r1[1]) / (1 + r1[0] * r1[1])
    ne = max(3.0, n * min(1.0, max(f, 1e-6)))
    t = r * np.sqrt((ne - 2) / max(1e-12, 1 - r ** 2))
    return r, 2 * stats.t.sf(abs(t), ne - 2), int(ne)


def wls(X, w):
    """Weighted least-squares projector: beta = data @ P.T, plus the hat
    matrix so a weighted R2 can be formed."""
    Wm = np.diag(w)
    P = np.linalg.solve(X.T @ Wm @ X, X.T @ Wm)
    return P, X @ P


# ------------------------------------------------------------- geometry ---
dstr = xr.open_dataset(tef2 / ('structure_%s_%s_%s.nc' % (args.ds0, args.ds1,
                                                          gctag)))
CEN = {sn: (float(np.mean(dstr['%s_lon' % sn].values)),
            float(np.mean(dstr['%s_lat' % sn].values))) for sn in SECTS}
lat0 = np.mean([c[1] for c in CEN.values()])
COS = np.cos(np.deg2rad(lat0))


def xy_km(lo, la):
    return lo * COS * 111.32, la * 111.32


x0, y0 = xy_km(*CEN[SECTS[0]])
xN, yN = xy_km(*CEN[SECTS[-1]])
axl = np.hypot(xN - x0, yN - y0)
ax_, ay_ = (xN - x0) / axl, (yN - y0) / axl        # mouth -> head
cx_, cy_ = ay_, -ax_                               # right-hand normal
print('--- axes ---')
print('along  (%.4f, %.4f)  mouth -> head' % (ax_, ay_))
print('cross  (%.4f, %.4f)  positive toward the NORTH shore' % (cx_, cy_))
# the check the two older wind scripts fail
print('a due-north wind (u=0, v=1) projects to cross = %+.3f here, and to '
      '%+.3f\nwith the (-u*ay + v*ax) form used by reduce_wind_cove.py and '
      '20260806_wind_characterize.py --\nso those two call southward '
      '"toward the north shore". Sign corrected here.' % (cy_, ax_))

# ------------------------------------------------------ flood sign check ---
dflux = xr.open_dataset(tef2 / ('hourly_flux_%s_%s_%s.nc'
                                % (args.ds0, args.ds1, gctag)))
SGN = {}
for sn in SECTS:
    q = dflux.qnet.sel(sect=sn).values
    z = dflux.ssh.sel(sect=sn).values
    SGN[sn] = -1.0 if np.corrcoef(q, np.gradient(z))[0, 1] < 0 else 1.0
dflux.close()

# ------------------------------------------- per-face y, h and the N/S pair ---
print('\n--- cross-section geometry, and the depth-matched N/S pair ---')
GEO, PAIR = {}, {}
for sn in SECTS:
    lon = dstr['%s_lon' % sn].values
    lat = dstr['%s_lat' % sn].values
    h = dstr['%s_h' % sn].values
    dd = dstr['%s_dd' % sn].values
    qbar = dstr['%s_qbar' % sn].values.sum(axis=0)
    xs, ys = xy_km(lon, lat)
    y = (xs - xs.mean()) * cx_ + (ys - ys.mean()) * cy_      # km, + = north
    rl = np.corrcoef(y, lat)[0, 1]
    assert rl > 0.999, '%s: y is not northward (corr with lat %.3f)' % (sn, rl)
    GEO[sn] = dict(y=y, h=h, dd=dd, qbar=qbar, lat=lat)

    # one face each side of the transport sign change, depth-matched and far
    # enough apart to be a lateral contrast -- the rule from
    # 20260806_pc_sections_series.py, kept identical so the two agree
    iN, iS = np.where(qbar < 0)[0], np.where(qbar > 0)[0]
    span = y.max() - y.min()
    need = args.minsep_face * span
    dh, kN, kS = args.dhmax, None, None
    while kN is None and dh <= span * 1e9:
        best = -np.inf
        for a in iN:
            for b in iS:
                if y[a] - y[b] < need or abs(h[a] - h[b]) > dh:
                    continue
                if abs(qbar[a]) + abs(qbar[b]) > best:
                    best, kN, kS = abs(qbar[a]) + abs(qbar[b]), a, b
        if kN is None:
            dh += 0.25
    PAIR[sn] = (kN, kS, dh)
    nsign = int(np.sum(np.diff(np.sign(qbar)) != 0))
    print('  %-6s width %.2f km, corr(y,h) = %+.2f, qbar changes sign %d '
          'time(s)' % (sn, dd.sum() / 1e3, np.corrcoef(y, h)[0, 1], nsign))
    print('         N/S pair p=%d (lat %.4f, h %.1f) vs p=%d (lat %.4f, '
          'h %.1f), tol %.2f m' % (kN, lat[kN], h[kN], kS, lat[kS], h[kS], dh))
    print('         qbar per face, north -> south: %s'
          % ' '.join('%+.0f' % v for v in qbar))

# ------------------------------------------------- per-section reduction ---
print('\n--- reducing sections (top/bottom %.1f m layers, per FACE) ---'
      % args.hlay)
S = pd.DataFrame()
FACE = {}
for sn in SECTS:
    g = GEO[sn]
    ds = xr.open_dataset(in_dir / (sn + '.nc'))
    tt = pd.DatetimeIndex(ds.time.values)
    DZ = ds.DZ.values
    salt = ds.salt.values
    oxy = ds.oxygen.values * O2_MMOL_TO_MGL          # mmol m-3 -> mg/L
    q = ds.q.values * SGN[sn]                        # + = into the cove
    dd3 = g['dd'][None, None, :]

    cum_hi = np.cumsum(DZ, axis=1)
    cum_lo = cum_hi - DZ
    H = cum_hi[:, -1:, :]
    w_bot = np.clip(args.hlay - cum_lo, 0, DZ) / DZ
    w_top = np.clip(args.hlay - (H - cum_hi), 0, DZ) / DZ

    F = {}
    for nm, w in [('top', w_top), ('bot', w_bot), ('bar', np.ones_like(DZ))]:
        A = dd3 * DZ * w                             # (t,z,p) face area
        As = A.sum(axis=1)                           # (t,p)
        F['s_' + nm] = (salt * A).sum(axis=1) / As   # (t,p)
        F['o_' + nm] = (oxy * A).sum(axis=1) / As
        F['u_' + nm] = (q * w).sum(axis=1) / As
    FACE[sn] = F
    ds.close()

    # the two designs: y alone, and y with depth controlled
    ones = np.ones_like(g['y'])
    Xy = np.column_stack([ones, g['y']])
    Xyh = np.column_stack([ones, g['y'], g['h']])
    Py, Hy = wls(Xy, g['dd'])
    Pyh, _ = wls(Xyh, g['dd'])
    wsum = g['dd'].sum()

    if S.empty:
        S = pd.DataFrame(index=tt)
    for vn in ['s_top', 's_bot', 's_bar', 'o_top', 'o_bot', 'o_bar',
               'u_top', 'u_bot', 'u_bar']:
        M = F[vn]                                    # (t,p)
        S['%s_%s_dy' % (sn, vn)] = godin(M @ Py.T[:, 1])
        S['%s_%s_dy_h' % (sn, vn)] = godin(M @ Pyh.T[:, 1])
        # weighted R2 of the straight line, so the slope can be judged
        fit = M @ Hy.T
        mu = (M * g['dd']).sum(axis=1) / wsum
        ssr = ((M - fit) ** 2 * g['dd']).sum(axis=1)
        sst = ((M - mu[:, None]) ** 2 * g['dd']).sum(axis=1)
        S['%s_%s_r2' % (sn, vn)] = godin(1 - ssr / np.maximum(sst, 1e-12))
        kN, kS, _ = PAIR[sn]
        S['%s_%s_ns' % (sn, vn)] = godin(M[:, kN] - M[:, kS])

TT = S.index
# section-mean bottom DO, width-weighted like every other section mean here
for sn in SECTS:
    w_ = GEO[sn]['dd']
    S['%s_o_bot_mean' % sn] = godin(
        (FACE[sn]['o_bot'] * w_).sum(axis=1) / w_.sum())
S['do_cove'] = S[['%s_o_bot_mean' % sn for sn in SECTS]].mean(axis=1)

print('  linear fit quality (mean weighted R2 of value vs y across the '
      'section)')
print('  %-6s %8s %8s %8s %8s' % ('sect', 's_top', 's_bot', 'u_top', 'u_bot'))
for sn in SECTS:
    print('  %-6s %8.2f %8.2f %8.2f %8.2f'
          % (sn, S['%s_s_top_r2' % sn].mean(), S['%s_s_bot_r2' % sn].mean(),
             S['%s_u_top_r2' % sn].mean(), S['%s_u_bot_r2' % sn].mean()))
print('  a low R2 means the lateral structure is not a ramp and the slope is')
print('  a poor summary of it -- quote the N-S pair difference instead.')

# ------------------------------------------------------------------ wind ---
C = pd.read_pickle(wind_fn)
W = C['W']
wa = W.u_pc.values * ax_ + W.v_pc.values * ay_
wc = W.u_pc.values * cx_ + W.v_pc.values * cy_       # + = toward north shore
spd = W.spd_pc.values
Wd = pd.DataFrame({'w_along': wa, 'w_cross': wc, 'spd': spd,
                   'tau_along': W.tau_pc.values * wa / np.maximum(spd, 1e-6),
                   'tau_cross': W.tau_pc.values * wc / np.maximum(spd, 1e-6)},
                  index=W.index)
Wd = Wd.reindex(Wd.index.union(TT)).interpolate('time').reindex(TT)
for c_ in Wd.columns:
    S[c_] = godin(Wd[c_].values)

print('\n--- cross-cove wind at Penn Cove, subtidal ---')
print('  mean %+.2f m/s, sd %.2f (+ = toward the north shore)'
      % (S.w_cross.mean(), S.w_cross.std()))
print('  blows toward the north shore %.0f%% of hours; |cross| / speed = %.2f'
      % (100 * (S.w_cross > 0).mean(),
         np.nanmean(np.abs(S.w_cross)) / np.nanmean(S.spd)))
print('  |cross| exceeds |along| in %.0f%% of hours -- the cove axis is 255 '
      'deg\n  and the prevailing wind is not on it'
      % (100 * (S.w_cross.abs() > S.w_along.abs()).mean()))

A = pd.DataFrame({c_: anom(S[c_]) for c_ in S.columns}, index=TT)

# ------------------------------------------------------------- the events ---
thr = A.tau_cross.abs().quantile(args.pct / 100)
dist = int(args.minsep * 24)
fin = A.notna().all(axis=1)
v0, v1 = TT[fin][0], TT[fin][-1]
EV, ndrop = {}, 0
for nm, sgn in [('northward', 1.0), ('southward', -1.0)]:
    v = sgn * A.tau_cross.values
    v = np.where(np.isfinite(v), v, -np.inf)
    pk, _ = find_peaks(v, height=thr, distance=dist)
    t = TT[pk]
    keep = ((t - pd.Timedelta(days=args.lead) >= v0)
            & (t + pd.Timedelta(days=args.lag) <= v1))
    ndrop += int((~keep).sum())
    EV[nm] = t[keep]


def met_dir(u, v):
    return (270 - np.rad2deg(np.arctan2(v, u))) % 360


def circ_stats(deg):
    """Vector-mean direction and circular sd. A plain mean/std of compass
    bearings wraps at 0/360 and reports nonsense for any family straddling
    north -- the southward family did, at 'sd 105 deg'."""
    a = np.deg2rad(np.asarray(deg, dtype=float))
    C_, S_ = np.cos(a).mean(), np.sin(a).mean()
    R = np.hypot(C_, S_)
    return (np.rad2deg(np.arctan2(S_, C_)) % 360,
            np.rad2deg(np.sqrt(max(0.0, -2 * np.log(max(R, 1e-12))))))


print('\n--- wind events: peaks in the CROSS-cove stress anomaly ---')
print('  threshold |tau_cross anomaly| > %.4f Pa (the %.0fth percentile), '
      'min %.0f d apart; %d dropped for an incomplete window'
      % (thr, args.pct, args.minsep, ndrop))
ERows = []
for nm in EV:
    for t in EV[nm]:
        ERows.append(dict(family=nm, time=t, tau_cross=S.tau_cross[t],
                          w_cross=S.w_cross[t], w_along=S.w_along[t],
                          spd=S.spd[t], month=t.month,
                          from_deg=met_dir(S.w_along[t] * ax_
                                           + S.w_cross[t] * cx_,
                                           S.w_along[t] * ay_
                                           + S.w_cross[t] * cy_)))
E = pd.DataFrame(ERows).sort_values(['family', 'time'])
E.to_csv(out_dir / 'wind_events_crosscove.csv', index=False,
         float_format='%.4f')
SEAS = dict(DJF=[12, 1, 2], MAM=[3, 4, 5], JJA=[6, 7, 8], SON=[9, 10, 11])
for nm in EV:
    g = E[E.family == nm]
    md, sd_ = circ_stats(g.from_deg)
    print('  %-9s n=%2d  from %3.0f +/- %2.0f deg (circular)  cross %.2f  '
          'along %.2f  speed %.2f'
          % (nm, len(g), md, sd_, g.w_cross.abs().mean(),
             g.w_along.abs().mean(), g.spd.mean()))
    print('            by season: %s' % '  '.join(
        '%s %d' % (k, int(g.month.isin(v).sum())) for k, v in SEAS.items()))

# ------------------------------------------------------------- composites ---
RESP = [('tau_cross', 'cross-cove wind stress', 'Pa', 1.0),
        ('tau_along', 'along-cove wind stress', 'Pa', 1.0),
        ('do_cove', 'bottom DO, cove mean', 'mg L$^{-1}$', 1.0)]
for sn in SECTS:
    RESP += [
        ('%s_s_top_dy' % sn, 'ds/dy surface, %s' % SLAB[sn],
         'g kg$^{-1}$ km$^{-1}$', 1.0),
        ('%s_s_bot_dy_h' % sn, 'ds/dy bottom (h controlled), %s' % SLAB[sn],
         'g kg$^{-1}$ km$^{-1}$', 1.0),
        ('%s_u_top_dy' % sn, 'du/dy surface, %s' % SLAB[sn],
         '10$^{-5}$ s$^{-1}$', 1.0),
        ('%s_u_bot_dy_h' % sn, 'du/dy bottom (h controlled), %s' % SLAB[sn],
         '10$^{-5}$ s$^{-1}$', 1.0),
        ('%s_s_top_ns' % sn, 's surface N - S, %s' % SLAB[sn],
         'g kg$^{-1}$', 1.0),
        ('%s_u_top_ns' % sn, 'u surface N - S, %s' % SLAB[sn],
         'm s$^{-1}$', 1.0),
        ('%s_u_bot_ns' % sn, 'u bottom N - S, %s' % SLAB[sn],
         'm s$^{-1}$', 1.0),
        ('%s_u_bar_ns' % sn, 'u depth-mean N - S, %s' % SLAB[sn],
         'm s$^{-1}$', 1.0),
        ('%s_o_bot_dy_h' % sn, 'dDO/dy bottom (h controlled), %s' % SLAB[sn],
         'mg L$^{-1}$ km$^{-1}$', 1.0),
        ('%s_o_bot_ns' % sn, 'bottom DO N - S, %s' % SLAB[sn],
         'mg L$^{-1}$', 1.0),
        ('%s_o_bot_mean' % sn, 'bottom DO, %s' % SLAB[sn],
         'mg L$^{-1}$', 1.0)]
# du/dy is (m/s)/km = 1e-3 s-1; rescale to 1e-5 s-1 for readability
for c_ in [c for c in S.columns if '_u_' in c and c.endswith(('_dy', '_dy_h'))]:
    S[c_] = S[c_] * 100.0
    A[c_] = A[c_] * 100.0

lags = np.arange(-int(args.lead * 24), int(args.lag * 24) + 1)
lagd = lags / 24.0
BASE = (lagd >= -args.lead) & (lagd <= -args.lead + 1)
COMP, crows = {}, []
for nm in EV:
    D = {}
    for vn, lab, un, sc in RESP:
        v = A[vn].values
        M = np.full((len(EV[nm]), len(lags)), np.nan)
        for i, t in enumerate(EV[nm]):
            j = TT.get_loc(t) + lags
            ok = (j >= 0) & (j < len(TT))
            M[i, ok] = v[j[ok]]
        M = M - np.nanmean(M[:, BASE], axis=1, keepdims=True)
        n = np.sum(np.isfinite(M), axis=0)
        mu = np.nanmean(M, axis=0)
        se = np.nanstd(M, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
        D[vn] = dict(mean=mu * sc, se=se * sc, n=n)
        k = np.nanargmax(np.abs(mu))
        crows.append(dict(family=nm, var=vn, label=lab, unit=un,
                          n_events=len(EV[nm]), peak_dev=mu[k] * sc,
                          peak_lag_d=lagd[k], peak_se=se[k] * sc,
                          sig=abs(mu[k]) > 2 * se[k]))
    COMP[nm] = D
CO = pd.DataFrame(crows)
CO.to_csv(out_dir / 'event_composites.csv', index=False, float_format='%.5f')

print('\n--- composite response to cross-cove wind, deviation from lead-in ---')
print('%-10s %-38s %10s %8s %6s' % ('family', 'response', 'peak dev',
                                    'lag (d)', '2SE?'))
for nm in EV:
    for vn, lab, un, sc in RESP:
        r = CO[(CO.family == nm) & (CO['var'] == vn)].iloc[0]
        print('%-10s %-38s %+10.4f %8.2f %6s'
              % (nm, lab, r.peak_dev, r.peak_lag_d, 'yes' if r.sig else '--'))

# ------------------------------------------------------ lagged correlation ---
# Against BOTH stress components, because the whole point is to find out which
# wind the lateral structure answers to.
AD = A.resample('1D').mean()
LAGS = np.arange(-3, 8)
lrows = []
for frc in ['tau_cross', 'tau_along']:
    tv = AD[frc].values
    for vn, lab, un, sc in RESP:
        if vn.startswith('tau_'):
            continue
        v = AD[vn].values
        for L in LAGS:
            a_ = tv[:len(tv) - L] if L >= 0 else tv[-L:]
            b_ = v[L:] if L >= 0 else v[:len(v) + L]
            r, pp, ne = neff_r(a_, b_)
            lrows.append(dict(forcing=frc, var=vn, label=lab, lag_d=float(L),
                              r=r, p=pp, n_eff=ne))
LC = pd.DataFrame(lrows)
LC.to_csv(out_dir / 'lagged_correlations.csv', index=False,
          float_format='%.4f')

print('\n--- which wind does the lateral structure answer to? ---')
print('%-38s %16s %16s' % ('response', 'vs CROSS stress', 'vs ALONG stress'))
for vn, lab, un, sc in RESP:
    if vn.startswith('tau_'):
        continue
    out = []
    for frc in ['tau_cross', 'tau_along']:
        g = LC[(LC.forcing == frc) & (LC['var'] == vn)].dropna(subset=['r'])
        b = g.iloc[g.r.abs().values.argmax()]
        out.append('%+.2f @%+.0fd%s' % (b.r, b.lag_d,
                                        '*' if b.p < 0.05 else ' '))
    print('%-38s %16s %16s' % (lab, out[0], out[1]))
print('  * = p < 0.05 on %d daily means with an autocorrelation-corrected '
      'n_eff' % len(AD))

# Bottom DO answers the cross-cove wind strongly, but the northward family is
# nearly absent in summer (JJA 2 of 30), so that answer is measured almost
# entirely in water that is already well oxygenated. The season the question
# is actually about is the low-DO one, so it is tested separately here.
print('\n--- bottom DO vs cross-cove stress, by season ---')
print('  the response above is dominated by the oxygenated half of the year;')
print('  this is the same correlation inside each season, at the best lag.')
print('%-8s %6s %10s %8s %8s %10s' % ('season', 'n_days', 'mean DO', 'r',
                                      'lag (d)', 'p'))
mo = AD.index.month
for k, v in SEAS.items():
    m = np.isin(mo, v)
    best = (None, 0, np.nan, 0)
    for L in LAGS:
        a_ = AD.tau_cross.values[m]
        b_ = pd.Series(AD.do_cove.values).shift(-L).values[m]
        r, pp, ne = neff_r(a_, b_)
        if np.isfinite(r) and abs(r) > abs(best[1]):
            best = (L, r, pp, ne)
    print('%-8s %6d %10.2f %+8.2f %8d %10.2g'
          % (k, int(m.sum()), S.do_cove[np.isin(S.index.month, v)].mean(),
             best[1], best[0], best[2]))
print('  a strong r in DJF/MAM and a weak one in JJA would mean the wind')
print('  ventilates the cove only when it is not the season that needs it.')

# =========================================================== figure 1: series
sub = S.dropna(subset=['tau_cross'])
fig, axs = plt.subplots(8, 1, figsize=(15, 20), sharex=True,
                        layout='constrained')
ax = axs[0]
ax.fill_between(sub.index, 0, sub.w_cross, where=sub.w_cross >= 0,
                color=CB['blue'], alpha=0.6, lw=0, label='toward north shore')
ax.fill_between(sub.index, 0, sub.w_cross, where=sub.w_cross < 0,
                color=CB['orange'], alpha=0.6, lw=0, label='toward south shore')
ax.set_ylabel('cross-cove wind\n(m s$^{-1}$), subtidal')
ax.legend(fontsize=8, ncol=2, loc='upper left')

ax = axs[1]
for sn in SECTS:
    ax.plot(sub.index, sub['%s_s_top_dy' % sn], lw=1.3, color=SC[sn],
            label=SLAB[sn])
ax.axhline(0, color='0.5', lw=0.8)
ax.set_ylabel('ds/dy surface\n(g kg$^{-1}$ km$^{-1}$)')
ax.legend(fontsize=8, ncol=3, loc='upper left')

ax = axs[2]
for sn in SECTS:
    ax.plot(sub.index, sub['%s_s_bot_dy_h' % sn], lw=1.3, color=SC[sn],
            label='%s (h controlled)' % SLAB[sn])
ax.axhline(0, color='0.5', lw=0.8)
ax.set_ylabel('ds/dy bottom\n(g kg$^{-1}$ km$^{-1}$)')
ax.legend(fontsize=8, ncol=3, loc='upper left')

ax = axs[3]
for sn in SECTS:
    ax.plot(sub.index, sub['%s_u_top_dy' % sn], lw=1.3, color=SC[sn],
            label='%s surface' % SLAB[sn])
ax.axhline(0, color='0.5', lw=0.8)
ax.set_ylabel('du/dy surface\n(10$^{-5}$ s$^{-1}$)')
ax.legend(fontsize=8, ncol=3, loc='upper left')

# du/dy and the N-S contrast both get one panel PER LAYER. Surface and bottom
# lateral velocity differ by roughly a factor of three here, and stacking them
# on one axis buries the bottom signal -- which is the one that matters for
# what gets flushed out of the deep cove.
ax = axs[4]
for sn in SECTS:
    ax.plot(sub.index, sub['%s_u_bot_dy_h' % sn], lw=1.3, color=SC[sn],
            label='%s (h controlled)' % SLAB[sn])
ax.axhline(0, color='0.5', lw=0.8)
ax.set_ylabel('du/dy BOTTOM\n(10$^{-5}$ s$^{-1}$)')
ax.legend(fontsize=8, ncol=3, loc='upper left')

ax = axs[5]
for sn in SECTS:
    ax.plot(sub.index, sub['%s_u_top_ns' % sn], lw=1.3, color=SC[sn],
            label='%s (rms %.3f)' % (SLAB[sn],
                                     np.sqrt(np.nanmean(
                                         sub['%s_u_top_ns' % sn] ** 2))))
ax.axhline(0, color='0.5', lw=0.8)
ax.set_ylabel('u SURFACE, N - S\n(m s$^{-1}$), depth-matched pair')
ax.legend(fontsize=7, ncol=3, loc='upper left')

ax = axs[6]
for sn in SECTS:
    ax.plot(sub.index, sub['%s_u_bot_ns' % sn], lw=1.3, color=SC[sn],
            label='%s (rms %.3f)' % (SLAB[sn],
                                     np.sqrt(np.nanmean(
                                         sub['%s_u_bot_ns' % sn] ** 2))))
ax.axhline(0, color='0.5', lw=0.8)
ax.set_ylabel('u BOTTOM, N - S\n(m s$^{-1}$), depth-matched pair')
ax.legend(fontsize=7, ncol=3, loc='upper left')

ax = axs[7]
for sn in SECTS:
    ax.plot(sub.index, sub['%s_o_bot_mean' % sn], lw=1.3, color=SC[sn],
            label='%s section mean' % SLAB[sn])
for sn in SECTS:
    ax.plot(sub.index, sub['%s_o_bot_ns' % sn], ':', lw=1.1, color=SC[sn],
            alpha=0.8, label='%s N - S' % SLAB[sn])
ax.axhline(2.0, color=CB['red'], lw=1.0, ls='--', alpha=0.8)
ax.axhline(0, color='0.5', lw=0.8)
ax.text(sub.index[5], 2.05, 'hypoxic 2 mg/L', fontsize=7, color=CB['red'])
ax.set_ylabel('bottom DO\n(mg L$^{-1}$)')
ax.legend(fontsize=7, ncol=3, loc='upper left')

for ax in axs:
    ax.grid(**GRID)
    for nm, c_ in [('northward', CB['blue']), ('southward', CB['orange'])]:
        for t in EV[nm]:
            ax.axvline(t, color=c_, lw=0.8, alpha=0.35, zorder=0)
axs[-1].xaxis.set_major_locator(MonthLocator())
axs[-1].xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
axs[-1].set_xlim(sub.index[0], sub.index[-1])
fig.suptitle('%s -- Penn Cove CROSS-channel structure, Godin-lowpassed.  '
             'y is positive toward the NORTH shore.\n'
             'vertical lines: cross-cove wind events '
             '(blue = toward north, orange = toward south)' % args.gtx,
             fontsize=12)
fn = out_dir / 'fig1_crosschannel_series.png'
fig.savefig(fn, dpi=170, bbox_inches='tight')
plt.close(fig)
print('\nsaved %s' % fn)

# ====================================================== figure 2: composites
LOOK = {vn: (lab, un, sc) for vn, lab, un, sc in RESP}
SHOW = ['tau_cross', 'do_cove']
for sn in SECTS:
    SHOW += ['%s_s_top_dy' % sn, '%s_u_top_ns' % sn, '%s_u_bot_ns' % sn,
             '%s_o_bot_ns' % sn]
SHOW = SHOW[:14]
fig, axs = plt.subplots(7, 2, figsize=(13, 19), sharex=True,
                        layout='constrained')
FC = {'southward': CB['orange'], 'northward': CB['blue']}
for i, vn in enumerate(SHOW):
    ax = axs[i % 7][i // 7]
    lab, un, sc = LOOK[vn]
    for nm in ['southward', 'northward']:
        d = COMP[nm][vn]
        ax.plot(lagd, d['mean'], lw=2, color=FC[nm],
                label='%s (n=%d)' % (nm, len(EV[nm])))
        ax.fill_between(lagd, d['mean'] - 2 * d['se'], d['mean'] + 2 * d['se'],
                        color=FC[nm], alpha=0.16, lw=0)
    ax.axhline(0, color='0.5', lw=0.8)
    ax.axvline(0, color='0.5', lw=0.8)
    ax.grid(**GRID)
    ax.set_title('%s  (%s)' % (lab, un), fontsize=10)
    yl = ax.get_ylim()
    ax.fill_between([lagd[BASE][0], lagd[BASE][-1]], yl[0], yl[1], color='0.6',
                    alpha=0.12, lw=0, zorder=0)
    ax.set_ylim(yl)
    if i == 0:
        ax.legend(fontsize=8)
for ax in axs[-1]:
    ax.set_xlabel('days from the peak of the cross-cove wind event')
fig.suptitle('Penn Cove: composite lateral response to CROSS-cove wind '
             'events\n30-day rolling anomalies of the subtidal series, '
             'referenced to the shaded lead-in day; band = +/- 2 SE',
             fontsize=12)
fn = out_dir / 'fig2_event_composites.png'
fig.savefig(fn, dpi=170, bbox_inches='tight')
plt.close(fig)
print('saved %s' % fn)

# ============================================== figure 3: which wind wins
fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), layout='constrained',
                        sharey=True)
for j, frc in enumerate(['tau_cross', 'tau_along']):
    ax = axs[j]
    for sn in SECTS:
        for vn, ls in [('%s_s_top_dy' % sn, '-'), ('%s_u_top_dy' % sn, '--')]:
            g = LC[(LC.forcing == frc) & (LC['var'] == vn)]
            ax.plot(g.lag_d, g.r, ls, lw=2, color=SC[sn], marker='o', ms=4,
                    mfc='none',
                    label='%s %s' % (SLAB[sn],
                                     'ds/dy' if '_s_' in vn else 'du/dy'))
            m = (g.p < 0.05).values
            ax.plot(g.lag_d[m], g.r[m], 'o', ms=7, color=SC[sn])
    ax.axhline(0, color='0.5', lw=0.8)
    ax.axvline(0, color='0.5', lw=0.8)
    ax.grid(**GRID)
    ax.set_xlabel('lag (days); positive = the response follows the wind')
    ax.set_title('vs %s-cove stress' % frc.split('_')[1], fontsize=11)
    if j == 0:
        ax.set_ylabel('correlation with the stress anomaly')
        ax.legend(fontsize=8, ncol=2)
fig.suptitle('Surface cross-channel gradients against each wind component.  '
             'daily means; large filled markers: p < 0.05 with an '
             'autocorrelation-corrected n_eff', fontsize=12)
fn = out_dir / 'fig3_which_wind.png'
fig.savefig(fn, dpi=170, bbox_inches='tight')
plt.close(fig)
print('saved %s' % fn)

# ------------------------------------------------------------------ tables ---
M = pd.DataFrame(index=SECTS)
for sn in SECTS:
    kN, kS, _ = PAIR[sn]
    M.loc[sn, 'width_km'] = GEO[sn]['dd'].sum() / 1e3
    M.loc[sn, 'corr_y_h'] = np.corrcoef(GEO[sn]['y'], GEO[sn]['h'])[0, 1]
    for vn in ['s_top', 's_bot', 'u_top', 'u_bot']:
        M.loc[sn, vn + '_dy'] = S['%s_%s_dy' % (sn, vn)].mean()
        M.loc[sn, vn + '_dy_h'] = S['%s_%s_dy_h' % (sn, vn)].mean()
        M.loc[sn, vn + '_ns'] = S['%s_%s_ns' % (sn, vn)].mean()
M.to_csv(out_dir / 'section_cross_means.csv', float_format='%.5f')
S.to_csv(out_dir / 'crosschannel_subtidal.csv', float_format='%.5f')

print('\n--- time-mean cross-channel state ---')
print('  ds/dy in g/kg/km, du/dy in 1e-5 s-1, N-S in g/kg or m/s')
print(M.round(4).to_string())
for fn_ in ['wind_events_crosscove.csv', 'event_composites.csv',
            'lagged_correlations.csv', 'section_cross_means.csv',
            'crosschannel_subtidal.csv']:
    print('saved %s' % (out_dir / fn_))
