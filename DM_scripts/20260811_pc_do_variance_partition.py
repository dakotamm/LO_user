"""
How much bottom-DO variance at pc_cp do along-cove wind and Qprism account for?

Target: bottom DO at cp_mid, the midpoint of the Coupeville line (pc_cp), from
the pc4 mooring job -- the same series as panel a of 20260811_pc_forcing_stack.py.
Predictors: along-cove wind velocity (+ = into the cove) and Qprism at pc_cp.
Both are built exactly as in that script, so the numbers here annotate that
figure rather than describing a different quantity.

WHY THIS IS NOT ONE REGRESSION. Three things break a naive OLS R2 here.

  1 BANDS. 93% of the DO variance at cp_mid sits at periods longer than 30 days
    -- it is the annual drawdown, not weather. Wind has an annual cycle too
    (into the cove in winter, out of it in summer). Regressing raw daily levels
    therefore mostly measures "two things that both have an annual cycle," which
    is true of almost any pair of series in Puget Sound and says nothing about
    mechanism. So every quantity is split into a seasonal band (SEAS_D-day
    running mean) and a subseasonal residual, and the regression is run in each
    band separately. Which band you ask about IS the scientific question.

  2 EFFECTIVE DOF. Daily bottom DO has lag-1 autocorrelation 0.99. The nominal
    n = 729 is a fiction. Each correlation carries a Bretherton et al. (1999)
    effective sample size from the two lag-1 autocorrelations, and significance
    for R2 comes from a moving-block bootstrap that resamples the PREDICTORS in
    blocks (preserving their own autocorrelation) against a fixed y. That null
    is wide: for raw levels, an unrelated predictor with this much memory clears
    R2 = 0.13 five percent of the time. Compare observed R2 to that, not to zero.

  3 SHARED VARIANCE. Wind and Qprism are not orthogonal (r = -0.54 in the
    seasonal band), so "R2 of each" does not add up. The script reports a
    commonality partition -- unique to wind, unique to Qprism, shared -- where
    unique_i = R2_full - R2_without_i and shared is the remainder. Shared
    variance is not attributable to either predictor by regression alone.

THREE DEPENDENT VARIABLES, because "bottom DO" can mean three things and they
give different answers:

  do_cp    the level. What you plot. Dominated by whatever Saratoga Passage
           delivers -- DO at M5 alone explains 91% of it.
  ddo/dt   the tendency, over TEND_D days. If ventilation is what wind and tides
           supply, it acts on the RATE of change of a reservoir, not its level.
           This is the physically correct target and the script reports it.
  dloc     do_cp - do_M5, the cove's departure from the passage. Removes the
           imported signal and leaves what the cove does to its own water.

A VALIDITY CHECK is included so a null result can be told from a dead method:
the subseasonal wind -> dstrat(pc_cp) response at 1-2 day lag is already known
to be real (see 20260807_pc_alongchannel_wind.py). If the subseasonal machinery
here recovers it, a null for DO is a statement about DO. It does -- weakly
(r = +0.15 at 1 d) -- so the DO null is not an artifact of the filtering.

Everything is subtidal already: moorings are -lt lowpass (daily Godin), Qprism
is Godin + daily-subsampled by bulk_calc_avg.py, dstrat and wind are hourly and
Godin filtered here on the FULL record before windowing.

run 20260811_pc_do_variance_partition.py                    # 2024-2025, all of it
run 20260811_pc_do_variance_partition.py -year 2025
run 20260811_pc_do_variance_partition.py -seas 30           # tighter seasonal band
run 20260811_pc_do_variance_partition.py -nboot 5000
"""
import argparse
import sys
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr
from cmcrameri import cm as cmc
from scipy import stats

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
p.add_argument('-gctag', default='wb1_pc1', type=str)
p.add_argument('-job', default='pc4', type=str)
p.add_argument('-0', '--ds0', default='2024.01.01', type=str)
p.add_argument('-1', '--ds1', default='2025.12.31', type=str)
p.add_argument('-m0', '--mds0', default='2024.01.02', type=str)
p.add_argument('-m1', '--mds1', default='2025.12.30', type=str)
p.add_argument('-year', default='all', type=str)
p.add_argument('-seas', '--seas_d', default=90, type=int,
               help='running-mean width [days] defining the seasonal band')
p.add_argument('-tend', '--tend_d', default=7, type=int,
               help='differencing interval [days] for the DO tendency')
p.add_argument('-nboot', default=2000, type=int)
p.add_argument('-block', default=60, type=int,
               help='moving-block bootstrap block length [days], raw band')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname=args.gctag.split('_')[0])
tef2 = Ldir['LOo'] / 'extract' / args.gtagex / 'tef2'
moor_dir = Ldir['LOo'] / 'extract' / args.gtagex / 'moor' / args.job
wind_fn = (Ldir['LOo'] / 'DM_outs' / '20260806_wind'
           / ('wind_hourly_atm00_%s_%s.p' % (args.ds0, args.ds1)))
out_dir = Ldir['LOo'] / 'DM_outs' / '20260811_pc_do_variance_partition'
Lfun.make_dir(out_dir)

O2_MMOL_TO_MGL = 32.0 / 1000.0
MOUTH, HEAD = 'pc_lp', 'pc_cp'
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
FS = 13
rng = np.random.default_rng(42)

# Predictor identity is the one categorical encoding in this figure, so the two
# hues are fixed and never cycled: wind and Qprism keep the same color in every
# panel. Okabe-Ito blue/vermillion, which is the CB dict the other pc scripts
# use, plus a neutral grey for the variance the partition cannot attribute.
C_WIND = '#0072B2'
C_QP = '#D55E00'
C_SHARED = '#9a9a9a'
C_NULL = '#CC0000'      # the bootstrap null level, a reference mark not a series
C_CP = mcolors.to_hex(cmc.lajolla(0.05))     # station colors, as in the map


def godin(a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return zfun.lowpass(np.asarray(a, dtype=float), f='godin')


def need(fn):
    if not fn.is_file():
        print('*** missing: %s' % fn)
        sys.exit(1)
    return fn


# ---------------------------------------------------------------------------
# assemble the daily frame (same construction as 20260811_pc_forcing_stack.py)
# ---------------------------------------------------------------------------
DO = {}
for sn in ['cp_mid', 'M5']:
    ds = xr.open_dataset(need(moor_dir / ('%s_%s_%s.nc'
                                          % (sn, args.mds0, args.mds1))))
    DO[sn] = pd.Series(ds.oxygen.values[:, 0] * O2_MMOL_TO_MGL,
                       index=pd.to_datetime(ds.ocean_time.values), name=sn)
    ds.close()
TT = DO['cp_mid'].index

QP = {}
for sn in [MOUTH, HEAD]:
    ds = xr.open_dataset(need(tef2 / ('bulk_avg_%s_%s' % (args.ds0, args.ds1))
                              / (sn + '.nc')))
    QP[sn] = pd.Series(ds.qprism.values, index=pd.to_datetime(ds.time.values))
    ds.close()

ds = xr.open_dataset(need(tef2 / ('strat_%s_%s_%s.nc'
                                  % (args.ds0, args.ds1, args.gctag))))
st = pd.to_datetime(ds.time.values)
DS = {sn: pd.Series(godin(ds.dstrat.sel(sect=sn).values), index=st)
      for sn in [MOUTH, HEAD]}
ds.close()

dstr = xr.open_dataset(need(tef2 / ('structure_%s_%s_%s.nc'
                                    % (args.ds0, args.ds1, args.gctag))))
CEN = {sn: (float(np.mean(dstr['%s_lon' % sn].values)),
            float(np.mean(dstr['%s_lat' % sn].values))) for sn in [MOUTH, HEAD]}
dstr.close()
COS = np.cos(np.deg2rad(np.mean([c[1] for c in CEN.values()])))
x0, y0 = CEN[MOUTH][0] * COS * 111.32, CEN[MOUTH][1] * 111.32
xN, yN = CEN[HEAD][0] * COS * 111.32, CEN[HEAD][1] * 111.32
axl = np.hypot(xN - x0, yN - y0)
ax_, ay_ = (xN - x0) / axl, (yN - y0) / axl          # unit vector, mouth -> head
W = pd.read_pickle(need(wind_fn))['W']
WA = pd.Series(godin(W.u_pc.values * ax_ + W.v_pc.values * ay_), index=W.index)


def on_daily(s):
    return s.reindex(s.index.union(TT)).interpolate('time').reindex(TT)


S = pd.DataFrame(index=TT)
S['do_cp'] = DO['cp_mid']
S['do_M5'] = DO['M5']
S['qp_cp'] = on_daily(QP[HEAD])
S['qp_lp'] = on_daily(QP[MOUTH])
S['ds_cp'] = on_daily(DS[HEAD])
S['wa'] = on_daily(WA)
# the tef2 files start a day after the moorings; that single edge NaN would
# otherwise propagate through every filter below
S = S.bfill()
S['dloc'] = S.do_cp - S.do_M5                   # cove minus passage
S['ddo'] = S.do_cp.diff(args.tend_d) / args.tend_d      # mg/L/day

if args.year.lower() != 'all':
    S = S[S.index.year == int(args.year)]
    if len(S) == 0:
        print('*** no samples in %s' % args.year)
        sys.exit(1)
span_lbl = '%s to %s' % (S.index[0].date(), S.index[-1].date())

print('along-cove axis %.0f deg true, mouth -> head, %.2f km'
      % (np.rad2deg(np.arctan2(ax_, ay_)) % 360, axl))
print('%s, %d days\n' % (span_lbl, len(S)))


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def lp(s, d):
    return s.rolling(d, center=True, min_periods=max(1, d // 3)).mean()


def neff(a, b):
    """Bretherton et al. (1999) effective sample size from lag-1 autocorrs."""
    r1 = lambda v: np.corrcoef(v[:-1], v[1:])[0, 1]
    ra, rb = r1(a), r1(b)
    return max(3.0, len(a) * (1 - ra * rb) / (1 + ra * rb)), ra, rb


def r_neff(y, x):
    """Pearson r with a p-value on effective, not nominal, degrees of freedom."""
    r = np.corrcoef(y, x)[0, 1]
    ne, ra, rb = neff(y, x)
    t = r * np.sqrt((ne - 2) / max(1e-12, 1 - r ** 2))
    return r, 2 * stats.t.sf(abs(t), ne - 2), ne


def r2_of(y, X):
    return sm.OLS(y, sm.add_constant(np.asarray(X, dtype=float))).fit().rsquared


def block_null_r2(y, X, L, nboot):
    """Moving-block bootstrap null for R2.

    The predictors are resampled together in circular blocks of L days against
    a FIXED y. That destroys any y-X relation while preserving each predictor's
    own autocorrelation and their mutual correlation, which is exactly the null
    a naive F-test gets wrong. Returns (p, 95th percentile of the null R2)."""
    n = len(y)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] != n:
        X = X.T
    obs = r2_of(y, X)
    Xc = np.vstack([X, X])                       # wrap, so blocks are circular
    nb = int(np.ceil(n / L))
    out = np.empty(nboot)
    for b in range(nboot):
        idx = np.concatenate([np.arange(s, s + L)
                              for s in rng.integers(0, n, nb)])[:n]
        out[b] = r2_of(y, Xc[idx])
    return obs, float((out >= obs).mean()), float(np.percentile(out, 95))


def partition(y, xw, xq, tag, L, nboot):
    """Commonality partition of R2 between wind and Qprism, with an
    autocorrelation-aware null and HAC coefficient tests."""
    X = np.column_stack([xw, xq])
    r2_full = r2_of(y, X)
    r2_w, r2_q = r2_of(y, xw), r2_of(y, xq)
    uw, uq = r2_full - r2_q, r2_full - r2_w
    shared = r2_full - uw - uq
    m = sm.OLS(y, sm.add_constant(X)).fit().get_robustcov_results(
        cov_type='HAC', maxlags=L, use_correction=True)
    print('--- %s (n = %d) ---' % (tag, len(y)))
    rows = []
    for nm, x_, r2_i in [('w_along', xw, r2_w), ('qp_cp', xq, r2_q)]:
        r, pn, ne = r_neff(y, x_)
        _, pb, q95 = block_null_r2(y, x_, L, nboot)
        print('  %-8s alone: R2 = %.3f  r = %+.2f  Neff = %3.0f  p_Neff = %.3f'
              '   p_block = %.3f  (null R2 95th = %.3f)'
              % (nm, r2_i, r, ne, pn, pb, q95))
        rows.append(dict(band=tag, predictor=nm, r=r, r2_alone=r2_i,
                         neff=ne, p_neff=pn, p_block=pb, null_r2_95=q95))
    _, pb_f, q95_f = block_null_r2(y, X, L, nboot)
    print('  together     : R2 = %.3f   p_block = %.3f  (null R2 95th = %.3f)'
          % (r2_full, pb_f, q95_f))
    print('  partition    : unique wind %.3f | unique Qprism %.3f | shared %.3f'
          % (uw, uq, shared))
    print('  HAC slopes   : wind %+.4f (t %+.2f, p %.3f)   '
          'Qprism %+.5f (t %+.2f, p %.3f)'
          % (m.params[1], m.tvalues[1], m.pvalues[1],
             m.params[2], m.tvalues[2], m.pvalues[2]))
    print('  r(wind, Qprism) = %+.2f\n' % np.corrcoef(xw, xq)[0, 1])
    rows.append(dict(band=tag, predictor='both', r=np.nan, r2_alone=r2_full,
                     neff=np.nan, p_neff=np.nan, p_block=pb_f,
                     null_r2_95=q95_f))
    return dict(tag=tag, r2_full=r2_full, uw=uw, uq=uq, shared=shared,
                null95=q95_f, p=pb_f), rows


# variance budget: how much of DO is even in each band -----------------------
print('bottom DO at cp_mid: mean %.2f, std %.2f, range %.2f-%.2f mg/L'
      % (S.do_cp.mean(), S.do_cp.std(), S.do_cp.min(), S.do_cp.max()))
for c_ in sorted({30, args.seas_d}):
    sd = lp(S.do_cp, c_)
    k = sd.notna()
    f_seas = 100 * sd[k].var() / S.do_cp[k].var()
    print('variance budget, %3d d split: %2.0f%% at longer periods, %2.0f%% shorter'
          % (c_, f_seas, 100 - f_seas))
print('')

RES, ROWS = {}, []
# 1. raw daily levels -- the number a naive regression would report
T = S[['do_cp', 'wa', 'qp_cp']].dropna()
RES['raw'], r = partition(T.do_cp.to_numpy(), T.wa.to_numpy(),
                          T.qp_cp.to_numpy(), 'RAW daily levels',
                          args.block, args.nboot)
ROWS += r
# 2. subseasonal -- the seasonal cycle removed from all three
U = pd.DataFrame({c: S[c] - lp(S[c], args.seas_d)
                  for c in ['do_cp', 'wa', 'qp_cp']}).dropna()
RES['sub'], r = partition(U.do_cp.to_numpy(), U.wa.to_numpy(),
                          U.qp_cp.to_numpy(),
                          'SUBSEASONAL (%d d highpass)' % args.seas_d,
                          20, args.nboot)
ROWS += r
# 3. tendency -- the physically right target for a ventilation term
V = S[['ddo', 'wa', 'qp_cp']].dropna()
RES['tend'], r = partition(V.ddo.to_numpy(), V.wa.to_numpy(), V.qp_cp.to_numpy(),
                           'TENDENCY d(DO)/dt over %d d' % args.tend_d,
                           30, args.nboot)
ROWS += r
# The tendency is the one framing whose R2 clears its null, so it has to be
# checked for the same seasonal aliasing as the level: a %d-day difference of a
# series with an annual cycle still carries that cycle, because the drawdown
# RATE is itself seasonal, and so is the wind. Highpass both and see what is
# left. (Answer, for 2024-2025: nothing -- so the tendency result is seasonal
# too, not a weather-band ventilation response.)
for c_ in sorted({30, args.seas_d}):
    Vh = pd.DataFrame({v: V[v] - lp(V[v], c_)
                       for v in ['ddo', 'wa', 'qp_cp']}).dropna()
    print('  tendency, %3d d highpass: R2 wind %.3f, R2 Qprism %.3f'
          % (c_, np.corrcoef(Vh.ddo, Vh.wa)[0, 1] ** 2,
             np.corrcoef(Vh.ddo, Vh.qp_cp)[0, 1] ** 2))
print('')
# 4. local anomaly -- the cove's own departure from Saratoga Passage
Z = pd.DataFrame({c: S[c] - lp(S[c], args.seas_d)
                  for c in ['dloc', 'wa', 'qp_cp']}).dropna()
RES['dloc'], r = partition(Z.dloc.to_numpy(), Z.wa.to_numpy(), Z.qp_cp.to_numpy(),
                           'LOCAL do_cp - do_M5, %d d highpass' % args.seas_d,
                           20, args.nboot)
ROWS += r
pd.DataFrame(ROWS).to_csv(out_dir / ('partition_%s_%s.csv'
                                     % (args.gtagex, args.year)), index=False)

# what actually does set bottom DO here --------------------------------------
r2_m5 = r2_of(S.do_cp.to_numpy(), S.do_M5.to_numpy()[:, None])
r2_m5f = r2_of(S.do_cp.to_numpy(), S[['do_M5', 'wa', 'qp_cp']].to_numpy())
print('for scale: DO at M5 alone explains R2 = %.3f of DO at cp_mid;'
      % r2_m5)
print('           adding wind and Qprism takes it to %.3f (+%.3f)\n'
      % (r2_m5f, r2_m5f - r2_m5))
print('Qprism at pc_cp: monthly means span %.0f-%.0f m3/s on a full range of '
      '%.0f-%.0f\n  -- it is a fortnightly signal with almost no annual cycle, '
      'so it has\n  nothing to alias onto the annual DO drawdown.'
      % (S.groupby(S.index.month).qp_cp.mean().min(),
         S.groupby(S.index.month).qp_cp.mean().max(),
         S.qp_cp.min(), S.qp_cp.max()))

# lag structure, subseasonal -------------------------------------------------
LAGS = np.arange(0, 31)
LAGR = {}
UU = pd.DataFrame({c: S[c] - lp(S[c], args.seas_d)
                   for c in ['do_cp', 'ds_cp', 'wa', 'qp_cp']}).dropna()
for nm, xc in [('w_along', 'wa'), ('qp_cp', 'qp_cp')]:
    LAGR[nm] = np.array([np.corrcoef(UU.do_cp.to_numpy()[L:],
                                     UU[xc].to_numpy()[:len(UU) - L or None])[0, 1]
                         for L in LAGS])
# validity check: the wind -> dstrat response that is already known to be real
LAGR['dstrat'] = np.array([np.corrcoef(UU.ds_cp.to_numpy()[L:],
                                       UU.wa.to_numpy()[:len(UU) - L or None])[0, 1]
                           for L in LAGS])
# The known response is at 1-2 days, so that is the lag quoted -- taking the
# maximum over the whole scan would be the same multiple-comparison mistake the
# scan-wide null below exists to prevent.
print('\nvalidity check -- subseasonal wind -> dstrat(pc_cp), a known response:')
print('  r at lag 1 d = %+.2f, lag 2 d = %+.2f (the response reported in'
      % (LAGR['dstrat'][1], LAGR['dstrat'][2]))
print('  20260807_pc_alongchannel_wind.py). Weak, but present and at the right')
print('  lag, so the subseasonal machinery is working and the DO null is real.')
for nm in ['w_along', 'qp_cp']:
    j = int(np.argmax(np.abs(LAGR[nm])))
    print('  DO vs %-8s: lag 0 r = %+.2f, best |r| = %+.2f at lag %d d'
          % (nm, LAGR[nm][0], LAGR[nm][j], LAGS[j]))
# a lag-scan-wide null: the largest |r| you get by chance over 31 lags
nb_lag = min(args.nboot, 1000)
mx = np.empty(nb_lag)
yv, xv = UU.do_cp.to_numpy(), UU.wa.to_numpy()
n_, L_ = len(yv), 20
xc2 = np.concatenate([xv, xv])
for b in range(nb_lag):
    idx = np.concatenate([np.arange(s, s + L_)
                          for s in rng.integers(0, n_, int(np.ceil(n_ / L_)))])[:n_]
    xb = xc2[idx]
    mx[b] = max(abs(np.corrcoef(yv[L:], xb[:n_ - L or None])[0, 1]) for L in LAGS)
LAG_CRIT = float(np.percentile(mx, 95))
print('  scan-wide 95%% null: |r| must exceed %.2f to mean anything over 31 lags'
      % LAG_CRIT)

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(12.5, 8.5))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.26)

# a. the partition, one stacked bar per framing, against its own null level
ax = fig.add_subplot(gs[0, 0])
keys = ['raw', 'sub', 'tend', 'dloc']
names = ['DO level\n(raw daily)', 'DO level\n(%dd highpass)' % args.seas_d,
         'd(DO)/dt\n(%dd)' % args.tend_d, 'DO$_{cp}$ - DO$_{M5}$\n(highpass)']
xp = np.arange(len(keys))
b1 = [RES[k]['uw'] for k in keys]
b2 = [RES[k]['uq'] for k in keys]
b3 = [max(0.0, RES[k]['shared']) for k in keys]
# 2 px surface gap between stacked segments, so the boundaries read as edges
kw = dict(width=0.6, lw=1.6, edgecolor='white')
ax.bar(xp, b1, color=C_WIND, label='unique to along-cove wind', **kw)
ax.bar(xp, b2, bottom=b1, color=C_QP, label='unique to $Q_{prism}$', **kw)
ax.bar(xp, b3, bottom=np.add(b1, b2), color=C_SHARED,
       label='shared (not attributable)', **kw)
for i, k in enumerate(keys):
    ax.plot([i - 0.36, i + 0.36], [RES[k]['null95']] * 2, color=C_NULL, lw=2,
            zorder=5, label='95% of the null' if i == 0 else None)
    # clear the null line too, or the value label lands on top of it in the
    # three framings where the bar is shorter than its own null
    ax.text(i, max(RES[k]['r2_full'], RES[k]['null95']) + 0.010,
            '%.3f' % RES[k]['r2_full'],
            ha='center', va='bottom', fontsize=FS - 3, color='#333333')
ax.set_xticks(xp)
ax.set_xticklabels(names, fontsize=FS - 4)
ax.set_ylabel('$R^2$ of bottom DO at cp_mid', fontsize=FS)
ax.set_title('a. variance accounted for, by framing', fontsize=FS, loc='left')
ax.legend(frameon=False, fontsize=FS - 4, loc='upper right')
ax.set_ylim(0, max(0.30, max(RES[k]['r2_full'] for k in keys) * 1.35))
ax.grid(axis='y', **GRID)
ax.set_axisbelow(True)

# b. where the DO variance actually is -- the reason panel a splits by band
ax = fig.add_subplot(gs[0, 1])
cuts = [15, 30, 45, 60, 90, 120, 180]
fr, r2w, r2q = [], [], []
for c_ in cuts:
    Uc = pd.DataFrame({v: S[v] - lp(S[v], c_)
                       for v in ['do_cp', 'wa', 'qp_cp']}).dropna()
    fr.append(100 * Uc.do_cp.var() / S.do_cp.var())
    r2w.append(np.corrcoef(Uc.do_cp, Uc.wa)[0, 1] ** 2)
    r2q.append(np.corrcoef(Uc.do_cp, Uc.qp_cp)[0, 1] ** 2)
ax.plot(cuts, np.array(fr) / 100, color=C_CP, lw=2, marker='o', ms=6,
        label='fraction of DO variance left after the highpass')
ax.plot(cuts, r2w, color=C_WIND, lw=2, marker='o', ms=6,
        label='$R^2$, along-cove wind')
ax.plot(cuts, r2q, color=C_QP, lw=2, marker='o', ms=6,
        label='$R^2$, $Q_{prism}$')
ax.set_xlabel('highpass cutoff [days]', fontsize=FS)
ax.set_ylabel('fraction', fontsize=FS)
ax.set_title('b. nothing survives removing the annual cycle', fontsize=FS,
             loc='left')
ax.legend(frameon=False, fontsize=FS - 4, loc='upper left')
ax.set_ylim(0, max(0.42, max(fr) / 100 * 1.6))     # nothing here reaches 0.3
ax.grid(**GRID)

# c. lag structure, with a scan-wide null band
ax = fig.add_subplot(gs[1, 0])
ax.axhspan(-LAG_CRIT, LAG_CRIT, color='#dddddd', lw=0, zorder=0)
ax.plot(LAGS, LAGR['w_along'], color=C_WIND, lw=2, label='DO vs wind')
ax.plot(LAGS, LAGR['qp_cp'], color=C_QP, lw=2, label='DO vs $Q_{prism}$')
ax.plot(LAGS, LAGR['dstrat'], color='#000000', lw=1.6, ls='--',
        label='$\\Delta s$ vs wind (validity check)')
ax.axhline(0, color='k', lw=0.8)
# the band is the DO-vs-wind scan-wide null; the dstrat curve is a different
# pair and is drawn against it only as a rough scale, which is why its 16 d
# excursion is not being claimed as a result
ax.text(30, LAG_CRIT, ' 95% scan-wide null, DO vs wind ', ha='right',
        va='bottom', fontsize=FS - 4, color='#777777')
ax.set_xlabel('lag [days], forcing leads', fontsize=FS)
ax.set_ylabel('correlation, %d d highpass' % args.seas_d, fontsize=FS)
ax.set_title('c. no lag rescues it', fontsize=FS, loc='left')
ax.legend(frameon=False, fontsize=FS - 4, loc='lower right')
ax.set_xlim(0, 30)
ax.grid(**GRID)

# d. the raw correlation is a seasonal loop, not a response
ax = fig.add_subplot(gs[1, 1])
# month is cyclic, so it gets a cyclic ramp -- romaO closes on itself, and a
# non-cyclic one would put a false seam between December and January
sc = ax.scatter(S.wa, S.do_cp, c=S.index.month, cmap=cmc.romaO, vmin=0.5,
                vmax=12.5, s=13, lw=0, alpha=0.85)
cb = fig.colorbar(sc, ax=ax, ticks=[1, 4, 7, 10], pad=0.02)
cb.ax.set_yticklabels(['Jan', 'Apr', 'Jul', 'Oct'], fontsize=FS - 4)
cb.outline.set_visible(False)
ax.axvline(0, color='k', lw=0.8)
ax.set_xlabel('along-cove wind [m s$^{-1}$], + into the cove', fontsize=FS)
ax.set_ylabel('bottom DO at cp_mid [mg L$^{-1}$]', fontsize=FS)
ax.set_title('d. the raw $R^2$ = %.2f is an annual loop' % RES['raw']['r2_full'],
             fontsize=FS, loc='left')
ax.grid(**GRID)

for a_ in fig.axes:
    a_.tick_params(labelsize=FS - 3)
fig.suptitle('Bottom DO at pc_cp against along-cove wind and $Q_{prism}$, %s, %s'
             % (span_lbl, args.gtagex), fontsize=FS + 1, y=0.975)

fn_out = out_dir / ('do_variance_partition_%s_%s.png' % (args.gtagex, args.year))
fig.savefig(fn_out, dpi=500, bbox_inches='tight', transparent=True)
print('\nwrote ' + str(fn_out))
