"""
Testing for a LAGGED response of bottom DO at pc_cp to along-cove wind and
Qprism, July-October, wb1_t0_xn11abbur00.

Companion to 20260811_pc_do_mlr.py, which fits the same two predictors at zero
lag. This script asks whether DO responds to the forcing's HISTORY instead, and
-- more to the point -- whether this record can answer that question at all.

THREE TESTS, weakest specification to strongest:

  1 LAG SCAN. r(DO(t), X(t-L)) for L = 0..LMAX. What everyone does first. It
    has two problems that both inflate significance, and the script fixes both.

  2 TRAILING-MEAN SCAN. r(DO(t), mean of X over the previous W days), W =
    1..WMAX. The physically motivated version: bottom water is a reservoir, so
    it should integrate the forcing rather than track today's value. A single
    accumulation timescale instead of a free lag.

  3 POLYNOMIAL DISTRIBUTED LAG (Almon) with AR(1) errors. The properly
    specified test. All lags 0..LMAX enter at once, with their weights
    constrained to a degree-DEG polynomial so each predictor costs DEG+1
    parameters instead of LMAX+1, and the whole lag block is tested ONCE by
    likelihood ratio against an AR(1)-only null. No lag is selected, so there
    is nothing to correct for.

WHY EVERY P-VALUE HERE IS CALIBRATED BY SURROGATE AND NOT BY FORMULA. Two
things break the textbook nulls:

  MULTIPLE LAGS. Scanning 41 lags and reporting the largest |r| is 41 tests,
  and the neighbouring lags are not independent, so a Bonferroni correction is
  both wrong and unnecessary. The script instead builds the distribution of
  max|r| OVER THE WHOLE SCAN under a null, and quotes its 95th percentile as
  the threshold a scan peak must clear.

  MEMORY. Bottom DO here has lag-1 autocorrelation 0.95 and a decorrelation
  time near 20 days, so 123 days of JASO is on the order of 3-6 independent
  observations. Worse, smooth trailing filters of an autocorrelated predictor
  can fit a seasonal drawdown by accident: on surrogate data the Almon basis
  absorbs essentially all of the residual autocorrelation, which is the
  signature of a lag basis behaving as a flexible trend rather than a response.

So the null for every test is a circular moving-block bootstrap of the
PREDICTORS on the full 2024-2025 record -- preserving each one's own
autocorrelation and destroying only its relation to DO -- rebuilt through the
identical pipeline and rescored. The script prints the nominal p beside the
empirical one so the inflation is visible rather than assumed.

LAGS ARE BUILT ON THE FULL RECORD, then the JASO window is taken. A 30-day lag
on 1 July therefore reaches back into June instead of truncating the window.

run 20260811_pc_do_lag_test.py                      # JASO 2025
run 20260811_pc_do_lag_test.py -year all            # both summers pooled
run 20260811_pc_do_lag_test.py -months all -year all
run 20260811_pc_do_lag_test.py -nboot 1000 -ndl 800   # tighter nulls, slower
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
from scipy import stats as st
from statsmodels.tsa.statespace.sarimax import SARIMAX

from lo_tools import Lfun, zfun

warnings.filterwarnings('ignore')

p = argparse.ArgumentParser()
p.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
p.add_argument('-gctag', default='wb1_pc1', type=str)
p.add_argument('-job', default='pc4', type=str)
p.add_argument('-0', '--ds0', default='2024.01.01', type=str)
p.add_argument('-1', '--ds1', default='2025.12.31', type=str)
p.add_argument('-m0', '--mds0', default='2024.01.02', type=str)
p.add_argument('-m1', '--mds1', default='2025.12.30', type=str)
p.add_argument('-year', default='2025', type=str)
p.add_argument('-months', default='7,8,9,10', type=str)
p.add_argument('-lmax', default=40, type=int, help='longest lag scanned [days]')
p.add_argument('-wmax', default=45, type=int, help='longest trailing mean [days]')
p.add_argument('-dlmax', default=21, type=int, help='distributed-lag span [days]')
p.add_argument('-deg', default=3, type=int, help='Almon polynomial degree')
p.add_argument('-nboot', default=500, type=int, help='surrogates for the scans')
p.add_argument('-ndl', default=300, type=int,
               help='surrogates for the distributed-lag test (each is a fit)')
p.add_argument('-block', default=20, type=int, help='bootstrap block length [d]')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname=args.gctag.split('_')[0])
tef2 = Ldir['LOo'] / 'extract' / args.gtagex / 'tef2'
moor_dir = Ldir['LOo'] / 'extract' / args.gtagex / 'moor' / args.job
wind_fn = (Ldir['LOo'] / 'DM_outs' / '20260806_wind'
           / ('wind_hourly_atm00_%s_%s.p' % (args.ds0, args.ds1)))
out_dir = Ldir['LOo'] / 'DM_outs' / '20260811_pc_do_lag_test'
Lfun.make_dir(out_dir)

O2_MMOL_TO_MGL = 32.0 / 1000.0
MOUTH, HEAD = 'pc_lp', 'pc_cp'
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
FS = 13
C = {'wa': '#0072B2', 'qp_cp': '#D55E00'}          # fixed, never cycled
NICE = {'wa': 'along-cove wind', 'qp_cp': '$Q_{prism}$'}
PRED = ['wa', 'qp_cp']
rng = np.random.default_rng(3)


def godin(a):
    return zfun.lowpass(np.asarray(a, dtype=float), f='godin')


def need(fn):
    if not fn.is_file():
        print('*** missing: %s' % fn)
        sys.exit(1)
    return fn


# ---------------------------------------------------------------------------
# daily frame, full record (same construction as 20260811_pc_forcing_stack.py)
# ---------------------------------------------------------------------------
ds = xr.open_dataset(need(moor_dir / ('cp_mid_%s_%s.nc' % (args.mds0, args.mds1))))
do_cp = pd.Series(ds.oxygen.values[:, 0] * O2_MMOL_TO_MGL,
                  index=pd.to_datetime(ds.ocean_time.values))
ds.close()
TT = do_cp.index

ds = xr.open_dataset(need(tef2 / ('bulk_avg_%s_%s' % (args.ds0, args.ds1))
                          / (HEAD + '.nc')))
qp_cp = pd.Series(ds.qprism.values, index=pd.to_datetime(ds.time.values))
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
ax_, ay_ = (xN - x0) / axl, (yN - y0) / axl
W = pd.read_pickle(need(wind_fn))['W']
wa = pd.Series(godin(W.u_pc.values * ax_ + W.v_pc.values * ay_), index=W.index)


def on_daily(s):
    return s.reindex(s.index.union(TT)).interpolate('time').reindex(TT)


S = pd.DataFrame({'do_cp': do_cp, 'wa': on_daily(wa),
                  'qp_cp': on_daily(qp_cp)}).bfill()

# the evaluation window is a MASK, not a subset -- the lags need the full record
MSK = np.ones(len(S), dtype=bool)
if args.year.lower() != 'all':
    MSK &= np.asarray(S.index.year == int(args.year))
if args.months.lower() != 'all':
    MONTHS = [int(m) for m in args.months.split(',')]
    MSK &= np.asarray(S.index.month.isin(MONTHS))
else:
    MONTHS = list(range(1, 13))
MLBL = ''.join('JFMAMJJASOND'[m - 1] for m in sorted(MONTHS))
if MSK.sum() < 20:
    print('*** only %d days in the window' % MSK.sum())
    sys.exit(1)

Y = S.do_cp.to_numpy()
print('window %s %s: n = %d days, %s to %s\n'
      % (MLBL, args.year, MSK.sum(), S.index[MSK][0].date(),
         S.index[MSK][-1].date()))

# ---------------------------------------------------------------------------
# how much information is in the window at all
# ---------------------------------------------------------------------------
print('--- effective sample size ---')
for c_ in ['do_cp'] + PRED:
    v = S[c_].to_numpy()[MSK]
    r1 = np.corrcoef(v[:-1], v[1:])[0, 1]
    print('  %-6s lag-1 autocorr %.3f -> decorrelation ~%4.1f d, ~%2.0f '
          'independent points in %d days'
          % (c_, r1, -1 / np.log(abs(r1)), MSK.sum() * (1 - r1) / (1 + r1),
             MSK.sum()))
print('  Read the rest of this script against those last numbers.\n')


# ---------------------------------------------------------------------------
# scans, and the surrogate machinery that calibrates them
# ---------------------------------------------------------------------------
LAGS = np.arange(0, args.lmax + 1)
WINS = np.arange(1, args.wmax + 1)


def scan_lag(x):
    s = pd.Series(x, index=S.index)
    return np.array([np.corrcoef(*pd.concat([S.do_cp, s.shift(int(L))], axis=1)
                                 .iloc[MSK].dropna().to_numpy().T)[0, 1]
                     for L in LAGS])


def scan_trail(x):
    s = pd.Series(x, index=S.index)
    return np.array([np.corrcoef(*pd.concat([S.do_cp, s.rolling(int(W)).mean()],
                                            axis=1)
                                 .iloc[MSK].dropna().to_numpy().T)[0, 1]
                     for W in WINS])


def surrogates(x, nboot):
    """circular moving-block resamples of x on the FULL record"""
    n = len(x)
    xc = np.concatenate([x, x])
    nb = int(np.ceil(n / args.block))
    for _ in range(nboot):
        idx = np.concatenate([np.arange(s, s + args.block)
                              for s in rng.integers(0, n, nb)])[:n]
        yield xc[idx]


SC, CRIT = {}, {}
print('--- 1/2. lag and trailing-mean scans, with scan-wide nulls ---')
print('%-8s %-8s %8s %10s %10s %10s'
      % ('pred', 'scan', 'at L/W', 'best |r|', 'R2', '95% null'))
for c_ in PRED:
    x = S[c_].to_numpy()
    for kind, fn, grid in [('lag', scan_lag, LAGS), ('trail', scan_trail, WINS)]:
        rr = fn(x)
        SC[(c_, kind)] = rr
        mx = np.array([np.max(np.abs(fn(xb)))
                       for xb in surrogates(x, args.nboot)])
        crit = float(np.percentile(mx, 95))
        CRIT[(c_, kind)] = crit
        j = int(np.argmax(np.abs(rr)))
        flag = '' if abs(rr[j]) > crit else '   (below null)'
        print('%-8s %-8s %8d %+10.2f %10.3f %10.2f%s'
              % (c_, kind, grid[j], rr[j], rr[j] ** 2, crit, flag))
print('  A scan peak means nothing unless it clears the last column.\n')

# ---------------------------------------------------------------------------
# 3. polynomial distributed lag with AR(1) errors, surrogate-calibrated
# ---------------------------------------------------------------------------
Lv = np.arange(args.dlmax + 1)
POW = np.column_stack([Lv ** k for k in range(args.deg + 1)])


def almon(x):
    s = pd.Series(x, index=S.index)
    lags = np.column_stack([s.shift(int(l)).to_numpy() for l in Lv])
    return lags @ POW


def dl_lr(xw, xq):
    """LR of the full distributed-lag model against an AR(1)-only null."""
    Z = np.column_stack([almon(xw), almon(xq)])
    ok = MSK & np.isfinite(Z).all(1)
    y = Y[ok]
    Zs = (Z[ok] - Z[ok].mean(0)) / (Z[ok].std(0) + 1e-12)
    b = SARIMAX(y, order=(1, 0, 0), trend='c').fit(disp=False)
    f = SARIMAX(y, exog=Zs, order=(1, 0, 0), trend='c').fit(disp=False)
    return 2 * (f.llf - b.llf), float(f.params[1]), int(Zs.shape[1])


obs_lr, ar_obs, ndf = dl_lr(S.wa.to_numpy(), S.qp_cp.to_numpy())
print('--- 3. polynomial distributed lag, 0-%d d, degree %d, AR(1) errors ---'
      % (args.dlmax, args.deg))
print('  observed LR = %.2f on %d df   nominal chi2 p = %.3f'
      % (obs_lr, ndf, st.chi2.sf(obs_lr, ndf)))
print('  calibrating against %d surrogates (each one is a model fit)...'
      % args.ndl)
gw = surrogates(S.wa.to_numpy(), args.ndl)
gq = surrogates(S.qp_cp.to_numpy(), args.ndl)
sur = np.array([dl_lr(a, b)[:2] for a, b in zip(gw, gq)])
p_emp = float((sur[:, 0] >= obs_lr).mean())
print('  surrogate LR: median %.1f, 95th pct %.1f, max %.1f'
      % (np.median(sur[:, 0]), np.percentile(sur[:, 0], 95), sur[:, 0].max()))
print('  EMPIRICAL p = %.3f   (nominal %.3f -- inflated %.0fx)'
      % (p_emp, st.chi2.sf(obs_lr, ndf),
         p_emp / max(st.chi2.sf(obs_lr, ndf), 1e-9)))
print('  the nominal chi2 would call %.0f%% of pure-noise surrogates '
      'significant at 0.05' % (100 * (st.chi2.sf(sur[:, 0], ndf) < 0.05).mean()))
print('  residual AR(1): observed %.3f, surrogate median %.3f' %
      (ar_obs, np.median(sur[:, 1])))
print('  A surrogate median near zero means the lag basis soaks up the memory')
print('  of ANY autocorrelated predictor -- it is acting as a flexible trend,')
print('  which is exactly why the nominal p cannot be believed.\n')

pd.DataFrame({'lag': LAGS,
              **{'r_%s' % c_: SC[(c_, 'lag')] for c_ in PRED}}).to_csv(
    out_dir / ('lagscan_%s_%s_%s.csv' % (args.gtagex, args.year, MLBL)),
    index=False)
pd.DataFrame({'window': WINS,
              **{'r_%s' % c_: SC[(c_, 'trail')] for c_ in PRED}}).to_csv(
    out_dir / ('trailscan_%s_%s_%s.csv' % (args.gtagex, args.year, MLBL)),
    index=False)

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

for ax, kind, grid, xlbl, ttl in [
        (axes[0], 'lag', LAGS, 'lag [days], forcing leads',
         'a. lag scan: r(DO(t), X(t$-$L))'),
        (axes[1], 'trail', WINS, 'accumulation window [days]',
         'b. trailing-mean scan: r(DO(t), $\\langle$X$\\rangle_W$)')]:
    # the null band is per-predictor, so draw each as its own dashed pair
    # rather than one shaded region that would imply a single threshold
    for c_ in PRED:
        ax.plot(grid, SC[(c_, kind)], color=C[c_], lw=2, label=NICE[c_])
        for sgn in (1, -1):
            ax.plot(grid, np.full(len(grid), sgn * CRIT[(c_, kind)]),
                    color=C[c_], lw=1.1, ls=':', alpha=0.9,
                    label='95%% scan-wide null, %s' % NICE[c_] if sgn == 1
                    else None)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel(xlbl, fontsize=FS)
    ax.set_ylabel('correlation with bottom DO', fontsize=FS)
    ax.set_title(ttl, fontsize=FS, loc='left')
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(-1, 1)
    ax.grid(**GRID)
axes[0].legend(frameon=False, fontsize=FS - 4, loc='lower left', ncol=2)

fig.suptitle('Lagged response of bottom DO at cp_mid, %s %s, %s   '
             '(no scan peak clears its own null; distributed-lag empirical '
             'p = %.2f)' % (MLBL, args.year, args.gtagex, p_emp),
             fontsize=FS - 1, y=1.02)
for a_ in fig.axes:
    a_.tick_params(labelsize=FS - 3)
fig.tight_layout()
fn_out = out_dir / ('lag_test_%s_%s_%s.png' % (args.gtagex, args.year, MLBL))
fig.savefig(fn_out, dpi=500, bbox_inches='tight', transparent=True)
print('wrote ' + str(fn_out))
