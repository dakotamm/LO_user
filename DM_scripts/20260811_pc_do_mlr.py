"""
Multiple linear regression of bottom DO at pc_cp on along-cove wind and Qprism,
July-October, wb1_t0_xn11abbur00.

    do_cp  ~  b0  +  b1 * w_along  +  b2 * qp_cp

do_cp    bottom DO (s_rho index 0) at cp_mid, the midpoint of the Coupeville
         line, from the pc4 mooring job. Extracted with -lt lowpass, so it is
         already one Godin-filtered value per day.
w_along  region-mean wind velocity projected on the pc_lp -> pc_cp axis,
         positive INTO the cove, Godin filtered here. m/s.
qp_cp    Qprism at the pc_cp section, Godin + daily-subsampled upstream by
         bulk_calc_avg.py. m3/s.

All three are built exactly as in 20260811_pc_forcing_stack.py, so the numbers
here annotate that figure rather than describing different quantities.

JULY-OCTOBER. The window is the stratified drawdown-and-reventilation season:
the September DO minimum and the October recovery both sit inside it, and the
winter half of the year -- where DO and wind co-vary purely because both have
an annual cycle -- is excluded. Restricting to it is what makes a plain
regression on levels a reasonable thing to run at all.

TWO PRACTICAL POINTS, not caveats that change the fit:

  Filtering happens on the FULL 2024-2025 record and only then is the window
  taken, so the July edge is not blanked by the Godin half-width.

  July-October is two disjoint ~123-day segments, one per year. The regression
  pools them, which is fine for the fit, and each year is also reported
  separately so a result that only exists in one summer is visible as such.

Alongside the ordinary OLS table the script prints standardized coefficients
(so the two predictors' magnitudes are comparable across their different
units), the variance each predictor explains alone, and VIF. It also prints
Durbin-Watson and HAC standard errors: daily DO is strongly autocorrelated, so
the OLS p-values are optimistic, and the HAC column is there to show by how
much. The coefficients themselves are unaffected.

run 20260811_pc_do_mlr.py                        # Jul-Oct, both years pooled
run 20260811_pc_do_mlr.py -months all            # whole record
run 20260811_pc_do_mlr.py -months 6,7,8,9        # a different window
run 20260811_pc_do_mlr.py -year 2025
"""
import argparse
import sys
import warnings

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr
from cmcrameri import cm as cmc
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

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
p.add_argument('-months', default='7,8,9,10', type=str,
               help="months to fit over, comma-separated, or 'all'")
args = p.parse_args()

Ldir = Lfun.Lstart(gridname=args.gctag.split('_')[0])
tef2 = Ldir['LOo'] / 'extract' / args.gtagex / 'tef2'
moor_dir = Ldir['LOo'] / 'extract' / args.gtagex / 'moor' / args.job
wind_fn = (Ldir['LOo'] / 'DM_outs' / '20260806_wind'
           / ('wind_hourly_atm00_%s_%s.p' % (args.ds0, args.ds1)))
out_dir = Ldir['LOo'] / 'DM_outs' / '20260811_pc_do_mlr'
Lfun.make_dir(out_dir)

O2_MMOL_TO_MGL = 32.0 / 1000.0
MOUTH, HEAD = 'pc_lp', 'pc_cp'
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
FS = 13
# predictor identity, fixed and never cycled: wind keeps this blue and Qprism
# this vermillion in every panel. Okabe-Ito, as in the other pc scripts.
C_WIND = '#0072B2'
C_QP = '#D55E00'
C_OBS = mcolors.to_hex(cmc.lajolla(0.05))    # cp_mid, as in the pc4 map
C_FIT = '#111111'


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
ds = xr.open_dataset(need(moor_dir / ('cp_mid_%s_%s.nc'
                                      % (args.mds0, args.mds1))))
do_cp = pd.Series(ds.oxygen.values[:, 0] * O2_MMOL_TO_MGL,
                  index=pd.to_datetime(ds.ocean_time.values))
h_cp = float(ds.h)
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
ax_, ay_ = (xN - x0) / axl, (yN - y0) / axl          # unit vector, mouth -> head
W = pd.read_pickle(need(wind_fn))['W']
wa = pd.Series(godin(W.u_pc.values * ax_ + W.v_pc.values * ay_), index=W.index)


def on_daily(s):
    """Sample onto TT; the hourly series are on the hour, TT on the half hour."""
    return s.reindex(s.index.union(TT)).interpolate('time').reindex(TT)


S = pd.DataFrame({'do_cp': do_cp, 'wa': on_daily(wa), 'qp_cp': on_daily(qp_cp)})
S = S.bfill()          # the tef2 files start one day after the moorings

# window AFTER filtering, so the July edge keeps its Godin half-width
if args.year.lower() != 'all':
    S = S[S.index.year == int(args.year)]
if args.months.lower() != 'all':
    MONTHS = [int(m) for m in args.months.split(',')]
    S = S[S.index.month.isin(MONTHS)]
else:
    MONTHS = list(range(1, 13))
S = S.dropna()
if len(S) < 10:
    print('*** only %d days in the window' % len(S))
    sys.exit(1)

MLBL = ''.join('JFMAMJJASOND'[m - 1] for m in sorted(MONTHS))
print('along-cove axis %.0f deg true, mouth -> head, %.2f km'
      % (np.rad2deg(np.arctan2(ax_, ay_)) % 360, axl))
print('cp_mid, h = %.1f m. Months %s, %d days, %s to %s\n'
      % (h_cp, MLBL, len(S), S.index[0].date(), S.index[-1].date()))

# ---------------------------------------------------------------------------
# the regression
# ---------------------------------------------------------------------------
PRED = ['wa', 'qp_cp']
NICE = {'wa': 'w_along [m/s]', 'qp_cp': 'qp_cp [m3/s]'}


def fit(df, tag):
    y = df.do_cp.to_numpy()
    X = sm.add_constant(df[PRED].to_numpy())
    m = sm.OLS(y, X).fit()
    mh = m.get_robustcov_results(cov_type='HAC', maxlags=30, use_correction=True)
    print('=== %s: n = %d ===' % (tag, len(df)))
    print('  R2 = %.3f   adj R2 = %.3f   F p = %.2g   RMSE = %.2f mg/L'
          % (m.rsquared, m.rsquared_adj, m.f_pvalue, np.sqrt(m.mse_resid)))
    print('  %-16s %10s %9s %8s %9s %10s'
          % ('term', 'coef', 'SE', 't', 'p', 'p (HAC)'))
    print('  %-16s %10.4f %9.4f %8.2f %9.2g %10s'
          % ('intercept', m.params[0], m.bse[0], m.tvalues[0], m.pvalues[0], '-'))
    for i, c_ in enumerate(PRED):
        print('  %-16s %10.5f %9.5f %8.2f %9.2g %10.2g'
              % (NICE[c_], m.params[i + 1], m.bse[i + 1], m.tvalues[i + 1],
                 m.pvalues[i + 1], mh.pvalues[i + 1]))
    # standardized betas: the two predictors differ by four orders of magnitude
    # in units, so the raw coefficients cannot be compared to each other
    sd_y = df.do_cp.std()
    print('  standardized beta:  ' + '   '.join(
        '%s %+.3f' % (c_, m.params[i + 1] * df[c_].std() / sd_y)
        for i, c_ in enumerate(PRED)))
    print('  R2 alone:           ' + '   '.join(
        '%s %.3f' % (c_, sm.OLS(y, sm.add_constant(df[c_].to_numpy()))
                     .fit().rsquared) for c_ in PRED))
    Xv = df[PRED].to_numpy()
    print('  VIF:                ' + '   '.join(
        '%s %.2f' % (c_, variance_inflation_factor(sm.add_constant(Xv), i + 1))
        for i, c_ in enumerate(PRED)))
    print('  r(wind, Qprism) = %+.2f' % df.wa.corr(df.qp_cp))
    print('  Durbin-Watson = %.2f  (2 = no residual autocorrelation; well below'
          % durbin_watson(m.resid))
    print('    2 means the OLS p-values above are optimistic -- compare the HAC'
          ' column)\n')
    return m


YRS = sorted(S.index.year.unique())
M = fit(S, '%s %s' % (MLBL, 'pooled' if len(YRS) > 1 else YRS[0]))
if len(YRS) > 1:
    for yr in YRS:
        fit(S[S.index.year == yr], '%d only' % yr)

S['fit'] = M.fittedvalues
S['resid'] = M.resid
S.to_csv(out_dir / ('do_mlr_%s_%s_%s.csv' % (args.gtagex, args.year, MLBL)))

print('mean bottom DO %.2f mg/L, std %.2f, range %.2f-%.2f'
      % (S.do_cp.mean(), S.do_cp.std(), S.do_cp.min(), S.do_cp.max()))
print('a 1 m/s increase in along-cove wind (into the cove) goes with %+.2f mg/L;'
      % M.params[1])
print('a 100 m3/s increase in Qprism goes with %+.3f mg/L.' % (100 * M.params[2]))

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
nyr = len(YRS)
fig = plt.figure(figsize=(12.5, 3.6 + 2.9 * nyr))
gs = fig.add_gridspec(nyr + 1, 2, hspace=0.45, wspace=0.24,
                      height_ratios=[1] * nyr + [1.15])

# top: observed and fitted, one row per year (the window is disjoint across
# years, so plotting them on one axis would draw a line through the gap)
for k, yr in enumerate(YRS):
    ax = fig.add_subplot(gs[k, :])
    Y = S[S.index.year == yr]
    ax.plot(Y.index, Y.do_cp, color=C_OBS, lw=2.2, label='observed')
    ax.plot(Y.index, Y.fit, color=C_FIT, lw=1.6, ls='--', label='fitted')
    ax.axhline(2.0, color='#7f2704', lw=0.8, ls=':')
    ax.text(0.997, 2.0, ' 2 mg/L ', transform=ax.get_yaxis_transform(),
            ha='right', va='bottom', fontsize=FS - 4, color='#7f2704')
    ax.set_ylabel('bottom DO\n[mg L$^{-1}$]', fontsize=FS)
    ax.set_title('%s %d: observed vs fitted, $R^2$ = %.2f%s'
                 % (MLBL, yr, M.rsquared, ' (pooled)' if nyr > 1 else ''),
                 fontsize=FS, loc='left')
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.set_xlim(Y.index[0], Y.index[-1])
    if k == 0:
        ax.legend(frameon=False, fontsize=FS - 3, loc='upper right', ncol=2)
    ax.grid(**GRID)

# bottom: partial regression (added-variable) plots -- the slope in each is the
# fitted coefficient, with the OTHER predictor already removed from both axes,
# so these show what each term contributes rather than its raw scatter
for i, (c_, col) in enumerate(zip(PRED, [C_WIND, C_QP])):
    ax = fig.add_subplot(gs[nyr, i])
    other = [q for q in PRED if q != c_]
    Xo = sm.add_constant(S[other].to_numpy())
    ey = sm.OLS(S.do_cp.to_numpy(), Xo).fit().resid
    ex = sm.OLS(S[c_].to_numpy(), Xo).fit().resid
    ax.scatter(ex, ey, s=16, color=col, lw=0, alpha=0.55)
    xx = np.linspace(ex.min(), ex.max(), 2)
    b = np.polyfit(ex, ey, 1)
    ax.plot(xx, np.polyval(b, xx), color='k', lw=2)
    ax.axhline(0, color='#999999', lw=0.7)
    ax.axvline(0, color='#999999', lw=0.7)
    ax.set_xlabel('%s | other' % NICE[c_], fontsize=FS)
    ax.set_ylabel('bottom DO | other [mg L$^{-1}$]' if i == 0 else '',
                  fontsize=FS)
    ax.set_title('%s. partial regression, %s: slope %+.4g, p = %.2g'
                 % ('cd'[i], c_, M.params[i + 1], M.pvalues[i + 1]),
                 fontsize=FS - 1, loc='left')
    ax.grid(**GRID)

for a_ in fig.axes:
    a_.tick_params(labelsize=FS - 3)
fig.suptitle('Bottom DO at cp_mid ~ along-cove wind + $Q_{prism}$, %s %s, %s'
             % (MLBL, '-'.join(str(y) for y in YRS), args.gtagex),
             fontsize=FS + 1, y=0.98)

fn_out = out_dir / ('do_mlr_%s_%s_%s.png' % (args.gtagex, args.year, MLBL))
fig.savefig(fn_out, dpi=500, bbox_inches='tight', transparent=True)
print('\nwrote ' + str(fn_out))
