"""
Does the along-cove wind move the Penn Cove turning point, and at what lag?

20260807_pc_turning_point.py regressed the turning point on the DAILY wind at
ZERO LAG and got r = +0.20 raw, +0.07 on 30-day anomalies -- which is close to
no answer, because zero lag is the wrong test. A wind stress does not move a
gyre instantly: it spins one up over hours to a day, and the response outlives
the forcing. This script does the test the question deserves, on the hourly
wind field that is already reduced in
LO_output/DM_outs/20260806_wind/wind_hourly_atm00_*.p:

  1. LAGGED CROSS-CORRELATION out to +/- several days, so the response is
     allowed to arrive late. Both series are Godin-filtered first (the turning
     point's tidal band is 86% of its variance and would swamp everything) and
     then reduced to 30-day anomalies, because the along-cove wind and the
     cove's circulation both have strong seasonal cycles and a raw correlation
     is mostly that shared season. Both are reported.
  2. A BINNED RESPONSE at the best lag, so a nonlinear or one-sided response
     (down-cove wind reinforcing the gyre, up-cove wind opposing it) is not
     hidden inside a single correlation coefficient.
  3. EVENT COMPOSITES around the strongest up-cove and down-cove wind events,
     which is where a wind response should be visible if it exists anywhere.
  4. Whether the wind changes the SIZE of the tidal swing, not just the mean
     position -- a different question, and the one that matters if the wind
     modulates the exchange rather than displacing it.

SIGN: along_pc > 0 blows INTO the cove (mouth -> head, up-cove), matching
20260806_wind_characterize.py. The axis unit vector is recovered by least
squares from that script's own daily csv rather than recomputed, so the two
cannot drift apart. x50 is reported as km WEST of the mouth (negative), so a
NEGATIVE correlation means up-cove wind pushes the turning point deeper into
the cove.

p-values use n_eff from the lag-1 autocorrelation of both series (Bretherton
et al. 1999). Daily subtidal values are strongly autocorrelated and the
nominal n would call almost anything significant. With ~730 days and n_eff of
order 200-400, |r| below about 0.12 is not distinguishable from zero.

Runs on the mac, off the CSVs the turning-point script already wrote.
run 20260810_pc_turning_wind.py
"""
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as tdist

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('-tz', default='America/Los_Angeles')
p.add_argument('-maxlag', default=7, type=int, help='days, wind leading')
p.add_argument('-minlag', default=-3, type=int)
args = p.parse_args()
warnings.simplefilter('ignore', RuntimeWarning)

Ldir = Lfun.Lstart(gridname='wb1')
out_dir = Ldir['LOo'] / 'DM_outs' / '20260807_pc_turning'
wind_dir = Ldir['LOo'] / 'DM_outs' / '20260806_wind'

C_UP, C_DOWN = '#0072B2', '#D55E00'
C_X, C_LIMB = '#009E73', '#CC79A7'
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
ROLL = 30                                   # days, the anomaly window


def local(ix, assume_utc=True):
    return (pd.to_datetime(ix, utc=True) if assume_utc
            else pd.to_datetime(ix)).tz_convert(args.tz)


# ------------------------------------------------------------------- load ---
H = pd.read_csv(out_dir / 'turning_point_hourly.csv', index_col=0)
H.index = local(H.index)
print('turning point: %d hourly steps, %s to %s'
      % (len(H), H.index[0], H.index[-1]))

Wd = pd.read_pickle(sorted(wind_dir.glob('wind_hourly_atm00_*.p'))[-1])
W = Wd['W'].copy()
W.index = local(W.index)

# recover the along-cove axis from the daily csv instead of recomputing it:
# along_pc = a*u_pc + b*v_pc, solved by least squares, so this script and
# 20260806_wind_characterize.py cannot disagree about which way is up-cove
Wdaily = pd.read_csv(wind_dir / 'daily_wind.csv')
G = Wdaily[['u_pc', 'v_pc', 'along_pc', 'cross_pc']].dropna()
ab, *_ = np.linalg.lstsq(G[['u_pc', 'v_pc']].values, G.along_pc.values,
                         rcond=None)
resid = np.max(np.abs(G[['u_pc', 'v_pc']].values @ ab - G.along_pc.values))
print('along-cove axis (%.3f, %.3f), reproduces the daily csv to %.2e m/s'
      % (ab[0], ab[1], resid))
W['along'] = W.u_pc * ab[0] + W.v_pc * ab[1]
cd, *_ = np.linalg.lstsq(G[['u_pc', 'v_pc']].values, G.cross_pc.values,
                         rcond=None)
W['cross'] = W.u_pc * cd[0] + W.v_pc * cd[1]

# ------------------------------------------------------- subtidal series ---
# The turning point's tidal band carries 86% of its variance. A wind
# correlation computed on the raw hourly series would be measuring how much
# tide leaks into the wind record, so both sides are Godin-filtered.
QCOLS = sorted([c for c in H.columns if c.startswith('km_q')])
BOTH = H[['km50', 'km90'] + QCOLS].join(
    W[['along', 'cross', 'spd_pc', 'ustar3_pc']], how='inner')
BOTH['limb'] = -H.QN.reindex(BOTH.index)          # + = gyre sense
print('%d hours in common' % len(BOTH))

SUB = pd.DataFrame(index=BOTH.index)
for c in BOTH.columns:
    v = BOTH[c].interpolate(limit=6, limit_area='inside').values.astype(float)
    SUB[c] = zfun.lowpass(v, f='godin')

# the tidal SIZE, as a daily number: how far the point swings each day
swing = (BOTH.km50 - SUB.km50).abs().resample('D').max()

DAY = SUB.resample('D').mean()
DAY['swing'] = swing
DAY = DAY.dropna(how='all')


def anom(s, w=ROLL):
    return s - s.rolling(w, center=True, min_periods=w // 2).mean()


A = DAY.apply(anom)


def eff_corr(x, y):
    """Pearson r with the p-value corrected for serial correlation."""
    g = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(g) < 20:
        return np.nan, np.nan, np.nan, 0
    r = g.x.corr(g.y)
    r1x, r1y = g.x.autocorr(1), g.y.autocorr(1)
    ne = float(np.clip(len(g) * (1 - r1x * r1y) / (1 + r1x * r1y), 3, len(g)))
    ts = r * np.sqrt((ne - 2) / max(1e-12, 1 - r ** 2))
    return r, 2 * (1 - tdist.cdf(abs(ts), ne - 2)), ne, len(g)


def lagged(xname, yname, frame):
    """r(lag) with the wind leading by `lag` days."""
    rows = []
    for L in range(args.minlag, args.maxlag + 1):
        r, pv, ne, n = eff_corr(frame[xname].shift(L), frame[yname])
        rows.append(dict(lag_days=L, r=r, p=pv, n_eff=ne, n=n))
    return pd.DataFrame(rows)


# ------------------------------------------------------------- the tests ---
# x50 and x90 are normalised by the mouth value, so they cannot see the
# profile being scaled up or down as a whole -- which is precisely what a wind
# that strengthens or weakens the gyre does. The km_q* metrics are the
# longitude of a FIXED transport contour and are not scale-invariant, so they
# are the honest position test against a forcing like this.
RESP = ([('km50', 'x50 position'), ('km90', 'x90 position')]
        + [(c, 'x at %s m3/s' % c[4:]) for c in QCOLS]
        + [('limb', 'limb strength'), ('swing', 'tidal swing')])

RES, best = {}, {}
for yn, lab in RESP:
    for xn in ['along', 'cross']:
        for det, fr in [('30 d anomaly', A), ('none', DAY)]:
            L = lagged(xn, yn, fr)
            L['wind'], L['response'], L['detrended'] = xn, yn, det
            RES['%s|%s|%s' % (xn, yn, det)] = L
            if xn == 'along' and det == '30 d anomaly':
                best[yn] = L.loc[L.r.abs().idxmax()]
R = pd.concat(RES.values(), ignore_index=True)
R.to_csv(out_dir / 'wind_lagged_correlation.csv', index=False,
         float_format='%.5f')

print('\nALONG-COVE WIND vs THE TURNING POINT (30-day anomalies)')
print('%-16s %9s %8s %8s %8s   %s' % ('response', 'best lag', 'r', 'p',
                                       'n_eff', 'reading'))
for yn, lab in RESP:
    b = best[yn]
    sig = 'significant' if b.p < 0.05 else 'not distinguishable from zero'
    print('%-16s %+8.0f d %+8.2f %8.1e %8.0f   %s'
          % (lab, b.lag_days, b.r, b.p, b.n_eff, sig))
print('  positive lag = wind leads. x50 is km WEST of the mouth, so r < 0 '
      'means\n  up-cove wind pushes the turning point deeper into the cove.')

# ------------------------------------------------------------- figure ---
fig, axs = plt.subplots(2, 2, figsize=(14.5, 9), layout='constrained')

Ax = axs[0][0]
PANEL = [('km50', C_X, 'x50 position (scale-invariant)'),
         ('limb', C_LIMB, 'limb strength')]
if QCOLS:
    PANEL.insert(1, (QCOLS[0], '#4565e8', 'x at %s m3/s (not scale-invariant)'
                     % QCOLS[0][4:]))
for yn, c, lab in PANEL:
    L = RES['along|%s|30 d anomaly' % yn]
    Ax.plot(L.lag_days, L.r, 'o-', color=c, lw=2, ms=4, label=lab)
    L0 = RES['along|%s|none' % yn]
    Ax.plot(L0.lag_days, L0.r, '--', color=c, lw=1.2, alpha=0.6,
            label='%s, no anomaly' % lab)
# the |r| a series this autocorrelated needs to clear
ne_typ = np.nanmedian(RES['along|km50|30 d anomaly'].n_eff)
rcrit = tdist.ppf(0.975, ne_typ - 2) / np.sqrt(ne_typ - 2 +
                                               tdist.ppf(0.975, ne_typ - 2) ** 2)
Ax.axhspan(-rcrit, rcrit, color='0.5', alpha=0.15, lw=0,
           label='not significant (n_eff = %.0f)' % ne_typ)
Ax.axhline(0, color='0.5', lw=0.8)
Ax.axvline(0, color='0.5', lw=0.8)
Ax.set_xlabel('lag (days, wind leading)')
Ax.set_ylabel('correlation with along-cove wind')
Ax.set_title('does up-cove wind move the turning point?\nGodin-filtered, '
             '30-day anomalies', fontsize=10)
Ax.grid(**GRID)
Ax.legend(fontsize=7)

Bx = axs[0][1]
Lb = int(best['km50'].lag_days)
xw = A.along.shift(Lb)
bins = np.nanpercentile(xw, [0, 10, 25, 50, 75, 90, 100])
bins = np.unique(np.round(bins, 3))
cut = pd.cut(xw, bins)
for yn, c, lab in [('km50', C_X, 'x50 (km west of mouth)')]:
    g = A[yn].groupby(cut, observed=False)
    ctr = [iv.mid for iv in g.mean().index]
    Bx.errorbar(ctr, g.mean().values, yerr=2 * (g.std() / np.sqrt(g.count())),
                fmt='o-', color=c, lw=2, ms=5, capsize=3, label=lab)
Bx2 = Bx.twinx()
g = A['limb'].groupby(cut, observed=False)
Bx2.errorbar([iv.mid for iv in g.mean().index], g.mean().values,
             yerr=2 * (g.std() / np.sqrt(g.count())), fmt='s--', color=C_LIMB,
             lw=1.5, ms=4, capsize=3, alpha=0.8)
Bx2.set_ylabel('limb transport anomaly (m$^3$ s$^{-1}$)', color=C_LIMB)
Bx2.tick_params(axis='y', colors=C_LIMB)
Bx.axhline(0, color='0.5', lw=0.8)
Bx.axvline(0, color='0.5', lw=0.8)
Bx.set_xlabel('along-cove wind anomaly at lag %+d d (m s$^{-1}$, + = up-cove)'
              % Lb)
Bx.set_ylabel('x50 anomaly (km)', color=C_X)
Bx.set_title('the response, binned\nis it linear, or one-sided?', fontsize=10)
Bx.grid(**GRID)

Cx = axs[1][0]
sd = A.along.std()
ev = {}
for nm, sign, c in [('up-cove', 1, C_UP), ('down-cove', -1, C_DOWN)]:
    strong = A.along * sign > 1.5 * sd
    # one date per event: the peak of each run of strong days
    grp = (strong != strong.shift()).cumsum()[strong]
    peaks = [A.along[g.index].abs().idxmax() for _, g in A[strong].groupby(grp)]
    ev[nm] = peaks
    if not peaks:
        continue
    win = np.arange(-3, 6)
    comp = np.full((len(peaks), len(win)), np.nan)
    for i, t0 in enumerate(peaks):
        for j, dd in enumerate(win):
            t = t0 + pd.Timedelta(days=int(dd))
            if t in A.index:
                comp[i, j] = A.km50.get(t, np.nan)
    m = np.nanmean(comp, axis=0)
    se = np.nanstd(comp, axis=0) / np.sqrt(np.sum(np.isfinite(comp), axis=0))
    Cx.plot(win, m, 'o-', color=c, lw=2, ms=4,
            label='%s, %d events' % (nm, len(peaks)))
    Cx.fill_between(win, m - 2 * se, m + 2 * se, color=c, alpha=0.18, lw=0)
Cx.axhline(0, color='0.5', lw=0.8)
Cx.axvline(0, color='0.5', lw=0.8)
Cx.set_xlabel('days from the wind peak')
Cx.set_ylabel('x50 anomaly (km)')
Cx.set_title('composite around the strongest wind events\n(> 1.5 sd of the '
             'along-cove anomaly)', fontsize=10)
Cx.grid(**GRID)
Cx.legend(fontsize=8)

Dx = axs[1][1]
r, pv, ne, n = eff_corr(A.along.shift(Lb), A.km50)
Dx.plot(A.along.shift(Lb), A.km50, '.', ms=3, alpha=0.35, color=C_X)
g = pd.DataFrame({'x': A.along.shift(Lb), 'y': A.km50}).dropna()
xx = np.linspace(g.x.min(), g.x.max(), 10)
slope = np.polyfit(g.x, g.y, 1)[0]
Dx.plot(xx, np.polyval(np.polyfit(g.x, g.y, 1), xx), '-', color='k', lw=2)
Dx.axhline(0, color='0.5', lw=0.8)
Dx.axvline(0, color='0.5', lw=0.8)
Dx.set_xlabel('along-cove wind anomaly at lag %+d d (m s$^{-1}$)' % Lb)
Dx.set_ylabel('x50 anomaly (km)')
Dx.set_title('r = %+.2f, %.3f km per m s$^{-1}$ (p = %.1e, n_eff = %.0f)\n'
             'a 5 m/s wind anomaly moves it %.2f km'
             % (r, slope, pv, ne, abs(5 * slope)), fontsize=10)
Dx.grid(**GRID)

fig.suptitle('Penn Cove turning point vs the along-cove wind -- lagged, '
             'Godin-filtered, 30-day anomalies', fontsize=12)
fn = out_dir / 'fig6_wind_response.png'
fig.savefig(fn, dpi=200, bbox_inches='tight')
print('\nsaved %s' % fn)

# ------------------------------------------------------------- summary ---
S = pd.DataFrame([dict(
    response=yn, best_lag_days=best[yn].lag_days, r=best[yn].r,
    p=best[yn].p, n_eff=best[yn].n_eff,
    r_lag0=RES['along|%s|30 d anomaly' % yn].set_index('lag_days').r.get(0),
    r_raw_lag0=RES['along|%s|none' % yn].set_index('lag_days').r.get(0),
    r_cross_best=RES['along|%s|30 d anomaly' % yn].r.abs().max())
    for yn, _ in RESP])
S['units_per_ms'] = [np.polyfit(*pd.DataFrame(
    {'x': A.along.shift(int(best[yn].lag_days)), 'y': A[yn]}).dropna()
    .T.values, 1)[0] for yn, _ in RESP]
S.to_csv(out_dir / 'wind_response_summary.csv', index=False,
         float_format='%.5f')
print(S.round(3).to_string(index=False))
print('\nup-cove events: %d, down-cove events: %d'
      % (len(ev.get('up-cove', [])), len(ev.get('down-cove', []))))
print('saved %s and %s' % ('wind_lagged_correlation.csv',
                           'wind_response_summary.csv'))
