"""
Qprism, bottom DO and along-cove wind on one time axis, 2024-2025.

Three forcings of the Penn Cove oxygen budget plotted against each other over
the whole run, so the fortnightly and the synoptic bands can be separated by
eye before anything is fitted:

  Qprism     tidal prism transport out of bulk_calc_avg (qabs = |qnet -
             lowpass(qnet)|, Qprism = 1/2 <qabs> Godin filtered, subsampled
             daily). This is the SPRING-NEAP forcing -- fortnightly, almost
             perfectly periodic, and set by the astronomy rather than by
             anything happening in the cove.
  bottom DO  bottom-3 m mean oxygen at the three cross-cove sections, Godin
             lowpassed and daily averaged. Seasonal at first order.
  w_along    along-cove wind and its signed stress, positive blowing from the
             mouth toward the head, same axis as 20260807_pc_alongchannel_wind
             defines it. This is the SYNOPTIC forcing -- events of 2-5 days.

WHY THE ANOMALY PANEL IS THE ONE TO READ

The three series live in different frequency bands and different units, and
the raw panels are dominated by the one thing they share: the season. Bottom
DO swings ~4 mg/L from spring to autumn, which is several times anything a
spring tide or a wind event does, so a correlation on the raw series just
measures how well each forcing happens to line up with summer. Every
comparison below is therefore on 30-day centred rolling anomalies, which is
the same treatment the tidal work already uses -- a raw regression here says
"no effect" for a signal that is plainly there once the season is out.

Panel d puts all three anomalies on one axis as z-scores. That is a display
choice, not a statistical one: the correlations in panel e are computed on
the anomalies in their own units.

LAGS AND SIGNIFICANCE

Panel e is a lagged cross-correlation of each forcing anomaly against the
bottom-DO anomaly, on daily values, with positive lag meaning the forcing
LEADS the DO. Subtidal daily series are heavily autocorrelated, so the
significance envelope uses an autocorrelation-corrected n_eff (lag-1,
Bretherton et al. 1999) rather than the nominal 729 points; at nominal n
essentially any correlation clears p = 0.05 and the panel would be
meaningless.

Both the Qprism series and the DO series are already Godin filtered, so lags
shorter than about a day are not resolvable and a peak at 0 d means
simultaneous within the filter's reach.

WHAT COMES OUT (default settings, 2024-2025)

  Qprism        r = +0.08 at +4 d, p = 0.50, n_eff = 74.  NOTHING.  The
                spring-neap cycle is the cleanest periodic signal in the whole
                figure -- Qprism runs 300 to 925 m3/s at the mouth, a factor
                of 3 -- and the bottom-DO anomaly does not follow it. In the
                zoom the two anomalies run together for a cycle or two and
                then slip out of phase, which is what a null correlation looks
                like when both series are narrowband. Whatever sets bottom
                oxygen here, it is not the fortnightly modulation of the tidal
                prism.
  wind stress   r = +0.19 at +1 d, p = 0.002, n_eff = 253.  Small but real and
                correctly ordered -- the wind leads. Positive means UP-cove
                stress goes with higher bottom DO one day later. n_eff is more
                than three times Qprism's because synoptic wind decorrelates
                in days rather than weeks, so the same |r| is far better
                resolved.
  stratifica-   r = -0.29 at 0 d, p = 0.004, n_eff = 92, and the strongest of
  tion          the three: more top-to-bottom salinity difference, less bottom
                oxygen, simultaneously. Seasonally it is a January-March
                signal (r = -0.58) and absent in summer, when the stratified
                cove is already drawn down.

  So the ranking is stratification > wind >> tide, and the tidal term is not
  merely weak but indistinguishable from zero at every lag out to +/- 21 days.

Runs on the mac from the local bulk_avg + extractions_avg + wind pickle.

run 20260807_pc_qprism_do_wind.py
run 20260807_pc_qprism_do_wind.py --anom_days 45 --maxlag 30
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
from scipy import stats

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('--gtx', default='wb1_t0_xn11abbur00')
p.add_argument('--coll', default='wb1_pc1')
p.add_argument('--ds0', default='2024.01.01')
p.add_argument('--ds1', default='2025.12.31')
p.add_argument('--sects', default='pc_lp,pc_lj,pc_cp', help='mouth first')
p.add_argument('--qsect', default='pc_lp',
               help='section whose Qprism is used in the analysis')
p.add_argument('--hlay', type=float, default=3.0,
               help='thickness (m) of the bottom layer for DO')
p.add_argument('--anom_days', type=float, default=30.0,
               help='window for the rolling mean that defines the anomaly')
p.add_argument('--maxlag', type=int, default=21, help='days, each direction')
p.add_argument('--z0', default='2024.06.01', help='zoom window start')
p.add_argument('--z1', default='2024.09.30', help='zoom window end')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
gctag = 'wb1_' + args.coll.split('_')[-1]
tef2 = Ldir['LOo'] / 'extract' / args.gtx / 'tef2'
in_dir = tef2 / ('extractions_avg_%s_%s' % (args.ds0, args.ds1))
bulk_dir = tef2 / ('bulk_avg_%s_%s' % (args.ds0, args.ds1))
wind_fn = (Ldir['LOo'] / 'DM_outs' / '20260806_wind'
           / ('wind_hourly_atm00_%s_%s.p' % (args.ds0, args.ds1)))
out_dir = Ldir['LOo'] / 'DM_outs' / '20260807_pc_qprism_do_wind'
Lfun.make_dir(out_dir)

SECTS = [s.strip() for s in args.sects.split(',')]
SLAB = {'pc_lp': 'mouth', 'pc_lj': 'mid-cove', 'pc_cp': 'head',
        'sp_mid': 'Saratoga mid', 'skagit_sp': 'Skagit'}
CB = dict(blue='#0072B2', orange='#D55E00', green='#009E73', red='#CC0000',
          purple='#5D3A9B', yellow='#E69F00', pink='#CC79A7', grey='#7f7f7f')
SC = {'pc_lp': CB['blue'], 'pc_lj': CB['green'], 'pc_cp': CB['orange'],
      'sp_mid': CB['pink'], 'skagit_sp': CB['purple']}
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
O2_MMOL_TO_MGL = 32.0 / 1000.0
LOWDO = 5.0                                   # the low-DO line used in wb1


def godin(a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return zfun.lowpass(np.asarray(a, dtype=float), f='godin')


def anom(s, per_day=1):
    """Deviation from a centred rolling mean -- the seasonal cycle removed."""
    w = int(round(args.anom_days * per_day))
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


def rcrit(ne, alpha=0.05):
    """|r| that would be significant at n_eff independent points."""
    if ne < 4:
        return np.nan
    t = stats.t.isf(alpha / 2, ne - 2)
    return t / np.sqrt(ne - 2 + t ** 2)


# ------------------------------------------------------------- geometry ---
# The cove axis is the line joining the mouth section's centroid to the head
# section's, which is the same axis 20260807_pc_alongchannel_wind projects the
# wind onto. Recomputed here rather than hard-coded so the two scripts cannot
# drift apart in the sign of w_along.
dstr = xr.open_dataset(tef2 / ('structure_%s_%s_%s.nc'
                               % (args.ds0, args.ds1, gctag)))
CEN = {sn: (float(np.mean(dstr['%s_lon' % sn].values)),
            float(np.mean(dstr['%s_lat' % sn].values))) for sn in SECTS}
dstr.close()
lat0 = np.mean([c[1] for c in CEN.values()])
COS = np.cos(np.deg2rad(lat0))
x0, y0 = CEN[SECTS[0]][0] * COS * 111.32, CEN[SECTS[0]][1] * 111.32
xN, yN = CEN[SECTS[-1]][0] * COS * 111.32, CEN[SECTS[-1]][1] * 111.32
axl = np.hypot(xN - x0, yN - y0)
ax_, ay_ = (xN - x0) / axl, (yN - y0) / axl        # unit vector, mouth -> head
print('--- cove axis: (%.4f, %.4f), %.0f deg true, mouth -> head, %.2f km ---'
      % (ax_, ay_, np.rad2deg(np.arctan2(ax_, ay_)) % 360, axl))

# ---------------------------------------------------------------- Qprism ---
# Already Godin filtered and subsampled daily by bulk_calc_avg -- there is no
# unfiltered qprism series to work from, and none is wanted: the fortnightly
# modulation is what survives the filter and is exactly the signal here.
QP = {}
for sn in SECTS + ['sp_mid']:
    fn = bulk_dir / (sn + '.nc')
    if not fn.is_file():
        continue
    ds = xr.open_dataset(fn)
    QP[sn] = pd.Series(ds.qprism.values,
                       index=pd.DatetimeIndex(ds.time.values).normalize())
    ds.close()
QP = pd.DataFrame(QP)
print('\n--- Qprism [m3 s-1], %s .. %s ---' % (args.ds0, args.ds1))
print(pd.DataFrame({'mean': QP.mean(), 'min': QP.min(), 'max': QP.max(),
                    'max/min': QP.max() / QP.min(),
                    'sd/mean': QP.std() / QP.mean()}).round(2).to_string())

# ------------------------------------------------------------- bottom DO ---
# Bottom layer means use PARTIAL cells: a sigma cell straddling the hlay level
# contributes only the fraction of itself inside the layer. Otherwise the
# "bottom 3 m" is however much the deepest few sigma cells happen to add up
# to, which moves with the tide and with h across the section.
print('\n--- bottom %.1f m DO from the section extractions ---' % args.hlay)
R = {}
for sn in SECTS:
    ds = xr.open_dataset(in_dir / (sn + '.nc'))
    tt = pd.DatetimeIndex(ds.time.values)
    dd = ds.dd.values[None, None, :]
    DZ = ds.DZ.values
    oxy = ds.oxygen.values * O2_MMOL_TO_MGL            # mmol m-3 -> mg/L
    salt = ds.salt.values
    cum_hi = np.cumsum(DZ, axis=1)
    cum_lo = cum_hi - DZ
    H = cum_hi[:, -1:, :]
    d_hi = H - cum_hi
    w_bot = np.clip(args.hlay - cum_lo, 0, DZ) / DZ
    w_top = np.clip(args.hlay - d_hi, 0, DZ) / DZ
    d = {}
    for nm, w in [('top', w_top), ('bot', w_bot)]:
        A = dd * DZ * w
        As = A.sum(axis=(1, 2))
        d['o_' + nm] = (oxy * A).sum(axis=(1, 2)) / As
        d['s_' + nm] = (salt * A).sum(axis=(1, 2)) / As
    R[sn] = pd.DataFrame(d, index=tt)
    print('  %-6s (%-8s) mean bottom DO %.2f mg/L, min %.2f'
          % (sn, SLAB[sn], R[sn].o_bot.mean(), R[sn].o_bot.min()))
    ds.close()
TT = R[SECTS[0]].index
for sn in SECTS:
    assert R[sn].index.equals(TT), 'section time axes differ'

# --------------------------------------------------------------- the wind ---
C = pd.read_pickle(wind_fn)
W = C['W']
wa = W.u_pc.values * ax_ + W.v_pc.values * ay_          # + = INTO the cove
spd = W.spd_pc.values
# Signed along-cove stress: stress magnitude (quadratic in speed) times the
# fraction of the wind lying on the cove axis, sign kept. Momentum input goes
# as speed squared and the sign is the whole point.
tau = W.tau_pc.values * wa / np.maximum(spd, 1e-6)
Wd = pd.DataFrame({'w_along': wa, 'spd': spd, 'tau_along': tau}, index=W.index)
# section times sit on the half hour, the wind on the hour
Wd = Wd.reindex(Wd.index.union(TT)).interpolate('time').reindex(TT)

# ------------------------------------------------- hourly subtidal -> daily ---
# Godin first, then a daily mean. Doing it the other way round leaves the
# diurnal sea breeze aliased into the wind series, and the tide into DO.
S = pd.DataFrame(index=TT)
for sn in SECTS:
    S['%s_o_bot' % sn] = godin(R[sn].o_bot.values)
    S['%s_dstrat' % sn] = godin((R[sn].s_bot - R[sn].s_top).values)
for c_ in ['w_along', 'spd', 'tau_along']:
    S[c_] = godin(Wd[c_].values)
S['do_cove'] = S[['%s_o_bot' % sn for sn in SECTS]].mean(axis=1)
S['dstrat_cove'] = S[['%s_dstrat' % sn for sn in SECTS]].mean(axis=1)

D = S.resample('1D').mean()
D.index = D.index.normalize()
D['qprism'] = QP[args.qsect].reindex(D.index)
for sn in QP.columns:
    D['qp_' + sn] = QP[sn].reindex(D.index)
D = D.loc[D.qprism.notna() | D.do_cove.notna()]

print('\n--- along-cove wind, subtidal daily (+ = into the cove) ---')
print('  mean %+.2f m/s, sd %.2f, range %+.2f to %+.2f; blows in %.0f%% of days'
      % (D.w_along.mean(), D.w_along.std(), D.w_along.min(), D.w_along.max(),
         100 * (D.w_along > 0).mean()))

# --------------------------------------------------------------- anomalies ---
A = pd.DataFrame({c_: anom(D[c_]) for c_ in D.columns}, index=D.index)

# ------------------------------------------------------- lag correlations ---
# Positive lag = the forcing LEADS the DO.
PAIRS = [('qprism', r'$Q_{prism}$ (%s)' % SLAB.get(args.qsect, args.qsect),
          CB['purple']),
         ('tau_along', 'along-cove stress', CB['red']),
         ('dstrat_cove', 'stratification', CB['grey'])]
lags = np.arange(-args.maxlag, args.maxlag + 1)
LC, best = {}, {}
for k, lab, _c in PAIRS:
    rr, pp, nn = [], [], []
    for L in lags:
        r, pv, ne = neff_r(A[k].shift(L).values, A.do_cove.values)
        rr.append(r)
        pp.append(pv)
        nn.append(ne)
    LC[k] = pd.DataFrame({'lag_d': lags, 'r': rr, 'p': pp, 'n_eff': nn})
    i = int(np.nanargmax(np.abs(rr)))
    best[k] = LC[k].iloc[i]

print('\n--- lagged correlation against the bottom-DO anomaly '
      '(%.0f-day anomalies, daily) ---' % args.anom_days)
print('  positive lag = the forcing leads DO')
for k, lab, _c in PAIRS:
    b, z = best[k], LC[k].set_index('lag_d').loc[0]
    print('  %-22s lag 0: r = %+.2f (p = %.3f, n_eff = %d)   |   best: '
          'r = %+.2f at %+d d (p = %.3f, n_eff = %d)'
          % (lab.replace('$Q_{prism}$', 'Qprism'), z.r, z.p, z.n_eff,
             b.r, b.lag_d, b.p, b.n_eff))

# seasonal split: the two forcings do not act in the same season
print('\n--- lag-0 correlation with bottom DO, by season ---')
srows = []
for nm, mo in [('Jan-Mar', [1, 2, 3]), ('Apr-Jun', [4, 5, 6]),
               ('Jul-Sep', [7, 8, 9]), ('Oct-Dec', [10, 11, 12])]:
    m = A.index.month.isin(mo)
    row = {'season': nm, 'n_days': int(m.sum())}
    for k, lab, _c in PAIRS:
        r, pv, ne = neff_r(A[k].values[m], A.do_cove.values[m])
        row['r_' + k] = r
        row['p_' + k] = pv
        row['neff_' + k] = ne
    srows.append(row)
SEAS = pd.DataFrame(srows)
print(SEAS.round(3).to_string(index=False))

# ------------------------------------------------------------------ figure ---
plt.close('all')
fig = plt.figure(figsize=(14, 15), layout='constrained')
gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1.1, 1.2])
axs = [fig.add_subplot(gs[0])]
# the four time-series panels share x so the anomaly panel cannot drift off
# the raw ones -- its NaN margins would otherwise set it a few weeks short
axs += [fig.add_subplot(gs[i], sharex=axs[0]) for i in range(1, 4)]
axs += [fig.add_subplot(gs[4])]

# ---- a: Qprism at every section
ax = axs[0]
for sn in QP.columns:
    lw = 2.0 if sn == args.qsect else 1.0
    al = 1.0 if sn == args.qsect else 0.55
    ax.plot(D.index, D['qp_' + sn], lw=lw, alpha=al, color=SC[sn],
            label='%s (%s)' % (sn, SLAB[sn]))
ax.set_yscale('log')
ax.set_ylabel(r'$Q_{prism}$  [m$^3$ s$^{-1}$]')
ax.legend(fontsize=8, ncol=4, loc='lower left')
ax.set_title('a   tidal prism transport -- Godin filtered, daily subsampled; '
             'the fortnightly spring-neap cycle is what survives the filter',
             fontsize=10, loc='left')

# ---- b: bottom DO
ax = axs[1]
for sn in SECTS:
    ax.plot(D.index, D['%s_o_bot' % sn], lw=1.2, alpha=0.8, color=SC[sn],
            label='%s (%s)' % (sn, SLAB[sn]))
ax.plot(D.index, D.do_cove, lw=2.2, color='k', label='cove mean')
ax.axhline(LOWDO, color=CB['red'], ls='--', lw=1.2)
ax.text(D.index[5], LOWDO, ' low-DO line, %.0f mg L$^{-1}$' % LOWDO,
        color=CB['red'], fontsize=8, va='bottom')
ax.set_ylabel('bottom %.0f m DO\n[mg L$^{-1}$]' % args.hlay)
ax.legend(fontsize=8, ncol=4, loc='lower left')
ax.set_title('b   bottom-layer oxygen -- seasonal at first order, which is '
             'why every comparison below is on anomalies', fontsize=10,
             loc='left')

# ---- c: along-cove wind
ax = axs[2]
ax.fill_between(D.index, 0, D.tau_along, where=D.tau_along >= 0,
                color=CB['red'], alpha=0.55, lw=0, label='up-cove (toward head)')
ax.fill_between(D.index, 0, D.tau_along, where=D.tau_along < 0,
                color=CB['blue'], alpha=0.55, lw=0, label='down-cove (toward mouth)')
ax.axhline(0, color='0.4', lw=0.8)
ax.set_ylabel('along-cove stress\n[Pa]')
ax.legend(fontsize=8, ncol=2, loc='lower left')
axb = ax.twinx()
axb.plot(D.index, D.w_along, lw=0.8, color='k', alpha=0.5)
axb.set_ylabel(r'$w_{along}$ [m s$^{-1}$]', fontsize=9)
ax.set_title('c   along-cove wind stress, signed (+ = blowing from the mouth '
             'toward the head); black = along-cove wind speed', fontsize=10,
             loc='left')

# ---- d: the three anomalies as z-scores on one axis
ax = axs[3]
for k, lab, c_ in [('qprism', r'$Q_{prism}$', CB['purple']),
                   ('do_cove', 'bottom DO', 'k'),
                   ('tau_along', 'along-cove stress', CB['red'])]:
    z = A[k] / A[k].std()
    ax.plot(D.index, z, lw=1.6 if k == 'do_cove' else 1.1,
            alpha=1.0 if k == 'do_cove' else 0.75, color=c_, label=lab)
ax.axhline(0, color='0.4', lw=0.8)
ax.set_ylabel('%.0f-day anomaly\n[standard deviations]' % args.anom_days)
ax.legend(fontsize=9, ncol=3, loc='lower left')
ax.set_title('d   the same three series as %.0f-day rolling anomalies, scaled '
             'to unit variance -- the season removed, the fortnightly and '
             'synoptic bands left' % args.anom_days, fontsize=10, loc='left')

for ax in axs[:4]:
    ax.grid(**GRID)
    ax.margins(x=0.005)
    ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_minor_locator(MonthLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))

# ---- e: lagged cross-correlations, with an autocorrelation-aware envelope
ax = axs[4]
for k, lab, c_ in PAIRS:
    L = LC[k]
    ax.plot(L.lag_d, L.r, lw=2, color=c_, label=lab)
    b = best[k]
    ax.plot(b.lag_d, b.r, 'o', ms=7, mfc='none', mec=c_, mew=2)
    ax.annotate('%+.2f at %+d d (p=%.3f)' % (b.r, b.lag_d, b.p),
                (b.lag_d, b.r), textcoords='offset points',
                xytext=(8, 6 if b.r > 0 else 8), fontsize=8, color=c_,
                ha='left')
# One significance line PER FORCING, not one band for all three. The three
# series have very different memory -- the wind decorrelates in days, Qprism
# and stratification in weeks -- so n_eff ranges over a factor of three and a
# single envelope would call the wind insignificant and the tide significant
# at the same |r|.
for k, lab, c_ in PAIRS:
    ne = int(LC[k].n_eff.median())
    rc = rcrit(ne)
    for s in (-1, 1):
        ax.axhline(s * rc, color=c_, ls=':', lw=1.2, alpha=0.8,
                   label=(r'p = 0.05 for %s (n$_{eff}$=%d)' % (lab, ne))
                   if s == 1 else None)
ax.axhline(0, color='0.4', lw=0.8)
ax.axvline(0, color='0.4', lw=0.8)
ax.set_xlabel('lag [days] -- positive means the forcing LEADS bottom DO')
ax.set_ylabel('correlation with the\nbottom-DO anomaly')
ax.grid(**GRID)
ax.margins(x=0.01, y=0.12)
ax.legend(fontsize=8, loc='lower left', ncol=2, framealpha=0.9)
ax.set_title('e   lagged correlation of each forcing anomaly against the '
             'bottom-DO anomaly; n$_{eff}$ is autocorrelation-corrected, not '
             'the %d nominal days' % len(D), fontsize=10, loc='left')

fig.suptitle('%s -- Penn Cove: tidal prism, bottom oxygen and along-cove wind, '
             '%s to %s' % (args.gtx, args.ds0, args.ds1), fontsize=13)
fn = out_dir / 'qprism_do_wind_series.png'
fig.savefig(fn, dpi=200, bbox_inches='tight')
print('\nsaved %s' % fn)

# ------------------------------------------------------------------- zoom ---
# One summer window at readable scale: at two years across a page the
# fortnightly cycle is a smear, and whether DO tracks it is exactly the thing
# being asked.
z = D.loc[args.z0.replace('.', '-'):args.z1.replace('.', '-')]
za = A.loc[z.index]
fig, axs = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                        layout='constrained')

ax = axs[0]
ax.plot(z.index, z.qprism, '-o', ms=3, lw=1.5, color=CB['purple'])
ax.set_ylabel(r'$Q_{prism}$ at %s' % args.qsect + '\n[m$^3$ s$^{-1}$]',
              color=CB['purple'])
ax.tick_params(axis='y', colors=CB['purple'])
axb = ax.twinx()
axb.plot(z.index, z.do_cove, lw=2, color='k')
axb.set_ylabel('bottom DO, cove mean\n[mg L$^{-1}$]')
ax.set_title('a   raw subtidal series -- the season dominates and hides '
             'whatever the spring-neap cycle is doing', fontsize=10, loc='left')

ax = axs[1]
ax.plot(z.index, za.qprism / A.qprism.std(), lw=1.8, color=CB['purple'],
        label=r'$Q_{prism}$')
ax.plot(z.index, za.do_cove / A.do_cove.std(), lw=2, color='k',
        label='bottom DO')
ax.axhline(0, color='0.4', lw=0.8)
ax.set_ylabel('%.0f-day anomaly\n[std devs]' % args.anom_days)
ax.legend(fontsize=9, ncol=2)
ax.set_title('b   the same window as anomalies -- Qprism is cleanly '
             'fortnightly, the DO anomaly is not phase-locked to it: they '
             'run together for a cycle or two and then slip', fontsize=10,
             loc='left')

ax = axs[2]
ax.fill_between(z.index, 0, z.tau_along, where=z.tau_along >= 0,
                color=CB['red'], alpha=0.55, lw=0)
ax.fill_between(z.index, 0, z.tau_along, where=z.tau_along < 0,
                color=CB['blue'], alpha=0.55, lw=0)
ax.axhline(0, color='0.4', lw=0.8)
ax.set_ylabel('along-cove stress\n[Pa]')
ax.set_title('c   along-cove wind stress (red = up-cove, blue = down-cove)',
             fontsize=10, loc='left')

for ax in axs:
    ax.grid(**GRID)
    ax.margins(x=0.005)
    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))

fig.suptitle('%s -- zoom, %s to %s' % (args.gtx, args.z0, args.z1),
             fontsize=13)
fn = out_dir / 'qprism_do_wind_zoom.png'
fig.savefig(fn, dpi=200, bbox_inches='tight')
print('saved %s' % fn)

# ------------------------------------------------------------------ tables ---
D.to_csv(out_dir / 'qprism_do_wind_daily.csv', float_format='%.5f')
pd.concat([LC[k].assign(forcing=k) for k, _l, _c in PAIRS]).to_csv(
    out_dir / 'lag_correlations.csv', index=False, float_format='%.5f')
SEAS.to_csv(out_dir / 'seasonal_correlations.csv', index=False,
            float_format='%.4f')
print('saved %s, lag_correlations.csv, seasonal_correlations.csv'
      % (out_dir / 'qprism_do_wind_daily.csv'))
