"""
Is there seasonal variation in the tides of wb1_t0_xn11abbur00?

"Seasonal variation in the tides" is two different questions and they have
different answers, so both are done here:

  RANGE   does the tidal excursion itself get bigger in some seasons?
          Puget Sound is diurnal-dominant, so the declinational constituents
          (K1, O1, P1) carry much of the range. K1 and P1 are locked to the
          SOLAR declination cycle, which peaks at both solstices. The
          prediction is a SEMIANNUAL signal with maxima near Jun and Dec,
          not an annual one -- and a semiannual signal is exactly what an
          annual-only fit would miss.

  LEVEL   does mean sea level itself have a seasonal cycle? Yes, and for a
          different reason: steric expansion, seasonal winds, and the river
          hydrograph. This shows up in the Godin-filtered ssh, not in the
          range, and peaks in winter here.

Both are fit with a harmonic model (annual + semiannual) and reported with
the variance each term explains. utide gives the constituent amplitudes that
explain WHY the semiannual term exists.

Two years is two samples per calendar month. That is enough to see a
solstitial signal that is large and phase-locked; it is NOT enough to call
any single month anomalous. Reported accordingly.

run 20260806_tidal_seasonality.py
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from lo_tools import Lfun, zfun

Ldir = Lfun.Lstart(gridname='wb1')
gtx = 'wb1_t0_xn11abbur00'
coll = 'wb1_pc1'
SECT = 'pc_lp'
TZ = 'America/Los_Angeles'
out_dir = Ldir['LOo'] / 'DM_outs' / '20260806_tidal_phase'
Lfun.make_dir(out_dir)

# ------------------------------------------------------------------- load ---
fn = sorted((Ldir['LOo'] / 'extract' / gtx / 'tef2').glob(
    'hourly_flux_*_%s.nc' % coll))[0]
d = xr.open_dataset(fn)
ssh = d.ssh.sel(sect=SECT).to_pandas()
d.close()
ssh.index = ssh.index.tz_localize('UTC').tz_convert(TZ)
t0, t1 = ssh.index[0], ssh.index[-1]

# daily range on fixed 24 h local windows, same convention as the calendar
midn = pd.date_range(t0.ceil('D'), t1.floor('D'), freq='D', tz=TZ)
v = ssh.values
pos = ssh.index.searchsorted(midn)
ok = (pos + 24) <= len(v)
midn, pos = midn[ok], pos[ok]
w = np.lib.stride_tricks.sliding_window_view(v, 24)[pos]
rng = pd.Series(w.max(axis=1) - w.min(axis=1), index=midn, name='range_m')

# tidally filtered sea level -> daily mean of the Godin-ish lowpass
lp = pd.Series(zfun.lowpass(ssh.values, f='godin'), index=ssh.index)
msl = lp.resample('D').mean().reindex(rng.index)

print('%s  section %s' % (gtx, SECT))
print('%d days, %s to %s (%s)' % (len(rng), rng.index[0].date(),
                                  rng.index[-1].date(), TZ))


# ------------------------------------------------------- harmonic seasonal ---
def seasonal_fit(y):
    """Least squares mean + annual + semiannual. Returns dict of results."""
    g = y.dropna()
    t = (g.index - g.index[0]).total_seconds().values / 86400.0
    w1 = 2 * np.pi / 365.25
    X = np.column_stack([np.ones_like(t),
                         np.cos(w1 * t), np.sin(w1 * t),
                         np.cos(2 * w1 * t), np.sin(2 * w1 * t)])
    b, *_ = np.linalg.lstsq(X, g.values, rcond=None)
    fit = X @ b
    resid = g.values - fit
    a1 = np.hypot(b[1], b[2])
    a2 = np.hypot(b[3], b[4])
    # day of year of each term's first maximum
    ph1 = (-np.arctan2(b[2], b[1]) / w1) % 365.25
    ph2 = (-np.arctan2(b[4], b[3]) / (2 * w1)) % 182.6
    doy0 = g.index[0].dayofyear
    var = g.values.var()
    # variance explained by each term alone
    v1 = 1 - (g.values - (X[:, [0, 1, 2]] @ b[[0, 1, 2]])).var() / var
    v2 = 1 - (g.values - (X[:, [0, 3, 4]] @ b[[0, 3, 4]])).var() / var
    return dict(mean=b[0], ann_amp=a1, semi_amp=a2,
                ann_peak_doy=(ph1 + doy0) % 365.25,
                semi_peak_doy=(ph2 + doy0) % 182.6,
                var_ann=v1, var_semi=v2,
                var_both=1 - resid.var() / var,
                fit=pd.Series(fit, index=g.index))


def _doy_label(doy):
    return (pd.Timestamp('2025-01-01') + pd.Timedelta(days=float(doy) - 1)
            ).strftime('%d %b')


R = seasonal_fit(rng)
M = seasonal_fit(msl)

print('\n--- TIDAL RANGE seasonality ---')
print('  mean %.3f m,  day-to-day sd %.3f m' % (R['mean'], rng.std()))
print('  annual     amp %.3f m  peak %s   var explained %5.1f%%'
      % (R['ann_amp'], _doy_label(R['ann_peak_doy']), 100 * R['var_ann']))
print('  semiannual amp %.3f m  peaks %s and %s   var explained %5.1f%%'
      % (R['semi_amp'], _doy_label(R['semi_peak_doy']),
         _doy_label(R['semi_peak_doy'] + 182.6), 100 * R['var_semi']))
print('  both terms together explain %.1f%% of daily-range variance'
      % (100 * R['var_both']))

print('\n--- MEAN SEA LEVEL seasonality (Godin filtered) ---')
print('  mean %.3f m,  sd %.3f m' % (M['mean'], msl.std()))
print('  annual     amp %.3f m  peak %s   var explained %5.1f%%'
      % (M['ann_amp'], _doy_label(M['ann_peak_doy']), 100 * M['var_ann']))
print('  semiannual amp %.3f m  peaks %s and %s   var explained %5.1f%%'
      % (M['semi_amp'], _doy_label(M['semi_peak_doy']),
         _doy_label(M['semi_peak_doy'] + 182.6), 100 * M['var_semi']))
print('  both terms together explain %.1f%% of MSL variance'
      % (100 * M['var_both']))

# ------------------------------------------------------ monthly by the eye ---
mon = pd.DataFrame(dict(range_m=rng, msl_m=msl))
mon['year'] = mon.index.year
mon['month'] = mon.index.month
piv = mon.pivot_table(index='month', columns='year', values='range_m',
                      aggfunc='mean')
piv['both'] = mon.groupby('month').range_m.mean()
piv['n_days'] = mon.groupby('month').range_m.count()
piv_m = mon.pivot_table(index='month', values='msl_m', aggfunc='mean')
print('\nmonthly mean daily tidal range (m), and mean sea level (m):')
print('%5s %8s %8s %8s %7s %9s' % ('month', '2024', '2025', 'both', 'days', 'msl'))
for m in range(1, 13):
    print('%5s %8.3f %8.3f %8.3f %7d %9.3f'
          % (pd.Timestamp(2025, m, 1).strftime('%b'), piv.loc[m, 2024],
             piv.loc[m, 2025], piv.loc[m, 'both'], piv.loc[m, 'n_days'],
             piv_m.loc[m, 'msl_m']))
hi = piv['both'].idxmax()
lo = piv['both'].idxmin()
print('  strongest %s (%.3f m), weakest %s (%.3f m), spread %.3f m (%.0f%% of mean)'
      % (pd.Timestamp(2025, hi, 1).strftime('%b'), piv.loc[hi, 'both'],
         pd.Timestamp(2025, lo, 1).strftime('%b'), piv.loc[lo, 'both'],
         piv['both'].max() - piv['both'].min(),
         100 * (piv['both'].max() - piv['both'].min()) / piv['both'].mean()))
# do the two years agree on the shape? if not, it is weather, not season
print('  2024 vs 2025 monthly correlation: r = %.2f'
      % piv[2024].corr(piv[2025]))

# ------------------------------------------------------- why: constituents ---
try:
    import utide
    # utide wants datetimes; handing it float days silently returns NO
    # constituents rather than raising, so keep it as datetimes.
    sol = utide.solve(ssh.index.tz_convert('UTC').tz_localize(None)
                      .to_pydatetime(), ssh.values, lat=48.22, nodal=True,
                      trend=False, method='ols', conf_int='none',
                      verbose=False)
    amp = pd.Series(sol.A, index=sol.name)
    if not len(amp):
        raise RuntimeError('utide returned no constituents')
    print('\nconstituent amplitudes (m), utide, lat 48.22, %d constituents:'
          % len(amp))
    for c in ['M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1', 'SA', 'SSA', 'MF']:
        if c in amp.index:
            print('  %-4s %6.3f' % (c, amp[c]))
    d1 = amp.reindex(['K1', 'O1']).sum()
    d2 = amp.reindex(['M2', 'S2']).sum()
    print('  form factor (K1+O1)/(M2+S2) = %.2f -> %s' % (d1 / d2,
          'mixed, mainly diurnal' if d1 / d2 > 1.5 else
          'mixed, mainly semidiurnal'))
    # K1-P1 and S2-K2 both beat at 182.6 d -- that beat IS the solstitial
    # modulation of the range, and it is why the seasonal signal is
    # semiannual rather than annual.
    for a, b in (('P1', 'K1'), ('K2', 'S2')):
        if a in amp.index and b in amp.index:
            print('  %s/%s = %.2f  (beat period 182.6 d, peaks at solstices)'
                  % (a, b, amp[a] / amp[b]))
except ImportError:
    print('\n(utide not available, skipping constituent decomposition)')

# ----------------------------------------------------------------- figure ---
plt.close('all')
fig, ax = plt.subplot_mosaic([['a', 'a'], ['b', 'c']], figsize=(14, 9),
                             layout='constrained')

A = ax['a']
A.plot(rng.index, rng.values, color='0.8', lw=0.8, label='daily range')
# full window only -- a partial window at the ends drops the mean and looks
# like a real late-December dip
A.plot(rng.index, rng.rolling(29, center=True).mean(),
       color='#0072B2', lw=2, label='29-day mean (removes spring-neap)')
A.plot(R['fit'].index, R['fit'].values, color='#CC0000', lw=1.6,
       label='annual + semiannual fit')
for yr in (2024, 2025):
    for m, lab in ((6, 'solstice'), (12, 'solstice')):
        A.axvline(pd.Timestamp(yr, m, 21, tz=TZ), color='0.6', ls=':', lw=1)
    for m in (3, 9):
        A.axvline(pd.Timestamp(yr, m, 21, tz=TZ), color='0.85', ls='--', lw=1)
A.set_ylabel('daily tidal range (m)')
A.set_title('%s -- seasonality of the tidal range at %s\n'
            'dotted = solstices, dashed = equinoxes' % (gtx, SECT))
A.grid(color='lightgray', ls='--', alpha=0.5)
A.legend(fontsize=8, ncol=3, loc='upper right')

B = ax['b']
bx = [mon.loc[mon.month == m, 'range_m'].values for m in range(1, 13)]
bp = B.boxplot(bx, positions=range(1, 13), widths=0.6, patch_artist=True,
               medianprops=dict(color='k'), flierprops=dict(ms=2, mfc='0.6',
                                                            mec='none'))
for p in bp['boxes']:
    p.set(facecolor='#9ECAE1', edgecolor='0.4')
B.plot(range(1, 13), piv['both'].values, 'o-', color='#CC0000', lw=1.6, ms=5,
       label='monthly mean')
B.set_xticks(range(1, 13))
B.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
B.set_ylabel('daily tidal range (m)')
B.set_title('b. Monthly distribution of daily range\n'
            '(spread within a month is the spring-neap cycle)')
B.grid(color='lightgray', ls='--', alpha=0.5, axis='y')
B.legend(fontsize=8)

C = ax['c']
C.plot(msl.index, msl.values, color='0.75', lw=0.8, label='daily mean, filtered')
C.plot(M['fit'].index, M['fit'].values, color='#009E73', lw=2,
       label='annual + semiannual fit')
C.set_ylabel('mean sea level (m)')
C.set_title('c. Seasonal cycle of mean sea level\n(steric + wind + rivers, '
            'not an astronomical tide)')
C.grid(color='lightgray', ls='--', alpha=0.5)
C.legend(fontsize=8)
for lab in C.get_xticklabels():
    lab.set_rotation(20)

fn_out = out_dir / 'seasonality.png'
fig.savefig(fn_out, dpi=200, bbox_inches='tight')
pd.DataFrame(dict(range_m=rng, msl_m=msl)).to_csv(
    out_dir / 'seasonality_daily.csv',
    index_label='date_local', float_format='%.4f')
print('\nsaved %s' % fn_out)
