"""
How does the TWO-LAYER part of the tidal flow at the Penn Cove mouth vary with
spring/neap, tropic/equatorial, season (stratification) and the daily wind?

From the pc_cove box, mouth section (last rho column west of pc_lp), hourly:
  u_out   along-cove velocity (+ = OUT), faces averaged to rho
  tidal   u_out minus its Godin low-pass (the tidal band, plus anything else
          faster than ~2 d, e.g. the sea-breeze current)
  BT(t)   whole-section mean of the tidal velocity (dz-weighted): the part that
          moves as one slab, i.e. filling and draining
  BC(t)   rms over the section of (tidal velocity - its column depth-mean):
          the two-layer / sheared part
Daily (UTC day): BT_d = rms of BT, BC_d = rms of BC, plus
  strat   section-mean bottom-minus-top salinity (daily mean)
  wind    daily range of the along-cove wind (sea-breeze strength proxy)

Spring/neap and tropic/equatorial bins: the band-fit tercile flags
(20261027_tidal_ellipses_windowed.py). Fortnightly comparisons use anomalies
from a 31-day running mean ([[analyze-tides-on-anomalies]]). A multiple
regression on daily anomalies gives the share of BC set by the tide, the
stratification and the wind.

Bottom-water focus: daily tidal EXCURSION (range of the time-integrated tidal
velocity within each UTC day) of the bottom 3 m, the top 3 m and the whole
section (slab), section means.

Output: LO_output/DM_outs/20261027_pc_twolayer_tides/pc_twolayer_tides.png
        (bottom-water excursion by cycle, month and stratification),
        report.txt (incl. the two-layer BT/BC numbers), daily.csv

  python 20261027_pc_twolayer_tides.py
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun, zfun

GTX, JOB, DS0, DS1 = 'wb1_t0_xn11abbur00', 'pc_cove', '2024.01.01', '2025.12.31'
Ldir = Lfun.Lstart(gridname='wb1')
box_fn = Ldir['LOo'] / 'extract' / GTX / 'box' / ('%s_%s_%s.nc' % (JOB, DS0, DS1))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261027_pc_twolayer_tides'
Lfun.make_dir(out_dir)
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
OUT = np.array([0.9789, 0.2043])
AX, AY = -0.9789, -0.2043
C_BT, C_BC = '#0072B2', '#D55E00'

ds = xr.open_dataset(box_fn)
tt = pd.DatetimeIndex(ds.ocean_time.values)
lon = ds.lon_rho.values
mask = ds.mask_rho.values == 1
LON_LP = pd.read_pickle(Ldir['LOo'] / 'section_lines' / 'pc_lp.p').x.mean()
i = int(np.flatnonzero(lon[0] < LON_LP)[-1])
jw = np.array([j for j in np.flatnonzero(mask[:, i]) if 0 < j < mask.shape[0] - 1])
u = np.nan_to_num(ds.u[:, :, :, i - 1:i + 1].values).mean(axis=-1)[:, :, jw] # (T, NZ, nj)
v = np.nan_to_num(ds.v[:, :, :, i].values)
v = 0.5 * (v[:, :, jw - 1] + v[:, :, jw])
zw = ds.z_w[:, :, :, i].values[:, :, jw] # (T, NZ+1, nj)
salt = ds.salt[:, :, :, i].values[:, :, jw]
ds.close()
uo = u * OUT[0] + v * OUT[1]
dz = np.diff(zw, axis=1)
NT = len(tt)
print('mouth column %d, %d cells, %d hours' % (i, len(jw), NT))

ulp = zfun.lowpass(uo.reshape(NT, -1).astype(float), f='godin').reshape(uo.shape)
ut = uo - ulp # tidal band (and other fast motions)
col_mean = (ut * dz).sum(1) / dz.sum(1) # (T, nj) depth mean per column
area = dz.sum(1) # (T, nj) column thickness (unit width)
BT = (col_mean * area).sum(1) / area.sum(1) # section-mean tidal velocity
BC = np.sqrt(((ut - col_mean[:, None, :]) ** 2 * dz).sum((1, 2)) / dz.sum((1, 2)))
S = (salt[:, 0, :] - salt[:, -1, :]).mean(1) # bottom minus top, section mean
zmid = 0.5 * (zw[:, :-1] + zw[:, 1:])
def layer_mean(sel):
    w = np.where(sel, dz, 0.0)
    return ((ut * w).sum((1, 2)) / w.sum((1, 2)))
BOT = layer_mean(zmid - zw[:, :1] <= 3.0) # bottom 3 m, section mean
SUR = layer_mean(zw[:, -1:] - zmid <= 3.0) # top 3 m
H = pd.DataFrame({'BT': BT, 'BC': BC, 'BOT': BOT, 'SUR': SUR, 'strat': S}, index=tt)
H = H[np.isfinite(ulp[:, 0, 0])]

W = pd.read_pickle(Ldir['LOo'] / 'DM_outs' / '20260806_wind' / 'wind_hourly_atm00_2024.01.01_2025.12.31.p')['W']
wa = (W.u_pc * AX + W.v_pc * AY)
wa = wa[wa.index.hour != 0] # drop the 00 UTC forcing seam hour from the daily range
D = pd.DataFrame({'BT': 100 * np.sqrt((H.BT ** 2).resample('D').mean()), 'BC': 100 * np.sqrt((H.BC ** 2).resample('D').mean()),
                  'strat': H.strat.resample('D').mean(), 'wind': wa.resample('D').max() - wa.resample('D').min()})
for c in ['BOT', 'SUR', 'BT']:
    D['X' + c] = H[c].resample('D').apply(lambda x: np.ptp(np.cumsum(x.values) * 3600) / 1000 if len(x) >= 20 else np.nan) # km
D = D[H.resample('D').size().reindex(D.index).fillna(0) >= 20].dropna()
BF = xr.open_dataset(Ldir['LOo'] / 'extract' / GTX / 'tidal_ellipses' / ('%s_%s_%s_bands.nc' % (JOB, DS0, DS1)))
F = pd.DataFrame({nm: BF['is_' + nm].values for nm in ['spring', 'neap', 'tropic', 'equatorial']}, index=pd.DatetimeIndex(BF.time.values).normalize())
F['zD2'], F['zD1'] = BF.zidx_D2.values, BF.zidx_D1.values
D = D.join(F, how='inner')
D.to_csv(out_dir / 'daily.csv', float_format='%.4f')
VAR = ['BT', 'BC', 'XBOT', 'XSUR', 'XBT', 'strat', 'wind', 'zD2', 'zD1']
A = D[VAR] - D[VAR].rolling(31, center=True, min_periods=15).mean()

rep = ['mouth section, tidal band = hourly minus Godin; daily rms; %d days' % len(D)]
rep.append('mean BT %.2f cm/s, mean BC %.2f cm/s (BC/BT %.2f)' % (D.BT.mean(), D.BC.mean(), D.BC.mean() / D.BT.mean()))
rep.append('daily tidal excursion (range of time-integrated velocity): bottom 3 m %.2f km, top 3 m %.2f km, section mean %.2f km'
           % (D.XBOT.mean(), D.XSUR.mean(), D.XBT.mean()))
bins = {}
for hi, lo in [('spring', 'neap'), ('tropic', 'equatorial')]:
    rep.append('%s/%s excursion: bottom %.2f -> %.2f km (x%.2f), surface %.2f -> %.2f (x%.2f), section mean %.2f -> %.2f (x%.2f)'
               % (hi, lo, D[D[lo] == 1].XBOT.mean(), D[D[hi] == 1].XBOT.mean(), D[D[hi] == 1].XBOT.mean() / D[D[lo] == 1].XBOT.mean(),
                  D[D[lo] == 1].XSUR.mean(), D[D[hi] == 1].XSUR.mean(), D[D[hi] == 1].XSUR.mean() / D[D[lo] == 1].XSUR.mean(),
                  D[D[lo] == 1].XBT.mean(), D[D[hi] == 1].XBT.mean(), D[D[hi] == 1].XBT.mean() / D[D[lo] == 1].XBT.mean()))
    for q in ['BT', 'BC', 'XBOT', 'XSUR', 'XBT']:
        bins[(hi, q)] = D[D[hi] == 1][q].mean()
        bins[(lo, q)] = D[D[lo] == 1][q].mean()
    rep.append('%s/%s: BT %.2f -> %.2f (x%.2f), BC %.2f -> %.2f (x%.2f)' % (hi, lo, bins[(lo, 'BT')], bins[(hi, 'BT')], bins[(hi, 'BT')] / bins[(lo, 'BT')],
                                                                           bins[(lo, 'BC')], bins[(hi, 'BC')], bins[(hi, 'BC')] / bins[(lo, 'BC')]))
for a, b in [('BC', 'BT'), ('BC', 'strat'), ('BC', 'wind'), ('XBOT', 'XBT'), ('XBOT', 'strat'), ('XBOT', 'wind'), ('XSUR', 'wind')]:
    rep.append('daily r(%s, %s): raw %+.2f, 31-d anomalies %+.2f' % (a, b, D[a].corr(D[b]), A[a].corr(A[b])))
# multiple regression on standardized daily anomalies
Z = A[['BC', 'BT', 'strat', 'wind']].dropna()
Z = (Z - Z.mean()) / Z.std()
X = np.column_stack([Z.BT, Z.strat, Z.wind, np.ones(len(Z))])
beta, *_ = np.linalg.lstsq(X, Z.BC.values, rcond=None)
r2 = 1 - np.var(Z.BC.values - X @ beta) / np.var(Z.BC.values)
rep.append('BC anomaly ~ BT + strat + wind (standardized): beta BT %+.2f, strat %+.2f, wind %+.2f, R2 %.2f' % (beta[0], beta[1], beta[2], r2))
Mo = D[['BT', 'BC', 'XBOT', 'XSUR', 'XBT', 'strat', 'wind']].resample('MS').mean()
rep.append('monthly r(BC, strat) %+.2f, r(BC, BT) %+.2f, r(BC, wind) %+.2f' % (Mo.BC.corr(Mo.strat), Mo.BC.corr(Mo.BT), Mo.BC.corr(Mo.wind)))
rep.append('monthly r(bottom excursion, strat) %+.2f, r(bottom excursion, section-mean excursion) %+.2f, r(bottom excursion, wind) %+.2f'
           % (Mo.XBOT.corr(Mo.strat), Mo.XBOT.corr(Mo.XBT), Mo.XBOT.corr(Mo.wind)))
rep.append('monthly bottom excursion range %.2f-%.2f km (min %s, max %s)' % (Mo.XBOT.min(), Mo.XBOT.max(), Mo.XBOT.idxmin().strftime('%Y-%m'), Mo.XBOT.idxmax().strftime('%Y-%m')))

C_SUR, C_BOT, C_SL = '#56B4E9', '#D55E00', '0.45'
fig, axd = plt.subplot_mosaic([['a', 'b', 'b'], ['c', 'd', 'd']], layout='constrained', figsize=(15, 9.5))
ax = axd['a']
names = ['neap', 'spring', 'equatorial', 'tropic']
x = np.arange(4)
for q, off, col, lab in [('XSUR', -0.27, C_SUR, 'top 3 m'), ('XBOT', 0.0, C_BOT, 'bottom 3 m'), ('XBT', 0.27, C_SL, 'whole section (slab)')]:
    vals = [bins[(n, q)] for n in names]
    ax.bar(x + off, vals, 0.26, color=col, label=lab)
    for xi, vv in zip(x, vals):
        ax.text(xi + off, vv + 0.05, '%.1f' % vv, ha='center', fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel('daily tidal excursion at the mouth [km]')
ax.set_title('bottom water sloshes ~1.8 km a day,\nthe same at spring and neap', fontsize=10, loc='left')
ax.set_ylim(0, 5.2)
ax.legend(fontsize=8, loc='upper center', ncol=3)
ax.grid(axis='y', **GRID)

ax = axd['b']
for q, col, lab in [('XSUR', C_SUR, 'top 3 m'), ('XBOT', C_BOT, 'bottom 3 m'), ('XBT', C_SL, 'whole section (slab)')]:
    ax.plot(Mo.index + pd.Timedelta(days=14), Mo[q], '-o', ms=4, lw=2, color=col, label=lab)
ax.set_ylabel('monthly mean daily excursion [km]')
ax.set_title('by month: the layers move far more than the slab, most in winter (Dec 2025 = freshwater flood)', fontsize=10, loc='left')
ax.legend(fontsize=8, ncol=3, loc='upper center')
ax.grid(**GRID)

ax = axd['c']
ax.scatter(Mo.strat, Mo.XBOT, s=30, color=C_BOT, zorder=3)
for t, r_ in Mo.iterrows():
    if t.month in (12, 9, 10):
        ax.annotate(t.strftime('%b %y'), (r_.strat, r_.XBOT), xytext=(3, 3), textcoords='offset points', fontsize=7)
ax.set_xlabel('stratification at the mouth, bottom - top salinity [g/kg] (monthly)')
ax.set_ylabel('bottom-water daily excursion [km] (monthly)')
ax.set_title('set by stratification: r = %+.2f\n(whole-section slab excursion: r = %+.2f)' % (Mo.XBOT.corr(Mo.strat), Mo.XBOT.corr(Mo.XBT)), fontsize=10, loc='left')
ax.grid(**GRID)

ax = axd['d']
ax.plot(Mo.index + pd.Timedelta(days=14), Mo.strat, '-o', ms=4, lw=2, color='#009E73')
ax.set_ylabel('bottom - top salinity [g/kg]')
ax.set_title('stratification at the mouth, monthly mean', fontsize=10, loc='left')
ax.grid(**GRID)
for k in 'abcd':
    axd[k].text(0.015, 0.97, k, transform=axd[k].transAxes, fontsize=13, fontweight='bold', va='top', ha='left')
fig.suptitle('How far bottom water at the Penn Cove mouth moves with the tide (hourly minus Godin, range of the integrated velocity per day), %s 2024-2025' % GTX, fontsize=11)
fig.savefig(out_dir / 'pc_twolayer_tides.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
(out_dir / 'report.txt').write_text('\n'.join(rep) + '\n')
print('\n'.join(rep))
