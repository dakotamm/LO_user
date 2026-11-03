"""
TIMING test for the inner Penn Cove cell: does it FOLLOW the inflow that
passes it (cell lags inflow by a spin-up time, ~1 km / 2-3 cm/s ~ 10 h), or do
both move together because something else (wind) drives both (no lag, and the
relation weakens once wind is held fixed)?

Input: hourly Godin-filtered series from 20261027_pc_lobe_cavity.py
(pc_lobe_cavity_hourly.csv: lobe = mean vorticity within 500 m of the eddy
centre [f]; inflow = westward u on the inflow side of pc_cp [m/s]; bands
z0-4, z4-10, z10-bed, dm), and the hourly cove wind from
20260806_wind_reduce.py (u_pc, v_pc), Godin-filtered the same way.
Wind: along = toward the cove head (axis (-0.9789, -0.2043)), cross = toward
the NORTH shore (right-hand normal; see [[wcross-sign-bug]]).

All series as anomalies from a 31-day running mean (removes the shared
seasonal cycle). The Godin filter is symmetric, so it blurs but does not shift
lags.

  1. lagged correlation corr(lobe(t + lag), inflow(t)), lag -72..+72 h
     (> 0: lobe lags), with a 95% CI on the PEAK LAG from a block bootstrap
     over 30-day blocks
  2. the same after regressing wind out of both (along + cross, at lags
     0, 12, 24, 36, 48 h before), i.e. a partial correlation on wind
  3. event composite: inflow peaks (local maxima of the inflow anomaly
     > 1 sd, >= 72 h apart); mean standardized inflow and lobe from -48 to
     +72 h; lobe peak time relative to the inflow peak, bootstrap over events

Output: LO_output/DM_outs/20261027_pc_lobe_timing/pc_lobe_timing.png, report.txt

  python 20261027_pc_lobe_timing.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun, zfun

Ldir = Lfun.Lstart(gridname='wb1')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261027_pc_lobe_timing'
Lfun.make_dir(out_dir)
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
C_LOBE, C_IN, C_PART = '#0072B2', '#D55E00', '#009E73'
AX, AY = -0.9789, -0.2043
BANDS = ['z0-4', 'z4-10', 'z10-bed', 'dm']
LAGS = np.arange(-72, 73)
WLAGS = [0, 12, 24, 36, 48]
NBOOT = 500
BLOCK = 30 * 24
rng = np.random.default_rng(0)

H = pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20261027_pc_lobe_cavity' / 'pc_lobe_cavity_hourly.csv', index_col=0, parse_dates=True)
H = H.asfreq('h')
W = pd.read_pickle(Ldir['LOo'] / 'DM_outs' / '20260806_wind' / 'wind_hourly_atm00_2024.01.01_2025.12.31.p')['W'][['u_pc', 'v_pc']].asfreq('h')
W['along'] = W.u_pc * AX + W.v_pc * AY
W['cross'] = W.u_pc * AY - W.v_pc * AX
for c in ['along', 'cross']:
    W[c] = zfun.lowpass(W[c].interpolate(limit=6).values.astype(float), f='godin')
H = H.join(W[['along', 'cross']], how='left')
anom = lambda s: s - s.rolling(31 * 24, center=True, min_periods=15 * 24).mean()
A = H.apply(anom)

def lagcorr(y, x, lags):
    """corr(y(t + lag), x(t)) for each lag [h]."""
    return np.array([pd.Series(y).shift(-k).corr(pd.Series(x)) for k in lags])

def lagcorr_np(y, x, lags):
    out = np.full(len(lags), np.nan)
    for n, k in enumerate(lags):
        a = y[k:] if k >= 0 else y[:k]
        b = x[:len(x) - k] if k >= 0 else x[-k:]
        ok = np.isfinite(a) & np.isfinite(b)
        out[n] = np.corrcoef(a[ok], b[ok])[0, 1]
    return out

def boot_peak(y, x, lags):
    """95% CI of the peak lag: resample 30-day blocks of the paired series."""
    n = len(y)
    nb = n // BLOCK
    pk = []
    for _ in range(NBOOT):
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in rng.integers(0, n - BLOCK, nb)])
        # lags are computed within the concatenated record; block edges add a little noise only
        r = lagcorr_np(y[idx], x[idx], lags)
        pk.append(lags[np.nanargmax(r)])
    return np.percentile(pk, [2.5, 97.5])

def wind_resid(s, Aw):
    """Residual of s after OLS on wind along/cross at lags WLAGS h before."""
    X = np.column_stack([Aw[c].shift(k).values for c in ['along', 'cross'] for k in WLAGS] + [np.ones(len(s))])
    ok = np.isfinite(X).all(1) & np.isfinite(s)
    beta = np.linalg.lstsq(X[ok], s[ok], rcond=None)[0]
    r = np.full(len(s), np.nan)
    r[ok] = s[ok] - X[ok] @ beta
    return r, 1 - np.nanvar(r) / np.nanvar(np.where(ok, s, np.nan))

report = ['anomalies from 31-d running mean; lag > 0 = lobe lags inflow; peak-lag CI from %d block bootstraps (30-d blocks)' % NBOOT]
fig, axs = plt.subplots(len(BANDS), 2, figsize=(13, 3.2 * len(BANDS)), layout='constrained')
for n, b in enumerate(BANDS):
    y, x = A['lobe_' + b].values, A['inflow_' + b].values
    r = lagcorr_np(y, x, LAGS)
    k = int(np.nanargmax(r))
    ci = boot_peak(y, x, LAGS)
    yr, fy = wind_resid(y, A)
    xr_, fx = wind_resid(x, A)
    rp = lagcorr_np(yr, xr_, LAGS)
    kp = int(np.nanargmax(rp))
    cip = boot_peak(yr, xr_, LAGS)
    # event composite on inflow peaks
    sd = np.nanstd(x)
    cand = [i for i in range(1, len(x) - 1) if np.isfinite(x[i]) and x[i] > sd and x[i] >= np.nanmax(x[max(0, i - 36):i + 37])]
    ev = []
    for i in cand:
        if not ev or i - ev[-1] >= 72:
            ev.append(i)
    T = np.arange(-48, 73)
    ev = [i for i in ev if i + T[0] >= 0 and i + T[-1] < len(x)]
    zx, zy = x / np.nanstd(x), y / np.nanstd(y)
    CX = np.array([zx[i + T] for i in ev])
    CY = np.array([zy[i + T] for i in ev])
    pk_ev = []
    for _ in range(NBOOT):
        s = rng.integers(0, len(ev), len(ev))
        pk_ev.append(T[np.nanargmax(np.nanmean(CY[s], axis=0))])
    tpk = T[np.nanargmax(np.nanmean(CY, axis=0))]
    cie = np.percentile(pk_ev, [2.5, 97.5])

    ax = axs[n, 0]
    ax.plot(LAGS, r, '-', color='k', lw=2, label='anomalies: peak r %+.2f at %+d h [%+d, %+d]' % (r[k], LAGS[k], ci[0], ci[1]))
    ax.plot(LAGS, rp, '-', color=C_PART, lw=2, label='wind removed: peak r %+.2f at %+d h [%+d, %+d]' % (rp[kp], LAGS[kp], cip[0], cip[1]))
    ax.axvspan(ci[0], ci[1], color='k', alpha=0.08, lw=0)
    ax.axvline(0, color='0.5', lw=0.8)
    ax.axhline(0, color='0.5', lw=0.8)
    ax.set_xlabel('lag [h]  (> 0: lobe lags inflow)')
    ax.set_ylabel('r')
    ax.set_title('%s: lobe vorticity vs inflow at pc_cp' % b, fontsize=10, loc='left')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(**GRID)
    ax = axs[n, 1]
    for C, col, lab in [(CX, C_IN, 'inflow'), (CY, C_LOBE, 'lobe vorticity')]:
        m, se = np.nanmean(C, axis=0), np.nanstd(C, axis=0) / np.sqrt(len(ev))
        ax.fill_between(T, m - se, m + se, color=col, alpha=0.2, lw=0)
        ax.plot(T, m, '-', color=col, lw=2, label=lab)
    ax.axvline(0, color='0.5', lw=0.8)
    ax.axhline(0, color='0.5', lw=0.8)
    ax.set_xlabel('hours from inflow peak')
    ax.set_ylabel('standardized anomaly')
    ax.set_title('%d inflow peaks; lobe peaks at %+d h [%+d, %+d]' % (len(ev), tpk, cie[0], cie[1]), fontsize=10, loc='left')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(**GRID)
    report.append('%-8s r peak %+.2f at %+d h, CI [%+d, %+d] (r at 0 h %+.2f) | wind removed (wind explains %.0f%% of lobe, %.0f%% of inflow var): peak %+.2f at %+d h, CI [%+d, %+d] | events n=%d, lobe peak %+d h, CI [%+d, %+d]'
                  % (b, r[k], LAGS[k], ci[0], ci[1], r[72], 100 * fy, 100 * fx, rp[kp], LAGS[kp], cip[0], cip[1], len(ev), tpk, cie[0], cie[1]))
for k, ax in enumerate(axs.flat):
    ax.text(0.01, 0.97, 'abcdefgh'[k], transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')
fig.suptitle('Does the inner-cove cell lag the inflow? Godin-filtered hourly anomalies (31-d running mean removed), wb1_t0_xn11abbur00', fontsize=10)
fig.savefig(out_dir / 'pc_lobe_timing.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
(out_dir / 'report.txt').write_text('\n'.join(report) + '\n')
print('\n'.join(report))
