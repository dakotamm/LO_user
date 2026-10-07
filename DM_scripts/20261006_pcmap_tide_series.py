"""
Time series of the two tidal descriptors used to class the pcmap releases,
through the year:

  top     F = A_diurnal / A_semidiurnal at pc_lp: least-squares fit of a mean
          plus one diurnal (24.84 h) and one semidiurnal (12.42 h) harmonic to
          ssh (tef2 hourly_flux) over a -win_days window, stepped every 6 h and
          plotted at the window CENTRE. Low F = two near-equal tides a day,
          high F = one dominant tide a day.
  bottom  qprism at pc_lp (tef2 bulk_avg, Godin-filtered, daily): the
          spring-neap signal.

Shading = the year-wide tercile edges of the release classes in
20261006_pcmap_retention_bulk.py (F: semidiurnal / mid / diurnal; qprism:
neap / mid / spring), taken from each release's first -win_days. For each
lunar day in -mark (default the 7/10 and 7/17 spaghetti pairs) the window the
classes are computed over -- from the E release to win_days after the F
release -- is shaded and labelled, so the curve inside the band is what the
releases saw. The last 2 days of qprism are dropped (Godin filter edge).

A second figure shows the fit itself for each -mark lunar day, over the window
of its first release: hourly ssh at pc_lp, the fitted mean + diurnal +
semidiurnal curve, the two harmonics separately (about the mean), and the
residual, with F = A_diurnal / A_semidiurnal and the fit R^2.

Output: LO_output/DM_outs/20261006_pcmap_tide_series/
  pcmap_tide_series_<year>.png, pcmap_tide_fit_<year>.png

run 20261006_pcmap_tide_series.py
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-year', type=int, default=2025)
p.add_argument('-win_days', type=float, default=3.0)
p.add_argument('-mark', default='2025.07.10,2025.07.17', help='lunar days of releases to mark')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
tef2 = Ldir['LOo'] / 'extract' / args.gtx / 'tef2'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261006_pcmap_tide_series'
Lfun.make_dir(out_dir)
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
yr0, yr1 = pd.Timestamp('%d-01-01' % args.year), pd.Timestamp('%d-01-01' % (args.year + 1))

hf = xr.open_dataset(tef2 / 'hourly_flux_2024.01.01_2025.12.31_wb1_pc1.nc')
th = pd.to_datetime(hf.time.values)
ssh = hf.ssh.sel(sect='pc_lp').values
hf.close()
dq = xr.open_dataset(tef2 / 'bulk_avg_2024.01.01_2025.12.31' / 'pc_lp.nc')
tq = pd.to_datetime(dq.time.values)
qp = dq.qprism.values
dq.close()
WD, WS = 2 * np.pi / 24.84, 2 * np.pi / 12.42
win_h = args.win_days * 24


def f_ratio(t_start):
    m = (th >= t_start) & (th < t_start + pd.Timedelta(hours=win_h))
    tt = (th[m] - t_start) / pd.Timedelta(hours=1)
    y = ssh[m]
    ok = np.isfinite(y)
    if ok.sum() < 0.8 * win_h:
        return np.nan
    X = np.column_stack([np.ones(ok.sum()), np.cos(WD * tt[ok]), np.sin(WD * tt[ok]),
                         np.cos(WS * tt[ok]), np.sin(WS * tt[ok])])
    b = np.linalg.lstsq(X, y[ok], rcond=None)[0]
    return np.hypot(b[1], b[2]) / np.hypot(b[3], b[4])


half = pd.Timedelta(hours=win_h / 2)
tstart = pd.date_range(yr0 - half, yr1 - half, freq='6h')
Fs = np.array([f_ratio(t) for t in tstart])
tF = tstart + half                                  # plot at the window centre

# release classes: the same first-window values and terciles as the bulk script
R = pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_release_times'
                / ('pcmap_release_times_%d_every3.csv' % args.year), parse_dates=['t_release'])
okq = np.isfinite(qp)
Fr = np.array([f_ratio(t) for t in R.t_release])
Qr = np.array([np.interp(np.arange(t, t + pd.Timedelta(hours=win_h), pd.Timedelta(hours=1))
                         .astype('datetime64[ns]').astype('int64'),
                         tq[okq].values.astype('int64'), qp[okq]).mean() for t in R.t_release])
Fe = np.percentile(Fr, [100 / 3, 200 / 3])
Qe = np.percentile(Qr, [100 / 3, 200 / 3])
print('F terciles %.2f / %.2f, qprism terciles %.0f / %.0f m3/s, corr(F, qprism) over releases r = %+.2f'
      % (Fe[0], Fe[1], Qe[0], Qe[1], np.corrcoef(Fr, Qr)[0, 1]))

marks = []
MCOL = ['#e8455e', '#4565e8', '#45a85b', '#f0a04b']
for k, day in enumerate([d for d in args.mark.split(',') if d]):
    r = R[R.sub_tag.str.endswith(day)]
    if len(r):
        marks.append((r.t_release.min(), r.t_release.max() + pd.Timedelta(hours=win_h), day[5:],
                      MCOL[k % len(MCOL)]))
        print('%s: F %s, qprism %s' % (day, ', '.join('%.2f' % x for x in Fr[r.index]),
                                       ', '.join('%.0f' % x for x in Qr[r.index])))

mq = (tq >= yr0) & (tq < min(yr1, tq.max() - pd.Timedelta(days=2)))
fig, axs = plt.subplots(2, 1, figsize=(15, 7.5), sharex=True)
for ax, t, v, e, labs, ylab, col in [
        (axs[0], tF, Fs, Fe, ('semidiurnal', 'mid', 'diurnal'), 'F = diurnal / semidiurnal\namplitude at pc_lp', 'k'),
        (axs[1], tq[mq], qp[mq], Qe, ('neap', 'mid', 'spring'), 'qprism at pc_lp [m$^3$ s$^{-1}$]', 'k')]:
    lo = np.nanmin(v) - 0.05 * np.nanmax(v); hi = np.nanmax(v) * 1.08
    ax.axhspan(lo, e[0], color='#e4f0f7', lw=0, zorder=0)
    ax.axhspan(e[1], hi, color='#dfe5ef', lw=0, zorder=0)
    for x in e:
        ax.axhline(x, color='0.3', lw=0.8, ls='--')
    xt = yr1 - pd.Timedelta(days=3)
    for y, lab in zip([(lo + e[0]) / 2, e.mean(), (e[1] + hi) / 2], labs):
        ax.text(xt, y, lab, ha='right', va='center', fontsize=10, color='0.25')
    ax.plot(t, v, color=col, lw=1.0, zorder=3)
    for ta, tb, lab, c in marks:
        for tx in (ta, tb):
            ax.axvline(tx, color=c, lw=1.3, zorder=4)
    ax.set_ylim(lo, hi)
    ax.set_ylabel(ylab)
    ax.grid(**GRID)
ytop = axs[0].get_ylim()[1]
for k, (ta, tb, lab, c) in enumerate(marks):
    axs[0].text(tb + pd.Timedelta(days=0.5), ytop * (0.95 - 0.09 * k), lab, va='top', ha='left',
                fontsize=10, color=c, fontweight='bold', zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', fc='w', ec='none'))
axs[0].set_title('tidal asymmetry: %g-day windows, plotted at the window centre' % args.win_days, fontsize=10)
axs[1].set_title('spring-neap: Godin-filtered daily tidal prism transport', fontsize=10)
axs[1].set_xlim(yr0, yr1)
fig.suptitle('%s: tidal asymmetry and spring-neap at pc_lp, %d (shading = release-class terciles)'
             % (args.gtx, args.year), fontsize=12)
fig.tight_layout()
fn_out = out_dir / ('pcmap_tide_series_%d.png' % args.year)
fig.savefig(fn_out, dpi=200, transparent=True)
plt.close(fig)
print('wrote %s' % fn_out)

# ------------------------------------------------- the fit, per window ---
def fit_parts(t_start):
    m = (th >= t_start) & (th < t_start + pd.Timedelta(hours=win_h))
    tt = (th[m] - t_start) / pd.Timedelta(hours=1)
    y = ssh[m]
    X = np.column_stack([np.ones(len(tt)), np.cos(WD * tt), np.sin(WD * tt),
                         np.cos(WS * tt), np.sin(WS * tt)])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    tf = np.linspace(0, win_h, int(win_h * 6) + 1)
    Xf = np.column_stack([np.ones(len(tf)), np.cos(WD * tf), np.sin(WD * tf),
                          np.cos(WS * tf), np.sin(WS * tf)])
    di = Xf[:, 1:3] @ b[1:3]; se = Xf[:, 3:5] @ b[3:5]
    resid = y - X @ b
    r2 = 1 - np.var(resid) / np.var(y)
    return dict(tt=tt, y=y, tf=tf, mean=b[0], di=di, se=se, fit=b[0] + di + se, resid=resid,
                Ad=np.hypot(b[1], b[2]), As=np.hypot(b[3], b[4]), r2=r2)


wins = []
for day in [d for d in args.mark.split(',') if d]:
    r = R[R.sub_tag.str.endswith(day)].sort_values('t_release')
    if len(r):
        wins.append((day, r.sub_tag.iloc[0], r.t_release.iloc[0]))
if wins:
    fig, axs = plt.subplots(2, len(wins), figsize=(7.5 * len(wins), 7.5), sharex=True,
                            gridspec_kw=dict(height_ratios=[3, 1]), squeeze=False)
    ylim = None
    for c, (day, tag, t0) in enumerate(wins):
        Pf = fit_parts(t0)
        ax = axs[0, c]
        ax.plot(Pf['tt'], Pf['y'], 'o', ms=3.5, color='k', label='ssh at pc_lp (hourly)', zorder=5)
        ax.plot(Pf['tf'], Pf['fit'], '-', color='0.35', lw=2.2, label='fit: mean + diurnal + semidiurnal')
        ax.plot(Pf['tf'], Pf['mean'] + Pf['di'], '-', color='#762a83', lw=1.6,
                label='diurnal (24.84 h), A = %.2f m' % Pf['Ad'])
        ax.plot(Pf['tf'], Pf['mean'] + Pf['se'], '-', color='#1b7837', lw=1.6,
                label='semidiurnal (12.42 h), A = %.2f m' % Pf['As'])
        ax.axhline(Pf['mean'], color='0.6', lw=0.8, ls=':')
        ax.set_title('%s window (%s, %s UTC + %g d)\nF = %.2f / %.2f = %.2f,  fit R$^2$ = %.3f'
                     % (day[5:], tag, t0.strftime('%Y-%m-%d %H:%M'), args.win_days,
                        Pf['Ad'], Pf['As'], Pf['Ad'] / Pf['As'], Pf['r2']), fontsize=11)
        ax.grid(**GRID)
        ax.legend(fontsize=8.5, loc='lower left')
        if c == 0:
            ax.set_ylabel('ssh [m]')
        lo_, hi_ = np.nanmin(Pf['y']), np.nanmax(Pf['y'])
        ylim = (lo_, hi_) if ylim is None else (min(ylim[0], lo_), max(ylim[1], hi_))
        axr = axs[1, c]
        axr.plot(Pf['tt'], Pf['resid'], '-o', ms=2.5, lw=0.8, color='0.3')
        axr.axhline(0, color='0.6', lw=0.8)
        axr.set_xlabel('hours from release')
        axr.grid(**GRID)
        if c == 0:
            axr.set_ylabel('residual [m]')
        print('%s fit: A_diurnal %.2f m, A_semidiurnal %.2f m, F %.2f, R2 %.3f, residual rms %.2f m'
              % (tag, Pf['Ad'], Pf['As'], Pf['Ad'] / Pf['As'], Pf['r2'], np.sqrt(np.mean(Pf['resid'] ** 2))))
    pad = 0.1 * (ylim[1] - ylim[0])
    for c in range(len(wins)):
        axs[0, c].set_ylim(ylim[0] - 2.5 * pad, ylim[1] + pad)
        axs[0, c].set_xlim(0, win_h)
        axs[0, c].set_xticks(np.arange(0, win_h + 1, 12))
    rlim = max(abs(np.array(axs[1, 0].get_ylim())).max(), abs(np.array(axs[1, -1].get_ylim())).max())
    for c in range(len(wins)):
        axs[1, c].set_ylim(-rlim, rlim)
    fig.suptitle('%s: diurnal / semidiurnal harmonic fit to ssh at pc_lp over each release window'
                 % args.gtx, fontsize=12)
    fig.tight_layout()
    fn_out = out_dir / ('pcmap_tide_fit_%d.png' % args.year)
    fig.savefig(fn_out, dpi=200, transparent=True)
    plt.close(fig)
    print('wrote %s' % fn_out)
