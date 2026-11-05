"""
Time series of the Penn Cove flow metrics from 20261104_pc_flow_metrics.py set
against their drivers, and which driver explains each metric.

DAILY VALUES (calendar days, UTC)
    exchange fractions  sum_day X / sum_day G for X = Qx, LC, VC: share of the gross
                        transport that is counterflow (all / lateral / vertical); a ratio
                        of sums, so slack hours (Q ~ 0, where an hourly fraction is noise)
                        carry little weight
    peak |u|            the day's max of the hourly umax
    everything else     daily mean of the hourly values
DRIVERS
    D2, D1   semidiurnal / diurnal envelope of the mouth transport (pc_tidal_cycles.py):
             spring-neap and tropic-equatorial
    tau_al   along-cove wind stress ROMS applied, + toward the head
    tau_cr   cross-cove wind stress, + toward the north shore
    ustar3   (|tau|/rho_w)^1.5, the wind's mixing power
    drho     the section's own bottom - top sigma0 (a driver of the flow, though itself
             partly a response to wind and tide)

DRIVER TABLE. Each daily metric at each section regressed on the six drivers together,
all standardized (z-scores), so a coefficient is the change in the metric, in its own
standard deviations, per standard deviation of that driver with the others held fixed.
Done on the raw daily values (VERSIONS = ['raw']). Adding 'deseasoned' also fits the
anomalies from a centred 30-day running mean, which removes the seasonal cycle (season can
otherwise hide the tides; [[analyze-tides-on-anomalies]]). drho is not regressed on itself. No p-values: the daily
series are strongly autocorrelated, so the coefficients are descriptive.

SHADING (time series) the classes of pc_forcing_states.py: high / low stratification and
up- / down-cove wind stress, as thin bars at the top.

INPUT  LO_output/DM_outs/20261104_pc_flow_metrics/flow_metrics.nc
OUTPUT LO_output/DM_outs/20261104_pc_flow_metrics/
    flow_metrics_timeseries.png   drivers, then metrics, daily, one line per section
    flow_metrics_drivers.png      standardized regression coefficients and R2, one panel per section
    flow_metrics_drivers.csv      the same numbers

run 20261104_pc_flow_metrics_plot.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun
from pc_tidal_cycles import envelopes
from pc_forcing_states import state_masks

GTAGEX = 'wb1_t0_xn11abbur00'
SECTS = ['pc_cp', 'pc_lj', 'pc_lp']       # west to east
NAME = {'pc_lp': 'pc_lp (mouth)', 'pc_lj': 'pc_lj (middle)', 'pc_cp': 'pc_cp (Coupeville)'}
SCOL = {'pc_cp': '#CC79A7', 'pc_lj': '#E69F00', 'pc_lp': '#0072B2'}
DRIVERS = ['D2', 'D1', 'tau_al', 'tau_cr', 'ustar3', 'drho']
VERSIONS = ['raw']                        # add 'deseasoned' for anomalies from a 30-day running mean
DLAB = {'D2': 'spring-neap (D2)', 'D1': 'tropic-equat. (D1)', 'tau_al': 'along-cove stress', 'tau_cr': 'cross-cove stress',
        'ustar3': 'wind mixing u*$^3$', 'drho': r'$\Delta\sigma_0$'}
# daily metric: (label, unit scale, unit)
MET = {'Qx_G': ('counterflow share of gross transport', 1, ''),
       'LC_G': ('lateral counterflow share', 1, ''),
       'VC_G': ('layered counterflow share', 1, ''),
       'upeak': ('daily peak |u|', 100, 'cm/s'),
       'u95': ('95th pctl |u| (daily mean)', 100, 'cm/s'),
       'Srms': ('rms shear', 1000, '10$^{-3}$ s$^{-1}$'),
       'dU': ('u(top) - u(bottom), + surface into cove', 100, 'cm/s'),
       'fcrit': ('share of section with Ri < 1/4', 1, ''),
       'drho': (r'$\Delta\sigma_0$ bottom - top', 1, 'kg m$^{-3}$')}

Ldir = Lfun.Lstart(gridname='wb1')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261104_pc_flow_metrics'
ds = xr.open_dataset(out_dir / 'flow_metrics.nc')
T = pd.DatetimeIndex(ds.time.values)
day = T.floor('D')

# ------------------------------------------------------------------ daily metrics per section, and daily drivers
D = {}
for SN in SECTS:
    h = ds.sel(sect=SN).to_dataframe()[['Q', 'G', 'Qx', 'LC', 'VC', 'umax', 'u95', 'Srms', 'dU', 'drho', 'fcrit']]
    g = h.groupby(day)
    S = g.sum()
    d = pd.DataFrame({'Qx_G': S.Qx / S.G, 'LC_G': S.LC / S.G, 'VC_G': S.VC / S.G, 'upeak': g.umax.max()})
    d = d.join(g[['u95', 'Srms', 'dU', 'fcrit', 'drho']].mean())
    D[SN] = d[g.size() >= 20]             # days with nearly all their hours (drops the stub at each end)
E = envelopes(ds.time.values)
drv = pd.DataFrame({'D2': E['D2'], 'D1': E['D1'], 'tau_al': ds.tau_al.values, 'tau_cr': ds.tau_cr.values,
                    'ustar3': ds.ustar3.values, 'w_al': ds.w_al.values}, index=T).groupby(day).mean()
drv = drv.loc[D[SECTS[0]].index]
CL = state_masks(ds.time.values)          # forcing classes, hourly
CLd = pd.DataFrame({k: CL[k] for k in ['strat_high', 'strat_low', 'wind_up', 'wind_down']}, index=T).groupby(day).mean() > 0.5
CLd = CLd.loc[drv.index]
print('%d days' % len(drv))
for SN in SECTS:
    print('%s mean: ' % SN + ', '.join('%s %.3g' % (k, D[SN][k].mean() * MET[k][1]) for k in MET))


# ------------------------------------------------------------------ plot: time series
def bar(ax, keys, cols, labs):
    """A thin class strip: each class shaded where it holds (daily majority)."""
    for k, c, l in zip(keys, cols, labs):
        ax.fill_between(drv.index, 0, 1, where=CLd[k].values, color=c, lw=0, step='mid', label=l)
    ax.set_yticks([])
    ax.legend(fontsize=7, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)


roll = lambda s: s.rolling(7, center=True, min_periods=4).mean()
panels = ['strat bar', 'wind bar', 'tide', 'stress', 'ustar3'] + list(MET)
hr = [0.25, 0.25] + [1] * (len(panels) - 2)
fig, axs = plt.subplots(len(panels), 1, figsize=(14, 2.0 * sum(hr) + 1), sharex=True, layout='constrained',
                        gridspec_kw={'height_ratios': hr})
bar(axs[0], ['strat_high', 'strat_low'], ['#F4C9A8', '#A9CFE8'], ['high stratification', 'low stratification'])
bar(axs[1], ['wind_up', 'wind_down'], ['#A8DCCB', '#EBC4DA'], ['up-cove stress', 'down-cove stress'])
ax = axs[2]
ax.plot(drv.index, drv.D2, color='#0072B2', lw=1, label='semidiurnal envelope (spring-neap)')
ax.plot(drv.index, drv.D1, color='#D55E00', lw=1, label='diurnal envelope (tropic-equatorial)')
ax.set_ylabel('tidal Q at\nthe mouth [m$^3$ s$^{-1}$]', fontsize=8)
ax = axs[3]
ax.plot(drv.index, drv.tau_al, color='0.6', lw=0.6)
ax.plot(drv.index, roll(drv.tau_al), color='k', lw=1.2, label='along-cove, + toward the head')
ax.plot(drv.index, roll(drv.tau_cr), color='#009E73', lw=1.2, label='cross-cove, + toward the north shore (7-d)')
ax.axhline(0, color='k', lw=0.6)
ax.set_ylabel('wind stress [Pa]', fontsize=8)
ax = axs[4]
ax.plot(drv.index, 1e6 * drv.ustar3, color='0.6', lw=0.6)
ax.plot(drv.index, 1e6 * roll(drv.ustar3), color='k', lw=1.2, label='7-day mean')
ax.set_ylabel('u*$^3$ [10$^{-6}$ m$^3$ s$^{-3}$]', fontsize=8)
for ax, k in zip(axs[5:], MET):
    lab, sc, un = MET[k]
    for SN in SECTS:
        ax.plot(D[SN].index, sc * D[SN][k], color=SCOL[SN], lw=0.4, alpha=0.5)
        ax.plot(D[SN].index, sc * roll(D[SN][k]), color=SCOL[SN], lw=1.2, label=NAME[SN] + ' (7-day mean)' if k == 'Qx_G' else None)
    if k == 'dU':
        ax.axhline(0, color='k', lw=0.6)
    ax.set_ylabel(lab + ('\n[%s]' % un if un else ''), fontsize=8)
for ax in axs[2:]:
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, loc='upper left', ncol=3)
axs[-1].set_xlim(drv.index[0], drv.index[-1])
fig.suptitle('Penn Cove flow metrics and their drivers, daily (thin) and 7-day running mean (thick), hourly snapshots, %s' % GTAGEX, fontsize=11)
fig.savefig(out_dir / 'flow_metrics_timeseries.png', dpi=200, transparent=True)
plt.close(fig)


# ------------------------------------------------------------------ driver table: standardized multiple regression
deseason = lambda df: df - df.rolling(30, center=True, min_periods=15).mean()
z = lambda df: (df - df.mean()) / df.std()
rows = []
for version in VERSIONS:
    for SN in SECTS:
        X = drv[DRIVERS[:-1]].copy()
        X['drho'] = D[SN]['drho']
        Y = D[SN][list(MET)]
        if version == 'deseasoned':
            X, Y = deseason(X), deseason(Y)
        for k in MET:
            xs = [d_ for d_ in DRIVERS if not (k == 'drho' and d_ == 'drho')]
            dfk = pd.concat([X[xs], Y[k]], axis=1).dropna()
            Z = z(dfk)
            A = np.column_stack([Z[xs].values, np.ones(len(Z))])
            b, *_ = np.linalg.lstsq(A, Z[k].values, rcond=None)
            r2 = 1 - np.sum((Z[k].values - A @ b)**2) / np.sum((Z[k].values - Z[k].mean())**2)
            row = dict(version=version, sect=SN, metric=k, R2=r2, n=len(Z))
            row.update({d_: np.nan for d_ in DRIVERS})
            row.update(dict(zip(xs, b[:-1])))
            rows.append(row)
tab = pd.DataFrame(rows)
tab.to_csv(out_dir / 'flow_metrics_drivers.csv', index=False, float_format='%.3f')

fig, axs = plt.subplots(len(VERSIONS), 3, figsize=(16, 5.5 * len(VERSIONS) + 0.5), layout='constrained', squeeze=False)   # not sharey: each panel has its own R2 labels
for i, version in enumerate(VERSIONS):
    for j, SN in enumerate(SECTS):
        ax = axs[i, j]
        t = tab[(tab.version == version) & (tab.sect == SN)].set_index('metric').loc[list(MET)]
        B = t[DRIVERS].values
        im = ax.imshow(B, cmap='RdBu_r', vmin=-0.8, vmax=0.8, aspect='auto')
        for (r, c), b in np.ndenumerate(B):
            if np.isfinite(b):
                ax.text(c, r, '%+.2f' % b, ha='center', va='center', fontsize=7, color='w' if abs(b) > 0.5 else 'k')
        ax.set_xticks(range(len(DRIVERS)))
        ax.set_xticklabels([DLAB[d_] for d_ in DRIVERS], rotation=40, ha='right', fontsize=8)
        ax.set_yticks(range(len(MET)))
        ax.set_yticklabels(['%s  (R$^2$ %.2f)' % (MET[k][0], r2) for k, r2 in zip(MET, t.R2)], fontsize=8)
        ax.set_title('%s, %s' % (NAME[SN], version), fontsize=10)
        if j > 0:
            ax.yaxis.set_tick_params(labelleft=True)
fig.colorbar(im, ax=axs, location='bottom', shrink=0.5, aspect=40,
             label='standardized coefficient: change in the metric (in its SDs) per SD of the driver, the others held fixed')
fig.suptitle('Which driver explains each daily flow metric? Multiple regression on all six drivers together, %s daily values.\n'
             'Descriptive: no p-values (the daily series are autocorrelated)' % ' and '.join(VERSIONS), fontsize=10)
fig.savefig(out_dir / 'flow_metrics_drivers.png', dpi=200, transparent=True)
plt.close(fig)
print('saved to ' + str(out_dir))
