"""
Penn Cove metric time series (20261110_pc_metric_ts.py) against their drivers, and
standardized multiple regressions on the drivers.

DRIVERS (hourly, on the metric clock: ocean_his snapshots, hh:00)
    D2      semidiurnal envelope of the mouth transport: spring-neap   (pc_tidal_cycles.py)
    D1      diurnal envelope of the mouth transport: tropic-equatorial (pc_tidal_cycles.py)
    drho    Godin cove-mean sigma0(bottom) - sigma0(top), the strat-class series (pc_forcing_states.py)
    tau_al  Godin along-cove wind stress ROMS applied, + toward the head (pc_forcing_states.py)
    tau_cr  Godin cross-cove wind stress, + toward the north shore    (pc_forcing_states.py)

UNITS OF ANALYSIS
    TIDAL, per half-cycle. A flood (ebb) is a run of hours with mouth Q > 0 (< 0), Q at pc_lp,
    + into the cove; the one clock for every section, as in 20261110_pc_column_metrics.py.
    Runs shorter than MIN_RUN hours are slack jitter, absorbed into the run before them; the
    first and last (partial) runs are dropped. Per half-cycle and section:
        Qpk     max |Q|                                   [m3/s]
        Qprism  mean of the hourly Qprism                 [m3/s]
        umax    max of the hourly umax                    [m/s]
        fopp    |Q|-weighted mean share of the cross-section against Q (slack hours, when the
                share is noise, carry little weight; cf. the ratio of sums of 20261104)
        surf_in share of hours with the surface flowing INTO the cove
        dint    median surface-layer depth over the two-layer hours  [m]
      and for the curtain (one per half-cycle): fopp_cur, |Q_mouth|-weighted.
      Drivers: the half-cycle mean. Floods and ebbs are regressed separately.
    HOURLY (2026-11-11, for comparison with the half-cycles). The hourly tidal metrics as they
      are, |Q|, Qprism, umax, fopp, Usurf, dint (and fopp_cur), against the hourly drivers;
      flood / ebb = the half-cycle sign above, every hour kept. Within-tide variation (slack to
      peak) is in the metric but not in the drivers, so R2 is lower than per half-cycle.
    SUBTIDAL, daily. The Godin series sampled at 12:00 UTC: umax_res, fsea_res, Usurf_res,
      dint_res per section; fsea_cur_res, rot_U, rot_frac once. Drivers at the same hours.

DRIVER SETS (MODELS)
    base   D2, D1, drho, tau_al, tau_cr
    split  (2026-11-11) tau_al split by direction, tau_up = max(tau_al, 0), tau_dn = min(tau_al, 0)
           (tau_dn keeps tau_al's sign, so + still = the metric rises as the stress moves toward the
           head); both scaled by the SD of tau_al, not their own, so equal coefficients = one straight
           line, and up- and down-cove wind are compared per the same amount of stress.
           Half-cycles also get Qtide = max |Q_pred| over the half-cycle's hours, Q_pred the D2 + D1
           harmonic prediction of the mouth transport (pc_tidal_cycles fit): this tide's own strength,
           incl. the greater / lesser inequality the envelopes miss; D2 and D1 then show what spring-neap
           and tropic-equatorial add beyond it.

REGRESSION. Each metric on its model's drivers together, all z-scored, so a coefficient is the
change in the metric, in its own SDs, per SD of the driver with the others held fixed; R2 of
the fit. Two versions: 'raw', and 'deseasoned' (metric and drivers minus a centred 30-day
running mean, which removes the seasonal cycle; [[analyze-tides-on-anomalies]]). No p-values:
the series are strongly autocorrelated, so the coefficients are descriptive.

INPUT  LO_output/DM_outs/20261110_pc_metric_ts/pc_metric_ts.nc
OUTPUT LO_output/DM_outs/20261111_pc_metric_drivers/
    halfcycle.csv, daily.csv              the per-half-cycle and daily tables
    coefs.csv                             coefficients and R2
    tidal_<sect>.png, tidal_curtain.png   drivers + per-half-cycle metrics, flood vs ebb
    subtidal.png                          drivers + daily subtidal metrics
    coefs_<version>.png                   coefficient heat maps: flood | ebb | subtidal
    coefs_hourly_<version>.png            the same for the hourly fits: flood | ebb
    coefs_split_<version>.png, coefs_hourly_split_<version>.png   the same for the split model

run 20261111_pc_metric_drivers.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun
import utide
from pc_tidal_cycles import envelopes, mouth_fit, D2 as D2_CONS, D1 as D1_CONS
from pc_forcing_states import series, nearest

SECTS = ['pc_cp', 'pc_lj', 'pc_lp']
SCOL = {'pc_cp': '#CC79A7', 'pc_lj': '#E69F00', 'pc_lp': '#0072B2'}
PCOL = {'flood': '#009E73', 'ebb': '#D55E00'}
DRIVERS = ['D2', 'D1', 'drho', 'tau_al', 'tau_cr']
DLAB = {'D2': 'spring-neap D2 [m$^3$/s]', 'D1': 'tropic-equat. D1 [m$^3$/s]', 'drho': r'$\Delta\sigma_0$ [kg/m$^3$]',
        'tau_al': 'along-cove\nstress [Pa]', 'tau_cr': 'cross-cove\nstress [Pa]'}
DSHORT = {'D2': 'D2', 'D1': 'D1', 'drho': r'$\Delta\sigma_0$', 'tau_al': r'$\tau_{al}$', 'tau_cr': r'$\tau_{cr}$',
          'Qtide': 'Q$_{tide}$', 'tau_up': r'$\tau_{al}^+$', 'tau_dn': r'$\tau_{al}^-$'}
SPLIT = ['D2', 'D1', 'drho', 'tau_up', 'tau_dn', 'tau_cr']
MODELS = {'base': dict(tidal=DRIVERS, hourly=DRIVERS, subtidal=DRIVERS),
          'split': dict(tidal=['Qtide'] + SPLIT, hourly=SPLIT, subtidal=SPLIT)}
VERSIONS = ['raw', 'deseasoned']
MIN_RUN = 3                               # hours; shorter same-sign runs of mouth Q are slack jitter

Ldir = Lfun.Lstart(gridname='wb1')
dmo = Ldir['LOo'] / 'DM_outs'
out_dir = dmo / '20261111_pc_metric_drivers'
Lfun.make_dir(out_dir)

# ------------------------------------------------------------------ metrics and drivers on one hourly clock
d = xr.open_dataset(dmo / '20261110_pc_metric_ts' / 'pc_metric_ts.nc')
T = pd.DatetimeIndex(d.time.values)
NT = len(T)
E = envelopes(T)
Qpred = utide.reconstruct(T, mouth_fit(), constit=D2_CONS + D1_CONS, verbose=False).h   # harmonic tide at the mouth
X = series()
i_s, i_t = nearest(X['t_s'], T), nearest(X['t_t'], T)
DRV = pd.DataFrame({'D2': E['D2'], 'D1': E['D1'], 'drho': X['drho_lp'][i_s],
                    'tau_al': X['tau_lp'][i_t], 'tau_cr': X['tau_cr_lp'][i_t]}, index=T)
DRV['tau_up'], DRV['tau_dn'] = DRV.tau_al.clip(lower=0), DRV.tau_al.clip(upper=0)
Qm = d.Q.sel(sect='pc_lp').values
print('drivers on %d hours; finite in all five: %d' % (NT, DRV.notna().all(axis=1).sum()))


# ------------------------------------------------------------------ tidal: floods and ebbs of the mouth transport
def runs(s):
    """[(start, stop), ...] of constant s (stop exclusive)"""
    k = np.flatnonzero(np.diff(s) != 0) + 1
    return list(zip(np.r_[0, k], np.r_[k, len(s)]))


sg = np.sign(Qm)
for a, b in runs(sg):                                     # short runs -> 0, then take the previous sign
    if b - a < MIN_RUN:
        sg[a:b] = 0
sg = pd.Series(sg).replace(0, np.nan).ffill().bfill().to_numpy()
HC = runs(sg)[1:-1]                                       # the first and last runs are partial
wmean = lambda x, w: np.nansum(x * w) / np.nansum(w * np.isfinite(x))
rows = []
for a, b in HC:
    sl = slice(a, b)
    base = dict(phase='flood' if sg[a] > 0 else 'ebb', t0=T[a], t1=T[b - 1], tmid=T[a] + (T[b - 1] - T[a]) / 2, nh=b - a)
    base.update(DRV.iloc[sl].mean().to_dict())
    base['Qtide'] = np.abs(Qpred[sl]).max()
    for sn in SECTS:
        m = d.sel(sect=sn)
        Q = m.Q.values[sl]
        di = m.dint.values[sl]
        rows.append(dict(base, sect=sn, Qpk=np.abs(Q).max(), Qprism=np.nanmean(m.Qprism.values[sl]),
                         umax=m.umax.values[sl].max(), fopp=wmean(m.fopp.values[sl], np.abs(Q)),
                         surf_in=(m.Usurf.values[sl] > 0).mean(),
                         dint=np.nanmedian(di) if np.isfinite(di).any() else np.nan))
    rows.append(dict(base, sect='curtain', fopp_cur=wmean(d.fopp_cur.values[sl], np.abs(Qm[sl]))))
H = pd.DataFrame(rows)
H.to_csv(out_dir / 'halfcycle.csv', index=False, float_format='%.5g')
hc = H[H.sect == 'pc_lp']
print('%d floods (median %.0f h), %d ebbs (median %.0f h)'
      % ((hc.phase == 'flood').sum(), hc[hc.phase == 'flood'].nh.median(), (hc.phase == 'ebb').sum(), hc[hc.phase == 'ebb'].nh.median()))

# ------------------------------------------------------------------ tidal, hourly: every hour, flood / ebb by the same sign
PH = np.where(sg > 0, 'flood', 'ebb')
rows = []
for sn in SECTS:
    m = d.sel(sect=sn)
    rows.append(pd.DataFrame({'sect': sn, 'phase': PH, 'Qabs': np.abs(m.Q.values),
                              **{k: m[k].values for k in ['Qprism', 'umax', 'fopp', 'Usurf', 'dint']}}, index=T))
rows.append(pd.DataFrame({'sect': 'curtain', 'phase': PH, 'fopp_cur': d.fopp_cur.values}, index=T))
Hh = pd.concat(rows).join(DRV)

# ------------------------------------------------------------------ subtidal: Godin series at 12:00 UTC
noon = np.flatnonzero(T.hour == 12)
rows = []
for sn in SECTS:
    m = d.sel(sect=sn)
    rows.append(pd.DataFrame({'sect': sn, **{k: m[k].values[noon] for k in ['umax_res', 'fsea_res', 'Usurf_res', 'dint_res']}},
                             index=T[noon]))
rows.append(pd.DataFrame({'sect': 'curtain', **{k: d[k].values[noon] for k in ['fsea_cur_res', 'rot_U', 'rot_frac']}}, index=T[noon]))
D = pd.concat(rows).join(DRV.iloc[noon])
D.index.name = 'time'
D.to_csv(out_dir / 'daily.csv', float_format='%.5g')

# ------------------------------------------------------------------ regressions
MET = {   # unit: {sect: [metrics]}
    'tidal': {**{sn: ['Qpk', 'Qprism', 'umax', 'fopp', 'surf_in', 'dint'] for sn in SECTS}, 'curtain': ['fopp_cur']},
    'hourly': {**{sn: ['Qabs', 'Qprism', 'umax', 'fopp', 'Usurf', 'dint'] for sn in SECTS}, 'curtain': ['fopp_cur']},
    'subtidal': {**{sn: ['umax_res', 'fsea_res', 'Usurf_res', 'dint_res'] for sn in SECTS}, 'curtain': ['fsea_cur_res', 'rot_U', 'rot_frac']},
}


def fit(df, k, version, drv):
    """standardized multiple regression of df[k] on the drivers drv; df indexed by time"""
    Z = df[list(dict.fromkeys(drv + ['tau_al', k]))].astype(float)
    if version == 'deseasoned':
        span = (Z.index[-1] - Z.index[0]) / pd.Timedelta('1D')
        for c in Z:                                    # need half of the series' own points per 30 days
            n30 = 30 * Z[c].notna().sum() / span
            Z[c] = Z[c] - Z[c].rolling('30D', center=True, min_periods=max(int(n30 / 2), 1)).mean()
    Z = Z.dropna()
    sd = Z.std()
    sd[[c for c in ['tau_up', 'tau_dn'] if c in sd]] = sd['tau_al']   # the split parts on tau_al's scale
    Z = (Z - Z.mean()) / sd
    A = np.c_[Z[drv].values, np.ones(len(Z))]
    b, *_ = np.linalg.lstsq(A, Z[k].values, rcond=None)
    r2 = 1 - ((Z[k].values - A @ b)**2).sum() / (Z[k].values**2).sum()
    return dict(zip(drv, b[:-1]), R2=r2, n=len(Z))


rows = []
for model, version in [(mo, v) for mo in MODELS for v in VERSIONS]:
    for unit, sm in MET.items():
        for phase in (['subtidal'] if unit == 'subtidal' else ['flood', 'ebb']):
            for sn, ks in sm.items():
                df = {'tidal': lambda: H[(H.sect == sn) & (H.phase == phase)].set_index('tmid'),
                      'hourly': lambda: Hh[(Hh.sect == sn) & (Hh.phase == phase)],
                      'subtidal': lambda: D[D.sect == sn]}[unit]()
                for k in ks:
                    rows.append(dict(model=model, version=version, unit=unit, phase=phase, sect=sn, metric=k,
                                     **fit(df, k, version, MODELS[model][unit])))
C = pd.DataFrame(rows)
C.to_csv(out_dir / 'coefs.csv', index=False, float_format='%.3f')
print(C.groupby(['model', 'version', 'unit', 'phase']).R2.median().round(2).unstack('model').to_string())

# ------------------------------------------------------------------ plots
MLAB = {'Qpk': ('peak |Q| [m$^3$/s]', 1), 'Qabs': ('|Q| [m$^3$/s]', 1), 'Usurf': ('surface u [cm/s]', 100), 'Qprism': ('Q$_{prism}$ [m$^3$/s]', 1), 'umax': ('U$_{max}$ [cm/s]', 100),
        'fopp': ('cross-section\nagainst Q [%]', 100), 'surf_in': ('hours surface\ninto cove [%]', 100),
        'dint': ('surface layer\ndepth [m]', 1), 'fopp_cur': ('vertical section\nagainst Q [%]', 100),
        'umax_res': ('residual\nU$_{max}$ [cm/s]', 100), 'fsea_res': ('cross-section\nseaward [%]', 100),
        'Usurf_res': ('residual\nsurface u [cm/s]', 100), 'dint_res': ('residual surface\nlayer depth [m]', 1),
        'fsea_cur_res': ('vertical section\nseaward [%]', 100), 'rot_U': ('back-cove\nrotation [cm/s]', 100),
        'rot_frac': ('column rotating\nccw [%]', 100)}


def drivers(axs):
    for ax, k in zip(axs, DRIVERS):
        ax.plot(DRV.index, DRV[k], lw=0.6, color='k')
        ax.set_ylabel(DLAB[k], fontsize=8)
        if k.startswith('tau'):
            ax.axhline(0, color='0.4', lw=0.5)


def finish(fig, axs, name, title):
    for ax in axs:
        ax.grid(True, color='0.85', lw=0.5)
        ax.set_xlim(T[0], T[-1])
    loc = mdates.AutoDateLocator()
    axs[-1].xaxis.set_major_locator(loc)
    axs[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    fig.suptitle(title)
    fig.savefig(out_dir / name, dpi=200, transparent=True)
    plt.close(fig)
    print('saved', out_dir / name)


# tidal: one figure per section (and the curtain), floods and ebbs as dots at mid-half-cycle
for sn, ks in MET['tidal'].items():
    fig, axs = plt.subplots(len(DRIVERS) + len(ks), 1, figsize=(12, 1.35 * (len(DRIVERS) + len(ks)) + 1), sharex=True, layout='constrained')
    drivers(axs)
    for ax, k in zip(axs[len(DRIVERS):], ks):
        for ph in ['flood', 'ebb']:
            h = H[(H.sect == sn) & (H.phase == ph)]
            ax.plot(h.tmid, MLAB[k][1] * h[k], '.', ms=1.5, color=PCOL[ph], label=ph)
        ax.set_ylabel(MLAB[k][0], fontsize=8)
    axs[len(DRIVERS)].legend(loc='upper right', ncol=2, fontsize=8, frameon=False, markerscale=6)
    finish(fig, axs, 'tidal_%s.png' % sn, '%s: drivers (hourly) and per-half-cycle tidal metrics (+ into the cove)' % sn)

# subtidal: one figure, sections as colours, the curtain / rotation in black
ks = ['umax_res', 'fsea_res', 'Usurf_res', 'dint_res', 'fsea_cur_res', 'rot_U', 'rot_frac']
fig, axs = plt.subplots(len(DRIVERS) + len(ks), 1, figsize=(12, 17), sharex=True, layout='constrained')
drivers(axs)
for ax, k in zip(axs[len(DRIVERS):], ks):
    for sn in (SECTS if k in MET['subtidal']['pc_lp'] else ['curtain']):
        dd = D[D.sect == sn]
        ax.plot(dd.index, MLAB[k][1] * dd[k], '.-' if k == 'dint_res' else '-', ms=2, lw=0.8,
                color=SCOL.get(sn, 'k'), label=sn)
    ax.set_ylabel(MLAB[k][0], fontsize=8)
    if k in ['Usurf_res', 'rot_U']:
        ax.axhline(0, color='0.4', lw=0.5)
axs[len(DRIVERS)].legend(loc='upper right', ncol=3, fontsize=8, frameon=False)
finish(fig, axs, 'subtidal.png', 'Drivers and daily (12:00 UTC) subtidal metrics (+ into the cove; rotation + ccw)')

# coefficient heat maps: flood | ebb | subtidal, one figure per version
PANELS = {'coefs_%s%s.png': [('tidal', 'flood'), ('tidal', 'ebb'), ('subtidal', 'subtidal')],
          'coefs_hourly_%s%s.png': [('hourly', 'flood'), ('hourly', 'ebb')]}
for (fn, panels), model, version in [(p, mo, v) for p in PANELS.items() for mo in MODELS for v in VERSIONS]:
    fn = fn % ('' if model == 'base' else model + '_', version)
    fig, axs = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 8.5), layout='constrained')
    for ax, (unit, ph) in zip(axs, panels):
        drv = MODELS[model][unit]
        c = C[(C.model == model) & (C.version == version) & (C.unit == unit) & (C.phase == ph)]
        order = list(dict.fromkeys(c.metric))                         # rows grouped by metric, sections within
        c = c.assign(o=c.metric.map(order.index)).sort_values(['o'], kind='stable').reset_index(drop=True)
        B = c[drv].values
        im = ax.imshow(B, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        for (i, j), v in np.ndenumerate(B):
            ax.text(j, i, '%+.2f' % v, ha='center', va='center', fontsize=6.5, color='w' if abs(v) > 0.6 else 'k')
        for i, r2 in enumerate(c.R2):
            ax.text(len(drv) - 0.35, i, '%.2f' % r2, ha='left', va='center', fontsize=6.5)
        ax.text(len(drv) - 0.35, -0.9, 'R$^2$', ha='left', va='center', fontsize=8)
        ax.set_xticks(range(len(drv)), [DSHORT[k] for k in drv])
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(c)), ['%s  %s' % (MLAB[k][0].replace('\n', ' ').split(' [')[0], s) for k, s in zip(c.metric, c.sect)],
                      fontsize=7)
        for i in np.flatnonzero(c.metric.values[1:] != c.metric.values[:-1]):
            ax.axhline(i + 0.5, color='k', lw=0.8)                  # metric groups
        ax.set_title('%s, %s%s, %s' % (ph, version, ', hourly' if unit == 'hourly' else '', model), fontsize=10, pad=18)
    fig.colorbar(im, ax=axs, shrink=0.5, label='standardized coefficient: SD of metric per SD of driver, others held fixed')
    fig.savefig(out_dir / fn, dpi=200, transparent=True)
    plt.close(fig)
    print('saved', out_dir / fn)
d.close()
