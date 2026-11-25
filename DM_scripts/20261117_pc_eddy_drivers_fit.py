"""
Which drivers go with each aspect of the inner-cove eddy? Simple, one driver at a time, 2024 only.
Daily table from 20261117_pc_eddy_drivers.py.
Aspects: WHEN presence, HOW STRONG U_max and zeta_c / f (per 2 m bin, 0-12 m), WHERE centre x, y, r_max, and
VERTICAL presence depth, thickness and depth of the strongest swirl.

STEP 1  for every response x driver:
          r          Pearson correlation of the daily values
          sig        p < 0.05 with the effective n = n (1 - r1x r1y) / (1 + r1x r1y), with r1 the lag-1
                     autocorrelations (Bretherton et al. 1999), because the days are autocorrelated
          dhi_lo     composite: response mean on the driver's top-third days minus its mean on the bottom-third days,
                     in the response's own units
        Each driver is tested alone, so correlated drivers (strat, river, heat, season) will look alike.
STEP 2  figures + results.csv:
          correlation.png   r heat map, significant in bold
          composites.png    top-third minus bottom-third, coloured by the difference / response SD, labelled in units

INPUT  LO_output/DM_outs/20261117_pc_eddy_drivers/eddy_drivers_daily.csv
OUTPUT LO_output/DM_outs/20261117_pc_eddy_drivers/

run 20261117_pc_eddy_drivers_fit.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from lo_tools import Lfun

YEAR = 2024
ZB = [1, 3, 5, 7, 9, 11]
DRV = ['D2', 'D1', 'U_tide_max', 'U_res_max', 'drho', 'z_pyc', 'drhox_top', 'drhox_bot', 'tau_al', 'tau_al_prior', 'tau_cr', 'tau_cr_prior',
       'shflux', 'skagit', 'sin_doy', 'cos_doy']
RESP = ([('pres_%d' % z, 'presence %d-%d m' % (z - 1, z + 1), '%') for z in ZB]
        + [('Umax_%d' % z, 'U$_{max}$ %d-%d m' % (z - 1, z + 1), 'cm/s') for z in ZB]
        + [('Ro_%d' % z, '$\\zeta_c$/f %d-%d m' % (z - 1, z + 1), '') for z in ZB]
        + [('x_c', 'centre x (east)', 'm'), ('y_c', 'centre y (north)', 'm'), ('r_max', 'r$_{max}$', 'm'),
           ('z_pres', 'presence depth', 'm'), ('thick', 'presence thickness', 'm'), ('z_Umax', 'depth of max U$_{max}$', 'm')])

Ldir = Lfun.Lstart()
out_dir = Ldir['LOo'] / 'DM_outs' / '20261117_pc_eddy_drivers'
T = pd.read_csv(out_dir / 'eddy_drivers_daily.csv', parse_dates=['date']).set_index('date')
T = T[T.index.year == YEAR]
for z in ZB:
    T['pres_%d' % z] *= 100                                      # share -> % of hours
print('%d days in %d' % (len(T), YEAR))

# ------------------------------------------------------------------ STEP 1: r, effective-n significance, composites
r1 = lambda s: s.autocorr(lag=1)
OUT = []
for vn, lab, un in RESP:
    for dv in DRV:
        S = T[[vn, dv]].dropna()
        y, x = S[vn], S[dv]
        r = np.corrcoef(x, y)[0, 1]
        a = r1(x) * r1(y)                                        # lag-1 autocorrelations of the pair
        neff = len(S) * (1 - a) / (1 + a)
        t = r * np.sqrt((neff - 2) / (1 - r ** 2))
        p = 2 * stats.t.sf(abs(t), neff - 2)
        lo, hi = x.quantile([1 / 3, 2 / 3])
        OUT.append({'resp': vn, 'label': lab, 'units': un, 'driver': dv, 'n': len(S), 'n_eff': neff, 'r': r, 'p': p,
                    'sig': p < 0.05, 'dhi_lo': y[x >= hi].mean() - y[x <= lo].mean(), 'resp_sd': y.std()})
OUT = pd.DataFrame(OUT)
OUT.to_csv(out_dir / 'results.csv', index=False, float_format='%.4g')
print('saved %s' % (out_dir / 'results.csv'))
for vn, g in OUT.groupby('resp', sort=False):
    g = g[g['sig']].reindex(g[g['sig']]['r'].abs().sort_values(ascending=False).index)
    print('%-8s %s' % (vn, ', '.join('%s %+.2f' % (d, r) for d, r in zip(g['driver'][:4], g['r'][:4])) or '-'))

# ------------------------------------------------------------------ STEP 2: figures
plt.rcParams.update({'font.size': 12})
labs = [lab for _, lab, _ in RESP]
seps = [6, 12, 18, 21]                                           # presence | U_max | zeta/f | where | vertical
M = lambda c: OUT.pivot(index='resp', columns='driver', values=c).loc[[v for v, _, _ in RESP], DRV].to_numpy()
R_, S_, D_, SD = M('r'), M('sig'), M('dhi_lo'), M('resp_sd')
UN = np.array([un for _, _, un in RESP])

for name, C, txt, cl, title in [
        ('correlation', R_, lambda i, j: '%.2f' % R_[i, j], 'r',
         'r with each driver (daily, %d); bold = p < 0.05 with the effective n' % YEAR),
        ('composites', D_ / SD, lambda i, j: ('%.0f' if abs(D_[i, j]) >= 10 else '%.2g') % D_[i, j], '(top third - bottom third) / response SD',
         'Response on the driver\'s top-third days minus its bottom-third days (%d); text in the response\'s units, bold = r significant' % YEAR)]:
    fig, ax = plt.subplots(figsize=(17, 13), layout='constrained')
    im = ax.imshow(C, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if np.isfinite(C[i, j]):
                ax.text(j, i, txt(i, j), ha='center', va='center', fontsize=8,
                        fontweight='bold' if S_[i, j] else 'normal', color='k' if S_[i, j] else '0.5')
    ax.set_xticks(range(len(DRV)), DRV, rotation=45, ha='right')
    ax.set_yticks(range(len(labs)), ['%s [%s]' % (l, u) if (name == 'composites' and u) else l for l, u in zip(labs, UN)])
    for s in seps:
        ax.axhline(s - 0.5, color='k', lw=1.5)
    for s in [2, 4, 8, 12, 13, 14]:                              # tide | pc_lj | density | wind | heat | river | season
        ax.axvline(s - 0.5, color='0.4', lw=1)
    fig.colorbar(im, ax=ax, shrink=0.5, label=cl)
    ax.set_title(title + '\n', loc='left', fontweight='bold')
    fig.savefig(out_dir / ('%s.png' % name), dpi=200, transparent=True)
    plt.close(fig)
    print('saved %s' % (out_dir / ('%s.png' % name)))
