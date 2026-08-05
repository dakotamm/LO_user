"""
What ventilates Penn Cove? A synthesis of the wb1_pc1 TEF results.

Five panels, each answering one step of the argument:

  a  the cove's salinity is set remotely by the Skagit, not locally
  b  a weak along-cove gradient appears in winter and tracks the exchange
  c  but the exchange carries no vertical salinity contrast, in any month
  d  the flow at the mouth is overwhelmingly tidal, not subtidal
  e  and the tidal prism is small compared with the volume it has to flush

Together: a basin that is vigorously stirred but poorly renewed, with an
exchange whose seasonal cycle follows the Skagit yet never organises into a
two-layer estuarine circulation.

Needs bulk_avg (tidally averaged) and hourly_flux (unfiltered) on the mac.

run 20260805_penncove_exchange_synthesis.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun

Ldir = Lfun.Lstart(gridname='wb1')
gtx = 'wb1_t0_xn11abbur00'
ds0, ds1 = '2024.01.01', '2025.12.31'
tef2 = Ldir['LOo'] / 'extract' / gtx / 'tef2'
bulk_dir = tef2 / ('bulk_avg_' + ds0 + '_' + ds1)
out_dir = Path.home() / 'Desktop' / 'pltz'
Lfun.make_dir(out_dir)

# CVD-validated categorical order, same as the qprism and flushing figures so
# a section keeps its colour across every plot in this set
COLOR = {'pc_cp': '#009E73', 'pc_lj': '#0072B2', 'pc_lp': '#CC79A7',
         'sp_mid': '#D55E00', 'skagit_sp': '#5D3A9B'}
MON = list('JFMAMJJASOND')

# ------------------------------------------------------------------ load ----
S, Qin, Sin, Sout, Qpr = {}, {}, {}, {}, {}
for sn in COLOR:
    d = xr.open_dataset(bulk_dir / (sn + '.nc'))
    q = d.q.values
    sa = d.salt.values
    t = pd.to_datetime(d.time.values)
    w = np.abs(q)
    ok = np.isfinite(sa)
    S[sn] = pd.Series(np.nansum(np.where(ok, w * sa, 0), axis=1)
                      / np.nansum(np.where(ok, w, 0), axis=1), index=t)
    qi = np.nansum(np.where(q > 0, q, 0), axis=1)
    qo = np.nansum(np.where(q < 0, q, 0), axis=1)
    Qin[sn] = pd.Series(qi, index=t)
    Sin[sn] = pd.Series(np.nansum(np.where(q > 0, q * sa, 0), axis=1) / qi, index=t)
    Sout[sn] = pd.Series(np.nansum(np.where(q < 0, q * sa, 0), axis=1) / qo, index=t)
    Qpr[sn] = pd.Series(d.qprism.values, index=t)
    d.close()

hf = xr.open_dataset(tef2 / ('hourly_flux_' + ds0 + '_' + ds1 + '_wb1_pc1.nc'))
th = pd.to_datetime(hf.time.values)
k_lp = list(hf.sect.values).index('pc_lp')
qnet_h = pd.Series(hf.qnet.values[:, k_lp], index=th)

vol = pd.read_pickle(Ldir['LOo'] / 'extract' / 'tef2' / 'vol_df_wb1_pc1.p')
V = {'pc_cp': vol.loc['pc_cp_m', 'volume m3'],
     'pc_lj': vol.loc[['pc_cp_m', 'pc_cp_p'], 'volume m3'].sum(),
     'pc_lp': vol.loc[['pc_cp_m', 'pc_cp_p', 'pc_lp_m'], 'volume m3'].sum()}

# ---------------------------------------------------------------- derived ---
dS = (S['pc_lp'] - S['pc_cp']).rename('dS')      # >0 means the cove is fresher
T = (V['pc_lp'] / Qin['pc_lp'] / 86400).rename('T')
mon = pd.DataFrame({'dS': dS, 'Qin': Qin['pc_lp'], 'T': T})
mon = mon.groupby(mon.index.month).mean()

dso = (Sin['pc_lp'] - Sout['pc_lp']).rename('dSinout')
g = dso.groupby(dso.index.month)
ci = pd.DataFrame({'mean': g.mean(), 'sd': g.std(), 'n': g.count()})
ci['se'] = ci.sd / np.sqrt(ci.n)

# tidal prism per half cycle, and the return-flow factor implied by
# T_flush = V / ((P/T_tide)(1-b))
T_TIDE = 12.42 * 3600
rows = []
for sn in ['pc_cp', 'pc_lj', 'pc_lp']:
    kk = list(hf.sect.values).index(sn)
    P = np.nanmean(np.abs(hf.qnet.values[:, kk])) * T_TIDE / 2
    Tf = np.nanmean(V[sn] / Qin[sn] / 86400) * 86400
    one_minus_b = V[sn] / (Tf * P / T_TIDE)
    rows.append(dict(sect=sn, V=V[sn], P=P, frac=100 * P / V[sn],
                     b=1 - one_minus_b))
pr = pd.DataFrame(rows).set_index('sect')

print(pr.round(3).to_string())
print('\nmonthly dS vs Qin r = %+.3f   dS vs T r = %+.3f'
      % (mon.dS.corr(mon.Qin), mon.dS.corr(mon['T'])))

# ----------------------------------------------------------------- figure ---
plt.close('all')
fig, ax = plt.subplot_mosaic([['a', 'a', 'b'], ['c', 'd', 'e']],
                             figsize=(16, 9), layout='constrained')

# a: the Skagit signal
A = ax['a']
for sn in ['pc_cp', 'pc_lp', 'skagit_sp', 'sp_mid']:
    ms = S[sn].groupby(S[sn].index.month).mean()
    A.plot(ms.index, ms.values, '-o', lw=2, ms=5, color=COLOR[sn], label=sn)
A.set_xticks(range(1, 13)); A.set_xticklabels(MON)
A.set_ylabel('section salinity [g kg$^{-1}$]')
A.set_title('a. Penn Cove salinity is set remotely: 5.7 g kg$^{-1}$ seasonal swing,\n'
            'the largest of any section, freshest in December')
A.grid(color='lightgray', ls='--', alpha=0.5)
A.legend(fontsize=9, ncol=4)

# b: dS against Qin, monthly
B = ax['b']
B.scatter(mon.dS, mon.Qin, c=range(1, 13), cmap='twilight', s=90,
          edgecolor='k', linewidth=0.5, zorder=5)
for m in range(1, 13):
    B.annotate(MON[m - 1], (mon.dS[m], mon.Qin[m]), fontsize=8,
               xytext=(5, 4), textcoords='offset points')
B.set_xlabel(r'$\Delta S$ = s(pc_lp) - s(pc_cp)   [cove fresher $\rightarrow$]')
B.set_ylabel(r'$Q_{in}$ at pc_lp  [m$^3$ s$^{-1}$]')
B.set_title('b. Exchange tracks the along-cove\ngradient (monthly r = %+.2f)'
            % mon.dS.corr(mon.Qin))
B.grid(color='lightgray', ls='--', alpha=0.5)

# c: Sin - Sout, never significant
C = ax['c']
C.axhline(0, color='k', lw=1)
C.errorbar(range(1, 13), ci['mean'], yerr=1.96 * ci.se, fmt='o', ms=6,
           color=COLOR['pc_lp'], ecolor='0.4', capsize=3, lw=1.5)
C.set_xticks(range(1, 13)); C.set_xticklabels(MON)
C.set_ylabel(r'$S_{in} - S_{out}$  [g kg$^{-1}$]')
C.set_title('c. ...but the exchange carries no vertical\n'
            'salinity contrast, in any month (95% CI)')
C.grid(color='lightgray', ls='--', alpha=0.5)

# d: tidal vs subtidal
D = ax['d']
w = qnet_h.loc['2024-12-01':'2024-12-11']
D.plot(w.index, w.values, '-', lw=0.9, color=COLOR['pc_lp'], label='hourly')
dm = w.resample('1D').mean()
D.plot(dm.index, dm.values, '-o', lw=2.5, ms=5, color='k', label='daily mean')
D.axhline(0, color='0.5', lw=0.8)
D.set_ylabel(r'$q_{net}$ at pc_lp  [m$^3$ s$^{-1}$]')
D.set_title('d. The mouth is a tidal oscillator:\nhourly sd is 28x the subtidal sd')
D.tick_params(axis='x', labelrotation=30, labelsize=8)
D.grid(color='lightgray', ls='--', alpha=0.5)
D.legend(fontsize=9)

# e: prism vs volume, and return flow
E = ax['e']
x = np.arange(len(pr))
E.bar(x - 0.2, pr.frac, 0.4, color=COLOR['pc_lp'], label='tidal prism as % of V')
E.bar(x + 0.2, 100 * pr.b, 0.4, color='0.55', label='return flow b [%]')
E.set_xticks(x); E.set_xticklabels(pr.index)
E.set_ylabel('percent')
E.set_title('e. Small prism, high return flow:\nonly ~half of an ebb is really lost')
E.grid(color='lightgray', ls='--', alpha=0.5, axis='y')
E.legend(fontsize=9)

fn = out_dir / '20260805_penncove_exchange_synthesis.png'
fig.savefig(fn, dpi=200, bbox_inches='tight')
print('\nsaved %s' % fn)
