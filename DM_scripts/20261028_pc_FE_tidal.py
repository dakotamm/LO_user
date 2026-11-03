"""
How the Eulerian exchange at the Penn Cove mouth (pc_lp) changes with the tidal
cycles: spring/neap (D2), tropic/equatorial (D1) and month.

Input (no new extraction):
  DM_outs/20261013_maccready2011_sec2_avg/pc_lp_sec2.nc
      MacCready (2011) Sec 2 / Lerczak et al. (2006) decomposition from hourly
      AVERAGES: u1(t, k, p) = tidally averaged departure of each element from the
      section mean (elements = 30 s-levels x 12 faces, stretching with the
      tide), F_E = int u1 s1 dA0 (steady exchange salt flux), F_T =
      < int u2 s2 dA > (tidal salt flux). Positive = INTO the cove. p = 0 is
      the north end.
  DM_outs/20261020_pc_do_tef_avg/pc_lp_do_tef.nc
      the same for oxygen: G_E (steady exchange), G_T (tidal), stored in
      mmol/s and converted here to g O2/s.
  tidal_ellipses/pc_cove_*_bands.nc   daily spring/neap, tropic/equatorial flags

Derived from u1: the exchange INFLOW Q_Ein = sum of u1 dA0 over elements with
u1 > 0, and the height above the bed and the north-south position of its
transport-weighted centre -- where the water that comes in actually comes in,
defined by the flow rather than by a fixed bottom layer.

Daily means of Godin-valid hours; bins = days flagged in the band fits.

Output: LO_output/DM_outs/20261028_pc_FE_tidal/pc_FE_tidal.png, report.txt

  python 20261028_pc_FE_tidal.py
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cmocean

from lo_tools import Lfun

GTX, JOB, DS0, DS1 = 'wb1_t0_xn11abbur00', 'pc_cove', '2024.01.01', '2025.12.31'
Ldir = Lfun.Lstart(gridname='wb1')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261028_pc_FE_tidal'
Lfun.make_dir(out_dir)
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
C_E, C_T = '#0072B2', '#D55E00'
BINS = ['neap', 'spring', 'equatorial', 'tropic']

S2 = xr.open_dataset(Ldir['LOo'] / 'DM_outs' / '20261013_maccready2011_sec2_avg' / 'pc_lp_sec2.nc')
DO = xr.open_dataset(Ldir['LOo'] / 'DM_outs' / '20261020_pc_do_tef_avg' / 'pc_lp_do_tef.nc')
t = pd.DatetimeIndex(S2.time.values)
ok = S2.ok.values.astype(bool)
u1, dA0 = S2.u1.values, S2.dA0.values # (T, k, p)
z0, h = S2.z0_mean.values, S2.h.values # (k, p), (p)
hab = z0 + h[None, :] # height above bed of each element
NK, NP = z0.shape
Q = np.where(u1 > 0, u1 * dA0, 0.0)
Qin = Q.sum((1, 2))
hab_in = (Q * hab[None]).sum((1, 2)) / Qin
p_in = (Q * np.arange(NP)[None, None, :]).sum((1, 2)) / Qin
H = pd.DataFrame({'F_E': S2.F_E.values, 'F_T': S2.F_T.values, 'Qin': Qin, 'hab_in': hab_in, 'p_in': p_in}, index=t)[ok]
tO = pd.DatetimeIndex(DO.time.values)
okO = DO.ok.values.astype(bool)
MMOL_TO_G = 0.032 # pc_lp_do_tef.nc carries oxygen fluxes in mmol/s (DO in mmol m-3); 32 mg/mmol
G = pd.DataFrame({'G_E': MMOL_TO_G * DO.G_E.values, 'G_T': MMOL_TO_G * DO.G_T.values}, index=tO)[okO]
D = H.resample('D').mean().join(G.resample('D').mean(), how='outer')
BF = xr.open_dataset(Ldir['LOo'] / 'extract' / GTX / 'tidal_ellipses' / ('%s_%s_%s_bands.nc' % (JOB, DS0, DS1)))
F = pd.DataFrame({nm: BF['is_' + nm].values for nm in BINS}, index=pd.DatetimeIndex(BF.time.values).normalize())
D = D.join(F, how='inner')
U1d = pd.DataFrame(u1[ok].reshape(ok.sum(), -1), index=t[ok]).resample('D').mean()

rep = ['pc_lp, MacCready 2011 / Lerczak 2006 decomposition from hourly averages; + = INTO the cove; %d days' % len(D)]
rep.append('2-yr mean: F_E %+.0f, F_T %+.0f g/kg m3/s (F_T/F_E %.2f: they nearly cancel); G_E %+.0f, G_T %+.0f g O2/s; Q_Ein %.0f m3/s centred %.1f m above the bed, face %.1f (0 = north)'
           % (D.F_E.mean(), D.F_T.mean(), D.F_T.mean() / D.F_E.mean(), D.G_E.mean(), D.G_T.mean(), D.Qin.mean(), D.hab_in.mean(), D.p_in.mean()))
B = {}
for b in BINS:
    B[b] = D[D[b] == 1].mean()
    rep.append('%-10s F_E %+5.0f F_T %+5.0f | G_E %+5.0f G_T %+5.0f g/s | Q_Ein %4.0f m3/s at %.1f m above bed, face %.1f'
               % (b, B[b].F_E, B[b].F_T, B[b].G_E, B[b].G_T, B[b].Qin, B[b].hab_in, B[b].p_in))
Mo = D[['F_E', 'F_T', 'G_E', 'G_T', 'Qin', 'hab_in', 'p_in']].resample('MS').mean()
rep.append('monthly: corr(F_E, F_T) %+.2f; Q_Ein range %.0f-%.0f m3/s; inflow centre %.1f-%.1f m above bed'
           % (Mo.F_E.corr(Mo.F_T), Mo.Qin.min(), Mo.Qin.max(), Mo.hab_in.min(), Mo.hab_in.max()))

fig = plt.figure(figsize=(17, 10), layout='constrained')
gs = fig.add_gridspec(2, 4)
yy = np.broadcast_to(np.arange(NP)[None, :], (NK, NP))
umax = 0
comps = {}
for b in BINS:
    days = D.index[D[b] == 1]
    comps[b] = U1d.reindex(days).mean().values.reshape(NK, NP)
    umax = max(umax, np.nanmax(np.abs(comps[b])))
for n, b in enumerate(BINS):
    ax = fig.add_subplot(gs[0, n])
    cs = ax.pcolormesh(yy, z0, 100 * comps[b], cmap=cmocean.cm.balance_r, vmin=-100 * umax, vmax=100 * umax, shading='nearest')
    ax.contour(yy, z0, comps[b], levels=[0], colors='k', linewidths=1)
    ax.plot(B[b].p_in, B[b].hab_in - np.interp(B[b].p_in, np.arange(NP), h), 'k*', ms=14, mec='white')
    ax.set_title('%s: Q_Ein %.0f m3/s' % (b, B[b].Qin), fontsize=10, loc='left')
    ax.set_xlabel('face (0 = north)  ->  south')
    if n == 0:
        ax.set_ylabel('depth at mean sea level [m]')
    ax.grid(**GRID)
fig.colorbar(cs, ax=fig.axes[:4], shrink=0.8, label='u1, tidally averaged exchange velocity [cm/s] (blue = IN)')

ax = fig.add_subplot(gs[1, 0])
x = np.arange(len(BINS))
ax.bar(x - 0.18, [B[b].F_E for b in BINS], 0.36, color=C_E, label='F_E (steady exchange)')
ax.bar(x + 0.18, [B[b].F_T for b in BINS], 0.36, color=C_T, label='F_T (tidal)')
ax.axhline(0, color='0.3', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(BINS)
ax.set_ylabel('salt flux into the cove [g/kg m3/s]')
ax.set_title('salt: steady vs tidal exchange by tidal phase', fontsize=10, loc='left')
ax.legend(fontsize=8); ax.grid(axis='y', **GRID)

ax = fig.add_subplot(gs[1, 1])
ax.bar(x - 0.18, [B[b].G_E for b in BINS], 0.36, color=C_E, label='G_E (steady exchange)')
ax.bar(x + 0.18, [B[b].G_T for b in BINS], 0.36, color=C_T, label='G_T (tidal)')
ax.axhline(0, color='0.3', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(BINS)
ax.set_ylabel('oxygen flux into the cove [g O2/s]')
ax.set_title('oxygen: steady exchange exports, tides import', fontsize=10, loc='left')
ax.legend(fontsize=8); ax.grid(axis='y', **GRID)

ax = fig.add_subplot(gs[1, 2:])
tm = Mo.index + pd.Timedelta(days=14)
ax.plot(tm, Mo.F_E, '-o', ms=4, lw=2, color=C_E, label='F_E salt (steady exchange)')
ax.plot(tm, Mo.F_T, '-o', ms=4, lw=2, color=C_T, label='F_T salt (tidal)')
ax.plot(tm, Mo.F_E + Mo.F_T, '-', lw=1.5, color='k', label='F_E + F_T')
ax.axhline(0, color='0.3', lw=0.8)
ax.set_ylabel('salt flux into the cove [g/kg m3/s]')
ax.set_title('by month; inflow core %.1f-%.1f m above the bed (stars above mark it in each section)' % (Mo.hab_in.min(), Mo.hab_in.max()), fontsize=10, loc='left')
ax.legend(fontsize=8, ncol=3, loc='upper left'); ax.grid(**GRID)
fig.suptitle('Eulerian exchange at the Penn Cove mouth (pc_lp) through the tidal cycles: where the exchange inflow sits and how steady vs tidal fluxes split, %s' % GTX, fontsize=11)
fig.savefig(out_dir / 'pc_FE_tidal.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
(out_dir / 'report.txt').write_text('\n'.join(rep) + '\n')
print('\n'.join(rep))
