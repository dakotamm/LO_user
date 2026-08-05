"""
Does the Penn Cove lateral gyre have the same seasonal cycle as the exchange?

The residual circulation across the cove sections turned out to be a lateral
gyre -- in along the north shore, out along the south shore, at all depths.
This asks whether it strengthens and weakens with the season, and whether that
matches the seasonal cycle already found in TEF Qin and in the along-cove
salinity gradient.

Gyre strength is defined as the sum of the positive part of the lateral
profile, i.e. the total inflowing limb, which by construction equals minus the
outflowing limb since the net is zero.

All fluxes are m3 s-1, so gyre strength and Qin share one axis honestly.

run 20260805_plot_gyre_seasonality.py
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
st = xr.open_dataset(tef2 / ('structure_' + ds0 + '_' + ds1 + '_wb1_pc1.nc'))
bulk_dir = tef2 / ('bulk_avg_' + ds0 + '_' + ds1)

SECTS = ['pc_cp', 'pc_lj', 'pc_lp']
COLOR = {'pc_cp': '#009E73', 'pc_lj': '#0072B2', 'pc_lp': '#CC79A7'}
MON = list('JFMAMJJASOND')

# monthly gyre strength, and the monthly TEF Qin for comparison
gyre = {}
Qin_m = {}
Sm = {}
for sn in SECTS:
    qm = st[sn + '_qmon'].values                 # (month, z, p)
    pp = -np.nansum(qm, axis=1)                  # (month, p), + = into cove
    gyre[sn] = np.array([np.nansum(r[r > 0]) for r in pp])
    b = xr.open_dataset(bulk_dir / (sn + '.nc'))
    q = b.q.values
    sa = b.salt.values
    t = pd.to_datetime(b.time.values)
    qi = pd.Series(np.nansum(np.where(q > 0, q, 0), axis=1), index=t)
    Qin_m[sn] = qi.groupby(qi.index.month).mean().values
    w = np.abs(q)
    ok = np.isfinite(sa)
    s = pd.Series(np.nansum(np.where(ok, w * sa, 0), axis=1)
                  / np.nansum(np.where(ok, w, 0), axis=1), index=t)
    Sm[sn] = s.groupby(s.index.month).mean().values
    b.close()

dS = Sm['pc_lp'] - Sm['pc_cp']          # cove fresher when positive

print('%-8s %s' % ('sect', '  '.join('%5s' % m for m in MON)))
for sn in SECTS:
    print('  %-8s %s' % (sn, '  '.join('%5.0f' % v for v in gyre[sn])))
print('\ncorrelations of monthly gyre strength:')
for sn in SECTS:
    print('  %-8s vs Qin r=%+.3f   vs dS r=%+.3f'
          % (sn, np.corrcoef(gyre[sn], Qin_m[sn])[0, 1],
             np.corrcoef(gyre[sn], dS)[0, 1]))

# ----------------------------------------------------------------- figure ---
out_dir = Path.home() / 'Desktop' / 'pltz'
Lfun.make_dir(out_dir)
plt.close('all')
fig, ax = plt.subplot_mosaic([['ts', 'sc'], ['win', 'sum']], figsize=(14, 10),
                             layout='constrained')

A = ax['ts']
for sn in SECTS:
    A.plot(range(1, 13), gyre[sn], '-o', lw=2.5, ms=6, color=COLOR[sn],
           label=sn + ' gyre')
A.plot(range(1, 13), Qin_m['pc_lp'], '--s', lw=2, ms=5, color='0.35',
       label='pc_lp TEF Qin')
A.set_xticks(range(1, 13)); A.set_xticklabels(MON)
A.set_ylabel('transport [m$^3$ s$^{-1}$]')
A.set_title('a. Gyre strength by month, with TEF Qin for scale')
A.grid(color='lightgray', ls='--', alpha=0.5)
A.legend(fontsize=9, ncol=2)

B = ax['sc']
for sn in SECTS:
    B.scatter(dS, gyre[sn], s=70, color=COLOR[sn], edgecolor='k', lw=0.4,
              label='%s (r=%+.2f)' % (sn, np.corrcoef(gyre[sn], dS)[0, 1]))
for m in range(12):
    B.annotate(MON[m], (dS[m], gyre['pc_lp'][m]), fontsize=8,
               xytext=(4, 3), textcoords='offset points')
B.set_xlabel(r'$\Delta S$ = s(pc_lp) - s(pc_cp)  [cove fresher $\rightarrow$]')
B.set_ylabel('gyre strength [m$^3$ s$^{-1}$]')
B.set_title('b. Gyre against the along-cove gradient')
B.grid(color='lightgray', ls='--', alpha=0.5)
B.legend(fontsize=8)

# winter and summer section views at the mouth
qm = st['pc_lp_qmon'].values
lat = st['pc_lp_lat'].values
dist = (lat - lat.min()) * 111.0
vmax = np.nanpercentile(np.abs(qm), 99)
for key, mo, lab in (('win', 12, 'December'), ('sum', 9, 'September')):
    C = ax[key]
    cs = C.pcolormesh(dist, np.arange(qm.shape[1]), -qm[mo - 1], cmap='RdBu_r',
                      vmin=-vmax, vmax=vmax, shading='nearest')
    plt.colorbar(cs, ax=C, label='m$^3$ s$^{-1}$ per cell, red = in')
    C.set_xlabel('distance across cove from south [km]')
    C.set_ylabel('sigma level (0 = bed)')
    C.set_title('%s. pc_lp, %s  (gyre = %.0f m$^3$ s$^{-1}$)'
                % ('c' if key == 'win' else 'd', lab, gyre['pc_lp'][mo - 1]))

fn_out = out_dir / '20260805_gyre_seasonality.png'
fig.savefig(fn_out, dpi=200, bbox_inches='tight')
print('\nsaved %s' % fn_out)
