"""
Does the Penn Cove exchange flow ever run backwards? (wb1_t0_xn11abbur00)

Follow-on to 20260818_gm2014_parameter_space.py. "Reverse exchange flow" is
ambiguous at Penn Cove, so this separates three distinct questions:

 1. VERTICAL mode -- is the subtidal flow deep-in / surface-out (normal
    estuarine) or surface-in / deep-out (reversed)?
 2. LATERAL mode  -- the mouth exchange is mostly a horizontal gyre, in along
    the north shore and out along the south (see reduce_wind_cove.py and the
    pc_lp mouth-point work). Does that ever flip?
 3. The BUOYANCY CONFIGURATION -- Penn Cove has no river, and the freshwater
    (Skagit) arrives from OUTSIDE, in the surface of Saratoga Passage. That
    puts the buoyancy source at the mouth rather than the head, which is
    geometrically an inverted estuary and should reverse the sign of the
    surface density gradient. Tested here as surface salinity of the outer
    cove minus upper Saratoga Passage.

Note this is NOT the evaporative inverse-estuary case: EminusP is identically
zero in every segment of this run, so that mechanism cannot operate at all.

Indices (both in m3/s, positive = normal sense, negative = reversed):
    Iv = 0.5 * (Q_upper - Q_lower)      split at z index 14 of 30
    Il = 0.5 * (Q_south - Q_north)      split between faces 5 and 6
built from Godin-filtered hourly q(t, z, p). Filtering the hourly transport
directly (not <u><dz>) matters -- see the Stokes-transport note.

Sign convention: positive q at pc_lp is OUT of the cove, z index 0 is the
BOTTOM, faces 0-5 are the north side.

Run:
    python 20260818_pc_exchange_reversal.py
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr
import gsw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('--gtx', default='wb1_t0_xn11abbur00')
p.add_argument('--coll', default='wb1_pc1')
p.add_argument('--dates', default='2024.01.01_2025.12.31')
p.add_argument('--sect', default='pc_lp')
p.add_argument('--zsplit', default=14, type=int)
p.add_argument('--psplit', default=6, type=int)
args = p.parse_args()

Ldir = Lfun.Lstart()
in_dir = Ldir['LOo'] / 'extract' / args.gtx / 'tef2'
out_dir = Ldir['LOo'] / 'DM_outs' / '20260818_pc_exchange_reversal'
Lfun.make_dir(out_dir)

# ---- subtidal transport structure at the mouth ----------------------------
ex = xr.open_dataset(in_dir / ('extractions_avg_' + args.dates) / (args.sect + '.nc'))
t = pd.to_datetime(ex.time.values)
q_sub = zfun.lowpass(ex.q.values, f='godin')      # (time, z, p)

qz = np.nansum(q_sub, axis=2)                     # vertical structure (time, z)
qp = np.nansum(q_sub, axis=1)                     # lateral structure  (time, p)

Iv = 0.5 * (qz[:, args.zsplit:].sum(axis=1) - qz[:, :args.zsplit].sum(axis=1))
Il = 0.5 * (qp[:, args.psplit:].sum(axis=1) - qp[:, :args.psplit].sum(axis=1))

S = pd.DataFrame({'Iv': Iv, 'Il': Il}, index=t).resample('1D').mean()

# ---- candidate drivers ----------------------------------------------------
bulk = xr.open_dataset(in_dir / ('bulk_avg_' + args.dates) / (args.sect + '.nc'))
S['qprism'] = pd.Series(bulk.qprism.values,
                        index=pd.to_datetime(bulk.time.values).floor('D')).reindex(S.index)

wd = xr.open_dataset(in_dir / ('wind_%s_%s.nc' % (args.dates, args.coll)))
wt = pd.to_datetime(wd.day.values)
# w_along positive blows INTO the cove (mouth -> head).
# w_cross is MISLABELLED in reduce_wind_cove.py: positive is SOUTHWARD.
S['w_along'] = pd.Series(wd.w_along.values, index=wt).reindex(S.index)
S['w_cross'] = pd.Series(wd.w_cross.values, index=wt).reindex(S.index)

seg = xr.open_dataset(in_dir / ('segments_%s_%s_trapsN00.nc' % (args.dates, args.coll)))
gt = pd.to_datetime(seg.time.values)
segs = list(seg.seg.values)
vol = seg.volume.values
sig = gsw.sigma0(seg.salt.values, seg.temp.values)
i_cove = [segs.index(x) for x in ['pc_lp_m', 'pc_cp_m', 'pc_cp_p']]
i_pas = segs.index('pc_lp_p')

# Volume-mean density difference. NOTE this is depth-confounded: pc_lp_p is a
# 3.3 km3 segment carrying deep water the shallow cove simply does not have,
# so a negative value does not by itself mean the cove is buoyant.
sig_cove = (sig[:, i_cove] * vol[:, i_cove]).sum(axis=1) / vol[:, i_cove].sum(axis=1)
S['dsig'] = pd.Series(sig_cove - sig[:, i_pas],
                      index=gt).resample('1D').mean().reindex(S.index)

# Surface salinity contrast: positive = cove surface SALTIER than the passage
# surface = the freshwater is outside the mouth = inverted buoyancy geometry.
ss = seg.salt_surf.values
S['ds_surf'] = pd.Series(ss[:, segs.index('pc_lp_m')] - ss[:, i_pas],
                         index=gt).resample('1D').mean().reindex(S.index)
S['EminusP'] = pd.Series(seg.EminusP.values[:, i_cove].sum(axis=1),
                         index=gt).resample('1D').mean().reindex(S.index)

S = S.dropna()
S.to_csv(out_dir / 'pc_exchange_reversal_daily.csv')

# ---- report ---------------------------------------------------------------
rev = S.Iv < 0
print('\n=== Penn Cove exchange reversal, %s (%s) ===' % (args.gtx, args.sect))
print('n days = %d' % len(S))
for nm, v in [('VERTICAL Iv (deep-in / surf-out)', S.Iv),
              ('LATERAL  Il (north-in / south-out)', S.Il)]:
    print('  %-36s mean %7.1f  std %6.1f  reversed %5.1f%% of days'
          % (nm, v.mean(), v.std(), 100 * (v < 0).mean()))
print('  corr(Iv, Il) = %+.2f  (daily)' % S.Iv.corr(S.Il))

print('\nE-P summed over the cove segments: mean %.3g  '
      '-> evaporative inverse estuary is impossible in this run'
      % S.EminusP.mean())

print('\nBuoyancy geometry -- surface salinity, outer cove MINUS upper Saratoga:')
print('  mean %+.3f g/kg, cove surface saltier on %.0f%% of days'
      % (S.ds_surf.mean(), 100 * (S.ds_surf > 0).mean()))
print('  monthly: ' + ' '.join('%.2f' % x for x in S.groupby(S.index.month).ds_surf.mean()))

print('\nLagged correlation with Iv (driver leads by L days):')
for v in ['w_along', 'w_cross', 'qprism', 'ds_surf', 'dsig']:
    print('  %-8s ' % v + '  '.join('L%d %+.2f' % (L, S.Iv.corr(S[v].shift(L)))
                                    for L in range(5)))

print('\nComposite, reversed vs normal days:')
print(S.groupby(rev)[['w_along', 'w_cross', 'qprism', 'ds_surf']].mean().round(3).to_string())

runs = []
c = 0
for r in rev.values:
    if r:
        c += 1
    elif c:
        runs.append(c)
        c = 0
if c:
    runs.append(c)
runs = np.array(runs)
print('\nReversal events: %d, %d days total, median %.0f d, max %d d'
      % (len(runs), runs.sum(), np.median(runs), runs.max()))

# ---- figure ---------------------------------------------------------------
fig = plt.figure(figsize=(13.5, 8))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], hspace=0.55, wspace=0.32)
a0 = fig.add_subplot(gs[0, 0])
a1 = fig.add_subplot(gs[0, 1])
a2 = fig.add_subplot(gs[0, 2])
a3 = fig.add_subplot(gs[1, :])
a4 = fig.add_subplot(gs[2, 0])
a5 = fig.add_subplot(gs[2, 1])
a6 = fig.add_subplot(gs[2, 2])

# (a) mean vertical mode
zi = np.arange(qz.shape[1])
qzm = np.nanmean(qz, axis=0)
a0.barh(zi, qzm, color=np.where(qzm < 0, 'tab:blue', 'tab:red'))
a0.axvline(0, color='k', lw=0.8)
a0.axhline(args.zsplit - 0.5, color='0.4', ls='--', lw=1)
a0.set_ylabel('z index (0 = bottom)')
a0.set_xlabel('mean subtidal q (m$^3$/s)')
a0.set_title('vertical mode\nblue = INTO cove', fontsize=10)

# (b) mean lateral mode
pi = np.arange(qp.shape[1])
qpm = np.nanmean(qp, axis=0)
a1.bar(pi, qpm, color=np.where(qpm < 0, 'tab:blue', 'tab:red'))
a1.axhline(0, color='k', lw=0.8)
a1.axvline(args.psplit - 0.5, color='0.4', ls='--', lw=1)
a1.set_xlabel('face index (0 = north)')
a1.set_ylabel('mean subtidal q (m$^3$/s)')
a1.set_title('lateral mode\n(dominant, 4x larger)', fontsize=10)

# (c) distributions
a2.hist(S.Iv, bins=40, color='tab:purple', alpha=0.75, label='$I_v$ vertical')
a2.hist(S.Il, bins=40, color='tab:green', alpha=0.55, label='$I_l$ lateral')
a2.axvline(0, color='k', lw=1.2)
a2.set_xlabel('exchange index (m$^3$/s)')
a2.set_ylabel('days')
a2.set_title('negative = reversed', fontsize=10)
a2.legend(fontsize=8)

# (d) time series
a3.fill_between(S.index, 0, 1, where=rev, transform=a3.get_xaxis_transform(),
                color='crimson', alpha=0.16, lw=0)
a3.plot(S.index, S.Il, color='tab:green', lw=1.0, label='$I_l$ lateral')
a3.plot(S.index, S.Iv, color='tab:purple', lw=1.0, label='$I_v$ vertical')
a3.axhline(0, color='k', lw=1.0)
a3.set_ylabel('m$^3$/s')
a3.set_title('subtidal exchange at %s; shading = vertically reversed days '
             '(%.0f%%)' % (args.sect, 100 * rev.mean()), fontsize=10)
a3.legend(fontsize=8, ncol=2)
a3.grid(True, alpha=0.25, lw=0.5)

# (e) wind is the driver
a4.scatter(S.w_along, S.Iv, s=9, c=np.where(rev, 'crimson', '0.5'), alpha=0.7)
a4.axhline(0, color='k', lw=0.8)
a4.axvline(0, color='k', lw=0.8)
a4.set_xlabel('w_along (m/s, + = INTO cove)')
a4.set_ylabel('$I_v$ (m$^3$/s)')
a4.set_title('r = %+.2f at lag 0' % S.Iv.corr(S.w_along), fontsize=10)
a4.grid(True, alpha=0.25, lw=0.5)

# (f) buoyancy geometry is NOT the driver
a5.scatter(S.ds_surf, S.Iv, s=9, c=np.where(rev, 'crimson', '0.5'), alpha=0.7)
a5.axhline(0, color='k', lw=0.8)
a5.axvline(0, color='k', lw=0.8)
a5.set_xlabel(r'$s_{surf}$ cove $-$ passage (g/kg)')
a5.set_ylabel('$I_v$ (m$^3$/s)')
a5.set_title('inverted buoyancy geometry,\nbut r = %+.2f' % S.Iv.corr(S.ds_surf),
             fontsize=10)
a5.grid(True, alpha=0.25, lw=0.5)

# (g) seasonality
m_ds = S.groupby(S.index.month).ds_surf.mean()
m_iv = S.groupby(S.index.month).Iv.mean()
a6.bar(m_ds.index, m_ds.values, color='tab:orange', alpha=0.8)
a6.set_xlabel('month')
a6.set_ylabel(r'$s_{surf}$ cove $-$ passage', color='tab:orange')
a6.tick_params(axis='y', labelcolor='tab:orange')
a62 = a6.twinx()
a62.plot(m_iv.index, m_iv.values, 'o-', color='tab:purple')
a62.set_ylabel('$I_v$ (m$^3$/s)', color='tab:purple')
a62.tick_params(axis='y', labelcolor='tab:purple')
a62.set_ylim(0, None)
a6.set_title('freshet peaks the inverted gradient\nAND the normal exchange',
             fontsize=9.5)
a6.set_xticks(range(1, 13))

fig.suptitle('Does the Penn Cove exchange reverse?  %s' % args.gtx, fontsize=12)
fig.savefig(out_dir / 'pc_exchange_reversal.png', dpi=200,
            bbox_inches='tight', transparent=True)
plt.close(fig)
print('\nsaved %s' % (out_dir / 'pc_exchange_reversal.png'))
