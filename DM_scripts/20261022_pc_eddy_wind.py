"""
How does the inner-cove eddy's presence (and strength) vary with wind?

EDDY: hourly present flag and core circulation from 20261021_pc_eddy_center.py
(band -band; Godin-filtered field, so the eddy series is itself smooth over
~1.5 d), averaged to days on a fixed UTC-8 clock (to match the wind file).
WIND: daily cove wind from 20260806_wind_characterize.py
(DM_outs/20260806_wind/daily_wind.csv: u_pc, v_pc vector means, spd_pc mean
speed; days on UTC-8). Along and cross are RECOMPUTED here because that file's
cross_pc has the wrong sign ([[wcross_sign_bug]]):
  along  = wind toward the cove HEAD (mouth -> head axis (-0.9789, -0.2043),
           i.e. blowing westward into the cove) [m/s]
  cross  = wind toward the NORTH shore (right-hand normal (-0.2043, 0.9789))
  dir    = direction the wind blows FROM [deg true]

Figures (DM_outs/20261022_pc_eddy_wind/):
  eddy_wind_lagcorr_<band>.png  correlation of daily eddy presence (fraction of
      the day present) and strength (circ) with along, cross and speed at lags
      -3..+6 d (+ = eddy AFTER wind), raw and with the seasonal cycle removed
      (anomalies from a 31-d centred running mean of each series)
  eddy_wind_rose_<band>.png  share of days present by the direction the wind
      blew from (8 sectors) x speed class, wind averaged over the -win days
      ending on the eddy day; n days in each box
  eddy_wind_monthly_<band>.png  monthly mean presence vs monthly wind (the
      seasonal co-variation the anomaly correlations remove)
  eddy_wind_<band>.csv  the daily table;  summary_<band>.txt

  python 20261022_pc_eddy_wind.py
  python 20261022_pc_eddy_wind.py -band lower
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun

p = argparse.ArgumentParser()
p.add_argument('-band', default='z4-10')
p.add_argument('-region', default='pc_cp')
p.add_argument('-win', type=int, default=2, help='days of wind averaged for the rose (ending on the eddy day)')
p.add_argument('-maxlag', type=int, default=6)
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261022_pc_eddy_wind'
Lfun.make_dir(out_dir)
AX, AY = -0.9789, -0.2043

C = pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20261021_pc_eddy_center' / args.region /
                ('pc_eddy_center_%s.csv' % args.band), usecols=['time', 'present', 'circ'], parse_dates=['time'])
C['day'] = (C.time - pd.Timedelta('8h')).dt.floor('D')
C['present'] = C.present.astype(float)
E = C.groupby('day').agg(pres=('present', 'mean'), circ=('circ', 'mean'), nh=('present', 'size'))
E = E[E.nh >= 20]

W = pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20260806_wind' / 'daily_wind.csv')
W['day'] = pd.to_datetime(W.date.str[:10])
W = W.set_index('day')
W['along'] = W.u_pc * AX + W.v_pc * AY
W['cross'] = W.u_pc * AY - W.v_pc * AX
W['spd'] = W.spd_pc
W['dir'] = (270 - np.degrees(np.arctan2(W.v_pc, W.u_pc))) % 360
D = E.join(W[['along', 'cross', 'spd', 'dir', 'u_pc', 'v_pc']], how='inner')
D = D.asfreq('D')
D.to_csv(out_dir / ('eddy_wind_%s.csv' % args.band), float_format='%.4g')
print('%d days with both (band %s); presence %.0f%% of hours' % (D.pres.notna().sum(), args.band, 100 * D.pres.mean()))
anom = lambda s: s - s.rolling(31, center=True, min_periods=20).mean()

txt = ['EDDY PRESENCE vs WIND -- band %s, %s; daily, %s to %s' % (args.band, args.region, D.index[0].date(),
                                                                 D.index[-1].date()),
       'along + = toward the cove head (westward into the cove); cross + = toward the north shore', '']
LAG = range(-3, args.maxlag + 1)
LC = {}
for ev in ['pres', 'circ']:
    for kind in ['raw', 'anom']:
        for wv in ['along', 'cross', 'spd']:
            e = D[ev] if kind == 'raw' else anom(D[ev])
            w = D[wv] if kind == 'raw' else anom(D[wv])
            LC[(ev, kind, wv)] = [e.corr(w.shift(L)) for L in LAG]
for ev in ['pres', 'circ']:
    for kind in ['raw', 'anom']:
        txt.append('%s, %s:' % (ev, kind))
        for wv in ['along', 'cross', 'spd']:
            r = np.array(LC[(ev, kind, wv)])
            k = np.nanargmax(np.abs(r))
            txt.append('   %-6s r at lag 0 %+.2f, lag 1 %+.2f; max |r| %+.2f at lag %+d d'
                       % (wv, r[list(LAG).index(0)], r[list(LAG).index(1)], r[k], list(LAG)[k]))

# composites on the wind of the previous -win days
Wm = D[['u_pc', 'v_pc', 'spd']].rolling(args.win, min_periods=args.win).mean()
Wm['dir'] = (270 - np.degrees(np.arctan2(Wm.v_pc, Wm.u_pc))) % 360
Wm['along'] = Wm.u_pc * AX + Wm.v_pc * AY
Wm['cross'] = Wm.u_pc * AY - Wm.v_pc * AX
SEC = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
ks = ((Wm.dir + 22.5) % 360) // 45
Wm['sector'] = pd.Categorical([SEC[int(k)] if np.isfinite(k) else None for k in ks], SEC)
Wm['spdc'] = pd.cut(Wm.spd, [0, 3, 5, 99], labels=['< 3 m/s', '3-5 m/s', '> 5 m/s'])
G = pd.concat([D[['pres', 'circ']], Wm[['sector', 'spdc', 'along', 'cross']]], axis=1).dropna()
G['month'] = G.index.month
T = G.pivot_table(index='spdc', columns='sector', values='pres', aggfunc='mean', observed=False)
N = G.pivot_table(index='spdc', columns='sector', values='pres', aggfunc='size', observed=False)
txt += ['', 'share of days present by direction the wind came FROM (mean of the %d d ending on the day) x speed:'
        % args.win, (100 * T).round(0).to_string(), 'n days:', N.to_string()]
for lab, m in [('Nov-Dec', G.month.isin([11, 12])), ('Aug-Oct', G.month.isin([8, 9, 10]))]:
    s = G[m].groupby('sector', observed=False).pres.agg(['mean', 'size'])
    txt += ['%s by sector: %s' % (lab, ', '.join('%s %.0f%% (%d)' % (k, 100 * v.mean_, v.size_)
                                                   for k, v in s.rename(columns={'mean': 'mean_', 'size': 'size_'})
                                                   .iterrows() if v.size_ > 0))]
(out_dir / ('summary_%s.txt' % args.band)).write_text('\n'.join(txt) + '\n')
print('\n'.join(txt))

# ------------------------------------------------------------- figures ---
CW = {'along': '#2a6fdb', 'cross': '#e8590c', 'spd': '#4d4d4d'}
LS = {'along': '-', 'cross': '--', 'spd': ':'}
fig, axs = plt.subplots(2, 2, figsize=(12, 7.5), layout='constrained', sharex=True, sharey=True)
for i, ev in enumerate(['pres', 'circ']):
    for j, kind in enumerate(['raw', 'anom']):
        ax = axs[i, j]
        for wv in ['along', 'cross', 'spd']:
            ax.plot(list(LAG), LC[(ev, kind, wv)], LS[wv], color=CW[wv], lw=2, marker='o', ms=4,
                    label={'along': 'along-cove wind (+ into cove)', 'cross': 'cross-cove wind (+ northward)',
                           'spd': 'wind speed'}[wv])
        ax.axhline(0, color='0.3', lw=0.8)
        ax.axvline(0, color='0.3', lw=0.8, ls=':')
        ax.set_title('%s, %s' % ({'pres': 'eddy presence (share of day)', 'circ': 'eddy strength (circ)'}[ev],
                                 {'raw': 'raw', 'anom': 'seasonal cycle removed (31-d anomalies)'}[kind]),
                     fontsize=10, loc='left')
        ax.grid(color='0.9', lw=0.6)
for ax in axs[-1]:
    ax.set_xlabel('lag [days]; + = eddy after the wind')
for ax in axs[:, 0]:
    ax.set_ylabel('correlation r')
axs[0, 0].legend(fontsize=8, frameon=False)
fig.suptitle('Daily eddy presence / strength vs daily cove wind (%s band)' % args.band, fontsize=11)
fn = out_dir / ('eddy_wind_lagcorr_%s.png' % args.band)
fig.savefig(fn, dpi=200, transparent=True, bbox_inches='tight')
plt.close(fig)
print('saved ' + str(fn))

fig, axs = plt.subplots(1, 2, figsize=(14, 4.4), layout='constrained', gridspec_kw=dict(width_ratios=[1.35, 1]))
ax = axs[0]
h = ax.imshow(100 * T.values, cmap='Blues', vmin=0, vmax=100, aspect='auto')
for (r, c), v in np.ndenumerate(T.values):
    n = N.values[r, c]
    if n > 0:
        ax.text(c, r, '%.0f%%\nn=%d' % (100 * v, n), ha='center', va='center', fontsize=8,
                color='white' if v > 0.6 else 'black')
ax.set_xticks(range(len(SEC)), SEC)
ax.set_yticks(range(len(T.index)), T.index)
ax.set_xlabel('direction the wind blew FROM (mean of the %d d ending on the day)' % args.win)
ax.set_title('(a) share of days with the eddy present', fontsize=10, loc='left')
fig.colorbar(h, ax=ax, label='% of the day present', shrink=0.8)
ax = axs[1]
sc = ax.scatter(G.cross, G.along, c=100 * G.pres, cmap='Blues', vmin=0, vmax=100, s=14, ec='0.5', lw=0.2)
ax.axhline(0, color='0.3', lw=0.8); ax.axvline(0, color='0.3', lw=0.8)
ax.set_xlabel('cross-cove wind [m/s] (+ toward the north shore)')
ax.set_ylabel('along-cove wind [m/s] (+ into the cove)')
ax.set_title('(b) each day, wind (%d-d mean) coloured by presence' % args.win, fontsize=10, loc='left')
ax.set_aspect('equal')
ax.grid(color='0.9', lw=0.6)
fig.colorbar(sc, ax=ax, label='% of the day present', shrink=0.8)
fig.suptitle('Eddy presence by wind direction and speed (%s band)' % args.band, fontsize=11)
fn = out_dir / ('eddy_wind_rose_%s.png' % args.band)
fig.savefig(fn, dpi=200, transparent=True, bbox_inches='tight')
plt.close(fig)
print('saved ' + str(fn))

M = D.groupby(D.index.month).agg(pres=('pres', 'mean'), along=('along', 'mean'), cross=('cross', 'mean'),
                                  spd=('spd', 'mean'))
fig, axs = plt.subplots(2, 1, figsize=(9, 6), layout='constrained', sharex=True)
axs[0].bar(M.index, 100 * M.pres, color='#2a6fdb')
axs[0].set_ylabel('% of hours present')
axs[0].set_title('(a) eddy presence by month (2024-25)', fontsize=10, loc='left')
for wv in ['along', 'cross', 'spd']:
    axs[1].plot(M.index, M[wv], LS[wv], color=CW[wv], lw=2, marker='o',
                label={'along': 'along (+ into cove)', 'cross': 'cross (+ northward)', 'spd': 'mean speed'}[wv])
axs[1].axhline(0, color='0.3', lw=0.8)
axs[1].set_ylabel('wind [m/s]')
axs[1].set_xticks(range(1, 13))
axs[1].set_xlabel('month')
axs[1].legend(fontsize=8, frameon=False)
axs[1].set_title('(b) cove wind by month', fontsize=10, loc='left')
for ax in axs:
    ax.grid(color='0.9', lw=0.6)
fn = out_dir / ('eddy_wind_monthly_%s.png' % args.band)
fig.savefig(fn, dpi=200, transparent=True, bbox_inches='tight')
plt.close(fig)
print('saved ' + str(fn))
