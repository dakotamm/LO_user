"""
Plots that explain the once-a-day SEAM in the wb1 wind forcing (atm00).

Each day's forcing file is built from that day's own WRF forecast, hours
f00-f24, and the run uses each file's own 24 hours. So every 00 UTC
(16:00 PST / 17:00 PDT) the wind switches from the end of yesterday's forecast
(f23) to the first hour of a freshly initialised one (f00). The two forecasts
disagree, so the wind jumps. Measured in 20260806_wind_characterize.py from the
raw files on apogee: the f24 and f00 fields for the same valid time differ by
~2.9 m/s (mean vector, over water).

  a  five example days of hourly wind (Penn Cove and domain mean, along-cove
     component) with 00 UTC marked
  b  mean size of the hour-to-hour wind change by UTC hour (2 years)
  c  spike or step? the change INTO an hour against the change OUT of it:
     at 00 UTC they cancel (the f00 hour is an outlier that snaps back), at
     other hours they are unrelated
  d  does the ocean feel it? mean size of the hour-to-hour change of the
     surface-layer and bottom-layer current, cove mean, by UTC hour. Tides are
     not locked to the clock (except S2, a smooth 12 h cycle), so over two years
     a spike at one UTC hour would be forcing. There is none worth the name:
     a one-hour blip is too short to move the cove.

Inputs: 20260806_wind/wind_hourly_atm00_*.p (hourly region-mean forcing wind),
pc_cove box (hourly his u, v).
Output: LO_output/DM_outs/20261027_wind_seam/wind_seam.png

  python 20261027_wind_seam_plots.py
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun

Ldir = Lfun.Lstart(gridname='wb1')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261027_wind_seam'
Lfun.make_dir(out_dir)
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
AX, AY = -0.9789, -0.2043 # cove axis, mouth -> head
C_PC, C_DOM, C_SEAM = '#0072B2', '#E69F00', '#D55E00'

W = pd.read_pickle(Ldir['LOo'] / 'DM_outs' / '20260806_wind' / 'wind_hourly_atm00_2024.01.01_2025.12.31.p')['W'].asfreq('h')
W['along_pc'] = W.u_pc * AX + W.v_pc * AY
W['along_dom'] = W.u_domain * AX + W.v_domain * AY
dW = {k: np.hypot(W['u_' + k].diff(), W['v_' + k].diff()) for k in ['pc', 'domain']} # vector change, m/s per hour
hr = W.index.hour

box = xr.open_dataset(Ldir['LOo'] / 'extract' / 'wb1_t0_xn11abbur00' / 'box' / 'pc_cove_2024.01.01_2025.12.31.nc')
mask = box.mask_rho.values == 1
tt = pd.DatetimeIndex(box.ocean_time.values)
dU = {}
for nm, k in [('surface layer', -1), ('bottom layer', 0)]:
    u = np.nan_to_num(box.u[:, k].values)
    v = np.nan_to_num(box.v[:, k].values)
    ur = 0.5 * (u[:, 1:-1, :-1] + u[:, 1:-1, 1:]) # interior rho
    vr = 0.5 * (v[:, :-1, 1:-1] + v[:, 1:, 1:-1])
    m = mask[1:-1, 1:-1]
    du = np.hypot(np.diff(ur, axis=0), np.diff(vr, axis=0))[:, m].mean(axis=1)
    dU[nm] = pd.Series(du, index=tt[1:])
box.close()

fig, axd = plt.subplot_mosaic([['a', 'a'], ['b', 'c'], ['d', 'd']], layout='constrained', figsize=(13, 12))

ax = axd['a']
t0, t1 = '2024-07-08', '2024-07-13'
for k, c, lab in [('along_pc', C_PC, 'Penn Cove'), ('along_dom', C_DOM, 'whole wb1 domain')]:
    ax.plot(W[t0:t1].index, W[t0:t1][k], '-o', ms=2.5, color=c, lw=1.5, label=lab)
for d in pd.date_range(t0, t1, freq='D'):
    ax.axvline(d, color=C_SEAM, lw=1.2, ls='--')
ax.axhline(0, color='0.4', lw=0.8)
ax.set_ylabel('along-cove wind [m/s]  (+ into the cove)')
ax.set_title('a  five days of hourly forcing wind; dashed = 00 UTC, when each day\'s file starts a new WRF forecast', loc='left', fontsize=10)
ax.legend(fontsize=8, loc='upper right')
ax.grid(**GRID)

ax = axd['b']
for k, c, lab, off in [('pc', C_PC, 'Penn Cove', -0.2), ('domain', C_DOM, 'whole domain', 0.2)]:
    g = dW[k].groupby(hr).mean()
    ax.bar(g.index + off, g.values, 0.4, color=[C_SEAM if h == 0 else c for h in g.index], label=lab)
    print('%s: |dW| into 00 UTC %.2f m/s, other hours %.2f m/s (x%.1f)' % (lab, g[0], g.drop(0).mean(), g[0] / g.drop(0).mean()))
ax.set_xticks(range(0, 24, 3))
ax.set_xlabel('UTC hour (00 = 16:00 PST / 17:00 PDT)')
ax.set_ylabel('mean size of hour-to-hour wind change [m/s]')
ax.set_title('b  the change into 00 UTC (orange) is ~2x any other hour', loc='left', fontsize=10)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_PC), plt.Rectangle((0, 0), 1, 1, color=C_DOM)], labels=['Penn Cove', 'whole domain'], fontsize=8, loc='upper right')
ax.grid(axis='y', **GRID)

ax = axd['c']
din, dout = W.along_pc.diff(), -W.along_pc.diff(-1) # change into the hour, change out of it
for H, col, lab in [(12, C_PC, '12 UTC (any ordinary hour)'), (0, C_SEAM, '00 UTC (new forecast)')]:
    m = hr == H
    ax.plot(din[m], dout[m], '.', ms=3, color=col, alpha=0.6, label='%s: r = %+.2f' % (lab, din[m].corr(dout[m])))
ax.plot([-8, 8], [8, -8], color='0.4', lw=0.8, ls='--')
ax.text(4.3, -6.2, 'spike: jumps,\nthen snaps back', fontsize=8, color='0.3')
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.set_xlabel('wind change INTO the hour [m/s]')
ax.set_ylabel('wind change OUT of the hour [m/s]')
ax.set_title('c  a one-hour spike, not a step: the 00 UTC wind is an outlier\n(%.1f m/s from its neighbours vs %.1f at other hours)'
             % ((W.along_pc - (W.along_pc.shift(1) + W.along_pc.shift(-1)) / 2)[hr == 0].abs().mean(),
                (W.along_pc - (W.along_pc.shift(1) + W.along_pc.shift(-1)) / 2)[hr != 0].abs().mean()), loc='left', fontsize=10)
ax.legend(fontsize=8, loc='upper right', markerscale=3)
ax.grid(**GRID)

ax = axd['d']
for (nm, s), c, off in zip(dU.items(), [C_PC, '0.5'], [-0.2, 0.2]):
    g = 100 * s.groupby(s.index.hour).mean()
    ax.bar(g.index + off, g.values, 0.4, color=c, label=nm)
    nb = g.drop([0]).loc[[23, 1]].mean()
    print('ocean %s: |du| into 00 UTC %.3f cm/s, neighbours (23, 01) %.3f (x%.2f)' % (nm, g[0], nb, g[0] / nb))
ax.set_xticks(range(0, 24, 1))
ax.set_xlabel('UTC hour')
ax.set_ylabel('mean size of hour-to-hour current change [cm/s]')
ax.axvspan(-0.5, 1.5, color=C_SEAM, alpha=0.12, lw=0)
ax.set_title('d  the cove barely notices: no spike at 00-01 UTC (shaded) beyond the smooth daily cycle of the surface current', loc='left', fontsize=10)
ax.legend(fontsize=8, loc='upper right')
ax.grid(axis='y', **GRID)

fig.suptitle('The daily seam in the wb1 wind forcing (atm00, wb1_t0_xn11abbur00, 2024-2025)', fontsize=12)
fig.savefig(out_dir / 'wind_seam.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved ' + str(out_dir / 'wind_seam.png'))
