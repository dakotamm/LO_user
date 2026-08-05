"""
TEF flushing time for the wb1_pc1 volumes, and how it varies through the basin.

Flushing time here is the TEF exchange time

    T = V / Qin

with Qin the tidally-averaged transport INTO the volume, summed over that
volume's bounding sections, taken from the two-layer bulk_calc_avg output.
It is a bulk replacement timescale: how long the exchange flow needs to supply
a volume of water equal to the volume itself. It is NOT a Lagrangian residence
time -- it assumes the inflow mixes through the whole box, which in a
stratified cove where the inflow is a deep layer is an approximation.

Reported alongside is the tidal-prism timescale

    T_prism = V / Qprism,     Qprism = 1/2 <|qnet - qnet_lowpass|>

which is the timescale if the tide exchanged with perfect efficiency. The
ratio T_prism/T is then an exchange efficiency: how much of the tidal
excursion actually does net exchange.

WHY THE COVE VOLUMES ARE NESTED, NOT PER-SEGMENT
The three cove volumes are cumulative -- everything landward of pc_cp, of
pc_lj, of pc_lp -- so each has exactly ONE open boundary. Per-segment boxes
between adjacent lines have two open ends, and summing the inflow at both
counts the same water sloshing between neighbours twice, which makes a middle
box look far better flushed than an inner box for reasons of bookkeeping
rather than physics. Nested volumes are the comparable choice along the axis.
Upper Saratoga necessarily has three open boundaries, so its T is a
ventilation time and includes the ~1500 m3/s throughflow -- read it as a
different quantity from the cove numbers, not a fourth point on the gradient.

Freshwater flushing time (V_fw/Qr) is deliberately absent: the cove has no
meaningful local freshwater source, so it is not defined here.

run 20260805_tef_flushing_time.py
run 20260805_tef_flushing_time.py -gtx wb1_t0_xn11abbur00 -0 2024.01.01 -1 2025.12.31
"""
import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
parser.add_argument('-ctag', default='pc1', type=str)
parser.add_argument('-riv', default='riv00', type=str)
parser.add_argument('-0', '--ds0', default='2024.01.01', type=str)
parser.add_argument('-1', '--ds1', default='2025.12.31', type=str)
args = parser.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
gctag = 'wb1_' + args.ctag
tef2_dir = Ldir['LOo'] / 'extract' / 'tef2'
bulk_dir = (Ldir['LOo'] / 'extract' / args.gtagex / 'tef2'
            / ('bulk_avg_' + args.ds0 + '_' + args.ds1))
out_dir = Path.home() / 'Desktop' / 'pltz'
Lfun.make_dir(out_dir)

vol_df = pd.read_pickle(tef2_dir / ('vol_df_' + gctag + '.p'))

# Each volume: the segments it contains, and (section, sign) for every open
# boundary. sign = +1 if the volume sits on that section's plus side, -1 if on
# its minus side, so that inflow = sum over layers of max(sign*q, 0).
# Verified 2026.08.04 from rho-cell adjacency; all faces of all sections agreed.
VOLUMES = {
    'Penn Cove, in from pc_cp': dict(
        segs=['pc_cp_m'], sects=[('pc_cp', -1)], seaward='pc_cp'),
    'Penn Cove, in from pc_lj': dict(
        segs=['pc_cp_m', 'pc_cp_p'], sects=[('pc_lj', -1)], seaward='pc_lj'),
    'Penn Cove, in from pc_lp': dict(
        segs=['pc_cp_m', 'pc_cp_p', 'pc_lp_m'], sects=[('pc_lp', -1)],
        seaward='pc_lp'),
    'Upper Saratoga': dict(
        segs=['pc_lp_p'],
        sects=[('pc_lp', 1), ('sp_mid', 1), ('skagit_sp', -1)],
        seaward=None),
}

# CVD-validated categorical order (validate_palette.js, light surface):
# worst adjacent pair dE 9.6 protan, all other checks PASS.
COLORS = ['#009E73', '#0072B2', '#CC79A7', '#D55E00']
vol_color = {k: COLORS[i] for i, k in enumerate(VOLUMES)}

# ------------------------------------------------------------------ load ----
bulk = {}
for sn in sorted({s for v in VOLUMES.values() for s, _ in v['sects']}):
    bulk[sn] = xr.open_dataset(bulk_dir / (sn + '.nc'))
    print('%-10s %d times' % (sn, bulk[sn].sizes['time']))

time = pd.to_datetime(bulk[list(bulk)[0]].time.values)


def inflow(sn, sgn):
    """Transport into the volume across section sn [m3/s], per time."""
    q = bulk[sn].q.values                       # (time, layer), NaN padded
    sq = sgn * q
    return np.nansum(np.where(sq > 0, sq, 0), axis=1)


rows = {}
for name, v in VOLUMES.items():
    V = vol_df.loc[v['segs'], 'volume m3'].sum()
    Qin = np.zeros(len(time))
    for sn, sgn in v['sects']:
        Qin += inflow(sn, sgn)
    # tidal prism: the seaward section for the nested cove volumes, the sum
    # over all openings where there is no single seaward face
    if v['seaward'] is not None:
        Qp = bulk[v['seaward']].qprism.values
    else:
        Qp = np.sum([bulk[sn].qprism.values for sn, _ in v['sects']], axis=0)
    rows[name] = pd.DataFrame(
        {'V': V, 'Qin': Qin, 'Qprism': Qp,
         'T_days': V / Qin / 86400, 'Tprism_days': V / Qp / 86400},
        index=time)

df = pd.concat(rows, names=['volume', 'time']).reset_index()
df['month'] = df.time.dt.month
df['year'] = df.time.dt.year

# Dakota's 3 season bins, December folding forward into the next year
SEASON = {12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Winter',
          4: 'Spring', 5: 'Spring', 6: 'Spring', 7: 'Spring',
          8: 'Low-DO', 9: 'Low-DO', 10: 'Low-DO', 11: 'Low-DO'}
df['season'] = df.month.map(SEASON)

# --------------------------------------------------------------- summary ---
print('\n' + '=' * 78)
print('TEF flushing time  T = V/Qin   [days]   %s .. %s' % (args.ds0, args.ds1))
print('=' * 78)
summ = df.groupby('volume').agg(
    V_km3=('V', lambda s: s.iloc[0] / 1e9),
    Qin=('Qin', 'mean'), Qprism=('Qprism', 'mean'),
    T_mean=('T_days', 'mean'), T_min=('T_days', 'min'), T_max=('T_days', 'max'),
    Tprism=('Tprism_days', 'mean'))
summ['efficiency'] = summ.Tprism / summ.T_mean
summ = summ.reindex(list(VOLUMES))
print(summ.round(3).to_string())

print('\nSeasonal mean T [days]:')
seas = df.pivot_table(index='volume', columns='season', values='T_days',
                      aggfunc='mean').reindex(list(VOLUMES))
print(seas[['Winter', 'Spring', 'Low-DO']].round(2).to_string())

csv_fn = out_dir / '20260805_tef_flushing_time.csv'
df.to_csv(csv_fn, index=False)
summ.to_csv(out_dir / '20260805_tef_flushing_time_summary.csv')
print('\nsaved %s' % csv_fn)

# --------------------------------------------------------------- figure ----
plt.close('all')
fig, axes = plt.subplot_mosaic([['ts', 'ts'], ['clim', 'prism']],
                               figsize=(13, 8), layout='constrained')

ax = axes['ts']
for name in VOLUMES:
    d = df[df.volume == name]
    ax.plot(d.time, d.T_days, lw=0.6, alpha=0.3, color=vol_color[name])
    sm = d.set_index('time').T_days.rolling(30, center=True, min_periods=10).mean()
    ax.plot(sm.index, sm.values, lw=2, color=vol_color[name], label=name)
ax.set_ylabel('flushing time T = V/Qin  [days]')
ax.set_title('TEF flushing time through the basin\n'
             'thin = Godin-filtered, daily-subsampled (every bulk_avg point); '
             'thick = 30-day rolling mean')
ax.grid(color='lightgray', linestyle='--', alpha=0.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
# legend outside the axes: the traces run right across the top-left corner, and
# the four labels are too long to sit inside without covering data
ax.legend(loc='upper left', bbox_to_anchor=(0, -0.02, 1, 1), ncol=4,
          fontsize=8, framealpha=0.9, mode='expand', borderaxespad=0)
ax.margins(x=0.01)

ax = axes['clim']
for name in VOLUMES:
    d = df[df.volume == name].groupby('month').T_days.agg(['mean', 'std'])
    ax.plot(d.index, d['mean'], '-o', lw=2, ms=5, color=vol_color[name], label=name)
    ax.fill_between(d.index, d['mean'] - d['std'], d['mean'] + d['std'],
                    color=vol_color[name], alpha=0.12, lw=0)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(list('JFMAMJJASOND'))
ax.set_xlabel('month')
ax.set_ylabel('T [days]')
ax.set_title('Seasonal climatology (shading = 1 sd)')
ax.grid(color='lightgray', linestyle='--', alpha=0.5)

ax = axes['prism']
# Plotted as anomalies from each volume's own mean. On absolute axes Saratoga
# sits near 1.5e4 m3/s and the cove near 5e2, so the two just form separate
# clusters and the spring-neap response inside each is unreadable. Normalising
# puts them on a common footing and shows the actual relationship.
for name in VOLUMES:
    d = df[df.volume == name]
    ax.plot(d.Qprism / d.Qprism.mean(), d.T_days / d.T_days.mean(), '.',
            ms=3, alpha=0.3, color=vol_color[name], label=name)
    # binned median, so the trend is visible through the scatter
    x = d.Qprism / d.Qprism.mean()
    y = d.T_days / d.T_days.mean()
    bins = np.linspace(x.min(), x.max(), 9)
    ib = np.digitize(x, bins) - 1
    bx = [x[ib == k].mean() for k in range(len(bins) - 1) if (ib == k).sum() > 5]
    by = [np.median(y[ib == k]) for k in range(len(bins) - 1) if (ib == k).sum() > 5]
    ax.plot(bx, by, '-', lw=2, color=vol_color[name])
ax.axhline(1, color='gray', lw=0.8, ls=':')
ax.axvline(1, color='gray', lw=0.8, ls=':')
ax.set_xlabel(r'$Q_{prism}$ / mean   (neap $\rightarrow$ spring)')
ax.set_ylabel('T / mean T')
ax.set_title('Spring-neap control (lines = binned median)')
ax.grid(color='lightgray', linestyle='--', alpha=0.5)

for k, letter in zip(['ts', 'clim', 'prism'], 'abc'):
    axes[k].text(0.012, 1.02, letter, transform=axes[k].transAxes,
                 fontsize=14, fontweight='bold', va='bottom')

fn_out = out_dir / '20260805_tef_flushing_time.png'
fig.savefig(fn_out, dpi=200, bbox_inches='tight')
print('saved %s' % fn_out)
