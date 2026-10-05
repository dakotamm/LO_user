"""
Both exchange limbs at the Penn Cove mouth, three ways, on one panel.

The single-panel cut of 20260921_pc_Q_three_methods.py: one section (pc_lp, the
mouth), both limbs, three methods overlaid. Qin is positive and Qout negative,
so the two limbs separate above and below zero on a single axis and colour is
free to carry the method.

  TEF, salinity coordinate      bulk_avg_[dates]/pc_lp.nc
  TEF, oxygen coordinate        bulk_avg_DO_[dates]/pc_lp.nc
  Eulerian, sign split          extractions_avg_[dates]/pc_lp.nc

The two TEF curves are the Lorenz multi-layer bulk values collapsed to two
layers by xfun.two_layer(), summing every layer with q > 0 into Qin and every
layer with q < 0 into Qout. They differ ONLY in the coordinate the hourly
transport was binned into before the divider ran, so the gap between them is
what changing the sorting coordinate does and nothing else.

The Eulerian curves are not sorted at all: Godin filter the hourly extraction
and split the (z, p) cells on the sign of <q>_c. A sign split never lets an
inflowing cell cancel an outflowing cell at the same tracer value, which a
sorted calculation does, so this is a GROSS measure against two NET ones. At
pc_lp the inflow and outflow sit side by side at nearly equal salinity -- the
lateral gyre -- so the cancellation is large and the Eulerian curves run about
twice the salinity-TEF ones. That is the method, not an error.

Qin and Qout are near mirror images in every method because Qnet is ~-0.02 m3/s
-- Penn Cove has essentially no river input, so there is nothing for a net
transport to balance and the two limbs must very nearly cancel. The visible
asymmetry between the limbs is cove volume storage following subtidal sea
level, worth a few m3/s against limbs of several hundred.

WHY QNET IS THE SAME IN ALL THREE
Qnet lives in the lower strip, at ~1/100 of the main panel's scale, and all
three methods land on one curve. That is an algebraic identity, not agreement
between independent estimates. Index the section cells by c = 1..N (N = 360
here, 30 sigma by 12 stairstep points) and write q_c(t) for the hourly Huon
flux and <.> for the Godin average.

The Godin filter is a fixed-weight convolution, <x>(t) = sum_k w_k x(t - k), so
it is LINEAR: sum_c <q_c> = < sum_c q_c > = <q_net>. Order of summing and
filtering does not matter. (Checked directly: max |difference| ~1e-12 m3/s.)

Each method then partitions the same fluxes and adds every part back:

  Eulerian   groups are {c : <q_c> > 0} and {c : <q_c> < 0}. Every cell is in
             exactly one, so
                 Qin + Qout = sum_c <q_c> = <q_net>.

  TEF salt   bin hourly: T_n(t) = sum over {c : s_c(t) in bin n} of q_c(t).
             Bins partition the salinity axis and each cell has exactly one
             salinity at each t, so sum_n T_n(t) = sum_c q_c(t) = q_net(t) at
             EVERY hour. Godin average, then the Lorenz divider groups the
             bins into layers l -- another partition -- giving Q_l, and
             two_layer() sums the positive Q_l into Qin and the negative into
             Qout. So
                 Qin + Qout = sum_l Q_l = sum_n <T_n> = <q_net>.

  TEF DO     identical, with o_c(t) in place of s_c(t).

The tracer only decides WHICH bin a cell's flux is dropped into. It never
changes the flux value and never decides whether it is counted. A partition's
total does not depend on how it was partitioned, so Qnet is invariant under all
of this, and equals the qnet stored in the bulk files, computed straight from
the raw section sum.

The one thing that can break the identity is min_trans: the divider discards
layers carrying less than that, so sum_l Q_l stops covering the whole section.
At pc_lp it never fires; at pc_cp in oxygen coordinates it costs up to
0.93 m3/s on individual days.

SIGN positive is INTO Penn Cove, via xfun.INFLOW_SIGN. The stored bulk_avg_*
files are in the raw section frame, whose positive direction points OUT of the
cove, so they are flipped on load.

CAVEAT the isohaline exchange flow here is largely numerical floor
(20260916_frozen_field_control.py: ~88% of Qin at pc_lp), and the frozen-field
control has never been run in oxygen coordinates. Read the shapes and the
method-to-method separation, not the absolute magnitudes.

run 20260921_pc_lp_Qin_three_methods.py
"""
import argparse
import importlib.util
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
parser.add_argument('-0', '--ds0', default='2024.01.01', type=str)
parser.add_argument('-1', '--ds1', default='2025.12.31', type=str)
parser.add_argument('-sect', default='pc_lp', type=str)
parser.add_argument('-mode', default='cell', type=str,
                    help="Eulerian split: 'cell' or 'vertical'")
parser.add_argument('-lp', default=30, type=int,
                    help='window in days for the centred rolling mean')
args = parser.parse_args()

_fn = Path(__file__).parent / '20260916_exchange_fun.py'
_spec = importlib.util.spec_from_file_location('exchange_fun', _fn)
xfun = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xfun)

Ldir = Lfun.Lstart(gridname='wb1')
tef_dir = Ldir['LOo'] / 'extract' / args.gtagex / 'tef2'
dates = args.ds0 + '_' + args.ds1
out_dir = Path.home() / 'Desktop' / 'pltz'
Lfun.make_dir(out_dir)

sn = args.sect
if sn not in xfun.INFLOW_SIGN:
    raise SystemExit('no INFLOW_SIGN entry for ' + sn)
sgn = xfun.INFLOW_SIGN[sn]

MLAB = {'tef_salt': 'TEF, salinity coordinate',
        'tef_do': 'TEF, oxygen coordinate',
        'eul': 'Eulerian, hourly avg, %s sign split' % args.mode}
C_M = {'tef_salt': '#000000', 'tef_do': '#009E73', 'eul': '#CC79A7'}
LOWDO = [8, 11]          # Low-DO season, shaded


def smooth(s):
    """House rolling mean: centred, window in days, tolerant of short ends."""
    return s.rolling(args.lp, center=True, min_periods=max(args.lp // 3, 1)).mean()


def bulk_limbs(sub_dir, tracer):
    """Two-layer Qin/Qout from a stored multi-layer bulk file, flipped to inflow."""
    fn = tef_dir / (sub_dir + '_' + dates) / (sn + '.nc')
    if not fn.is_file():
        raise SystemExit('missing ' + str(fn))
    ds = xr.open_dataset(fn)
    two = xfun.two_layer(sgn * ds.q.to_numpy(), ds[tracer].to_numpy())
    t = pd.to_datetime(ds.time.to_numpy())
    ds.close()
    return pd.DataFrame({'Qin': two['Qin'], 'Qout': two['Qout']}, index=t)


# ------------------------------------------------------------------- load ---
print('loading ' + sn + ' ...')
D = dict()
D['tef_salt'] = bulk_limbs('bulk_avg', 'salt')
D['tef_do'] = bulk_limbs('bulk_avg_DO', 'oxygen')
S = xfun.load_section(sn, args.gtagex, args.ds0, args.ds1, Ldir=Ldir)
E = xfun.eulerian_bulk(S, mode=args.mode)
D['eul'] = pd.DataFrame({'Qin': E['Qin'], 'Qout': E['Qout']},
                        index=pd.to_datetime(xfun.daily_time(S['time'])))
METHODS = ['tef_salt', 'tef_do', 'eul']
for m in METHODS:
    D[m]['Qnet'] = D[m]['Qin'] + D[m]['Qout']
# the identity the docstring argues for, checked rather than asserted
spread = np.nanmax(np.abs(np.diff(np.column_stack(
    [D[m]['Qnet'].to_numpy() for m in METHODS]), axis=1)))
print('max |Qnet difference| between methods: %.2e m3 s-1' % spread)
df = pd.concat({MLAB[m]: D[m] for m in METHODS}, axis=1)

# ---------------------------------------------------------------- summary ---
print('\nExchange limbs at %s [m3 s-1], positive = INTO Penn Cove, %s .. %s'
      % (sn, args.ds0, args.ds1))
print(pd.DataFrame({MLAB[m]: {'Qin mean': D[m]['Qin'].mean(),
                              'Qin std': D[m]['Qin'].std(),
                              'Qout mean': D[m]['Qout'].mean(),
                              'Qout std': D[m]['Qout'].std(),
                              'Qnet mean': (D[m]['Qin'] + D[m]['Qout']).mean(),
                              '|Qout|/Qin': -D[m]['Qout'].mean()
                              / D[m]['Qin'].mean()}
                    for m in METHODS}).round(3).to_string())
print('\ngross/net ratio, Eulerian / TEF salt:  Qin %.2f   Qout %.2f'
      % (D['eul']['Qin'].mean() / D['tef_salt']['Qin'].mean(),
         D['eul']['Qout'].mean() / D['tef_salt']['Qout'].mean()))
print('oxygen TEF / salinity TEF:             Qin %.3f   Qout %.3f'
      % (D['tef_do']['Qin'].mean() / D['tef_salt']['Qin'].mean(),
         D['tef_do']['Qout'].mean() / D['tef_salt']['Qout'].mean()))
for vn in ('Qin', 'Qout'):
    print('\ncorrelation of the %d-day rolling means, %s:' % (args.lp, vn))
    print(pd.DataFrame({MLAB[m]: smooth(D[m][vn]) for m in METHODS})
          .corr().round(3).to_string())

out_csv = out_dir / ('20260921_%s_limbs_three_methods.csv' % sn)
df.to_csv(out_csv)
print('\nsaved ' + str(out_csv))

# ----------------------------------------------------------------- figure ---
plt.close('all')
fig, ax = plt.subplots(figsize=(12, 5.5), layout='constrained')

for yr in sorted(set(df.index.year)):
    ax.axvspan(pd.Timestamp(yr, LOWDO[0], 1),
               pd.Timestamp(yr, LOWDO[1], 1) + pd.offsets.MonthEnd(1),
               color='0.85', alpha=0.45, lw=0, zorder=0)

for m in METHODS:
    # Qin is positive and Qout negative, so the limbs separate above and below
    # zero on their own and colour is free to carry the method. Six daily
    # clouds on one axis would be unreadable, so daily goes down faint and the
    # rolling mean carries the colour.
    for vn in ('Qin', 'Qout'):
        ax.plot(D[m].index, D[m][vn], lw=0.6, alpha=0.13, color=C_M[m], zorder=2)
        ax.plot(D[m].index, smooth(D[m][vn]), lw=2.4, color=C_M[m], zorder=3,
                label=('%s   (%+.0f / %+.0f)'
                       % (MLAB[m], D[m]['Qin'].mean(), D[m]['Qout'].mean()))
                if vn == 'Qin' else None)

# Qnet goes on the SAME axis as the limbs, at true scale. It lies on zero and
# all three methods lie on each other, which is the honest picture: Qnet is
# ~1/30000 of Qin and is identical across methods by construction (docstring).
# Decreasing linewidth so all three are visibly still there.
for m, lw_, ls_ in [('tef_salt', 4, '-'), ('tef_do', 2.4, '-'),
                    ('eul', 1.2, '--')]:
    ax.plot(D[m].index, D[m]['Qnet'], lw=lw_, ls=ls_, color=C_M[m], zorder=5,
            label=r'$Q_{net}$, all three (%+.2f)' % D[m]['Qnet'].mean()
            if m == 'tef_salt' else None)

lim = 1.05 * max(np.nanmax(np.abs(D[m][['Qin', 'Qout']].to_numpy()))
                 for m in METHODS)
ax.set_ylim(-lim, lim)
ax.axhline(0, color='k', lw=0.9, alpha=0.6, zorder=4)
ax.set_ylabel(r'$Q$  [m$^3$ s$^{-1}$]')
ax.grid(color='lightgray', linestyle='--', alpha=0.5, zorder=1)
ax.margins(x=0.01)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.legend(loc='upper left', fontsize=9, framealpha=0.9,
          title=r'method   ($\bar{Q}_{in}$ / $\bar{Q}_{out}$)',
          title_fontsize=9)
# the limbs are labelled on the axis rather than in the legend, which would
# otherwise need six entries to say what the sign already says
for y, lab in [(0.965, r'$Q_{in}$, into the cove'),
               (0.035, r'$Q_{out}$, out of the cove')]:
    ax.text(0.995, y, lab, transform=ax.transAxes, ha='right',
            va='top' if y > 0.5 else 'bottom', fontsize=10, alpha=0.75)
# just above the zero line is the one empty band on this axis
ax.text(0.008, 0.515, r'$Q_{net}$, all three methods (max spread %.0e)' % spread,
        transform=ax.transAxes, ha='left', va='bottom', fontsize=9, alpha=0.75)
ax.set_title('Exchange limbs at %s, three ways   (positive = into Penn Cove)\n'
             'faint = Godin filtered, daily subsampled; heavy = %d-day rolling '
             'mean; shading = Low-DO season\nTEF is a net (sorted) measure, the '
             'Eulerian sign split a gross one' % (sn, args.lp), fontsize=11)

fn_out = out_dir / ('20260921_%s_limbs_three_methods.png' % sn)
fig.savefig(fn_out, dpi=200, bbox_inches='tight', transparent=True)
print('saved ' + str(fn_out))
