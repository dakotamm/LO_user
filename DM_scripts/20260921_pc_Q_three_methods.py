"""
Qin / Qout / Qnet at the Penn Cove sections, three ways.

One figure, sections across (landward -> seaward), quantity down, with all
three methods overlaid in every panel:

  row 1  Qin      all three methods
  row 2  Qout     all three methods
  row 3  Qnet     ONE line, because all three are the same number (see below)

The three methods are

  TEF, salinity coordinate      bulk_avg_[dates]/[sn].nc
  TEF, oxygen coordinate        bulk_avg_DO_[dates]/[sn].nc
  Eulerian, sign split          extractions_avg_[dates]/[sn].nc

The two TEF methods are the Lorenz multi-layer bulk values collapsed to two
layers by
xfun.two_layer(): every layer with q > 0 summed into Qin, every layer with
q < 0 into Qout. The two files differ ONLY in the coordinate the hourly
transport was binned into before the divider ran -- salinity in row 1, oxygen in
row 2 -- so the difference between those rows is what changing the sorting
coordinate does, nothing else.

The Eulerian method is not a sorted calculation at all. It Godin filters the hourly section
extraction and splits the section by the sign of the subtidal transport in each
(z, p) cell,

    Qin_E = sum over {<q>_c > 0} of <q>_c

(xfun.eulerian_bulk, mode 'cell'; -mode vertical sums over p first for the
textbook two-layer version, which is much smaller here because it cancels the
north inflow against the south outflow before taking a sign). A sign split
never lets an inflowing cell cancel an outflowing cell at the same salinity,
which a sorted calculation does, so the Eulerian method is a GROSS measure and
the two TEF methods are NET measures. Eulerian Qin coming out roughly twice
salinity-TEF Qin at pc_lp is that difference, not an error.

WHY QNET IS ONE LINE AND NOT THREE
Qnet is the sum of the subtidal transport over the WHOLE section, and all three
methods partition the same set of cell transports <q>_c and then add every part
back up:

    Eulerian    {<q> > 0} + {<q> < 0}          = all cells
    TEF salt    cells grouped into salt bins   = all cells
    TEF DO      cells grouped into oxygen bins = all cells

A partition's total does not depend on how it was partitioned, and the Godin
filter is linear so filter-then-sum equals sum-then-filter. So the three Qnet
series are identical by construction -- checked, max |difference| ~1e-13 m3/s
-- and equal to the qnet stored in the bulk files, which is computed straight
from the raw section sum. Plotting them on top of each other shows agreement
that was never in question, so row 3 draws Qnet once.

The ONE thing that can break the identity is the Lorenz divider discarding a
layer under min_trans, which stops the all-layer sum from being the full
section. That happens once in this set -- pc_cp in oxygen coordinates, up to
0.93 m3/s on individual days, 0.0022 m3/s in the mean -- so row 3 draws that
residual, Qnet(two-layer) - Qnet(stored), as a second line. It is the only
method-discriminating information Qnet carries.

Qnet itself sits at ~-0.02 m3/s in the two-year mean because Penn Cove has
essentially no river input, so there is nothing for a net transport to balance;
the +/- 10 m3/s daily swings are cove volume storage following subtidal sea
level.

SIGN positive is INTO Penn Cove everywhere, via xfun.INFLOW_SIGN. The stored
bulk_avg_* files are in the raw section frame, whose positive direction points
OUT of the cove, so they are flipped on load.

SECTIONS only pc_cp, pc_lj and pc_lp. bulk_avg_* also holds skagit_sp and
sp_mid, but neither the DO bulk nor the hourly extractions were run for those,
so they cannot be shown all three ways.

CAVEAT the isohaline exchange flow at these sections is largely numerical floor
(20260916_frozen_field_control.py: ~88% of Qin at pc_lp). The frozen-field
control has not been run in oxygen coordinates. Read the shapes and the
method-to-method differences, not the absolute magnitudes.

run 20260921_pc_Q_three_methods.py
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
parser.add_argument('-sect', default='pc_cp,pc_lj,pc_lp', type=str,
                    help='comma separated, ordered landward -> seaward')
parser.add_argument('-mode', default='cell', type=str,
                    help="Eulerian split: 'cell' or 'vertical'")
parser.add_argument('-lp', default=30, type=int,
                    help='window in days for the centred rolling mean')
args = parser.parse_args()

# load the sibling function module by path, since DM_scripts is not a package
# and the file name starts with a digit
_fn = Path(__file__).parent / '20260916_exchange_fun.py'
_spec = importlib.util.spec_from_file_location('exchange_fun', _fn)
xfun = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xfun)

Ldir = Lfun.Lstart(gridname='wb1')
tef_dir = Ldir['LOo'] / 'extract' / args.gtagex / 'tef2'
dates = args.ds0 + '_' + args.ds1
out_dir = Path.home() / 'Desktop' / 'pltz'
Lfun.make_dir(out_dir)

SECTS = [s.strip() for s in args.sect.split(',') if s.strip()]
METHODS = ['tef_salt', 'tef_do', 'eul']
MLAB = {'tef_salt': 'TEF, salinity coordinate',
        'tef_do': 'TEF, oxygen coordinate',
        'eul': 'Eulerian, hourly avg, %s sign split' % args.mode}

# CVD-validated categorical palette, same as 20260917_bulk_plot_avg_DO_lp.py
C_IN = '#D55E00'    # vermillion, inflow
C_OUT = '#0072B2'   # blue, outflow
C_M = {'tef_salt': '#000000', 'tef_do': '#009E73', 'eul': '#CC79A7'}


def smooth(s):
    """House rolling mean: centred, window in days, tolerant of short ends."""
    return s.rolling(args.lp, center=True, min_periods=max(args.lp // 3, 1)).mean()


def bulk_two_layer(sub_dir, sn, tracer):
    """Two-layer Qin/Qout from a stored multi-layer bulk file, flipped to inflow."""
    fn = tef_dir / (sub_dir + '_' + dates) / (sn + '.nc')
    if not fn.is_file():
        raise SystemExit('missing ' + str(fn))
    ds = xr.open_dataset(fn)
    sgn = xfun.INFLOW_SIGN[sn]
    two = xfun.two_layer(sgn * ds.q.to_numpy(), ds[tracer].to_numpy())
    t = pd.to_datetime(ds.time.to_numpy())
    # stored qnet is the all-layer sum before the two-layer collapse, so it is
    # the reference Qnet; keep it to report what the collapse drops
    qnet_stored = sgn * ds.qnet.to_numpy()
    ds.close()
    return pd.DataFrame({'Qin': two['Qin'], 'Qout': two['Qout'],
                         'Qnet': two['Qin'] + two['Qout'],
                         'Qnet_stored': qnet_stored}, index=t)


# ------------------------------------------------------------------- load ---
D = {m: dict() for m in METHODS}
for sn in SECTS:
    if sn not in xfun.INFLOW_SIGN:
        raise SystemExit('no INFLOW_SIGN entry for ' + sn)
    print('loading ' + sn + ' ...')
    D['tef_salt'][sn] = bulk_two_layer('bulk_avg', sn, 'salt')
    D['tef_do'][sn] = bulk_two_layer('bulk_avg_DO', sn, 'oxygen')

    S = xfun.load_section(sn, args.gtagex, args.ds0, args.ds1, Ldir=Ldir)
    E = xfun.eulerian_bulk(S, mode=args.mode)
    t = pd.to_datetime(xfun.daily_time(S['time']))
    df = pd.DataFrame({'Qin': E['Qin'], 'Qout': E['Qout'],
                       'Qnet': E['Qin'] + E['Qout']}, index=t)
    if args.mode == 'cell':
        # the exact all-cell sum, no sign split involved
        df['Qnet_stored'] = E['Qnet']
    D['eul'][sn] = df

# ---------------------------------------------------------------- summary ---
print('\nrecord-mean transport [m3 s-1], positive = INTO Penn Cove,'
      ' %s .. %s' % (args.ds0, args.ds1))
rows = dict()
for m in METHODS:
    for k in ['Qin', 'Qout', 'Qnet']:
        rows[(MLAB[m], k)] = {sn: D[m][sn][k].mean() for sn in SECTS}
summ = pd.DataFrame(rows).T
print(summ.round(2).to_string())

print('\nQnet [m3 s-1]: two-layer collapse vs the all-layer sum in the file')
chk = dict()
for m in METHODS:
    for sn in SECTS:
        df = D[m][sn]
        if 'Qnet_stored' in df:
            chk[(MLAB[m], sn)] = {'two-layer': df['Qnet'].mean(),
                                  'all-layer': df['Qnet_stored'].mean(),
                                  'dropped': (df['Qnet']
                                              - df['Qnet_stored']).mean()}
print(pd.DataFrame(chk).T.round(4).to_string())

print('\ngross/net ratio, Qin(Eulerian) / Qin(TEF salt):')
print(pd.Series({sn: D['eul'][sn]['Qin'].mean() / D['tef_salt'][sn]['Qin'].mean()
                 for sn in SECTS}).round(2).to_string())

out_csv = out_dir / '20260921_pc_Q_three_methods.csv'
pd.concat({MLAB[m]: pd.concat(D[m], axis=1) for m in METHODS},
          axis=1).to_csv(out_csv)
print('\nsaved ' + str(out_csv))

# ----------------------------------------------------------------- figure ---
plt.close('all')
mosaic = [[r + '_' + sn for sn in SECTS] for r in ['Qin', 'Qout', 'Qnet']]
fig, axes = plt.subplot_mosaic(mosaic, figsize=(5.2 * len(SECTS), 9.5),
                               layout='constrained', sharex=True)

for sn in SECTS:
    # Qin and Qout get the same magnitude scale down each column, so the
    # inflowing and outflowing limbs can be compared by eye.
    lim = 1.05 * max(np.nanmax(np.abs(D[m][sn][['Qin', 'Qout']].to_numpy()))
                     for m in METHODS)
    for vn, sgn_lim in [('Qin', 1), ('Qout', -1)]:
        ax = axes[vn + '_' + sn]
        for m in METHODS:
            df = D[m][sn]
            # three daily clouds on one axis would be unreadable, so the daily
            # series goes down faint and the rolling mean carries the colour
            ax.plot(df.index, df[vn], lw=0.6, alpha=0.15, color=C_M[m])
            ax.plot(df.index, smooth(df[vn]), lw=2, color=C_M[m], label=MLAB[m])
        ax.axhline(0, color='k', lw=0.8, alpha=0.4)
        ax.set_ylim(*sorted([0, sgn_lim * lim]))
        if sn == SECTS[0]:
            ax.set_ylabel(('$Q_{in}$' if vn == 'Qin' else '$Q_{out}$')
                          + r'  [m$^3$ s$^{-1}$]')
        if vn == 'Qin':
            ax.set_title(sn, fontweight='bold')
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

    # ---- Qnet. All three methods give this identically (module docstring), so
    # it is drawn ONCE, with the only method-dependent piece -- the transport in
    # layers the Lorenz divider dropped under min_trans -- over the top.
    ax = axes['Qnet_' + sn]
    qn = D['tef_salt'][sn]['Qnet']
    ax.plot(qn.index, qn, lw=0.7, alpha=0.35, color='0.45')
    ax.plot(qn.index, smooth(qn), lw=2, color='0.15',
            label=r'$Q_{net}$, all three methods')
    for m in ['tef_salt', 'tef_do']:
        df = D[m][sn]
        ax.plot(df.index, df['Qnet'] - df['Qnet_stored'], lw=1, color=C_M[m],
                label='dropped layers, ' + MLAB[m].split(',')[1].strip())
    ax.axhline(0, color='k', lw=0.8, alpha=0.4)
    # on a +/- 10 m3/s axis the dropped-layer lines sit on zero, which is the
    # honest picture but unreadable, so give the worst day as a number too
    worst = max(np.nanmax(np.abs((D[m][sn]['Qnet']
                                  - D[m][sn]['Qnet_stored']).to_numpy()))
                for m in ['tef_salt', 'tef_do'])
    ax.text(0.99, 0.04, 'largest dropped-layer day  %.2g m$^3$ s$^{-1}$' % worst,
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            alpha=0.8)
    if sn == SECTS[0]:
        ax.set_ylabel(r'$Q_{net}$  [m$^3$ s$^{-1}$]')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

for k in axes:
    axes[k].grid(color='lightgray', linestyle='--', alpha=0.5)
    axes[k].margins(x=0.01)
    axes[k].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axes[k].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

fig.suptitle('Penn Cove exchange transport three ways   (positive = into the '
             'cove)\nfaint = Godin filtered, daily subsampled; heavy = %d-day '
             'rolling mean   |   TEF is a net (sorted) measure, the Eulerian '
             'sign split a gross one' % args.lp, fontsize=12)

for k, letter in zip([r + '_' + sn for r in ['Qin', 'Qout', 'Qnet']
                      for sn in SECTS], 'abcdefghi'):
    axes[k].text(0.008, 1.02, letter, transform=axes[k].transAxes,
                 fontsize=13, fontweight='bold', va='bottom')

fn_out = out_dir / '20260921_pc_Q_three_methods.png'
fig.savefig(fn_out, dpi=200, bbox_inches='tight', transparent=True)
print('saved ' + str(fn_out))
