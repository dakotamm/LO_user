"""
Oxygen-coordinate exchange flow at the Penn Cove sections, low-pass filtered.

Same three quantities as LO_user/extract/tef2/bulk_plot_avg_DO.py, but laid out
as one figure with the sections side by side, and with a rolling mean over the
top so the seasonal signal is readable under the fortnightly one.

NOTE every point here is ALREADY tidally averaged. bulk_calc_avg_DO.py Godin
filters and then takes every 24th value, so "daily" is the sampling interval,
not raw daily data -- there is no unfiltered series in bulk_avg_DO. What the
rolling mean removes is the spring-neap modulation, which Godin leaves alone.
Same situation as 20260805_tef_qprism.py.

SIGN CONVENTION -- DIFFERENT FROM THE LO_user SCRIPT
bulk_avg_DO_* is stored in the raw section frame, where positive is the
section's own direction (pm = +1), which at the pc sections points OUT of Penn
Cove. That is right for a file that has to match bulk_avg_*, and wrong for
reading a figure. Here q is multiplied by INFLOW_SIGN from
20260916_exchange_fun.py, so positive is INTO Penn Cove everywhere and Qin is
the inflowing limb in the ordinary estuarine sense.

WHAT IS PLOTTED
row 1   Qin, Qout            two-layer volume transport [m3 s-1]
row 2   O_in, O_out          flux-weighted oxygen of each limb [mg L-1]
row 3   F_net, Qin*dO        oxygen flux [g s-1], with Qprism on the twin
                             -- rolling means only, see the note in the code

F_net is summed over the multi-layer bulk values, not rebuilt from the
two-layer collapse, so it carries the layers that the two-layer step drops.

UNITS the model carries oxygen in mmol m-3 (= uM) and bulk_avg_DO_* stores it
that way. This script converts to O2 mass for display: concentrations in
mg L-1, fluxes in g s-1. See CONV below. The dotted line on row 2 is the
conventional 2 mg L-1 hypoxia threshold, which only fits on the axis once the
units are mg L-1.

The third row deliberately does NOT show Qin*dS's oxygen analogue as an
"exchange strength" -- dO between the limbs is small and noisy compared to the
seasonal swing in O itself, so Qin*dO is plotted next to F_net rather than
instead of it.

CAVEAT the isohaline exchange flow at these sections is largely numerical
floor (see 20260916_frozen_field_control.py). Nothing about switching the
coordinate to oxygen fixes that, and the frozen-field control has not been run
in oxygen coordinates yet. Read the shapes, not the magnitudes.

run 20260917_bulk_plot_avg_DO_lp.py
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
bulk_dir = (Ldir['LOo'] / 'extract' / args.gtagex / 'tef2'
            / ('bulk_avg_DO_' + args.ds0 + '_' + args.ds1))
out_dir = Path.home() / 'Desktop' / 'pltz'
Lfun.make_dir(out_dir)

if not bulk_dir.is_dir():
    raise SystemExit('no ' + str(bulk_dir)
                     + '\nrun process_sections_avg_DO.py then bulk_calc_avg_DO.py')

SECTS = [s.strip() for s in args.sect.split(',') if s.strip()]

# The model carries oxygen in mmol m-3 (= uM); everything below is reported as
# O2 mass. Both conversions are the molar mass over 1000:
#     1 mmol m-3  =  31.998 mg m-3  =  0.031998 mg L-1
#     1 mmol      =  0.031998 g
# so concentrations become mg L-1 and fluxes become g s-1. The stored
# bulk_avg_DO_* files stay in model units -- this is a presentation change only,
# and the binning coordinate in process_sections_avg_DO.py is untouched.
O2_MM = 31.998          # g mol-1
CONV = O2_MM / 1000

# the conventional hypoxia threshold, now that the axis is in mg L-1
HYPOXIC = 2.0

# CVD-validated categorical pair, same palette as 20260805_tef_qprism.py
C_IN = '#D55E00'   # vermillion, inflow
C_OUT = '#0072B2'  # blue, outflow
C_NET = '#000000'
C_EXCH = '#6E6E6E'
C_PRISM = '#56B4E9'


def smooth(s):
    """House rolling mean: centred, window in days, tolerant of short ends."""
    return s.rolling(args.lp, center=True, min_periods=max(args.lp // 3, 1)).mean()


# ------------------------------------------------------------------- load ---
D = dict()
for sn in SECTS:
    ds = xr.open_dataset(bulk_dir / (sn + '.nc'))
    if sn not in xfun.INFLOW_SIGN:
        raise SystemExit('no INFLOW_SIGN entry for ' + sn)
    sgn = xfun.INFLOW_SIGN[sn]
    q = sgn * ds.q.to_numpy()          # (time, layer), positive INTO the cove
    o = ds.oxygen.to_numpy()
    t = pd.to_datetime(ds.time.to_numpy())

    two = xfun.two_layer(q, o)         # generic in the tracer, despite the name
    df = pd.DataFrame(index=t)
    df['Qin'] = two['Qin']
    df['Qout'] = two['Qout']
    df['O_in'] = two['sin'] * CONV
    df['O_out'] = two['sout'] * CONV
    df['dO'] = df['O_in'] - df['O_out']
    # oxygen flux [g s-1], from the full multi-layer set. q*o is mmol s-1.
    df['F_net'] = np.nansum(q * o, axis=1) * CONV
    # Qin [m3 s-1] * dO [mg L-1] = g s-1 directly, since mg L-1 == g m-3.
    df['F_exch'] = df['Qin'] * df['dO']
    df['qprism'] = ds.qprism.to_numpy()
    ds.close()
    D[sn] = df

# ---------------------------------------------------------------- summary ---
print('\nOxygen-coordinate exchange, %s .. %s   (positive = INTO Penn Cove)'
      % (args.ds0, args.ds1))
summ = pd.DataFrame({sn: {'Qin [m3/s]': D[sn]['Qin'].mean(),
                          'Qout [m3/s]': D[sn]['Qout'].mean(),
                          'O_in [mg/L]': D[sn]['O_in'].mean(),
                          'O_out [mg/L]': D[sn]['O_out'].mean(),
                          'dO [mg/L]': D[sn]['dO'].mean(),
                          'F_net [g/s]': D[sn]['F_net'].mean(),
                          'Qprism [m3/s]': D[sn]['qprism'].mean()}
                     for sn in SECTS})
print(summ.round(2).to_string())
pd.concat(D, axis=1).to_csv(out_dir / '20260917_bulk_plot_avg_DO_lp.csv')

# ----------------------------------------------------------------- figure ---
plt.close('all')
mosaic = [[r + '_' + sn for sn in SECTS] for r in ['Q', 'O', 'F']]
fig, axes = plt.subplot_mosaic(mosaic, figsize=(5.2 * len(SECTS), 9),
                               layout='constrained', sharex=True)

for sn in SECTS:
    df = D[sn]

    # ---- row 1: transport
    ax = axes['Q_' + sn]
    for vn, c in [('Qin', C_IN), ('Qout', C_OUT)]:
        ax.plot(df.index, df[vn], lw=0.7, alpha=0.35, color=c)
        ax.plot(df.index, smooth(df[vn]), lw=2, color=c, label=vn)
    ax.axhline(0, color='k', lw=0.8, alpha=0.4)
    ax.set_title(sn, fontweight='bold')
    if sn == SECTS[0]:
        ax.set_ylabel(r'Transport  [m$^3$ s$^{-1}$]')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=2)

    # ---- row 2: flux-weighted oxygen
    ax = axes['O_' + sn]
    for vn, c, lab in [('O_in', C_IN, r'$O_{in}$'), ('O_out', C_OUT, r'$O_{out}$')]:
        ax.plot(df.index, df[vn], lw=0.7, alpha=0.35, color=c)
        ax.plot(df.index, smooth(df[vn]), lw=2, color=c, label=lab)
    ax.axhline(HYPOXIC, color='k', lw=1, linestyle=':', alpha=0.6)
    ax.text(0.995, HYPOXIC, ' %g mg L$^{-1}$' % HYPOXIC, transform=ax.get_yaxis_transform(),
            ha='right', va='bottom', fontsize=8, alpha=0.7)
    if sn == SECTS[0]:
        ax.set_ylabel(r'Oxygen  [mg L$^{-1}$]')
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9, ncol=2)

    # ---- row 3: oxygen flux, with Qprism behind
    ax = axes['F_' + sn]
    axqp = ax.twinx()
    axqp.plot(df.index, smooth(df['qprism']), color=C_PRISM, lw=3, alpha=0.4,
              zorder=0)
    axqp.set_ylim(bottom=0)
    axqp.tick_params(axis='y', colors=C_PRISM, labelsize=8)
    if sn == SECTS[-1]:
        axqp.set_ylabel(r'$Q_{prism}$  [m$^3$ s$^{-1}$]', color=C_PRISM)
    else:
        # one Qprism axis is enough; the others are clutter
        axqp.tick_params(axis='y', labelright=False)
    sm = dict()
    for vn, c, lab in [('F_net', C_NET, r'$F_{net}$'),
                       ('F_exch', C_EXCH, r'$Q_{in}\Delta O$')]:
        sm[vn] = smooth(df[vn])
    # These two sit almost exactly on top of each other -- the net oxygen flux
    # is nearly all exchange-carried -- so F_net goes down heavy and Qin*dO
    # goes over it dashed, and the places they separate are visible.
    ax.plot(df.index, sm['F_net'], lw=2.5, color=C_NET, label=r'$F_{net}$',
            zorder=3)
    ax.plot(df.index, sm['F_exch'], lw=1.5, color=C_EXCH, linestyle='--',
            label=r'$Q_{in}\Delta O$', zorder=4)
    ax.axhline(0, color='k', lw=0.8, alpha=0.4, zorder=1)
    ax.set_zorder(axqp.get_zorder() + 1)
    ax.patch.set_visible(False)
    if sn == SECTS[0]:
        ax.set_ylabel(r'Oxygen flux  [g s$^{-1}$]')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=2)
    # NOTE this row shows the rolling means ONLY. The daily F_net swings
    # +/- 40 mol s-1 about a mean near -2, so it either sets the scale and
    # flattens both rolling means onto the zero line, or gets clipped and fills
    # the panel with a solid picket fence. Neither is readable, and this is the
    # low-pass figure. Rows 1 and 2 keep their daily series because there the
    # cloud is legible and worth seeing.
    both = np.concatenate([sm['F_net'].to_numpy(), sm['F_exch'].to_numpy()])
    lo, hi = np.nanmin(both), np.nanmax(both)
    mid, half = (lo + hi) / 2, max((hi - lo) / 2, 1e-9) * 1.4
    ax.set_ylim(mid - half, mid + half)

for k in axes:
    axes[k].grid(color='lightgray', linestyle='--', alpha=0.5)
    axes[k].margins(x=0.01)
    axes[k].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axes[k].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

fig.suptitle('Oxygen-coordinate exchange flow at the Penn Cove sections   '
             '(positive = into the cove)\n'
             'a-f  thin = Godin-filtered, daily-subsampled, thick = %d-day '
             'rolling mean   |   g-i  %d-day rolling mean only'
             % (args.lp, args.lp), fontsize=12)

for k, letter in zip([r + '_' + sn for r in ['Q', 'O', 'F'] for sn in SECTS],
                     'abcdefghi'):
    axes[k].text(0.008, 1.02, letter, transform=axes[k].transAxes,
                 fontsize=13, fontweight='bold', va='bottom')

fn_out = out_dir / '20260917_bulk_plot_avg_DO_lp.png'
fig.savefig(fn_out, dpi=200, bbox_inches='tight', transparent=True)
print('\nsaved %s' % fn_out)
