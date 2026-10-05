"""
Subtidal oxygen flux at the Penn Cove sections as a time series, split into
its advective and tidal-pumping parts.

Same decomposition as 20260917_pc_o2_flux_structure.py, which shows it resolved
in (z, p) and time averaged. This is the complement: summed over the section
and resolved in time.

    <q O>   =   <q> <O>   +   <q' O'>
     total      advective    tidal pumping

<> is the Godin average; <O> is area weighted. Only total and advective are
measured -- tidal pumping is the residual, so it absorbs anything the advective
term gets wrong. See the sibling script's docstring for the full argument.

WHY THE TWO TERMS ARE WORTH SEEING SEPARATELY IN TIME
They oppose each other and nearly cancel. At pc_lp the two-year means are
advective -308 and tidal +215 g s-1, so the -93 that survives is a small
difference of two large terms. A time series of the total alone hides that
completely, and hides the fact that the cancellation is stable: tidal pumping
offsets 82 / 67 / 68 % of the advective term in Winter / Spring / Low-DO. Any
story about a change in the total has to say which of the two moved.

LEFT the daily series (thin) with a centred rolling mean (thick). Every point
is already Godin filtered and daily subsampled, so the rolling mean is removing
the spring-neap modulation, not the tide.

RIGHT the monthly climatology of the same three terms, both years pooled.

SIGN positive is INTO the cove (INFLOW_SIGN), so negative = export. Shaded
bands mark the Low-DO season (Aug-Nov).

run 20260917_pc_o2_flux_decomp_ts.py
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
parser.add_argument('-sect', default='pc_cp,pc_lj,pc_lp', type=str)
parser.add_argument('-lp', default=30, type=int,
                    help='window in days for the centred rolling mean')
parser.add_argument('-layer', default='all', type=str,
                    help="'all', or a fraction of the water column measured from "
                         "the BED, e.g. 'bot0.33' for the bottom third, "
                         "'top0.33' for the top third")
args = parser.parse_args()

_fn = Path(__file__).parent / '20260916_exchange_fun.py'
_spec = importlib.util.spec_from_file_location('exchange_fun', _fn)
xfun = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xfun)

Ldir = Lfun.Lstart(gridname='wb1')
in_dir = xfun.section_dir(args.gtagex, args.ds0, args.ds1, Ldir=Ldir)
out_dir = Path.home() / 'Desktop' / 'pltz'
Lfun.make_dir(out_dir)

SECTS = [s.strip() for s in args.sect.split(',') if s.strip()]
CONV = 31.998 / 1000
PAD = xfun.GODIN_PAD
TERMS = [('total', r'total  $\langle qO \rangle$', '#000000'),
         ('adv', r'advective  $\langle q \rangle \langle O \rangle$', '#0072B2'),
         ('tid', r"tidal pumping  $\langle q'O' \rangle$", '#D55E00')]
LOWDO = [8, 9, 10, 11]
def smooth(s):
    return s.rolling(args.lp, center=True,
                     min_periods=max(args.lp // 3, 1)).mean()


D = dict()
for sn in SECTS:
    ds = xr.open_dataset(in_dir / (sn + '.nc'))
    sgn = xfun.INFLOW_SIGN[sn]
    q = sgn * ds.q.to_numpy()
    o = ds.oxygen.to_numpy().astype(float)
    ds_DZ = ds.DZ.to_numpy()
    dA = ds_DZ * ds.dd.to_numpy()[np.newaxis, np.newaxis, :]
    t = pd.to_datetime(xfun.daily_time(ds.time.to_numpy()))
    ds.close()

    # per-cell subtidal terms [g s-1], still resolved in (z, p)
    qo = xfun.godin_daily(q * o, pad=PAD) * CONV
    q_lp = xfun.godin_daily(q, pad=PAD)
    dA_lp = xfun.godin_daily(dA, pad=PAD)
    with np.errstate(invalid='ignore', divide='ignore'):
        o_lp = xfun.godin_daily(o * dA, pad=PAD) / dA_lp
    adv = q_lp * o_lp * CONV

    # Layer mask. The per-cell subtidal terms above do NOT depend on the mask --
    # they are properties of each cell. The mask only chooses which cells to add
    # up, so a layer is just a subset of the same numbers and the layers sum
    # back to the whole-section total exactly.
    #
    # The boundary is a fraction of the local water column measured from the
    # bed, so it follows the bathymetry across the section (16 m at the ends,
    # 27 m in the channel) instead of cutting a flat plane through it. Built
    # from the DAILY DZ, so the layer is fixed within a day and the tide does
    # not slosh cells across the boundary.
    DZ_lp = xfun.godin_daily(ds_DZ, pad=PAD)
    hab = np.cumsum(DZ_lp, axis=1) - DZ_lp / 2          # height above bed
    frac = hab / DZ_lp.sum(axis=1)[:, np.newaxis, :]    # 0 at bed, 1 at surface
    if args.layer == 'all':
        mask = np.ones_like(frac, dtype=bool)
    elif args.layer.startswith('bot'):
        mask = frac <= float(args.layer[3:])
    elif args.layer.startswith('top'):
        mask = frac >= 1 - float(args.layer[3:])
    else:
        raise SystemExit("-layer must be 'all', 'botX' or 'topX'")

    tot = np.where(mask, qo, 0.0).sum(axis=(1, 2))
    advs = np.where(mask, adv, 0.0).sum(axis=(1, 2))
    D[sn] = pd.DataFrame({'total': tot, 'adv': advs, 'tid': tot - advs}, index=t)

# ---------------------------------------------------------------- summary ---
print('\nO2 flux decomposition [g/s], + = into the cove, %s .. %s   layer=%s'
      % (args.ds0, args.ds1, args.layer))
summ = pd.DataFrame({sn: {'total': D[sn]['total'].mean(),
                          'advective': D[sn]['adv'].mean(),
                          'tidal': D[sn]['tid'].mean(),
                          'tidal/|adv| %': 100 * D[sn]['tid'].mean()
                          / abs(D[sn]['adv'].mean())}
                     for sn in SECTS})
print(summ.round(1).to_string())
pd.concat(D, axis=1).to_csv(out_dir / ('20260917_pc_o2_flux_decomp_ts_'
                                       + args.layer + '.csv'))

# ----------------------------------------------------------------- figure ---
plt.close('all')
fig, axes = plt.subplots(len(SECTS), 2, figsize=(16, 3.2 * len(SECTS)),
                         gridspec_kw=dict(width_ratios=[3, 1]),
                         sharex='col', layout='constrained')

for r, sn in enumerate(SECTS):
    df = D[sn]

    ax = axes[r, 0]
    # shade the Low-DO season
    for yr in sorted(set(df.index.year)):
        ax.axvspan(pd.Timestamp(yr, LOWDO[0], 1),
                   pd.Timestamp(yr, LOWDO[-1], 1) + pd.offsets.MonthEnd(1),
                   color='0.85', alpha=0.45, lw=0, zorder=0)
    for key, lab, col in TERMS:
        ax.plot(df.index, df[key], lw=0.6, alpha=0.2, color=col, zorder=2)
        ax.plot(df.index, smooth(df[key]), lw=2, color=col, label=lab, zorder=3)
    ax.axhline(0, color='k', lw=0.8, alpha=0.5, zorder=1)
    ax.set_ylabel('%s\nO$_2$ flux [g s$^{-1}$]\n+ = into the cove' % sn)
    ax.grid(color='lightgray', ls='--', alpha=0.5)
    ax.margins(x=0.01)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    if r == 0:
        ax.legend(loc='upper left', ncol=3, fontsize=9, framealpha=0.9)
        ax.set_title('daily (thin) and %d-day rolling mean (thick); '
                     'shading = Low-DO season (Aug-Nov)' % args.lp, fontsize=10)

    ax = axes[r, 1]
    clim = df.groupby(df.index.month).mean()
    for key, lab, col in TERMS:
        ax.plot(clim.index, clim[key], '-o', ms=4, lw=2, color=col)
    ax.axhline(0, color='k', lw=0.8, alpha=0.5)
    ax.axvspan(LOWDO[0] - 0.5, LOWDO[-1] + 0.5, color='0.85', alpha=0.45, lw=0,
               zorder=0)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(list('JFMAMJJASOND'))
    ax.grid(color='lightgray', ls='--', alpha=0.5)
    if r == 0:
        ax.set_title('monthly climatology', fontsize=10)

LAYLAB = {'all': 'whole section'}.get(
    args.layer, 'bottom %g of the water column' % float(args.layer[3:])
    if args.layer.startswith('bot') else 'top %g' % float(args.layer[3:]))
fig.suptitle('Subtidal oxygen flux at the Penn Cove sections: %s\n'
             'split into advective and tidal-pumping parts, which oppose each '
             'other; the total is the difference that survives' % LAYLAB,
             fontsize=13)
fn_out = out_dir / ('20260917_pc_o2_flux_decomp_ts_' + args.layer + '.png')
fig.savefig(fn_out, dpi=200, bbox_inches='tight', transparent=True)
print('\nsaved %s' % fn_out)
