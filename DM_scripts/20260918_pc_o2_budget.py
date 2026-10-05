"""
Penn Cove oxygen budget, whole cove and bottom layer.

Closes two budgets on the same daily axis:

    WHOLE COVE      dI/dt      = F_mouth        + internal
    BOTTOM LAYER    dI_bot/dt  = F_mouth,bot    + (vertical + bio)

I is the oxygen inventory of the cove [g O2], F the net oxygen flux through
pc_lp, and the last column is in both cases a RESIDUAL -- everything the other
two terms do not account for. For the whole cove that residual is the net
internal source (production minus respiration minus SOD, plus air-sea). For the
bottom layer it additionally contains the vertical exchange with the layer
above, which is why it cannot be read as respiration.

WHY THE BOTTOM LAYER IS THE ONE THAT MATTERS
The whole-cove residual is POSITIVE in every season (Winter +63, Spring +90,
Low-DO +126 g/s): depth integrated, the cove produces more oxygen than it
consumes all year and exports the surplus. So the whole-cove budget cannot
explain hypoxia at all. Splitting vertically reverses the sign of both the
flux and the residual, and that is where the hypoxia signal lives.

CONTROL VOLUME
The cove is the box's wet cells with xi_rho <= 30, which slices off the 13
Saratoga cells the box footprint necessarily includes (see the pc_cove job
comment in LO_user/extract/box/job_definitions.py). That gives 317 cells and a
plan area of 1.273e7 m2, which matches the three tef2 cove segments exactly,
so the box inventory and the section fluxes refer to the same volume. The
whole-cove inventory agrees with the segments-file route to 0.02%.

THE LAYER BOUNDARY MOVES
The bottom layer is a fixed FRACTION of the local water column measured from
the bed, so its top follows the bathymetry and breathes with zeta. dI_bot/dt
therefore includes flux through a moving surface, which lands in the residual.
Small for a subtidal mean, but it is there, and it is a reason not to read
R_bot as a physical rate.

STATE AND FLUX PAIRING
The inventory comes from the box, which is ocean_his -- instantaneous, hourly.
The flux comes from the tef2 section extraction, which is ocean_avg -- an
average over each hour. So the state is DIFFERENCED across the hour that the
flux averages over, np.diff(I)/3600, before either is Godin filtered. This is
the same argument as the docstring of LO_user/extract/tef2/extract_segments_SV.py.
Getting it wrong puts a spurious residual straight into the term of interest.

INPUTS
    box      LO_output/extract/[gtx]/box/pc_cove_o2_[ds0]_[ds1].nc
    section  LO_output/extract/[gtx]/tef2/extractions_avg_[ds0]_[ds1]/pc_lp.nc

run 20260918_pc_o2_budget.py
run 20260918_pc_o2_budget.py -frac 0.25
"""
import argparse
import importlib.util
from pathlib import Path
from time import time as clock

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
parser.add_argument('-sect', default='pc_lp', type=str,
                    help='the section bounding the volume')
parser.add_argument('-frac', default=1/3., type=float,
                    help='bottom layer as a fraction of the water column; '
                         'ignored when -zint is given')
parser.add_argument('-zint', default=8.0, type=float,
                    help='FLAT interface depth [m below surface] defining the '
                         'bottom layer. Set 0 to fall back on -frac (no '
                         'vertical flux term is available in that case).')
parser.add_argument('-lp', default=30, type=int,
                    help='window in days for the centred rolling mean')
parser.add_argument('-nchunk', default=1500, type=int,
                    help='time steps per chunk when reading the box')
args = parser.parse_args()

_fn = Path(__file__).parent / '20260916_exchange_fun.py'
_spec = importlib.util.spec_from_file_location('exchange_fun', _fn)
xfun = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xfun)

Ldir = Lfun.Lstart(gridname='wb1')
CONV = 31.998 / 1000            # mmol -> g of O2
PAD = xfun.GODIN_PAD
XI_COVE = 30                    # slice off the Saratoga cells in the box
SEASON = {12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Winter',
          4: 'Spring', 5: 'Spring', 6: 'Spring', 7: 'Spring',
          8: 'Low-DO', 9: 'Low-DO', 10: 'Low-DO', 11: 'Low-DO'}
ORD = ['Winter', 'Spring', 'Low-DO']
out_dir = Path.home() / 'Desktop' / 'pltz'
Lfun.make_dir(out_dir)

def _interp_to_z(zsrc, vsrc, zt):
    """
    Linear interpolation of vsrc onto the level zt, per column and per time.

    zsrc and vsrc are (t, k, y, x) with zsrc increasing along k. Returns
    (t, y, x), nan where zt is outside the column -- i.e. where the bed is
    shallower than the interface, which is the 5% of cove area with h < 8 m.
    """
    nk = zsrc.shape[1]
    kb = (zsrc < zt).sum(axis=1) - 1
    good = (kb >= 0) & (kb < nk - 1)
    k0 = np.clip(kb, 0, nk - 2)[:, np.newaxis]
    k1 = k0 + 1
    z0 = np.take_along_axis(zsrc, k0, axis=1)[:, 0]
    z1 = np.take_along_axis(zsrc, k1, axis=1)[:, 0]
    v0 = np.take_along_axis(vsrc, k0, axis=1)[:, 0]
    v1 = np.take_along_axis(vsrc, k1, axis=1)[:, 0]
    dz = z1 - z0
    f = np.divide(zt - z0, dz, out=np.zeros_like(dz), where=dz != 0)
    return np.where(good, v0 + f * (v1 - v0), np.nan)


# ------------------------------------------------------- inventory, box ----
box_fn = (Ldir['LOo'] / 'extract' / args.gtagex / 'box'
          / ('pc_cove_o2_' + args.ds0 + '_' + args.ds1 + '.nc'))
if not box_fn.is_file():
    raise SystemExit('missing %s\nrun extract_box.py -job pc_cove_o2 -lt hourly0 '
                     'on apogee and bring it back' % box_fn)
bx = xr.open_dataset(box_fn)
USE_Z = args.zint > 0
vel_fn = box_fn.parent / ('pc_cove_' + args.ds0 + '_' + args.ds1 + '.nc')
if USE_Z:
    if not vel_fn.is_file():
        raise SystemExit('missing %s -- needed for w' % vel_fn)
    vb = xr.open_dataset(vel_fn)
    if not np.array_equal(vb.ocean_time.to_numpy(), bx.ocean_time.to_numpy()):
        raise SystemExit('velocity and oxygen boxes have different time axes')
    if not np.array_equal(np.asarray(vb.h), np.asarray(bx.h)):
        raise SystemExit('velocity and oxygen boxes are on different grids')
msk = np.asarray(bx.mask_rho).astype(bool)
xi = np.arange(bx.sizes['xi_rho'])[None, :] * np.ones((bx.sizes['eta_rho'], 1))
cove = msk & (xi <= XI_COVE)
# plan area, zeroed outside the cove so the sums below need no further masking
dA = (1 / np.asarray(bx.pm)) * (1 / np.asarray(bx.pn)) * cove

NT = bx.sizes['ocean_time']
I_all = np.zeros(NT); I_bot = np.zeros(NT)
V_all = np.zeros(NT); V_bot = np.zeros(NT)
W_adv = np.zeros(NT)            # resolved vertical advective O2 flux [g/s]
print('reading the box in chunks of %d ...' % args.nchunk)
t0 = clock()
for s in range(0, NT, args.nchunk):
    sl = slice(s, min(s + args.nchunk, NT))
    zw = bx.z_w.isel(ocean_time=sl).to_numpy()
    # land cells are nan and their area is 0, so nan*0 = nan would poison the
    # sums -- zero the tracer rather than relying on the area mask
    o = np.nan_to_num(bx.oxygen.isel(ocean_time=sl).to_numpy().astype(float),
                      nan=0.0)
    DZ = np.diff(zw, axis=1)
    dV = DZ * dA[None, None, :, :]
    if USE_Z:
        # FLAT interface. The flux through it is exactly w*O*dA -- no moving or
        # sloping surface corrections, which is the whole reason for preferring
        # a fixed depth over a fraction of the column here.
        zr = 0.5 * (zw[:, :-1] + zw[:, 1:])
        bot = zr <= -args.zint
        w_i = _interp_to_z(zw, vb.w.isel(ocean_time=sl).to_numpy().astype(float),
                           -args.zint)
        o_i = _interp_to_z(zr, o, -args.zint)
        # positive w carries oxygen UP and OUT of the bottom layer, so the
        # contribution to the layer is the negative of it
        W_adv[sl] = -np.nansum(np.where(np.isfinite(w_i * o_i),
                                        w_i * o_i * dA[None, :, :], 0.0),
                               axis=(1, 2)) * CONV
    else:
        hab = np.cumsum(DZ, axis=1) - DZ / 2        # height above bed
        bot = (hab / DZ.sum(axis=1)[:, None, :, :]) <= args.frac
    I_all[sl] = (o * dV).sum(axis=(1, 2, 3)) * CONV
    I_bot[sl] = np.where(bot, o * dV, 0.0).sum(axis=(1, 2, 3)) * CONV
    V_all[sl] = dV.sum(axis=(1, 2, 3))
    V_bot[sl] = np.where(bot, dV, 0.0).sum(axis=(1, 2, 3))
print('  %.0f s' % (clock() - t0))
bx.close()

# ---------------------------------------------------- flux, tef2 section ----
ds = xr.open_dataset(xfun.section_dir(args.gtagex, args.ds0, args.ds1,
                                      Ldir=Ldir) / (args.sect + '.nc'))
sgn = xfun.INFLOW_SIGN[args.sect]
q = sgn * ds.q.to_numpy()
o = ds.oxygen.to_numpy().astype(float)
DZ = ds.DZ.to_numpy()
tt = ds.time.to_numpy()
ds.close()
if USE_Z:
    zw_s = np.cumsum(DZ, axis=1) - np.asarray(
        xr.open_dataset(xfun.section_dir(args.gtagex, args.ds0, args.ds1,
                                         Ldir=Ldir) / (args.sect + '.nc')).h)[None, None, :]
    sbot = (zw_s - DZ / 2) <= -args.zint
else:
    hab = np.cumsum(DZ, axis=1) - DZ / 2
    sbot = (hab / DZ.sum(axis=1)[:, np.newaxis, :]) <= args.frac
F_all = np.nansum(q * o, axis=(1, 2)) * CONV
F_bot = np.nansum(np.where(sbot, q * o, 0.0), axis=(1, 2)) * CONV

# ------------------------------------------------------------- budgets ----
# instantaneous state differenced across the hour the flux averages over
dI_all = np.diff(I_all) / 3600.0
dI_bot = np.diff(I_bot) / 3600.0
assert len(dI_all) == len(F_all), 'box and section time axes do not pair'

g = lambda x: xfun.godin_daily(x, pad=PAD)
t = pd.to_datetime(xfun.daily_time(tt, pad=PAD))
cols = {'dIall': g(dI_all), 'F_all': g(F_all),
        'dIbot': g(dI_bot), 'F_bot': g(F_bot)}
if USE_Z:
    cols['W_adv'] = g(W_adv[1:])         # drop the first hour, as dI did
df = pd.DataFrame(cols, index=t)
df['R_all'] = df.dIall - df.F_all        # net internal source, whole cove
if USE_Z:
    # the resolved vertical advection is now explicit, so what is left is
    # vertical DIFFUSION plus biology -- still two things, but one fewer
    df['R_bot'] = df.dIbot - df.F_bot - df.W_adv
else:
    df['R_bot'] = df.dIbot - df.F_bot
df['season'] = pd.Series(df.index.month, index=df.index).map(SEASON)

A = dA.sum()
print('\ncontrol volume: %d cove cells, plan area %.3e m2, mean volume %.3e m3'
      % (cove.sum(), A, V_all.mean()))
print('bottom %.2f of the column = %.1f%% of the volume, mean [O2] %.2f mg/L '
      '(whole cove %.2f)' % (args.frac, 100 * V_bot.mean() / V_all.mean(),
                             I_bot.mean() / V_bot.mean(), I_all.mean() / V_all.mean()))
print('\nPENN COVE O2 BUDGETS [g/s], + = into the volume')
print('\nWHOLE COVE      dI/dt = F + internal')
print(df.groupby('season')[['dIall', 'F_all', 'R_all']].mean().round(1)
      .reindex(ORD).to_string())
if USE_Z:
    print('\nBOTTOM LAYER (below %.1f m)   dI/dt = F + W_adv + (diffusion + bio)'
          % args.zint)
    print(df.groupby('season')[['dIbot', 'F_bot', 'W_adv', 'R_bot']].mean()
          .round(1).reindex(ORD).to_string())
else:
    print('\nBOTTOM LAYER    dI/dt = F + (vertical + bio)')
    print(df.groupby('season')[['dIbot', 'F_bot', 'R_bot']].mean().round(1)
          .reindex(ORD).to_string())
print('\nR_bot per unit area [g O2 m-2 d-1]:')
print((df.groupby('season')['R_bot'].mean() * 86400 / A).round(2)
      .reindex(ORD).to_string())
print('\nrecord means [g/s]:')
print(df[[c for c in ['dIall', 'F_all', 'R_all', 'dIbot', 'F_bot', 'W_adv',
                      'R_bot'] if c in df]].mean().round(2).to_string())
df.to_csv(out_dir / '20260918_pc_o2_budget.csv')

# -------------------------------------------------------------- figure ----
def smooth(s):
    return s.rolling(args.lp, center=True,
                     min_periods=max(args.lp // 3, 1)).mean()

if USE_Z:
    ROWS = [('whole cove', ['dIall', 'F_all', 'R_all'],
             [r'$dI/dt$', r'$F$ at %s' % args.sect, 'internal (residual)']),
            ('below %.0f m' % args.zint, ['dIbot', 'F_bot', 'W_adv', 'R_bot'],
             [r'$dI/dt$', r'$F$ at %s' % args.sect, r'$-wO$ at the interface',
              'diffusion + bio (residual)'])]
else:
    ROWS = [('whole cove', ['dIall', 'F_all', 'R_all'],
             [r'$dI/dt$', r'$F$ at %s' % args.sect, 'internal (residual)']),
            ('bottom %g of the column' % args.frac, ['dIbot', 'F_bot', 'R_bot'],
             [r'$dI/dt$', r'$F$ at %s' % args.sect, 'vertical + bio (residual)'])]
COLS = ['#000000', '#0072B2', '#009E73', '#D55E00']
LOWDO = [8, 11]

plt.close('all')
fig, axes = plt.subplots(2, 2, figsize=(16, 7),
                         gridspec_kw=dict(width_ratios=[3, 1]),
                         sharex='col', layout='constrained')
for r, (lab, keys, names) in enumerate(ROWS):
    ax = axes[r, 0]
    for yr in sorted(set(df.index.year)):
        ax.axvspan(pd.Timestamp(yr, LOWDO[0], 1),
                   pd.Timestamp(yr, LOWDO[1], 1) + pd.offsets.MonthEnd(1),
                   color='0.85', alpha=0.45, lw=0, zorder=0)
    cyc = COLS if len(keys) == 4 else [COLS[0], COLS[1], COLS[3]]
    for k, c, nm in zip(keys, cyc, names):
        ax.plot(df.index, df[k], lw=0.6, alpha=0.2, color=c, zorder=2)
        ax.plot(df.index, smooth(df[k]), lw=2, color=c, label=nm, zorder=3)
    ax.axhline(0, color='k', lw=0.8, alpha=0.5, zorder=1)
    ax.set_ylabel('%s\nO$_2$ [g s$^{-1}$]\n+ = into the volume' % lab)
    ax.grid(color='lightgray', ls='--', alpha=0.5)
    ax.margins(x=0.01)
    ax.legend(loc='upper left', ncol=len(keys), fontsize=9, framealpha=0.9)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    if r == 0:
        ax.set_title('daily (thin), %d-day rolling mean (thick); '
                     'shading = Low-DO season' % args.lp, fontsize=10)

    ax = axes[r, 1]
    clim = df.groupby(df.index.month)[keys].mean()
    for k, c in zip(keys, cyc):
        ax.plot(clim.index, clim[k], '-o', ms=4, lw=2, color=c)
    ax.axvspan(LOWDO[0] - 0.5, LOWDO[1] + 0.5, color='0.85', alpha=0.45, lw=0,
               zorder=0)
    ax.axhline(0, color='k', lw=0.8, alpha=0.5)
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(list('JFMAMJJASOND'))
    ax.grid(color='lightgray', ls='--', alpha=0.5)
    if r == 0:
        ax.set_title('monthly climatology', fontsize=10)

fig.suptitle('Penn Cove oxygen budget, %s to %s\n'
             'depth integrated the cove is a net oxygen SOURCE all year; '
             'the bottom layer is supplied horizontally and drained upward'
             % (args.ds0, args.ds1), fontsize=13)
fn_out = out_dir / '20260918_pc_o2_budget.png'
fig.savefig(fn_out, dpi=200, bbox_inches='tight', transparent=True)
print('\nsaved %s' % fn_out)
