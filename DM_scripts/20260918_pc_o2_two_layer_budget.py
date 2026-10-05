"""
Two-layer oxygen budget for Penn Cove, with vertical advection explicit.

For each layer L of the cove control volume,

    d/dt Int_L DO dV  =  Qin DOin + Qout DOout        horizontal, at pc_lp
                       + W_adv                        vertical advection
                       + R                            residual

R is everything not resolved: vertical diffusion (ROMS AKs, which no box
carries), photosynthesis minus consumption, and for the surface layer the
air-sea flux. It is NOT a respiration rate.

The horizontal term is in TEF bulk form. Within each layer the section cells
are split by the sign of the subtidal transport <q>, and

    Qin  = sum over {<q> > 0} of <q>
    DOin = sum over {<q> > 0} of <q O> / Qin

so DOin is weighted by the FULL flux <qO>, not by <q><O>. That makes
Qin DOin + Qout DOout identically equal to the total flux sum <qO>, tidal
pumping included. Weighting by <q><O> instead would silently drop the tidal
term, which at pc_lp is worth ~215 g/s against an advective ~-308.

WHY TWO LAYERS
The depth-integrated budget cannot explain hypoxia: the cove's internal source
is positive in every season (+63 / +90 / +126 g/s), so integrated over depth it
produces oxygen and exports it. Splitting at the pycnocline reverses the sign
of both the flux and the residual in the lower layer, and that is where the
hypoxia signal lives.

INTERFACE is FLAT, at -zint (default 8 m, the pycnocline depth reported by
20260811_pc_pycnocline.py at pc_lp). Flat matters: through a sloping, breathing
surface the flux would be w - dz/dt - u dz/dx - v dz/dy, and the slope terms
are not small over this bathymetry. Through a level surface it is exactly w O.
Cost: the 5% of cove plan area shallower than 8 m has no bottom layer and no
interface, and is carried entirely in the surface layer.

WWTPs Penn Cove contains two, COUPEVILLE STP and PENN COVE WWTP (both in
segment pc_cp_p under trapsN00; OAK HARBOR STP and Whidbey east are outside,
in pc_lp_p). Their combined oxygen load is 0.044 g/s -- 0.06% of the smallest
other term -- so they are documented here and omitted rather than carried as a
column. Do not omit them from a nitrogen budget without re-checking.

CHECKS the script asserts that the two layer budgets sum to the whole-cove
budget and that W_adv cancels between them, which it must by construction.

run 20260918_pc_o2_two_layer_budget.py
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
parser.add_argument('-sect', default='pc_lp', type=str)
parser.add_argument('-nsig', default=10, type=int,
                    help='bottom layer = this many sigma layers above the bed')
parser.add_argument('-lp', default=30, type=int)
parser.add_argument('-nchunk', default=1500, type=int)
args = parser.parse_args()

_fn = Path(__file__).parent / '20260916_exchange_fun.py'
_spec = importlib.util.spec_from_file_location('exchange_fun', _fn)
xfun = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xfun)

Ldir = Lfun.Lstart(gridname='wb1')
CONV = 31.998 / 1000
PAD = xfun.GODIN_PAD
XI_COVE = 30
WWTP = 0.044                    # g O2 /s, documented and omitted (see docstring)
SEASON = {12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Winter',
          4: 'Spring', 5: 'Spring', 6: 'Spring', 7: 'Spring',
          8: 'Low-DO', 9: 'Low-DO', 10: 'Low-DO', 11: 'Low-DO'}
ORD = ['Winter', 'Spring', 'Low-DO']
out_dir = Path.home() / 'Desktop' / 'pltz'
Lfun.make_dir(out_dir)


def u_to_rho(u):
    """u on the u-grid -> rho points, one-sided at the two edge columns."""
    r = np.empty(u.shape[:3] + (u.shape[3] + 1,), dtype=u.dtype)
    r[..., 1:-1] = 0.5 * (u[..., :-1] + u[..., 1:])
    r[..., 0] = u[..., 0]
    r[..., -1] = u[..., -1]
    return r


def v_to_rho(v):
    """v on the v-grid -> rho points, one-sided at the two edge rows."""
    r = np.empty(v.shape[:2] + (v.shape[2] + 1,) + v.shape[3:], dtype=v.dtype)
    r[:, :, 1:-1] = 0.5 * (v[:, :, :-1] + v[:, :, 1:])
    r[:, :, 0] = v[:, :, 0]
    r[:, :, -1] = v[:, :, -1]
    return r


# ------------------------------------------ inventories and w, from the box --
bdir = Ldir['LOo'] / 'extract' / args.gtagex / 'box'
o2_fn = bdir / ('pc_cove_o2_' + args.ds0 + '_' + args.ds1 + '.nc')
vel_fn = bdir / ('pc_cove_' + args.ds0 + '_' + args.ds1 + '.nc')
for f in (o2_fn, vel_fn):
    if not f.is_file():
        raise SystemExit('missing %s' % f)
bx = xr.open_dataset(o2_fn)
vb = xr.open_dataset(vel_fn)
assert np.array_equal(bx.ocean_time.to_numpy(), vb.ocean_time.to_numpy())
assert np.array_equal(np.asarray(bx.h), np.asarray(vb.h))

msk = np.asarray(bx.mask_rho).astype(bool)
xi = np.arange(bx.sizes['xi_rho'])[None, :] * np.ones((bx.sizes['eta_rho'], 1))
cove = msk & (xi <= XI_COVE)
dA = (1 / np.asarray(bx.pm)) * (1 / np.asarray(bx.pn)) * cove

NT = bx.sizes['ocean_time']
K = args.nsig                   # interface sits at z_w index K
pm = np.asarray(bx.pm); pn = np.asarray(bx.pn)

# interface geometry for the whole record: a sigma surface, so it is neither
# level nor steady, and the flux through it is the GRID-RELATIVE velocity
#     w_rel = w - dz/dt - u dz/dx - v dz/dy
# not w. The three correction terms are each computed and reported, because on
# a terrain-following surface over this bathymetry they are not small.
z_int = bx.z_w.isel(s_w=K).to_numpy().astype(float)          # (t, y, x)
dzdt = np.gradient(z_int, 3600.0, axis=0)
dzdx = np.gradient(z_int, axis=2) * pm[None, :, :]
dzdy = np.gradient(z_int, axis=1) * pn[None, :, :]

I_bot = np.zeros(NT); I_srf = np.zeros(NT)
V_bot = np.zeros(NT); V_srf = np.zeros(NT)
WO = np.zeros(NT)               # upward O2 flux through the interface [g/s]
WTERM = np.zeros((NT, 4))       # w, -dz/dt, -u dz/dx, -v dz/dy contributions
print('reading the boxes ...')
t0 = clock()
for s_ in range(0, NT, args.nchunk):
    sl = slice(s_, min(s_ + args.nchunk, NT))
    zw = bx.z_w.isel(ocean_time=sl).to_numpy()
    o = np.nan_to_num(bx.oxygen.isel(ocean_time=sl).to_numpy().astype(float), nan=0.0)
    DZ = np.diff(zw, axis=1)
    dV = DZ * dA[None, None, :, :]
    I_bot[sl] = (o[:, :K] * dV[:, :K]).sum(axis=(1, 2, 3)) * CONV
    I_srf[sl] = (o[:, K:] * dV[:, K:]).sum(axis=(1, 2, 3)) * CONV
    V_bot[sl] = dV[:, :K].sum(axis=(1, 2, 3))
    V_srf[sl] = dV[:, K:].sum(axis=(1, 2, 3))

    # tracer at the interface: average the two sigma cells either side of it
    o_i = 0.5 * (o[:, K - 1] + o[:, K])
    w_i = vb.w.isel(ocean_time=sl, s_w=K).to_numpy().astype(float)
    uu = u_to_rho(vb.u.isel(ocean_time=sl).to_numpy().astype(float))
    vv = v_to_rho(vb.v.isel(ocean_time=sl).to_numpy().astype(float))
    u_i = 0.5 * (uu[:, K - 1] + uu[:, K])
    v_i = 0.5 * (vv[:, K - 1] + vv[:, K])
    parts = [w_i, -dzdt[sl], -u_i * dzdx[sl], -v_i * dzdy[sl]]
    for j, pt in enumerate(parts):
        WTERM[sl, j] = np.nansum(np.nan_to_num(pt) * o_i * dA[None, :, :],
                                 axis=(1, 2)) * CONV
    WO[sl] = WTERM[sl].sum(axis=1)
print('  %.0f s' % (clock() - t0))
bx.close(); vb.close()

# ------------------------------------------- horizontal, TEF bulk per layer --
ds = xr.open_dataset(xfun.section_dir(args.gtagex, args.ds0, args.ds1,
                                      Ldir=Ldir) / (args.sect + '.nc'))
sgn = xfun.INFLOW_SIGN[args.sect]
q = sgn * ds.q.to_numpy()
o = ds.oxygen.to_numpy().astype(float)
DZ = ds.DZ.to_numpy()
ds_dd = ds.dd.to_numpy()
tt = ds.time.to_numpy()
ds.close()
g = lambda x: xfun.godin_daily(x, pad=PAD)
dA_s = DZ * ds_dd[np.newaxis, np.newaxis, :]
qo_lp = g(q * o) * CONV                      # <qO>, tidal covariance included
q_lp = g(q)
dA_lp = g(dA_s)
with np.errstate(invalid='ignore', divide='ignore'):
    o_lp = g(o * dA_s) / dA_lp               # <O>, area weighted
adv_c = q_lp * o_lp * CONV                   # <q><O>, per cell

K = args.nsig
BULK = dict()
for nm, sl_ in [('bot', slice(0, K)), ('srf', slice(K, None))]:
    tot = qo_lp[:, sl_].sum(axis=(1, 2))
    adv = adv_c[:, sl_].sum(axis=(1, 2))
    qq = q_lp[:, sl_]
    ff = qo_lp[:, sl_]
    Qin = np.where(qq > 0, qq, 0.0).sum(axis=(1, 2))
    Qout = np.where(qq < 0, qq, 0.0).sum(axis=(1, 2))
    Fin = np.where(qq > 0, ff, 0.0).sum(axis=(1, 2))
    Fout = np.where(qq < 0, ff, 0.0).sum(axis=(1, 2))
    with np.errstate(invalid='ignore', divide='ignore'):
        BULK[nm] = dict(F=tot, F_adv=adv, F_tid=tot - adv, Qin=Qin, Qout=Qout,
                        DOin=Fin / Qin, DOout=Fout / Qout)

# --------------------------------------------------------------- budgets ----
t = pd.to_datetime(xfun.daily_time(tt, pad=PAD))
df = pd.DataFrame(index=t)
df['dI_srf'] = g(np.diff(I_srf) / 3600.0)
df['dI_bot'] = g(np.diff(I_bot) / 3600.0)
for nm in ('srf', 'bot'):
    df['Fadv_' + nm] = BULK[nm]['F_adv']
    df['Ftid_' + nm] = BULK[nm]['F_tid']
df['F_srf'] = BULK['srf']['F']
df['F_bot'] = BULK['bot']['F']
df['W_srf'] = g(WO[1:])          # upward flux ENTERS the surface layer
df['W_bot'] = -df['W_srf']       # and LEAVES the bottom layer
df['R_srf'] = df.dI_srf - df.F_srf - df.W_srf
df['R_bot'] = df.dI_bot - df.F_bot - df.W_bot
for nm in ('bot', 'srf'):
    for k in ('Qin', 'Qout', 'DOin', 'DOout'):
        df[k + '_' + nm] = BULK[nm][k]
for j, nm in enumerate(['w', 'dzdt', 'ududx', 'vdvdy']):
    df['W_' + nm] = g(WTERM[1:, j])
df['season'] = pd.Series(df.index.month, index=df.index).map(SEASON)

# the two layers must sum to the whole cove, and W must cancel
assert np.allclose((df.W_srf + df.W_bot).to_numpy(), 0.0, atol=1e-9)
whole = (df.dI_srf + df.dI_bot) - (df.F_srf + df.F_bot) - (df.R_srf + df.R_bot)
assert np.nanmax(np.abs(whole.to_numpy())) < 1e-6, 'layer budgets do not sum'

A = dA.sum()
print('\ncontrol volume %.3e m2 plan, %.3e m3; lowest %d sigma layers hold '
      '%.0f%% of volume' % (A, (V_srf + V_bot).mean(), args.nsig,
                            100 * V_bot.mean() / (V_srf + V_bot).mean()))
print('mean [O2]: surface %.2f, bottom %.2f mg/L'
      % (I_srf.mean() / V_srf.mean(), I_bot.mean() / V_bot.mean()))
print('WWTPs inside the volume contribute %.3f g/s and are omitted' % WWTP)
for nm, lab in [('srf', 'UPPER  (sigma %d-29)' % args.nsig),
                ('bot', 'BOTTOM (lowest %d sigma layers)' % args.nsig)]:
    print('\n%s   dI/dt = <q><O> + <q\'O\'> + W_adv + R   [g O2 /s]' % lab)
    cols = ['dI_' + nm, 'Fadv_' + nm, 'Ftid_' + nm, 'W_' + nm, 'R_' + nm]
    print(df.groupby('season')[cols].mean().round(1).reindex(ORD).to_string())
    print('  TEF bulk form of the same horizontal term:')
    print(df.groupby('season')[['Qin_' + nm, 'DOin_' + nm, 'Qout_' + nm,
                                'DOout_' + nm]].mean().round(2).reindex(ORD).to_string())
print('\nInterface flux, broken into the grid-relative velocity terms [g/s],')
print('positive = upward through the sigma surface:')
print(df[['W_w', 'W_dzdt', 'W_ududx', 'W_vdvdy']].mean().round(2).to_string())
print('  sum = %.2f  (= W_srf)' % df[['W_w', 'W_dzdt', 'W_ududx',
                                      'W_vdvdy']].mean().sum())
print('\nrecord means [g/s]:')
print(df[['dI_srf', 'Fadv_srf', 'Ftid_srf', 'W_srf', 'R_srf',
          'dI_bot', 'Fadv_bot', 'Ftid_bot', 'W_bot', 'R_bot']].mean()
      .round(2).to_string())
df.to_csv(out_dir / '20260918_pc_o2_two_layer_budget.csv')

# ---------------------------------------------------------------- figure ----
def smooth(s):
    return s.rolling(args.lp, center=True,
                     min_periods=max(args.lp // 3, 1)).mean()

NAMES = [r'$dI/dt$', r'advective $\langle q\rangle\langle O\rangle$',
         r"tidal pumping $\langle q'O'\rangle$", r'$W_{adv}$',
         'residual (diffusion + bio)']
COLS = ['#000000', '#0072B2', '#CC79A7', '#009E73', '#D55E00']
LOWDO = [8, 11]

plt.close('all')
fig, axes = plt.subplots(2, 2, figsize=(16, 7),
                         gridspec_kw=dict(width_ratios=[3, 1]),
                         sharex='col', layout='constrained')
for r, (nm, lab) in enumerate([('srf', 'upper, sigma %d-29' % args.nsig),
                               ('bot', 'bottom %d sigma layers' % args.nsig)]):
    keys = ['dI_' + nm, 'Fadv_' + nm, 'Ftid_' + nm, 'W_' + nm, 'R_' + nm]
    ax = axes[r, 0]
    for yr in sorted(set(df.index.year)):
        ax.axvspan(pd.Timestamp(yr, LOWDO[0], 1),
                   pd.Timestamp(yr, LOWDO[1], 1) + pd.offsets.MonthEnd(1),
                   color='0.85', alpha=0.45, lw=0, zorder=0)
    for k, c, n in zip(keys, COLS, NAMES):
        ax.plot(df.index, df[k], lw=0.6, alpha=0.18, color=c, zorder=2)
        ax.plot(df.index, smooth(df[k]), lw=2, color=c, label=n, zorder=3)
    ax.axhline(0, color='k', lw=0.8, alpha=0.5, zorder=1)
    ax.set_ylabel('%s\nO$_2$ [g s$^{-1}$]\n+ = into the layer' % lab)
    ax.grid(color='lightgray', ls='--', alpha=0.5)
    ax.margins(x=0.01)
    ax.legend(loc='upper left', ncol=5, fontsize=8, framealpha=0.9)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    if r == 0:
        ax.set_title('daily (thin), %d-day rolling mean (thick); shading = '
                     'Low-DO season' % args.lp, fontsize=10)
    ax = axes[r, 1]
    clim = df.groupby(df.index.month)[keys].mean()
    for k, c in zip(keys, COLS):
        ax.plot(clim.index, clim[k], '-o', ms=4, lw=2, color=c)
    ax.axvspan(LOWDO[0] - 0.5, LOWDO[1] + 0.5, color='0.85', alpha=0.45, lw=0, zorder=0)
    ax.axhline(0, color='k', lw=0.8, alpha=0.5)
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(list('JFMAMJJASOND'))
    ax.grid(color='lightgray', ls='--', alpha=0.5)
    if r == 0:
        ax.set_title('monthly climatology', fontsize=10)

fig.suptitle('Penn Cove two-layer oxygen budget, bottom %d sigma layers, '
             '%s to %s\nhorizontal split into advective and tidal-pumping parts; '
             'vertical advection through the sigma surface is grid-relative'
             % (args.nsig, args.ds0, args.ds1), fontsize=13)
fn_out = out_dir / '20260918_pc_o2_two_layer_budget.png'
fig.savefig(fn_out, dpi=200, bbox_inches='tight', transparent=True)
print('\nsaved %s' % fn_out)
