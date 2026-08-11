"""
When is the velocity profile at pc_lp (Penn Cove mouth) baroclinic, and when is
it barotropic?

THE QUESTION, MADE MEASURABLE
"Barotropic" means the section moves as a slab: one velocity, no sign change
with depth. "Baroclinic" means the profile is sheared enough to reverse, water
going in at one depth and out at another. Both live in the same field, so the
split is a ratio, not a category. For every hour the width-averaged profile

    U(t,z) = sum_p q / sum_p (dd * DZ)                            [m s-1]

is decomposed into its area-weighted depth mean and the deviation from it

    U0(t)  = qnet / area                    the barotropic part
    U'(t,z)= U(t,z) - U0(t)                 the baroclinic part
    Urms   = sqrt( sum_z A U'^2 / A_tot )   its amplitude

and the hour is called BAROTROPIC when |U0| > 2 Urms, BAROCLINIC when
Urms > 2 |U0|, MIXED in between. The continuous version of the same thing is
the barotropic energy fraction fbt = U0^2 / (U0^2 + Urms^2).

WHY THE ANSWER IS NOT "IT DEPENDS ON THE TIDE"
Urms turns out to be almost independent of how hard the tide is running
(corr(Urms, |U0|) = -0.04 hourly) and strongly tied to stratification
(corr = +0.72 on daily means). So the shear is not a frictional boundary layer
dragged along by the barotropic tide -- if it were, it would scale with the
tidal current. It is a separate, density-supported motion sitting underneath a
tide that swings past it.

The harmonic fit (panel a) is what settles it. M2 is barotropic: amplitude
constant above the bottom boundary layer and phase constant with depth. The
DIURNAL band is baroclinic: K1 and S1 both have a node ~10 m below the surface
(56 % of the 23 m column up from the bed) with a ~180 deg phase flip across it,
and they are LARGER at the surface (K1 0.052, S1 0.073 m/s) than M2 is anywhere
(0.042 m/s). Verified face by face -- p=6 alone shows the same node -- so it is
not an artifact of averaging sigma levels across faces of different depth.

The diurnal energy sits in a K1 / P1 / S1 triplet, which is what an
astronomical K1 internal tide looks like when its amplitude is modulated
annually by stratification (K1 +/- SA = P1 and psi1). A purely radiational
(sea-breeze / heating) driver is not a good fit: the diurnal shear is as strong
in winter as in summer, its hour-of-day phase moves with the season, and its
diurnal-band correlation with along-cove wind is only +0.10 (+0.38 at 5 h lag).

FIVE PANELS
  a, b  amplitude and phase of M2, K1, S1 against depth. The mode structure.
  c     composite profiles for four regimes. What it actually looks like.
  d     the seasonal cycle: diurnal baroclinic envelope vs M2 barotropic
        envelope vs stratification. One is flat all year; the other is not.
  e     median fbt binned on stratification x tidal-current strength. The
        "when", as a lookup table.

SIGN. Positive q at pc_lp runs minus-side -> plus-side = eastward = OUT of the
cove, so everything here is negated: positive means INTO Penn Cove. Consistent
with 20260806_pc_mouth_salinity_tides.py and the corr(qnet, d(ssh)/dt) = -1.00
check recorded there.

CAVEAT. u is the section-NORMAL component only; the along-section component
never enters the tef2 extraction, so these are lower bounds on the true speed.
Levels are sigma levels, so "depth" in panels a and b is the record-mean depth
of each level, not a fixed z.

Runs on the mac from the local extractions_avg.
run 20260811_pc_lp_baroclinic_barotropic.py
"""
import argparse
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utide
import xarray as xr
from cmcrameri import cm as cmc
from scipy.signal import butter, filtfilt, hilbert

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
p.add_argument('-gctag', default='wb1_pc1', type=str)
p.add_argument('-sect', default='pc_lp', type=str)
p.add_argument('-0', '--ds0', default='2024.01.01', type=str)
p.add_argument('-1', '--ds1', default='2025.12.31', type=str)
args = p.parse_args()

warnings.simplefilter('ignore')
Ldir = Lfun.Lstart(gridname=args.gctag.split('_')[0])
tef2 = Ldir['LOo'] / 'extract' / args.gtagex / 'tef2'
ex_fn = tef2 / ('extractions_avg_%s_%s' % (args.ds0, args.ds1)) / (args.sect + '.nc')
st_fn = tef2 / ('strat_%s_%s_%s.nc' % (args.ds0, args.ds1, args.gctag))
out_dir = Ldir['LOo'] / 'DM_outs' / '20260811_pc_lp_modes'
Lfun.make_dir(out_dir)

CB = dict(blue='#0072B2', red='#CC0000', green='#009E73', orange='#D55E00',
          purple='#CC79A7', grey='#7f7f7f')
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
FS = 13
LAT = 48.24


def godin(a):
    return zfun.lowpass(np.asarray(a, dtype=float), f='godin')


def bandpass(v, lo, hi, fs=24.):
    b, a = butter(3, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    return filtfilt(b, a, np.nan_to_num(v))


def envelope(v):
    return np.abs(hilbert(v))


def solve(t, y):
    s = utide.solve(t, np.nan_to_num(y), lat=LAT, method='ols', conf_int='none',
                    nodal=True, trend=False, verbose=False)
    return pd.Series(s.A, index=s.name), pd.Series(s.g, index=s.name)


# ---------------------------------------------------------------------------
# load and decompose
# ---------------------------------------------------------------------------
ds = xr.open_dataset(ex_fn)
tt = pd.to_datetime(ds.time.values)
dd, hh = ds.dd.values, ds.h.values
q = -ds.q.values                      # + = INTO the cove
A = ds.DZ.values * dd[None, None, :]  # face-cell area (time, z, p)
ds.close()
NZ, NP = A.shape[1], A.shape[2]

Az = A.sum(axis=2)                    # (time, z) area of each sigma layer
Atot = Az.sum(axis=1)
Uz = q.sum(axis=2) / Az               # width-mean u at each level
U0 = q.sum(axis=(1, 2)) / Atot        # barotropic
Urms = np.sqrt((Az * (Uz - U0[:, None]) ** 2).sum(axis=1) / Atot)

# the lateral mode, for scale: the residual at pc_lp is known to be lateral
Ap = A.sum(axis=1)
Lrms = np.sqrt((Ap * (q.sum(axis=1) / Ap - U0[:, None]) ** 2).sum(axis=1) / Atot)

# top and bottom third by level count, for a single scalar shear
ub = q[:, :10, :].sum(axis=(1, 2)) / A[:, :10, :].sum(axis=(1, 2))
ut = q[:, 20:, :].sum(axis=(1, 2)) / A[:, 20:, :].sum(axis=(1, 2))

dsrt = xr.open_dataset(st_fn)
dstrat = pd.Series(godin(dsrt.dstrat.sel(sect=args.sect).values),
                   index=pd.to_datetime(dsrt.time.values))
dsrt.close()

S = pd.DataFrame(dict(U0=U0, Urms=Urms, Lrms=Lrms, ut=ut, ub=ub), index=tt)
S['shear'] = S.ut - S.ub
S['fbt'] = S.U0 ** 2 / (S.U0 ** 2 + S.Urms ** 2)
S['dstrat'] = dstrat
S['cls'] = np.where(S.U0.abs() > 2 * S.Urms, 'barotropic',
                    np.where(S.Urms > 2 * S.U0.abs(), 'baroclinic', 'mixed'))
S['revs'] = np.sign(Uz).min(axis=1) != np.sign(Uz).max(axis=1)

# record-mean depth of each sigma level, measured down from the surface
lay = Az.mean(axis=0)
Hm = (Atot / dd.sum()).mean()
zmid = Hm * (1 - (np.cumsum(lay) - 0.5 * lay) / lay.sum())   # z=0 is the bed

print('section %s: %d faces, h %.1f to %.1f m, mean depth %.1f m, %d hours'
      % (args.sect, NP, hh.min(), hh.max(), Hm, len(S)))

# ---------------------------------------------------------------------------
# a. per-level harmonic structure
# ---------------------------------------------------------------------------
tp = tt.to_pydatetime()
rows = []
for k in range(NZ):
    a_, g_ = solve(tp, Uz[:, k])
    rows.append(dict(z=k, depth=zmid[k],
                     **{c: a_.get(c, np.nan) for c in ['M2', 'S2', 'K1', 'O1', 'P1', 'S1']},
                     **{'g' + c: g_.get(c, np.nan) for c in ['M2', 'K1', 'S1']}))
H = pd.DataFrame(rows)
H.to_csv(out_dir / ('level_constituents_%s.csv' % args.sect), index=False,
         float_format='%.5f')

a_sh, _ = solve(tp, S.shear.values)
a_bt, _ = solve(tp, S.U0.values)
print('\nharmonic amplitudes (m/s)          M2      S2      K1      O1      P1      S1')
print('  barotropic U0                 %s'
      % '  '.join('%.4f' % a_bt.get(c, np.nan) for c in ['M2', 'S2', 'K1', 'O1', 'P1', 'S1']))
print('  shear (top third - bottom)    %s'
      % '  '.join('%.4f' % a_sh.get(c, np.nan) for c in ['M2', 'S2', 'K1', 'O1', 'P1', 'S1']))
knode = int(np.nanargmin(H.S1.values[:20]))
print('  diurnal node at sigma level %d = %.1f m below the surface (%.0f%% of the '
      'column above the bed)' % (knode, zmid[knode], 100 * (1 - zmid[knode] / Hm)))

# ---------------------------------------------------------------------------
# b. composite profiles
# ---------------------------------------------------------------------------
strong = (S.U0.abs() > S.U0.abs().quantile(0.8)).values
REG = [('peak flood', (S.U0 > 0).values & strong, CB['red'], '-'),
       ('peak ebb', (S.U0 < 0).values & strong, CB['blue'], '-'),
       ('stratified\n(top 20% $\\Delta s$)', (S.dstrat > S.dstrat.quantile(0.8)).values,
        CB['purple'], '--'),
       ('well mixed\n(bottom 20% $\\Delta s$)', (S.dstrat < S.dstrat.quantile(0.2)).values,
        CB['orange'], '--')]
PROF = {lbl: np.nanmean(Uz[m], axis=0) for lbl, m, _, _ in REG}

# ---------------------------------------------------------------------------
# c. seasonal envelopes
# ---------------------------------------------------------------------------
E = pd.DataFrame(dict(
    bc_diurnal=godin(envelope(bandpass(S.shear.values, 0.7, 1.4))),
    bt_semi=godin(envelope(bandpass(S.U0.values, 1.6, 2.4))),
    dstrat=S.dstrat.values), index=tt).dropna()
MO = E.groupby(E.index.month).mean()

# ---------------------------------------------------------------------------
# d. the when-table
# ---------------------------------------------------------------------------
T = S.dropna(subset=['dstrat']).copy()
SQ = ['weakest', 'q2', 'q3', 'q4', 'strongest']
UQ = ['slack', 'q2', 'q3', 'q4', 'peak']
T['sq'] = pd.qcut(T.dstrat, 5, labels=SQ)
T['uq'] = pd.qcut(T.U0.abs(), 5, labels=UQ)
FBT = pd.crosstab(T.sq, T.uq, T.fbt, aggfunc='median').loc[SQ, UQ]
BT = pd.crosstab(T.sq, T.uq, T.cls.eq('barotropic'), aggfunc='mean').loc[SQ, UQ] * 100

print('\n--- how the hours split ---')
for k, v in (S.cls.value_counts(normalize=True) * 100).round(1).items():
    print('  %-11s %5.1f %%' % (k, v))
print('  profile reverses sign somewhere in the vertical: %.0f %% of hours'
      % (100 * S.revs.mean()))
print('  median fbt %.2f   |  corr(Urms, |U0|) %+.2f   corr(Urms, dstrat) daily %+.2f'
      % (S.fbt.median(), S.Urms.corr(S.U0.abs()),
         S.Urms.resample('D').mean().corr(S.dstrat.resample('D').mean())))
print('  vertical shear rms %.3f  vs lateral rms %.3f  vs |U0| %.3f  m/s'
      % (S.Urms.mean(), S.Lrms.mean(), S.U0.abs().mean()))

print('\nmedian barotropic energy fraction fbt, stratification x tidal current:')
print(FBT.round(2).to_string())
print('\n%% of hours barotropic-dominated:')
print(BT.round(0).to_string())

print('\nby month:')
MM = pd.DataFrame({
    'baroclinic_%': (pd.crosstab(S.index.month, S.cls, normalize='index') * 100)['baroclinic'],
    'barotropic_%': (pd.crosstab(S.index.month, S.cls, normalize='index') * 100)['barotropic'],
    'bc_diurnal': MO.bc_diurnal, 'bt_M2': MO.bt_semi, 'dstrat': MO.dstrat})
MM['bc/bt'] = MM.bc_diurnal / MM.bt_M2
print(MM.round(3).to_string())
MM.to_csv(out_dir / ('monthly_modes_%s.csv' % args.sect), float_format='%.4f')

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(12.5, 9))
gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 0.55, 1.0], hspace=0.32, wspace=0.55)

# a. amplitude vs depth, with phase alongside
ax = fig.add_subplot(gs[0, 0])
for c, col in [('M2', CB['blue']), ('K1', CB['red']), ('S1', CB['orange'])]:
    ax.plot(H[c], H.depth, color=col, lw=2.0, marker='o', ms=3, label=c)
ax.axhline(zmid[knode], color=CB['grey'], lw=1.0, ls=':')
ax.text(0.97, zmid[knode] / Hm + 0.015, 'diurnal node, %.1f m' % zmid[knode],
        transform=ax.get_yaxis_transform(), ha='right', va='bottom',
        fontsize=FS - 3, color=CB['grey'])
ax.invert_yaxis()
ax.set_xlabel('amplitude [m s$^{-1}$]', fontsize=FS)
ax.set_ylabel('depth below surface [m]', fontsize=FS)
ax.set_title('a  vertical structure of each constituent', fontsize=FS, loc='left')
ax.legend(frameon=False, fontsize=FS - 2, loc='lower right')

ax = fig.add_subplot(gs[0, 1])
for c, col in [('M2', CB['blue']), ('K1', CB['red']), ('S1', CB['orange'])]:
    ax.plot(H['g' + c], H.depth, color=col, lw=2.0, marker='o', ms=3)
ax.axhline(zmid[knode], color=CB['grey'], lw=1.0, ls=':')
ax.invert_yaxis()
ax.set_xlabel('phase [deg]', fontsize=FS)
ax.set_title('b  phase', fontsize=FS, loc='left')
ax.tick_params(labelleft=False)

# c. composite profiles
ax = fig.add_subplot(gs[0, 2])
for lbl, _, col, ls in REG:
    ax.plot(PROF[lbl], zmid, color=col, lw=2.0, ls=ls, label=lbl)
ax.axvline(0, color='k', lw=0.9)
ax.invert_yaxis()
ax.set_xlabel('width-mean $u$ [m s$^{-1}$], + into the cove', fontsize=FS)
ax.set_ylabel('depth below surface [m]', fontsize=FS)
ax.set_title('c  mean profile by regime', fontsize=FS, loc='left')
ax.legend(frameon=False, fontsize=FS - 4, loc='lower right')

# d. seasonal cycle
ax = fig.add_subplot(gs[1, :2])
mo = MO.index.values
ax.plot(mo, MO.bc_diurnal, color=CB['red'], lw=2.2, marker='o',
        label='baroclinic: diurnal shear envelope')
ax.plot(mo, MO.bt_semi, color=CB['blue'], lw=2.2, marker='s',
        label='barotropic: M2 $U_0$ envelope')
ax.set_ylabel('velocity [m s$^{-1}$]', fontsize=FS)
ax.set_ylim(0, None)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
ax.set_title('d  the barotropic tide is flat all year; the baroclinic mode is not',
             fontsize=FS, loc='left')
axb = ax.twinx()
axb.plot(mo, MO.dstrat, color=CB['grey'], lw=1.6, ls='--',
         label='stratification $\\Delta s$ (right axis)')
axb.set_ylabel('$\\Delta s$ top - bottom [g kg$^{-1}$]', fontsize=FS - 1,
               color=CB['grey'])
hl = ax.get_legend_handles_labels()
hr = axb.get_legend_handles_labels()
ax.legend(hl[0] + hr[0], hl[1] + hr[1], frameon=False, fontsize=FS - 4,
          loc='lower center', ncol=3)
axb.tick_params(colors=CB['grey'], labelsize=FS - 3)
axb.set_ylim(0, None)

# e. the when-table
ax = fig.add_subplot(gs[1, 2])
im = ax.imshow(FBT.values, cmap=cmc.vik, vmin=0, vmax=1, origin='lower',
               aspect='auto')
for i in range(5):
    for j in range(5):
        v = FBT.values[i, j]
        ax.text(j, i, '%.2f' % v, ha='center', va='center', fontsize=FS - 3,
                color='w' if abs(v - 0.5) > 0.28 else 'k')
ax.set_xticks(range(5)); ax.set_xticklabels(UQ, fontsize=FS - 3)
ax.set_yticks(range(5)); ax.set_yticklabels(SQ, fontsize=FS - 3)
ax.set_xlabel('tidal current $|U_0|$ quintile', fontsize=FS - 1)

ax.set_title('e  median $f_{bt}$: rows $\\Delta s$, columns $|U_0|$',
             fontsize=FS, loc='left')
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.set_label('0 = all baroclinic,  1 = all barotropic', fontsize=FS - 4)
cb.ax.tick_params(labelsize=FS - 4)

for a_ in fig.axes:
    a_.grid(**GRID)
    a_.tick_params(labelsize=FS - 2)

fn_out = out_dir / ('pc_lp_modes_%s_%s.png' % (args.gtagex, args.sect))
fig.savefig(fn_out, dpi=400, bbox_inches='tight', transparent=True)
print('\nwrote ' + str(fn_out))
