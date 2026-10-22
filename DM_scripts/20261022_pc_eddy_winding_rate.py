"""
Selection-robust versions of the winding test (W_fix about the fixed inner-cove
eddy centre, from 20261021_pc_eddy_winding.py). The >= 24 h eligibility cut
keeps fewer particles when the eddy is present (the cove flushes faster), so:

RATE  per release, turns per day experienced near the centre:
        rate = sum(W_fix) / sum(h_fix / 24)  over particles with h_fix > -hmin
      (h_fix = hours counted 200-1200 m from the centre before first exit;
      W only accumulates in those hours, so the ratio is consistent). A particle
      that leaves after 30 h contributes 30 h, not nothing.
      Compared with the Eulerian core rate: solid-body rotation at the mean core
      vorticity turns 24 / T_turn_h times a day (T_turn_h = 4 pi / zeta_core from
      20261021_pc_eddy_center.py, averaged over the release's first 3 d). The
      core is the strongest part, and the 200-1200 m annulus reaches beyond it,
      so the particle rate should be below it, but track it.
ALL   per release, mean W over ALL cove-start particles (W = 0 if never near):
      no selection at all, but mixes rotation with flushing (robustness only).
CELL-BY-CELL  releases compared within the same starting rho cell, so a
      different mix of starting places cannot make the difference:
        - pooled rate per start cell in eddy-present releases (pres3d = 1) and in
          eddy-absent releases (pres3d = 0), and their difference (map)
        - within-cell (cell fixed effects) weighted regression of the
          release x cell rate on pres3d, weights = exposure days; the slope is
          the rate change from eddy absent to present
        - W histograms of eligible particles with the absent group reweighted to
          the present group's start-cell mix (and vice versa)
Uncertainty: block bootstrap over releases (-block_days blocks of release
time, -nboot draws), because releases ~1.5 d apart share eddy states.

Output: LO_output/DM_outs/20261022_pc_eddy_winding_rate/
  rate_<band>.png        (a) release rate vs pres3d, (b) particle rate vs
                         Eulerian core rate, (c) all-particle mean W vs pres3d
  cells_<band>.png       start-cell maps: pooled rate present / absent / difference
  reweighted_<band>.png  W histograms, raw and start-cell reweighted
  rate_releases_<band>.csv, summary_<band>.txt

  python 20261022_pc_eddy_winding_rate.py
  python 20261022_pc_eddy_winding_rate.py -band z4-10
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-band', default='lower')
p.add_argument('-region', default='pc_cp')
p.add_argument('-hmin', type=float, default=0, help='rate: min counted hours per particle (0 = any)')
p.add_argument('-hmin_hist', type=float, default=24, help='reweighted histograms: eligibility, as before')
p.add_argument('-min_days', type=float, default=2.0, help='cell maps: min pooled exposure days per group')
p.add_argument('-block_days', type=float, default=14)
p.add_argument('-nboot', type=int, default=1000)
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
win = Ldir['LOo'] / 'DM_outs' / '20261021_pc_eddy_winding' / args.region / args.band
cen_fn = Ldir['LOo'] / 'DM_outs' / '20261021_pc_eddy_center' / args.region / ('pc_eddy_center_%s.csv' % args.band)
out_dir = Ldir['LOo'] / 'DM_outs' / '20261022_pc_eddy_winding_rate'
Lfun.make_dir(out_dir)
rng = np.random.default_rng(0)
BLUE, GREY, ORANGE = '#2a6fdb', '#8a8a8a', '#e8590c'
SCOL = {'Nov-Dec': BLUE, 'Aug-Oct': ORANGE, 'other': GREY}

A = pd.read_pickle(win / ('winding_particles_%s.p' % args.gtx))
keep = set(pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_release_times' /
                       'pcmap_release_times_2025_every3.csv').out_name)
A = A[A.release.str.split('__').str[0].isin(keep)].copy()
A['t0'] = pd.to_datetime(A.t0)
A['days'] = A.h_fix / 24
mon = A.t0.dt.month
A['season'] = np.where(mon.isin([11, 12]), 'Nov-Dec', np.where(mon.isin([8, 9, 10]), 'Aug-Oct', 'other'))
A['grp'] = np.where(A.pres3d >= 0.999, 'present', np.where(A.pres3d <= 0.001, 'absent', 'partial'))

CEN = pd.read_csv(cen_fn, parse_dates=['time']).set_index('time')

# ------------------------------------------------------- release level ---
X = A[A.h_fix > args.hmin]
R = X.groupby('release').agg(t0=('t0', 'first'), season=('season', 'first'), grp=('grp', 'first'),
                             pres3d=('pres3d', 'first'), sumW=('W_fix', 'sum'), days=('days', 'sum'),
                             n=('W_fix', 'size'))
R['rate'] = R.sumW / R.days
R['meanW_all'] = A.groupby('release').W_fix.mean()
R['n_all'] = A.groupby('release').size()
R['euler_rate'] = [24 / CEN.T_turn_h[(CEN.index >= t) & (CEN.index < t + pd.Timedelta('3D'))].mean()
                   for t in R.t0]
R = R.reset_index().sort_values('t0')
R.to_csv(out_dir / ('rate_releases_%s.csv' % args.band), index=False, float_format='%.4g')
R['block'] = ((R.t0 - R.t0.min()).dt.total_seconds() // (args.block_days * 86400)).astype(int)
BLK = {b: g.release.values for b, g in R.groupby('block')}


def boot(stat):
    """Block bootstrap over releases: stat(list of release names) -> value; returns (est, lo, hi)."""
    est = stat(R.release.values)
    keys = list(BLK)
    v = []
    for _ in range(args.nboot):
        pick = rng.choice(keys, len(keys), replace=True)
        v.append(stat(np.concatenate([BLK[k] for k in pick])))
    v = np.array(v, float)
    return est, np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)


Ri = R.set_index('release')


def grp_rate(g):
    def f(rels):
        r = Ri.loc[rels]
        r = r[r.grp == g]
        return r.sumW.sum() / r.days.sum()
    return f


txt = ['WINDING RATE ABOUT THE INNER-COVE EDDY CENTRE -- %s band, %s, %d releases' % (args.band, args.region, len(R)),
       'rate = sum W / sum days counted 200-1200 m from the centre before first exit (particles with h_fix > %g h)'
       % args.hmin, 'CI = 95%% block bootstrap over releases (%g-d blocks, %d draws)' % (args.block_days, args.nboot), '']
for g in ['absent', 'partial', 'present']:
    e, lo, hi = boot(grp_rate(g))
    txt.append('  pooled rate, eddy %-8s %+.3f turns/day [%+.3f, %+.3f]  (%d releases)'
               % (g, e, lo, hi, (R.grp == g).sum()))
e, lo, hi = boot(lambda rels: grp_rate('present')(rels) - grp_rate('absent')(rels))
txt.append('  present - absent              %+.3f turns/day [%+.3f, %+.3f]' % (e, lo, hi))
for col, nm in [('rate', 'release rate'), ('meanW_all', 'all-particle mean W'), ('euler_rate', 'Eulerian core rate')]:
    e, lo, hi = boot(lambda rels, col=col: np.corrcoef(Ri.loc[rels, 'pres3d'], Ri.loc[rels, col])[0, 1])
    txt.append('  corr(pres3d, %-20s) %+.2f [%+.2f, %+.2f]' % (nm, e, lo, hi))
e, lo, hi = boot(lambda rels: np.corrcoef(Ri.loc[rels, 'euler_rate'], Ri.loc[rels, 'rate'])[0, 1])
txt.append('  corr(Eulerian core rate, particle rate) %+.2f [%+.2f, %+.2f]; median ratio particle/Eulerian %.2f'
           % (e, lo, hi, (R.rate / R.euler_rate).median()))

# -------------------------------------------------------- cell by cell ---
C = X.groupby(['release', 'j0', 'i0']).agg(sumW=('W_fix', 'sum'), days=('days', 'sum')).reset_index()
C = C.merge(R[['release', 'pres3d', 'grp']], on='release')
C['rate'] = C.sumW / C.days
Cr = {r: g for r, g in C.groupby('release')}


def fe_slope(rels):
    """Within-cell weighted regression of release x cell rate on pres3d (cell fixed effects)."""
    d = pd.concat([Cr[r] for r in rels if r in Cr], ignore_index=True)
    w = d.days.values
    key = d.j0.values * 1000 + d.i0.values
    df = pd.DataFrame(dict(k=key, w=w, x=d.pres3d.values, y=d.rate.values))
    sw = df.groupby('k').w.transform('sum')
    xm = (df.w * df.x).groupby(df.k).transform('sum') / sw
    ym = (df.w * df.y).groupby(df.k).transform('sum') / sw
    xd, yd = df.x - xm, df.y - ym
    return (df.w * xd * yd).sum() / (df.w * xd * xd).sum()


def cell_diff(rels):
    """Exposure-weighted mean over cells of pooled rate(present) - pooled rate(absent)."""
    d = pd.concat([Cr[r] for r in rels if r in Cr], ignore_index=True)
    d = d[d.grp != 'partial']
    P = d.groupby(['j0', 'i0', 'grp'])[['sumW', 'days']].sum().unstack('grp')
    P = P.dropna()
    P = P[(P['days']['present'] >= args.min_days) & (P['days']['absent'] >= args.min_days)]
    rp = P['sumW']['present'] / P['days']['present']
    ra = P['sumW']['absent'] / P['days']['absent']
    w = 2 / (1 / P['days']['present'] + 1 / P['days']['absent'])
    return ((rp - ra) * w).sum() / w.sum()


e, lo, hi = boot(fe_slope)
txt.append('  within-cell (cell fixed effects) slope of rate on pres3d: %+.3f turns/day per unit presence [%+.3f, %+.3f]'
           % (e, lo, hi))
e, lo, hi = boot(cell_diff)
txt.append('  within-cell present - absent (exposure-weighted over cells with >= %g d in each): %+.3f [%+.3f, %+.3f]'
           % (args.min_days, e, lo, hi))

# pooled cell maps
d = C[C.grp != 'partial'].groupby(['j0', 'i0', 'grp'])[['sumW', 'days']].sum().unstack('grp')
MAP = {}
for g in ['present', 'absent']:
    ok = d['days'][g] >= args.min_days
    MAP[g] = (d['sumW'][g] / d['days'][g]).where(ok)
MAP['diff'] = MAP['present'] - MAP['absent']
txt.append('  cells with >= %g d in both groups: %d; of those, present > absent in %d (%.0f%%)'
           % (args.min_days, MAP['diff'].notna().sum(), (MAP['diff'] > 0).sum(),
              100 * (MAP['diff'] > 0).sum() / max(MAP['diff'].notna().sum(), 1)))

# reweighted histograms (eligible particles)
E = A[(A.h_fix >= args.hmin_hist) & (A.grp != 'partial')].copy()
E['cell'] = E.j0 * 1000 + E.i0
comp = E.groupby(['grp', 'cell']).size().unstack('grp').fillna(0)
comp = comp[(comp.present > 0) & (comp.absent > 0)]
share = comp / comp.sum()
E = E[E.cell.isin(comp.index)]


def wmedian(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return x[o][np.searchsorted(c, 0.5 * c[-1])]


HW = {}
for g, other in [('present', 'absent'), ('absent', 'present')]:
    e_ = E[E.grp == g]
    w_raw = np.ones(len(e_))
    w_to = (share[other] / share[g]).reindex(e_.cell).values          # reweight g to other's start-cell mix
    HW[g] = (e_.W_fix.values, w_raw, w_to)
    txt.append('  eligible %-7s median W raw %+.2f; reweighted to the %s start-cell mix %+.2f'
               % (g, np.median(e_.W_fix), other, wmedian(e_.W_fix.values, w_to)))
txt.append('  (start cells used in both groups: %d)' % len(comp))
(out_dir / ('summary_%s.txt' % args.band)).write_text('\n'.join(txt) + '\n')
print('\n'.join(txt))

# ------------------------------------------------------------- figures ---
fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), layout='constrained')
for ax, col, yl in [(axs[0], 'rate', 'particle winding rate [turns/day]'),
                    (axs[2], 'meanW_all', 'mean W over ALL cove-start particles [turns]')]:
    for s_, r in R.groupby('season'):
        ax.scatter(r.pres3d, r[col], s=16, color=SCOL[s_], label=s_, alpha=0.85, ec='none')
    b = pd.cut(R.pres3d, [-0.01, 0.001, 0.25, 0.5, 0.75, 0.999, 1.0])
    m = R.groupby(b, observed=True)[col].median()
    xs = [0, 0.125, 0.375, 0.625, 0.875, 1][:len(m)] if len(m) == 6 else [iv.mid for iv in m.index]
    ax.plot(xs, m.values, 'k-o', ms=5, lw=1.5, label='bin median')
    ax.axhline(0, color='0.3', lw=0.8, ls=':')
    ax.set_xlabel('share of the first 3 d with the eddy present (pres3d)')
    ax.set_ylabel(yl)
    ax.set_title('r = %.2f' % R.pres3d.corr(R[col]), fontsize=10)
    ax.grid(color='0.9', lw=0.6)
axs[0].legend(fontsize=8, frameon=False)
ax = axs[1]
for s_, r in R.groupby('season'):
    ax.scatter(r.euler_rate, r.rate, s=16, color=SCOL[s_], alpha=0.85, ec='none')
lim = [0, max(R.euler_rate.max(), R.rate.max()) * 1.05]
ax.plot(lim, lim, color='0.3', lw=0.8, ls='--', label='1:1')
ax.set_xlim(lim); ax.set_ylim(min(0, R.rate.min()) - 0.02, lim[1])
ax.set_xlabel('Eulerian core rate 24 / T_turn [turns/day], first 3 d')
ax.set_ylabel('particle winding rate [turns/day]')
ax.set_title('r = %.2f, median particle / core = %.2f' % (R.euler_rate.corr(R.rate), (R.rate / R.euler_rate).median()),
             fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(color='0.9', lw=0.6)
axs[0].set_title('(a) release winding rate vs eddy presence\n' + axs[0].get_title(), fontsize=10, loc='left')
axs[1].set_title('(b) particles vs Eulerian core\n' + axs[1].get_title(), fontsize=10, loc='left')
axs[2].set_title('(c) no eligibility cut (robustness)\n' + axs[2].get_title(), fontsize=10, loc='left')
for ax in axs:
    ax.set_title('', loc='center')
fig.suptitle('Winding rate about the inner-cove eddy centre (%s band; one dot per release; particles with any time '
             '200-1200 m from the centre before first exit)' % args.band, fontsize=10)
fn = out_dir / ('rate_%s.png' % args.band)
fig.savefig(fn, dpi=200, transparent=True, bbox_inches='tight')
plt.close(fig)
print('saved ' + str(fn))

g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
glon, glat = g.lon_rho.values, g.lat_rho.values
g.close()
xc, yc = CEN.x_c[CEN.present.astype(bool)].mean(), CEN.y_c[CEN.present.astype(bool)].mean()
LON_MOUTH = pd.read_pickle(Ldir['LOo'] / 'section_lines' / 'pc_lp.p').x.mean()
lonc, latc = LON_MOUTH + xc / (111.32e3 * np.cos(np.deg2rad(48.23))), 48.23 + yc / 111.32e3
vm = np.nanpercentile(np.abs(pd.concat([MAP['present'], MAP['absent']])), 98)
fig, axs = plt.subplots(1, 3, figsize=(17, 4.6), layout='constrained')
for ax, k, ttl in [(axs[0], 'present', 'eddy present (pres3d = 1)'), (axs[1], 'absent', 'eddy absent (pres3d = 0)'),
                   (axs[2], 'diff', 'present - absent')]:
    F = np.full(glon.shape, np.nan)
    s = MAP[k].dropna()
    F[s.index.get_level_values(0), s.index.get_level_values(1)] = s.values
    v = vm if k != 'diff' else np.nanpercentile(np.abs(s.values), 98)
    h = ax.pcolormesh(glon, glat, F, cmap='RdBu_r', vmin=-v, vmax=v, shading='nearest')
    pfun.add_coast(ax, color='gray', linewidth=0.6)
    ax.plot(lonc, latc, 'k+', ms=12, mew=2)
    ax.axis([-122.737, -122.652, 48.212, 48.246])
    pfun.dar(ax)
    ax.tick_params(labelsize=8)
    ax.set_title('%s\npooled winding rate by START cell [turns/day]' % ttl, fontsize=10)
    fig.colorbar(h, ax=ax, shrink=0.8)
fig.suptitle('Cell-by-cell comparison (%s band; cells with >= %g exposure days in the group; + = eddy centre)'
             % (args.band, args.min_days), fontsize=10)
fn = out_dir / ('cells_%s.png' % args.band)
fig.savefig(fn, dpi=200, transparent=True, bbox_inches='tight')
plt.close(fig)
print('saved ' + str(fn))

bins = np.arange(-2, 3.05, 0.1)
fig, axs = plt.subplots(1, 2, figsize=(13, 4.4), layout='constrained', sharey=True)
for ax, (g_, c_) in zip(axs, [('present', BLUE), ('absent', ORANGE)]):
    w, w_raw, w_to = HW[g_]
    o_, oc = ('absent', ORANGE) if g_ == 'present' else ('present', BLUE)
    wo = HW[o_][0]
    wc = lambda a: np.clip(a, -2 + 1e-9, 3 - 1e-9)
    ax.hist(wc(w), bins=bins, density=True, histtype='step', color=c_, lw=2,
            label='eddy %s, raw (median %+.2f)' % (g_, np.median(w)))
    ax.hist(wc(w), bins=bins, weights=w_to, density=True, histtype='step', color=c_, lw=2, ls='--',
            label='eddy %s, reweighted to %s start cells (median %+.2f)' % (g_, o_, wmedian(w, w_to)))
    ax.hist(wc(wo), bins=bins, density=True, histtype='step', color=oc, lw=1.2, alpha=0.7,
            label='eddy %s, raw (median %+.2f)' % (o_, np.median(wo)))
    for x in [0, 1]:
        ax.axvline(x, color='0.3', lw=0.8, ls=':')
    ax.set_xlim(-2, 3)
    ax.set_xlabel('winding number W about the eddy centre [turns]')
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    ax.grid(color='0.9', lw=0.6)
axs[0].set_ylabel('probability density')
fig.suptitle('W of eligible particles (>= %g h), raw vs reweighted to the other group\'s start-cell mix (%s band)'
             % (args.hmin_hist, args.band), fontsize=10)
fn = out_dir / ('reweighted_%s.png' % args.band)
fig.savefig(fn, dpi=200, transparent=True, bbox_inches='tight')
plt.close(fig)
print('saved ' + str(fn))
