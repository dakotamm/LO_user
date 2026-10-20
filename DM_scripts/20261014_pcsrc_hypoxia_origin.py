"""
Where does Penn Cove's hypoxic water come from: made inside the cove, or
imported already hypoxic? A Lagrangian classification of the hypoxic water in
a target, from the pcsrc reduced files (20261013_pcsrc_reduce.py: upper-Saratoga
cohort + the pcmap cove cohort) and DO along every path
(20261013_pcsrc_do_path.py). Mac only.

Every hour, each particle inside the target with DO < -thresh (2 mg/L) is put
in one class from its own history up to that hour:

  never_left      cove cohort, inside the cove at every hour since release
  hypoxic_inside  its MOST RECENT entry into the cove was at DO >= thresh, so it
                  went hypoxic after getting in (either cohort)
  reentered       cove cohort that left the cove and came back already hypoxic
                  (DO < thresh at entry): the cove's own water, recycled
  imported        upper-Saratoga cohort that entered already hypoxic

never_left, hypoxic_inside and reentered all mean the hypoxia was made in the
cove (reentered: made earlier, exported, brought back); imported is the only
class that means hypoxic water arrived from upper Saratoga. Caveat: DO is the
model field sampled along the path, so a particle's DO also reflects mixing with
its surroundings; an "imported" particle went hypoxic in water outside the cove
that may itself contain exported cove water.

Entry = first hourly frame inside the cove after a frame outside it; DO at entry
is DO at that frame (up to an hour after the crossing). "Inside the cove" =
pcsrc_regions 'cove' cells (landward of pc_lp), for both targets.

TARGETS  cove       landward of pc_lp
         inner_bot  pc_cp_m, <= 5 m off the bed and lower half (pcbot definition)
Volumes: particle volumes at the release tide (pcsrc_regions.release_tide_vol);
target volume from the tef2 segments (inner_bot layer from the pc_cp_m mean
surface), as in 20261013_pcsrc_source.py.

Reported per hour (tidal-day 25 h running mean) and averaged over releases:
  hyp_<class>   class volume / target volume
  hyp_lab       labelled hypoxic volume / target volume (sum of classes)
  lab           all labelled volume / target volume: the coverage. Water that is
                neither cohort (unlabelled, grows with time) is not classified.
  share_<class> class volume / labelled hypoxic volume

run 20261014_pcsrc_hypoxia_origin.py -month 8
"""
import argparse
import pickle

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from lo_tools import Lfun

import pcsrc_regions as pr

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-month', type=int, default=0)
p.add_argument('-set', default='both', choices=['both', 'E', 'F'])
p.add_argument('-days', type=float, default=14.0)
p.add_argument('-thresh', type=float, default=2.0, help='hypoxia threshold [mg/L]')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
red_dir = Ldir['LOo'] / 'DM_outs' / '20261013_pcsrc_reduce' / args.gtx
do_dir = Ldir['LOo'] / 'DM_outs' / '20261013_pcsrc_do_path' / args.gtx
out_dir = Ldir['LOo'] / 'DM_outs' / '20261014_pcsrc_hypoxia_origin' / args.gtx
Lfun.make_dir(out_dir)

g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat, h = g.lon_rho.values, g.lat_rho.values, g.h.values
A = 1 / (g.pm.values * g.pn.values)
wet = g.mask_rho.values == 1
g.close()
R = pr.regions(Ldir, lon, lat, wet)
seg = pickle.load(open(sorted((Ldir['LOo'] / 'extract' / 'tef2').glob('seg_info_dict_wb1_pc1_*.p'))[0], 'rb'))
inner = np.zeros(lon.shape, dtype=bool)
a = np.array(seg['pc_cp_m']['ji_list']); inner[a[:, 0], a[:, 1]] = True

segds = xr.open_dataset(Ldir['LOo'] / 'extract' / args.gtx / 'tef2'
                        / 'segments_2024.01.01_2025.12.31_wb1_pc1_trapsN00.nc')
Vseg = segds.volume.to_pandas()
segds.close()
V_cove = Vseg[pr.COVE_SEGS].sum(axis=1)
h_in, A_in = h[inner], A[inner]
zeta_in = (Vseg['pc_cp_m'] - (h_in * A_in).sum()) / A_in.sum()
V_ibot = pd.Series([(A_in * np.minimum(5, 0.5 * (h_in + zt))).sum() for zt in zeta_in.values],
                   index=Vseg.index)

TARGETS = ['cove', 'inner_bot']
TLAB = {'cove': 'Penn Cove', 'inner_bot': 'inner-cove bottom water'}
CLASSES = ['never_left', 'hypoxic_inside', 'reentered', 'imported']
CLAB = {'never_left': 'cove water, never left',
        'hypoxic_inside': 'entered >= %g, went hypoxic inside' % args.thresh,
        'reentered': 'cove water re-entering already hypoxic',
        'imported': 'upper-SP water entering already hypoxic'}
CCOL = {'never_left': '#52514e', 'hypoxic_inside': '#2a78d6',
        'reentered': '#1baf7a', 'imported': '#eb6834'}

curves = {tg: {} for tg in TARGETS}
rows = []
for fn in sorted(red_dir.glob('*.p')):
    D = pickle.load(open(fn, 'rb'))
    meta, P, H = D['meta'], D['P'], D['H']
    st = meta['sub_tag']
    t0 = pd.Timestamp(meta['t0'])
    if args.month and t0.month != args.month:
        continue
    if args.set != 'both' and meta.get('set') != args.set:
        continue
    do_fn = do_dir / ('%s_DO.p' % st)
    if not do_fn.is_file():
        print('no DO for %s, skipped' % st)
        continue
    nf = min(meta['nf'], int(round(args.days * 24)) + 1)
    DO = pickle.load(open(do_fn, 'rb'))['DO'][:nf].astype(float)
    J = H['J'][:nf].astype(int); I = H['I'][:nf].astype(int)
    ok = J >= 0
    Jc, Ic = np.where(ok, J, 0), np.where(ok, I, 0)
    cs = H['CS'][:nf] / 10000
    hab = H['Z'][:nf].astype(float) + h[Jc, Ic]
    inside = ok & R['cove'][Jc, Ic]
    in_t = dict(cove=inside, inner_bot=ok & inner[Jc, Ic] & (hab <= 5) & (cs <= -0.5))
    t = pd.DatetimeIndex(meta['ot'][:nf])
    w = pr.release_tide_vol(P, t0, Vseg, h, A, R)
    is_cove0 = (P.cohort == 'cove').values

    # history: entries, DO at the most recent entry, never-left
    entry = np.zeros_like(inside)
    entry[1:] = inside[1:] & ~inside[:-1]
    do_entry = pd.DataFrame(np.where(entry, DO, np.nan)).ffill().values   # NaN until a first entry
    never_left = np.logical_and.accumulate(inside, axis=0) & is_cove0[None, :]
    hypoxic = DO < args.thresh
    cls = {'never_left': never_left,
           'hypoxic_inside': ~never_left & (do_entry >= args.thresh),
           'reentered': ~never_left & is_cove0[None, :] & (do_entry < args.thresh),
           'imported': ~never_left & ~is_cove0[None, :] & (do_entry < args.thresh)}

    for tg in TARGETS:
        V = (V_cove if tg == 'cove' else V_ibot).reindex(t, method='nearest').values
        m = in_t[tg] & hypoxic
        df = pd.DataFrame({'hyp_' + c: ((m & cls[c]) * w).sum(axis=1) / V for c in CLASSES}, index=t)
        df['lab'] = (in_t[tg] * w).sum(axis=1) / V
        # hypoxic but unclassified (inside the target with no entry record yet
        # and not never_left) should not happen; keep it visible if it does
        df['hyp_other'] = ((m & ~np.any([cls[c] for c in CLASSES], axis=0)) * w).sum(axis=1) / V
        df = df.rolling(25, center=True, min_periods=25).mean()
        df['hyp_lab'] = df[['hyp_' + c for c in CLASSES]].sum(axis=1)
        for c in CLASSES:
            df['share_' + c] = df['hyp_' + c] / df['hyp_lab'].where(df['hyp_lab'] > 0)
        df['days'] = np.arange(nf) / 24
        curves[tg][st] = df
        r = dict(sub_tag=st, set=meta.get('set'), t0=t0, target=tg)
        for d in [1, 3, 7, 10, 13]:
            k = int(round(d * 24))
            if k < nf:
                for c in ['hyp_lab', 'lab', 'hyp_other'] + ['share_' + c for c in CLASSES]:
                    r['%s_d%d' % (c, d)] = df[c].values[k]
        rows.append(r)

res = pd.DataFrame(rows)
if len(res) == 0:
    raise SystemExit('no releases with both reduced and DO files')
sfx = ('_m%02d' % args.month if args.month else '') + ('_%s' % args.set if args.set != 'both' else '')
res.to_csv(out_dir / ('pcsrc_hypoxia_origin%s.csv' % sfx), index=False)
pd.set_option('display.width', 220)
print('%d releases' % res.sub_tag.nunique())
for tg in TARGETS:
    s = res[res.target == tg]
    print('\n%s: share of labelled hypoxic volume (mean over releases)' % TLAB[tg])
    print('  day  hypoxic/target  coverage  ' + '  '.join('%14s' % c for c in CLASSES))
    for d in [1, 3, 7, 10, 13]:
        if 'hyp_lab_d%d' % d in s:
            print('  %3d  %14.3f  %8.2f  ' % (d, s['hyp_lab_d%d' % d].mean(), s['lab_d%d' % d].mean())
                  + '  '.join('%14.2f' % s['share_%s_d%d' % (c, d)].mean() for c in CLASSES))
    if s.filter(like='hyp_other').abs().max().max() > 1e-6:
        print('  NOTE: unclassified hypoxic volume present (hyp_other)')

# ---------------------------------------------------------------- figure
plt.close('all')
fs = 11
plt.rc('font', size=fs)
fig, axs = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True, layout='constrained')
for r, tg in enumerate(TARGETS):
    cv = curves[tg]
    days = next(iter(cv.values())).days.values
    mean = {c: np.nanmean(np.vstack([cv[k]['hyp_' + c].values for k in cv]), axis=0) for c in CLASSES}
    lab = np.nanmean(np.vstack([cv[k]['lab'].values for k in cv]), axis=0)
    ax = axs[r, 0]
    ax.stackplot(days, [mean[c] for c in CLASSES], colors=[CCOL[c] for c in CLASSES], lw=0)
    ax.set_xlim(0, args.days); ax.set_ylim(0, None)
    ax.set_ylabel('hypoxic volume / target volume')
    ax.set_title('%s: DO < %g mg/L (mean of %d)' % (TLAB[tg], args.thresh, len(cv)),
                 loc='left', fontsize=fs)
    ax2 = axs[r, 1]
    tot = np.sum([mean[c] for c in CLASSES], axis=0)
    share = [np.where(tot > 0, mean[c] / np.where(tot > 0, tot, 1), np.nan) for c in CLASSES]
    ax2.stackplot(days, share, colors=[CCOL[c] for c in CLASSES], lw=0)
    ax2.plot(days, lab, color='#0b0b0b', lw=1.5, ls='--')
    ax2.set_xlim(0, args.days); ax2.set_ylim(0, 1)
    ax2.set_ylabel('share of labelled hypoxic volume')
    ax2.set_title('as shares; dashed = coverage (labelled share)', loc='left', fontsize=fs)
    for a_ in (ax, ax2):
        a_.grid(lw=0.3, alpha=0.5)
for a_ in axs[1]:
    a_.set_xlabel('days since release')
fig.legend(handles=[Patch(fc=CCOL[c], label=CLAB[c]) for c in CLASSES], loc='outside lower center',
           ncol=4, fontsize=fs - 1, frameon=False)
fn_out = out_dir / ('pcsrc_hypoxia_origin%s.png' % sfx)
fig.savefig(fn_out, dpi=200, transparent=True)
plt.close(fig)
print('\nsaved %s' % fn_out)
