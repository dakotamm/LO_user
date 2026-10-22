"""
Winding rate about the inner-cove eddy centre vs eddy presence AT OTHER DEPTHS
(mac; no tracks needed). Two questions:

(1) WHICH DEPTH'S EDDY predicts particle rotation? The particles' winding
    (W_fix, h_fix from 20261021_pc_eddy_winding.py, -wband, i.e. about that
    band's fixed centre) is held fixed; each release's eddy presence over its
    first 3 d (pres3d) is recomputed from every band's centre file
    (20261021_pc_eddy_center.py: surf, mid, bot, lower, dm, z0-4, z4-10,
    z10-bed). Per band: pooled rate (sum W / sum days near the centre) in
    releases with the eddy absent (pres3d = 0) and present (= 1), the
    difference, and corr(pres3d, release rate). A band that is rarely present
    or never absent gives few releases in one group (n shown).
(2) DO PARTICLES AT DIFFERENT DEPTHS ROTATE DIFFERENTLY? Only by STARTING depth
    here (surface vs bottom half, cs0 >= -0.5), with presence from -wband.
    Particles move vertically (~40% switch halves within 3 d), so this is a
    proxy; depth-resolved winding needs the hourly tracks (apogee).

CI = 95% block bootstrap over releases (-block_days blocks).
Output: LO_output/DM_outs/20261022_pc_eddy_rate_by_depth/
  rate_by_depth_<wband>.png, rate_by_depth_<wband>.csv, summary_<wband>.txt

  python 20261022_pc_eddy_rate_by_depth.py
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-wband', default='lower', help='band whose winding run (centre) is used')
p.add_argument('-region', default='pc_cp')
p.add_argument('-bands', default='z0-4,z4-10,z10-bed,surf,mid,bot,lower,dm')
p.add_argument('-block_days', type=float, default=14)
p.add_argument('-nboot', type=int, default=500)
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
win = Ldir['LOo'] / 'DM_outs' / '20261021_pc_eddy_winding' / args.region / args.wband
cdir = Ldir['LOo'] / 'DM_outs' / '20261021_pc_eddy_center' / args.region
out_dir = Ldir['LOo'] / 'DM_outs' / '20261022_pc_eddy_rate_by_depth'
Lfun.make_dir(out_dir)
rng = np.random.default_rng(0)

A = pd.read_pickle(win / ('winding_particles_%s.p' % args.gtx))
keep = set(pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_release_times' /
                       'pcmap_release_times_2025_every3.csv').out_name)
A = A[A.release.str.split('__').str[0].isin(keep) & (A.h_fix > 0)].copy()
A['t0'] = pd.to_datetime(A.t0)
A['days'] = A.h_fix / 24
BANDS = args.bands.split(',')

# per release (and per release x start half): sum W, sum days
R = A.groupby('release').agg(t0=('t0', 'first'), sumW=('W_fix', 'sum'), days=('days', 'sum')).reset_index()
H = A.groupby(['release', 'surf0']).agg(sumW=('W_fix', 'sum'), days=('days', 'sum')).reset_index()
for b in BANDS:
    C = pd.read_csv(cdir / ('pc_eddy_center_%s.csv' % b), usecols=['time', 'present'], parse_dates=['time'])
    C = C.set_index('time').present.astype(float)
    R['p_' + b] = [C[(C.index >= t) & (C.index < t + pd.Timedelta('3D'))].mean() for t in R.t0]
R['rate'] = R.sumW / R.days
R = R.sort_values('t0').reset_index(drop=True)
R['block'] = ((R.t0 - R.t0.min()).dt.total_seconds() // (args.block_days * 86400)).astype(int)
BLK = [g.index.values for _, g in R.groupby('block')]
H = H.merge(R[['release', 'p_' + args.wband]], on='release')


def boot(stat):
    est = stat(np.arange(len(R)))
    v = [stat(np.concatenate([BLK[k] for k in rng.integers(0, len(BLK), len(BLK))])) for _ in range(args.nboot)]
    return est, np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)


def pooled(r, col, lo, hi):
    s = r[(r[col] >= lo) & (r[col] <= hi)]
    return s.sumW.sum() / s.days.sum() if len(s) else np.nan


rows = []
for b in BANDS:
    c = 'p_' + b
    ab = lambda ix: pooled(R.iloc[ix], c, 0, 0.001)
    pr = lambda ix: pooled(R.iloc[ix], c, 0.999, 1)
    e_a, la, ha = boot(ab)
    e_p, lp, hp = boot(pr)
    e_d, ld, hd = boot(lambda ix: pr(ix) - ab(ix))
    e_r, lr, hr = boot(lambda ix: np.corrcoef(R.iloc[ix][c], R.iloc[ix].rate)[0, 1])
    rows.append(dict(kind='presence band', band=b, mean_pres=R[c].mean(), n_absent=int((R[c] <= 0.001).sum()),
                     n_present=int((R[c] >= 0.999).sum()), rate_absent=e_a, rate_absent_lo=la, rate_absent_hi=ha,
                     rate_present=e_p, rate_present_lo=lp, rate_present_hi=hp, diff=e_d, diff_lo=ld, diff_hi=hd,
                     r=e_r, r_lo=lr, r_hi=hr))
c = 'p_' + args.wband
for half, lab in [(True, 'surface-half starters'), (False, 'bottom-half starters')]:
    Hh = H[H.surf0 == half].set_index('release').reindex(R.release)
    Hh['block'] = R.block.values
    f = lambda ix, lo, hi: pooled(Hh.iloc[ix].dropna(), c, lo, hi)
    e_a, la, ha = boot(lambda ix: f(ix, 0, 0.001))
    e_p, lp, hp = boot(lambda ix: f(ix, 0.999, 1))
    e_d, ld, hd = boot(lambda ix: f(ix, 0.999, 1) - f(ix, 0, 0.001))
    rr = (Hh.sumW / Hh.days).values
    e_r, lr, hr = boot(lambda ix: pd.Series(Hh[c].values[ix]).corr(pd.Series(rr[ix])))
    rows.append(dict(kind='start depth', band=lab, mean_pres=np.nan, n_absent=int((Hh[c] <= 0.001).sum()),
                     n_present=int((Hh[c] >= 0.999).sum()), rate_absent=e_a, rate_absent_lo=la, rate_absent_hi=ha,
                     rate_present=e_p, rate_present_lo=lp, rate_present_hi=hp, diff=e_d, diff_lo=ld, diff_hi=hd,
                     r=e_r, r_lo=lr, r_hi=hr))
S = pd.DataFrame(rows)
S.to_csv(out_dir / ('rate_by_depth_%s.csv' % args.wband), index=False, float_format='%.4g')
txt = ['WINDING RATE vs EDDY PRESENCE BY DEPTH -- particles from the %s-band winding run (%d releases)'
       % (args.wband, len(R)), 'rate in turns/day; CI = 95%% block bootstrap (%g-d blocks)' % args.block_days, '',
       S.to_string(index=False, float_format=lambda v: '%.3f' % v)]
(out_dir / ('summary_%s.txt' % args.wband)).write_text('\n'.join(txt) + '\n')
print('\n'.join(txt))

# ------------------------------------------------------------- figure ---
fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), layout='constrained')
ax = axs[0]
zb = ['z0-4', 'z4-10', 'z10-bed']
zc = dict(zip(zb, ['#9ecae1', '#3182bd', '#08306b']))
for b in [x for x in zb if x in BANDS]:
    bins = pd.cut(R['p_' + b], [-0.01, 0.001, 0.25, 0.5, 0.75, 0.999, 1.0])
    m = R.groupby(bins, observed=True).rate.median()
    xs = {iv: (0 if iv.right <= 0.001 else 1 if iv.left >= 0.999 else iv.mid) for iv in m.index}
    ax.plot([xs[iv] for iv in m.index], m.values, '-o', color=zc[b], lw=2, ms=5,
            label='eddy presence from %s (r = %.2f)' % (b, R['p_' + b].corr(R.rate)))
ax.axhline(0, color='0.3', lw=0.8, ls=':')
ax.set_xlabel('share of the first 3 d with the eddy present (pres3d)')
ax.set_ylabel('release winding rate, bin median [turns/day]')
ax.set_title('(a) same particles, presence measured at each depth band', fontsize=10, loc='left')
ax.legend(fontsize=8, frameon=False)
ax.grid(color='0.9', lw=0.6)

for ax, kind, ttl in [(axs[1], 'presence band', '(b) present - absent, by the band defining presence'),
                      (axs[2], 'start depth', '(c) present - absent (%s presence), by START depth' % args.wband)]:
    s = S[S.kind == kind].reset_index(drop=True)
    y = np.arange(len(s))[::-1]
    ax.errorbar(s['diff'], y, xerr=[s['diff'] - s.diff_lo, s.diff_hi - s['diff']], fmt='o', color='k',
                ecolor='0.4', capsize=3, ms=6)
    ax.set_yticks(y)
    ax.set_yticklabels(['%s  (n abs %d / pres %d)' % (b, a, pr_) for b, a, pr_ in zip(s.band, s.n_absent, s.n_present)],
                       fontsize=8)
    ax.axvline(0, color='0.3', lw=0.8, ls=':')
    ax.set_xlabel('rate present - absent [turns/day], 95% CI')
    ax.set_title(ttl, fontsize=10, loc='left')
    ax.grid(color='0.9', lw=0.6, axis='x')
fig.suptitle('Particle winding rate about the eddy centre vs eddy presence at other depths (winding from the %s run)'
             % args.wband, fontsize=10)
fn = out_dir / ('rate_by_depth_%s.png' % args.wband)
fig.savefig(fn, dpi=200, transparent=True, bbox_inches='tight')
plt.close(fig)
print('saved ' + str(fn))
