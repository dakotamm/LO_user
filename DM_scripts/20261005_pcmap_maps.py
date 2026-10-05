"""
First-pass pcmap analysis from the reduced files of 20261005_pcmap_reduce.py:
residence-time maps by ORIGIN cell, a region summary table, and region means
through the year for the E (peak of the strongest ebb) and F (strongest flood)
sets.

  fig 1  maps, rows = first-exit time / exposure at -cut days, columns = whole
         column / surface half / bottom half, pooled over the selected
         releases. Each cell shows the mean over the particles that STARTED
         there. Only cove cells are drawn, in a box around the cove.
  fig 2  quadrant-mean exposure vs release time, E and F as separate markers
         (needs many releases to be worth looking at)
  csv    region x set: n, mean/median first exit and exposure, fraction
         censored, fraction still inside at each cutoff

The set (E / F) comes from the directory name (..._E_<date> / ..._F_<date>);
anything else, e.g. the pcret/pcbot test runs, is set "other".

Forcing regressions (qprism, w_along, stratification) are not here yet --
they wait until enough of the year is reduced to be worth fitting.

run 20261005_pcmap_maps.py
run 20261005_pcmap_maps.py -set E -cut 10
run 20261005_pcmap_maps.py -glob 'pcret*'     (mac test)
"""
import argparse
import pickle
import re

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-glob', default='pcmap_3d*', help='reduced files to use')
p.add_argument('-set', default='all', choices=['all', 'E', 'F', 'other'])
p.add_argument('-cut', type=float, default=14.0, help='exposure cutoff [days]')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
red_dir = Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_reduce' / args.gtx
out_dir = Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_maps' / args.gtx
Lfun.make_dir(out_dir)
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
QNAMES = ['inner-N', 'inner-S', 'outer-N', 'outer-S']
ecol = 'exp_%gd_h' % args.cut

# ------------------------------------------------------------------ load ---
frames, curves, meta = [], [], []
for fn in sorted(red_dir.glob(args.glob + '.p')):
    D = pickle.load(open(fn, 'rb'))
    m = re.search(r'_([EF])_\d{4}\.\d{2}\.\d{2}$', D['meta']['dir'])
    s = m.group(1) if m else 'other'
    if args.set != 'all' and s != args.set:
        continue
    if ecol not in D['P']:
        raise SystemExit('%s has no %s; cutoffs were %s' % (fn.name, ecol, D['meta']['cutoffs']))
    P = D['P'].copy()
    P['set'] = s
    P['t0'] = D['meta']['t0']
    frames.append(P)
    meta.append(dict(set=s, t0=D['meta']['t0'], file=fn.name, n=len(P)))
    curves.append((s, D['curves']))
if not frames:
    raise SystemExit('no reduced files match %s/%s (set %s)' % (red_dir, args.glob, args.set))
A = pd.concat(frames, ignore_index=True)
M = pd.DataFrame(meta)
A['days_exit'] = A.first_exit_h / 24
A['days_exp'] = A[ecol] / 24
A['quad'] = np.array(QNAMES)[A.quad0]
A['half'] = np.where(A.surf0, 'surf', 'bot')
print('%d releases (%s), %d particles'
      % (len(M), ', '.join('%s %d' % kv for kv in M.set.value_counts().items()), len(A)))
tag = '%s_%s_cut%g' % (args.glob.replace('*', '').rstrip('_'), args.set, args.cut)

# ----------------------------------------------------------------- table ---
rows = []
for s in sorted(A.set.unique()) + (['pooled'] if A.set.nunique() > 1 else []):
    a = A if s == 'pooled' else A[A.set == s]
    for gname, g in [('cove', a)] + [(q, a[a.quad == q]) for q in QNAMES] + \
            [('%s-%s' % (q, hf), a[(a.quad == q) & (a.half == hf)])
             for q in QNAMES for hf in ['surf', 'bot']]:
        if len(g) == 0:
            continue
        rows.append(dict(set=s, region=gname, n=len(g),
                         exit_mean_d=g.days_exit.mean(), exit_med_d=g.days_exit.median(),
                         exp_mean_d=g.days_exp.mean(), exp_med_d=g.days_exp.median(),
                         censored=g.censored.mean()))
T = pd.DataFrame(rows)
# still-inside at each cutoff, release-averaged, from the stored curves
for s0 in T.set.unique():
    for gname in T.region[T.set == s0].unique():
        for c in frames[0].filter(like='exp_').columns:
            nh = int(c.split('_')[1][:-1]) * 24
            v = [cv[gname]['still'][nh] for s, cv in curves
                 if gname in cv and (s0 == 'pooled' or s == s0) and nh < len(cv[gname]['still'])]
            T.loc[(T.set == s0) & (T.region == gname), 'still_%s' % c.split('_')[1]] = \
                np.mean(v) if v else np.nan
pd.set_option('display.width', 220)
print(T[T.region.isin(['cove'] + QNAMES)].to_string(index=False, float_format=lambda x: '%.2f' % x))
T.to_csv(out_dir / ('pcmap_regions_%s.csv' % tag), index=False)

# ------------------------------------------------------------------ maps ---
g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat = g.lon_rho.values, g.lat_rho.values
g.close()
lon_ax, lat_ax = lon[0, :], lat[:, 0]
dlon, dlat = lon_ax[1] - lon_ax[0], lat_ax[1] - lat_ax[0]
NR, NC = lon.shape
cells = A[['j0', 'i0']].drop_duplicates()
j1, j2 = cells.j0.min() - 3, cells.j0.max() + 4
i1, i2 = cells.i0.min() - 3, cells.i0.max() + 4
xe = np.append(lon_ax[i1:i2] - dlon / 2, lon_ax[i2 - 1] + dlon / 2)
ye = np.append(lat_ax[j1:j2] - dlat / 2, lat_ax[j2 - 1] + dlat / 2)


def cell_field(a, col):
    f = np.full((NR, NC), np.nan)
    m = a.groupby(['j0', 'i0'])[col].mean()
    f[m.index.get_level_values(0), m.index.get_level_values(1)] = m.values
    return f[j1:j2, i1:i2]


fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True)
for r, (col, lab) in enumerate([('days_exit', 'first exit [d]'),
                                ('days_exp', 'exposure to %g d [d]' % args.cut)]):
    fields = [cell_field(A, col), cell_field(A[A.surf0], col), cell_field(A[~A.surf0], col)]
    vmax = np.nanpercentile(np.concatenate([f[np.isfinite(f)] for f in fields]), 98)
    for c, (f, ttl) in enumerate(zip(fields, ['whole column', 'surface half', 'bottom half'])):
        ax = axs[r, c]
        pc = ax.pcolormesh(xe, ye, f, vmin=0, vmax=vmax, cmap='viridis')
        ax.set_title('%s, %s' % (lab, ttl), fontsize=10)
        ax.grid(**GRID)
        ax.locator_params(axis='x', nbins=4)
        ax.set_aspect(1 / np.cos(np.deg2rad(lat_ax[j1:j2].mean())))
    fig.colorbar(pc, ax=axs[r, :], shrink=0.9, label=lab)
fig.suptitle('%s pcmap: residence time by origin cell, set %s, %d releases'
             % (args.gtx, args.set, len(M)), fontsize=12)
fn_out = out_dir / ('pcmap_maps_%s.png' % tag)
fig.savefig(fn_out, dpi=200, transparent=True)
plt.close(fig)
print('wrote %s' % fn_out)

# ------------------------------------------------------- through the year ---
if len(M) >= 4:
    R = A.groupby(['set', 't0', 'quad']).days_exp.mean().reset_index()
    fig, ax = plt.subplots(figsize=(13, 4.5))
    colors = dict(zip(QNAMES, ['#e8455e', '#f0a04b', '#4565e8', '#45b0a8']))
    for q in QNAMES:
        for s, mk in [('E', 'o'), ('F', 's'), ('other', '^')]:
            r = R[(R.quad == q) & (R.set == s)]
            if len(r):
                ax.plot(r.t0, r.days_exp, mk, ms=3, color=colors[q], alpha=0.7,
                        label='%s %s' % (q, s))
    ax.set_ylabel('mean exposure to %g d [d]' % args.cut)
    ax.set_title('quadrant-mean exposure by release (o = E, s = F)')
    ax.grid(**GRID)
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fn_out = out_dir / ('pcmap_series_%s.png' % tag)
    fig.savefig(fn_out, dpi=200, transparent=True)
    plt.close(fig)
    print('wrote %s' % fn_out)
