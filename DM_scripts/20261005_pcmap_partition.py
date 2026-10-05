"""
Partition pcmap particles by where they started and by the dissolved oxygen
they started in, and compare residence time across the partitions.

Reads the reduced files of 20261005_pcmap_reduce.py. Every particle carries
its starting cell, depth, height above bed, salt, temp and DO, plus its full
hourly inside-the-cove record, so any partition gets real retention curves --
nothing has to be re-reduced to try a new one.

FACTORS (-by, comma-separated; two or more are crossed, e.g. -by DO,quad)
  quad     inner-N / inner-S / outer-N / outer-S          (starting cell)
  inner    inner (pc_cp_m) / outer
  ns       north / south of the along-cove centreline
  half     surface / bottom half of the column             (cs >= -0.5)
  x        along-cove position, -nx equal bins of starting rho column i
  hab      height above bed at release, bins -hab_edges [m]
  depth    depth at release, bins -depth_edges [m]
  DO       DO at release, bins -do_edges [mg/L]; default <2 (hypoxic), 2-4,
           4-6, >6
  DOrel    DO tercile WITHIN each release (low / mid / high for that day).
           DO is strongly seasonal and sits mostly in the inner bottom water,
           so absolute DO bins mostly re-sort particles by season and place;
           DOrel asks whether the relatively low-DO water of a given day is
           retained longer than that day's high-DO water.
  season   DJF / MAM / JJA / SON of the release
  set      E (strongest ebb) / F (strongest flood)

For each group: number of particles and of releases contributing, the
"still inside" and "never left" curves (particle-pooled), the e-fold of
"still inside", median first exit, mean exposure to -cut days and the fraction
censored. Pooled curves weight releases by how many of their particles fall
in the group, so a DO<2 group is almost all late summer by construction --
the composition table printed with it says so.

Outputs, to LO_output/DM_outs/20261005_pcmap_partition/<gtx>/:
  pcmap_partition_<by>.csv          group metrics
  pcmap_partition_<by>_curves.png   curves; panels = levels of the 2nd
                                    factor (if any), lines = levels of the 1st
  pcmap_partition_<by>_comp.csv     particle counts by group x season

run 20261005_pcmap_partition.py -by quad
run 20261005_pcmap_partition.py -by DO
run 20261005_pcmap_partition.py -by DOrel,quad
run 20261005_pcmap_partition.py -by DO,season -do_edges 3,5
run 20261005_pcmap_partition.py -by hab -glob 'pcret*'          (mac test)
"""
import argparse
import pickle
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun

FACTORS = ['quad', 'inner', 'ns', 'half', 'x', 'hab', 'depth', 'DO', 'DOrel', 'season', 'set']
p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-glob', default='pcmap_3d*', help='reduced files to use')
p.add_argument('-set', default='all', choices=['all', 'E', 'F', 'other'])
p.add_argument('-by', default='quad', help='comma list from: ' + ', '.join(FACTORS))
p.add_argument('-cut', type=float, default=14.0, help='exposure cutoff [days]')
p.add_argument('-do_edges', default='2,4,6', help='DO bin edges [mg/L]')
p.add_argument('-hab_edges', default='2,5,10', help='height-above-bed bin edges [m]')
p.add_argument('-depth_edges', default='5,10,15', help='depth bin edges [m, positive]')
p.add_argument('-nx', type=int, default=4, help='along-cove bins for factor x')
args = p.parse_args()
by = args.by.split(',')
bad = [f for f in by if f not in FACTORS]
if bad:
    raise SystemExit('unknown factor(s) %s; choose from %s' % (bad, FACTORS))

Ldir = Lfun.Lstart(gridname='wb1')
red_dir = Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_reduce' / args.gtx
out_dir = Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_partition' / args.gtx
Lfun.make_dir(out_dir)
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
QNAMES = np.array(['inner-N', 'inner-S', 'outer-N', 'outer-S'])
SEASON = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM',
          6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON'}
ecol = 'exp_%gd_h' % args.cut


def bin_labels(edges, unit):
    e = [float(v) for v in edges.split(',')]
    labs = ['<%g' % e[0]] + ['%g-%g' % (a, b) for a, b in zip(e[:-1], e[1:])] + ['>%g' % e[-1]]
    return e, [l + ' ' + unit for l in labs]


def cut(x, edges, labels):
    k = np.searchsorted(edges, x, side='right')
    out = np.array(labels, dtype=object)[k]
    out[~np.isfinite(x)] = 'nan'
    return out


fns = sorted(red_dir.glob(args.glob + '.p'))
rel = []
for fn in fns:
    D = pickle.load(open(fn, 'rb'))
    if 'inside_bits' not in D:
        raise SystemExit('%s predates the partition fields; rerun the reduce with -clobber' % fn.name)
    m = re.search(r'_([EF])_\d{4}\.\d{2}\.\d{2}$', D['meta']['dir'])
    s = m.group(1) if m else 'other'
    if args.set == 'all' or s == args.set:
        rel.append((fn, s, D))
if not rel:
    raise SystemExit('no reduced files match %s/%s (set %s)' % (red_dir, args.glob, args.set))
if 'DO' in by or 'DOrel' in by:
    if all(D['P'].DO0.isna().all() for _, _, D in rel):
        raise SystemExit('no DO in these reduced files (reduced with -no_do?)')
nf = min(D['meta']['nf'] for _, _, D in rel)
i_all = np.concatenate([D['P'].i0.values for _, _, D in rel])
x_edges = np.quantile(i_all, np.linspace(0, 1, args.nx + 1))[1:-1]


def factor(name, P, s, t0):
    if name == 'quad':
        return QNAMES[P.quad0.values]
    if name == 'inner':
        return np.where(P.quad0.values < 2, 'inner', 'outer')
    if name == 'ns':
        return np.where(P.quad0.values % 2 == 0, 'north', 'south')
    if name == 'half':
        return np.where(P.surf0.values, 'surface', 'bottom')
    if name == 'x':
        # i increases toward the mouth; label 1 = innermost
        return np.array(['x%d' % (k + 1) for k in np.searchsorted(x_edges, P.i0.values, side='right')])
    if name == 'hab':
        e, l = bin_labels(args.hab_edges, 'm ab')
        return cut(P.hab0.values, e, l)
    if name == 'depth':
        e, l = bin_labels(args.depth_edges, 'm deep')
        return cut(-P.z0.values, e, l)
    if name == 'DO':
        e, l = bin_labels(args.do_edges, 'mg/L')
        return cut(P.DO0.values, e, l)
    if name == 'DOrel':
        q = P.DO0.rank(pct=True).values
        return np.select([q <= 1 / 3, q <= 2 / 3], ['DO low', 'DO mid'], 'DO high')
    if name == 'season':
        return np.full(len(P), SEASON[t0.month], dtype=object)
    if name == 'set':
        return np.full(len(P), s, dtype=object)


ORDER = {'DJF': 0, 'MAM': 1, 'JJA': 2, 'SON': 3, 'DO low': 0, 'DO mid': 1, 'DO high': 2,
         'inner': 0, 'outer': 1, 'north': 0, 'south': 1, 'surface': 0, 'bottom': 1}


def lev_key(lab):
    """Natural order for one factor level: bins by their numbers, seasons and
    terciles in sequence, everything else alphabetical."""
    if lab in ORDER:
        return (0, ORDER[lab], '')
    m = re.match(r'([<>]?)([\d.]+)', lab)
    if m:
        return (1, float(m.group(2)) + {'<': -1e-6, '>': 1e-6, '': 0}[m.group(1)], '')
    return (2, 0, lab)


def grp_key(g):
    return tuple(lev_key(l) for l in g.split(' | '))


# accumulate per group: particle counts, curve sums, metric lists
acc = {}
comp = []
for fn, s, D in rel:
    P = D['P']
    t0 = pd.Timestamp(D['meta']['t0'])
    ins = np.unpackbits(D['inside_bits'], axis=0)[:nf].astype(bool)
    nev = np.minimum.accumulate(ins, axis=0)
    key = np.array([' | '.join(t) for t in zip(*[factor(f, P, s, t0) for f in by])])
    for g in np.unique(key):
        m = key == g
        a = acc.setdefault(g, dict(n=0, nrel=0, still=np.zeros(nf), never=np.zeros(nf),
                                   exit=[], exp=[], cens=[]))
        a['n'] += m.sum(); a['nrel'] += 1
        a['still'] += ins[:, m].sum(axis=1); a['never'] += nev[:, m].sum(axis=1)
        a['exit'].append(P.first_exit_h.values[m]); a['exp'].append(P[ecol].values[m])
        a['cens'].append(P.censored.values[m])
        comp.append(dict(group=g, season=SEASON[t0.month], n=int(m.sum())))

hours = np.arange(nf)
rows = []
for g in sorted(acc, key=grp_key):
    a = acc[g]
    st = a['still'] / a['n']
    k = np.where(st < 1 / np.e)[0]
    rows.append(dict(group=g, n=a['n'], n_releases=a['nrel'],
                     efold_still_d=hours[k[0]] / 24 if len(k) else np.nan,
                     exit_med_d=np.median(np.concatenate(a['exit'])) / 24,
                     exp_mean_d=np.mean(np.concatenate(a['exp'])) / 24,
                     censored=np.mean(np.concatenate(a['cens'])),
                     still_7d=st[min(7 * 24, nf - 1)], still_end=st[-1]))
T = pd.DataFrame(rows)
tag = '_'.join(by) + ('' if args.set == 'all' else '_' + args.set)
pd.set_option('display.width', 220)
print('%d releases, %d particles, record %.1f d, partition by %s'
      % (len(rel), T.n.sum(), (nf - 1) / 24, ' x '.join(by)))
print(T.to_string(index=False, float_format=lambda v: '%.2f' % v))
T.to_csv(out_dir / ('pcmap_partition_%s.csv' % tag), index=False)

C = pd.DataFrame(comp).groupby(['group', 'season']).n.sum().unstack(fill_value=0)
C = C.loc[sorted(C.index, key=grp_key)]
C = C.reindex(columns=[c for c in ['DJF', 'MAM', 'JJA', 'SON'] if c in C.columns])
print('\nparticles by group x season (what the pooled curves are made of):')
print(C.to_string())
C.to_csv(out_dir / ('pcmap_partition_%s_comp.csv' % tag))

# ----------------------------------------------------------------- figure ---
levels1 = sorted(set(g.split(' | ')[0] for g in acc), key=lev_key)
levels2 = sorted(set(g.split(' | ')[1] for g in acc), key=lev_key) if len(by) > 1 else ['']
cmap = plt.get_cmap('viridis', max(len(levels1), 2))
fig, axs = plt.subplots(1, len(levels2), figsize=(4.2 * len(levels2) + 1, 4.2),
                        sharey=True, squeeze=False)
dd = hours / 24
for ax, l2 in zip(axs[0], levels2):
    for c, l1 in enumerate(levels1):
        g = l1 if len(by) == 1 else '%s | %s' % (l1, l2)
        if g not in acc:
            continue
        a = acc[g]
        ax.plot(dd, a['still'] / a['n'], color=cmap(c), lw=1.4,
                label='%s (n %d)' % (l1, a['n']))
        ax.plot(dd, a['never'] / a['n'], color=cmap(c), lw=0.9, ls='--')
    ax.axhline(1 / np.e, color='0.5', lw=0.8, ls=':')
    ax.set_title(l2 if l2 else 'by %s' % by[0], fontsize=10)
    ax.set_xlabel('days from release')
    ax.grid(**GRID)
    ax.legend(fontsize=7, loc='upper right')
axs[0, 0].set_ylabel('fraction still inside the cove\n(dashed = never left)')
axs[0, 0].set_ylim(0, 1.02)
fig.suptitle('%s pcmap: retention by %s, %d releases (set %s)'
             % (args.gtx, ' x '.join(by), len(rel), args.set), fontsize=11)
fig.tight_layout()
fn_out = out_dir / ('pcmap_partition_%s_curves.png' % tag)
fig.savefig(fn_out, dpi=200, transparent=True)
plt.close(fig)
print('\nwrote %s' % out_dir)
