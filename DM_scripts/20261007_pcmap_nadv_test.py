"""
Does the tracker's vertical random walk keep particles evenly spread through
the water column? Compares no-advection tests (tracker.py -no_advection True,
sub_tag <release>_nadvtest) with the normal runs of the same releases.

With no advection a particle stays in its starting column and moves only by the
turbulent random walk (dAKs/dz drift + sqrt(2 AKs / dt) noise, reflection at
surface and bed). A correct scheme preserves an even distribution (the
well-mixed condition) whatever the AKs profile, so any thinning of the top and
bottom of the column in the no-advection run is a NUMERICAL artefact. The
normal run's inside-the-cove profile is drawn alongside to show how much of its
shape the artefact accounts for.

Height is fractional, cs + 1 (0 = bed, 1 = surface), in -nbins bins. The
starting distribution is not exactly flat in 10 bins (particles start at sigma
cell centres ~2 m apart), so the day-0 profile is drawn as the reference.
Columns are also split by local depth h at the start: shallow (< -h_split[0]),
mid, deep (> -h_split[1]).

  fig 1  per test release (rows): profiles at -snap hours, no-advection
         (solid) vs normal run inside the cove (dashed), all columns
  fig 2  time-height maps of fraction / even, no-advection runs, one row per
         release, columns = shallow / mid / deep columns
  stdout fraction in the bed and surface tenths over time, both runs

Reads the track files, so it runs on apogee.

Output: LO_output/DM_outs/20261007_pcmap_nadv_test/

run 20261007_pcmap_nadv_test.py
run 20261007_pcmap_nadv_test.py -rels E_2025.07.10
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from lo_tools import Lfun
from pcmap_regions import regions

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-rels', default='E_2025.02.02,E_2025.07.10,E_2025.12.12')
p.add_argument('-nbins', type=int, default=10)
p.add_argument('-hours', type=int, default=72, help='length compared [h]')
p.add_argument('-snap', default='0,6,24,72', help='snapshot hours')
p.add_argument('-h_split', default='12,18', help='shallow / deep column split [m]')
args = p.parse_args()
snaps = [int(x) for x in args.snap.split(',')]
hs = [float(x) for x in args.h_split.split(',')]

Ldir = Lfun.Lstart(gridname='wb1')
trk = Ldir['LOo'] / 'tracks2' / args.gtx
out_dir = Ldir['LOo'] / 'DM_outs' / '20261007_pcmap_nadv_test'
Lfun.make_dir(out_dir)
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat = g.lon_rho.values, g.lat_rho.values
g.close()
lon_ax, lat_ax = lon[0, :], lat[:, 0]
dlon, dlat = lon_ax[1] - lon_ax[0], lat_ax[1] - lat_ax[0]
NR, NC = lon.shape
cove = regions(Ldir, lon, lat)['cove']
edges = np.linspace(0, 1, args.nbins + 1)
even = 1 / args.nbins


def find_dir(pattern):
    dd = sorted(d for d in trk.glob(pattern) if d.is_dir())
    if len(dd) != 1:
        raise SystemExit('found %d dirs for %s in %s' % (len(dd), pattern, trk))
    return sorted(dd[0].glob('release_*.nc'))[0]


def load(fn, nh):
    d = xr.open_dataset(fn)
    n = min(nh + 1, d.sizes['Time'])
    plon, plat, cs, h0 = d.lon.values[:n], d.lat.values[:n], d.cs.values[:n], d.h.values[0]
    d.close()
    ok = np.isfinite(plon) & np.isfinite(plat)
    i = np.zeros(plon.shape, dtype=int); j = np.zeros(plon.shape, dtype=int)
    i[ok] = np.clip(np.round((plon[ok] - lon_ax[0]) / dlon), 0, NC - 1).astype(int)
    j[ok] = np.clip(np.round((plat[ok] - lat_ax[0]) / dlat), 0, NR - 1).astype(int)
    ins = ok & cove[j, i]
    return cs + 1, ins, h0


def frac(H, sel):
    """Fraction per height bin per hour over the selected particle-hours."""
    out = np.full((H.shape[0], args.nbins), np.nan)
    for t in range(H.shape[0]):
        x = H[t][sel[t] & np.isfinite(H[t])]
        if len(x) >= 20:
            out[t] = np.histogram(np.clip(x, 0, 1 - 1e-9), bins=edges)[0] / len(x)
    return out


R = {}
for rel in args.rels.split(','):
    Hn, In, h0 = load(find_dir('pcmap_3d*_nadv_%s_nadvtest' % rel), args.hours)
    Hr, Ir, _ = load(find_dir('pcmap_3d*_%s' % rel), args.hours)
    keep = In[0]
    Hn, In, h0n = Hn[:, keep], In[:, keep], h0[keep]
    keep_r = Ir[0]
    Hr, Ir = Hr[:, keep_r], Ir[:, keep_r]
    allt = np.ones_like(In)
    depth_cls = {'shallow (h < %g m)' % hs[0]: h0n < hs[0],
                 'mid (%g-%g m)' % (hs[0], hs[1]): (h0n >= hs[0]) & (h0n <= hs[1]),
                 'deep (h > %g m)' % hs[1]: h0n > hs[1]}
    R[rel] = dict(nadv=frac(Hn, allt), real_in=frac(Hr, Ir),
                  bydepth={k: frac(Hn[:, m], allt[:, m]) for k, m in depth_cls.items()},
                  n=int(keep.sum()), ndepth={k: int(m.sum()) for k, m in depth_cls.items()})
    print('\n%s: %d particles; columns %s' % (rel, keep.sum(),
          ', '.join('%s %d' % kv for kv in R[rel]['ndepth'].items())))
    print('  %5s | %-30s | %-30s' % ('hour', 'no advection: bed / surface tenth', 'normal run (in cove): bed / surface'))
    for t in snaps:
        if t < R[rel]['nadv'].shape[0]:
            a, b = R[rel]['nadv'][t], R[rel]['real_in'][t]
            print('  %5d | %.3f / %.3f (even %.2f)        | %.3f / %.3f'
                  % (t, a[0], a[-1], even, b[0], b[-1]))
    for k, f in R[rel]['bydepth'].items():
        t = min(args.hours, f.shape[0] - 1)
        print('  no advection, %-20s at %d h: bed %.3f, surface %.3f' % (k, t, f[t, 0], f[t, -1]))

rels = list(R)
# ---------------------------------------------------------- fig 1 -------
cmap = plt.get_cmap('viridis')
fig, axs = plt.subplots(1, len(rels), figsize=(5.2 * len(rels), 6), sharey=True, squeeze=False)
for c, rel in enumerate(rels):
    ax = axs[0, c]
    for k, t in enumerate(snaps):
        if t >= R[rel]['nadv'].shape[0]:
            continue
        col = '0.2' if t == 0 else cmap(k / max(len(snaps) - 1, 1))
        a = R[rel]['nadv'][t]
        ax.plot(np.r_[a, a[-1]], edges, drawstyle='steps-post', color=col, lw=2.0,
                ls=':' if t == 0 else '-', label='%d h, no advection' % t)
        if t > 0:
            b = R[rel]['real_in'][t]
            ax.plot(np.r_[b, b[-1]], edges, drawstyle='steps-post', color=col, lw=1.2, ls='--',
                    label='%d h, normal run (in cove)' % t)
    ax.axvline(even, color='k', lw=1, ls='--')
    ax.set_ylim(0, 1)
    ax.set_title('%s (n %d)' % (rel, R[rel]['n']), fontsize=11)
    ax.set_xlabel('fraction of particles in bin')
    ax.grid(**GRID)
axs[0, 0].set_ylabel('height in column [0 bed, 1 surface]')
axs[0, 0].legend(fontsize=7.5, loc='upper right')
fig.suptitle('%s: vertical random walk alone (-no_advection) vs normal run; even = %.2f'
             % (args.gtx, even), fontsize=12)
fig.tight_layout()
fn_out = out_dir / 'pcmap_nadv_test_profiles.png'
fig.savefig(fn_out, dpi=200, transparent=True)
plt.close(fig)
print('\nwrote %s' % fn_out)

# ---------------------------------------------------------- fig 2 -------
dk = list(R[rels[0]]['bydepth'])
fig, axs = plt.subplots(len(rels), len(dk), figsize=(5 * len(dk), 3.2 * len(rels)), sharex=True,
                        sharey=True, squeeze=False)
norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=2)
for r, rel in enumerate(rels):
    for c, k in enumerate(dk):
        ax = axs[r, c]
        f = R[rel]['bydepth'][k]
        pc = ax.pcolormesh(np.arange(f.shape[0] + 1), edges, (f / even).T, cmap='RdBu_r', norm=norm,
                           shading='flat')
        ax.set_title('%s, %s columns (n %d)' % (rel, k, R[rel]['ndepth'][k]), fontsize=9.5)
        if r == len(rels) - 1:
            ax.set_xlabel('hours from release')
        if c == 0:
            ax.set_ylabel('height in column')
fig.colorbar(pc, ax=axs, shrink=0.85, pad=0.01, label='fraction / even (1 = even)')
fig.suptitle('%s: no-advection runs, particle height distribution by column depth' % args.gtx, fontsize=12)
fn_out = out_dir / 'pcmap_nadv_test_timeheight.png'
fig.savefig(fn_out, dpi=200, transparent=True, bbox_inches='tight')
plt.close(fig)
print('wrote %s' % fn_out)
