"""
Quick look at finished pcmap releases straight from the track files -- no
reduce step. Meant for checking the first releases while the launcher is still
running, so by default it only opens releases the launcher has logged as
finished (returncode 0 in 20261005_pcmap_launch/pcmap_timing.csv); files of
releases still running are partly written.

Same cove and quadrants as 20261005_pcmap_reduce.py (from each particle's
initial cell): cove = pc_cp_m + pc_cp_p + pc_lp_m, inner = pc_cp_m, N/S split
at each column's mean j, surface/bottom split at cs = -0.5.

  fig 1  fraction still inside the cove (solid) and never left (dashed) vs
         days, whole cove and each quadrant of origin, one line per release,
         red = E (strongest ebb), blue = F (strongest flood)
  fig 2  maps by origin cell, pooled over releases: first exit and exposure
         (hours inside, re-entry counted) to -days, for the whole column,
         surface half and bottom half
  stdout one line per release: e-fold of "still inside", median first exit,
         fraction still inside at 7 / 10 / 14 d

Output: LO_output/DM_outs/20261005_pcmap_quicklook/

run 20261005_pcmap_quicklook.py
run 20261005_pcmap_quicklook.py -all      (every complete file, ignore the timing log)
"""
import argparse
import pickle

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-dir_glob', default='pcmap_3d*')
p.add_argument('-days', type=float, default=14.0)
p.add_argument('-all', action='store_true', help='do not filter on the timing log')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
trk = Ldir['LOo'] / 'tracks2' / args.gtx
out_dir = Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_quicklook'
Lfun.make_dir(out_dir)
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
QNAMES = ['inner-N', 'inner-S', 'outer-N', 'outer-S']
SETC = {'E': '#e8455e', 'F': '#4565e8'}

# ------------------------------------------------------------- regions ---
g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat = g.lon_rho.values, g.lat_rho.values
g.close()
lon_ax, lat_ax = lon[0, :], lat[:, 0]
dlon, dlat = lon_ax[1] - lon_ax[0], lat_ax[1] - lat_ax[0]
NR, NC = lon.shape
seg = pickle.load(open(sorted((Ldir['LOo'] / 'extract' / 'tef2').glob(
    'seg_info_dict_wb1_pc1_*.p'))[0], 'rb'))


def seg_mask(names):
    m = np.zeros((NR, NC), dtype=bool)
    for s in names:
        a = np.array(seg[s]['ji_list'])
        m[a[:, 0], a[:, 1]] = True
    return m


cove = seg_mask(['pc_cp_m', 'pc_cp_p', 'pc_lp_m'])
inner = seg_mask(['pc_cp_m'])
jj, ii = np.where(cove)
north = np.zeros((NR, NC), dtype=bool)
for i in np.unique(ii):
    north[jj[(ii == i) & (jj > jj[ii == i].mean())], i] = True
QUAD = np.full((NR, NC), -1, dtype=int)
QUAD[cove] = (2 * (~inner) + (~north))[cove]

# ---------------------------------------------------------------- files ---
fns = sorted(f for dd in sorted(trk.glob(args.dir_glob)) if dd.is_dir()
             for f in dd.glob('release_*.nc'))
if not args.all:
    T = pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_launch' / 'pcmap_timing.csv')
    ok = set(T.log[T.returncode == 0].str.replace('.log', '', regex=False))
    fns = [f for f in fns if f.parent.name in ok]
print('%d finished releases' % len(fns))
if not fns:
    raise SystemExit('nothing to plot')

nf = int(round(args.days * 24)) + 1
dd = np.arange(nf) / 24
rel, parts = [], []
for fn in fns:
    s = fn.parent.name.split('_')[-2]                       # E or F
    d = xr.open_dataset(fn)
    if d.sizes['Time'] < nf:
        print('SKIP %s: %d frames' % (fn.parent.name, d.sizes['Time']))
        d.close()
        continue
    plon = d.lon.values[:nf]; plat = d.lat.values[:nf]; cs0 = d.cs.values[0]
    t0 = pd.Timestamp(d.ot.values[0])
    d.close()
    okp = np.isfinite(plon) & np.isfinite(plat)
    i = np.zeros(plon.shape, dtype=int); j = np.zeros(plon.shape, dtype=int)
    i[okp] = np.clip(np.round((plon[okp] - lon_ax[0]) / dlon), 0, NC - 1).astype(int)
    j[okp] = np.clip(np.round((plat[okp] - lat_ax[0]) / dlat), 0, NR - 1).astype(int)
    q = np.where(okp, QUAD[j, i], -1)
    keep = q[0] >= 0
    q, cs0, j0, i0 = q[:, keep], cs0[keep], j[0, keep], i[0, keep]
    inside = q >= 0
    never = np.minimum.accumulate(inside, axis=0)
    b = ~inside[1:]
    k = np.argmax(b, axis=0) + 1
    k[~b.any(axis=0)] = nf - 1
    curves = {'cove': (inside.mean(1), never.mean(1))}
    for kq, nq in enumerate(QNAMES):
        m = q[0] == kq
        curves[nq] = (inside[:, m].mean(1), never[:, m].mean(1))
    rel.append(dict(name=fn.parent.name, set=s, t0=t0, curves=curves))
    parts.append(pd.DataFrame(dict(j0=j0, i0=i0, surf=cs0 >= -0.5,
                                   exit_d=k / 24, exp_d=inside[1:].sum(0) / 24)))
    st = curves['cove'][0]
    ef = np.where(st < 1 / np.e)[0]
    print('%-32s %s  NP %d  e-fold %5.2f d  median first exit %5.2f d  '
          'still in @7/10/14 d %.3f / %.3f / %.3f'
          % (fn.parent.name, t0, keep.sum(), dd[ef[0]] if len(ef) else np.nan,
             np.median(k) / 24, st[7 * 24], st[10 * 24], st[-1]))

# --------------------------------------------------------------- curves ---
fig, axs = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
for ax, gname in zip(axs, ['cove'] + QNAMES):
    for r in rel:
        still, nev = r['curves'][gname]
        ax.plot(dd, still, color=SETC.get(r['set'], '0.4'), lw=0.8, alpha=0.7)
        ax.plot(dd, nev, color=SETC.get(r['set'], '0.4'), lw=0.6, ls='--', alpha=0.7)
    ax.axhline(1 / np.e, color='0.5', lw=0.8, ls=':')
    ax.set_title('whole cove' if gname == 'cove' else '%s (started there)' % gname, fontsize=10)
    ax.set_xlabel('days from release')
    ax.grid(**GRID)
axs[0].set_ylabel('fraction still inside the cove')
axs[0].set_ylim(0, 1.02)
for s in ['E', 'F']:
    axs[-1].plot([], [], color=SETC[s], label=s)
axs[-1].plot([], [], color='0.3', ls='--', label='never left')
axs[-1].legend(fontsize=8)
fig.suptitle('%s pcmap quick look: %d finished releases (%s to %s)'
             % (args.gtx, len(rel), min(r['t0'] for r in rel).date(),
                max(r['t0'] for r in rel).date()), fontsize=12)
fig.tight_layout()
fig.savefig(out_dir / 'pcmap_quicklook_curves.png', dpi=200, transparent=True)
plt.close(fig)

# ----------------------------------------------------------------- maps ---
A = pd.concat(parts, ignore_index=True)
j1, j2 = A.j0.min() - 3, A.j0.max() + 4
i1, i2 = A.i0.min() - 3, A.i0.max() + 4
xe = np.append(lon_ax[i1:i2] - dlon / 2, lon_ax[i2 - 1] + dlon / 2)
ye = np.append(lat_ax[j1:j2] - dlat / 2, lat_ax[j2 - 1] + dlat / 2)


def cell_field(a, col):
    f = np.full((NR, NC), np.nan)
    m = a.groupby(['j0', 'i0'])[col].mean()
    f[m.index.get_level_values(0), m.index.get_level_values(1)] = m.values
    return f[j1:j2, i1:i2]


fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True)
for r, (col, lab) in enumerate([('exit_d', 'first exit [d]'),
                                ('exp_d', 'exposure to %g d [d]' % args.days)]):
    fields = [cell_field(A, col), cell_field(A[A.surf], col), cell_field(A[~A.surf], col)]
    vmax = np.nanpercentile(np.concatenate([f[np.isfinite(f)] for f in fields]), 98)
    for c, (f, ttl) in enumerate(zip(fields, ['whole column', 'surface half', 'bottom half'])):
        ax = axs[r, c]
        pc = ax.pcolormesh(xe, ye, f, vmin=0, vmax=vmax, cmap='viridis')
        ax.set_title('%s, %s' % (lab, ttl), fontsize=10)
        ax.grid(**GRID)
        ax.locator_params(axis='x', nbins=4)
        ax.set_aspect(1 / np.cos(np.deg2rad(lat_ax[j1:j2].mean())))
    fig.colorbar(pc, ax=axs[r, :], shrink=0.9, label=lab)
fig.suptitle('%s pcmap quick look: residence time by origin cell, %d releases pooled'
             % (args.gtx, len(rel)), fontsize=12)
fig.savefig(out_dir / 'pcmap_quicklook_maps.png', dpi=200, transparent=True)
plt.close(fig)
print('wrote %s' % out_dir)
