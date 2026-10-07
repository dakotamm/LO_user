"""
How fast the particles of one or more pcmap releases move, by starting class
(4 quadrants x surface / bottom half, pcmap_regions.py).

Two measures, both hourly, only while the particle is INSIDE the cove:
  speed      horizontal speed sqrt(u^2 + v^2) [cm/s] of the model velocity the
             tracker interpolated to the particle (instantaneous, at the hour)
  hourly     net displacement between consecutive hourly positions / 1 h
  displ.     [cm/s] -- what the hourly paths in the spaghetti plots show; lower
             than speed because tidal and turbulent motion within the hour
             partly cancels
plus |w| [mm/s], the vertical velocity at the particle.

Prints, for each release and class, the median and 90th percentile over all
inside-the-cove particle-hours in the first -hours (default 24.84, one tidal
day) and the first -days (default 7). Figure: the class-median horizontal
speed against hours from release for the first tidal day, one panel per
release (surface solid, bottom dashed, colour = quadrant), with ssh at pc_lp.

Reads the track files, so it runs on apogee.

Output: LO_output/DM_outs/20261007_pcmap_particle_speed/
  pcmap_particle_speed_<tag>.csv, pcmap_particle_speed_<tag>.png

run 20261007_pcmap_particle_speed.py                         (E and F 2025.07.10)
run 20261007_pcmap_particle_speed.py -rels E_2025.07.17,F_2025.07.17
run 20261007_pcmap_particle_speed.py -dir pcret_3d -file release_2024.02.15.nc   (mac test)
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun
from pcmap_regions import regions, QNAMES

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-rels', default='E_2025.07.10,F_2025.07.10', help='launcher sub_tags, comma-separated')
p.add_argument('-dir', default='', help='tracks2 output dir instead of -rels')
p.add_argument('-file', default='')
p.add_argument('-hours', type=float, default=24.84, help='first window [h]')
p.add_argument('-days', type=float, default=7.0, help='second window [d]')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
trk = Ldir['LOo'] / 'tracks2' / args.gtx
out_dir = Ldir['LOo'] / 'DM_outs' / '20261007_pcmap_particle_speed'
Lfun.make_dir(out_dir)
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
QCOL = dict(zip(QNAMES, ['#e8455e', '#f0a04b', '#4565e8', '#45b0a8']))

g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat = g.lon_rho.values, g.lat_rho.values
g.close()
lon_ax, lat_ax = lon[0, :], lat[:, 0]
dlon, dlat = lon_ax[1] - lon_ax[0], lat_ax[1] - lat_ax[0]
NR, NC = lon.shape
REG = regions(Ldir, lon, lat)
QUAD = REG['QUAD']

if args.dir:
    fns = [(args.dir, trk / args.dir / args.file)]
else:
    fns = []
    for rel in args.rels.split(','):
        dd = sorted(d for d in trk.glob('pcmap_3d*_' + rel) if d.is_dir())
        if len(dd) != 1:
            raise SystemExit('found %d dirs for %s' % (len(dd), rel))
        fns.append((rel, sorted(dd[0].glob('release_*.nc'))[0]))

hf_fn = Ldir['LOo'] / 'extract' / args.gtx / 'tef2' / 'hourly_flux_2024.01.01_2025.12.31_wb1_pc1.nc'
S = None
if hf_fn.is_file():
    hf = xr.open_dataset(hf_fn)
    S = pd.Series(hf.ssh.sel(sect='pc_lp').values, index=pd.to_datetime(hf.time.values))
    hf.close()

n1 = int(round(args.hours)) + 1
n2 = int(round(args.days * 24)) + 1
rows, curves = [], []
for tag, fn in fns:
    d = xr.open_dataset(fn)
    n2_ = min(n2, d.sizes['Time'])
    plon = d.lon.values[:n2_]; plat = d.lat.values[:n2_]; cs = d.cs.values[:n2_]
    u = d.u.values[:n2_]; v = d.v.values[:n2_]; w = d.w.values[:n2_]
    ot = pd.to_datetime(d.ot.values[:n2_])
    d.close()
    ok = np.isfinite(plon) & np.isfinite(plat)
    i = np.zeros(plon.shape, dtype=int); j = np.zeros(plon.shape, dtype=int)
    i[ok] = np.clip(np.round((plon[ok] - lon_ax[0]) / dlon), 0, NC - 1).astype(int)
    j[ok] = np.clip(np.round((plat[ok] - lat_ax[0]) / dlat), 0, NR - 1).astype(int)
    q = np.where(ok, QUAD[j, i], -1)
    keep = q[0] >= 0
    q, cs, plon, plat, u, v, w = q[:, keep], cs[:, keep], plon[:, keep], plat[:, keep], \
        u[:, keep], v[:, keep], w[:, keep]
    inside = q >= 0
    spd = np.hypot(u, v) * 100                                  # cm/s
    mx = 111320 * np.cos(np.deg2rad(plat)); my = 110540
    dx = np.diff(plon, axis=0) * mx[1:]; dy = np.diff(plat, axis=0) * my
    dsp = np.vstack([np.full((1, plon.shape[1]), np.nan), np.hypot(dx, dy) / 3600 * 100])   # cm/s
    dsp[1:][~(inside[1:] & inside[:-1])] = np.nan                # both ends inside the cove
    aw = np.abs(w) * 1000                                       # mm/s
    hrs = (ot - ot[0]) / pd.Timedelta(hours=1)
    classes = [('cove', np.ones(q.shape[1], dtype=bool))]
    for k, qn in enumerate(QNAMES):
        for hn, hv in [('surf', True), ('bot', False)]:
            classes.append(('%s-%s' % (qn, hn), (q[0] == k) & ((cs[0] >= -0.5) == hv)))
    for cname, m in classes:
        r = dict(release=tag, cls=cname, n=int(m.sum()))
        for wlab, nn in [('tday', n1), ('%gd' % args.days, n2_)]:
            ins = inside[:nn, m]
            for vname, A in [('speed', spd), ('displ', dsp), ('absw', aw)]:
                a = A[:nn, m][ins]
                a = a[np.isfinite(a)]
                r['%s_med_%s' % (vname, wlab)] = np.median(a) if len(a) else np.nan
                r['%s_p90_%s' % (vname, wlab)] = np.percentile(a, 90) if len(a) else np.nan
        rows.append(r)
        if cname != 'cove':
            A = np.where(inside[:n1, m], spd[:n1, m], np.nan)
            with np.errstate(all='ignore'):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    curves.append((tag, cname, hrs[:n1], np.nanmedian(A, axis=1)))
    if S is not None:
        curves.append((tag, 'ssh', hrs[:n1],
                       np.interp(ot[:n1].values.astype('int64'), S.index.values.astype('int64'), S.values)))

T = pd.DataFrame(rows)
otag = ('_'.join(t for t, _ in fns)).replace('.', '')
T.to_csv(out_dir / ('pcmap_particle_speed_%s.csv' % otag), index=False)
pd.set_option('display.width', 220)
show = ['release', 'cls', 'n', 'speed_med_tday', 'speed_p90_tday', 'displ_med_tday',
        'speed_med_%gd' % args.days, 'speed_p90_%gd' % args.days, 'absw_med_tday']
print('horizontal speed and hourly displacement [cm/s], |w| [mm/s]; particles inside the cove;'
      ' tday = first %.2f h' % args.hours)
print(T[show].to_string(index=False, float_format=lambda x: '%.1f' % x))

fig, axs = plt.subplots(2, len(fns), figsize=(7.5 * len(fns), 7), sharex=True, squeeze=False,
                        gridspec_kw=dict(height_ratios=[3, 1]))
for c, (tag, fn) in enumerate(fns):
    ax = axs[0, c]
    for t, cname, h, y in curves:
        if t != tag or cname == 'ssh':
            continue
        qn, hn = cname.rsplit('-', 1)
        ax.plot(h, y, color=QCOL[qn], lw=1.6, ls='-' if hn == 'surf' else '--',
                label='%s %s' % (qn, 'surface' if hn == 'surf' else 'bottom'))
    ax.set_title('%s: median horizontal speed of particles still in the cove' % tag, fontsize=10)
    ax.set_ylabel('speed [cm s$^{-1}$]')
    ax.set_ylim(bottom=0)
    ax.grid(**GRID)
    if c == 0:
        ax.legend(fontsize=7.5, ncol=2, loc='upper right')
    axs_ = axs[1, c]
    for t, cname, h, y in curves:
        if t == tag and cname == 'ssh':
            axs_.plot(h, y, color='#3b0f70', lw=1.2)
    axs_.set_ylabel('ssh at pc_lp [m]')
    axs_.set_xlabel('hours from release')
    axs_.set_xlim(0, args.hours)
    axs_.grid(**GRID)
fig.suptitle('%s pcmap particle speeds, first tidal day' % args.gtx, fontsize=12)
fig.tight_layout()
fn_out = out_dir / ('pcmap_particle_speed_%s.png' % otag)
fig.savefig(fn_out, dpi=200, transparent=True)
plt.close(fig)
print('wrote %s' % fn_out)
