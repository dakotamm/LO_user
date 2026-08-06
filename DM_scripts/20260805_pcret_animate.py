"""
Animate the pcret particle releases, one movie per starting segment.

Each movie shows a SINGLE cohort, February on the left and August on the right
on the same clock, so within a movie the comparison is seasonal and between
movies it is where the water started. Overlaying all cohorts hides exactly what
you want to see -- with three clouds superimposed you cannot tell whether the
inner-cove water is lingering or has been replaced by mid-cove water drifting
in.

Below the maps the retention curves for that cohort run with a marker showing
where the animation is.

Only the first NDAYS are animated: the curves converge by about day 14 and the
long tail adds runtime without adding anything to watch. Sampled every 2 hours,
which resolves the 12.4 h tidal cycle -- 6-hourly aliases it and the particles
appear to jitter rather than breathe.

run 20260805_pcret_animate.py
run 20260805_pcret_animate.py -cohorts pc_cp_m -ndays 30 -step 1
"""
import argparse
import pickle
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

parser = argparse.ArgumentParser()
parser.add_argument('-cohorts', default='pc_cp_m,pc_cp_p,pc_lp_m,pc_lp_p',
                    type=str, help='comma-separated segments, one movie each')
parser.add_argument('-ndays', default=20, type=int)
parser.add_argument('-step', default=2, type=int, help='hours between frames')
parser.add_argument('-fps', default=20, type=int)
parser.add_argument('-tag', default='', type=str, help='suffix for the output filename')
args = parser.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
trk = Ldir['LOo'] / 'tracks2' / 'wb1_t0_xn11abbur00' / 'pcret_3d'
out_dir = Path.home() / 'Desktop' / 'pltz'
Lfun.make_dir(out_dir)

g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon = g.lon_rho.values
lat = g.lat_rho.values
mask = g.mask_rho.values
h = g.h.values
g.close()
lx, ly = lon[0, :], lat[:, 0]
dx, dy = lx[1] - lx[0], ly[1] - ly[0]
NR, NC = lon.shape

tef2 = Ldir['LOo'] / 'extract' / 'tef2'
seg = pickle.load(open(sorted(tef2.glob('seg_info_dict_wb1_pc1_*.p'))[0], 'rb'))
COVE = ['pc_cp_m', 'pc_cp_p', 'pc_lp_m']
LABEL = {'pc_cp_m': 'inner cove', 'pc_cp_p': 'mid cove',
         'pc_lp_m': 'outer cove', 'pc_lp_p': 'upper Saratoga'}
COLOR = {'pc_cp_m': '#009E73', 'pc_cp_p': '#0072B2',
         'pc_lp_m': '#CC79A7', 'pc_lp_p': '#D55E00'}

seg_mask = {}
for s in seg:
    m = np.zeros((NR, NC), dtype=bool)
    a = np.array(seg[s]['ji_list'])
    m[a[:, 0], a[:, 1]] = True
    seg_mask[s] = m
cove_mask = np.zeros((NR, NC), dtype=bool)
for s in COVE:
    cove_mask |= seg_mask[s]

rels = sorted(f.stem.replace('release_', '') for f in trk.glob('release_*.nc'))
D = {}
for rel in rels:
    d = xr.open_dataset(trk / ('release_' + rel + '.nc'))
    plon = d.lon.values
    plat = d.lat.values
    d.close()
    cs0 = xr.open_dataset(trk / ('release_' + rel + '.nc')).cs.values[0, :]
    ok = np.isfinite(plon) & np.isfinite(plat)
    i = np.full(plon.shape, -1, dtype=int)
    j = np.full(plon.shape, -1, dtype=int)
    i[ok] = np.clip(np.round((plon[ok] - lx[0]) / dx), 0, NC - 1).astype(int)
    j[ok] = np.clip(np.round((plat[ok] - ly[0]) / dy), 0, NR - 1).astype(int)
    ins = np.zeros(plon.shape, dtype=bool)
    ins[ok] = cove_mask[j[ok], i[ok]]
    home = {s: np.zeros(plon.shape, dtype=bool) for s in seg}
    for s in seg:
        home[s][ok] = seg_mask[s][j[ok], i[ok]]
    coh = np.array(['none'] * plon.shape[1], dtype=object)
    for s in seg:
        coh[seg_mask[s][j[0, :], i[0, :]]] = s
    D[rel] = dict(lon=plon, lat=plat, coh=coh, ins=ins, home=home, cs0=cs0)

hm = np.ma.masked_where(mask == 0, h)
hours = np.arange(D[rels[0]]['lon'].shape[0])
frames = np.arange(0, min(args.ndays * 24 + 1, len(hours)), args.step)

# Sea surface height at the cove mouth, so particle motion can be read against
# tidal phase. It comes from the hourly_flux reduction, which is unfiltered --
# bulk_avg is Godin filtered and has no tide left in it by construction.
# Aligned to each release using the tracker's own ocean_time, not by assuming
# the two files start on the same timestamp.
import pandas as pd
hf_fn = (Ldir['LOo'] / 'extract' / 'wb1_t0_xn11abbur00' / 'tef2'
         / 'hourly_flux_2024.01.01_2025.12.31_wb1_pc1.nc')
SSH = {}
if hf_fn.is_file():
    hf = xr.open_dataset(hf_fn)
    k_lp = list(hf.sect.values).index('pc_lp')
    ssh_all = pd.Series(hf.ssh.values[:, k_lp], index=pd.to_datetime(hf.time.values))
    hf.close()
    for rel in rels:
        d = xr.open_dataset(trk / ('release_' + rel + '.nc'))
        t0 = pd.to_datetime(d.ot.values[0], unit='s')
        d.close()
        idx = ssh_all.index.get_indexer([t0], method='nearest')[0]
        SSH[rel] = ssh_all.values[idx:idx + len(hours)]
        print('%s ssh aligned at %s' % (rel, ssh_all.index[idx]))
else:
    print('no hourly_flux file -- ssh panel will be skipped')

# Release-depth bins, in FRACTIONAL depth (cs) rather than metres. The cove
# shoals from 26 m at the mouth to 8 m at the head, so a fixed depth in metres
# would be "on the bed" inland and "mid column" at the mouth, mixing two
# different things. cs is comparable everywhere, and the release was seeded at
# sigma cell centres so the bins are evenly populated by construction.
DEPTH_BINS = [(-1.01, -2 / 3, 'bottom third'),
              (-2 / 3, -1 / 3, 'middle third'),
              (-1 / 3, 0.01, 'top third')]
# sequential ramp, light = surface to dark = deep, since depth is ordered.
# Warm hues so they stay legible over the blue bathymetry.
DEPTH_COLOR = ['#7F2704', '#E6550D', '#FDBE85']


def make_movie(sn):
    """One movie for the cohort that started in segment sn."""
    # Saratoga particles are tracked against their own segment; cove cohorts
    # against the whole cove, which is the volume the retention question is about
    ref = 'home' if sn == 'pc_lp_p' else 'ins'
    npart = int((D[rels[0]]['coh'] == sn).sum())
    print('%s (%s): %d particles' % (sn, LABEL[sn], npart))

    pad = 0.01
    if sn == 'pc_lp_p':
        base = seg_mask['pc_lp_p']
        aa = [lon[base].min() - pad, lon[base].max() + pad,
              lat[base].min() - pad, lat[base].max() + pad]
    else:
        aa = [lon[cove_mask].min() - pad, lon[cove_mask].max() + 0.09,
              lat[cove_mask].min() - 0.02, lat[cove_mask].max() + 0.03]

    plt.close('all')
    fig = plt.figure(figsize=(15, 9))
    nrow = 3 if SSH else 2
    hr = [3, 1, 1] if SSH else [3, 1]
    gs = fig.add_gridspec(nrow, 2, height_ratios=hr, hspace=0.22, wspace=0.08)
    axm = [fig.add_subplot(gs[0, k]) for k in range(2)]
    axc = fig.add_subplot(gs[1, :])
    axs = fig.add_subplot(gs[2, :], sharex=axc) if SSH else None

    dots, ttl = {}, {}
    for k, rel in enumerate(rels):
        ax = axm[k]
        ax.pcolormesh(lon, lat, hm, cmap='Blues', vmin=0, vmax=60,
                      shading='nearest', zorder=1, alpha=0.7)
        pfun.add_coast(ax, color='k', linewidth=0.6)
        ax.contour(lon, lat, cove_mask.astype(float), [0.5], colors='k',
                   linewidths=1.2, zorder=6)
        # outline the segment this cohort started in
        ax.contour(lon, lat, seg_mask[sn].astype(float), [0.5],
                   colors=[COLOR[sn]], linewidths=2.0, zorder=7)
        pfun.dar(ax)
        ax.axis(aa)
        ax.set_xlabel('Longitude')
        if k == 0:
            ax.set_ylabel('Latitude')
        else:
            ax.set_yticklabels([])
        dots[rel] = []
        for (lo, hi, lab), c in zip(DEPTH_BINS, DEPTH_COLOR):
            dots[rel].append(ax.plot([], [], '.', ms=3.5, color=c, alpha=0.85,
                                     zorder=10, label=lab)[0])
        ax.legend(loc='upper right', fontsize=9, markerscale=4, framealpha=0.9,
                  title='release depth')
        ttl[rel] = ax.set_title('%s\n' % rel, fontsize=12)

    for k, rel in enumerate(rels):
        m = D[rel]['coh'] == sn
        cs = D[rel]['cs0'][m]
        arr = (D[rel][ref][sn] if ref == 'home' else D[rel]['ins'])[:, m]
        for (lo, hi, lab), c in zip(DEPTH_BINS, DEPTH_COLOR):
            b = (cs >= lo) & (cs < hi)
            if b.sum() == 0:
                continue
            axc.plot(hours / 24, arr[:, b].mean(axis=1), '-' if k == 0 else '--',
                     lw=2, color=c,
                     label='%s %s' % (lab, rel[5:7]))
    vline = axc.axvline(0, color='k', lw=2)
    vline2 = axs.axvline(0, color='k', lw=2) if axs else None
    if axs:
        for k, rel in enumerate(rels):
            axs.plot(hours / 24, SSH[rel], '-' if k == 0 else '--', lw=1.6,
                     color='#3B0F70', label=rel)
        axs.axhline(0, color='0.5', lw=0.8)
        axs.set_ylabel('ssh at pc_lp [m]')
        axs.set_xlabel('days since release')
        axs.grid(color='lightgray', ls='--', alpha=0.5)
        axs.legend(fontsize=8, ncol=2, loc='upper right')
        axc.set_xlabel('')
    axc.set_xlim(0, args.ndays)
    axc.set_ylim(0, 1.02)
    axc.set_ylabel('fraction still in %s' % ('segment' if ref == 'home' else 'cove'))
    axc.grid(color='lightgray', ls='--', alpha=0.5)
    axc.legend(fontsize=9, ncol=2, loc='upper right')

    fig.suptitle('pcret: particles released in %s (%s), n=%d per release'
                 % (LABEL[sn], sn, npart), fontsize=14)

    def update(fi):
        for rel in rels:
            m = D[rel]['coh'] == sn
            cs = D[rel]['cs0'][m]
            xl, yl = D[rel]['lon'][fi, m], D[rel]['lat'][fi, m]
            for art, (lo, hi, lab) in zip(dots[rel], DEPTH_BINS):
                b = (cs >= lo) & (cs < hi)
                art.set_data(xl[b], yl[b])
            fr = (D[rel][ref][sn] if ref == 'home' else D[rel]['ins'])[fi, m].mean()
            ttl[rel].set_text('%s\nday %.1f   %.0f%% still in %s'
                              % (rel, fi / 24, 100 * fr,
                                 'segment' if ref == 'home' else 'cove'))
        vline.set_xdata([fi / 24, fi / 24])
        if vline2 is not None:
            vline2.set_xdata([fi / 24, fi / 24])
        return []

    anim = animation.FuncAnimation(fig, update, frames=frames,
                                   interval=1000 / args.fps, blit=False)
    fn_out = out_dir / ('20260805_pcret_anim_%s_bydepth%s.mp4' % (sn, args.tag))
    anim.save(fn_out, writer=animation.FFMpegWriter(fps=args.fps, bitrate=2400))
    print('  saved %s' % fn_out)


print('%d frames, %d hours apart\n' % (len(frames), args.step))
for sn in [s for s in args.cohorts.split(',') if s]:
    make_movie(sn)
