"""
T-S diagram of the BOTTOM cell at a Penn Cove mooring, colored by bottom DO.

One dot per lowpassed (daily, Godin-filtered) time step, so the cloud is the
two-year population of bottom water that actually sits at the station rather
than a seasonal average of it. The question it answers: which water masses
arriving at the cove mouth are the low-oxygen ones -- is the hypoxic bottom
water the coldest and saltiest (deep Saratoga Passage water pushed in), the
warmest and freshest (locally stagnant water that has sat and respired), or
neither, in which case DO is set by residence time and not by origin.

DEFAULT STATION lp_mid, the midpoint of the pc_lp section = the cove mouth
(see 20260811_pc4_points_map.py for where the pc4 points are). -sn cp_mid for
the inner cove or M5 for Saratoga Passage outside.

BOTTOM CELL means s_rho index 0, which is the same cell the bottom-DO series
in 20260811_pc_cp_mid_bottom_DO.py uses -- about 0.5 m above the bed here, not
a fixed depth. Its z and thickness move with the tide, so the printed depth is
a record mean. -k takes any other level; -k -1 is the surface.

SA AND CT, not raw salt and temp, so the density contours are TEOS-10 and the
axes match everything else in this project (obs and model are compared in
SA/CT throughout the paper work). Pressure comes from the cell's own z at
each step, so it follows zeta rather than assuming a fixed depth.

COLOR IS A SMOOTH GRADIENT (cmocean `deep_r`, dark = low DO), spanning 0 to the
record maximum so the whole seasonal swing in DO is legible as a gradient
rather than compressed into a few bands. The cost of a smooth ramp is that it
cannot resolve a THRESHOLD -- 1.9 and 2.1 mg/L are the same color to the eye --
so the hypoxic points are additionally ringed in crimson and enlarged, and
drawn last. The ring, not the color, is what says "hypoxic"; the fill still
carries the actual value, so a ringed point that is nearly black is a much
worse day than a ringed point at the edge of the threshold. -hyp moves the
threshold, -vmax pins the top of the ramp, -cmap swaps the gradient.

Density is sigma0 (potential density anomaly referenced to the surface), which
at 27 m is within ~0.01 kg/m3 of in-situ -- the contours are there to show
which direction along the cloud is a density change and which is a
compensating T-S swap, not for a stability calculation.

run 20260811_pc_moor_ts_bottom.py
run 20260811_pc_moor_ts_bottom.py -sn cp_mid
run 20260811_pc_moor_ts_bottom.py -sn lp_mid,cp_mid,M5      # one panel each
run 20260811_pc_moor_ts_bottom.py -track                    # join in time order
run 20260811_pc_moor_ts_bottom.py -k -1                     # surface instead
"""
import argparse
import sys

import gsw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cmocean import cm as cmo

from lo_tools import Lfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
p.add_argument('-job', default='pc4', type=str)
p.add_argument('-sn', default='lp_mid', type=str,
               help='comma-separated station names, one panel each')
p.add_argument('-k', default=0, type=int,
               help='s_rho index to plot; 0 = bottom cell, -1 = surface')
p.add_argument('-vmax', default=None, type=float,
               help='top of the DO color range, mg/L; default is the record '
                    'max rounded up, so the ramp uses its whole span')
p.add_argument('-cmap', default='deep_r', type=str,
               help='cmocean colormap name for DO; dark should be low DO')
p.add_argument('-hyp', default=2.0, type=float,
               help='hypoxic threshold, mg/L -- points below it get a ring')
p.add_argument('-track', action='store_true',
               help='join the points in time order with a faint line')
p.add_argument('-year', default='all', type=str,
               help="calendar year to plot, or 'all' for the whole record")
args = p.parse_args()

Ldir = Lfun.Lstart(gridname=args.gtagex.split('_')[0])
moor_dir = Ldir['LOo'] / 'extract' / args.gtagex / 'moor'
out_dir = Ldir['LOo'] / 'DM_outs' / '20260811_pc_moor_ts_bottom'
Lfun.make_dir(out_dir)

SN_LIST = [s.strip() for s in args.sn.split(',') if s.strip()]
O2_MMOL_TO_MGL = 32.0 / 1000.0
HYPOXIC_MGL, LOWDO_MGL = args.hyp, 5.0
HYP_RING = '#e8000d'      # crimson, absent from the deep_r ramp so it reads as
HYP_S, BASE_S = 48, 26    # a flag rather than as another DO value
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
LBL = {'cp_mid': 'cp_mid — Coupeville line, inner cove',
       'lp_mid': 'lp_mid — Long Point line, cove mouth',
       'M5': 'M5 — Saratoga Passage, outside'}

EXTRACT_MSG = (
    'Extract on apogee, where the run lives:\n'
    '  cd ~/LO/extract/moor\n'
    '  python multi_mooring_driver.py -gtx %s -ro 2 -0 2024.01.01 '
    '-1 2025.12.31 -lt lowpass -job %s -get_all True -Nproc 100 > %s.log &\n'
    % (args.gtagex, args.job, args.job))


def read_station(sn):
    """SA, CT, DO and z at level -k for one station, indexed in UTC.

    The extraction dates are read off the filename rather than passed in --
    the lowpass list trims to 2024.01.02_2025.12.30, not the dates handed to
    the driver, and hardcoding either one goes stale on the next run.
    """
    # multi_mooring_driver.py files under moor/<job>/; a bare extract_moor.py
    # call leaves them in moor/ -- accept either, newest match wins.
    cands = sorted(list((moor_dir / args.job).glob('%s_*.nc' % sn))
                   + list(moor_dir.glob('%s_*.nc' % sn)))
    if not cands:
        print('No mooring file for %s under %s' % (sn, moor_dir))
        print('\n' + EXTRACT_MSG)
        sys.exit(1)
    fn = cands[-1]
    print('reading ' + str(fn))

    ds = xr.open_dataset(fn, decode_times=True)
    if 'oxygen' not in ds.data_vars:
        print('*** no oxygen in %s -- the extraction needs -get_bio True '
              '(or -get_all True)' % fn.name)
        sys.exit(1)

    k = args.k
    lon, lat = float(ds.lon_rho), float(ds.lat_rho)
    z = ds.z_rho.values[:, k]                    # m, negative down, moves with zeta
    SP = ds.salt.values[:, k]                    # ROMS salt = practical salinity
    pt = ds.temp.values[:, k]                    # ROMS temp = potential temperature

    pres = gsw.p_from_z(z, lat)
    SA = gsw.SA_from_SP(SP, pres, lon, lat)
    CT = gsw.CT_from_pt(SA, pt)

    df = pd.DataFrame(
        {'SA': SA, 'CT': CT, 'sigma0': gsw.sigma0(SA, CT),
         'do_mgL': ds.oxygen.values[:, k] * O2_MMOL_TO_MGL,
         'z': z, 'hab_m': z - ds.z_w.values[:, 0]},
        index=pd.to_datetime(ds.ocean_time.values).tz_localize('UTC'))
    meta = dict(lon=lon, lat=lat, h=float(ds.h), fn=fn.name,
                N=ds.sizes['s_rho'])
    ds.close()
    return df, meta


DFS = {}
for sn in SN_LIST:
    df, meta = read_station(sn)
    if args.year != 'all':
        df = df[df.index.year == int(args.year)]
        if df.empty:
            print('*** no %s data for %s -- skipping' % (args.year, sn))
            continue
    DFS[sn] = (df, meta)
if not DFS:
    sys.exit('nothing to plot')

# Common axes across panels so the stations are directly comparable, padded so
# no marker sits on the frame.
allSA = np.concatenate([d.SA.values for d, _ in DFS.values()])
allCT = np.concatenate([d.CT.values for d, _ in DFS.values()])
pad_s = 0.05 * (np.nanmax(allSA) - np.nanmin(allSA))
pad_t = 0.05 * (np.nanmax(allCT) - np.nanmin(allCT))
SLIM = (np.nanmin(allSA) - pad_s, np.nanmax(allSA) + pad_s)
TLIM = (np.nanmin(allCT) - pad_t, np.nanmax(allCT) + pad_t)

sg, tg = np.meshgrid(np.linspace(*SLIM, 200), np.linspace(*TLIM, 200))
SIG = gsw.sigma0(sg, tg)

DOMAX = max(d.do_mgL.max() for d, _ in DFS.values())
VMAX = args.vmax if args.vmax is not None else float(np.ceil(DOMAX))
CMAP = getattr(cmo, args.cmap)

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
nax = len(DFS)
fig, axes = plt.subplots(1, nax, figsize=(6.2 * nax + 1.4, 6.4), squeeze=False,
                         sharex=True, sharey=True)
axes = axes[0]
kname = ('bottom' if args.k == 0 else
         'surface' if args.k in (-1, 29) else 's_rho %d' % args.k)

for ax, sn in zip(axes, DFS):
    df, meta = DFS[sn]

    cs = ax.contour(sg, tg, SIG, levels=np.arange(np.floor(SIG.min()),
                                                  np.ceil(SIG.max()) + 0.5, 0.5),
                    colors='0.6', linewidths=0.7, zorder=1)
    ax.clabel(cs, fmt='%.1f', fontsize=7, inline=True)

    if args.track:
        ax.plot(df.SA, df.CT, '-', color='0.75', lw=0.4, alpha=0.7, zorder=2)

    # Lowest DO drawn last so the hypoxic points are never buried under the
    # oxygenated cloud -- with 700+ overlapping dots, plot order is the whole
    # story. The hypoxic subset is then re-drawn on top with a ring, because a
    # smooth ramp cannot show a threshold on its own.
    hyp = (df.do_mgL < HYPOXIC_MGL).values
    o = np.argsort(-df.do_mgL.values)
    sc = ax.scatter(df.SA.values[o], df.CT.values[o], c=df.do_mgL.values[o],
                    cmap=CMAP, vmin=0, vmax=VMAX, s=BASE_S,
                    edgecolors='k', linewidths=0.25, zorder=3)
    if hyp.any():
        ax.scatter(df.SA.values[hyp], df.CT.values[hyp],
                   c=df.do_mgL.values[hyp], cmap=CMAP, vmin=0, vmax=VMAX,
                   s=HYP_S, edgecolors=HYP_RING, linewidths=1.0, zorder=4,
                   label='DO < %g mg L$^{-1}$  (n = %d)'
                         % (HYPOXIC_MGL, int(hyp.sum())))
        ax.legend(loc='lower right', fontsize=8, framealpha=0.9,
                  scatterpoints=1)

    nhyp = int(hyp.sum())
    nlow = int((df.do_mgL < LOWDO_MGL).sum())
    ax.set_title('%s\n%s cell, %s to %s'
                 % (LBL.get(sn, sn), kname,
                    df.index[0].strftime('%Y-%m-%d'),
                    df.index[-1].strftime('%Y-%m-%d')), fontsize=10)
    ax.text(0.025, 0.975,
            'n = %d lowpassed days\nh = %.1f m, cell z = %.1f m '
            '(%.1f m above bed)\nDO %.2f–%.2f mg/L\n'
            '< %g mg/L: %d d (%.0f%%)   < %g mg/L: %d d (%.0f%%)'
            % (len(df), meta['h'], df.z.mean(), df.hab_m.mean(),
               df.do_mgL.min(), df.do_mgL.max(),
               HYPOXIC_MGL, nhyp, 100 * nhyp / len(df),
               LOWDO_MGL, nlow, 100 * nlow / len(df)),
            transform=ax.transAxes, va='top', ha='left', fontsize=8,
            bbox=dict(fc='white', ec='0.8', alpha=0.85, pad=3.5))
    ax.set_xlabel('$S_A$ (g kg$^{-1}$)')
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)

axes[0].set_ylabel('$\\Theta$ (°C)')
axes[0].set_xlim(SLIM)
axes[0].set_ylim(TLIM)

cb = fig.colorbar(sc, ax=axes, fraction=0.035, pad=0.02,
                  extend='max' if DOMAX > VMAX else 'neither')
cb.set_label('%s DO (mg L$^{-1}$)' % kname)
cb.set_ticks(sorted({0.0, HYPOXIC_MGL, LOWDO_MGL, VMAX}
                    | set(np.arange(0, VMAX + 0.01, 2.0))))
# The hypoxic cut in the ring color, so the bar explains the rings; the low-DO
# line stays black because nothing on the plot is marked with it.
cb.ax.axhline(HYPOXIC_MGL, color=HYP_RING, lw=1.8)
cb.ax.axhline(LOWDO_MGL, color='k', lw=1.0)

fig.suptitle('Penn Cove %s-water T-S, %s (lowpassed)' % (kname, args.gtagex),
             fontsize=12)

out_fn = out_dir / ('ts_%s_%s_k%d_%s.png'
                    % (args.gtagex, '-'.join(DFS), args.k, args.year))
fig.savefig(out_fn, dpi=200, bbox_inches='tight', transparent=True)
print('\nsaved %s' % out_fn)

# ---------------------------------------------------------------------------
# what the picture is claiming, as numbers
# ---------------------------------------------------------------------------
for sn, (df, _) in DFS.items():
    lo = df[df.do_mgL < LOWDO_MGL]
    hi = df[df.do_mgL >= LOWDO_MGL]
    print('\n%s (%s cell)' % (sn, kname))
    print('  %-16s %8s %8s %8s' % ('', 'SA', 'CT', 'sigma0'))
    for lab, d in [('all', df), ('DO < %g' % LOWDO_MGL, lo),
                   ('DO >= %g' % LOWDO_MGL, hi)]:
        if len(d):
            print('  %-16s %8.3f %8.3f %8.3f  (n=%d)'
                  % (lab, d.SA.mean(), d.CT.mean(), d.sigma0.mean(), len(d)))
    for v in ['SA', 'CT', 'sigma0']:
        r = df[[v, 'do_mgL']].corr().iloc[0, 1]
        print('  corr(%s, DO) = %+.3f' % (v, r))
    print('  ^ a negative corr means the low-DO water is the SALTIER / WARMER / '
          'DENSER end.\n    Both SA and CT negative = warm salty water, the '
          'late-summer end of the seasonal\n    cycle; SA negative with CT '
          'positive = cold salty water, a dense intrusion.')

plt.close(fig)
