"""
Where the six points actually are.

Every figure in this set is a time series at named faces of two tef2 sections,
and a reader has no way to check that "north" and "south" mean what they
sound like, or how far apart they are, or how deep. This is that check: the
Penn Cove bathymetry, both section lines drawn face by face, and the six
points marked, labelled and coloured exactly as they are in the series plots.

THE POINTS ARE READ, NOT RE-DERIVED. 20260806_pc_sections_series.py writes
point_selection.csv when it runs; this script reads it. Recomputing the
selection rule here would let the map and the time series drift apart the
first time the rule changed, which is the one failure a location map must not
have. If the CSV is missing, run the series script first.

Left panel is Penn Cove with the bathymetry and the points. Right panel is
the Whidbey Basin context with the zoom box, because the cove on its own
gives no sense of where the exchange water is coming from.

The per-face mean transport (qbar, from structure_*.nc) is drawn along each
section as a filled profile, so the sign change the north/south pair was
chosen to straddle is visible on the map rather than asserted in a caption.

run 20260807_pc_points_map.py
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

p = argparse.ArgumentParser()
p.add_argument('--gtx', default='wb1_t0_xn11abbur00')
p.add_argument('--gctag', default='wb1_pc1')
p.add_argument('--ds0', default='2024.01.01')
p.add_argument('--ds1', default='2025.12.31')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname=args.gctag.split('_')[0])
tef2 = Ldir['LOo'] / 'extract' / args.gtx / 'tef2'
ser_dir = Ldir['LOo'] / 'DM_outs' / '20260806_pc_sections_series'
out_dir = Ldir['LOo'] / 'DM_outs' / '20260807_pc_points_map'
Lfun.make_dir(out_dir)

PCOLOR = {'north': '#0072B2', 'center': '#009E73', 'south': '#D55E00'}
SLABEL = {'pc_lp': 'pc_lp  (mouth)', 'pc_cp': 'pc_cp  (Coupeville)',
          'pc_lj': 'pc_lj  (mid-cove)'}

fn_sel = ser_dir / 'point_selection.csv'
if not fn_sel.is_file():
    raise SystemExit('%s not found -- run 20260806_pc_sections_series.py first'
                     % fn_sel)
SEL = pd.read_csv(fn_sel)
SECTS = list(dict.fromkeys(SEL.sect))
print('read %d points from %s' % (len(SEL), fn_sel.name))
print(SEL.round(4).to_string(index=False))

# ------------------------------------------------------------------ grid ---
dsg = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat = dsg.lon_rho.values, dsg.lat_rho.values
lon_u, lat_u = dsg.lon_u.values, dsg.lat_u.values
lon_v, lat_v = dsg.lon_v.values, dsg.lat_v.values
mask, hh = dsg.mask_rho.values, dsg.h.values
dsg.close()

sect_df = pd.read_pickle(Ldir['LOo'] / 'extract' / 'tef2'
                         / ('sect_df_%s.p' % args.gctag))
dstr = xr.open_dataset(tef2 / ('structure_%s_%s_%s.nc'
                               % (args.ds0, args.ds1, args.gctag)))

hw = np.where(mask == 1, hh, np.nan)          # depth over water only


def draw_base(ax, aa, vmax=None):
    # Penn Cove bottoms out near 27 m; Saratoga Passage immediately east is
    # over 70. Scaling the zoom to the deepest cell in view would paint the
    # whole cove a single pale blue, so the zoom is clipped to the cove's own
    # range and the channel is allowed to saturate.
    if vmax is None:
        vmax = np.nanmax(hw[(lon > aa[0]) & (lon < aa[1])
                            & (lat > aa[2]) & (lat < aa[3])])
    pc = ax.pcolormesh(lon, lat, np.ma.masked_invalid(hw), cmap='Blues',
                       vmin=0, vmax=vmax, shading='nearest', zorder=1)
    ax.pcolormesh(lon, lat, np.ma.masked_invalid(np.where(mask == 1, np.nan, 1.0)),
                  cmap='Greys', vmin=0, vmax=3, shading='nearest', zorder=2)
    pfun.add_coast(ax, color='k', linewidth=0.6)
    pfun.dar(ax)
    ax.axis(aa)
    ax.set_xlabel('Longitude')
    return pc


def sect_xy(sn):
    d = sect_df[sect_df.sn == sn]
    du, dv = d[d.uv == 'u'], d[d.uv == 'v']
    xs = np.concatenate([lon_u[du.j.values, du.i.values],
                         lon_v[dv.j.values, dv.i.values]])
    ys = np.concatenate([lat_u[du.j.values, du.i.values],
                         lat_v[dv.j.values, dv.i.values]])
    return xs, ys


# --------------------------------------------------------------- extents ---
PLON = SEL.lon.values
PLAT = SEL.lat.values
pad = 0.030
aa_zoom = [PLON.min() - pad * 2.2, PLON.max() + pad * 2.2,
           PLAT.min() - pad, PLAT.max() + pad]
aa_ctx = [lon.min(), lon.max(), lat.min(), lat.max()]

plt.close('all')
fig, axs = plt.subplots(1, 2, figsize=(17, 9.5), layout='constrained',
                        gridspec_kw=dict(width_ratios=[1.35, 1]))

# =========================================================== left: zoom ===
ax = axs[0]
VMAX_ZOOM = 32.0
pc = draw_base(ax, aa_zoom, vmax=VMAX_ZOOM)
cb = fig.colorbar(pc, ax=ax, shrink=0.72, pad=0.01, extend='max')
cb.set_label('depth (m), clipped at %.0f -- Saratoga Passage is deeper'
             % VMAX_ZOOM)

for sn in SECTS:
    xs, ys = sect_xy(sn)
    ax.plot(xs, ys, 's', color='magenta', ms=3.5, zorder=8)
    ax.text(xs.mean(), ys.max() + 0.0016, SLABEL.get(sn, sn), color='magenta',
            fontsize=10, fontweight='bold', ha='center', va='bottom', zorder=12,
            bbox=dict(fc='w', ec='none', alpha=0.7, pad=1))

    # per-face mean transport, drawn as a profile hanging off the section, so
    # the sign change the pair straddles is visible rather than asserted
    flon = dstr['%s_lon' % sn].values
    flat = dstr['%s_lat' % sn].values
    qbar = dstr['%s_qbar' % sn].values.sum(axis=0)
    scale = 0.016 / np.abs(qbar).max()
    ax.plot(flon + qbar * scale, flat, '-', color='0.35', lw=1.2, zorder=9)
    ax.fill_betweenx(flat, flon, flon + qbar * scale, where=qbar < 0,
                     color='#0072B2', alpha=0.35, lw=0, zorder=7)
    ax.fill_betweenx(flat, flon, flon + qbar * scale, where=qbar > 0,
                     color='#D55E00', alpha=0.35, lw=0, zorder=7)

# pc_cp's three faces are ~200 m apart, so labels placed at the point sit on
# top of each other. Fan them vertically and draw a leader line back.
DY = {'north': +0.011, 'center': 0.0, 'south': -0.011}
DX = {'pc_lp': +0.020, 'pc_cp': -0.020}
for _, r in SEL.iterrows():
    ax.plot(r.lon, r.lat, 'o', ms=13, mfc=PCOLOR[r.point], mec='k', mew=1.4,
            zorder=15)
    tx, ty = r.lon + DX.get(r.sect, 0.02), r.lat + DY[r.point]
    ax.annotate('%s  p=%d\nh %.1f m' % (r.point, r.face, r.h),
                xy=(r.lon, r.lat), xytext=(tx, ty),
                fontsize=8.5, ha='center', va='center', zorder=16, color='k',
                bbox=dict(fc='w', ec=PCOLOR[r.point], alpha=0.9, pad=2),
                arrowprops=dict(arrowstyle='-', color=PCOLOR[r.point], lw=1.0,
                                shrinkA=0, shrinkB=6))

ax.set_ylabel('Latitude')
ax.set_title('Penn Cove -- the six points\n'
             'shaded profile on each section is the mean per-face transport: '
             'blue INTO the cove, orange OUT', fontsize=11)
hand = [Line2D([], [], marker='o', ls='', mfc=PCOLOR[k], mec='k', ms=10,
               label=k) for k in ['north', 'center', 'south']]
hand.append(Line2D([], [], marker='s', ls='', color='magenta', ms=6,
                   label='section faces'))
ax.legend(handles=hand, loc='lower left', fontsize=9, framealpha=0.9)

# ======================================================== right: context ===
ax = axs[1]
draw_base(ax, aa_ctx)
for sn in SECTS:
    xs, ys = sect_xy(sn)
    ax.plot(xs, ys, '-', color='magenta', lw=2.5, zorder=8)
ax.plot(PLON, PLAT, 'o', ms=4, mfc='k', mec='k', zorder=9)
ax.plot([aa_zoom[0], aa_zoom[1], aa_zoom[1], aa_zoom[0], aa_zoom[0]],
        [aa_zoom[2], aa_zoom[2], aa_zoom[3], aa_zoom[3], aa_zoom[2]],
        '-', color='r', lw=2, zorder=20)
ax.set_title('wb1 grid -- Whidbey Basin context\nred box = left panel',
             fontsize=11)

fig.suptitle('%s -- location of the points used in the Penn Cove time series'
             % args.gtx, fontsize=13, y=1.01)
fn = out_dir / 'pc_points_map.png'
fig.savefig(fn, dpi=200, bbox_inches='tight')
dstr.close()
print('\nsaved %s' % fn)
