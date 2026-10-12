"""
Proposed seeding map for pcsrc, the Penn Cove source-water particle experiment.

Regions and densities come from pcsrc_regions.py (lean A, 2026-10-12).

Seeded domain = everything inside wb.p, i.e. Whidbey Basin north of to_mb
(wb.p's southern edge IS to_mb). It is split by the existing hand-drawn
lines into four origin regions, in priority order:

  cove   tef2 wb1_pc1 segments pc_cp_m + pc_cp_p + pc_lp_m (landward of pc_lp)
  spu    upper Saratoga = sp_head polygon minus the cove (pc_lp, skagit_sp, sp_mid)
  skb    Skagit Bay = wb_north minus sp_head (closed by deception_pass, the
         Port Susan flats and the Swinomish cut on wb.p's north edge)
  south  the rest of wb.p: lower Saratoga + Port Susan + Possession, to to_mb

Each region gets its own vertical spacing DZ and horizontal stride, so the
density falls off away from the cove. Particle counts follow the
ic_from_tef2_segs rule (n = max(1, round(h/DZ)) levels per kept cell), and each
particle stands for h*A*stride^2/n of water.

Gates for post-processing are drawn as lines: pc_cp, pc_lj, pc_lp, sp_mid,
skagit_sp, deception_pass, to_mb.

Mac only: needs grid.nc, section_lines and the wb1_pc1 seg_info_dict, no
ROMS output (t0 bathymetry matches grid.nc).

python 20261012_pcsrc_seed_map.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

import pcsrc_regions as pr

Ldir = Lfun.Lstart(gridname='wb1')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261012_pcsrc_seed_map'
Lfun.make_dir(out_dir)
sl_dir = Ldir['LOo'] / 'section_lines'

# region name, label, colour, DZ (m), horizontal stride (cells) -- densities from pcsrc_regions
COLS = {'cove': '#0b0b0b', 'spu': '#2a78d6', 'skb': '#eb6834', 'south': '#1baf7a'}
REG = [(rn, pr.RLABELS[rn], COLS[rn], pr.SEED[rn]['DZ'], pr.SEED[rn]['stride'])
       for rn in pr.RNAMES]
GATES = ['pc_cp', 'pc_lj', 'pc_lp', 'sp_mid', 'skagit_sp', 'deception_pass', 'to_mb']
PAD_CELLS = 10

g = xr.open_dataset(Ldir['data'] / 'grids' / 'wb1' / 'grid.nc')
lon = g.lon_rho.values; lat = g.lat_rho.values
h = g.h.values; wet = g.mask_rho.values == 1
A = 1 / (g.pm.values * g.pn.values)
plon, plat = pfun.get_plon_plat(lon, lat)
NR, NC = lon.shape


def sect(name):
    return pd.read_pickle(sl_dir / (name + '.p'))


R = pr.regions(Ldir, lon, lat, wet)
in_wb = R['in_wb']
masks = {rn: R[rn] for rn in pr.RNAMES}
leftover = in_wb & (R['REGID'] < 0)

# particle counts and weights
rows = []
REGID = np.full((NR, NC), np.nan)
SEED = np.zeros((NR, NC), dtype=bool)
for k, (key, label, col, DZ, st) in enumerate(REG):
    m = masks[key]
    keep = pr.seed_mask(R, key)
    nlev = pr.n_levels(h, key)
    NP = int(nlev[keep].sum())
    V = (h * A)[m].sum()
    REGID[m] = k
    SEED |= keep
    rows.append(dict(region=key, label=label, cells=int(m.sum()),
                     cells_seeded=int(keep.sum()), h_med=np.median(h[m]),
                     V_km3=V / 1e9, DZ=DZ, stride=st, particles=NP,
                     m3_per_particle=V / NP, per_km3=NP / (V / 1e9)))
df = pd.DataFrame(rows).set_index('region')
pd.set_option('display.width', 200)
print(df.round(3).to_string())
print('total particles per release: %d' % df.particles.sum())
print('wet cells inside wb.p not assigned to any region: %d' % leftover.sum())
df.to_csv(out_dir / 'pcsrc_seed_regions.csv')

# ---------------------------------------------------------------- figure
plt.close('all')
fs = 10
plt.rc('font', size=fs)
fig = plt.figure(figsize=(14, 9.5))
ax1 = fig.add_axes([0.04, 0.06, 0.48, 0.9])
ax2 = fig.add_axes([0.55, 0.42, 0.43, 0.54])
ax3 = fig.add_axes([0.55, 0.06, 0.43, 0.28]); ax3.axis('off')

cmap = ListedColormap([r[2] for r in REG])
hh = np.where(wet, h, np.nan)


def draw(ax, aa, gates, label_gates=True, lw=1.6):
    show = np.where(in_wb, REGID, np.nan)
    ax.pcolormesh(plon, plat, show, cmap=cmap, vmin=-0.5, vmax=len(REG) - 0.5,
                  shading='flat')
    ax.contour(lon, lat, np.where(in_wb, hh, np.nan), levels=[10, 50, 100, 200],
               colors='#52514e', linewidths=0.4, alpha=0.6)
    # seeded columns
    sx, sy = lon[SEED & in_wb], lat[SEED & in_wb]
    inbox = (sx > aa[0]) & (sx < aa[1]) & (sy > aa[2]) & (sy < aa[3])
    ax.plot(sx[inbox], sy[inbox], '.', color='#0b0b0b', ms=0.6 if aa[1] - aa[0] > 0.3 else 1.2,
            alpha=0.35, mec='none')
    for gname in gates:
        d = sect(gname)
        x, y = d.x.values.astype(float), d.y.values.astype(float)
        ax.plot(x, y, '-', color='#e34948', lw=lw, solid_capstyle='round')
        if label_gates:
            if np.ptp(x) < 1e-3:   # N-S line: label above its north end
                xm, ym, ha, va = x.mean(), y.max(), 'center', 'bottom'
            else:
                xm, ym, ha, va = x.mean(), y.mean(), 'left', 'center'
            ax.text(xm, ym, ' ' + gname + ' ', color='#0b0b0b', fontsize=fs - 1,
                    ha=ha, va=va, clip_on=True,
                    bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.5))
    pfun.dar(ax)
    ax.axis(aa)
    ax.tick_params(labelsize=fs - 2)
    ax.grid(lw=0.3, alpha=0.4)


# full domain: rectangle around wb.p plus margin
dwb = sect('wb')
dxc = np.median(np.diff(lon[0])); dyc = np.median(np.diff(lat[:, 0]))
aa1 = [dwb.x.min() - PAD_CELLS * dxc, dwb.x.max() + PAD_CELLS * dxc,
       dwb.y.min() - PAD_CELLS * dyc, dwb.y.max() + PAD_CELLS * dyc]
draw(ax1, aa1, ['pc_lp', 'sp_mid', 'skagit_sp', 'deception_pass', 'to_mb'])
ax1.set_title('pcsrc seeded domain (wb.p, north of to_mb)', fontsize=fs + 1, loc='left')

# Penn Cove + upper Saratoga zoom
aa2 = [-122.76, -122.43, 48.17, 48.31]
draw(ax2, aa2, ['pc_cp', 'pc_lj', 'pc_lp', 'sp_mid', 'skagit_sp'], lw=2)
ax2.set_title('Penn Cove and upper Saratoga', fontsize=fs + 1, loc='left')
ax1.add_patch(plt.Rectangle((aa2[0], aa2[2]), aa2[1] - aa2[0], aa2[3] - aa2[2],
                            fill=False, ec='#0b0b0b', lw=0.8, ls='--'))

handles = [Patch(fc=r[2], label=r[1]) for r in REG]
handles += [Line2D([], [], color='#e34948', lw=2, label='gate (section line)'),
            Line2D([], [], color='#0b0b0b', marker='.', ls='none', ms=4,
                   label='seeded column'),
            Line2D([], [], color='#52514e', lw=0.6, label='10/50/100/200 m')]
ax1.legend(handles=handles, loc='lower left', fontsize=fs - 1, framealpha=0.9)

# table
cols = ['region', 'cells', 'V (km$^3$)', 'h med (m)', 'DZ (m)', 'stride', 'particles', 'per km$^3$']
cell_text = [[lab, f'{r.cells:,}', f'{r.V_km3:.2f}', f'{r.h_med:.0f}', f'{r.DZ}', f'{r.stride}',
              f'{r.particles:,}', f'{r.per_km3:,.0f}']
             for lab, r in zip(['Penn Cove', 'Upper Saratoga', 'Skagit Bay',
                                'Lower SP + Port Susan\n+ Possession'], df.itertuples())]
cell_text.append(['total', f'{df.cells.sum():,}', f'{df.V_km3.sum():.2f}', '', '', '',
                  f'{df.particles.sum():,}', ''])
tb = ax3.table(cellText=cell_text, colLabels=cols,
               loc='upper left', colWidths=[0.27, 0.1, 0.1, 0.11, 0.09, 0.08, 0.12, 0.13],
               cellLoc='right', colLoc='right', edges='horizontal')
tb.auto_set_font_size(False); tb.set_fontsize(fs - 1); tb.scale(1, 1.6)
tb[(4, 0)].set_height(tb[(4, 0)].get_height() * 1.6)
for k, r in enumerate(REG):
    c = tb[(k + 1, 0)]; c.set_text_props(ha='left', fontweight='bold')
    c.get_text().set_color(r[2])
    for jc in range(len(cols)):
        tb[(4, jc)].set_height(tb[(4, 0)].get_height())
tb[(0, 0)].set_text_props(ha='left'); tb[(5, 0)].set_text_props(ha='left')

fn = out_dir / 'pcsrc_seed_map.png'
fig.savefig(fn, dpi=200, transparent=True)
plt.close(fig)
print('saved %s' % fn)
