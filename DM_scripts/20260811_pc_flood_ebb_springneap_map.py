"""
The average flood and the average ebb OVER PENN COVE, all hours and by spring
and neap.

The map counterpart of the two pc_lp section figures, and it writes both of
them from one read of the transports:

  flood_ebb_map_*.png             1 x 2, all hours, the pair to
                                  20260811_pc_lp_flood_ebb_section.py
  flood_ebb_springneap_map_*.png  2 x 2, rows spring and neap, the pair to
                                  20260811_pc_lp_springneap_section.py

Same bins, same sign convention, same colour-scale logic as those, but the
whole cove in plan view instead of one section. Each figure is scaled by its
OWN panels, exactly as in the section pair: the spring flood is faster than the
all-hours flood, and one scale across both figures would flatten neap.

Every panel is the depth-averaged velocity,

    ubar = <QU2> / (dy * (h_u + <zeta>))                           [m s-1]

shaded by its ALONG-COVE (westward) component so that BLUE is INTO Penn Cove
and RED is out of it, exactly as in the section figures, with the full vector
drawn on top. The sections say how the flood is arranged over depth at the
mouth; these say where in the cove it goes.

BINNING IS IDENTICAL TO THE SECTION FIGURES, and deliberately comes from the
same file rather than being redefined here: flood/ebb is the sign of the pc_lp
net transport, and spring/neap is the upper/lower quartile of the 30-day
anomaly of the Godin-filtered |Qnet| (binning on the raw envelope would put
most "spring" hours in whichever season has the biggest tides). Those series
are on the tef2 clock -- ocean_avg, hourly means stamped at :30 -- and the maps
are on the ocean_his clock, on the hour, so the two are interpolated onto the
his times rather than assumed to line up. The pickle's own mouth transport is
computed as well, and the fraction of hours the two definitions of "flood"
agree on is printed: that number is the check that this figure is binned the
same way as the sections.

TRANSPORT VELOCITY, NOT MEAN VELOCITY, and note what that means in a tidal bin:
each panel is the bin's mean transport over the bin's mean area, with the area
built on the mean zeta OF THAT BIN. Flood and ebb sit at different mean sea
level (printed), so using one record-mean area for all four would put a few
percent of the difference between panels into the geometry instead of the flow.

WHY THE PANELS ARE NOT THE RESIDUAL. These are raw hourly composites, so what
dominates them is the barotropic tide -- ~10x the residual circulation in
20260811_pc_mean_circulation.py, which is what is left after flood and ebb
cancel. Flood and ebb are near mirror images by construction; the interest is
in where they are NOT, and in how that changes between spring and neap.

Runs on the mac. Needs the hourly transports from
20260807_pc_turning_reduce.py and the local extractions_avg.
run 20260811_pc_flood_ebb_springneap_map.py
run 20260811_pc_flood_ebb_springneap_map.py -qq 0.15
"""
import argparse
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import ListedColormap
from cmcrameri import cm as cmc

from lo_tools import Lfun, zfun
from lo_tools import plotting_functions as pfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
p.add_argument('-gctag', default='wb1_pc1', type=str)
p.add_argument('-sect', default='pc_lp', type=str,
               help='section whose net transport defines flood/ebb and the '
                    'spring/neap envelope')
p.add_argument('-0', '--ds0', default='2024.01.01', type=str)
p.add_argument('-1', '--ds1', default='2025.12.31', type=str)
p.add_argument('-f', '--fn', default='turning_his_wb1_t0_xn11abbur00_'
                                     '2024.01.01_2025.12.31.p',
               help='pickle from 20260807_pc_turning_reduce.py')
p.add_argument('-qq', default=0.25, type=float,
               help='quantile defining spring/neap; 0.25 = outer quartiles')
p.add_argument('--quiver-step', default=1, type=int, dest='quiver_step')
p.add_argument('--vmax', type=float, help='colour limit for all four panels [m/s]')
args = p.parse_args()

warnings.simplefilter('ignore')
Ldir = Lfun.Lstart(gridname=args.gctag.split('_')[0])
tef2 = Ldir['LOo'] / 'extract' / args.gtagex / 'tef2'
ex_fn = tef2 / ('extractions_avg_%s_%s' % (args.ds0, args.ds1)) / (args.sect + '.nc')
turn_fn = Ldir['LOo'] / 'DM_outs' / '20260807_pc_turning' / args.fn
out_dir = Ldir['LOo'] / 'DM_outs' / '20260811_pc_lp_flood_ebb'
Lfun.make_dir(out_dir)
for fn in [ex_fn, turn_fn]:
    if not fn.is_file():
        print('*** missing: %s' % fn)
        sys.exit(1)

CMAP = cmc.vik_r         # blue = into the cove, as in the sections

# ---------------------------------------------------------------------------
# style: the wb1 grid plots of 20260807_grid_bathy_ppt.py, minus the
# bathymetry. Slide-sized fonts, transparent background, flat land fill.
# TEXT_COLOR flips the whole figure between light-slide and dark-slide use.
# ---------------------------------------------------------------------------
TEXT_COLOR = 'k'                     # 'w' for dark slides
LAND_COLOR = '#e8e4dc'
mpl.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'text.color': TEXT_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'axes.edgecolor': TEXT_COLOR,
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR,
    'savefig.transparent': True,
    'figure.facecolor': 'none',
    'axes.facecolor': 'none',
})
SAVE_KW = dict(dpi=300, bbox_inches='tight', transparent=True, facecolor='none')
# the panel label, in the axes rather than above it, one step up from the
# axis labels so it reads as the panel's name
LBL_SIZE = mpl.rcParams['axes.labelsize'] + 2

# ---------------------------------------------------------------------------
# the bins, from the section -- identical definitions to the section figures
# ---------------------------------------------------------------------------
ds = xr.open_dataset(ex_fn)
ts = pd.to_datetime(ds.time.values)
dd, hh = ds.dd.values, ds.h.values
q_s = -ds.q.values                                  # + = INTO the cove
DZ_s = ds.DZ.values
ds.close()
u_s = q_s / (DZ_s * dd[None, None, :])              # section-normal velocity
qnet_s = pd.Series(q_s.sum(axis=(1, 2)), index=ts)

env = pd.Series(zfun.lowpass(np.abs(qnet_s.values), f='godin'), index=ts)
anom_s = env - env.rolling(30 * 24, center=True, min_periods=200).mean()
hi, lo = anom_s.quantile(1 - args.qq), anom_s.quantile(args.qq)
print('%s: %d hourly steps, envelope mean %.0f m3/s (%.0f to %.0f)'
      % (args.sect, len(ts), env.mean(), env.min(), env.max()))
print('spring = 30-day anomaly >= %+.0f m3/s, neap <= %+.0f' % (hi, lo))

# ---------------------------------------------------------------------------
# the maps, and the two clocks
# ---------------------------------------------------------------------------
D = pd.read_pickle(turn_fn)
INFO = D['info']
th = pd.DatetimeIndex(D['time_h'])
h, cove, UM, VM_, lonr, latr = (D['h'], D['cove'], D['UM'], D['VM'],
                                D['lon_rho'], D['lat_rho'])
QU2h = np.where(UM[None], D['QU2_h'], np.nan)
QV2h = np.where(VM_[None], D['QV2_h'], np.nan)
zeta_h = D['zeta_h']
h_u = 0.5 * (h[:, :-1] + h[:, 1:])
h_v = 0.5 * (h[:-1, :] + h[1:, :])


def onto_his(s):
    """A tef2 series (hourly means stamped at :30) on the his clock (on the
    hour). Interpolated, not nearest-matched: half an hour is 15 degrees of M2
    phase, which is enough to put a slack-water hour in the wrong bin."""
    return s.reindex(s.index.union(th)).interpolate('time').reindex(th)


qn = onto_his(qnet_s)
an = onto_his(anom_s)
print('clocks: tef2 %s .. %s, his %s .. %s; %d his hours outside the tef2 span'
      % (ts[0], ts[-1], th[0], th[-1], int(qn.isna().sum())))

flood, ebb = (qn > 0).values, (qn < 0).values
spring, neap = (an >= hi).values, (an <= lo).values
# The section panels are composited on the SECTION's own clock, with the same
# definition applied to its own qnet -- so no field is ever interpolated, only
# the bin definitions are carried across.
flood_s, ebb_s = (qnet_s > 0).values, (qnet_s < 0).values

# The same flood/ebb call made from the pickle's own mouth column. It is the
# same faces as the section, but ocean_his rather than ocean_avg, so this is
# the check that the two figure families really are binned alike.
im = int(list(D['iu_glob']).index(D['mouth_iu']))
q_mouth = -np.nansum(QU2h[:, :, im], axis=1)                    # + = INTO cove
agree = float(np.mean(np.sign(q_mouth) == np.sign(qn.values)))
print('flood/ebb from the pickle mouth column agrees with the section on '
      '%.1f%% of hours\n  (r = %+.3f; the disagreement is slack water, where '
      'the half-hour offset flips the sign)'
      % (100 * agree, np.corrcoef(q_mouth, np.nan_to_num(qn.values))[0, 1]))

# ---------------------------------------------------------------------------
# the four composites
# ---------------------------------------------------------------------------
AREA = np.where(cove, D['area'], np.nan)


def composite(m, label):
    """Bin-mean depth-averaged velocity over the cove for the hours in m."""
    if m.sum() == 0:
        print('*** no hours in %s' % label)
        sys.exit(1)
    # the bin's own mean sea level goes into the bin's face areas
    zb = float(np.nanmean(zeta_h[m]))
    ub = np.nanmean(QU2h[m], axis=0) / (D['DYU'] * (h_u + zb))
    vb = np.nanmean(QV2h[m], axis=0) / (D['DXV'] * (h_v + zb))
    u_rho = np.full(h.shape, np.nan)
    v_rho = np.full(h.shape, np.nan)
    # a dry face carries no flow, so it enters as zero rather than as a NaN
    # that would eat the whole cell
    u_rho[:, 1:-1] = 0.5 * (np.nan_to_num(ub[:, :-1]) + np.nan_to_num(ub[:, 1:]))
    v_rho[1:-1, :] = 0.5 * (np.nan_to_num(vb[:-1, :]) + np.nan_to_num(vb[1:, :]))
    u_rho[~cove], v_rho[~cove] = np.nan, np.nan
    u_in = -u_rho                           # + = westward = INTO the cove
    f_in = 100 * np.nansum(AREA * (u_in > 0)) / np.nansum(AREA)
    print('%-18s %8d %12.0f %10.3f %10.3f %8.0f%%'
          % (label, m.sum(), qn[m].mean(), zb,
             np.nanmax(np.hypot(u_rho, v_rho)), f_in))
    return dict(u_in=u_in, u=u_rho, v=v_rho, m=m, zb=zb)


def composite_sect(m, label):
    """Bin-mean section-normal velocity, and the mean cell thicknesses to draw
    it on. Same construction as the pc_lp section figures."""
    uu, dzm = np.nanmean(u_s[m], axis=0), DZ_s[m].mean(axis=0)
    a = dzm * dd[None, :]
    print('%-18s %8d %12.0f %10s %10.3f %8.0f%%'
          % (label, m.sum(), qnet_s[m].mean(), '-', np.nanmax(np.abs(uu)),
             100 * (a * (uu > 0)).sum() / a.sum()))
    return dict(u=uu, dz=dzm, m=m)


print('\n%-18s %8s %12s %10s %10s %9s'
      % ('bin', 'n [h]', 'mean Qnet', 'mean zeta', 'max |u|', 'area in'))
ALL = {k: composite(mk, k) for k, mk in [('flood', flood), ('ebb', ebb)]}
SEC = {k: composite_sect(mk, k + ' (section)')
       for k, mk in [('flood', flood_s), ('ebb', ebb_s)]}
SN = {(tide, k): composite(mt & mk, '%s %s' % (tide, k))
      for tide, mt in [('spring', spring), ('neap', neap)]
      for k, mk in [('flood', flood), ('ebb', ebb)]}

# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
DAR = 1 / np.cos(np.deg2rad(float(np.mean(latr))))
# cell CORNERS and shading='flat', as in the grid plots: the mesh then covers
# exactly the cells it describes, with no half-cell margin to patch up
plonr, platr = pfun.get_plon_plat(lonr, latr)
aa = [plonr.min(), plonr.max(), platr.min(), platr.max()]
land = ~D['mask_rho']
SL = pd.read_pickle(Ldir['LOo'] / 'section_lines' / (args.sect + '.p'))
qs = max(1, args.quiver_step)
RED = '#e04256'                      # the section, on the map and as its frame
# The map's rendered shape, height / width, from dar and the window. The
# section is given the same box shape so a row is two panels of one size --
# left to itself the section has no aspect constraint, fills the whole row and
# leaves the map floating in a short, wide cell.
BOX_AR = (aa[3] - aa[2]) * DAR / (aa[1] - aa[0])

# section geometry, shared by both section panels
xe = np.concatenate([[0], np.cumsum(dd)]) / 1000.               # km
xc = 0.5 * (xe[:-1] + xe[1:])
zg = np.linspace(-hh.max(), 0, 120)


def z_edges(dzm):
    """Cell interfaces in metres below the surface, built up from the bed."""
    zw = np.zeros((dzm.shape[0] + 1, dzm.shape[1]))
    zw[1:, :] = np.cumsum(dzm, axis=0)
    return -(zw[-1:, :] - zw)


def on_common_grid(uu, dzm):
    """Each face's profile interpolated onto one depth grid, masked below its
    own bed -- only so the u = 0 line can be contoured across faces. The
    colours are drawn from the untouched sigma cells, not from this."""
    G = np.full((len(zg), dzm.shape[1]), np.nan)
    zw = z_edges(dzm)
    for j in range(dzm.shape[1]):
        zc = 0.5 * (zw[:-1, j] + zw[1:, j])
        ok = zg >= zw[0, j]
        G[ok, j] = np.interp(zg[ok], zc, uu[:, j])
    return np.ma.masked_invalid(G)


def draw_sect(ax, C, vm):
    """One section panel: colours on the real cell thicknesses, the u = 0
    contour, and the bed in the same fill as the land on the map."""
    uu, dzm = C['u'], C['dz']
    zw = z_edges(dzm)
    # one quadmesh PER FACE -- a single mesh spanning all faces interpolates
    # the sigma levels across columns of different depth and smears the water
    # column past the stepped bathymetry
    for j in range(len(dd)):
        m = ax.pcolormesh(xe[j:j + 2], zw[:, j], uu[:, j:j + 1], cmap=CMAP,
                          vmin=-vm, vmax=vm, shading='flat', rasterized=True)
    ax.contour(xc, zg, on_common_grid(uu, dzm), levels=[0], colors=TEXT_COLOR,
               linewidths=1.4)
    # the bed, drawn at the bottom of the cells actually plotted so no white
    # seam opens up between the deepest cell and the fill
    hb = dzm.sum(axis=0)
    xs, zs = np.repeat(xe, 2)[1:-1], np.repeat(-hb, 2)
    ax.fill_between(xs, zs, -hb.max() - 3, color=LAND_COLOR, lw=0, zorder=5)
    ax.plot(xs, zs, color=TEXT_COLOR, lw=1.0, zorder=6)
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(-hh.max() - 3, 0)
    ax.set_box_aspect(BOX_AR)          # same shape as the map beside it
    ax.tick_params(length=6)
    # framed in the same red as the section line on the map, so the section and
    # its location read as one object
    for sp in ax.spines.values():
        sp.set_color(RED)
        sp.set_linewidth(3)
    ax.tick_params(color=RED)
    return m


def draw(ax, C, vm, qscale, lbl):
    ax.pcolormesh(plonr, platr, np.ma.masked_where(~land, np.ones(land.shape)),
                  cmap=ListedColormap([LAND_COLOR]), shading='flat', zorder=0,
                  rasterized=True)
    ax.pcolormesh(plonr, platr, C['u_in'], cmap=CMAP, vmin=-vm, vmax=vm,
                  shading='flat', zorder=1, rasterized=True)
    Q = ax.quiver(lonr[::qs, ::qs], latr[::qs, ::qs],
                  C['u'][::qs, ::qs], C['v'][::qs, ::qs], scale=qscale,
                  scale_units='width', units='width', width=0.0030,
                  color=TEXT_COLOR, zorder=6)
    # the section, drawn but not named: it is one line in the same place on
    # every panel and carries no direction the reader has to keep track of
    ax.plot(SL.x.astype(float), SL.y.astype(float), '-', color=RED,
            lw=3.5, zorder=9, solid_capstyle='butt')
    pfun.add_coast(ax, color=TEXT_COLOR, linewidth=0.8)
    # set_xticks re-autoscales, and a rounded tick outside the grid then drags
    # the view past the domain edge -- so pin the limits afterwards. Three
    # decimals, not the one the wb1-wide plots use: the cove spans 0.08 deg.
    ax.set_xticks(np.linspace(aa[0], aa[1], 4).round(3))
    ax.set_yticks(np.linspace(aa[2], aa[3], 5).round(3))
    ax.axis(aa)
    ax.set_autoscale_on(False)
    pfun.dar(ax)
    ax.tick_params(length=6, labelrotation=0)
    # the panel label goes IN the axes, on the land at the top left, where
    # nothing is ever plotted -- the counts and transports are in stdout
    ax.text(0.015, 0.97, lbl, transform=ax.transAxes, ha='left', va='top',
            fontsize=LBL_SIZE, color=TEXT_COLOR, zorder=10)
    return Q


def color_limit(fields):
    """Symmetric colour limit from THIS figure's panels only, rounded up to a
    tidy 0.02 -- the all-hours flood is slower than the spring flood, and one
    scale across both figures would flatten neap."""
    if args.vmax:
        return args.vmax
    return np.ceil(max(np.nanmax(np.abs(a)) for a in fields) * 50) / 50.


def quiver_scale(cells):
    """Arrow scale from the map panels: a 95th-pctl arrow spans ~1.2 cells."""
    s95 = float(np.nanpercentile(
        np.concatenate([np.hypot(C['u'], C['v'])[cove] for C in cells]), 95))
    return s95, s95 * lonr.shape[1] / (1.2 * qs)


def quiver_key(Q, s95):
    # on the axes that OWNS Q -- coordinates='axes' resolves against Q's own
    # axes, so keying it to any other panel puts the arrow outside that panel
    # and it is clipped away
    Q.axes.quiverkey(Q, 0.97, 0.06, round(s95, 2), '%.2f m s$^{-1}$' % s95,
                     labelpos='W', coordinates='axes',
                     fontproperties=dict(size=mpl.rcParams['xtick.labelsize']))


def cbar(fig, mappable, axs, what):
    cb = fig.colorbar(mappable, ax=axs, fraction=0.035, pad=0.02)
    # two lines, not one: rotated vertically at this font size the one-line
    # version is longer than the colourbar and gets clipped at both ends
    cb.set_label('%s $u$ [m s$^{-1}$]\nblue = into Penn Cove' % what,
                 color=TEXT_COLOR)
    cb.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelcolor=TEXT_COLOR)
    cb.outline.set_edgecolor(TEXT_COLOR)
    return cb


# --- all hours: rows are flood and ebb, columns are the map and the section ---
# ONE COLOUR SCALE AND ONE COLOURBAR for all four panels, set by whichever is
# larger. They are the same quantity in the same units with the same sign, so a
# shared scale is what makes the four directly comparable. The cost is worth
# stating rather than tuning away: the map is depth-averaged, so the in- and
# outflow largely cancel within each column of water, and it comes out paler
# than the section, which resolves them. How much paler is printed below.
vm = color_limit([C['u_in'] for C in ALL.values()]
                 + [C['u'] for C in SEC.values()])
s95, qsc = quiver_scale(list(ALL.values()))
print('\nall hours:   shared colour scale +/- %.2f m/s, quiver key %.3f m/s'
      % (vm, s95))
for lbl, cells, key in [('map', ALL.values(), 'u_in'),
                        ('section', SEC.values(), 'u')]:
    print('  %-7s |u| reaches %.0f%% of the scale'
          % (lbl, 100 * max(np.nanmax(np.abs(C[key])) for C in cells) / vm))
# Both panels carry the same box aspect now, so they want equal widths -- a
# width_ratios here would only open a gap. The figure is sized to be about
# twice as wide as it is tall, which is what two rows of that shape need.
fig, axes = plt.subplots(2, 2, figsize=(17, 9.5), layout='constrained')
for i, k in enumerate(['flood', 'ebb']):
    Q = draw(axes[i, 0], ALL[k], vm, qsc, k)
    draw_sect(axes[i, 1], SEC[k], vm)
    axes[i, 0].set_ylabel('Latitude [$^{\\circ}$N]')
    axes[i, 1].set_ylabel('Z [m]')
    if i == 0:                                  # only the bottom row is labelled
        for ax in axes[i]:
            ax.tick_params(labelbottom=False)
axes[1, 0].set_xlabel('Longitude [$^{\\circ}$E]')
axes[1, 1].set_xlabel('Distance along section [km]')
quiver_key(Q, s95)
# one colourbar for all four panels. The label has to cover both a
# depth-averaged along-cove velocity and a section-normal one; at this section
# the normal IS the along-cove direction, so "along-cove" is true of both.
cbar(fig, plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(-vm, vm)),
     axes, 'Along-cove')
fn_out = out_dir / ('flood_ebb_map_section_%s.png' % args.gtagex)
fig.savefig(fn_out, **SAVE_KW)
print('wrote ' + str(fn_out))

# --- spring / neap, 2 x 2 of maps ------------------------------------------
# This one stays four maps: the breakdown is already two-dimensional (tide
# strength x tidal phase), so there is no column left for a section.
vm = color_limit([C['u_in'] for C in SN.values()])
s95, qsc = quiver_scale(list(SN.values()))
print('spring/neap: colour +/- %.2f m/s, quiver key %.3f m/s' % (vm, s95))
fig, axes = plt.subplots(2, 2, figsize=(18, 10.4), sharex=True, sharey=True,
                         layout='constrained')
for i, tide in enumerate(['spring', 'neap']):
    for j, k in enumerate(['flood', 'ebb']):
        Q = draw(axes[i, j], SN[(tide, k)], vm, qsc, '%s %s' % (tide, k))
    axes[i, 0].set_ylabel('Latitude [$^{\\circ}$N]')
for j in range(2):
    axes[1, j].set_xlabel('Longitude [$^{\\circ}$E]')
quiver_key(Q, s95)
cbar(fig, plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(-vm, vm)),
     axes, 'Depth-averaged along-cove')
fn_out = out_dir / ('flood_ebb_springneap_map_%s.png' % args.gtagex)
fig.savefig(fn_out, **SAVE_KW)
print('wrote ' + str(fn_out))
