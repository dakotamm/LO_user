"""
Conceptual map: how the flood, ebb and residual cross-sections of Penn Cove are modified by
the tide (spring / neap, tropic / equatorial), stratification and wind stress, with the
model's own composites as the evidence.

LAYOUT
    top row       baseline: average flood | ebb | residual at SECT, all hours
    one row each  driver box -> 'what it changes' (numbers from the 2026-11-03..05 analyses)
                  -> the DIFFERENCE it makes to the flood, ebb and residual sections
                  (high minus low half, e.g. spring minus neap), so each arrow points at data
The composites are from 20261104_pc_condition_sections.py (hourly snapshots, transport-
weighted, flood / ebb from the section's own Q); stratification is compared over ALL hours
(record thirds of drho). Sections look into the cove (west), north at the right.

SOURCES of the numbers in the text: 20261103_pc_section_profiles (velocity profiles),
20261103_pc_section_strat (N2), 20261104_pc_flow_metrics_plot (driver regression),
20261104_pc_condition_sections (residual rms), 20261104_pc_esco_response (ESCO ranking).

-kind profiles: the thumbnails are cross-section-averaged velocity PROFILES instead (1 m depth-
below-surface bins; each cell's transport-weighted mean velocity split among the bins by overlap
of its mean geometry, weighted by its mean area), the two halves overlaid (first solid, second
dashed) rather than differenced; each column shares one x-range down the page; into the cove
(west) at the LEFT, as on the velocity profiles.

-kind salt: the thumbnails are cross-section-averaged SALINITY profiles (1 m depth-below-surface
bins, hourly binning from 20261105_pc_salt_binned.py, summed over the hours of each condition),
columns flood | ebb | all hours, the two halves overlaid (the wind row of both profile kinds also has CALM,
no stress either way, dotted); the 'what it changes' boxes are filled in
from the profiles themselves (surface, 15 m, and the surface-to-15 m difference, i.e. how stratified).

-kind saltsec: SALINITY cross-sections (cell composites from 20261104_pc_condition_sections.py,
area-weighted): the baseline row is the salinity itself, each driver row the difference between its
two halves; same text boxes as -kind salt.

-kind lateral (2026-11-10): ACROSS-CHANNEL profiles of the DEPTH-AVERAGED velocity, the lateral counterpart of
-kind profiles: per face, U_d = sum_k U dz / sum_k dz over the composite's mean interfaces (U = the transport-
weighted cell composite), plotted at the face centre; y = distance from the north shore with NORTH AT THE TOP;
x as -kind profiles (into the cove at the left, one x-range per column down the page); same line styles.
The text boxes are computed from the composites: north-half and south-half area-weighted depth-averaged
velocity (+ into the cove) and the lateral exchange 0.5 sum_p |(U_d,p - Ubar) A_p| for flood, ebb and residual.

OUTPUT LO_output/DM_outs/20261105_pc_conceptual_map/pc_conceptual_map[_profiles|_lateral|_salt|_saltsec]_[SECT].png

-month YYYY-MM: only the hours of that month (composites from 20261104_pc_condition_sections.py -month),
with the classes unchanged, so the map compares directly with the record-wide one: same color scales and
x-ranges as the record-wide map; a class with no hours that month is a blank panel; the 'what it
changes' boxes give that month's hours and mean transports (or salinities) instead of record-wide numbers.

run 20261105_pc_conceptual_map.py [-sect pc_cp|pc_lj|pc_lp] [-kind sections|profiles|lateral|salt|saltsec] [-month YYYY-MM]
"""
import argparse
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import cmocean

from lo_tools import Lfun

parser = argparse.ArgumentParser()
parser.add_argument('-sect', default='pc_lp', choices=['pc_cp', 'pc_lj', 'pc_lp'])   # section shown in the thumbnails
parser.add_argument('-kind', default='sections', choices=['sections', 'profiles', 'lateral', 'salt', 'saltsec'])   # cross-sections, velocity profiles (vertical / lateral), or salinity
parser.add_argument('-month', default='', type=str)   # YYYY-MM: only the hours of that month
args = parser.parse_args()
SECT, KIND, MONTH = args.sect, args.kind, args.month
MLAB = pd.Timestamp(MONTH + '-01').strftime('%B %Y') if MONTH else ''
NAME = {'pc_lp': 'mouth (pc_lp)', 'pc_lj': 'middle (pc_lj)', 'pc_cp': 'Coupeville (pc_cp)'}
PHASES = ['flood', 'ebb', 'residual']

ROWS = [   # (high half, low half, driver box, what-it-changes box)
    ('spring', 'neap',
     'SPRING vs NEAP\nsemidiurnal tide (M2 + S2), ~14.8 d\nthirds of the D2 envelope of the\nmouth transport, within each month',
     'How MUCH water the tide moves: flood / ebb x1.6\n(mouth flood +1652 vs +1037 m$^3$ s$^{-1}$); same pattern.\n'
     'Residual nearly unchanged (not tidally rectified).\n'
     'Neap: more stratified (peak N$^2$ ~25% higher) and\nmore two-layer (counterflow share up).'),
    ('tropic', 'equatorial',
     'TROPIC vs EQUATORIAL\ndiurnal tide (K1 + O1), ~13.7 d\nthirds within each month\n(season balanced)',
     'Same transport, different SHAPE: ebb more\nsurface-trapped (mouth surface -5.0 vs -3.0 cm s$^{-1}$);\n'
     'stronger residual exchange (surface out -1.4 vs -0.3);\n'
     'more lateral counterflow. Stratification unchanged\n(the diurnal tide is partly two-layer).'),
    ('high strat', 'low strat',
     'HIGH vs LOW STRATIFICATION\n$\\Delta\\sigma_0$ bottom - top, record thirds,\nall hours; high: mostly Nov-Mar, Jun;\nlow: mostly Aug-Oct; imported from\nSaratoga Passage',
     'LAYERING and STRENGTH: faster, more sheared surface\nlayer (the top driver of speed and shear); more\n'
     'layered counterflow; residual exchange x1.2-1.9\n'
     '(x1.4-2.3 in calm hours, so not just the wind).\n'
     'Local tide and wind explain only ~5% of $\\Delta\\sigma_0$.'),
    ('up-cove', 'down-cove',
     'UP-COVE vs DOWN-COVE STRESS\nalong-cove stress ROMS applied,\n> +0.01 vs < -0.01 Pa (Godin)\nup: winter 13% | down: summer 34%',
     'The DIRECTION of the layered exchange\n(same sign on ~90% of clear-stress days):\n'
     'down-cove -> surface out, return inflow at 8-14 m;\n'
     'up-cove -> surface in, outflow below ~4 m.\n'
     'Adds to the surface ebb or flood; tidal Q unchanged.'),
]
BASE = {   # what the baseline (all-hours) sections look like, per section
    'pc_lp': 'Flood: inflow over most of the section, strongest at mid-depth;\na deep south-side outflow remains.\n'
             'Ebb: outflow concentrated near the surface; the north side\nkeeps flowing in.\n'
             'Residual: in on the north, out on the south (lateral exchange);\nhour by hour the counterflow is mostly layered.',
    'pc_lj': 'Flood: inflow over most of the section, strongest near the surface;\na deep south-side outflow remains.\n'
             'Ebb: outflow concentrated near the surface; the north side\nkeeps flowing in at depth.\n'
             'Residual: in on the north, out on the south (lateral exchange);\nhour by hour the counterflow is mostly layered.',
    'pc_cp': 'Flood: inflow strongest in the top ~5 m across the middle.\n'
             'Ebb: surface outflow over a subsurface inflow lens at ~10-17 m\nin the middle of the section.\n'
             'Residual: a subsurface inflow core near 13 m, with outflow at\nthe surface at the sides and below; mostly layered.'}

Ldir = Lfun.Lstart(gridname='wb1')
CS_DIR = Ldir['LOo'] / 'DM_outs' / '20261104_pc_condition_sections'
RA = pickle.load(open(CS_DIR / 'condition_sections.p', 'rb'))          # whole record: sets the color scales and x-ranges
R = pickle.load(open(CS_DIR / ('condition_sections_%s.p' % MONTH), 'rb')) if MONTH else RA   # what is plotted
out_dir = Ldir['LOo'] / 'DM_outs' / '20261105_pc_conceptual_map'
out_dir.mkdir(parents=True, exist_ok=True)
dy = R['dy'][SECT]
xe = np.concatenate([[0], np.cumsum(dy)]) / 1000


def thumb(ax, U, zw, vm, cmap):
    """One small cross-section: each face a column between its mean interfaces, north at right."""
    for p in range(len(dy)):
        cs = ax.pcolormesh(xe[p:p + 2], zw[:, p], 100 * U[:, p:p + 1], cmap=cmap, vmin=-vm, vmax=vm, shading='flat')
    X = np.broadcast_to(0.5 * (xe[1:] + xe[:-1]), U.shape)
    ax.contour(X, 0.5 * (zw[1:] + zw[:-1]), U, levels=[0], colors='k', linewidths=0.7)
    ax.set_xlim(xe[-1], xe[0])
    ax.set_ylim(-28, 0)
    ax.set_xticks([])
    ax.set_yticks([-20, -10, 0])
    ax.tick_params(labelsize=6, length=2)
    return cs


EDGES = np.arange(0, 31, 1.0)             # depth-below-surface bins [m]
DMID = 0.5 * (EDGES[1:] + EDGES[:-1])
PCOL = {'flood': '#D55E00', 'ebb': '#0072B2', 'residual': 'k'}


def hours(c):
    """Hours of condition c in the plotted period (0 if none)."""
    d = R[(SECT, c, 'residual')]
    return 0 if d is None else int(round(d['frac'] * R['n_all']))


def lab(c, name=None):
    """Legend label for condition c; says so when it has no hours in the period."""
    return (name or c) + (' (no hours)' if hours(c) == 0 else '')


def blank(ax, msg='no hours in this class'):
    """An empty panel for a class with no hours in the period."""
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha='center', va='center', fontsize=8, color='0.4')
    ax.set_xticks([]), ax.set_yticks([])


def profile(c, ph, RR=None):
    """Cross-section-averaged velocity [m/s] in 1 m bins below the surface, from a composite: each cell's
    transport (U x mean area) split among the bins by overlap of its mean extent, divided by the binned area.
    RR: the composites to use (default the plotted period); all NaN if the class has no hours."""
    d = (R if RR is None else RR)[(SECT, c, ph)]
    if d is None:
        return np.full(len(DMID), np.nan)
    U, zw = d['U'], d['zw']
    top, bot = -zw[1:], -zw[:-1]                       # depth below z = 0 of each cell's top and bottom (k, p)
    Qb, Ab = np.zeros(len(DMID)), np.zeros(len(DMID))
    for b in range(len(DMID)):
        ov = np.clip(np.minimum(bot, EDGES[b + 1]) - np.maximum(top, EDGES[b]), 0, None) * dy[None, :]
        Qb[b], Ab[b] = (U * ov).sum(), ov.sum()
    return np.where(Ab >= dy.mean() * 1.0, Qb / np.maximum(Ab, 1e-12), np.nan)   # bins covered by at least one face


YC = 0.5 * (xe[1:] + xe[:-1])            # face centres, km from the north shore


def lateral(c, ph, RR=None):
    """Depth-averaged velocity [m/s] of each face, sum_k U dz / sum_k dz over the composite's mean interfaces;
    all NaN if the class has no hours."""
    d = (R if RR is None else RR)[(SECT, c, ph)]
    if d is None:
        return np.full(len(dy), np.nan)
    dz = np.diff(d['zw'], axis=0)
    return (d['U'] * dz).sum(axis=0) / dz.sum(axis=0)


def lthumb(ax, curves, xl):
    """One small across-channel panel: (velocity, color, linestyle, label) curves; north at the top, into the cove at the left."""
    for x, col, ls, lab in curves:
        ax.plot(100 * x, YC, color=col, ls=ls, lw=1.8, marker='o', ms=3, label=lab)
    ax.axvline(0, color='k', lw=0.6)
    ax.set_xlim(xl, -xl)                  # reversed: + (into the cove, west) at the left
    ax.set_ylim(xe[-1], 0)                # north shore at the top
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=6, length=2)
    ax.text(0.98, 0.98, 'N', transform=ax.transAxes, ha='right', va='top', fontsize=7, fontweight='bold')
    ax.text(0.98, 0.02, 'S', transform=ax.transAxes, ha='right', va='bottom', fontsize=7, fontweight='bold')
    ax.legend(fontsize=6, loc='lower left', handlelength=1.5)


NORTH = YC < 0.5 * xe[-1]                # faces in the north half of the section (the rest are the south half)


def lstats(c, ph):
    """North-half and south-half area-weighted depth-averaged velocity [cm/s] and the lateral exchange
    0.5 sum_p |(U_d,p - Ubar) A_p| [m3/s] of a composite; None if the class has no hours."""
    d = R[(SECT, c, ph)]
    if d is None:
        return None
    A = np.diff(d['zw'], axis=0).sum(axis=0) * dy             # (p,) mean area of each face column
    Ud = lateral(c, ph)
    Ub = (Ud * A).sum() / A.sum()
    half = lambda m: 100 * (Ud[m] * A[m]).sum() / A[m].sum()
    return half(NORTH), half(~NORTH), 0.5 * np.abs((Ud - Ub) * A).sum()


def ltext(hi, lo, calm=False):
    """'What it changes' box for -kind lateral, from the composites."""
    L = lambda n: n.replace(', calm', '')
    cs = [hi, lo] + (['calm'] if calm else [])
    rows = ['depth-averaged u [cm/s], + into the cove', '            north half     south half    lateral exch.']
    for ph in PHASES:
        for c in cs:
            st = lstats(c, ph)
            nm = '%-8s %-10s' % (ph if c == hi else '', L(c)[:10])
            rows.append(nm + ('   no hours' if st is None else '  %+6.1f        %+6.1f       %5.0f m3/s' % st))
    sh, sl = lstats(hi, 'residual'), lstats(lo, 'residual')
    if sh is not None and sl is not None:
        rows.append('residual lateral exchange %s / %s: x%.2f' % (L(hi), L(lo), sh[2] / sl[2]))
    return '\n'.join(rows)


def pthumb(ax, curves, xl):
    """One small profile panel: (velocity, color, linestyle, label) curves; into the cove at the left."""
    for x, col, ls, lab in curves:
        ax.plot(100 * x, DMID, color=col, ls=ls, lw=1.8, label=lab)
    ax.axvline(0, color='k', lw=0.6)
    ax.set_xlim(xl, -xl)                  # reversed: + (into the cove, west) at the left
    ax.set_ylim(28, 0)
    ax.grid(True, alpha=0.3)
    ax.set_yticks([0, 10, 20])
    ax.tick_params(labelsize=6, length=2)
    ax.legend(fontsize=6, loc='lower left', handlelength=1.5)


if KIND in ('salt', 'saltsec'):           # hourly binned salinity, and the condition / phase masks
    import xarray as xr
    from pc_tidal_cycles import cycle_masks
    SB = xr.open_dataset(Ldir['LOo'] / 'DM_outs' / '20261105_pc_salt_binned' / 'salt_binned.nc')
    SA_, SS_, SQ_ = SB['A_' + SECT].values, SB['sA_' + SECT].values, SB['Q_' + SECT].values
    SCM = cycle_masks(SB.time.values)
    SCM['all'] = np.ones(len(SQ_), bool)
    SPH = {'flood': SQ_ > 0, 'ebb': SQ_ < 0, 'all': np.ones(len(SQ_), bool)}
    MM = (pd.DatetimeIndex(SB.time.values).strftime('%Y-%m') == MONTH) if MONTH else np.ones(len(SQ_), bool)
    DB = SB.bin.values                    # bin centres, depth below the surface


def sprofile(c, ph, mm=None):
    """Cross-section-averaged salinity [g/kg] in the 1 m bins over the hours of condition c and phase ph
    (and the month mask mm, default the plotted period); bins covered on average by less than one face are
    dropped; all NaN if there are no hours."""
    m = SCM[c] & SPH[ph] & (MM if mm is None else mm)
    if not m.any():
        return np.full(len(DB), np.nan)
    A = SA_[m].sum(axis=0)
    return np.where(SA_[m].mean(axis=0) >= dy.mean(), SS_[m].sum(axis=0) / np.maximum(A, 1e-12), np.nan)


def snums(c):
    """Surface, 15 m and their difference (all hours of condition c)."""
    p = sprofile(c, 'all')
    return p[0], p[np.argmin(np.abs(DB - 14.5))]


def shours(c):
    return int((SCM[c] & MM).sum())


def stext(hi, lo):
    """'What it changes' box for salinity, from the profiles."""
    (s0h, s15h), (s0l, s15l) = snums(hi), snums(lo)
    L = lambda n: n.replace(', calm', '')
    if MONTH and (shours(hi) == 0 or shours(lo) == 0):
        return '\n'.join('%s: %s' % (L(c), ('%d h, surface %.2f, 15 m %.2f g/kg' % ((shours(c),) + snums(c))) if shours(c) else 'no hours in %s' % MLAB)
                         for c in (hi, lo))
    return (('%s: %d h, %s: %d h in %s\n' % (L(hi), shours(hi), L(lo), shours(lo), MLAB)) if MONTH else '') + ('%s:  surface %.2f, 15 m %.2f g/kg\n          -> top-to-15 m difference %.2f\n'
            '%s:  surface %.2f, 15 m %.2f g/kg\n          -> top-to-15 m difference %.2f\n'
            'difference (%s - %s): surface %+.2f, 15 m %+.2f g/kg'
            % (L(hi), s0h, s15h, s15h - s0h, L(lo), s0l, s15l, s15l - s0l, L(hi), L(lo), s0h - s0l, s15h - s15l))


def textbox(ax, text, fc, fs=8.5, mono=False):
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0.02, 0.04), 0.96, 0.92, boxstyle='round,pad=0.01,rounding_size=0.03',
                                transform=ax.transAxes, fc=fc, ec='0.4', lw=1))
    ax.text(0.06, 0.5, text, transform=ax.transAxes, fontsize=fs, va='center', ha='left', linespacing=1.35,
            family='monospace' if mono else None)


def arrow(fig, ax0, ax1):
    """Arrow from the right edge of ax0 to the left edge of ax1 (figure coordinates)."""
    b0, b1 = ax0.get_position(), ax1.get_position()
    y = 0.5 * (b0.y0 + b0.y1)
    fig.patches.append(matplotlib.patches.FancyArrowPatch((b0.x1, y), (b1.x0, y), transform=fig.transFigure,
                                                          arrowstyle='-|>', mutation_scale=14, color='0.3', lw=1.5))


# ------------------------------------------------------------------ figure
vb = np.nanpercentile(np.concatenate([np.abs(RA[(SECT, 'all', ph)]['U']).ravel() for ph in PHASES]), 99) * 100   # scales: whole record
vd = np.nanpercentile(np.concatenate([np.abs(RA[(SECT, hi, ph)]['U'] - RA[(SECT, lo, ph)]['U']).ravel()
                                      for hi, lo, *_ in ROWS for ph in PHASES]), 98) * 100
nr = len(ROWS) + 1
fig = plt.figure(figsize=(20, 3.0 * nr))
gs = fig.add_gridspec(nr, 6, width_ratios=[1.35, 0.12, 1.75, 1, 1, 1], height_ratios=[1.1] + [1] * len(ROWS),
                      left=0.02, right=0.93, top=0.95, bottom=0.02, wspace=0.08, hspace=0.35)
# baseline row
if KIND == 'sections':
    btext = 'BASELINE, all hours\nsection: %s, looking into the\ncove (west), north at right;\nred = into the cove' % NAME[SECT]
elif KIND == 'profiles':
    btext = 'BASELINE, all hours\nsection: %s, averaged across\nthe section in 1 m depth bins;\ninto the cove (west) at the left' % NAME[SECT]
elif KIND == 'lateral':
    btext = 'BASELINE, all hours\nsection: %s, depth-averaged\nvelocity of each face across\nthe channel, north at the top;\ninto the cove (west) at the left' % NAME[SECT]
elif KIND == 'salt':
    btext = 'BASELINE, all hours\nsection: %s, salinity averaged\nacross the section in 1 m\ndepth bins below the surface' % NAME[SECT]
else:
    btext = 'BASELINE, all hours\nsection: %s, salinity,\nlooking into the cove (west),\nnorth at right' % NAME[SECT]
if MONTH:
    btext = btext.replace('BASELINE, all hours', 'BASELINE, %s ONLY\n(all %d h of the month)' % (MLAB.upper(), hours('all')))
textbox(fig.add_subplot(gs[0, 0]), btext, '#EEEEEE', fs=9)
if KIND in ('salt', 'saltsec'):
    pf, pe, pa = (sprofile('all', ph) for ph in ['flood', 'ebb', 'all'])
    k15 = np.argmin(np.abs(DB - 14.5))
    textbox(fig.add_subplot(gs[0, 2]), ('%s: ' % MLAB if MONTH else '') + 'Salinity rises from %.2f g/kg at the surface to %.2f at 15 m\n(all hours; the deepest bins reach %.2f).\n'
            'Flood minus ebb: surface %+.2f, 15 m %+.2f g/kg.\n'
            'The cove has no river: this layering is imported\nfrom Saratoga Passage through the mouth.'
            % (pa[0], pa[k15], np.nanmax(pa), pf[0] - pe[0], pf[k15] - pe[k15]), '#EEEEEE')
elif KIND == 'lateral':
    word = lambda v: 'in' if v > 0 else 'out'
    bl = ['%s: north %+.1f, south %+.1f cm s$^{-1}$\n    (%s on the north, %s on the south); lateral exchange %.0f m$^3$ s$^{-1}$'
          % ((ph.upper(),) + lstats('all', ph)[:2] + (word(lstats('all', ph)[0]), word(lstats('all', ph)[1]), lstats('all', ph)[2])) for ph in PHASES]
    textbox(fig.add_subplot(gs[0, 2]), ('%s only (%d h).\n' % (MLAB, hours('all')) if MONTH else '')
            + 'Depth-averaged velocity, area-weighted over each half\nof the section; + = into the cove.\n' + '\n'.join(bl), '#EEEEEE')
elif MONTH:
    qf, qe = R[(SECT, 'all', 'flood')]['Q'], R[(SECT, 'all', 'ebb')]['Q']
    textbox(fig.add_subplot(gs[0, 2]), '%s only: %d h.\nMean section transport: flood %+.0f, ebb %+.0f m$^3$ s$^{-1}$.\n'
            'Same classes and the same color scales / x-ranges as the\nwhole-record map, so the two compare panel by panel.\n'
            'Blank panels: classes with no hours this month.' % (MLAB, hours('all'), qf, qe), '#EEEEEE')
else:
    textbox(fig.add_subplot(gs[0, 2]), BASE[SECT], '#EEEEEE')
CPH = ['flood', 'ebb', 'all'] if KIND in ('salt', 'saltsec') else PHASES   # column phases
RPH = {'flood': 'flood', 'ebb': 'ebb', 'all': 'residual', 'residual': 'residual'}   # composite key ('residual' = all hours)
if KIND == 'saltsec':                     # salinity range of the baseline, and of the differences
    SV = [RA[(SECT, 'all', RPH[ph])]['S'] for ph in CPH]                 # scales: whole record
    SR = (np.floor(np.nanmin(SV) * 2) / 2, np.ceil(np.nanmax(SV) * 2) / 2)
    SD = np.nanpercentile(np.concatenate([np.abs(RA[(SECT, hi, RPH[ph])]['S'] - RA[(SECT, lo, RPH[ph])]['S']).ravel()
                                          for hi, lo, *_ in ROWS for ph in CPH]), 98)


def ssec(ax, S, zw, cmap, vmin, vmax, levels):
    """A salinity cross-section thumbnail: each face a column between its mean interfaces, north at right."""
    for p in range(len(dy)):
        cs = ax.pcolormesh(xe[p:p + 2], zw[:, p], S[:, p:p + 1], cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
    X = np.broadcast_to(0.5 * (xe[1:] + xe[:-1]), S.shape)
    cc = ax.contour(X, 0.5 * (zw[1:] + zw[:-1]), S, levels=levels, colors='k', linewidths=0.6)
    if len(levels) > 1:
        ax.clabel(cc, cc.levels[::2], fontsize=5, fmt='%.1f')
    ax.set_xlim(xe[-1], xe[0])
    ax.set_ylim(-28, 0)
    ax.set_xticks([])
    ax.set_yticks([-20, -10, 0])
    ax.tick_params(labelsize=6, length=2)
    return cs
if KIND == 'salt':                        # one salinity range for every panel
    allp = np.concatenate([sprofile(c, ph, np.ones(len(MM), bool)) for c in ['all', 'calm'] + [h_ for r_ in ROWS for h_ in r_[:2]] for ph in CPH])   # whole record
    SX = (np.floor(np.nanmin(allp) * 2) / 2, np.ceil(np.nanmax(allp) * 2) / 2)


def sthumb(ax, curves):
    for x, col, ls, lab in curves:
        ax.plot(x, DB, color=col, ls=ls, lw=1.8, label=lab)
    ax.set_xlim(*SX)
    ax.set_ylim(28, 0)
    ax.grid(True, alpha=0.3)
    ax.set_yticks([0, 10, 20])
    ax.tick_params(labelsize=6, length=2)
    ax.legend(fontsize=6, loc='lower left', handlelength=1.5)
if KIND in ('profiles', 'lateral'):      # one x-range per column, shared down the page
    PF = profile if KIND == 'profiles' else lateral
    XL = {ph: 1.1 * np.nanmax([np.nanmax(np.abs(100 * PF(c, ph, RA))) for c in ['all', 'calm'] + [h_ for r_ in ROWS for h_ in r_[:2]]]) for ph in PHASES}   # whole record
for j, ph in enumerate(CPH):
    ax = fig.add_subplot(gs[0, 3 + j])
    if KIND == 'saltsec':
        d = R[(SECT, 'all', RPH[ph])]
        cb = ssec(ax, d['S'], d['zw'], cmocean.cm.haline, SR[0], SR[1], np.arange(SR[0], SR[1] + 0.5, 0.5))
    elif KIND == 'salt':
        sthumb(ax, [(sprofile('all', ph), PCOL.get(ph, 'k'), '-', 'all hours')])
        ax.set_xlabel('salinity [g kg$^{-1}$]', fontsize=7)
    elif KIND == 'lateral':
        lthumb(ax, [(lateral('all', ph), PCOL[ph], '-', 'all hours')], XL[ph])
        ax.set_xlabel(r'$\leftarrow$ into the cove   [cm s$^{-1}$]   out $\rightarrow$', fontsize=7)
        if j == 0:
            ax.set_ylabel('km from the north shore', fontsize=6)
    elif KIND == 'profiles':
        pthumb(ax, [(profile('all', ph), PCOL[ph], '-', 'all hours')], XL[ph])
        ax.set_xlabel(r'$\leftarrow$ into the cove   [cm s$^{-1}$]   out $\rightarrow$', fontsize=7)
    else:
        cb = thumb(ax, R[(SECT, 'all', ph)]['U'], R[(SECT, 'all', ph)]['zw'], vb, cmocean.cm.balance)
    ax.set_title('ALL HOURS' if ph == 'all' else ph.upper(), fontsize=11, fontweight='bold')
if KIND == 'sections':
    cax = fig.add_axes([0.94, 0.80, 0.008, 0.12])
    fig.colorbar(cb, cax=cax).set_label('velocity [cm s$^{-1}$]\n+ into the cove', fontsize=8)
elif KIND == 'saltsec':
    cax = fig.add_axes([0.94, 0.80, 0.008, 0.12])
    fig.colorbar(cb, cax=cax).set_label('salinity [g kg$^{-1}$]\n(contours every 0.5)', fontsize=8)
# driver rows
COL = ['#DCEBF7', '#FCE6D4', '#F3E3F0', '#DFF2E7', '#FFF4CC']
for i, (hi, lo, dtext, ctext) in enumerate(ROWS):
    r = i + 1
    a0 = fig.add_subplot(gs[r, 0])
    textbox(a0, dtext, COL[i], fs=9)
    a2 = fig.add_subplot(gs[r, 2])
    SK = KIND in ('salt', 'saltsec')
    if KIND == 'lateral':
        tx = ltext(hi, lo, calm=(hi == 'up-cove'))
    elif SK:
        tx = stext(hi, lo)
    elif MONTH:                           # this month's hours and mean transports, not the record-wide text
        tl = []
        for c in (hi, lo):
            if hours(c):
                tl.append('%s: %d h; mean Q flood %+.0f, ebb %+.0f m$^3$ s$^{-1}$' % (c, hours(c), R[(SECT, c, 'flood')]['Q'], R[(SECT, c, 'ebb')]['Q']))
            else:
                tl.append('%s: no hours in %s' % (c, MLAB))
        if hi == 'up-cove':
            tl.append('calm (no wind): %d h' % hours('calm'))
        tx = '\n'.join(tl)
    else:
        tx = ctext
    if SK and hi == 'up-cove':            # the calm (no wind) reference
        c0, c15 = snums('calm')
        tx += '\ncalm (no wind):  surface %.2f, 15 m %.2f g/kg' % (c0, c15)
    textbox(a2, tx, 'white', fs=7 if KIND == 'lateral' else (7.5 if SK else (8 if MONTH else 8.5)), mono=SK or KIND == 'lateral')
    for j, ph in enumerate(CPH):
        ax = fig.add_subplot(gs[r, 3 + j])
        if KIND in ('sections', 'saltsec') and (R[(SECT, hi, RPH[ph])] is None or R[(SECT, lo, RPH[ph])] is None):
            blank(ax, 'no %s hours in %s' % (' or '.join(c for c in (hi, lo) if R[(SECT, c, RPH[ph])] is None), MLAB))
            ax.set_title('%s minus %s' % (hi, lo), fontsize=7)
        elif KIND == 'saltsec':
            dh, dl = R[(SECT, hi, RPH[ph])], R[(SECT, lo, RPH[ph])]
            cdd = ssec(ax, dh['S'] - dl['S'], dh['zw'], cmocean.cm.balance, -SD, SD, [0])
            ax.set_title('%s minus %s' % (hi.replace(', calm', ''), lo.replace(', calm', '')), fontsize=7)
        elif KIND == 'salt':
            sthumb(ax, [(sprofile(hi, ph), PCOL.get(ph, 'k'), '-', lab(hi)), (sprofile(lo, ph), PCOL.get(ph, 'k'), '--', lab(lo))]
                   + ([(sprofile('calm', ph), PCOL.get(ph, 'k'), ':', lab('calm', 'calm (no wind)'))] if hi == 'up-cove' else []))
            ax.set_title('%s (solid) vs %s (dashed)%s' % (hi.replace(', calm', ''), lo.replace(', calm', ''), ' vs calm (dotted)' if hi == 'up-cove' else ''), fontsize=7)
        elif KIND in ('profiles', 'lateral'):
            TH = pthumb if KIND == 'profiles' else lthumb
            TH(ax, [(PF(hi, ph), PCOL[ph], '-', lab(hi)), (PF(lo, ph), PCOL[ph], '--', lab(lo))]
               + ([(PF('calm', ph), PCOL[ph], ':', lab('calm', 'calm (no wind)'))] if hi == 'up-cove' else []), XL[ph])
            ax.set_title('%s (solid) vs %s (dashed)%s' % (hi.replace(', calm', ''), lo.replace(', calm', ''), ' vs calm (dotted)' if hi == 'up-cove' else ''), fontsize=7)
        else:
            cd = thumb(ax, R[(SECT, hi, ph)]['U'] - R[(SECT, lo, ph)]['U'], R[(SECT, hi, ph)]['zw'], vd, cmocean.cm.curl)
            ax.set_title('%s minus %s' % (hi.replace(', calm', ''), lo.replace(', calm', '')), fontsize=7)
    fig.canvas.draw()
    arrow(fig, a0, a2)
if KIND == 'saltsec':
    cax = fig.add_axes([0.94, 0.25, 0.008, 0.35])
    fig.colorbar(cdd, cax=cax).set_label('salinity difference [g kg$^{-1}$]\n+ = saltier in the\nfirst condition', fontsize=8)
if KIND == 'sections':
    cax = fig.add_axes([0.94, 0.25, 0.008, 0.35])
    fig.colorbar(cd, cax=cax).set_label('difference [cm s$^{-1}$]\n+ = more into the cove\nin the first condition', fontsize=8)
fig.suptitle(('How the flood / ebb / residual %s of Penn Cove are modified by the tide, stratification and wind  ' % {'sections': 'cross-sections', 'profiles': 'section-averaged velocity profiles', 'lateral': 'depth-averaged across-channel velocity profiles'}[KIND]
              if KIND not in ('salt', 'saltsec') else 'How the SALINITY %s of Penn Cove are modified by the tide, stratification and wind  '
              % ('profiles (section-averaged)' if KIND == 'salt' else 'cross-sections')) +
             ('(wb1_t0_xn11abbur00, %s ONLY, hourly snapshots)' % MLAB.upper() if MONTH else '(wb1_t0_xn11abbur00, 2024-2025, hourly snapshots)'), fontsize=13)
fn = out_dir / ('pc_conceptual_map%s_%s%s.png' % ({'sections': '', 'profiles': '_profiles', 'lateral': '_lateral', 'salt': '_salt', 'saltsec': '_saltsec'}[KIND], SECT, '_' + MONTH if MONTH else ''))
fig.savefig(fn, dpi=170, transparent=True)
plt.close(fig)
print('saved ' + str(fn))
