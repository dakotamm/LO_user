"""
Saratoga Passage version of the Penn Cove conceptual map (20261105_pc_conceptual_map.py -kind profiles / salt):
how the flood / ebb / residual section-averaged VELOCITY (or SALINITY) profiles of a Saratoga section are
modified by the tide (spring / neap, tropic / equatorial), stratification and wind stress.

SECTIONS (tef2 collection wb1_pc1): skagit_sp = the Skagit Bay entrance (NE), sp_mid = Saratoga Passage
    across lat 48.184, between the south of the box and the Penn Cove mouth. pc_lp also works (a check
    against the Penn Cove map).
CLASSES  exactly the Penn Cove map's (pc_tidal_cycles.cycle_masks), so the rows compare the same hours:
    the tidal classes come from the Penn Cove mouth transport (one tide for the whole region), but strat
    is the PENN COVE bottom-top delta sigma0 and wind the ALONG-COVE (~E-W) stress, not Saratoga's own.
PROFILES  from 20261111_sp_section_binned.py (hourly, 1 m bins of depth below the surface):
    velocity U = sum_t Q_b / sum_t A_b over a class's hours (transport-weighted), salinity
    sum_t (S A)_b / sum_t A_b; flood / ebb = sign of the section's own hourly Q; residual = all hours of
    the class. A bin is drawn where it is wetted, on average, by at least half a face.
ORIENTATION  + = the FLOOD direction, set per section by the sign of corr(Q, d zeta/dt) (water moving the
    way it goes while the sea rises), drawn at the LEFT as 'into the cove' on the Penn Cove map.
TEXT  the 'what it changes' boxes are computed from the profiles (no hand-written reading yet).
-kind lateral: ACROSS-CHANNEL profiles of the DEPTH-AVERAGED velocity (the counterpart of the Penn Cove map's
    -kind lateral): per face, U_d = sum_t Q_p / sum_t A_p over a class's hours (transport-weighted, depth-
    integrated per face, from the binned file's Qp, Ap); y = km along the section from its first end (sp_mid: W at
    the top; skagit_sp: NW at the top; pc_lp: N at the top; face positions from the tef2 structure file).
    Text: area-weighted U_d of each half of the section and the lateral exchange 0.5 sum_p |(U_d,p - Ubar) A_p|.
-norm True: velocities divided by the section's tidal velocity scale U_T = <|Q|> / <A> (all hours, one number
    per section), so the three sections can be compared by eye; the text keeps cm/s.

OUTPUT LO_output/DM_outs/20261111_sp_conceptual_map/sp_conceptual_map_[profiles|salt|lateral][_norm]_[sect].png

run 20261111_sp_conceptual_map.py [-sect skagit_sp|sp_mid|pc_lp] [-kind profiles|salt|lateral] [-norm True]
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import xarray as xr

from lo_tools import Lfun
from pc_tidal_cycles import cycle_masks

parser = argparse.ArgumentParser()
parser.add_argument('-sect', default='sp_mid', choices=['skagit_sp', 'sp_mid', 'pc_lp'])
parser.add_argument('-kind', default='profiles', choices=['profiles', 'salt', 'lateral'])
parser.add_argument('-norm', default=False, type=lambda x: x.lower() in ['true', '1', 'yes'])
args = parser.parse_args()
SECT, KIND, NORM = args.sect, args.kind, args.norm and args.kind != 'salt'
NAME = {'skagit_sp': 'Skagit Bay entrance (skagit_sp)', 'sp_mid': 'Saratoga Passage, lat 48.18 (sp_mid)',
        'pc_lp': 'Penn Cove mouth (pc_lp)'}
ENDS = {'skagit_sp': ('NW', 'SE'), 'sp_mid': ('W', 'E'), 'pc_lp': ('N', 'S')}   # first face, last face
FLOOD_DIR = {'skagit_sp': ('into Skagit Bay', 'out'), 'sp_mid': ('north', 'south'), 'pc_lp': ('into the cove', 'out')}
PHASES = ['flood', 'ebb', 'residual']
PCOL = {'flood': '#D55E00', 'ebb': '#0072B2', 'residual': 'k'}
ROWS = [   # (first class, second class, driver box); classes and wording as the Penn Cove map
    ('spring', 'neap', 'SPRING vs NEAP\nsemidiurnal tide (M2 + S2), ~14.8 d\nthirds of the D2 envelope of the\nPenn Cove mouth transport,\nwithin each month'),
    ('tropic', 'equatorial', 'TROPIC vs EQUATORIAL\ndiurnal tide (K1 + O1), ~13.7 d\nthirds within each month\n(season balanced)'),
    ('high strat', 'low strat', 'HIGH vs LOW STRATIFICATION\nPENN COVE $\\Delta\\sigma_0$ bottom - top,\nrecord thirds, all hours;\nhigh: mostly Nov-Mar, Jun;\nlow: mostly Aug-Oct'),
    ('up-cove', 'down-cove', 'UP-COVE vs DOWN-COVE STRESS\nalong-COVE (~E-W) stress ROMS applied,\n> +0.01 vs < -0.01 Pa (Godin)\nup: winter 13% | down: summer 34%'),
]

Ldir = Lfun.Lstart(gridname='wb1')
B = xr.open_dataset(Ldir['LOo'] / 'DM_outs' / '20261111_sp_section_binned' / ('binned_%s.nc' % SECT))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261111_sp_conceptual_map'
out_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ orientation, phases, classes
Q, zeta = B.Q.values, B.zeta.values
r = np.corrcoef(Q, np.gradient(zeta))[0, 1]
SGN = np.sign(r)                          # + = flood direction
Q = SGN * Q
Qb, Ab, SAb = SGN * B.Qb.values.astype(float), B.Ab.values.astype(float), B.SAb.values.astype(float)
if 'Qp' in B:                             # per face, depth-integrated (binned files from 2026-11-11 on)
    Qp, Ap = SGN * B.Qp.values.astype(float), B.Ap.values.astype(float)
else:
    assert KIND != 'lateral', 'binned_%s.nc has no per-face Qp / Ap: rerun 20261111_sp_section_binned.py' % SECT
    Qp, Ap = np.zeros((len(Q), B.sizes['p'])), np.ones((len(Q), B.sizes['p']))
UT = np.abs(Q).mean() / Ab.sum(axis=1).mean()                              # tidal velocity scale [m/s]
ST = xr.open_dataset(Ldir['LOo'] / 'extract' / B.attrs['gtagex'] / 'tef2' / 'structure_2024.01.01_2025.12.31_wb1_pc1.nc')
flon, flat = ST[SECT + '_lon'].values, ST[SECT + '_lat'].values
assert len(flon) == Qp.shape[1], 'structure file faces do not match the extraction'
YC = np.concatenate([[0], np.cumsum(np.hypot(np.diff(flon) * 111.32 * np.cos(np.deg2rad(flat.mean())), np.diff(flat) * 110.57))])   # km from the first face
YL = YC[-1] + 0.5 * (YC[1] - YC[0]) if len(YC) > 1 else 1
FIRST = YC < 0.5 * YC[-1]                 # faces in the first half (NW / W / N)
E = B.edges.values
DMID = 0.5 * (E[1:] + E[:-1])
AMIN = 0.5 * B.dd.values.mean()           # a bin counts where it is wetted by >= half a face on average
CM = cycle_masks(B.time.values)
CM['all'] = np.ones(len(Q), bool)
PH = {'flood': Q > 0, 'ebb': Q < 0, 'residual': np.ones(len(Q), bool)}
print('%s: corr(Q, dzeta/dt) %+.2f -> flood = %s; %d h, flood %.0f%% of hours' % (SECT, r, FLOOD_DIR[SECT][0], len(Q), 100 * PH['flood'].mean()))


def prof(c, ph):
    """class composite profile: velocity [m/s] or salinity, NaN where the bin is too rarely wet"""
    m = CM[c] & PH[ph]
    if m.sum() == 0:
        return np.full(len(DMID), np.nan)
    A = Ab[m].sum(axis=0)
    ok = A >= AMIN * m.sum()
    num = Qb[m].sum(axis=0) if KIND == 'profiles' else SAb[m].sum(axis=0)
    return np.where(ok, num / np.where(ok, A, 1), np.nan)


def lateral(c, ph):
    """per-face depth-averaged velocity [m/s], transport-weighted over the class's hours"""
    m = CM[c] & PH[ph]
    if m.sum() == 0:
        return np.full(Qp.shape[1], np.nan)
    A = Ap[m].sum(axis=0)
    return np.where(A > 0, Qp[m].sum(axis=0) / np.where(A > 0, A, 1), np.nan)


def lstats(c, ph):
    """first-half and second-half area-weighted U_d [cm/s] and the lateral exchange [m3/s]"""
    m = CM[c] & PH[ph]
    A = Ap[m].mean(axis=0)
    Ud = lateral(c, ph)
    Ub = np.nansum(Ud * A) / A.sum()
    half = lambda k: 100 * np.nansum(Ud[k] * A[k]) / A[k].sum()
    return half(FIRST), half(~FIRST), 0.5 * np.nansum(np.abs((Ud - Ub) * A))


def qmean(c, ph):
    m = CM[c] & PH[ph]
    return Q[m].mean() if m.any() else np.nan


def lab(c, name=None):
    return '%s (%d h)' % (name or c, (CM[c]).sum())


# ------------------------------------------------------------------ numbers for the text boxes
def vnums(c):
    """residual: surface bin, max landward (+) and its depth, max seaward (-) below the surface bin; cm/s"""
    u = 100 * prof(c, 'residual')
    k = np.nanargmax(u)
    kd = 1 + np.nanargmin(u[1:])
    return u[0], u[k], DMID[k], u[kd], DMID[kd]


def snums(c, ph='residual'):
    s = prof(c, ph)
    k15 = np.argmin(np.abs(DMID - 14.5))
    return s[0], s[k15]


def ctext(hi, lo):
    L = lambda c: c.replace(', calm', '')
    rows = []
    if KIND == 'lateral':
        a_, b_ = ENDS[SECT]
        cs = [hi, lo] + (['calm'] if hi == 'up-cove' else [])
        rows = ['depth-averaged U_d [cm/s], + = %s' % FLOOD_DIR[SECT][0], '                     %-2s half   %-2s half  lateral exch.' % (a_, b_)]
        for ph in PHASES:
            for c in cs:
                rows.append('%-8s %-11s %+6.1f   %+6.1f   %6.0f m3/s' % ((ph if c == hi else '', L(c)[:11]) + lstats(c, ph)))
        rows.append('residual lateral exchange %s / %s: x%.2f' % (L(hi), L(lo), lstats(hi, 'residual')[2] / lstats(lo, 'residual')[2]))
        return '\n'.join(rows)
    if KIND == 'profiles':
        for c in (hi, lo):
            u0, umax, dmax, umin, dmin = vnums(c)
            rows.append('%-11s Q flood %+6.0f  ebb %+6.0f m3/s' % (L(c), qmean(c, 'flood'), qmean(c, 'ebb')))
            rows.append('%-11s resid surf %+5.1f; in %+4.1f @%2.0f m; out %+4.1f @%2.0f m' % ('', u0, umax, dmax, umin, dmin))
        rows.append('flood x%.2f, ebb x%.2f (%s / %s)' % (qmean(hi, 'flood') / qmean(lo, 'flood'), qmean(hi, 'ebb') / qmean(lo, 'ebb'), L(hi), L(lo)))
        rows.append('residual in cm/s; + = %s' % FLOOD_DIR[SECT][0])
    else:
        for c in (hi, lo):
            s0, s15 = snums(c)
            rows.append('%-11s surface %.2f, 15 m %.2f, diff %.2f g/kg' % (L(c), s0, s15, s15 - s0))
        rows.append('%s - %s: surface %+.2f, 15 m %+.2f' % (L(hi), L(lo), snums(hi)[0] - snums(lo)[0], snums(hi)[1] - snums(lo)[1]))
    if hi == 'up-cove':
        if KIND == 'profiles':
            u0, umax, dmax, umin, dmin = vnums('calm')
            rows.append('calm: resid surf %+5.1f; in %+4.1f @%2.0f m' % (u0, umax, dmax))
        else:
            rows.append('calm: surface %.2f, 15 m %.2f g/kg' % snums('calm'))
    return '\n'.join(rows)


# ------------------------------------------------------------------ figure
def textbox(ax, text, fc, fs=8.5, mono=False):
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0.02, 0.04), 0.96, 0.92, boxstyle='round,pad=0.01,rounding_size=0.03',
                                transform=ax.transAxes, fc=fc, ec='0.4', lw=1))
    ax.text(0.05, 0.5, text, transform=ax.transAxes, fontsize=fs, va='center', ha='left', linespacing=1.35,
            family='monospace' if mono else None)


def arrow(fig, ax0, ax1):
    b0, b1 = ax0.get_position(), ax1.get_position()
    y = 0.5 * (b0.y0 + b0.y1)
    fig.patches.append(FancyArrowPatch((b0.x1, y), (b1.x0, y), transform=fig.transFigure,
                                       arrowstyle='-|>', mutation_scale=14, color='0.3', lw=1.5))


ALLC = ['all', 'calm'] + [c for r_ in ROWS for c in r_[:2]]
ZMAX = max(DMID[np.isfinite(prof('all', 'residual'))].max() + 0.5, 5)
V = (lambda x: x / UT) if NORM else (lambda x: 100 * x)                   # plotted units: U / U_T, or cm/s
FN = lateral if KIND == 'lateral' else prof
if KIND in ('profiles', 'lateral'):       # one x-range per column, shared down the page
    XL = {ph: 1.1 * np.nanmax([np.nanmax(np.abs(V(FN(c, ph)))) for c in ALLC]) for ph in PHASES}
else:                                     # one salinity range for the page
    SR = np.nanpercentile(np.concatenate([prof(c, ph) for c in ALLC for ph in PHASES]), [0.5, 99.5]) + np.array([-0.1, 0.1])


def thumb(ax, curves, ph):
    if KIND == 'lateral':
        for x, col, ls, lb in curves:
            ax.plot(V(x), YC, color=col, ls=ls, lw=1.8, marker='o', ms=2.5, label=lb)
        ax.axvline(0, color='k', lw=0.6)
        ax.set_xlim(XL[ph], -XL[ph])
        ax.set_ylim(YL, -0.5 * (YC[1] - YC[0]))   # first end at the top
        a_, b_ = ENDS[SECT]
        ax.text(0.98, 0.98, a_, transform=ax.transAxes, ha='right', va='top', fontsize=7, fontweight='bold')
        ax.text(0.98, 0.02, b_, transform=ax.transAxes, ha='right', va='bottom', fontsize=7, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6, length=2)
        ax.legend(fontsize=6, loc='lower left', handlelength=1.5)
        return
    for x, col, ls, lb in curves:
        ax.plot(V(x) if KIND == 'profiles' else x, DMID, color=col, ls=ls, lw=1.8, label=lb)
    if KIND == 'profiles':
        ax.axvline(0, color='k', lw=0.6)
        ax.set_xlim(XL[ph], -XL[ph])      # reversed: + (the flood direction) at the left
    else:
        ax.set_xlim(*SR)
    ax.set_ylim(ZMAX, 0)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=6, length=2)
    ax.legend(fontsize=6, loc='lower left', handlelength=1.5)


nr = len(ROWS) + 1
fig = plt.figure(figsize=(20, 3.0 * nr))
gs = fig.add_gridspec(nr, 6, width_ratios=[1.35, 0.12, 1.75, 1, 1, 1], height_ratios=[1.1] + [1] * len(ROWS),
                      left=0.02, right=0.97, top=0.93 if NORM else 0.95, bottom=0.02, wspace=0.08, hspace=0.35)
fl, eb = FLOOD_DIR[SECT]
UNIT = 'U / U$_T$' if NORM else 'cm s$^{-1}$'
XLAB = (r'$\leftarrow$ %s   [%s]   %s $\rightarrow$' % (fl, UNIT, eb)) if KIND != 'salt' else 'salinity [g kg$^{-1}$]'
textbox(fig.add_subplot(gs[0, 0]), 'BASELINE, all hours\nsection: %s,\naveraged across the section in\n1 m depth bins below the surface;\n%s'
        % (NAME[SECT], 'flood direction (%s) at the left' % fl if KIND == 'profiles' else 'salinity'), '#EEEEEE', fs=9) if KIND != 'lateral' else \
    textbox(fig.add_subplot(gs[0, 0]), 'BASELINE, all hours\nsection: %s,\ndepth-averaged velocity of each\nface across the channel,\n%s at the top;\nflood direction (%s) at the left'
            % (NAME[SECT], ENDS[SECT][0], fl), '#EEEEEE', fs=9)
if KIND == 'lateral':
    a_, b_ = ENDS[SECT]
    btx = 'depth-averaged U_d [cm/s], + = %s\n            %-2s half   %-2s half  lateral exch.\n' % (fl, a_, b_) + '\n'.join(
        '%-10s %+6.1f   %+6.1f   %6.0f m3/s' % ((ph,) + lstats('all', ph)) for ph in PHASES)
elif KIND == 'profiles':
    u0, umax, dmax, umin, dmin = vnums('all')
    btx = ('all hours: mean Q flood %+.0f, ebb %+.0f m3/s\nresidual: surface %+.1f cm/s; max %s %+.1f at %.0f m;\nmax %s %+.1f at %.0f m (below the surface bin)\n'
           '+ = %s (the flood direction: corr(Q, dzeta/dt) %+.2f)' % (qmean('all', 'flood'), qmean('all', 'ebb'), u0, fl, umax, dmax, eb, umin, dmin, fl, SGN * r))
else:
    s0, s15 = snums('all')
    btx = 'all hours: salinity %.2f at the surface, %.2f at 15 m\n(difference %.2f g/kg)' % (s0, s15, s15 - s0)
textbox(fig.add_subplot(gs[0, 2]), btx, '#EEEEEE', fs=8.5, mono=True)
for j, ph in enumerate(PHASES):
    ax = fig.add_subplot(gs[0, 3 + j])
    thumb(ax, [(FN('all', ph), PCOL[ph], '-', 'all hours')], ph)
    ax.set_xlabel(XLAB, fontsize=7)
    ax.set_title(ph.upper(), fontsize=11, fontweight='bold')
    if j == 0:
        ax.set_ylabel('km along the section' if KIND == 'lateral' else 'depth below surface [m]', fontsize=7)
COL = ['#DCEBF7', '#FCE6D4', '#F3E3F0', '#DFF2E7']
for i, (hi, lo, dtext) in enumerate(ROWS):
    a0 = fig.add_subplot(gs[i + 1, 0])
    textbox(a0, dtext, COL[i], fs=9)
    a2 = fig.add_subplot(gs[i + 1, 2])
    textbox(a2, ctext(hi, lo), 'white', fs=6.8 if KIND == 'lateral' else 7.5, mono=True)
    for j, ph in enumerate(PHASES):
        ax = fig.add_subplot(gs[i + 1, 3 + j])
        thumb(ax, [(FN(hi, ph), PCOL[ph], '-', lab(hi)), (FN(lo, ph), PCOL[ph], '--', lab(lo))]
              + ([(FN('calm', ph), PCOL[ph], ':', lab('calm', 'calm (no wind)'))] if hi == 'up-cove' else []), ph)
        ax.set_title('%s (solid) vs %s (dashed)%s' % (hi, lo, ' vs calm (dotted)' if hi == 'up-cove' else ''), fontsize=7)
    fig.canvas.draw()
    arrow(fig, a0, a2)
fig.suptitle('How the flood / ebb / residual %s of %s are modified by the tide, stratification and wind  '
             '(wb1_t0_xn11abbur00, 2024-2025, hourly)%s'
             % ({'profiles': 'section-averaged velocity profiles', 'salt': 'section-averaged SALINITY profiles',
                 'lateral': 'depth-averaged across-channel velocity profiles'}[KIND], NAME[SECT],
                '\nvelocities normalised by the tidal velocity scale U$_T$ = <|Q|>/<A> = %.1f cm s$^{-1}$' % (100 * UT) if NORM else ''), fontsize=13)
fn = out_dir / ('sp_conceptual_map_%s%s_%s.png' % (KIND, '_norm' if NORM else '', SECT))
fig.savefig(fn, dpi=170, transparent=True)
plt.close(fig)
print('saved', fn)
