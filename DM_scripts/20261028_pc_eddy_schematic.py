"""
Schematic of the inner Penn Cove eddy and how it varies, drawn from the
measured numbers of this analysis (wb1_t0_xn11abbur00, 2024-25; velocity field only).

(a) plan view on the real coastline: eddy centre and core radius (present-hour
    mean centre and median r_eq, 20261021_pc_eddy_center.py, pc_cp region) for
    the three depth bands; the cove-scale exchange at the mouth (in on the
    north side, out on the south side of pc_lp); lobe bathymetry.
(b) seasonal cycle: monthly share of hours with the eddy present, per depth band
(c) how fast the eddy turns, by depth, in hours with the eddy absent vs present
    (that band's hourly flag): median rim speed U_rot (ring-mean tangential
    velocity of the Godin-filtered field, 20261021_pc_eddy_center.py) and core
    radius; arrow width ~ rim speed. Velocity field only (no particles).
(d) along-cove wind events: change of each layer's eddy presence (share of the
    day, seasonal cycle removed) after strong into-cove / out-of-cove wind
    (composite of 2026-10-22: daily cove wind > +3 / < -3 m/s, 11 / 60 events).
    These values are typed in below (WIND) from that composite.
(e) stratification at the mouth: daily d(sigma0) bottom - top at pc_lp (tef2
    strat file, width-weighted top/bottom salt and temp, gsw sigma0, Godin-
    filtered, UTC-8 days), its 31-d running mean, and the onsets of
    stratifying / mixing events: the 31-d anomaly first rises above +1 sd /
    falls below -1 sd.
(f) as (d) for those stratification events (computed here): change of each
    layer's eddy presence (seasonal cycle removed), the largest change within
    days 0-2 of onset, and any opposite change of >= 10 pp within days 3-5.
(g) the same mouth d(sigma0) with FIXED levels (> 5 / < 2 kg m-3, no seasonal
    reference) and the onsets of runs above / below them.
(h) as (f), for the fixed-level events of (g).

Output: LO_output/DM_outs/20261028_pc_eddy_schematic/pc_eddy_schematic.png

  python 20261028_pc_eddy_schematic.py
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Polygon

import gsw

from lo_tools import Lfun, zfun
from lo_tools import plotting_functions as pfun

Ldir = Lfun.Lstart(gridname='wb1')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261028_pc_eddy_schematic'
Lfun.make_dir(out_dir)
cdir = Ldir['LOo'] / 'DM_outs' / '20261021_pc_eddy_center' / 'pc_cp'
M_PER_DEG, LAT0 = 111.32e3, 48.23
COS0 = np.cos(np.deg2rad(LAT0))
LON_MOUTH = pd.read_pickle(Ldir['LOo'] / 'section_lines' / 'pc_lp.p').x.mean()
SEC = {k: pd.read_pickle(Ldir['LOo'] / 'section_lines' / (k + '.p')) for k in ['pc_cp', 'pc_lp']}
CCW, CW, INK, MUTED = '#2a6fdb', '#e8590c', '#222222', '#6b6b6b'
BANDS = [('z0-4', '0-4 m'), ('z4-10', '4-10 m'), ('z10-bed', '10 m - bed')]
DEPTHC = {'z0-4': '#bdbdbd', 'z4-10': '#737373', 'z10-bed': '#252525'}
# (c) along-cove wind composite: presence change [percentage points], seasonal cycle removed
WIND = {'in': {'z0-4': (23, 'day 0'), 'z4-10': (20, 'day 0'), 'z10-bed': (12, 'day 0')},
        'out': {'z0-4': (-21, 'day 0; back to +13 by day 2'), 'z4-10': (14, 'peak of days 0-2'),
                'z10-bed': (16, 'day 1')}}

EDDY, ROT = {}, {}
for b, _ in BANDS:
    C = pd.read_csv(cdir / ('pc_eddy_center_%s.csv' % b))
    t = pd.to_datetime(C.time).dt.month
    P = C[C.present.astype(bool)]
    x, y = P.x_c.mean(), P.y_c.mean()
    EDDY[b] = dict(lon=LON_MOUTH + x / (M_PER_DEG * COS0), lat=LAT0 + y / M_PER_DEG, r=P.r_eq.median(),
                   T=P.T_turn_h.median(), p_all=C.present.mean(), p_nd=C.present[t.isin([11, 12])].mean(),
                   p_ao=C.present[t.isin([8, 9, 10])].mean(),
                   monthly=C.present.astype(float).groupby(t).mean())
    ROT[b] = {pr: dict(u=100 * C.U_rot[C.present.astype(bool) == pr].median(),
                       lo=100 * C.U_rot[C.present.astype(bool) == pr].quantile(0.25),
                       hi=100 * C.U_rot[C.present.astype(bool) == pr].quantile(0.75),
                       r=C.r_eq[C.present.astype(bool) == pr].median()) for pr in [False, True]}

# ---------------------------------------------- mouth stratification ---
anom = lambda x: x - x.rolling(31, center=True, min_periods=20).mean()
ds_ = xr.open_dataset(Ldir['LOo'] / 'extract' / 'wb1_t0_xn11abbur00' / 'tef2' / 'strat_2024.01.01_2025.12.31_wb1_pc1.nc')
ss = ds_.sel(sect='pc_lp')
drho = gsw.sigma0(ss.s_bot.values, ss.t_bot.values) - gsw.sigma0(ss.s_top.values, ss.t_top.values)
th = pd.to_datetime(ds_.time.values)
ds_.close()
gS = pd.Series(zfun.lowpass(drho, f='godin'), index=th)
DR = gS.groupby((gS.index - pd.Timedelta('8h')).floor('D')).mean().asfreq('D')
DRa = anom(DR)
SD = DRa.std()
PA = {}
for b, _ in BANDS:
    C = pd.read_csv(cdir / ('pc_eddy_center_%s.csv' % b), usecols=['time', 'present'], parse_dates=['time'])
    C['present'] = C.present.astype(float)
    PA[b] = anom(C.groupby((C.time - pd.Timedelta('8h')).dt.floor('D')).present.mean().asfreq('D'))
STRAT, ONSET, MAG = {}, {}, {}
for k, cond in [('strat', DRa > SD), ('mix', DRa < -SD)]:
    st = DRa.index[cond & ~cond.shift(1, fill_value=False)]
    ONSET[k] = st
    ev = np.array([[DRa.get(t + pd.Timedelta(days=L), np.nan) for L in range(3)] for t in st])
    pk = np.nanmean(ev, axis=0)
    MAG[k] = dict(onset=np.nanmean(ev[:, 0]), peak=pk[np.nanargmax(np.abs(pk))], peak_day=int(np.nanargmax(np.abs(pk))),
                  pct=100 * np.nanmean((DRa / (DR - DRa)).reindex(st).values))
    STRAT[k] = {}
    for b, _ in BANDS:
        comp = np.array([np.nanmean([PA[b].get(t + pd.Timedelta(days=L), np.nan) for t in st]) for L in range(6)])
        k0 = int(np.nanargmax(np.abs(comp[:3])))
        v = 100 * comp[k0]
        later = comp[3:] * np.sign(-v)
        note = 'day %d' % k0
        if np.nanmax(later) * 100 >= 10:
            k1 = 3 + int(np.nanargmax(later))
            note += '; then %+d by day %d' % (round(100 * comp[k1]), k1)
        STRAT[k][b] = (int(round(v)), note)
    print('%s events: %d; %s' % (k, len(st), STRAT[k]))



def compose(st):
    """Presence-anomaly composite after event onsets st: {band: (pp, note)}, as for STRAT."""
    out = {}
    for b, _ in BANDS:
        comp = np.array([np.nanmean([PA[b].get(t + pd.Timedelta(days=L), np.nan) for t in st]) for L in range(6)])
        k0 = int(np.nanargmax(np.abs(comp[:3])))
        v = 100 * comp[k0]
        later = comp[3:] * np.sign(-v)
        note = 'day %d' % k0
        if np.nanmax(later) * 100 >= 10:
            k1 = 3 + int(np.nanargmax(later))
            note += '; then %+d by day %d' % (round(100 * comp[k1]), k1)
        out[b] = (int(round(v)), note)
    return out


# (g, h) events on a FIXED level of mouth stratification (no seasonal reference)
SHI, SLO = 5.0, 2.0
ABS, AONSET, AMAG = {}, {}, {}
for k, cond in [('hi', DR > SHI), ('lo', DR < SLO)]:
    st = DR.index[cond & ~cond.shift(1, fill_value=False)]
    run = (cond != cond.shift()).cumsum()[cond]
    AONSET[k] = st
    ABS[k] = compose(st)
    AMAG[k] = dict(mean3=np.nanmean([DR.reindex(pd.date_range(t, periods=3)).mean() for t in st]),
                   run=np.median(run.value_counts().values))
    print('%s (fixed level) events: %d; %s' % (k, len(st), ABS[k]))

g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat, h = g.lon_rho.values, g.lat_rho.values, g.h.values
mask = g.mask_rho.values.astype(bool)
g.close()


def circ_arrow(ax, x, y, r, sense, color, lw, aspect=1.0):
    """Circular arrow centred (x, y), radius r (data units, x scaled by aspect): the arc stops short and the
    head is drawn past its end, so a thick arc does not hide the head."""
    sg = 1 if sense > 0 else -1
    t0, t1 = (30, 330) if sg > 0 else (330, 30)
    P = lambda t: (x + r * aspect * np.cos(np.radians(t)), y + r * np.sin(np.radians(t)))
    th = np.linspace(t0, t1 - sg * 14, 120)
    ax.plot(*P(th), color=color, lw=lw, solid_capstyle='butt', zorder=5)
    a = FancyArrowPatch(P(t1 - sg * 16), P(t1 + sg * 10), arrowstyle='-|>', mutation_scale=12 + 2.6 * lw,
                        color=color, lw=0, shrinkA=0, shrinkB=0, zorder=6)
    ax.add_patch(a)


fig = plt.figure(figsize=(15, 23.5))
gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.2, 1.2, 1.2], hspace=0.33)
g4 = gs[3].subgridspec(1, 2, wspace=0.14)
gt = gs[0].subgridspec(1, 2, width_ratios=[1.55, 1], wspace=0.12)
gb = gs[1].subgridspec(1, 2, wspace=0.14)
g3 = gs[2].subgridspec(1, 2, wspace=0.14)


def response_panel(ax, cols, vals, caption, title):
    """Two-column, three-layer panel of presence changes (pp); cols = [(key, header, icon_fn or None)]."""
    ax.set_xlim(0, 2); ax.set_ylim(0, 3.25)
    ax.axis('off')
    for c, col_ in enumerate(cols):
        k, hdr, icon = col_[:3]
        sub = col_[3] if len(col_) > 3 else None
        ax.text(c + 0.5, 3.17 if sub else 3.1, hdr, ha='center', va='center', fontsize=10, fontweight='bold',
                color=INK)
        if sub:
            ax.text(c + 0.5, 3.0, sub, ha='center', va='center', fontsize=8.2, color=INK)
        if icon is not None:
            icon(ax, c + 0.5, 2.8 if sub else 2.86)
        for r_, (b, lab) in enumerate(BANDS):
            yc = 2.3 - r_ * 0.9
            v, note = vals[k][b]
            L = 0.012 * abs(v)
            col = CCW if v > 0 else CW
            ax.add_patch(FancyArrowPatch((c + 0.3, yc - np.sign(v) * L / 2), (c + 0.3, yc + np.sign(v) * L / 2),
                                         arrowstyle='simple', mutation_scale=18, color=col))
            ax.text(c + 0.4, yc, '%s eddy: %+d pp\n(%s)' % ('more' if v > 0 else 'less', v, note), ha='left',
                    va='center', fontsize=8.5, color=INK)
    for r_, (b, lab) in enumerate(BANDS):
        ax.text(-0.02, 2.3 - r_ * 0.9, lab, ha='right', va='center', fontsize=10, color=INK)
    ax.plot([1, 1], [0.02, 3.0], color='0.85', lw=1)
    ax.text(-0.12, -0.06, caption, ha='left', va='top', fontsize=8.3, color=MUTED)
    ax.set_title(title, fontsize=11, loc='left', x=-0.12)


def mag_txt(k):
    m = MAG[k]
    return ('$\\Delta\\sigma_0$ %+.1f kg m$^{-3}$ (%+.0f%%)\nvs its 31-d mean at onset'
            % (m['onset'], m['pct']))


def wind_icon(sign):
    def f(ax, x, y):
        ax.add_patch(FancyArrowPatch((x + 0.17 * sign, y), (x - 0.17 * sign, y), arrowstyle='simple',
                                     mutation_scale=16, color='#5b5b5b'))
    return f


def layer_icon(strong):
    def f(ax, x, y):
        if strong:
            ax.add_patch(plt.Rectangle((x - 0.17, y), 0.34, 0.07, color='#c6dbef', lw=0))
            ax.add_patch(plt.Rectangle((x - 0.17, y - 0.07), 0.34, 0.07, color='#2171b5', lw=0))
        else:
            ax.add_patch(plt.Rectangle((x - 0.17, y - 0.07), 0.34, 0.14, color='#6baed6', lw=0))
    return f

# ------------------------------------------------------------ (a) map ---
ax = fig.add_subplot(gt[0])
hv = np.where(mask, h, np.nan)
pm = ax.pcolormesh(lon, lat, hv, cmap='Blues', vmin=0, vmax=40, shading='nearest', alpha=0.45, zorder=0)
pfun.add_coast(ax, color='0.45', linewidth=0.8)
for k, s in SEC.items():
    ax.plot(s.x, s.y, color=MUTED, lw=0.9, ls='--', zorder=2)
    if k == 'pc_lp':
        ax.text(s.x.mean(), s.y.max() + 0.0006, k, ha='center', va='bottom', fontsize=8, color=MUTED)
    else:
        ax.text(s.x.mean(), s.y.min() - 0.0008, k, ha='center', va='top', fontsize=8, color=MUTED)
for b, lab in BANDS:
    e = EDDY[b]
    ring = np.linspace(0, 2 * np.pi, 200)
    ax.plot(e['lon'] + e['r'] * np.cos(ring) / (M_PER_DEG * COS0), e['lat'] + e['r'] * np.sin(ring) / M_PER_DEG,
            color=DEPTHC[b], lw=1.6, ls='-', zorder=3,
            label='%s: core radius %.0f m' % (lab, e['r']))
    ax.plot(e['lon'], e['lat'], '+', color=DEPTHC[b], ms=11, mew=2.2, zorder=4)
e = EDDY['z4-10']
circ_arrow(ax, e['lon'], e['lat'], 0.62 * e['r'] / M_PER_DEG, +1, CCW, 3.0, aspect=1 / COS0)
ax.text(0.0, -0.08, 'Counterclockwise eddy over the deep southern lobe (16-17 m), strongest at 4-10 m:\n'
        'core radius %.0f m (~3 grid cells), turnover %.0f h, present %.0f%% of hours (Nov-Dec %.0f%%, Aug-Oct %.0f%%).\n'
        'Rings and + = present-hour mean core by depth band; the bed centre sits ~200 m west.'
        % (e['r'], e['T'], 100 * e['p_all'], 100 * e['p_nd'], 100 * e['p_ao']), transform=ax.transAxes,
        fontsize=8.3, color=MUTED, va='top')
ym = SEC['pc_lp'].y.mean()
for (y0, dx, txt) in [(ym + 0.0085, -0.011, 'in (north)'), (ym - 0.0035, +0.011, 'out (south)')]:
    x0 = LON_MOUTH + (0.006 if dx < 0 else -0.005)
    ax.add_patch(FancyArrowPatch((x0, y0), (x0 + dx, y0), arrowstyle='simple', mutation_scale=22,
                                 color='#5b5b5b', alpha=0.85, zorder=4))
    ax.text(x0 + dx / 2, y0 + 0.0016, txt, ha='center', va='bottom', fontsize=8.5, color=INK)
ax.text(LON_MOUTH + 0.0005, ym - 0.0105, 'lateral exchange\nat the mouth\n(never reverses)', ha='center',
        va='top', fontsize=8.5, color=INK)
ax.axis([-122.738, -122.638, 48.209, 48.256])
pfun.dar(ax)
ax.tick_params(labelsize=8)
ax.legend(loc='upper left', fontsize=8.5, frameon=False, title='present-hour mean eddy, by depth band',
          title_fontsize=8.5)
ax.set_title('(a) where: the inner-cove eddy and the cove exchange', fontsize=11, loc='left')
cb = fig.colorbar(pm, ax=ax, shrink=0.7, pad=0.01)
cb.set_label('depth [m]', fontsize=8)

# ------------------------------------------------------ (b) seasonal ---
ax = fig.add_subplot(gt[1])
for (m0, m1, sh) in [(7.5, 10.5, '0.95'), (10.5, 12.5, '0.88')]:
    ax.axvspan(m0, m1, color=sh, lw=0, zorder=0)
ax.axvline(10.5, color='0.75', lw=0.8, zorder=1)
ax.text(9, 101, 'Aug-Oct', ha='center', va='bottom', fontsize=8, color=MUTED)
ax.text(11.5, 101, 'Nov-Dec', ha='center', va='bottom', fontsize=8, color=MUTED)
for b, lab in BANDS:
    mm = EDDY[b]['monthly']
    ax.plot(mm.index, 100 * mm.values, '-o', color=DEPTHC[b], lw=2, ms=5, label=lab, zorder=3)
ax.set_xticks(range(1, 13), ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
ax.set_xlim(0.5, 12.5)
ax.set_ylim(0, 108)
ax.set_ylabel('% of hours with the eddy present', fontsize=9)
ax.grid(color='0.9', lw=0.6, axis='y')
ax.legend(fontsize=8.5, frameon=False, loc='upper left')
ax.tick_params(labelsize=8)
ax.set_title('(b) when: seasonal cycle by depth (2024-25)', fontsize=11, loc='left')

# ------------------------------------------------- (c) rotation by depth ---
ax = fig.add_subplot(gb[0])
ax.set_xlim(0, 2); ax.set_ylim(0, 3.25)
ax.axis('off')
for c, (pres, hdr) in enumerate([(False, 'eddy ABSENT (hours)'), (True, 'eddy PRESENT (hours)')]):
    ax.text(c + 0.5, 3.12, hdr, ha='center', va='center', fontsize=10, fontweight='bold', color=INK)
    for r_, (b, lab) in enumerate(BANDS):
        yc = 2.45 - r_ * 0.95
        R = ROT[b][pres]
        circ_arrow(ax, c + 0.5, yc + 0.09, 0.25, np.sign(R['u']), CCW if R['u'] > 0 else CW, 1.5 + 2.8 * abs(R['u']))
        ax.text(c + 0.5, yc - 0.40, 'rim speed %.1f cm/s (IQR %.1f-%.1f)\ncore radius %.0f m'
                % (R['u'], R['lo'], R['hi'], R['r']), ha='center', va='center', fontsize=8.5, color=INK)
for r_, (b, lab) in enumerate(BANDS):
    ax.text(-0.02, 2.45 - r_ * 0.95, lab, ha='right', va='center', fontsize=10, color=INK)
ax.plot([1, 1], [0.02, 3.0], color='0.85', lw=1)
ax.text(-0.12, -0.06, 'Blue = counterclockwise; arrow width ~ rim speed: the mean flow around the core on a ring at\n'
        '0.5-1 core radius (tide-filtered velocity, hourly, 2024-25; medians). In "absent" hours a small,\n'
        'weak counterclockwise patch (radius ~300 m, below the 400 m threshold) still turns at 1-1.5 cm/s;\n'
        'the eddy proper turns 2-4 cm/s, fastest near the surface and slowest near the bed.',
        ha='left', va='top', fontsize=8.3, color=MUTED)
ax.set_title('(c) how fast the eddy turns, by depth', fontsize=11, loc='left', x=-0.12)

# ----------------------------------------------------------- (d) wind ---
response_panel(fig.add_subplot(gb[1]),
               [('in', 'wind INTO the cove\n(> 3 m/s, 11 events)', wind_icon(+1)),
                ('out', 'wind OUT of the cove\n(< -3 m/s, 60 events)', wind_icon(-1))], WIND,
               'Change in the share of the day each layer\'s eddy is present (pp, seasonal cycle removed).\n'
               'Wind into the cove switches the surface eddy on; wind out of the cove switches it off and\n'
               'strengthens the deeper layers, as if each layer\'s eddy is spun up by that layer\'s inflow.\n'
               'Overall presence barely depends on wind (|r| < 0.15). Strong into-cove wind is rare.',
               '(d) how along-cove wind changes it, by depth')

# ------------------------------------------- (e) mouth stratification ---
ax = fig.add_subplot(g3[0])
for yr in [2024, 2025]:
    for (m0, m1, sh) in [((yr, 8, 1), (yr, 11, 1), '0.95'), ((yr, 11, 1), (yr + 1, 1, 1), '0.88')]:
        ax.axvspan(pd.Timestamp(*m0), pd.Timestamp(*m1), color=sh, lw=0, zorder=0)
ax.plot(DR.index, DR.values, color='#08519c', lw=1, label='daily (Godin-filtered)')
ax.plot(DR.index, (DR - DRa).values, color=INK, lw=1.6, ls='--', label='31-d running mean')
for k, mk, col, lab in [('strat', '^', CCW, 'stratifying event (anomaly > +1 sd)'),
                        ('mix', 'v', CW, 'mixing event (anomaly < -1 sd)')]:
    t = ONSET[k]
    ax.plot(t, DR.reindex(t).values, mk, color=col, ms=7, mec='k', mew=0.5, ls='none', zorder=4,
            label='%s, n = %d' % (lab, len(t)))
ax.set_ylabel('$\\Delta\\sigma_0$ bottom - top at pc_lp [kg m$^{-3}$]', fontsize=9)
ax.grid(color='0.9', lw=0.6)
ax.tick_params(labelsize=8)
ax.legend(fontsize=8, frameon=False, loc='upper left')
ax.text(0.0, -0.09, 'Shading: Aug-Oct (light), Nov-Dec (darker). Mouth stratification is highest in\n'
        'Nov-Dec (mean %.1f) and lowest in Aug-Oct (%.1f kg m$^{-3}$), like the eddy\'s seasonal\n'
        'cycle in (b). Events use the 31-d anomaly (sd %.2f kg m$^{-3}$).'
        % (DR[DR.index.month.isin([11, 12])].mean(), DR[DR.index.month.isin([8, 9, 10])].mean(), SD),
        transform=ax.transAxes, fontsize=8.3, color=MUTED, va='top')
ax.set_title('(e) stratification at the mouth (pc_lp), 2024-25', fontsize=11, loc='left')

# ------------------------------------------ (f) stratification events ---
response_panel(fig.add_subplot(g3[1]),
               [('strat', 'mouth STRATIFIES (%d events)' % len(ONSET['strat']), layer_icon(True), mag_txt('strat')),
                ('mix', 'mouth MIXES (%d events)' % len(ONSET['mix']), layer_icon(False), mag_txt('mix'))], STRAT,
               'As (d), for the stratification events of (e). A stratifying mouth switches the surface eddy on and\n'
               'the near-bed eddy off within a day; the 4-10 m eddy dips, then recovers. Mixing does roughly\n'
               'the opposite, more weakly. On event time scales stratification shifts the eddy between depths;\n'
               'its seasonal level follows the season (b, e).',
               '(f) how mouth stratification changes it, by depth')

# --------------------------------------- (g) fixed-level stratification ---
ax = fig.add_subplot(g4[0])
for yr in [2024, 2025]:
    for (m0, m1, sh) in [((yr, 8, 1), (yr, 11, 1), '0.95'), ((yr, 11, 1), (yr + 1, 1, 1), '0.88')]:
        ax.axvspan(pd.Timestamp(*m0), pd.Timestamp(*m1), color=sh, lw=0, zorder=0)
ax.plot(DR.index, DR.values, color='#08519c', lw=1, label='daily (Godin-filtered)')
ax.axhline(SHI, color=CCW, lw=1.2, ls='--')
ax.axhline(SLO, color=CW, lw=1.2, ls='--')
for k, mk, col, lab in [('hi', 's', CCW, 'onset of $\\Delta\\sigma_0$ > %g' % SHI),
                        ('lo', 's', CW, 'onset of $\\Delta\\sigma_0$ < %g' % SLO)]:
    t = AONSET[k]
    ax.plot(t, DR.reindex(t).values, mk, color=col, ms=6, mec='k', mew=0.5, ls='none', zorder=4,
            label='%s kg m$^{-3}$, n = %d' % (lab, len(t)))
ax.set_ylabel('$\\Delta\\sigma_0$ bottom - top at pc_lp [kg m$^{-3}$]', fontsize=9)
ax.grid(color='0.9', lw=0.6)
ax.tick_params(labelsize=8)
ax.legend(fontsize=8, frameon=False, loc='upper left')
ax.text(0.0, -0.09, 'Events on a fixed level of mouth stratification, with no seasonal reference: the first day\n'
        'of each run above %g (strongly stratified, ~%.0f%% of days) or below %g kg m$^{-3}$ (weakly\n'
        'stratified, ~%.0f%% of days). Shading: Aug-Oct (light), Nov-Dec (darker).'
        % (SHI, 100 * (DR > SHI).mean(), SLO, 100 * (DR < SLO).mean()),
        transform=ax.transAxes, fontsize=8.3, color=MUTED, va='top')
ax.set_title('(g) mouth stratification above / below a fixed level', fontsize=11, loc='left')

# ------------------------------------------ (h) fixed-level events ---
response_panel(fig.add_subplot(g4[1]),
               [('hi', 'STRONGLY STRATIFIED (%d)' % len(AONSET['hi']), layer_icon(True),
                 'mouth $\\Delta\\sigma_0$ > %g kg m$^{-3}$ (mean %.1f\nover the first 3 d); runs last %.0f d'
                 % (SHI, AMAG['hi']['mean3'], AMAG['hi']['run'])),
                ('lo', 'WEAKLY STRATIFIED (%d)' % len(AONSET['lo']), layer_icon(False),
                 'mouth $\\Delta\\sigma_0$ < %g kg m$^{-3}$ (mean %.1f\nover the first 3 d); runs last %.0f d'
                 % (SLO, AMAG['lo']['mean3'], AMAG['lo']['run']))], ABS,
               'As (d) and (f), for the fixed-level events of (g): change in the share of the day each layer\'s\n'
               'eddy is present (pp, seasonal cycle removed), largest change in days 0-2 of onset and any\n'
               'opposite change of >= 10 pp in days 3-5. Strongly stratified events fall mostly in Nov-Dec\n'
               'and June, weakly stratified ones mostly in Jan-Feb and Jul-Oct.',
               '(h) how a strongly / weakly stratified mouth changes it, by depth')

fig.suptitle('The inner Penn Cove eddy and how it varies (wb1_t0_xn11abbur00, 2024-25; from the tide-filtered '
             'velocity field)', fontsize=11.5, y=0.905)
fn = out_dir / 'pc_eddy_schematic.png'
fig.savefig(fn, dpi=200, transparent=True, bbox_inches='tight')
plt.close(fig)
print('saved ' + str(fn))
