"""
Animation comparing three lunar days at one Penn Cove section, hour by hour: the BASELINE day
(calm wind, median stratification, median spring/neap and tropic/equatorial) and the two ends of
one forcing axis (e.g. spring | neap), each a column, matched by hours since the start of each
lunar day. Same panels as 20261106_pc_section_tidal_animation.py -kind profiles.

DAYS  baseline = top of 20261105_pc_standard_cycle/lunar_days.csv; each variant = the best row
    of variants.csv whose tides come in the SAME ORDER as the baseline's (which flood and which
    ebb is the bigger: baseline little flood -> big ebb -> big flood -> little ebb), so that the
    same hour is the same point in the diurnal inequality in every column. The neap top pick
    (09-09 11:00) is the same neap one cycle later in the other order, so neap is the runner-up.
    Hand picks (PICK, see below; -pick variant=YYYY.MM.DD_HH adds more): spring 2024-09-02 07:00,
    up-cove wind 2024-01-19 13:00.
MATCHING  frame h = hour h after each day's start (the utide zero up-crossing of the mouth
    transport, i.e. the start of the first flood), from -3 h to 3 h past the longest day.

COLUMNS  baseline | high end | low end of the axis
ROWS     section     u normal to the section, + into the cove, BLUE = entering (balance_r);
                     looking out of the cove, north at LEFT; IN / OUT at the strongest cells
         velocity    U_b = Q_b / A_b in 1 m bins below the moving surface, arrows per bin
                     (blue in, red out), axis reversed (IN at left); grey dashed = the case's
                     lunar-day mean
         salinity    area-weighted in the same bins (s on the face = mean of the two rho cells)
         DO          the same, from the pc_cove_o2 box, mg/L
         strips      section-mean velocity Ubar = Q / A and sea level of all three cases vs
                     hours since the start of the lunar day, cursor at the frame; flood / ebb
                     names of the baseline
One scale per variable across all three columns and frames.

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc, pc_cove_o2_[ds0]_[ds1].nc
OUTPUT LO_output/DM_outs/20261106_pc_case_compare_animation/
    [sect]_[axis].mp4 (or .mov with -vformat prores), [sect]_[axis]_maxflood.png, _maxebb.png
    (transparent stills at the baseline's max flood / ebb hour)

run 20261106_pc_case_compare_animation.py [-axis springneap|tropeq|strat|wind] [-sect pc_lj|pc_cp|pc_lp]
    [-fps 1.5] [-vformat mp4|prores] [-pick spring=2024.09.02_07 up-cove_wind=2024.01.19_13]
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import animation
import numpy as np
import pandas as pd
import xarray as xr
import cmocean

from lo_tools import Lfun

parser = argparse.ArgumentParser()
parser.add_argument('-axis', default='springneap', choices=['springneap', 'tropeq', 'strat', 'wind'])
parser.add_argument('-sect', default='pc_lj', choices=['pc_cp', 'pc_lj', 'pc_lp'])
parser.add_argument('-fps', default=1.5, type=float)
parser.add_argument('-vformat', default='mp4', choices=['mp4', 'prores'])
parser.add_argument('-pick', nargs='*', default=[])   # override days: variant=YYYY.MM.DD_HH, spaces in names as _
args = parser.parse_args()

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SIGN = -1                                 # section-positive (east) is out of the cove
FACES = {                                 # u faces in the pc_cove box, north to south
    'pc_cp': dict(XI_U=15, ETA=np.arange(13, 5, -1)),   # global i 52, j 222..215
    'pc_lj': dict(XI_U=23, ETA=np.arange(14, 6, -1)),   # global i 60, j 223..216
    'pc_lp': dict(XI_U=30, ETA=np.arange(19, 7, -1)),   # global i 67, j 228..217
}
NAME = {'pc_lp': 'pc_lp (mouth)', 'pc_lj': 'pc_lj (Lovejoy, middle)', 'pc_cp': 'pc_cp (Coupeville)'}
AXES = {'springneap': ('spring', 'neap'), 'tropeq': ('tropic', 'equatorial'),
        'strat': ('high strat', 'low strat'), 'wind': ('up-cove wind', 'down-cove wind')}
CASES = ['baseline'] + list(AXES[args.axis])
CCOL = {CASES[0]: 'k', CASES[1]: '#E69F00', CASES[2]: '#CC79A7'}   # case colours on the strips and titles
PAD = 3                                   # hours before / after the lunar day
EDGES = np.arange(0, 31, 1.0)             # profile bins, depth below the surface [m]
DMID = 0.5 * (EDGES[1:] + EDGES[:-1])
NB = len(DMID)
MMOL_TO_MGL = 32 / 1000                   # O2 mmol m-3 -> mg L-1
COL_IN, COL_OUT = '#2166AC', '#B2182B'    # blue = entering the cove, red = leaving
UMIN_LAB = 2.0                            # cm/s: IN / OUT labels only where the flow is at least this strong
OUTLINE = [pe.withStroke(linewidth=2.5, foreground='k')]

Ldir = Lfun.Lstart(gridname='wb1')
box = Ldir['LOo'] / 'extract' / GTAGEX / 'box'
in_fn, o2_fn = box / ('pc_cove_%s_%s.nc' % (DS0, DS1)), box / ('pc_cove_o2_%s_%s.nc' % (DS0, DS1))
sel_dir = Ldir['LOo'] / 'DM_outs' / '20261105_pc_standard_cycle'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261106_pc_case_compare_animation'
out_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ the three days: variants in the baseline's tide order
# PICK: days chosen by hand (2026-11-06) over the automatic best, both in the baseline's tide order:
#   spring  2024-09-02 07:00 (D2 rank 0.72): same season as the baseline and steady (drho range 0.53 over
#           +-1 d) instead of 2025-06-22 (0.76; June, range 1.94, surface 24.9 g/kg)
#   up-cove 2024-01-19 13:00: along and cross stress both 0.02 Pa, instead of 2024-03-28 (cross 0.06 > along 0.02)
PICK = {'spring': '2024-09-02 07:00', 'up-cove wind': '2024-01-19 13:00'}
for kv in args.pick:
    k, v = kv.split('=')
    PICK[k.replace('_', ' ')] = pd.to_datetime(v, format='%Y.%m.%d_%H').strftime('%Y-%m-%d %H:%M')
order = lambda r: (np.sign(r.Qflood2 - r.Qflood1), np.sign(r.Qebb2 - r.Qebb1))
DAY = {'baseline': pd.read_csv(sel_dir / 'lunar_days.csv', parse_dates=['t0', 't1']).iloc[0]}
V = pd.read_csv(sel_dir / 'variants.csv', parse_dates=['t0', 't1'])
for c in CASES[1:]:
    g = V[V.variant == c]
    if c in PICK:
        g = g[g.t0 == pd.Timestamp(PICK[c])]
        assert len(g) == 1, '%s day %s is not in variants.csv' % (c, PICK[c])
        if order(g.iloc[0]) != order(DAY['baseline']):
            print('WARNING: %s day %s is not in the baseline tide order' % (c, PICK[c]))
        DAY[c] = g.iloc[0]
        continue
    same = [order(r) == order(DAY['baseline']) for _, r in g.iterrows()]
    assert any(same), 'no %s day in the baseline tide order' % c
    DAY[c] = g[same].iloc[0]
for c in CASES:
    d = DAY[c]
    print('%-15s %s to %s UTC; ranks D2 %.2f D1 %.2f strat %.2f; floods %.0f, %.0f, ebbs %.0f, %.0f m3/s'
          % (c, d.t0, d.t1, d.r_D2, d.r_D1, d.r_strat, d.Qflood1, d.Qflood2, d.Qebb1, d.Qebb2))
NH = max(int((DAY[c].t1 - DAY[c].t0) / pd.Timedelta('1h')) for c in CASES)
H = np.arange(-PAD, NH + PAD + 1)        # frame = hours since the start of each lunar day


# ------------------------------------------------------------------ one case at the section
def section(ds, do, it, XI_U, ETA, nday):
    """Velocity, interfaces, sea level, transport and the 1 m depth-bin profiles (U, S, DO) at the hours it."""
    sl = dict(ocean_time=slice(it[0], it[-1] + 1))
    u = SIGN * ds.u.isel(xi_u=XI_U, eta_u=ETA, **sl).to_numpy().astype(float)                 # (t, k, p) + into the cove
    zw = ds.z_w.isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1], **sl).to_numpy().mean(axis=-1)    # (t, w, p) at the face
    zeta = ds.zeta.isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1], **sl).to_numpy().mean(axis=(1, 2))
    dy = (1 / ds.pn.isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).to_numpy()).mean(axis=-1)
    salt = ds.salt.isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1], **sl).to_numpy().astype(float).mean(axis=-1)
    oxy = MMOL_TO_MGL * do.oxygen.isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1], **sl).to_numpy().astype(float).mean(axis=-1)
    dz = np.diff(zw, axis=1)
    q = u * dz * dy[None, None, :]
    Q = q.sum(axis=(1, 2))
    A = (dz * dy[None, None, :]).sum(axis=(1, 2))
    dw = zw[:, -1:, :] - zw
    d_top, d_bot = dw[:, 1:, :], dw[:, :-1, :]
    assert d_bot.max() < EDGES[-1]
    Qb, Ab, SAb, OAb = (np.zeros((len(u), NB)) for _ in range(4))
    for b in range(NB):
        ov = np.clip(np.minimum(d_bot, EDGES[b + 1]) - np.maximum(d_top, EDGES[b]), 0, None)   # (t, k, p) m of overlap
        Qb[:, b] = (q * ov / dz).sum(axis=(1, 2))
        Ab[:, b] = (ov * dy[None, None, :]).sum(axis=(1, 2))
        SAb[:, b] = (salt * ov * dy[None, None, :]).sum(axis=(1, 2))
        OAb[:, b] = (oxy * ov * dy[None, None, :]).sum(axis=(1, 2))
    assert np.abs(Qb.sum(axis=1) - Q).max() < 1e-6 * np.abs(Q).max()
    amin = 0.5 * dy.mean()                                           # half a face of wetted area
    iday = nday                                                      # the lunar day's hours within it
    Abd = Ab[iday].mean(axis=0)
    prof = lambda XAb: np.where(Ab >= amin, XAb / np.maximum(Ab, 1e-12), np.nan)
    pday = lambda XAb: np.where(Abd >= amin, XAb[iday].sum(axis=0) / np.maximum(Ab[iday].sum(axis=0), 1e-12), np.nan)
    return dict(u=u, zw=zw, zeta=zeta, dy=dy, Q=Q, UBAR=Q / A, xe=np.concatenate([[0], np.cumsum(dy)]) / 1000,
                Ub=prof(Qb), U_day=pday(Qb), Sb=prof(SAb), S_day=pday(SAb), Ob=prof(OAb), O_day=pday(OAb))


ds, do = xr.open_dataset(in_fn), xr.open_dataset(o2_fn)
assert (do.ocean_time.values == ds.ocean_time.values).all()
TA = pd.DatetimeIndex(ds.ocean_time.values)
R = {}
for c in CASES:
    i0 = TA.get_loc(DAY[c].t0)
    it = i0 + H
    nday = (H >= 0) & (H <= int((DAY[c].t1 - DAY[c].t0) / pd.Timedelta('1h')))
    R[c] = section(ds, do, it, nday=nday, **FACES[args.sect])
    R[c]['T'] = TA[it]
ds.close()
do.close()
for c in CASES:
    r = R[c]
    print('  %-15s Ubar %+.1f to %+.1f cm/s, zeta %+.2f to %+.2f m, S %.2f to %.2f, DO %.1f to %.1f mg/L'
          % (c, 100 * r['UBAR'].min(), 100 * r['UBAR'].max(), r['zeta'].min(), r['zeta'].max(),
             np.nanmin(r['Sb']), np.nanmax(r['Sb']), np.nanmin(r['Ob']), np.nanmax(r['Ob'])))


# ------------------------------------------------------------------ flood / ebb names per case (sign runs of Ubar, big / little in the day)
def tide_names(ub, inday):
    cuts = np.concatenate([[0], np.flatnonzero(np.diff(np.sign(ub)) != 0) + 1, [len(ub)]])
    runs = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        pk = a + np.argmax(np.abs(ub[a:b]))
        runs.append(dict(a=a, b=b, pk=pk, kind='flood' if ub[pk] > 0 else 'ebb', inday=bool(inday[pk])))
    for kind in ['flood', 'ebb']:
        rr = sorted([r_ for r_ in runs if r_['kind'] == kind and r_['inday']], key=lambda r_: -abs(ub[r_['pk']]))
        for r_, size in zip(rr, ['big', 'little'] if len(rr) == 2 else [''] * len(rr)):
            r_['name'] = ('%s %s' % (size, kind)).strip()
    for r_ in runs:
        r_.setdefault('name', r_['kind'])
    names = np.empty(len(ub), object)
    for r_ in runs:
        names[r_['a']:r_['b']] = r_['name']
    return runs, names


for c in CASES:
    nday = (H >= 0) & (H <= int((DAY[c].t1 - DAY[c].t0) / pd.Timedelta('1h')))
    R[c]['runs'], R[c]['tide'] = tide_names(R[c]['UBAR'], nday)

# ------------------------------------------------------------------ movie
EVEN = ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']
VFMT = {   # h264 has no alpha, so mp4 is rendered on white (a transparent figure would composite onto black)
    'mp4': dict(ext='.mp4', kw=dict(extra_args=EVEN + ['-crf', '18', '-pix_fmt', 'yuv420p']), save=dict(facecolor='white')),
    'prores': dict(ext='.mov', kw=dict(codec='prores_ks', extra_args=['-pix_fmt', 'yuva444p10le', '-profile:v', '4444', '-alpha_bits', '8']),
                   save=dict(transparent=True)),
}
cat_ = lambda key: np.concatenate([R[c][key].ravel() for c in CASES])
umax = np.ceil(100 * np.nanpercentile(np.abs(cat_('u')), 99))         # one scale per variable for all three columns
pmax = np.ceil(100 * np.nanmax(np.abs(cat_('Ub'))))
smin, smax = np.floor(np.nanmin(cat_('Sb')) * 4) / 4, np.ceil(np.nanmax(cat_('Sb')) * 4) / 4
omin, omax = np.floor(np.nanmin(cat_('Ob'))), np.ceil(np.nanmax(cat_('Ob')))
zbot = min(R[c]['zw'][:, 0].min() for c in CASES) - 1
ztop = max(R[c]['zw'][:, -1].max() for c in CASES) + 0.5
dmax = max(np.nanmax(np.where(np.isfinite(R[c]['Ub']), DMID, np.nan)) for c in CASES) + 1

fig = plt.figure(figsize=(18, 19), layout='constrained')
gs = fig.add_gridspec(6, 3, height_ratios=[2, 1.7, 1.4, 1.4, 0.9, 0.9])
ax_s = [fig.add_subplot(gs[0, k]) for k in range(3)]
ax_p = [fig.add_subplot(gs[1, k]) for k in range(3)]
ax_q = [fig.add_subplot(gs[2, k]) for k in range(3)]
ax_o = [fig.add_subplot(gs[3, k]) for k in range(3)]
sm = matplotlib.cm.ScalarMappable(cmap=cmocean.cm.balance_r, norm=matplotlib.colors.Normalize(-umax, umax))
cb = fig.colorbar(sm, ax=ax_s, location='right', shrink=0.9, pad=0.01, extend='both')
cb.set_label('u [cm s$^{-1}$]\n+ into the cove (west)')
cb.ax.text(0.5, 1.07, 'IN', transform=cb.ax.transAxes, ha='center', va='bottom', fontweight='bold', color=COL_IN)
cb.ax.text(0.5, -0.07, 'OUT', transform=cb.ax.transAxes, ha='center', va='top', fontweight='bold', color=COL_OUT)

# strips: Ubar and sea level of the three cases vs hours since the start of the lunar day, drawn once
ax_t = fig.add_subplot(gs[4, :])
ax_z = fig.add_subplot(gs[5, :], sharex=ax_t)
ax_t.axvspan(0, NH, color=(1, 0.93, 0.8), lw=0, zorder=0)            # opaque pale orange = the lunar day
ax_z.axvspan(0, NH, color=(1, 0.93, 0.8), lw=0, zorder=0)
for c in CASES:
    ax_t.plot(H, 100 * R[c]['UBAR'], color=CCOL[c], lw=1.8 if c == 'baseline' else 1.3,
              label='%s (%s UTC)' % (c, DAY[c].t0.strftime('%Y-%m-%d %H:%M')))
    ax_z.plot(H, R[c]['zeta'], color=CCOL[c], lw=1.8 if c == 'baseline' else 1.3, ls='--')
for r_ in R['baseline']['runs']:                                    # the baseline's tide names (same order in every case)
    if r_['inday']:
        y = 100 * R['baseline']['UBAR'][r_['pk']]
        ax_t.annotate(r_['name'].upper(), (H[r_['pk']], y), xytext=(0, 10 if y > 0 else -10), textcoords='offset points',
                      ha='center', va='bottom' if y > 0 else 'top', fontsize=9, fontweight='bold', color=COL_IN if y > 0 else COL_OUT)
ax_t.set_ylim(np.array(ax_t.get_ylim()) * 1.5)
ax_t.axhline(0, color='k', lw=0.5)
ax_t.set_ylabel('Ubar = Q / A [cm s$^{-1}$]\n+ into the cove')
ax_t.legend(fontsize=8, loc='upper left', ncol=3)
ax_z.set_ylabel('sea level [m]')
ax_z.set_xlabel('hours since the start of each lunar day (start of the first flood)')
for a_ in (ax_t, ax_z):
    a_.set_xlim(H[0], H[-1])
    a_.set_xticks(np.arange(0, H[-1] + 1, 3))
    a_.grid(True, alpha=0.3)
cur = [a_.axvline(H[0], color='k', lw=1.5) for a_ in (ax_t, ax_z)]
dots = [(ax_t.plot([], [], 'o', color=CCOL[c], ms=6)[0], ax_z.plot([], [], 'o', color=CCOL[c], ms=5)[0]) for c in CASES]


def prof_panel(ax, k, x, xday, lo, hi, xl, reverse=False):
    ax.cla()
    ax.plot(xday, DMID, color='0.5', lw=1.2, ls='--')
    ax.plot(x, DMID, color='k', lw=1.5, marker='o', ms=2.5, zorder=3)
    ax.set_xlim((hi, lo) if reverse else (lo, hi))
    ax.set_ylim(dmax, 0)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(xl + '; black = this hour, grey dashed = lunar-day mean', fontsize=8)
    if k == 0:
        ax.set_ylabel('depth below the surface [m]')
    else:
        ax.tick_params(labelleft=False)


def update(n):
    for k, c in enumerate(CASES):
        r = R[c]
        ax = ax_s[k]
        ax.cla()
        for p in range(len(r['dy'])):                                # each face between its own interfaces at this hour
            ax.pcolormesh(r['xe'][p:p + 2], r['zw'][n][:, p], 100 * r['u'][n][:, p:p + 1], cmap=sm.cmap, norm=sm.norm, shading='flat')
        un = 100 * r['u'][n]
        for sgn, lab in [(1, 'IN'), (-1, 'OUT')]:                    # at the strongest cell of each sign
            kk, p = np.unravel_index(np.nanargmax(sgn * un), un.shape)
            if sgn * un[kk, p] >= UMIN_LAB:
                ax.text(0.5 * (r['xe'][p] + r['xe'][p + 1]), 0.5 * (r['zw'][n][kk, p] + r['zw'][n][kk + 1, p]), lab,
                        ha='center', va='center', fontsize=12, fontweight='bold', color='w', path_effects=OUTLINE)
        ax.set_xlim(r['xe'][0], r['xe'][-1])                         # north at LEFT = looking east, out of the cove
        ax.set_ylim(zbot, ztop)
        ax.axhline(0, color='0.5', lw=0.5, ls=':')
        d = DAY[c]
        ax.set_title('%s: %s\n%s UTC; ranks D2 %.2f, D1 %.2f, strat %.2f\nQ %+.0f m$^3$ s$^{-1}$'
                     % (c.upper(), r['tide'][n].upper(), r['T'][n].strftime('%Y-%m-%d %H:%M'), d.r_D2, d.r_D1, d.r_strat, r['Q'][n]),
                     fontsize=10, color=CCOL[c] if c != 'baseline' else 'k')
        ax.set_xlabel('distance from north shore [km]  (N at left, looking out of the cove)', fontsize=8)
        if k == 0:
            ax.set_ylabel('z [m]')
        else:
            ax.tick_params(labelleft=False)
        # velocity profile with an arrow per bin
        U = 100 * r['Ub'][n]
        prof_panel(ax_p[k], k, U, 100 * r['U_day'], -pmax, pmax, 'U [cm s$^{-1}$]', reverse=True)
        for sgn, colr in [(1, COL_IN), (-1, COL_OUT)]:
            m = np.isfinite(U) & (sgn * U > 0.3)
            ax_p[k].quiver(np.zeros(m.sum()), DMID[m], U[m], np.zeros(m.sum()), color=colr, angles='xy', scale_units='xy',
                           scale=1, width=0.006, headwidth=3.5, headlength=4, headaxislength=3.5, zorder=2)
        ax_p[k].axvline(0, color='k', lw=0.5)
        ax_p[k].text(0.03, 0.03, '← IN', transform=ax_p[k].transAxes, ha='left', va='bottom', fontsize=10, fontweight='bold', color=COL_IN)
        ax_p[k].text(0.97, 0.03, 'OUT →', transform=ax_p[k].transAxes, ha='right', va='bottom', fontsize=10, fontweight='bold', color=COL_OUT)
        prof_panel(ax_q[k], k, r['Sb'][n], r['S_day'], smin, smax, 'salinity [g kg$^{-1}$]')
        prof_panel(ax_o[k], k, r['Ob'][n], r['O_day'], omin, omax, 'DO [mg L$^{-1}$]')
        dots[k][0].set_data([H[n]], [100 * r['UBAR'][n]])
        dots[k][1].set_data([H[n]], [r['zeta'][n]])
    for cc in cur:
        cc.set_xdata([H[n], H[n]])
    fig.suptitle('%s, %s vs the baseline, %+d h from the start of each lunar day\n'
                 'hourly ocean_his snapshots, %s; profiles in 1 m bins below the moving surface: U = Q$_b$ / A$_b$, S and DO area-weighted'
                 % (NAME[args.sect], ' | '.join(CASES[1:]), H[n], GTAGEX), fontsize=12)


fn = out_dir / ('%s_%s' % (args.sect, args.axis))
ub = R['baseline']['UBAR']
for lbl, n in [('maxflood', np.argmax(ub * (H >= 0))), ('maxebb', np.argmin(ub * (H >= 0)))]:
    update(n)
    fig.savefig(str(fn) + '_%s.png' % lbl, dpi=120, transparent=True)
anim = animation.FuncAnimation(fig, update, frames=len(H), interval=1000 / args.fps, blit=False)
out = str(fn) + VFMT[args.vformat]['ext']
anim.save(out, writer=animation.FFMpegWriter(fps=args.fps, **VFMT[args.vformat]['kw']), savefig_kwargs=VFMT[args.vformat]['save'], dpi=120)
plt.close(fig)
print('saved %s (%d frames at %g fps) + stills' % (out, len(H), args.fps))
