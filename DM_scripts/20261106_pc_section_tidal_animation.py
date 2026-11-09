"""
Animation of one lunar day at the three Penn Cove sections, pc_cp | pc_lj | pc_lp (west to east,
inner cove to mouth), hourly snapshots: each section's cross-section of section-normal velocity and
its cross-section averaged velocity profile, and the sea level and section-mean ("tidal") velocity
with a cursor at the current hour.

DAY  the baseline lunar day (calm wind, median strat, median spring/neap and tropic/equatorial)
    or one of the one-axis variants, read from 20261105_pc_standard_cycle (lunar_days.csv,
    variants.csv), as in 20261105_pc_tidal_day_schematic.py. Frames run from 3 h before to
    3 h after the lunar day.

PANELS (each frame = one hourly ocean_his snapshot); one COLUMN per section, west to east, each
as wide as the section really is; one colour scale and one profile range for every panel and frame
    section   u normal to the section, + = into the cove (west), drawn BLUE (cmocean balance_r; red =
              leaving); looking OUT of the cove (east), so north is at the LEFT (the user's choice
              for this script, unlike the 20261103 sections); each face drawn between its own interfaces at that hour (the
              surface moves); conventions of 20261103_pc_flood_ebb_sections.py
    profile   (arrows per bin from 0 to U_b, blue = in, red = out; |U_b| < 0.3 cm/s no arrow)
              U_b = Q_b / A_b in 1 m bins of depth below the moving surface: each cell's
              transport split among the bins it overlaps (20261103_pc_section_profiles.py),
              so sum_b Q_b = Q exactly; bins wetted less than half a face (dy x 0.5 m) are
              dropped. Velocity axis REVERSED (+, into the cove, at the left). Grey dashed:
              the lunar-day mean, sum_t Q_b / sum_t A_b over the lunar day.
    salinity, DO  (-kind profiles only) one row each under the velocity profile, same bins,
              area-weighted: S_b = sum_c s_c ov_cb dy / A_b, s on the face = mean of the two rho
              cells either side; grey dashed = lunar-day mean (sum_t S_b A_b / sum_t A_b); one range
              per variable for every panel and frame. DO from the pc_cove_o2 box (same footprint and
              hours), mmol m-3 x 32/1000 = mg L-1.
    strip     sea level (zeta, mean over the rho cells either side of the section's faces;
              on the combined page pc_lj's, the three differ by ~1 cm) and the section-mean
              velocity Ubar = Q / A, + into the cove; cursor at the frame.
LABELS  IN / OUT on each cross-section at the strongest inflow / outflow cell of that hour (only
    where it exceeds UMIN_LAB = 2 cm/s), and at the colour-bar ends; IN / OUT on the profile axes;
    on the strip each flood and ebb within the lunar day is "big" or "little" by comparing the peak
    |Ubar| of the two floods (and of the two ebbs) at the reference section (pc_lj on the combined
    page); tides cut by the window edges are plain "flood" / "ebb". The frame title names the tide.
-sect each (default): one movie per section; pc_cp | pc_lj | pc_lp: that one; all: the three as
    columns on one page.

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc (hourly ocean_his snapshots),
       pc_cove_o2_[ds0]_[ds1].nc (oxygen, same footprint and hours)
OUTPUT LO_output/DM_outs/20261106_pc_section_tidal_animation/
    [sect]_[variant][_profiles].mp4, sect = pc_cp | pc_lj | pc_lp | all
        (or .mov with -vformat prores, which keeps transparency)
    [sect]_[variant]_maxflood.png, _maxebb.png (transparent stills, at max flood / ebb of Ubar;
        on the combined page, of pc_lj)

run 20261106_pc_section_tidal_animation.py [-variant baseline|spring|...] [-sect each|all|pc_cp|pc_lj|pc_lp] [-kind velocity|profiles] [-fps 1.5] [-vformat mp4|prores]
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib import animation
import numpy as np
import pandas as pd
import xarray as xr
import cmocean

from lo_tools import Lfun

parser = argparse.ArgumentParser()
parser.add_argument('-variant', default='baseline')
parser.add_argument('-fps', default=1.5, type=float)
parser.add_argument('-vformat', default='mp4', choices=['mp4', 'prores'])
parser.add_argument('-kind', default='velocity', choices=['velocity', 'profiles'])   # profiles = + salinity and DO profile rows
parser.add_argument('-sect', default='each', choices=['each', 'all', 'pc_cp', 'pc_lj', 'pc_lp'])   # each = one movie per section
args = parser.parse_args()
VARIANT = args.variant.replace('_', ' ')

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SIGN = -1                                 # section-positive (east) is out of the cove
FACES = {                                 # u faces in the pc_cove box, north to south; west to east
    'pc_cp': dict(XI_U=15, ETA=np.arange(13, 5, -1)),   # global i 52, j 222..215
    'pc_lj': dict(XI_U=23, ETA=np.arange(14, 6, -1)),   # global i 60, j 223..216
    'pc_lp': dict(XI_U=30, ETA=np.arange(19, 7, -1)),   # global i 67, j 228..217
}
SECTS = list(FACES)
NAME = {'pc_lp': 'pc_lp (mouth)', 'pc_lj': 'pc_lj (middle)', 'pc_cp': 'pc_cp (Coupeville)'}
COL = {'pc_cp': '#009E73', 'pc_lj': '#CC79A7', 'pc_lp': '#E69F00'}
PAD = pd.Timedelta('3h')                  # frames before and after the lunar day
EDGES = np.arange(0, 31, 1.0)             # profile bins, depth below the surface [m]
DMID = 0.5 * (EDGES[1:] + EDGES[:-1])
COL_IN, COL_OUT = '#2166AC', '#B2182B'   # blue = entering the cove, red = leaving
UMIN_LAB = 2.0                            # cm/s: IN / OUT labels only where the flow is at least this strong
OUTLINE = [pe.withStroke(linewidth=2.5, foreground='k')]   # white text, black edge: readable on red, blue or white

Ldir = Lfun.Lstart(gridname='wb1')
in_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_%s_%s.nc' % (DS0, DS1))
o2_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_o2_%s_%s.nc' % (DS0, DS1))   # same footprint and hours, oxygen
MMOL_TO_MGL = 32 / 1000                   # O2 mmol m-3 -> mg L-1
sel_dir = Ldir['LOo'] / 'DM_outs' / '20261105_pc_standard_cycle'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261106_pc_section_tidal_animation'
out_dir.mkdir(parents=True, exist_ok=True)

if VARIANT == 'baseline':
    day = pd.read_csv(sel_dir / 'lunar_days.csv', parse_dates=['t0', 't1']).iloc[0]
else:
    V = pd.read_csv(sel_dir / 'variants.csv', parse_dates=['t0', 't1'])
    day = V[V.variant == VARIANT].iloc[0]
T0, T1 = day.t0, day.t1
print('%s: lunar day %s to %s UTC' % (VARIANT, T0, T1))

# ------------------------------------------------------------------ each section over the window
NB = len(DMID)


def section(ds, do, sl, XI_U, ETA):
    """Velocity, interfaces, sea level, transport and the 1 m depth-bin profile of one section."""
    u = SIGN * ds.u.isel(xi_u=XI_U, eta_u=ETA, **sl).to_numpy().astype(float)                 # (t, k, p) + into the cove
    zw = ds.z_w.isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1], **sl).to_numpy().mean(axis=-1)    # (t, w, p) at the face
    zeta = ds.zeta.isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1], **sl).to_numpy().mean(axis=(1, 2))
    dy = (1 / ds.pn.isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).to_numpy()).mean(axis=-1)     # (p,) face width [m]
    salt = ds.salt.isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1], **sl).to_numpy().astype(float).mean(axis=-1)   # (t, k, p) on the face
    oxy = MMOL_TO_MGL * do.oxygen.isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1], **sl).to_numpy().astype(float).mean(axis=-1)
    dz = np.diff(zw, axis=1)
    q = u * dz * dy[None, None, :]                                   # (t, k, p) transport per cell
    Q = q.sum(axis=(1, 2))
    A = (dz * dy[None, None, :]).sum(axis=(1, 2))
    # profile: each cell's transport split among the 1 m bins of depth below the surface it overlaps
    dw = zw[:, -1:, :] - zw
    d_top, d_bot = dw[:, 1:, :], dw[:, :-1, :]
    assert d_bot.max() < EDGES[-1]
    Qb, Ab, SAb, OAb = (np.zeros((len(u), NB)) for _ in range(4))
    for b in range(NB):
        ov = np.clip(np.minimum(d_bot, EDGES[b + 1]) - np.maximum(d_top, EDGES[b]), 0, None)   # (t, k, p) m of overlap
        Qb[:, b] = (q * ov / dz).sum(axis=(1, 2))
        Ab[:, b] = (ov * dy[None, None, :]).sum(axis=(1, 2))
        SAb[:, b] = (salt * ov * dy[None, None, :]).sum(axis=(1, 2))   # area-weighted salt
        OAb[:, b] = (oxy * ov * dy[None, None, :]).sum(axis=(1, 2))    # area-weighted oxygen
    assert np.abs(Qb.sum(axis=1) - Q).max() < 1e-6 * np.abs(Q).max()
    amin = 0.5 * dy.mean()                                           # half a face of wetted area
    Ab_day = Ab[iday].mean(axis=0)
    return dict(u=u, zw=zw, zeta=zeta, dy=dy, Q=Q, UBAR=Q / A,
                xe=np.concatenate([[0], np.cumsum(dy)]) / 1000,       # km, face edges from the north shore
                Ub=np.where(Ab >= amin, Qb / np.maximum(Ab, 1e-12), np.nan),
                U_day=np.where(Ab_day >= amin, Qb[iday].sum(axis=0) / np.maximum(Ab[iday].sum(axis=0), 1e-12), np.nan),
                Sb=np.where(Ab >= amin, SAb / np.maximum(Ab, 1e-12), np.nan),
                S_day=np.where(Ab_day >= amin, SAb[iday].sum(axis=0) / np.maximum(Ab[iday].sum(axis=0), 1e-12), np.nan),
                Ob=np.where(Ab >= amin, OAb / np.maximum(Ab, 1e-12), np.nan),
                O_day=np.where(Ab_day >= amin, OAb[iday].sum(axis=0) / np.maximum(Ab[iday].sum(axis=0), 1e-12), np.nan))


ds = xr.open_dataset(in_fn)
do = xr.open_dataset(o2_fn)
assert (do.ocean_time.values == ds.ocean_time.values).all()
TA = pd.DatetimeIndex(ds.ocean_time.values)
it = np.flatnonzero((TA >= T0 - PAD) & (TA <= T1 + PAD))
T = TA[it]
iday = (T >= T0) & (T <= T1)
R = {SN: section(ds, do, dict(ocean_time=slice(it[0], it[-1] + 1)), **F) for SN, F in FACES.items()}
ds.close()
do.close()
print('%d frames' % len(T))
for SN, r in R.items():
    print('  %-6s width %.2f km; Ubar %+.1f to %+.1f cm/s; zeta %+.2f to %+.2f m; lunar-day mean Q %+.0f m3/s'
          % (SN, r['xe'][-1], 100 * r['UBAR'].min(), 100 * r['UBAR'].max(), r['zeta'].min(), r['zeta'].max(), r['Q'][iday].mean()))
print('max sea-level difference between sections %.3f m' % max(np.abs(R[a]['zeta'] - R['pc_lj']['zeta']).max() for a in SECTS))


# ------------------------------------------------------------------ movie
EVEN = ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']
VFMT = {   # h264 has no alpha, so mp4 is rendered on white (a transparent figure would composite onto black)
    'mp4': dict(ext='.mp4', kw=dict(extra_args=EVEN + ['-crf', '18', '-pix_fmt', 'yuv420p']), save=dict(facecolor='white')),
    'prores': dict(ext='.mov', kw=dict(codec='prores_ks', extra_args=['-pix_fmt', 'yuva444p10le', '-profile:v', '4444', '-alpha_bits', '8']),
                   save=dict(transparent=True)),
}


def movie(sects):
    tag = sects[0] if len(sects) == 1 else 'all'
    ref = 'pc_lj' if 'pc_lj' in sects else sects[0]                  # sea level, flood / ebb label, stills
    umax = np.ceil(100 * np.nanpercentile(np.abs(np.concatenate([R[a]['u'].ravel() for a in sects])), 99))   # cm/s
    pmax = np.ceil(100 * max(np.nanmax(np.abs(R[a]['Ub'])) for a in sects))
    zbot = min(R[a]['zw'][:, 0].min() for a in sects) - 1
    ztop = max(R[a]['zw'][:, -1].max() for a in sects) + 0.5
    dmax = max(np.nanmax(np.where(np.isfinite(R[a]['Ub']), DMID, np.nan)) for a in sects) + 1
    widths = [R[a]['xe'][-1] for a in sects]
    PR = args.kind == 'profiles'                                      # salinity and DO rows under the velocity profile
    fig = plt.figure(figsize=(3.2 + 4.2 * sum(widths) / 1.6 if len(sects) > 1 else 9, 17 if PR else 11), layout='constrained')
    gs = fig.add_gridspec(5 if PR else 3, len(sects), width_ratios=widths,
                          height_ratios=[2, 1.7, 1.4, 1.4, 1.1] if PR else [2, 2, 1.1])
    ax_s = [fig.add_subplot(gs[0, c]) for c in range(len(sects))]
    ax_p = [fig.add_subplot(gs[1, c]) for c in range(len(sects))]
    ax_q = [fig.add_subplot(gs[2, c]) for c in range(len(sects))] if PR else []   # salinity profiles
    ax_o = [fig.add_subplot(gs[3, c]) for c in range(len(sects))] if PR else []   # DO profiles
    smin = np.floor(min(np.nanmin(R[a]['Sb']) for a in sects) * 4) / 4
    smax = np.ceil(max(np.nanmax(R[a]['Sb']) for a in sects) * 4) / 4
    omin = np.floor(min(np.nanmin(R[a]['Ob']) for a in sects))
    omax = np.ceil(max(np.nanmax(R[a]['Ob']) for a in sects))
    sm = matplotlib.cm.ScalarMappable(cmap=cmocean.cm.balance_r, norm=matplotlib.colors.Normalize(-umax, umax))
    cb = fig.colorbar(sm, ax=ax_s, location='right', shrink=0.9, pad=0.01, extend='both')
    cb.set_label('u [cm s$^{-1}$]\n+ into the cove (west)')
    cb.ax.text(0.5, 1.07, 'IN', transform=cb.ax.transAxes, ha='center', va='bottom', fontweight='bold', color=COL_IN)
    cb.ax.text(0.5, -0.07, 'OUT', transform=cb.ax.transAxes, ha='center', va='top', fontweight='bold', color=COL_OUT)

    # floods and ebbs of the reference section: sign runs of Ubar; big / little within the lunar day
    ub = R[ref]['UBAR']
    cuts = np.concatenate([[0], np.flatnonzero(np.diff(np.sign(ub)) != 0) + 1, [len(ub)]])
    runs = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        pk = a + np.argmax(np.abs(ub[a:b]))
        runs.append(dict(a=a, b=b, pk=pk, kind='flood' if ub[pk] > 0 else 'ebb', inday=bool(iday[pk])))
    for kind in ['flood', 'ebb']:
        rr = sorted([r_ for r_ in runs if r_['kind'] == kind and r_['inday']], key=lambda r_: -abs(ub[r_['pk']]))
        for r_, size in zip(rr, ['big', 'little'] if len(rr) == 2 else [''] * len(rr)):
            r_['name'] = ('%s %s' % (size, kind)).strip()
    for r_ in runs:
        r_.setdefault('name', r_['kind'])
    TIDE = np.empty(len(ub), object)
    for r_ in runs:
        TIDE[r_['a']:r_['b']] = r_['name']

    # strip: sea level and section-mean velocity over the window, drawn once
    ax_t = fig.add_subplot(gs[-1, :])
    ax_t.axvspan(T0, T1, color=(1, 0.93, 0.8), lw=0, zorder=0)       # opaque pale orange = the lunar day
    for a in sects:
        ax_t.plot(T, 100 * R[a]['UBAR'], color=COL[a] if len(sects) > 1 else 'k', lw=1.5, label='Ubar ' + a)
    ax_t.axhline(0, color='k', lw=0.5)
    ax_t.set_ylabel('section-mean velocity\nUbar = Q / A [cm s$^{-1}$]\n+ into the cove')
    ax_z = ax_t.twinx()
    ax_z.plot(T, R[ref]['zeta'], color='0.35', lw=1, ls='--', label='sea level ' + ref)
    ax_z.set_ylabel('sea level [m]', color='0.35')
    ax_z.tick_params(axis='y', colors='0.35')
    h1, l1 = ax_t.get_legend_handles_labels()
    h2, l2 = ax_z.get_legend_handles_labels()
    ax_t.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left', ncol=4)
    ax_t.set_xlim(T[0], T[-1])
    ax_t.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
    ax_t.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%m-%d'))
    ax_t.set_xlabel('UTC', fontsize=8)
    ax_t.grid(True, alpha=0.3)
    for r_ in runs:
        if r_['inday']:
            y = 100 * ub[r_['pk']]
            ax_t.annotate(r_['name'].upper(), (T[r_['pk']], y), xytext=(0, 10 if y > 0 else -10), textcoords='offset points',
                          ha='center', va='bottom' if y > 0 else 'top', fontsize=9, fontweight='bold',
                          color=COL_IN if y > 0 else COL_OUT)
    ax_t.set_ylim(np.array(ax_t.get_ylim()) * 1.5)                   # room for the labels
    cur = ax_t.axvline(T[0], color='k', lw=1.5)
    dots = [ax_t.plot([], [], 'o', color=COL[a] if len(sects) > 1 else 'k', ms=6)[0] for a in sects]
    dz_ = ax_z.plot([], [], 'o', color='0.35', ms=5)[0]

    def update(n):
        for c, a in enumerate(sects):
            r = R[a]
            ax = ax_s[c]
            ax.cla()
            for p in range(len(r['dy'])):                             # each face between its own interfaces at this hour
                ax.pcolormesh(r['xe'][p:p + 2], r['zw'][n][:, p], 100 * r['u'][n][:, p:p + 1], cmap=sm.cmap, norm=sm.norm, shading='flat')
            un = 100 * r['u'][n]
            for sgn, lab in [(1, 'IN'), (-1, 'OUT')]:                 # at the strongest cell of each sign
                k, p = np.unravel_index(np.nanargmax(sgn * un), un.shape)
                if sgn * un[k, p] >= UMIN_LAB:
                    ax.text(0.5 * (r['xe'][p] + r['xe'][p + 1]), 0.5 * (r['zw'][n][k, p] + r['zw'][n][k + 1, p]), lab,
                            ha='center', va='center', fontsize=12, fontweight='bold', color='w', path_effects=OUTLINE)
            ax.set_xlim(r['xe'][0], r['xe'][-1])                      # north at LEFT = looking east, out of the cove
            ax.set_ylim(zbot, ztop)
            ax.axhline(0, color='0.5', lw=0.5, ls=':')
            ax.set_title('%s\nQ %+.0f m$^3$ s$^{-1}$' % (NAME[a], r['Q'][n]), fontsize=10)
            ax.set_xlabel('distance from north shore [km]  (N at left, looking out of the cove)', fontsize=8)
            if c == 0:
                ax.set_ylabel('z [m]')
            ax = ax_p[c]
            ax.cla()
            ax.plot(100 * r['U_day'], DMID, color='0.5', lw=1.2, ls='--', label='lunar-day mean')
            U = 100 * r['Ub'][n]
            for sgn, colr in [(1, COL_IN), (-1, COL_OUT)]:           # an arrow per bin from 0 to U: blue in, red out
                m = np.isfinite(U) & (sgn * U > 0.3)
                ax.quiver(np.zeros(m.sum()), DMID[m], U[m], np.zeros(m.sum()), color=colr, angles='xy', scale_units='xy',
                          scale=1, width=0.006, headwidth=3.5, headlength=4, headaxislength=3.5, zorder=2)
            ax.plot(U, DMID, color='k', lw=1.5, marker='o', ms=2.5, label='this hour', zorder=3)
            ax.axvline(0, color='k', lw=0.5)
            ax.set_xlim(pmax, -pmax)                                  # reversed: + (into the cove, west) at left
            ax.set_ylim(dmax, 0)
            ax.text(0.03, 0.03, '\u2190 IN (into the cove)', transform=ax.transAxes, ha='left', va='bottom', fontsize=10, fontweight='bold', color=COL_IN)
            ax.text(0.97, 0.03, 'OUT \u2192', transform=ax.transAxes, ha='right', va='bottom', fontsize=10, fontweight='bold', color=COL_OUT)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('U [cm s$^{-1}$]; black = this hour, grey dashed = lunar-day mean', fontsize=8)
            if c == 0:
                ax.set_ylabel('depth below the surface [m]')
            if not PR:
                continue
            for ax, key, lo, hi, xl in [(ax_q[c], 'S', smin, smax, 'salinity [g kg$^{-1}$]'),
                                        (ax_o[c], 'O', omin, omax, 'DO [mg L$^{-1}$]')]:
                ax.cla()
                ax.plot(r[key + '_day'], DMID, color='0.5', lw=1.2, ls='--')
                ax.plot(r[key + 'b'][n], DMID, color='k', lw=1.5, marker='o', ms=2.5)
                ax.set_xlim(lo, hi)
                ax.set_ylim(dmax, 0)
                ax.grid(True, alpha=0.3)
                ax.set_xlabel(xl + '; black = this hour, grey dashed = lunar-day mean', fontsize=8)
                if c == 0:
                    ax.set_ylabel('depth below the surface [m]')
        cur.set_xdata([T[n], T[n]])
        for d, a in zip(dots, sects):
            d.set_data([T[n]], [100 * R[a]['UBAR'][n]])
        dz_.set_data([T[n]], [R[ref]['zeta'][n]])
        fig.suptitle(('%s, %s UTC (%+.0f h from the start of the lunar day): %s at %s\n'
                      'hourly ocean_his snapshot, %s\nprofiles in 1 m bins below the moving surface: U = Q$_b$ / A$_b$'
                      + (', S and DO area-weighted' if PR else ''))
                     % (VARIANT, T[n].strftime('%Y-%m-%d %H:%M'), (T[n] - T0).total_seconds() / 3600,
                        TIDE[n].upper(), ref, GTAGEX), fontsize=11)

    fn = out_dir / ('%s_%s%s' % (tag, VARIANT.replace(' ', '_'), '_profiles' if PR else ''))
    for lbl, n in [('maxflood', np.argmax(R[ref]['UBAR'] * iday)), ('maxebb', np.argmin(R[ref]['UBAR'] * iday))]:
        update(n)
        fig.savefig(str(fn) + '_%s.png' % lbl, dpi=150, transparent=True)
    anim = animation.FuncAnimation(fig, update, frames=len(T), interval=1000 / args.fps, blit=False)
    out = str(fn) + VFMT[args.vformat]['ext']
    anim.save(out, writer=animation.FFMpegWriter(fps=args.fps, **VFMT[args.vformat]['kw']),
              savefig_kwargs=VFMT[args.vformat]['save'], dpi=150)
    plt.close(fig)
    print('saved %s (%d frames at %g fps) + stills' % (out, len(T), args.fps))


for sects in ([[a] for a in SECTS] if args.sect == 'each' else [SECTS] if args.sect == 'all' else [[args.sect]]):
    movie(sects)
