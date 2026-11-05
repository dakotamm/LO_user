"""
The two leading terms of the TKE budget at the Penn Cove sections (pc_cp, pc_lj, pc_lp),
wb1_t0_xn11abbur00: shear production and buoyancy flux, from the GLS eddy coefficients. (TKE
itself is not written in this run: Hout(idMtke) = F.)

TERMS, hourly, on the interior w-levels of each face (between cell centres k-1 and k, where
AKv / AKs and the shear coincide):
    P  = AKv * S^2        shear production   [W/kg = m2/s3],  S^2 = (du/dz)^2 + (dv/dz)^2 (full shear;
                          u normal to the section, v along it from the 4 v points around the face)
    B  = AKs * N^2        buoyancy flux      [m2/s3], the TKE spent mixing the stratification
                          (N^2 from TEOS-10 sigma0; < 0 where unstable, a TKE source)
    Rf = <B> / <P>        flux Richardson number, a ratio of averages (P -> 0 at slack water);
                          ~0.2 or more = stratification limits the turbulence
In a near-steady boundary layer P ~ B + eps, so P is also a stand-in for the dissipation.
AKv, AKs: GLS closure only (numerical mixing is not in them); hourly snapshots.

COMPOSITES  per element, layer-thickness weighted, over the hours of each condition of
20261104_pc_condition_sections.py and phase (flood / ebb from the section's own Q, all hours).
Profiles: section averages in 1 m bins of depth below the mean surface, each layer split by
overlap of its mean extent.

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc       u, v, salt, temp, z_rho, z_w
       LO_output/extract/[gtagex]/box/pc_cove_mix_[ds0]_[ds1].nc   AKv, AKs
OUTPUT LO_output/DM_outs/20261105_pc_tke_production/
    tke_sections.png          all-hours cross-sections of P, B, Rf (sections west to east)
    tke_profiles_[sect].png   section-averaged profiles: baseline flood / ebb / all, then each condition pair
    tke_production.p          the composites (cached; delete to recompute)

run 20261105_pc_tke_production.py
"""
import pickle
from time import time as clock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import xarray as xr
import cmocean
import gsw

from lo_tools import Lfun
from pc_tidal_cycles import cycle_masks

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SIGN = -1        # section-positive (east) is out of the cove; v is not flipped (+ = north)
FACES = {        # box indices of each section's u-faces (global i - 37, global j - 209), north to south; west to east
    'pc_cp': dict(XI_U=15, ETA=np.arange(13, 5, -1)),
    'pc_lj': dict(XI_U=23, ETA=np.arange(14, 6, -1)),
    'pc_lp': dict(XI_U=30, ETA=np.arange(19, 7, -1)),
}
SECTS = list(FACES)
NAME = {'pc_lp': 'pc_lp (mouth)', 'pc_lj': 'pc_lj (middle)', 'pc_cp': 'pc_cp (Coupeville)'}
PAIRS = [('spring', 'neap'), ('tropic', 'equatorial'), ('high strat', 'low strat'), ('up-cove', 'down-cove')]   # strat over ALL hours
CONDS = ['all'] + [c for p in PAIRS for c in p]
PHASES = ['flood', 'ebb', 'all']
PCOL = {'flood': '#D55E00', 'ebb': '#0072B2', 'all': 'k'}
G, RHO0 = 9.81, 1025.0
LON, LAT = -122.69, 48.23
BLK = 744
EDGES = np.arange(0, 31, 1.0)
DMID = 0.5 * (EDGES[1:] + EDGES[:-1])

Ldir = Lfun.Lstart(gridname='wb1')
box = Ldir['LOo'] / 'extract' / GTAGEX / 'box'
vel_fn, mix_fn = box / ('pc_cove_%s_%s.nc' % (DS0, DS1)), box / ('pc_cove_mix_%s_%s.nc' % (DS0, DS1))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261105_pc_tke_production'
out_dir.mkdir(parents=True, exist_ok=True)
cache_fn = out_dir / 'tke_production.p'

# ------------------------------------------------------------------ hourly P and B, composited per condition and phase (cached)
if cache_fn.exists():
    R = pickle.load(open(cache_fn, 'rb'))
    print('loaded ' + str(cache_fn))
else:
    vb, mx = xr.open_dataset(vel_fn), xr.open_dataset(mix_fn)
    assert np.array_equal(vb.ocean_time.values, mx.ocean_time.values)
    T = vb.ocean_time.values
    CM = cycle_masks(T)
    CM['all'] = np.ones(len(T), bool)
    DY = {SN: (1 / vb.pn.values[F['ETA']][:, [F['XI_U'], F['XI_U'] + 1]]).mean(axis=-1) for SN, F in FACES.items()}
    acc = {(SN, c, ph): dict(P=0, B=0, w=0, zr=0, n=0) for SN in SECTS for c in CONDS for ph in PHASES}
    tt0 = clock()
    for t0 in range(0, len(T), BLK):
        tt = slice(t0, min(t0 + BLK, len(T)))
        V = {vn: vb[vn].isel(ocean_time=tt).to_numpy().astype(float) for vn in ['u', 'v', 'salt', 'temp', 'z_rho', 'z_w']}
        M = {vn: mx[vn].isel(ocean_time=tt).to_numpy().astype(float) for vn in ['AKv', 'AKs']}
        for SN, F in FACES.items():
            XI_U, ETA, dy = F['XI_U'], F['ETA'], DY[SN]
            IR = [XI_U, XI_U + 1]
            face = lambda x: x[:, :, ETA][:, :, :, IR].mean(axis=-1)                 # rho -> u-face, (t, lev, p)
            u = SIGN * V['u'][:, :, ETA, XI_U]
            v = np.nan_to_num(np.concatenate([V['v'][:, :, ETA + dj][:, :, :, IR] for dj in (-1, 0)], axis=-1)).mean(axis=-1)
            zr, zw = face(V['z_rho']), face(V['z_w'])
            Q = (u * np.diff(zw, axis=1) * dy[None, None, :]).sum(axis=(1, 2))
            dzr = np.diff(zr, axis=1)                                                    # (t, N-1, p) layer thickness
            S2 = (np.diff(u, axis=1) / dzr)**2 + (np.diff(v, axis=1) / dzr)**2
            SA = gsw.SA_from_SP(face(V['salt']), -zr, LON, LAT)
            N2 = -(G / RHO0) * np.diff(gsw.sigma0(SA, gsw.CT_from_pt(SA, face(V['temp']))), axis=1) / dzr
            P = face(M['AKv'])[:, 1:-1, :] * S2                                          # interior w-levels 1..N-1
            B = face(M['AKs'])[:, 1:-1, :] * N2
            PH = {'flood': Q > 0, 'ebb': Q < 0, 'all': np.ones(len(Q), bool)}
            for c in CONDS:
                for ph in PHASES:
                    m = PH[ph] & CM[c][tt]
                    a = acc[(SN, c, ph)]
                    a['P'] = a['P'] + (P * dzr)[m].sum(axis=0)
                    a['B'] = a['B'] + (B * dzr)[m].sum(axis=0)
                    a['w'] = a['w'] + dzr[m].sum(axis=0)
                    a['zr'] = a['zr'] + zr[m].sum(axis=0)
                    a['n'] += m.sum()
        print('  hours %d-%d of %d, %.0f s' % (tt.start, tt.stop, len(T), clock() - tt0))
    vb.close(), mx.close()
    R = {'dy': DY}
    for (SN, c, ph), a in acc.items():
        R[(SN, c, ph)] = dict(P=a['P'] / a['w'], B=a['B'] / a['w'], zr=a['zr'] / a['n'], frac=a['n'] / len(T))
    pickle.dump(R, open(cache_fn, 'wb'))
    print('saved ' + str(cache_fn))


def sec_profile(SN, c, ph, k):
    """Section average of term k in 1 m bins of depth below the mean surface (layers between cell centres)."""
    d = R[(SN, c, ph)]
    zr, dy = d['zr'], R['dy'][SN]
    top, bot = -zr[1:], -zr[:-1]                                   # layer extents, depth below z = 0
    out = np.full(len(DMID), np.nan)
    for b in range(len(DMID)):
        ov = np.clip(np.minimum(bot, EDGES[b + 1]) - np.maximum(top, EDGES[b]), 0, None) * dy[None, :]
        if ov.sum() >= 0.25 * dy.mean():
            out[b] = (d[k] * ov).sum() / ov.sum()
    return out


for SN in SECTS:
    d = R[(SN, 'all', 'all')]
    w = np.diff(d['zr'], axis=0) * R['dy'][SN][None, :]
    Pm, Bm = (d['P'] * w).sum() / w.sum(), (d['B'] * w).sum() / w.sum()
    print('%s, all hours: section-mean P %.2e, B %.2e m2/s3, Rf %.2f' % (SN, Pm, Bm, Bm / Pm))

# ------------------------------------------------------------------ figure 1: all-hours cross-sections of P, B, Rf
WR = [R['dy'][SN].sum() for SN in SECTS]
fig, axs = plt.subplots(3, 3, figsize=(14, 9), sharey=True, layout='constrained', gridspec_kw={'width_ratios': WR})
LN = LogNorm(1e-9, 1e-5)
for j, SN in enumerate(SECTS):
    d = R[(SN, 'all', 'all')]
    zr, dy = d['zr'], R['dy'][SN]
    xe = np.concatenate([[0], np.cumsum(dy)]) / 1000
    h = -np.diff(zr[:2], axis=0)[0] / 2 + zr[0]                 # approx bed: half a layer below the lowest centre
    for i, (k, lab) in enumerate([('P', 'shear production P = AKv S$^2$'), ('B', 'buoyancy flux B = AKs N$^2$'), ('Rf', 'flux Richardson number Rf = <B>/<P>')]):
        ax = axs[i, j]
        F = d['B'] / d['P'] if k == 'Rf' else d[k]
        for p in range(len(dy)):
            if k == 'Rf':
                cs = ax.pcolormesh(xe[p:p + 2], zr[:, p], F[:, p:p + 1], cmap=cmocean.cm.amp, vmin=0, vmax=0.5, shading='flat')
            else:
                cs = ax.pcolormesh(xe[p:p + 2], zr[:, p], np.clip(F[:, p:p + 1], 1e-12, None), cmap=cmocean.cm.thermal, norm=LN, shading='flat')
            ax.fill_between(xe[p:p + 2], zr[0, p] - 2, zr[0, p], color='0.8', lw=0)
            ax.fill_between(xe[p:p + 2], zr[-1, p], 0.5, color='0.8', lw=0)
        if k == 'Rf':
            X = np.broadcast_to(0.5 * (xe[1:] + xe[:-1]), F.shape)
            ax.contour(X, 0.5 * (zr[1:] + zr[:-1]), F, levels=[0.2], colors='k', linewidths=1)
            crf = cs
        else:
            cpb = cs
        ax.set_xlim(xe[-1], xe[0])           # reversed: north at right = looking west, into the cove
        ax.set_ylim(-28, 0.5)
        ax.grid(True, alpha=0.3)
        ax.set_title('%s: %s' % (NAME[SN], lab), fontsize=9)
        if j == 0:
            ax.set_ylabel('z [m]')
        if i == 2:
            ax.set_xlabel('distance from north shore [km]')
fig.colorbar(cpb, ax=axs[:2, :].ravel().tolist(), location='right', shrink=0.8, label='P, B [m$^2$ s$^{-3}$ = W kg$^{-1}$], log')
fig.colorbar(crf, ax=axs[2, :].ravel().tolist(), location='right', shrink=0.9, label='Rf (black: 0.2)')
fig.suptitle('TKE shear production and buoyancy flux on the sections, all hours (GLS AKv / AKs, hourly snapshots), west to east, '
             'each looking into the cove (north at right); grey = no resolved layer; %s' % GTAGEX, fontsize=10)
fig.savefig(out_dir / 'tke_sections.png', dpi=200, transparent=True)
plt.close(fig)

# ------------------------------------------------------------------ figure 2: section-averaged profiles, per section
for SN in SECTS:
    fig, axs = plt.subplots(1 + len(PAIRS), 3, figsize=(13, 3.2 * (1 + len(PAIRS))), sharey=True, layout='constrained')
    for i in range(1 + len(PAIRS)):
        curves = ([(c_, ph, PCOL[ph], '-', ph) for c_, ph in [('all', 'flood'), ('all', 'ebb'), ('all', 'all')]] if i == 0 else
                  [(PAIRS[i - 1][0], 'all', 'k', '-', PAIRS[i - 1][0]), (PAIRS[i - 1][1], 'all', 'k', '--', PAIRS[i - 1][1])])
        for j, k in enumerate(['P', 'B', 'Rf']):
            ax = axs[i, j]
            for c_, ph, col, ls, lab in curves:
                x = sec_profile(SN, c_, ph, 'B') / sec_profile(SN, c_, ph, 'P') if k == 'Rf' else sec_profile(SN, c_, ph, k)
                ax.plot(x, DMID, color=col, ls=ls, lw=1.8, label=lab)
            if k == 'Rf':
                ax.axvline(0.2, color='0.4', lw=0.8, ls=':')
                ax.set_xlim(0, 1.0)
            else:
                ax.set_xscale('log')
                ax.set_xlim(1e-9, 1e-5)
            ax.grid(True, alpha=0.3)
            ax.set_title('%s: %s' % ('all hours' if i == 0 else '%s vs %s' % PAIRS[i - 1], {'P': 'shear production', 'B': 'buoyancy flux', 'Rf': 'Rf = <B>/<P>'}[k]), fontsize=9)
            if i == len(PAIRS):
                ax.set_xlabel({'P': 'P [m$^2$ s$^{-3}$]', 'B': 'B [m$^2$ s$^{-3}$]', 'Rf': 'Rf (dotted: 0.2)'}[k], fontsize=8)
        axs[i, 0].legend(fontsize=7, loc='lower left')
        axs[i, 0].set_ylabel('depth below the surface [m]')
    axs[0, 0].invert_yaxis()
    fig.suptitle('%s: section-averaged TKE shear production, buoyancy flux and flux Richardson number (1 m bins), %s' % (NAME[SN], GTAGEX), fontsize=10)
    fig.savefig(out_dir / ('tke_profiles_%s.png' % SN), dpi=170, transparent=True)
    plt.close(fig)
print('saved figures to ' + str(out_dir))
