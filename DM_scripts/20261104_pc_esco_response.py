"""
Is ESCO a main driver of the Penn Cove exchange flow? For each face of pc_cp, pc_lj and
pc_lp, the residual velocity profile that each forcing ALONE would drive against the
tidally averaged eddy viscosity <A>, compared with the model's actual residual profile.

THE COLUMN SOLVE (the u1 equation of Dijkstra, Schuttelaars & Burchard 2017, eq. 7-10, in
its steady tidally averaged form, one water column at a time), with u + into the cove:
    d/dz( <A> du/dz + S_F ) = P        S_F = the forcing's own stress, P = a uniform barotropic
                                        pressure gradient fixed by: depth-mean u = 0
    surface: <A> du/dz + S_F = S_F(0)  (zero for ESCO and gravity, the wind stress for wind)
    bed:     u = 0 (no slip)
so  <A> du/dz = S(z) + P z,  integrated up from the bed, with
    ESCO     S = -ESCO(z)                                  (the covariance stress, 20261104_pc_esco.py)
    wind     S = tau_w / rho0                              (uniform: the surface stress carried down)
    gravity  S = int_z^0 F_G dz',  F_G(z) = (g/rho0) int_z^0 rho_x dz'   (baroclinic push INTO the
             cove; rho_x = east-west gradient across the face from the two rho cells either side,
             corrected to constant z: rho_x|z = rho_x|sigma - rho_z z_x|sigma)
Each response has zero depth mean, so it is compared with the actual residual minus its own
column mean: the LAYERED exchange in each column. The LATERAL part of the
exchange, the column means, is not addressed by any of these 1-D balances.

WHAT IS LEFT OUT  advection, Coriolis, lateral stresses, time-dependence: a first-order test
of which vertical forcing can account for the layered exchange, not a momentum budget (ROMS
momentum diagnostics were not written in this run).

INPUTS  esco.p (20261104_pc_esco.py: <A>, ESCO, mean z_rho per face, per condition)
        condition_sections.p (20261104_pc_condition_sections.py: actual residual U per condition)
        tau_cove.npz (pc_forcing_states.py: cove-mean along-cove stress ROMS applied, hourly)
        pc_cove box (salt, temp, z_rho of the rho cells either side of each face, for rho_x)
OUTPUT  LO_output/DM_outs/20261104_pc_esco_response/
    esco_response_profiles.png   actual vs ESCO / wind / gravity / sum, section-averaged, per condition row
    esco_response_summary.png    section rms of each response vs the actual, and pattern correlations
    esco_response.p              all profiles (cached density; delete rho_mean.p to recompute)

run 20261104_pc_esco_response.py
"""
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import gsw

from lo_tools import Lfun
from pc_tidal_cycles import cycle_masks

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
FACES = {
    'pc_cp': dict(XI_U=15, ETA=np.arange(13, 5, -1)),
    'pc_lj': dict(XI_U=23, ETA=np.arange(14, 6, -1)),
    'pc_lp': dict(XI_U=30, ETA=np.arange(19, 7, -1)),
}
SECTS = list(FACES)
NAME = {'pc_lp': 'pc_lp (mouth)', 'pc_lj': 'pc_lj (middle)', 'pc_cp': 'pc_cp (Coupeville)'}
CONDS = ['all', 'up-cove', 'down-cove', 'spring', 'neap', 'high strat', 'low strat']
G, RHO0 = 9.81, 1025.0
LON, LAT = -122.69, 48.23
BLK = 744
FORC = ['esco', 'wind', 'grav']
FLAB = {'esco': 'ESCO', 'wind': 'wind stress', 'grav': 'density gradient', 'sum': 'sum of the three', 'actual': 'model residual'}
FCOL = {'esco': '#D55E00', 'wind': '#009E73', 'grav': '#0072B2', 'sum': '0.5', 'actual': 'k'}

Ldir = Lfun.Lstart(gridname='wb1')
DM = Ldir['LOo'] / 'DM_outs'
box_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_%s_%s.nc' % (DS0, DS1))
out_dir = DM / '20261104_pc_esco_response'
out_dir.mkdir(parents=True, exist_ok=True)


def column_solve(S, A, zr, zwi, h):
    """Solve <A> du/dz = S(z) + P z, u(-h) = 0, depth-mean u = 0, for one face.
    S, A at the interior w-levels zwi (k-1 values); u returned at the cell centres zr (k values)."""
    def integrate(F):                     # u at the cell centres from du/dz = F / A at the w-levels
        g = F / A
        u = np.zeros(len(zr))
        u[0] = g[0] * (zr[0] + h)         # bottom half cell: no slip at -h, gradient of the lowest w-level
        for k in range(1, len(zr)):
            u[k] = u[k - 1] + g[k - 1] * (zr[k] - zr[k - 1])
        return u
    zw = np.concatenate([[-h], zwi, [2 * zr[-1] - zwi[-1]]])   # cell bounds, for the depth mean
    dz = np.diff(zw)
    u_s, u_p = integrate(S), integrate(zwi)                    # forcing part, and the response to P = 1
    P = -(u_s * dz).sum() / (u_p * dz).sum()                   # depth mean zero
    return u_s + P * u_p


# ------------------------------------------------------------------ time-mean sigma0 and z_rho either side of each face, per condition (cached)
rho_fn = out_dir / 'rho_mean.p'
if rho_fn.exists():
    RHO = pickle.load(open(rho_fn, 'rb'))
else:
    ds = xr.open_dataset(box_fn)
    T = ds.ocean_time.values
    CM = cycle_masks(T)
    CM['all'] = np.ones(len(T), bool)
    acc = {(SN, c): dict(sig=0, z=0, n=0) for SN in SECTS for c in CONDS}
    for t0 in range(0, len(T), BLK):
        tt = slice(t0, min(t0 + BLK, len(T)))
        S_b, T_b, Z_b = (ds[v].isel(ocean_time=tt).to_numpy().astype(float) for v in ['salt', 'temp', 'z_rho'])
        for SN, F in FACES.items():
            ix = [F['XI_U'], F['XI_U'] + 1]                    # rho cells west (minus) and east (plus) of the face
            s, t, z = (x[:, :, F['ETA']][:, :, :, ix] for x in (S_b, T_b, Z_b))   # (t, k, p, 2)
            SA = gsw.SA_from_SP(s, -z, LON, LAT)
            sig = gsw.sigma0(SA, gsw.CT_from_pt(SA, t))
            for c in CONDS:
                m = CM[c][tt]
                a = acc[(SN, c)]
                a['sig'] = a['sig'] + sig[m].sum(axis=0)
                a['z'] = a['z'] + z[m].sum(axis=0)
                a['n'] += m.sum()
        print('  hours %d-%d of %d' % (tt.start, tt.stop, len(T)))
    pm = ds.pm.values
    RHO = {'dx': {SN: (1 / pm[F['ETA']][:, [F['XI_U'], F['XI_U'] + 1]]).mean(axis=-1) for SN, F in FACES.items()}}
    ds.close()
    for (SN, c), a in acc.items():
        RHO[(SN, c)] = dict(sig=a['sig'] / a['n'], z=a['z'] / a['n'])    # (k, p, 2): west, east
    pickle.dump(RHO, open(rho_fn, 'wb'))
    print('saved ' + str(rho_fn))


def grav_stress(sig, z, dx, zr, h):
    """S_grav at the interior w-levels: int_z^0 F_G dz', F_G = (g/rho0) int_z^0 rho_x dz' (+ = push into the cove).
    sig, z: (k, p, 2) west/east time-mean sigma0 and z_rho; zr: (k, p) face cell centres; h: (p,) bed depth."""
    rx_s = (sig[..., 1] - sig[..., 0]) / dx[None, :]          # d(rho)/dx along s-levels, x east
    zx_s = (z[..., 1] - z[..., 0]) / dx[None, :]
    sf = sig.mean(axis=-1)
    rz = np.gradient(sf, axis=0) / np.gradient(zr, axis=0)
    rx = rx_s - rz * zx_s                                     # at constant z
    zwi = 0.5 * (zr[1:] + zr[:-1])
    top = 2 * zr[-1] - zwi[-1]                                # approximate mean surface of each face
    dzc = np.diff(np.concatenate([-h[None, :], zwi, top[None, :]], axis=0), axis=0)   # cell thicknesses, same bounds as column_solve
    # F_G at the cell centres: g/rho0 * integral of rho_x from the centre up to the surface
    cum = np.cumsum((rx * dzc)[::-1], axis=0)[::-1]           # integral from the bottom of each cell to the surface
    FG = (G / RHO0) * (cum - 0.5 * rx * dzc)                  # from each cell centre to the surface
    # S at the interior w-levels: integral of F_G from the w-level up to the surface
    cumF = np.cumsum((FG * dzc)[::-1], axis=0)[::-1]          # from each cell bottom to the surface
    return cumF[1:]                                           # w-level k (top of cell k-1) = bottom of cell k


# ------------------------------------------------------------------ solve every column, every condition
E = pickle.load(open(DM / '20261104_pc_esco' / 'esco.p', 'rb'))
CS = pickle.load(open(DM / '20261104_pc_condition_sections' / 'condition_sections.p', 'rb'))
tz = np.load(DM / '20261104_pc_forcing_states' / 'tau_cove.npz')
TAU = pd.Series(tz['tau'], index=pd.DatetimeIndex(tz['t']))      # cove-mean along-cove stress [Pa], + toward the head
CMt = cycle_masks(tz['t'])
CMt['all'] = np.ones(len(TAU), bool)
OUT = {}
for SN in SECTS:
    zr, h, dy = E[SN]['zr'], E[SN]['h'], E[SN]['dy']
    zwi = 0.5 * (zr[1:] + zr[:-1])
    dz = np.diff(np.concatenate([-h[None, :], zwi, (2 * zr[-1] - zwi[-1])[None, :]], axis=0), axis=0)   # (k, p)
    wa = dz * dy[None, :]                                      # cell areas
    for c in CONDS:
        A = E[SN][c]['A']                                      # <A> at the interior w-levels
        S = {'esco': -E[SN][c]['esco'],
             'wind': np.full(A.shape, TAU[CMt[c]].mean() / RHO0),
             'grav': grav_stress(RHO[(SN, c)]['sig'], RHO[(SN, c)]['z'], RHO['dx'][SN], zr, h)}
        r = {f: np.column_stack([column_solve(S[f][:, p], A[:, p], zr[:, p], zwi[:, p], h[p]) for p in range(len(h))]) for f in FORC}
        r['sum'] = r['esco'] + r['wind'] + r['grav']
        U = CS[(SN, c, 'residual')]['U']
        r['actual'] = U - (U * dz).sum(axis=0) / dz.sum(axis=0)    # minus each column's own mean: the layered exchange
        st = {}
        for f in FORC + ['sum']:
            st[f] = dict(rms=np.sqrt((r[f]**2 * wa).sum() / wa.sum()),
                         r=np.sum(r[f] * r['actual'] * wa) / np.sqrt(np.sum(r[f]**2 * wa) * np.sum(r['actual']**2 * wa)))
        st['actual'] = dict(rms=np.sqrt((r['actual']**2 * wa).sum() / wa.sum()))
        OUT[(SN, c)] = dict(r=r, st=st, zr=zr, wa=wa, tau=TAU[CMt[c]].mean())
    s = OUT[(SN, 'all')]['st']
    print('%s, all hours: actual layered exchange rms %.2f cm/s | ESCO %.2f (r %+.2f) | wind %.2f (r %+.2f) | gravity %.2f (r %+.2f) | sum %.2f (r %+.2f)'
          % (SN, 100 * s['actual']['rms'], *(v for f in FORC + ['sum'] for v in (100 * s[f]['rms'], s[f]['r']))))
pickle.dump(OUT, open(out_dir / 'esco_response.p', 'wb'))

# ------------------------------------------------------------------ figure 1: section-averaged profiles, rows = conditions
EDGES = np.arange(0, 31, 1.0)
DMID = 0.5 * (EDGES[1:] + EDGES[:-1])


def binned(x, d, w):
    out = np.full(len(DMID), np.nan)
    for b in range(len(DMID)):
        m = (d >= EDGES[b]) & (d < EDGES[b + 1])
        if w[m].sum() > 0:
            out[b] = (x[m] * w[m]).sum() / w[m].sum()
    return out


fig, axs = plt.subplots(len(CONDS), 3, figsize=(13, 3.2 * len(CONDS)), sharey=True, layout='constrained')
for i, c in enumerate(CONDS):
    for j, SN in enumerate(SECTS):
        o = OUT[(SN, c)]
        ax = axs[i, j]
        for f in ['actual'] + FORC + ['sum']:
            ax.plot(100 * binned(o['r'][f], -o['zr'], o['wa']), DMID, color=FCOL[f], lw=2.5 if f == 'actual' else 1.5,
                    ls='--' if f == 'sum' else '-', label='%s (r %+.2f)' % (FLAB[f], o['st'][f]['r']) if f != 'actual' else FLAB[f])
        ax.axvline(0, color='k', lw=0.6)
        ax.invert_xaxis()                 # west (into the cove) at left
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc='lower left')
        ax.set_title('%s, %s (mean along-cove stress %+.3f Pa)' % (NAME[SN], c, o['tau']), fontsize=9)
        if i == len(CONDS) - 1:
            ax.set_xlabel(r'$\leftarrow$ into the cove (+)   layered exchange [cm s$^{-1}$]   out (-) $\rightarrow$', fontsize=8)
    axs[i, 0].set_ylabel('depth below the surface [m]')
axs[0, 0].invert_yaxis()
fig.suptitle('Which forcing can drive the layered exchange? Response of <A> to ESCO, wind stress and the density gradient alone (1-D, depth-mean zero),\n'
             'vs the model residual minus each column mean; section averages in 1 m depth bins; r = pattern correlation with the model over the section', fontsize=10)
fig.savefig(out_dir / 'esco_response_profiles.png', dpi=170, transparent=True)
plt.close(fig)

# ------------------------------------------------------------------ figure 2: summary, size and shape
fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True, layout='constrained')
x = np.arange(len(CONDS))
for j, SN in enumerate(SECTS):
    for n, f in enumerate(FORC + ['sum']):
        rr = [OUT[(SN, c)]['st'][f]['rms'] / OUT[(SN, c)]['st']['actual']['rms'] for c in CONDS]
        pr = [OUT[(SN, c)]['st'][f]['r'] for c in CONDS]
        axs[0, j].bar(x + (n - 1.5) * 0.2, rr, 0.2, color=FCOL[f], label=FLAB[f])
        axs[1, j].bar(x + (n - 1.5) * 0.2, pr, 0.2, color=FCOL[f], label=FLAB[f])
    axs[0, j].axhline(1, color='k', lw=0.8, ls=':')
    axs[1, j].axhline(0, color='k', lw=0.8)
    axs[0, j].set_title('%s: size (rms response / rms model)' % NAME[SN], fontsize=10)
    axs[1, j].set_title('%s: shape (pattern correlation with the model)' % NAME[SN], fontsize=10)
    axs[1, j].set_xticks(x)
    axs[1, j].set_xticklabels(CONDS, rotation=40, ha='right', fontsize=8)
    axs[1, j].set_ylim(-1, 1)
    for ax in axs[:, j]:
        ax.grid(True, axis='y', alpha=0.3)
axs[0, 0].legend(fontsize=8)
fig.suptitle('Can ESCO, wind or the density gradient account for the model\'s layered exchange? (1-D column responses, %s)' % GTAGEX, fontsize=10)
fig.savefig(out_dir / 'esco_response_summary.png', dpi=200, transparent=True)
plt.close(fig)
print('saved to ' + str(out_dir))
