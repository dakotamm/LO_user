"""
How the plan-view RESIDUAL circulation of Penn Cove changes with the conditions of
pc_tidal_cycles.py, all on one page per layer, from the hourly layer transports of
20261110_pc_cove_layer_maps.py (wb1_t0_xn11abbur00, 2024-2025 hourly snapshots).

RESIDUAL per condition = sum_t Ux / sum_t H over all hours of the condition (transport-
weighted, as 20261110_pc_cove_layer_maps_plot.py); layers depth mean | upper | lower
(split at each column's median two-layer divide).

METRICS per condition and layer
    rms     area-weighted rms residual speed over the cove [cm/s]
    lobe    mean residual relative vorticity dv/dx - du/dy in the 500 m disk centred at
            (-122.708, 48.226) (the lobe cell of 20261027_pc_lobe_cavity.py), in units of f;
            + = counterclockwise. Derivatives by centred differences on the rho grid, cells
            next to land use the one-sided neighbour (np.gradient on NaN-free interior).
    r_all   vector pattern correlation with the all-hours residual,
            sum (u u_all + v v_all) dA / sqrt(sum |u|^2 dA * sum |u_all|^2 dA):
            1 = same pattern (any strength), 0 = unrelated

FIGURES (LO_output/DM_outs/20261110_pc_cove_residual_conditions/)
    residual_conditions_<layer>.png   one map per condition, ONE color / arrow scale on the
        page; color = along-cove component (+ toward the head), arrows = vector
    residual_conditions.csv           the metrics

run 20261110_pc_cove_residual_conditions.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun
from pc_tidal_cycles import cycle_masks

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
AXIS = (-0.9789, -0.2043)                 # cove axis, mouth -> head
SECT_XI_U = {'pc_cp': 15, 'pc_lj': 23, 'pc_lp': 30}
SCOL = {'pc_cp': '#CC79A7', 'pc_lj': '#E69F00', 'pc_lp': '#0072B2'}
LAYS = ['depth mean', 'upper', 'lower']
LOBE = (-122.708, 48.226, 500.0)          # lon, lat, radius [m]
# page layout: rows of conditions (None = empty panel)
GRID = [['all', 'calm', None, None],
        ['spring', 'neap', 'tropic', 'equatorial'],
        ['high strat', 'low strat', 'high strat, calm', 'low strat, calm'],
        ['up-cove', 'down-cove', 'northward', 'southward'],
        ['up-cove, high strat', 'down-cove, high strat', 'up-cove, low strat', 'down-cove, low strat']]

Ldir = Lfun.Lstart(gridname='wb1')
in_dir = Ldir['LOo'] / 'DM_outs' / '20261110_pc_cove_layer_maps'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261110_pc_cove_residual_conditions'
out_dir.mkdir(parents=True, exist_ok=True)

d = xr.open_dataset(in_dir / 'layers_hourly.nc')
g = xr.open_dataset(Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_' + DS0 + '_' + DS1 + '.nc'))
LON, LAT, MASK = g.lon_rho.values, g.lat_rho.values, g.mask_rho.values == 1
PM, PN, F = g.pm.values, g.pn.values, g.f.values
g.close()
JJ, II = d.eta.values, d.xi.values
dA = 1 / (PM[JJ, II] * PN[JJ, II])
Ux, Vy, H = d.Ux.values.astype(float), d.Vy.values.astype(float), d.H.values.astype(float)
Ux, Vy, H = [np.concatenate([a.sum(axis=1, keepdims=True), a], axis=1) for a in (Ux, Vy, H)]   # depth mean, upper, lower

CM = cycle_masks(d.time.values)
CM['all'] = np.ones(d.time.size, bool)
CONDS = [c for row in GRID for c in row if c]

# lobe disk
km = 111.32
rr = np.hypot((LON[JJ, II] - LOBE[0]) * km * np.cos(np.deg2rad(LOBE[1])), (LAT[JJ, II] - LOBE[1]) * km) * 1000
DISK = rr <= LOBE[2]
print('lobe disk: %d cells' % DISK.sum())


def to2d(a):
    out = np.full(LON.shape, np.nan)
    out[JJ, II] = a
    return out


def vorticity(u, v):
    """dv/dx - du/dy at the cove cells, centred differences on the rho grid (one-sided next to land)"""
    U, V = to2d(u), to2d(v)
    dx, dy = 1 / PM, 1 / PN
    dvdx = np.full(U.shape, np.nan)
    dudy = np.full(U.shape, np.nan)
    for j, i in zip(JJ, II):
        il, ir = (i - 1 if np.isfinite(V[j, i - 1]) else i), (i + 1 if np.isfinite(V[j, i + 1]) else i)
        jl, jr = (j - 1 if np.isfinite(U[j - 1, i]) else j), (j + 1 if np.isfinite(U[j + 1, i]) else j)
        if ir > il:
            dvdx[j, i] = (V[j, ir] - V[j, il]) / ((ir - il) * dx[j, i])
        if jr > jl:
            dudy[j, i] = (U[jr, i] - U[jl, i]) / ((jr - jl) * dy[j, i])
    return (dvdx - dudy)[JJ, II]


def basemap(ax):
    ax.pcolormesh(LON, LAT, np.where(MASK, np.nan, 1.0), cmap='Greys', vmin=0, vmax=2.5, shading='nearest')
    for SN, XI_U in SECT_XI_U.items():
        jw = np.flatnonzero(MASK[:, XI_U] & MASK[:, XI_U + 1])
        lo = 0.5 * (LON[0, XI_U] + LON[0, XI_U + 1])
        ax.plot([lo, lo], [LAT[jw.min(), 0] - 0.0009, LAT[jw.max(), 0] + 0.0009], color=SCOL[SN], lw=1.2)
    th = np.linspace(0, 2 * np.pi, 60)
    ax.plot(LOBE[0] + LOBE[2] / 1000 * np.cos(th) / (km * np.cos(np.deg2rad(LOBE[1]))),
            LOBE[1] + LOBE[2] / 1000 * np.sin(th) / km, color='0.3', lw=0.8, ls=':')
    ax.set_xlim(LON[0, 0], LON[0, -1])
    ax.set_ylim(LAT[0, 0], LAT[-1, 0])
    ax.set_aspect(1 / np.cos(np.deg2rad(48.23)))
    ax.set_xticks([])
    ax.set_yticks([])


# ------------------------------------------------------------------ residual per condition, and metrics
RES, rows = {}, []
for c in CONDS:
    m = CM[c]
    Hs = H[m].sum(axis=0)
    u, v = Ux[m].sum(axis=0) / Hs, Vy[m].sum(axis=0) / Hs           # (3, cell)
    RES[c] = (u, v)
f0 = F[JJ, II][DISK].mean()
for c in CONDS:
    for L, lay in enumerate(LAYS):
        u, v = RES[c][0][L], RES[c][1][L]
        ua, va = RES['all'][0][L], RES['all'][1][L]
        ok = np.isfinite(u) & np.isfinite(ua)
        w = dA[ok]
        rms = np.sqrt(((u[ok]**2 + v[ok]**2) * w).sum() / w.sum())
        r = ((u[ok] * ua[ok] + v[ok] * va[ok]) * w).sum() / np.sqrt(((u[ok]**2 + v[ok]**2) * w).sum() * ((ua[ok]**2 + va[ok]**2) * w).sum())
        z = vorticity(u, v)
        rows.append(dict(condition=c, layer=lay, hours=int(CM[c].sum()), rms_cm_s=100 * rms,
                         lobe_vort_f=np.nanmean(z[DISK]) / f0, r_all=r))
M = pd.DataFrame(rows)
M.to_csv(out_dir / 'residual_conditions.csv', index=False, float_format='%.3f')
pd.set_option('display.width', 200)
print(M.pivot(index='condition', columns='layer', values=['rms_cm_s', 'lobe_vort_f', 'r_all']).loc[CONDS].round(2).to_string())

# ------------------------------------------------------------------ one page per layer
al = lambda u, v: u * AXIS[0] + v * AXIS[1]
for L, lay in enumerate(LAYS):
    vm = 100 * np.nanpercentile(np.abs(np.concatenate([al(RES[c][0][L], RES[c][1][L]) for c in CONDS])), 99)
    fig, axs = plt.subplots(len(GRID), 4, figsize=(16, 15), layout='constrained')
    for r_, row in enumerate(GRID):
        for c_, c in enumerate(row):
            ax = axs[r_, c_]
            if c is None:
                ax.axis('off')
                continue
            u, v = RES[c][0][L], RES[c][1][L]
            basemap(ax)
            cs = ax.pcolormesh(LON, LAT, to2d(100 * al(u, v)), cmap='RdBu_r', vmin=-vm, vmax=vm, shading='nearest')
            ax.quiver(LON[JJ, II], LAT[JJ, II], u, v, scale=vm / 100 * 10, scale_units='width', width=0.004, color='k')
            mm = M[(M.condition == c) & (M.layer == lay)].iloc[0]
            ax.set_title('%s (%d h)\nrms %.1f cm/s, lobe %+.2f f, r %.2f' % (c, mm.hours, mm.rms_cm_s, mm.lobe_vort_f, mm.r_all), fontsize=8)
    fig.colorbar(cs, ax=axs, shrink=0.4, location='right', label='residual along-cove u [cm s$^{-1}$], + toward the head (one scale on the page)')
    fig.suptitle('Penn Cove residual circulation, %s layer, by condition (pc_tidal_cycles.py classes), %s, 2024-2025\n'
                 'lobe = mean vorticity in the dotted 500 m disk / f (+ counterclockwise); r = vector pattern correlation with the all-hours residual'
                 % (lay, GTAGEX), fontsize=10)
    fig.savefig(out_dir / ('residual_conditions_%s.png' % lay.replace(' ', '_')), dpi=200, transparent=True)
    plt.close(fig)
    print('saved residual_conditions_%s.png' % lay.replace(' ', '_'))
