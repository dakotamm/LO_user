"""
Flood / ebb / residual velocity MAPS of Penn Cove in flow-defined layers, the residual-
to-tidal ratio map, and the centre-line curtain, from 20261110_pc_cove_layer_maps.py
(wb1_t0_xn11abbur00, 2024-2025 hourly snapshots). Step 3 of the whole-cove plan.

COMPOSITES (per cell, transport-weighted as the section scripts): U = sum_t Ux / sum_t H
over the hours of the phase; layers upper (surface to the column's median divide D) and
lower (D to the bed); depth mean = (Ux_up + Ux_lo) / (H_up + H_lo).
Flood / ebb = sign of the hourly mouth Q (pc_lp); residual = all hours.
COLOR = along-cove component, + toward the head (into the cove, red), axis (-0.9789, -0.2043);
ARROWS = the full vector. Flood and ebb share one color / arrow scale, residual its own.

RATIO MAP: |residual vector| / rms tidal speed, tidal = hourly layer velocity (Ux / H)
minus its Godin low-pass, i.e. everything faster than ~2 days (includes the daily S1 wind
band, as u2 does in 20261110_pc_column_u0u1u2.py). The lower layer empties at low water in 38
shallow edge cells (h < D at low tide, mostly < 10% of hours): those hours are bridged by
linear interpolation before filtering and left out of the rms.

CURTAIN: along-cove u on the centre line, head (left) to mouth (right), flood / ebb /
residual, on the phase-mean cell interfaces; dashed line = D below the mean surface;
pc_cp, pc_lj, pc_lp at their u-face columns.

CYCLE (-cycle springneap | tropeq | strat | wind | windhs | windls | cross | stratcalm | all;
the classes of pc_tidal_cycles.py, as the 20261103 section scripts and
20261104_pc_condition_sections.py): every phase split into the high and low half of the
cycle (flood = mouth Q > 0 AND the half, etc.; residual = all hours of the half). Per cycle:
    maps_<cycle>_<layer>.png   one per layer (depth_mean, upper, lower): rows flood | ebb |
                               residual, columns high | low | high - low. High and low share a
                               color / arrow scale per row (flood and ebb one scale); the
                               difference has its own per row.
    curtain_<cycle>.png        the centre-line curtain, same rows and columns, from
                               curtain_hourly.nc
The ratio map is not redrawn. Default (none): unchanged.

INPUT  LO_output/DM_outs/20261110_pc_cove_layer_maps/layers_hourly.nc, curtain.p, curtain_hourly.nc
       LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc (grid only)
OUTPUT LO_output/DM_outs/20261110_pc_cove_layer_maps/
    maps_flood_ebb_residual.png, map_residual_tidal_ratio.png, curtain_flood_ebb_residual.png
    (-cycle) maps_<cycle>_<layer>.png, curtain_<cycle>.png

run 20261110_pc_cove_layer_maps_plot.py [-cycle springneap|tropeq|strat|wind|windhs|windls|cross|stratcalm|all]
"""
import argparse
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cmocean

from lo_tools import Lfun, zfun
from pc_tidal_cycles import cycle_masks, CYCLES

parser = argparse.ArgumentParser()
parser.add_argument('-cycle', default='none', choices=['none', 'all'] + list(CYCLES))   # split each phase by the cycle
CYCLE = parser.parse_args().cycle

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
AXIS = (-0.9789, -0.2043)                 # cove axis, mouth -> head
SECT_XI_U = {'pc_cp': 15, 'pc_lj': 23, 'pc_lp': 30}
SCOL = {'pc_cp': '#CC79A7', 'pc_lj': '#E69F00', 'pc_lp': '#0072B2'}
LAYS = ['depth mean', 'upper', 'lower']
PHS = ['flood', 'ebb', 'residual']

Ldir = Lfun.Lstart(gridname='wb1')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261110_pc_cove_layer_maps'
d = xr.open_dataset(out_dir / 'layers_hourly.nc')
g = xr.open_dataset(Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_' + DS0 + '_' + DS1 + '.nc'))
LON, LAT, MASK = g.lon_rho.values, g.lat_rho.values, g.mask_rho.values == 1
g.close()
JJ, II = d.eta.values, d.xi.values
Qm = d.Qmouth.values

# ------------------------------------------------------------------ hourly layer series (t, 3, cell): depth mean, upper, lower
Ux, Vy, H = d.Ux.values.astype(float), d.Vy.values.astype(float), d.H.values.astype(float)
Ux, Vy, H = [np.concatenate([a.sum(axis=1, keepdims=True), a], axis=1) for a in (Ux, Vy, H)]

# ------------------------------------------------------------------ composites
PM = {'flood': Qm > 0, 'ebb': Qm < 0, 'residual': np.ones(len(Qm), bool)}
COMP = {}
for ph, m in PM.items():
    Hs = H[m].sum(axis=0)
    with np.errstate(invalid='ignore'):
        COMP[ph] = (Ux[m].sum(axis=0) / Hs, Vy[m].sum(axis=0) / Hs)     # (3, cell) each

def to2d(a):
    out = np.full(LON.shape, np.nan)
    out[JJ, II] = a
    return out


def basemap(ax):
    land = np.where(MASK, np.nan, 1.0)
    ax.pcolormesh(LON, LAT, land, cmap='Greys', vmin=0, vmax=2.5, shading='nearest')
    for SN, XI_U in SECT_XI_U.items():
        jw = np.flatnonzero(MASK[:, XI_U] & MASK[:, XI_U + 1])
        lo = 0.5 * (LON[0, XI_U] + LON[0, XI_U + 1])
        ax.plot([lo, lo], [LAT[jw.min(), 0] - 0.0009, LAT[jw.max(), 0] + 0.0009], color=SCOL[SN], lw=1.5)
    ax.set_xlim(LON[0, 0], LON[0, -1])
    ax.set_ylim(LAT[0, 0], LAT[-1, 0])
    ax.set_aspect(1 / np.cos(np.deg2rad(48.23)))
    ax.tick_params(labelsize=6)


al = lambda u, v: u * AXIS[0] + v * AXIS[1]

# ------------------------------------------------------------------ -cycle: maps and curtain split by the high / low half of each cycle
if CYCLE != 'none':
    CM = cycle_masks(d.time.values)
    with open(out_dir / 'curtain.p', 'rb') as f:
        K = pickle.load(f)
    ch = xr.open_dataset(out_dir / 'curtain_hourly.nc')
    UAL, ZWC = ch.ual.values.astype(float), ch.zw.values.astype(float)
    ch.close()
    DZC = np.diff(ZWC, axis=1)
    xe = np.concatenate([[0], np.cumsum(K['dx'])]) / 1000          # km from the head, cell edges
    xc = 0.5 * (xe[1:] + xe[:-1])

    def composite(m):
        Hs = H[m].sum(axis=0)
        with np.errstate(invalid='ignore'):
            return Ux[m].sum(axis=0) / Hs, Vy[m].sum(axis=0) / Hs   # (3, cell) each

    def curtain(m):
        return (UAL[m] * DZC[m]).sum(axis=0) / DZC[m].sum(axis=0), ZWC[m].mean(axis=0)

    def draw_curtain(ax, u, zw, vm, label_sects):
        Xe = np.broadcast_to(xe[None, :], (zw.shape[0], len(xe)))
        Ze = np.concatenate([zw[:, :1], 0.5 * (zw[:, 1:] + zw[:, :-1]), zw[:, -1:]], axis=1)
        cs = ax.pcolormesh(Xe, Ze, 100 * u, cmap='RdBu_r', vmin=-vm, vmax=vm, shading='flat')
        zc = 0.5 * (zw[1:] + zw[:-1])
        if np.nanmin(u) < 0 < np.nanmax(u):
            ax.contour(np.broadcast_to(xc[None, :], zc.shape), zc, u, levels=[0], colors='k', linewidths=0.7)
        ax.plot(xc, zw[-1] - K['D'], 'k--', lw=0.8)
        for SN, XI_U in SECT_XI_U.items():
            ax.axvline(xe[XI_U], color=SCOL[SN], lw=1.2)
            if label_sects:
                ax.text(xe[XI_U], 1.01, SN, color=SCOL[SN], fontsize=7, ha='center', va='bottom', transform=ax.get_xaxis_transform())
        ax.fill_between(np.repeat(xe, 2)[1:-1], np.repeat(zw[0], 2), -30, color='0.6')
        ax.set_ylim(-27, zw[-1].max())
        ax.tick_params(labelsize=7)
        return cs

    for cyc in (list(CYCLES) if CYCLE == 'all' else [CYCLE]):
        hi, lo = CYCLES[cyc]
        R = {}
        for half in (hi, lo):
            for ph, m in PM.items():
                mm = m & CM[half]
                R[half, ph] = dict(c=composite(mm), cur=curtain(mm), n=int(mm.sum()))
        # one scale for flood and ebb, one for residual (both halves); the differences get their own per row
        vm_ph = {'flood': max(np.nanmax(np.abs(al(*R[hf, ph]['c']))) for hf in (hi, lo) for ph in ('flood', 'ebb'))}
        vm_ph['ebb'] = vm_ph['flood']
        vm_ph['residual'] = max(np.nanmax(np.abs(al(*R[hf, 'residual']['c']))) for hf in (hi, lo))

        # ---------------------------------------------------------- maps, one figure per layer
        for i, L in enumerate(LAYS):
            fig, axs = plt.subplots(3, 3, figsize=(16, 10), layout='constrained')
            for r, ph in enumerate(PHS):
                vm = 100 * vm_ph[ph]
                for c, half in enumerate((hi, lo)):
                    ax = axs[r, c]
                    u, v = R[half, ph]['c'][0][i], R[half, ph]['c'][1][i]
                    basemap(ax)
                    cs = ax.pcolormesh(LON, LAT, to2d(100 * al(u, v)), cmap='RdBu_r', vmin=-vm, vmax=vm, shading='nearest')
                    ax.quiver(LON[JJ, II], LAT[JJ, II], u, v, scale=vm / 100 * 12, scale_units='width', width=0.003, color='k')
                    ax.set_title('%s, %s (%d h)' % (ph, half, R[half, ph]['n']), fontsize=9)
                fig.colorbar(cs, ax=axs[r, 1], shrink=0.85, label='along-cove u [cm s$^{-1}$]')
                du = R[hi, ph]['c'][0][i] - R[lo, ph]['c'][0][i]
                dv = R[hi, ph]['c'][1][i] - R[lo, ph]['c'][1][i]
                vd = 100 * np.nanmax(np.abs(al(du, dv)))
                ax = axs[r, 2]
                basemap(ax)
                cd = ax.pcolormesh(LON, LAT, to2d(100 * al(du, dv)), cmap='PuOr_r', vmin=-vd, vmax=vd, shading='nearest')
                ax.quiver(LON[JJ, II], LAT[JJ, II], du, dv, scale=vd / 100 * 12, scale_units='width', width=0.003, color='k')
                ax.set_title('%s: %s minus %s' % (ph, hi, lo), fontsize=9)
                fig.colorbar(cd, ax=ax, shrink=0.85, label='difference [cm s$^{-1}$]\n(own scale)')
            fig.suptitle('Penn Cove %s velocity by %s: %s vs %s, transport-weighted composites, %s, 2024-2025\n'
                         'flood / ebb = sign of mouth Q; color = along-cove component (+ toward the head), arrows = vector; classes of pc_tidal_cycles.py'
                         % (L, cyc, hi, lo, GTAGEX), fontsize=10)
            fn = out_dir / ('maps_%s_%s.png' % (cyc, L.replace(' ', '_')))
            fig.savefig(fn, dpi=200, transparent=True)
            plt.close(fig)

        # ---------------------------------------------------------- curtain
        fig, axs = plt.subplots(3, 3, figsize=(18, 9), sharex=True, sharey=True, layout='constrained')
        for r, ph in enumerate(PHS):
            vm = 100 * max(np.nanmax(np.abs(R[hf, p]['cur'][0])) for hf in (hi, lo)
                           for p in (('flood', 'ebb') if ph != 'residual' else ('residual',)))
            for c, half in enumerate((hi, lo)):
                u, zw = R[half, ph]['cur']
                cs = draw_curtain(axs[r, c], u, zw, vm, r == 0)
                axs[r, c].set_title('%s, %s (%d h)' % (ph, half, R[half, ph]['n']), fontsize=9, loc='left')
            fig.colorbar(cs, ax=axs[r, 1], shrink=0.85, label='along-cove u [cm s$^{-1}$]')
            du = R[hi, ph]['cur'][0] - R[lo, ph]['cur'][0]
            zw = 0.5 * (R[hi, ph]['cur'][1] + R[lo, ph]['cur'][1])
            vd = 100 * np.nanmax(np.abs(du))
            Xe = np.broadcast_to(xe[None, :], (zw.shape[0], len(xe)))
            Ze = np.concatenate([zw[:, :1], 0.5 * (zw[:, 1:] + zw[:, :-1]), zw[:, -1:]], axis=1)
            ax = axs[r, 2]
            cd = ax.pcolormesh(Xe, Ze, 100 * du, cmap='PuOr_r', vmin=-vd, vmax=vd, shading='flat')
            ax.fill_between(np.repeat(xe, 2)[1:-1], np.repeat(zw[0], 2), -30, color='0.6')
            for SN, XI_U in SECT_XI_U.items():
                ax.axvline(xe[XI_U], color=SCOL[SN], lw=1.2)
            ax.tick_params(labelsize=7)
            ax.set_title('%s: %s minus %s' % (ph, hi, lo), fontsize=9, loc='left')
            fig.colorbar(cd, ax=ax, shrink=0.85, label='difference [cm s$^{-1}$]')
        for ax in axs[:, 0]:
            ax.set_ylabel('z [m]', fontsize=8)
        for ax in axs[-1]:
            ax.set_xlabel('km from the head (head left, mouth right)', fontsize=8)
        fig.suptitle('Along-cove velocity on the Penn Cove centre line by %s: %s vs %s, %s, 2024-2025; dashed = divide D'
                     % (cyc, hi, lo, GTAGEX), fontsize=10)
        fig.savefig(out_dir / ('curtain_%s.png' % cyc), dpi=200, transparent=True)
        plt.close(fig)

        # ---------------------------------------------------------- numbers
        for i, L in enumerate(LAYS):
            line = []
            for half in (hi, lo):
                fl = np.median(np.hypot(*[x[i] for x in R[half, 'flood']['c']]))
                eb = np.median(np.hypot(*[x[i] for x in R[half, 'ebb']['c']]))
                rs = np.median(np.hypot(*[x[i] for x in R[half, 'residual']['c']]))
                line.append('%s: flood %.1f ebb %.1f resid %.1f' % (half, 100 * fl, 100 * eb, 100 * rs))
            print('%-10s %-10s %s' % (cyc, L, ' | '.join(line)))
        print('saved maps_%s_*.png, curtain_%s.png' % (cyc, cyc))
    raise SystemExit

# ------------------------------------------------------------------ residual / tidal ratio
with np.errstate(invalid='ignore', divide='ignore'):
    uh, vh = Ux / H, Vy / H
valid = H > 0                                                            # the lower layer empties in 38 shallow
                                                                         # edge cells at low water (mostly < 10% of hours)


def fill(a):
    """linear interpolation in time across the empty-layer hours, so the Godin filter does not spread NaN"""
    sh = a.shape
    return pd.DataFrame(a.reshape(sh[0], -1)).interpolate(limit_direction='both').to_numpy().reshape(sh)


uf, vf = fill(uh), fill(vh)
ut, vt = uf - zfun.lowpass(uf, f='godin'), vf - zfun.lowpass(vf, f='godin')
tid = np.sqrt(np.nanmean(np.where(valid, ut**2 + vt**2, np.nan), axis=0))   # (3, cell), over the hours the layer exists
res = np.hypot(*COMP['residual'])
RATIO = res / tid
for n, L in enumerate(LAYS):
    print('%-10s residual %.1f cm/s, tidal rms %.1f cm/s, ratio median %.2f [%.2f-%.2f]'
          % (L, 100 * np.median(res[n]), 100 * np.median(tid[n]), np.median(RATIO[n]), *np.percentile(RATIO[n], [10, 90])))


# ------------------------------------------------------------------ maps: rows layers, columns flood | ebb | residual
vt_ = 100 * max(np.nanmax(np.abs(al(*COMP[ph]))) for ph in ['flood', 'ebb'])
vr_ = 100 * np.nanmax(np.abs(al(*COMP['residual'])))
fig, axs = plt.subplots(3, 3, figsize=(15, 10), layout='constrained')
for j, ph in enumerate(PHS):
    vm = vr_ if ph == 'residual' else vt_
    for i, L in enumerate(LAYS):
        ax = axs[i, j]
        u, v = COMP[ph][0][i], COMP[ph][1][i]
        basemap(ax)
        cs = ax.pcolormesh(LON, LAT, to2d(100 * al(u, v)), cmap='RdBu_r', vmin=-vm, vmax=vm, shading='nearest')
        ax.quiver(LON[JJ, II], LAT[JJ, II], u, v, scale=vm / 100 * 12, scale_units='width', width=0.003, color='k')
        ax.set_title('%s, %s' % (ph, L), fontsize=9)
    fig.colorbar(cs, ax=axs[:, j], location='bottom', shrink=0.8,
                 label='along-cove u [cm s$^{-1}$], + toward the head' + (' (own scale)' if ph == 'residual' else ''))
fig.suptitle('Penn Cove velocity, transport-weighted composites over all flood / ebb hours (sign of mouth Q) and all hours, %s, 2024-2025\n'
             'layers split at each column\'s median two-layer divide (%.1f m at the head to %.1f m at the mouth, below the moving surface)'
             % (GTAGEX, d.D.values.min(), d.D.values.max()), fontsize=10)
fig.savefig(out_dir / 'maps_flood_ebb_residual.png', dpi=200, transparent=True)
plt.close(fig)

# ------------------------------------------------------------------ ratio map
fig, axs = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
for i, L in enumerate(LAYS):
    ax = axs[i]
    basemap(ax)
    cs = ax.pcolormesh(LON, LAT, to2d(RATIO[i]), cmap=cmocean.cm.amp, vmin=0, vmax=np.nanpercentile(RATIO, 98), shading='nearest')
    ax.set_title('%s: median %.2f' % (L, np.median(RATIO[i])), fontsize=9)
fig.colorbar(cs, ax=axs, shrink=0.8, label='|residual| / rms tidal speed')
fig.suptitle('Residual relative to tidal flow (tidal = hourly minus Godin, incl. the daily wind band), %s, 2024-2025' % GTAGEX, fontsize=10)
fig.savefig(out_dir / 'map_residual_tidal_ratio.png', dpi=200, transparent=True)
plt.close(fig)

# ------------------------------------------------------------------ curtain
with open(out_dir / 'curtain.p', 'rb') as f:
    K = pickle.load(f)
xe = np.concatenate([[0], np.cumsum(K['dx'])]) / 1000          # km from the head, cell edges
xc = 0.5 * (xe[1:] + xe[:-1])
fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True, layout='constrained')
for ax, ph, key in zip(axs, PHS, ['flood', 'ebb', 'all']):
    C = K['C'][key]
    vm = vr_ if ph == 'residual' else vt_
    zw = C['zw']                                                # (w, n) mean interfaces
    Xe = np.broadcast_to(xe[None, :], (zw.shape[0], len(xe)))
    Ze = np.concatenate([zw[:, :1], 0.5 * (zw[:, 1:] + zw[:, :-1]), zw[:, -1:]], axis=1)
    cs = ax.pcolormesh(Xe, Ze, 100 * C['u'], cmap='RdBu_r', vmin=-vm, vmax=vm, shading='flat')
    zc = 0.5 * (zw[1:] + zw[:-1])
    ax.contour(np.broadcast_to(xc[None, :], zc.shape), zc, C['u'], levels=[0], colors='k', linewidths=0.8)
    ax.plot(xc, zw[-1] - K['D'], 'k--', lw=1, label='divide D')
    for SN, XI_U in SECT_XI_U.items():
        ax.axvline(xe[XI_U], color=SCOL[SN], lw=1.5)
        if ph == 'flood':
            ax.text(xe[XI_U], 1.01, SN, color=SCOL[SN], fontsize=8, ha='center', va='bottom', transform=ax.get_xaxis_transform())
    ax.set_ylabel('z [m]', fontsize=8)
    ax.set_title(ph + (' (all hours)' if ph == 'residual' else ' (%d h)' % C['n']), fontsize=9, loc='left')
    ax.fill_between(np.repeat(xe, 2)[1:-1], np.repeat(zw[0], 2), -30, color='0.6')   # bed (transparent=True drops the axes face)
    ax.set_ylim(-27, zw[-1].max())
    fig.colorbar(cs, ax=ax, label='along-cove u [cm s$^{-1}$]\n+ toward the head', shrink=0.9)
axs[0].legend(fontsize=7, loc='lower left')
axs[-1].set_xlabel('distance from the head along the centre line [km]  (head left, mouth right)', fontsize=8)
fig.suptitle('Along-cove velocity on the Penn Cove centre line, transport-weighted composites, %s, 2024-2025' % GTAGEX, fontsize=10)
fig.savefig(out_dir / 'curtain_flood_ebb_residual.png', dpi=200, transparent=True)
plt.close(fig)
print('saved to ' + str(out_dir))
