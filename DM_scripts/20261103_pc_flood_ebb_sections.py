"""
Average flood, average ebb, and residual velocity across the three Penn Cove
sections (pc_lp = mouth, pc_lj = middle, pc_cp = off Coupeville), plotted as
cross-sections looking into the cove, i.e. west, so north is at the right
(x = distance from the north shore, increasing to the left).

Input (SRC switch), wb1_t0_xn11abbur00, 2024.01.01-2025.12.31, hourly:
  SRC = 'his': ocean_his SNAPSHOTS on the hour, from the pc_cove box. Each face is
    a ROMS u-point between rho cells (i, i+1): u at the face, DZ = mean Hz of the
    two cells, dd = mean 1/pn, q = u*dd*DZ (as in 20261013_maccready2011_sec2.py).
    Snapshots alias sub-hourly motion into the subtidal band (worst for residual / u1).
  SRC = 'avg': tef2 extractions_avg, hour-mean Huon q and DZ per face, stamped hh:30.

Method (per section):
  u_in = -q / (dd * DZ)       sign flipped so + is INTO the cove (westward)
  flood = hours when the section-total Q_in > 0, ebb = hours when Q_in < 0
  U_phase(z, p) = average of u_in over the hours in that phase, either
    SRC = 'his'         # 'his': hourly snapshots (pc_cove box) | 'avg': hour-mean Huon (tef2 extractions_avg)
AVG = 'transport': sum_t q_in / sum_t (dd * DZ)   (integrates to the true mean Q)
    AVG = 'eulerian':  mean_t of q_in / (dd * DZ)     (plain time mean of velocity)
  residual = the same over ALL hours (2-year mean)
Each composite is plotted on its own phase-mean sigma-level depths.

  tidal flood / ebb = flood / ebb composite minus the residual (the oscillation alone)

Output: LO_output/DM_outs/20261103_pc_flood_ebb_sections/  (each name gets _his or _avg before .png)
  pc_flood_ebb_sections.png        full flood, full ebb, residual
  pc_tidal_flood_ebb_sections.png  tidal flood, tidal ebb, residual
  pc_max_flood_ebb_sections.png    max flood, max ebb, residual (max = the peak-|Q_in| hour of
                                   each flood / ebb, averaged over all of them; not minus residual)
  (the Eulerian u0 / u1 / u2 decomposition is in 20261103_pc_eulerian_u0u1u2.py)

  python 20261103_pc_flood_ebb_sections.py
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cmocean

from lo_tools import Lfun

GTX, DS0, DS1 = 'wb1_t0_xn11abbur00', '2024.01.01', '2025.12.31'
SRC = 'his'         # 'his': hourly snapshots (pc_cove box) | 'avg': hour-mean Huon (tef2 extractions_avg)
AVG = 'transport'   # 'transport': sum(q)/sum(A) per cell | 'eulerian': mean(q/A) per cell
SECTS = ['pc_cp', 'pc_lj', 'pc_lp']   # west to east: the column order of every figure
NAME = {'pc_lp': 'pc_lp (mouth)', 'pc_lj': 'pc_lj (middle)', 'pc_cp': 'pc_cp (Coupeville)'}
Ldir = Lfun.Lstart(gridname='wb1')
in_dir = Ldir['LOo'] / 'extract' / GTX / 'tef2' / ('extractions_avg_%s_%s' % (DS0, DS1))
box_fn = Ldir['LOo'] / 'extract' / GTX / 'box' / ('pc_cove_%s_%s.nc' % (DS0, DS1))
sect_df = pd.read_pickle(Ldir['LOo'] / 'extract' / 'tef2' / 'sect_df_wb1_pc1.p')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261103_pc_flood_ebb_sections'
Lfun.make_dir(out_dir)
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)

def peak_hours(mask, Q, nmin=3):
    # one hour per contiguous run of mask (one flood or one ebb): the hour of largest |Q|; runs shorter than nmin h are slack flicker, skipped
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    out = np.zeros_like(mask)
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= nmin:
            out[a + np.argmax(np.abs(Q[a:b]))] = True
    return out

def load_his(sect):
    # section u-faces from the pc_cove box of ocean_his snapshots; returns the same fields as the tef2 extraction, p = 0 at the north end
    bx = xr.open_dataset(box_fn)
    G = Ldir['data'] / 'grids' / 'wb1' / 'grid.nc'
    glon = xr.open_dataset(G).lon_rho.values
    glat = xr.open_dataset(G).lat_rho.values
    J0, I0 = np.unravel_index(np.argmin((glon - bx.lon_rho.values[0, 0]) ** 2 + (glat - bx.lat_rho.values[0, 0]) ** 2), glon.shape)   # box origin in the global grid
    sd = sect_df[sect_df.sn == sect]
    assert (sd.uv == 'u').all() and (sd.i.nunique() == 1)
    i = sd.i.values[0] - I0                              # box xi_u of the faces; rho neighbours are xi_rho i, i+1
    jj = sd.j.values - J0                                # box eta, north to south
    js = slice(jj.min(), jj.max() + 1)
    assert (np.diff(jj) == -1).all()                     # sect_df runs north to south, one row at a time
    rev = slice(None, None, -1)                          # the box slab runs south to north; flip it to match p
    u = bx.u.isel(eta_u=js, xi_u=i).values.astype(float)[:, :, rev]                  # (t, z, p) m/s, + eastward
    zw = bx.z_w.isel(eta_rho=js, xi_rho=slice(i, i + 2)).values[:, :, rev, :]        # (t, z+1, p, 2)
    pn = bx.pn.isel(eta_rho=js, xi_rho=slice(i, i + 2)).values[rev, :]               # (p, 2)
    zeta = bx.zeta.isel(eta_rho=js, xi_rho=slice(i, i + 2)).values.astype(float)[:, rev, :].mean(axis=-1)   # (t, p)
    bx.close()
    DZ = np.diff(zw, axis=1).mean(axis=-1)               # (t, z, p) u-point cell thickness
    dd = (1 / pn).mean(axis=-1)                          # (p,) face width
    h = -zw[0, 0].mean(axis=-1)                          # (p,) bed depth at the face
    A = dd[None, None, :] * DZ
    return -u * A, A, DZ, h, dd, zeta

R = {}
for sect in SECTS:
    if SRC == 'his':
        q_in, A, DZ, h, dd, zeta = load_his(sect)        # snapshots: q_in = -u * A
    else:
        ds = xr.open_dataset(in_dir / ('%s.nc' % sect))
        q_in = -ds.q.values                                  # (time, z, p) m3/s, + into the cove
        A = ds.dd.values[None, None, :] * ds.DZ.values      # (time, z, p) m2, cell face area this hour
        DZ = ds.DZ.values
        h = ds.h.values                                      # (p,) bed depth at each face
        dd = ds.dd.values                                    # (p,) face width
        zeta = ds.zeta.values                                # (time, p) hour-mean surface height
        ds.close()

    Q_in = q_in.sum(axis=(1, 2))                         # (time,) section-total transport into the cove
    F = Q_in > 0                                         # flood hours
    E = Q_in < 0                                         # ebb hours
    dzdt = np.gradient(zeta.mean(axis=1)) / 3600         # m/s, section-mean d(zeta)/dt
    print('%s: flood %.1f%% of hours, ebb %.1f%%, corr(Q_in, dzeta/dt) = %+.3f' % (sect, 100 * F.mean(), 100 * E.mean(), np.corrcoef(Q_in, dzdt)[0, 1]))

    PF = peak_hours(F, Q_in)                             # peak hour of each flood
    PE = peak_hours(E, Q_in)                             # peak hour of each ebb
    nrun = lambda m: int((np.diff(np.concatenate([[0], m.astype(int)])) == 1).sum())
    print('   peaks: %d floods (%d runs < 3 h skipped), %d ebbs (%d skipped)' % (PF.sum(), nrun(F) - PF.sum(), PE.sum(), nrun(E) - PE.sum()))

    out = {}
    for ph, m in [('flood', F), ('ebb', E), ('max flood', PF), ('max ebb', PE), ('residual', np.ones_like(F))]:
        if AVG == 'transport':
            U = q_in[m].sum(axis=0) / A[m].sum(axis=0)       # (z, p) m/s
        else:
            U = (q_in[m] / A[m]).mean(axis=0)                # (z, p) m/s
        zw = np.concatenate([-h[None, :], -h[None, :] + np.cumsum(DZ[m].mean(axis=0), axis=0)], axis=0)   # (z+1, p) phase-mean interface depths
        out[ph] = dict(U=U, zw=zw, zr=0.5 * (zw[1:] + zw[:-1]), A=A[m].mean(axis=0), Q=Q_in[m].mean(), frac=m.mean())
    out['dd'] = dd
    R[sect] = out

    f, e, r = out['flood'], out['ebb'], out['residual']
    if AVG == 'transport':
        recon = (f['frac'] * f['A'] * f['U'] + e['frac'] * e['A'] * e['U']) / r['A']
    else:
        recon = f['frac'] * f['U'] + e['frac'] * e['U']
    U_other = (q_in / A).mean(axis=0) if AVG == 'transport' else q_in.sum(axis=0) / A.sum(axis=0)
    print('   decomposition: max |U_res - recon| = %.1e cm/s' % (100 * np.abs(r['U'] - recon).max()))
    print('   integral: sum(U_res * A_res) = %+.2f m3/s vs mean Q_in = %+.2f m3/s' % ((r['U'] * r['A']).sum(), r['Q']))
    print('   method choice: max |U_res(transport) - U_res(eulerian)| = %.2f cm/s (residual rms %.2f cm/s)' % (100 * np.abs(r['U'] - U_other).max(), 100 * np.sqrt((r['U'] ** 2).mean())))
    print('   mean Q_in: flood %+.0f, ebb %+.0f m3/s' % (f['Q'], e['Q']))

def draw(ax, d, dd, vmax, dlev, cmap=cmocean.cm.balance, vmin=None):
    vmin = -vmax if vmin is None else vmin
    xe = np.concatenate([[0], np.cumsum(dd)]) / 1000                  # km, face edges from the north shore
    for p in range(len(dd)):
        cs = ax.pcolormesh(xe[p:p + 2], d['zw'][:, p], 100 * d['U'][:, p:p + 1], cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
    X = np.broadcast_to(0.5 * (xe[1:] + xe[:-1]), d['zr'].shape)      # (z, p) face centers
    lev = np.arange(np.floor(vmin / dlev) * dlev, vmax + dlev, dlev)
    ax.contour(X, d['zr'], 100 * d['U'], levels=lev[lev != 0], colors='0.3', linewidths=0.5)
    if vmin < 0:
        ax.contour(X, d['zr'], 100 * d['U'], levels=[0], colors='k', linewidths=1.5)
    ax.set_xlim(xe[-1], xe[0])           # reversed: north at right = looking west, into the cove
    ax.grid(**GRID)
    return cs

# tidal part = composite minus the residual (exact for either AVG, since U_res is fixed per cell)
for s in SECTS:
    for ph in ['flood', 'ebb']:
        d, r = R[s][ph], R[s]['residual']
        R[s]['tidal ' + ph] = dict(U=d['U'] - r['U'], zw=d['zw'], zr=d['zr'], A=d['A'], Q=d['Q'] - r['Q'], frac=d['frac'])

def make_fig(PH, dlev, cblab, fname):
    vt = np.ceil(np.nanmax([np.nanpercentile(np.abs(100 * R[s][ph]['U']), 99) for s in SECTS for ph in PH[:2]]))
    vr = np.ceil(np.nanmax([np.nanpercentile(np.abs(100 * R[s]['residual']['U']), 99) for s in SECTS]))
    # columns = sections west to east, each as wide as the section really is; rows = phases
    fig, axs = plt.subplots(3, 3, figsize=(14, 9), sharey=True, layout='constrained',
                            gridspec_kw={'width_ratios': [R[s]['dd'].sum() for s in SECTS]})
    for j, s in enumerate(SECTS):
        for i, ph in enumerate(PH):
            ax = axs[i, j]
            d = R[s][ph]
            cs = draw(ax, d, R[s]['dd'], vr if ph == 'residual' else vt, 1 if ph == 'residual' else dlev)
            if ph == 'residual':
                cr = cs
            else:
                ct = cs
            ax.set_title('%s, %s\n%.0f%% of hours, mean Q_in %+.*f m3/s' % (NAME[s], ph, 100 * d['frac'], 2 if ph == 'residual' else 0, d['Q']), fontsize=8)
            if j == 0:
                ax.set_ylabel('z [m]')
            if i == 2:
                ax.set_xlabel('distance from north shore [km]')
    fig.colorbar(ct, ax=axs[:2, :], location='right', shrink=0.8, label=cblab + ' [cm/s], + into the cove (contours every %g)' % dlev)
    fig.colorbar(cr, ax=axs[2, :], location='right', shrink=0.9, label='residual [cm/s], + into\nthe cove (contours every 1)')
    fig.suptitle('Penn Cove sections west to east, each looking into the cove (west; north at right), %s average, %s, %s, %s to %s' % (AVG, 'hourly snapshots' if SRC == 'his' else 'hour-mean Huon', GTX, DS0, DS1), fontsize=11)
    fn = out_dir / fname.replace('.png', '_%s.png' % SRC)
    fig.savefig(fn, dpi=200, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print('Saved ' + str(fn))

make_fig(['flood', 'ebb', 'residual'], 5, 'section-normal velocity', 'pc_flood_ebb_sections.png')
make_fig(['tidal flood', 'tidal ebb', 'residual'], 2, 'tidal velocity (composite minus residual)', 'pc_tidal_flood_ebb_sections.png')
make_fig(['max flood', 'max ebb', 'residual'], 5, 'velocity at the peak hour of each flood / ebb, averaged', 'pc_max_flood_ebb_sections.png')
