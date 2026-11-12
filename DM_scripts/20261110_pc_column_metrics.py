"""
Hourly flow metrics at EVERY column of u-faces in Penn Cove, head to mouth, of
wb1_t0_xn11abbur00: the 20261104_pc_flow_metrics.py set, taken from three sections
(pc_cp, pc_lj, pc_lp) to all 30, so the along-cove variation can be seen.

COLUMNS. Box u-face column XI_U = 1..30 lies between rho columns XI_U and XI_U + 1
(global i = XI_U + 37). Its faces are the rows where BOTH rho cells are wet. This rule
gives exactly the hand-set faces of pc_cp (XI_U 15), pc_lj (23) and pc_lp (30). The box's
west, north and south edges are land, so every column cuts the cove completely: it is a
closed section, and nothing enters the water landward of it except through its faces
(and point sources).

PER COLUMN AND HOUR, as in 20261104_pc_flow_metrics.py (copied, unchanged):
    Q, G, Qx, LC, VC, umax, u95, Srms, dU, drho, fcrit      (+ = into the cove)
PLUS
    Vland  = sum zeta / (pm pn) over the wet cove cells with xi_rho <= XI_U  [m3]
             the volume anomaly landward of the column. Continuity: Q = dVland/dt
             (+ sources); the plot script differentiates it. That is the prism
             expectation for the tidal transport at each column.
    Uprof  width-mean section-normal u on fixed depths DG BELOW THE MOVING SURFACE
             (0.25 m, every 0.5 m), dy-weighted over the faces deep enough
             (DG <= H - 0.25, as in 20261028_pc_mouth_interface.py; linear in the
             vertical, clamped to the top / bottom cell centre). Saved raw, so the
             two-layer divide (20261028 run rules) can be found, and its rules
             changed, without re-reading the box. NB the 20261028 mouth profile was the
             cove-axis component at the rho column west of pc_lp; this is u normal to
             the faces, at the faces.

GEOMETRY (per column): x_km = distance west of the mouth (pc_lp, XI_U 30), east-west
along the plaid grid; width, mean area, mean depth, n faces, landward surface area.

READING. The box is stored in compressed chunks of one hour x the whole box, so it is
read once, a month of hours at a time, and every column is cut from each block.

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc (hourly ocean_his snapshots)
OUTPUT LO_output/DM_outs/20261110_pc_column_metrics/column_metrics.nc   (time, col[, depth])

run 20261110_pc_column_metrics.py [-test True]
"""
import argparse
from time import time as clock

import numpy as np
import xarray as xr
import gsw

from lo_tools import Lfun

parser = argparse.ArgumentParser()
parser.add_argument('-test', default=False, type=Lfun.boolean_string)   # True: first month only
TEST = parser.parse_args().test

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SIGN = -1        # section-positive (east) is out of the cove; v is not flipped (+ = north)
XI_US = np.arange(1, 31)                  # u-face columns, head (west) to mouth (pc_lp = 30)
SECT_XI_U = {'pc_cp': 15, 'pc_lj': 23, 'pc_lp': 30}
G_, RHO0 = 9.81, 1025.0
LON, LAT = -122.69, 48.23
RI_CRIT = 0.25
DZG = 0.5                                 # profile depth step [m]
BLK = 744                                 # hours per read (31 days)
METRICS = ['Q', 'G', 'Qx', 'LC', 'VC', 'umax', 'u95', 'Srms', 'dU', 'drho', 'fcrit']
UNITS = {'Q': 'm3/s', 'G': 'm3/s', 'Qx': 'm3/s', 'LC': 'm3/s', 'VC': 'm3/s', 'umax': 'm/s', 'u95': 'm/s',
         'Srms': '1/s', 'dU': 'm/s', 'drho': 'kg/m3', 'fcrit': '1'}

Ldir = Lfun.Lstart(gridname='wb1')
in_fn = (Ldir['LOo'] / 'extract' / GTAGEX / 'box'
         / ('pc_cove_' + DS0 + '_' + DS1 + '.nc'))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261110_pc_column_metrics'
out_dir.mkdir(parents=True, exist_ok=True)

ds = xr.open_dataset(in_fn)

# ------------------------------------------------------------------ column geometry
mask = ds.mask_rho.values == 1
pm, pn, h = ds.pm.values, ds.pn.values, ds.h.values
COL = {}
for XI_U in XI_US:
    IRM, IRP = XI_U, XI_U + 1
    ETA = np.flatnonzero(mask[:, IRM] & mask[:, IRP])[::-1]              # north to south
    land = mask.copy()
    land[:, XI_U + 1:] = False                                           # wet cove cells landward (west)
    COL[XI_U] = dict(IRM=IRM, IRP=IRP, ETA=ETA,
                     dy=(1 / pn[ETA][:, [IRM, IRP]]).mean(axis=-1),
                     land=land)
for SN, XI_U in SECT_XI_U.items():
    print('%s: XI_U %d, faces eta %d..%d (%d)' % (SN, XI_U, COL[XI_U]['ETA'][0], COL[XI_U]['ETA'][-1], len(COL[XI_U]['ETA'])))
lon_u = ds.lon_u.values[mask.shape[0] // 2]
x_km = (lon_u[30] - lon_u[XI_US]) * 111.32 * np.cos(np.deg2rad(LAT))   # km west of the mouth
DG = np.arange(0.25, h[mask].max() + 1.0, DZG)                           # depth below the moving surface [m]


# ------------------------------------------------------------------ helpers (wpercentile and metrics copied from 20261104_pc_flow_metrics.py)
def wpercentile(x, w, p):
    """Weighted percentile p (0-1) along axis 1 of x (t, n) with weights w (t, n)."""
    i = np.argsort(x, axis=1)
    xs, ws = np.take_along_axis(x, i, axis=1), np.take_along_axis(w, i, axis=1)
    cw = np.cumsum(ws, axis=1) / ws.sum(axis=1, keepdims=True)
    j = np.argmax(cw >= p, axis=1)
    return xs[np.arange(len(x)), j]


def metrics(u, v, salt, temp, zw, zr, dy):
    """All metrics for one section over a block of hours. u, v, salt, temp, zr (t, k, p);
    zw (t, k+1, p); dy (p,). Returns a dict of (t,) arrays."""
    nt = len(u)
    dz = np.diff(zw, axis=1)                                  # (t, k, p) cell thickness
    dA = dz * dy[None, None, :]
    q = u * dA                                                # (t, k, p) transport per element

    # transport and counterflow
    Q = q.sum(axis=(1, 2))
    G = np.abs(q).sum(axis=(1, 2))
    Qin, Qout = np.clip(q, 0, None).sum(axis=(1, 2)), np.clip(q, None, 0).sum(axis=(1, 2))
    Qx = np.minimum(Qin, -Qout)
    Qp = q.sum(axis=1)                                        # (t, p) face transport
    LC = np.clip(-np.sign(Q)[:, None] * Qp, 0, None).sum(axis=1)
    VC = np.clip(-np.sign(Qp)[:, None, :] * q, 0, None).sum(axis=(1, 2))

    # speed
    au = np.abs(u).reshape(nt, -1)
    umax = au.max(axis=1)
    u95 = wpercentile(au, dA.reshape(nt, -1), 0.95)

    # shear, stratification, Ri on the layers between cell centres
    dzr = np.diff(zr, axis=1)                                 # (t, k-1, p)
    S2 = (np.diff(u, axis=1) / dzr)**2 + (np.diff(v, axis=1) / dzr)**2
    SA = gsw.SA_from_SP(salt, -zr, LON, LAT)
    sig = gsw.sigma0(SA, gsw.CT_from_pt(SA, temp))
    N2 = -(G_ / RHO0) * np.diff(sig, axis=1) / dzr
    wl = dzr * dy[None, None, :]                              # layer area
    Srms = np.sqrt((S2 * wl).sum(axis=(1, 2)) / wl.sum(axis=(1, 2)))
    fcrit = ((N2 < RI_CRIT * S2) * wl).sum(axis=(1, 2)) / wl.sum(axis=(1, 2))
    W = dy / dy.sum()                                         # width weights
    dU = ((u[:, -1, :] - u[:, 0, :]) * W).sum(axis=1)
    drho = ((sig[:, 0, :] - sig[:, -1, :]) * W).sum(axis=1)
    return dict(Q=Q, G=G, Qx=Qx, LC=LC, VC=VC, umax=umax, u95=u95, Srms=Srms, dU=dU, drho=drho, fcrit=fcrit)


def on_depths(x, zr, zw, dy):
    """Width-mean profile of x on the depths DG below the moving surface. x, zr (t, k, p) at
    cell centres; zw (t, k+1, p); dy (p,). Per face: linear in depth, clamped to the top and
    bottom cell centres, NaN deeper than H - 0.25 m (as 20261028_pc_mouth_interface.py).
    Then the dy-weighted mean over the faces that reach each depth. Returns (t, nd)."""
    d = (zw[:, -1:, :] - zr)[:, ::-1, :]                      # (t, k, p) depth, increasing down
    xr_ = x[:, ::-1, :]
    K = d.shape[1]
    i1 = np.clip((d[:, :, None, :] < DG[None, None, :, None]).sum(axis=1), 1, K - 1)   # (t, nd, p)
    i0 = i1 - 1
    d0, d1 = np.take_along_axis(d, i0, axis=1), np.take_along_axis(d, i1, axis=1)
    x0, x1 = np.take_along_axis(xr_, i0, axis=1), np.take_along_axis(xr_, i1, axis=1)
    f = np.clip((DG[None, :, None] - d0) / (d1 - d0), 0, 1)
    xp = x0 + f * (x1 - x0)
    ok = DG[None, :, None] <= (zw[:, -1, :] - zw[:, 0, :])[:, None, :] - 0.25
    wgt = ok * dy[None, None, :]
    with np.errstate(invalid='ignore'):
        return (np.where(ok, xp, 0) * wgt).sum(axis=2) / wgt.sum(axis=2)


# ------------------------------------------------------------------ main: read the box a block of hours at a time
T = ds.ocean_time.values
NT = BLK if TEST else len(T)
NC = len(XI_US)
OUT = {k: np.full((NT, NC), np.nan) for k in METRICS + ['A', 'Vland']}
UP = np.full((NT, NC, len(DG)), np.nan, np.float32)
cell_area = 1 / (pm * pn)
tt0 = clock()
for t0 in range(0, NT, BLK):
    tt = slice(t0, min(t0 + BLK, NT))
    B = {vn: ds[vn].isel(ocean_time=tt).to_numpy().astype(float) for vn in ['u', 'v', 'salt', 'temp', 'z_w', 'z_rho', 'zeta']}
    for c, XI_U in enumerate(XI_US):
        C = COL[XI_U]
        ETA, IRM, IRP, dy = C['ETA'], C['IRM'], C['IRP'], C['dy']
        face = lambda x: x[:, :, ETA][:, :, :, [IRM, IRP]].mean(axis=-1)               # rho -> face, (t, k, p)
        u = SIGN * B['u'][:, :, ETA, XI_U]
        v = np.nan_to_num(np.concatenate([B['v'][:, :, ETA + dj][:, :, :, [IRM, IRP]] for dj in (-1, 0)], axis=-1)).mean(axis=-1)
        zw, zr = face(B['z_w']), face(B['z_rho'])
        M = metrics(u, v, face(B['salt']), face(B['temp']), zw, zr, dy)
        for k in METRICS:
            OUT[k][tt, c] = M[k]
        OUT['A'][tt, c] = (np.diff(zw, axis=1) * dy[None, None, :]).sum(axis=(1, 2))
        OUT['Vland'][tt, c] = (B['zeta'][:, C['land']] * cell_area[C['land']][None, :]).sum(axis=1)
        UP[tt, c] = on_depths(u, zr, zw, dy)
    print('  hours %d-%d of %d, %.0f s' % (tt.start, tt.stop, NT, clock() - tt0))
ds.close()

# ------------------------------------------------------------------ save
UNITS.update(A='m2', Vland='m3')
dsout = xr.Dataset({k: (('time', 'col'), OUT[k], {'units': UNITS[k]}) for k in METRICS + ['A', 'Vland']},
                   coords={'time': T[:NT], 'col': XI_US, 'depth': DG},
                   attrs={'source': str(in_fn),
                          'sign': 'u, Q, dU positive into the cove (west)',
                          'col': 'box xi_u of the u-face column (global i = col + 37); pc_cp 15, pc_lj 23, pc_lp 30',
                          'note': 'hourly snapshots; see the docstring of 20261110_pc_column_metrics.py for definitions'})
dsout['Uprof'] = (('time', 'col', 'depth'), UP, {'units': 'm/s', 'long_name': 'width-mean u on depth below the moving surface'})
dsout['x_km'] = (('col',), x_km, {'units': 'km', 'long_name': 'distance west of the mouth (pc_lp)'})
dsout['width'] = (('col',), np.array([COL[x]['dy'].sum() for x in XI_US]), {'units': 'm'})
dsout['nface'] = (('col',), np.array([len(COL[x]['ETA']) for x in XI_US]))
dsout['Aland'] = (('col',), np.array([cell_area[COL[x]['land']].sum() for x in XI_US]), {'units': 'm2', 'long_name': 'surface area landward'})
fn = out_dir / ('column_metrics%s.nc' % ('_test' if TEST else ''))
dsout.to_netcdf(fn)
print('saved ' + str(fn))

# ------------------------------------------------------------------ check: the three named columns against 20261104_pc_flow_metrics.py
fm_fn = Ldir['LOo'] / 'DM_outs' / '20261104_pc_flow_metrics' / ('flow_metrics%s.nc' % ('_test' if TEST else ''))
if fm_fn.exists():
    fm = xr.open_dataset(fm_fn)
    for SN, XI_U in SECT_XI_U.items():
        c = int(np.flatnonzero(XI_US == XI_U)[0])
        n = min(NT, fm.time.size)
        dmax = max(np.nanmax(np.abs(OUT[k][:n, c] - fm[k].sel(sect=SN).values[:n])) / np.nanmax(np.abs(fm[k].sel(sect=SN).values[:n])) for k in METRICS)
        print('check %s vs flow_metrics: max relative difference over all metrics %.1e' % (SN, dmax))
for c, XI_U in enumerate(XI_US):
    G = OUT['G'][:, c].sum()
    print('col %2d (%4.2f km): %2d faces, A %5.0f m2, std Q %4.0f m3/s, Qx/G %.2f, LC/G %.2f, VC/G %.2f, u95 %4.1f cm/s, Srms %.3f, drho %.2f, fcrit %.2f'
          % (XI_U, x_km[c], len(COL[XI_U]['ETA']), OUT['A'][:, c].mean(), OUT['Q'][:, c].std(), OUT['Qx'][:, c].sum() / G,
             OUT['LC'][:, c].sum() / G, OUT['VC'][:, c].sum() / G, 100 * OUT['u95'][:, c].mean(), OUT['Srms'][:, c].mean(),
             OUT['drho'][:, c].mean(), OUT['fcrit'][:, c].mean()))
