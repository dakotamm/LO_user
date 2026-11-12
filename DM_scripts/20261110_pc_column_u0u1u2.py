"""
MacCready (2011, JPO 41, 1116-1124) Section 2 velocity decomposition, u = u0 + u1 + u2,
at EVERY u-face column of Penn Cove (box xi_u 1..30, head to mouth; the columns of
20261110_pc_column_metrics.py), of wb1_t0_xn11abbur00. Same equations as
20261103_pc_eulerian_u0u1u2.py (copied), which did pc_cp, pc_lj and pc_lp only:
    (2.2)  A0  = < int dA >,   dA0 = < dA >
    (2.3)  u0  = < int u dA > / A0                 section-mean subtidal velocity
    (2.4)  u1  = < u dA > / dA0 - u0               subtidal, varies across the section
    (2.5)  u2  = u - u0 - u1                       hourly; the tidal part
<> = Godin filter, kept hourly (NaN for 35 h at each end; statistics use the valid hours).

LATERAL / LAYERED SPLIT at each column (the names of [[exchange-terms-layered-lateral]]):
    time-mean u1 = lateral + layered
        lateral  = the dA0-weighted depth mean of time-mean u1 at each face (varies across only)
        layered  = time-mean u1 minus that (varies with depth, integrates to 0 on each face)
    hourly u2 = slab + lateral + layered
        slab     = the dA-weighted section mean of u2 (the tide moving as one block)
        lateral, layered as above, of u2 - slab, each hour
  Each split is orthogonal under its area weighting, so the mean squares add exactly:
  rms^2 = lateral^2 + layered^2 (u1); slab^2 + lateral^2 + layered^2 (u2).

MAX FLOOD / MAX EBB on ONE clock for every column: the peak-|Q| hour of each flood (ebb) of
the hourly MOUTH transport (col 30), runs < 3 h skipped (peak_hours of 20261103). The 20261103
script used each section's own Q; the cove floods and ebbs together (r 0.998), so at its three
sections the choice changes little.

READING. u and z_w only, the box read once in month blocks; each column's u and dA kept
(float32, ~1.3 GB together), then decomposed column by column.

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc (hourly ocean_his snapshots)
OUTPUT LO_output/DM_outs/20261110_pc_column_u0u1u2/column_u0u1u2.p
    per column: x_km, dy, h, z0 (time-mean cell-centre z), u0h (hourly), scalar statistics,
    time-mean u1 (and its lateral / layered parts), composites of u, u1, u2 at max flood /
    max ebb; plotted by 20261110_pc_column_u0u1u2_plot.py

run 20261110_pc_column_u0u1u2.py
"""
import pickle
from time import time as clock

import numpy as np
import xarray as xr

from lo_tools import Lfun, zfun

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SIGN = -1        # section-positive (east) is out of the cove
XI_US = np.arange(1, 31)                  # u-face columns, head (west) to mouth (pc_lp = 30)
LAT = 48.23
BLK = 744                                 # hours per read (31 days)

Ldir = Lfun.Lstart(gridname='wb1')
in_fn = (Ldir['LOo'] / 'extract' / GTAGEX / 'box'
         / ('pc_cove_' + DS0 + '_' + DS1 + '.nc'))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261110_pc_column_u0u1u2'
out_dir.mkdir(parents=True, exist_ok=True)


def lp(x):
    """Tidal low-pass <>: Godin filter on axis 0, hourly (NaN at the ends)."""
    return zfun.lowpass(x, f='godin')


def isum(x):
    """Integral over the section, int ... dA, as a sum over elements (k, p)."""
    return x.sum(axis=(-2, -1))


def peak_hours(mask, Q, nmin=3):
    """One hour per contiguous run of mask (one flood or one ebb): the hour of
    largest |Q|. Runs shorter than nmin hours are slack-water flicker, skipped."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    out = np.zeros_like(mask)
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= nmin:
            out[a + np.argmax(np.abs(Q[a:b]))] = True
    return out


def split_lat_lay(f, w):
    """f, w (..., k, p): lateral = w-weighted depth mean of f at each face, broadcast
    over k; layered = f - lateral."""
    lat = (f * w).sum(axis=-2, keepdims=True) / w.sum(axis=-2, keepdims=True)
    return np.broadcast_to(lat, f.shape), f - lat


# ------------------------------------------------------------------ column geometry (as 20261110_pc_column_metrics.py)
ds = xr.open_dataset(in_fn)
time = ds.ocean_time.to_numpy()
NT = len(time)
mask = ds.mask_rho.values == 1
pn = ds.pn.values
lon_u = ds.lon_u.values[mask.shape[0] // 2]
COL = {}
for XI_U in XI_US:
    IRM, IRP = XI_U, XI_U + 1
    ETA = np.flatnonzero(mask[:, IRM] & mask[:, IRP])[::-1]              # north to south
    COL[XI_U] = dict(IRM=IRM, IRP=IRP, ETA=ETA, dy=(1 / pn[ETA][:, [IRM, IRP]]).mean(axis=-1),
                     x_km=(lon_u[30] - lon_u[XI_U]) * 111.32 * np.cos(np.deg2rad(LAT)),
                     u=np.empty((NT, 30, len(ETA)), np.float32), dz=np.empty((NT, 30, len(ETA)), np.float32))

# ------------------------------------------------------------------ read u and z_w once, a block of hours at a time
tt0 = clock()
for t0 in range(0, NT, BLK):
    tt = slice(t0, min(t0 + BLK, NT))
    U = ds.u.isel(ocean_time=tt).to_numpy()
    ZW = ds.z_w.isel(ocean_time=tt).to_numpy()
    for XI_U, C in COL.items():
        ETA, IRM, IRP = C['ETA'], C['IRM'], C['IRP']
        C['u'][tt] = SIGN * U[:, :, ETA, XI_U]
        C['dz'][tt] = np.diff(ZW[:, :, ETA][:, :, :, [IRM, IRP]], axis=1).mean(axis=-1)
    print('  hours %d-%d of %d, %.0f s' % (tt.start, tt.stop, NT, clock() - tt0))
h_all = ds.h.values
ds.close()

# mouth transport (hourly, unfiltered): the flood / ebb clock for every column
Cm = COL[30]
Qm = isum(Cm['u'].astype(float) * Cm['dz'] * Cm['dy'][None, None, :])
PF0, PE0 = peak_hours(Qm > 0, Qm), peak_hours(Qm < 0, Qm)

# ------------------------------------------------------------------ decompose each column
R = {}
for XI_U, C in COL.items():
    u = C['u'].astype(float)                  # (t, k, p)
    dy = C['dy']
    dA = C['dz'].astype(float) * dy[None, None, :]
    del C['u'], C['dz']                       # free the float32 copies as we go

    A0h = lp(isum(dA))                        # (2.2) < int dA >
    u0h = lp(isum(u * dA)) / A0h              # (2.3)
    dA0h = lp(dA)                             #       < dA >
    u1h = lp(u * dA) / dA0h - u0h[:, None, None]   # (2.4)
    u2 = u - u0h[:, None, None] - u1h         # (2.5)
    ok = np.isfinite(u0h)

    # time-mean u1 and its lateral / layered parts, weighted by the time-mean dA0
    w0 = dA0h[ok].mean(axis=0)                # (k, p)
    u1m = u1h[ok].mean(axis=0)
    u1lat, u1lay = split_lat_lay(u1m, w0)
    wrms = lambda f, w: np.sqrt((f**2 * w).sum() / w.sum())

    # hourly u2: slab + lateral + layered, weighted by the hourly dA
    u2o, dAo = u2[ok], dA[ok]
    slab = isum(u2o * dAo) / isum(dAo)        # (t,)
    u2lat, u2lay = split_lat_lay(u2o - slab[:, None, None], dAo)
    wsum = dAo.sum()
    ms = lambda f: (f**2 * dAo).sum() / wsum if f.ndim == 3 else (f**2 * isum(dAo)).sum() / wsum

    # max flood / max ebb composites (mouth clock), plain mean over those hours, as 20261103
    comp = {}
    for ph, m in [('max flood', PF0 & ok), ('max ebb', PE0 & ok)]:
        comp[ph] = {'u': u[m].mean(axis=0), 'u0': u0h[m].mean(), 'u1': u1h[m].mean(axis=0), 'u2': u2[m].mean(axis=0)}

    # time-mean cell-centre depth from the tidally averaged element thickness (as 20261103)
    h = h_all[C['ETA']][:, [C['IRM'], C['IRP']]].mean(axis=-1)         # (p,) depth at the face
    dz0m = w0 / dy[None, :]
    z0 = -h[None, :] + np.cumsum(dz0m, axis=0) - dz0m / 2

    S = dict(x_km=C['x_km'], A0=A0h[ok].mean(), u0_mean=u0h[ok].mean(), u0_std=u0h[ok].std(),
             u1m_rms=wrms(u1m, w0), u1m_lat=wrms(u1lat, w0), u1m_lay=wrms(u1lay, w0),
             u1_rms=np.sqrt((u1h[ok]**2 * dAo).sum() / wsum),
             u2_rms=np.sqrt(ms(u2o)), u2_slab=np.sqrt(ms(slab)), u2_lat=np.sqrt(ms(u2lat)), u2_lay=np.sqrt(ms(u2lay)),
             chk_int_u1=np.nanmax(np.abs(isum(u1h * dA0h))),
             chk_u1split=abs(wrms(u1m, w0)**2 - wrms(u1lat, w0)**2 - wrms(u1lay, w0)**2),
             chk_u2split=abs(ms(u2o) - ms(slab) - ms(u2lat) - ms(u2lay)) / ms(u2o),
             n_flood=int((PF0 & ok).sum()), n_ebb=int((PE0 & ok).sum()))
    R[XI_U] = dict(S=S, dy=dy, h=h, z0=z0, u0h=u0h.astype(np.float32), u1m=u1m, u1lat=np.array(u1lat), u1lay=u1lay, comp=comp)
    print('col %2d (%.2f km): u0 %+.2f+-%.2f mm/s | mean u1 rms %.2f cm/s (lat %.2f, lay %.2f) | u2 rms %.2f (slab %.2f, lat %.2f, lay %.2f) | chk %.0e %.0e %.0e'
          % (XI_U, S['x_km'], 1e3 * S['u0_mean'], 1e3 * S['u0_std'], 100 * S['u1m_rms'], 100 * S['u1m_lat'], 100 * S['u1m_lay'],
             100 * S['u2_rms'], 100 * S['u2_slab'], 100 * S['u2_lat'], 100 * S['u2_lay'], S['chk_int_u1'], S['chk_u1split'], S['chk_u2split']))

with open(out_dir / 'column_u0u1u2.p', 'wb') as f:
    pickle.dump(dict(R=R, time=time, Qm=Qm, PF=PF0, PE=PE0, gtagex=GTAGEX), f)
print('saved ' + str(out_dir / 'column_u0u1u2.p'))
