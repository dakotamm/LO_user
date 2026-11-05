"""
Hourly depth-binned SALINITY at the three Penn Cove sections (pc_cp, pc_lj, pc_lp),
wb1_t0_xn11abbur00, saved so any condition's profile is a sum over its hours (used by
20261105_pc_conceptual_map.py -kind salt).

Per section and hour, whole width, 1 m bins of depth below the moving surface: salinity on the
cells, averaged onto the face from the two rho cells either side, each cell split among the bins
by overlap (as for velocity in 20261103_pc_section_profiles.py). Saved per bin: the area A and the
area-weighted sum s*A, so a condition's profile is sum_t(s*A) / sum_t(A) over its hours. Also
saved: the section transport Q (flood Q > 0 into the cove, ebb Q < 0).

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc (hourly ocean_his snapshots)
OUTPUT LO_output/DM_outs/20261105_pc_salt_binned/salt_binned.nc

run 20261105_pc_salt_binned.py
"""
from time import time as clock

import numpy as np
import xarray as xr

from lo_tools import Lfun

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SIGN = -1        # section-positive (east) is out of the cove
FACES = {        # box indices of each section's u-faces (global i - 37, global j - 209), north to south; west to east
    'pc_cp': dict(XI_U=15, ETA=np.arange(13, 5, -1)),
    'pc_lj': dict(XI_U=23, ETA=np.arange(14, 6, -1)),
    'pc_lp': dict(XI_U=30, ETA=np.arange(19, 7, -1)),
}
EDGES = np.arange(0, 31, 1.0)             # bin edges, depth below the surface [m]
NB = len(EDGES) - 1
BLK = 744

Ldir = Lfun.Lstart(gridname='wb1')
in_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_%s_%s.nc' % (DS0, DS1))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261105_pc_salt_binned'
out_dir.mkdir(parents=True, exist_ok=True)


def binsum(X, lo, hi, dy):
    """Per hour and bin: area A = sum ov dy and XA = sum X ov dy over all elements; lo < hi are
    each element's depths below the surface. X, lo, hi: (t, k, p)."""
    A = np.zeros((len(X), NB), np.float32)
    XA = np.zeros((len(X), NB), np.float32)
    for b in range(NB):
        ov = np.clip(np.minimum(hi, EDGES[b + 1]) - np.maximum(lo, EDGES[b]), 0, None) * dy[None, None, :]
        A[:, b] = ov.sum(axis=(1, 2))
        XA[:, b] = (X * ov).sum(axis=(1, 2))
    return A, XA


ds = xr.open_dataset(in_fn)
T = ds.ocean_time.values
NT = len(T)
OUT = {SN: {k: np.zeros((NT, NB), np.float32) for k in ['A', 'sA']} for SN in FACES}
for SN in FACES:
    OUT[SN]['Q'] = np.zeros(NT)
DY = {SN: (1 / ds.pn.values[F['ETA']][:, [F['XI_U'], F['XI_U'] + 1]]).mean(axis=-1) for SN, F in FACES.items()}
tt0 = clock()
for t0 in range(0, NT, BLK):
    tt = slice(t0, min(t0 + BLK, NT))
    B = {vn: ds[vn].isel(ocean_time=tt).to_numpy().astype(float) for vn in ['u', 'salt', 'z_w']}
    for SN, F in FACES.items():
        XI_U, ETA, dy = F['XI_U'], F['ETA'], DY[SN]
        face = lambda x: x[:, :, ETA][:, :, :, [XI_U, XI_U + 1]].mean(axis=-1)       # rho -> u-face, (t, lev, p)
        u = SIGN * B['u'][:, :, ETA, XI_U]
        zw = face(B['z_w'])
        OUT[SN]['Q'][tt] = (u * np.diff(zw, axis=1) * dy[None, None, :]).sum(axis=(1, 2))
        zeta = zw[:, -1:, :]
        OUT[SN]['A'][tt], OUT[SN]['sA'][tt] = binsum(face(B['salt']), zeta - zw[:, 1:, :], zeta - zw[:, :-1, :], dy)   # cells
    print('  hours %d-%d of %d, %.0f s' % (tt.start, tt.stop, NT, clock() - tt0))
ds.close()

dsout = xr.Dataset({'%s_%s' % (k, SN): ((('time', 'bin') if k != 'Q' else ('time',)), OUT[SN][k]) for SN in FACES for k in ['A', 'sA', 'Q']},
                   coords={'time': T, 'bin': 0.5 * (EDGES[1:] + EDGES[:-1])},
                   attrs={'note': 'hourly, whole-width, 1 m bins of depth below the surface; profile = sum_t XA / sum_t A over the hours of a condition',
                          'units': 'A [m2]; sA [g/kg m2]; Q [m3/s], + into the cove'})
for SN in FACES:
    dsout['dy_' + SN] = (('p_' + SN,), DY[SN])
fn = out_dir / 'salt_binned.nc'
dsout.to_netcdf(fn)
print('saved ' + str(fn))
