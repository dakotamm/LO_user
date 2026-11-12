"""
Hourly cross-section profiles of the Saratoga Passage tef2 sections, for the Saratoga version of the
Penn Cove conceptual map (20261111_sp_conceptual_map.py).

For each hour, each cell's transport q, area DZ*dd and salt*area are split among 1 m bins of depth
below the moving surface by the fraction of the cell each bin overlaps (as 20261110_pc_midchannel_u_timeseries.py),
so sum_b Q_b = Q exactly. A class composite is then sum_t Q_b / sum_t A_b (transport-weighted) and
sum_t SA_b / sum_t A_b (area-weighted salinity).

Dry cells: ocean_avg writes dry cell-hours as 0 in every variable ([[wb1-avg-dry-zeros]]); a cell
with salt == 0 gets zero area here (its q is 0 already).

INPUT  LO_output/extract/[gtagex]/tef2/extractions_avg_[ds0]_[ds1]/[sect].nc   (on apogee: skagit_sp,
       sp_mid are not mirrored to the Mac)
OUTPUT LO_output/DM_outs/20261111_sp_section_binned/binned_[sect].nc with
       Q (time), Qb, Ab, SAb (time, bin) [float32], zeta (time, width-mean), edges (bin + 1), h, dd (p)
       Signs are tef2's (+ from the section's minus side to its plus side); the plot script orients them.

run 20261111_sp_section_binned.py [-sects skagit_sp,sp_mid] [-gtx wb1_t0_xn11abbur00]
"""
import argparse

import numpy as np
import xarray as xr

from lo_tools import Lfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', default='wb1_t0_xn11abbur00')
parser.add_argument('-sects', default='skagit_sp,sp_mid')
parser.add_argument('-ds0', default='2024.01.01')
parser.add_argument('-ds1', default='2025.12.31')
parser.add_argument('-chunk', default=744, type=int)          # hours read at a time
args = parser.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
in_dir = Ldir['LOo'] / 'extract' / args.gtx / 'tef2' / ('extractions_avg_%s_%s' % (args.ds0, args.ds1))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261111_sp_section_binned'
out_dir.mkdir(parents=True, exist_ok=True)

for sect in args.sects.split(','):
    ds = xr.open_dataset(in_dir / ('%s.nc' % sect))
    h, dd = ds.h.values, ds.dd.values
    nt = ds.sizes['time']
    E = np.arange(0, np.ceil(h.max() + 4) + 1)                # 1 m bins; +4 m for high water
    nb = len(E) - 1
    Q = np.zeros(nt); Z = np.zeros(nt)
    Qb, Ab, SAb = (np.zeros((nt, nb), dtype=np.float32) for _ in range(3))
    n_dry = 0
    for t0 in range(0, nt, args.chunk):
        sl = slice(t0, min(t0 + args.chunk, nt))
        q = ds.q[sl].values                                   # (t, z, p) m3/s, bottom to top
        DZ = ds.DZ[sl].values
        s = ds.salt[sl].values.astype(float)
        zeta = ds.zeta[sl].values.astype(float)
        dry = s == 0
        n_dry += dry.sum()
        a = np.where(dry, 0, DZ * dd[None, None, :])          # cell area [m2]
        zw = -h[None, None, :] + np.concatenate([np.zeros_like(DZ[:, :1]), np.cumsum(DZ, axis=1)], axis=1)
        dw = zw[:, -1:, :] - zw                               # depth below the surface of each interface
        d_top, d_bot = dw[:, 1:, :], dw[:, :-1, :]
        assert d_bot.max() < E[-1], 'bins too shallow: %.1f m' % d_bot.max()
        with np.errstate(invalid='ignore', divide='ignore'):
            for b in range(nb):
                f = np.clip(np.minimum(d_bot, E[b + 1]) - np.maximum(d_top, E[b]), 0, None) / np.where(DZ > 0, DZ, np.inf)
                Qb[sl, b] = (q * f).sum(axis=(1, 2))
                Ab[sl, b] = (a * f).sum(axis=(1, 2))
                SAb[sl, b] = (s * a * f).sum(axis=(1, 2))
        Q[sl] = q.sum(axis=(1, 2))
        Z[sl] = (zeta * dd[None]).sum(axis=1) / dd.sum()
        assert np.allclose(Qb[sl].sum(axis=1), Q[sl], rtol=1e-4, atol=1e-2), 'bins do not sum to Q'
        print('  %s: %d / %d h' % (sect, sl.stop, nt), flush=True)
    out = xr.Dataset(dict(Q=('time', Q), zeta=('time', Z), Qb=(('time', 'bin'), Qb), Ab=(('time', 'bin'), Ab),
                          SAb=(('time', 'bin'), SAb), edges=('edge', E), h=('p', h), dd=('p', dd)),
                     coords=dict(time=ds.time.values))
    out.attrs.update(sect=sect, gtagex=args.gtx, note='1 m bins of depth below the moving surface; tef2 signs; '
                     'dry cell-hours (salt == 0) given zero area: %d of %d' % (n_dry, ds.salt.size))
    fn = out_dir / ('binned_%s.nc' % sect)
    out.to_netcdf(fn)
    print('%s: %d faces, h max %.1f m, %d bins, dry cell-hours %d -> %s' % (sect, len(h), h.max(), nb, n_dry, fn))
    ds.close()
