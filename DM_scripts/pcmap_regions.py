"""
Penn Cove regions for the pcmap analysis -- the ONE definition every pcmap
script imports, so the quadrants cannot drift apart between scripts.

  cove   tef2 wb1_pc1 segments pc_cp_m + pc_cp_p + pc_lp_m (all water
         landward of pc_lp, the release footprint)
  inner  pc_cp_m (landward of pc_cp); outer = the rest of the cove
  north  the cell centre lies north of the pc_ew line at that cell's
         longitude. pc_ew is the east-west line drawn along the middle of the
         cove (LO_output/section_lines/pc_ew.p, 2026-08-04), linearly
         interpolated in longitude.

Adopted 2026-10-06. It replaces the original per-column split (a cell was north
if its j was above the mean j of its column of cove cells), which followed the
local width of each column and bent south into the inner cove's southern lobe.
With pc_ew: inner-N 93 cells, inner-S 72, outer-N 75, outer-S 77.

The pc_ew points are copied in below because apogee may not have
LO_output/section_lines; where the file exists it is checked against them.

Quadrant codes: 0 inner-N, 1 inner-S, 2 outer-N, 3 outer-S, -1 not cove.

use (from LO_user/DM_scripts):
    from pcmap_regions import regions, QNAMES
    R = regions(Ldir, lon, lat)     # dict of (NR, NC) arrays: cove, inner, north, QUAD
"""
import pickle

import numpy as np
import pandas as pd

QNAMES = ['inner-N', 'inner-S', 'outer-N', 'outer-S']
# LO_output/section_lines/pc_ew.p, west -> east
PC_EW_X = np.array([-122.734335, -122.692975, -122.672886, -122.654688])
PC_EW_Y = np.array([48.220924, 48.230382, 48.231486, 48.236057])


def pc_ew_line(Ldir):
    fn = Ldir['LOo'] / 'section_lines' / 'pc_ew.p'
    if fn.is_file():
        L = pd.read_pickle(fn)
        o = np.argsort(L.x.values.astype(float))
        x, y = L.x.values.astype(float)[o], L.y.values.astype(float)[o]
        if not (np.allclose(x, PC_EW_X, atol=1e-5) and np.allclose(y, PC_EW_Y, atol=1e-5)):
            raise ValueError('%s differs from the pc_ew points in pcmap_regions.py -- '
                             'update PC_EW_X/Y if the line was redrawn on purpose' % fn)
    return PC_EW_X, PC_EW_Y


def regions(Ldir, lon, lat, gctag='wb1_pc1'):
    NR, NC = lon.shape
    seg = pickle.load(open(sorted((Ldir['LOo'] / 'extract' / 'tef2').glob(
        'seg_info_dict_%s_*.p' % gctag))[0], 'rb'))

    def seg_mask(names):
        m = np.zeros((NR, NC), dtype=bool)
        for s in names:
            a = np.array(seg[s]['ji_list'])
            m[a[:, 0], a[:, 1]] = True
        return m

    cove = seg_mask(['pc_cp_m', 'pc_cp_p', 'pc_lp_m'])
    inner = seg_mask(['pc_cp_m'])
    lx, ly = pc_ew_line(Ldir)
    north = cove & (lat > np.interp(lon, lx, ly))
    QUAD = np.full((NR, NC), -1, dtype=int)
    QUAD[cove] = (2 * (~inner) + (~north))[cove]
    return dict(cove=cove, inner=inner, north=north, QUAD=QUAD, line=(lx, ly))
