"""
Origin regions and seeding densities for pcsrc, the Penn Cove source-water
experiment -- the ONE definition imported by LO_user/tracker2/experiments.py
(to seed) and by every pcsrc analysis script (to tag and weight particles), so
the two cannot drift apart.

Seeded domain = wb.p, i.e. Whidbey Basin north of to_mb. Regions, in priority
order (a cell belongs to the first one that claims it):

  cove   tef2 wb1_pc1 segments pc_cp_m + pc_cp_p + pc_lp_m (landward of pc_lp)
  spu    upper Saratoga = sp_head polygon minus the cove
  skb    Skagit Bay = wb_north minus sp_head
  south  the rest of wb.p: lower Saratoga + Port Susan + Possession

SEED gives each region's vertical spacing DZ (m) and horizontal stride (keep a
cell only if j % stride == 0 and i % stride == 0). A kept cell gets
n = max(1, round(h/DZ)) particles at sigma centres -(k+0.5)/n, the rule used
by ic_from_tef2_segs, so the cove cohort is identical to pcmap's. Each
particle stands for h * A * stride**2 / n of water (see particle_volume).

Adopted 2026-10-12 ("lean A"): ~13k particles per release.

The polygon points are copied in below because apogee may not have
LO_output/section_lines; where the files exist they are checked against them.

use (from LO_user/DM_scripts, or with that folder on sys.path):
    import pcsrc_regions as pr
    R = pr.regions(Ldir, lon, lat, wet)   # dict of (NR, NC) bool masks + REGID
"""
import pickle

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

RNAMES = ['cove', 'spu', 'skb', 'south']
RLABELS = {'cove': 'Penn Cove', 'spu': 'Upper Saratoga', 'skb': 'Skagit Bay',
           'south': 'Lower Saratoga + Port Susan + Possession'}
SEED = {'cove': dict(DZ=2, stride=1),
        'spu': dict(DZ=4, stride=2),
        'skb': dict(DZ=4, stride=3),
        'south': dict(DZ=8, stride=4)}
COVE_SEGS = ['pc_cp_m', 'pc_cp_p', 'pc_lp_m']

# LO_output/section_lines/<name>.p as of 2026-10-12
POLYS = {
    'wb': (np.array([-122.643899, -122.587872, -122.538310, -122.512451, -122.445649, -122.346524, -122.352989, -122.335749, -122.290497, -122.165513, -122.192449, -122.219385, -122.281877, -122.298039, -122.389622, -122.441339, -122.495212, -122.527535, -122.557704, -122.572788, -122.584640, -122.592182, -122.638512, -122.707469, -122.747334, -122.740870, -122.682688, -122.643899, -122.643899]),
           np.array([48.432985, 48.460293, 48.457418, 48.422206, 48.389868, 48.305790, 48.238959, 48.187938, 48.101704, 48.054994, 47.987444, 47.951513, 47.934267, 47.901929, 47.944327, 48.010440, 48.014752, 48.003254, 48.014752, 48.067929, 48.108171, 48.151288, 48.182189, 48.195124, 48.217401, 48.245427, 48.305790, 48.384119, 48.432985])),
    'wb_north': (np.array([-122.643899, -122.587872, -122.539387, -122.512451, -122.446727, -122.346524, -122.352989, -122.396087, -122.453191, -122.494134, -122.503831, -122.638512, -122.707469, -122.748412, -122.739792, -122.682688, -122.644977, -122.642822, -122.643899]),
                 np.array([48.432985, 48.461011, 48.458137, 48.422925, 48.389868, 48.305790, 48.238959, 48.235366, 48.251176, 48.227461, 48.185063, 48.182907, 48.195124, 48.218119, 48.246145, 48.306509, 48.384119, 48.432985, 48.432985])),
    'sp_head': (np.array([-122.526307, -122.454349, -122.494927, -122.502501, -122.638302, -122.707015, -122.748134, -122.741100, -122.684291, -122.527389]),
                np.array([48.302445, 48.253730, 48.226666, 48.184446, 48.182642, 48.194550, 48.217283, 48.245791, 48.306053, 48.301723])),
}


def poly_xy(Ldir, name):
    x, y = POLYS[name]
    fn = Ldir['LOo'] / 'section_lines' / (name + '.p')
    if fn.is_file():
        d = pd.read_pickle(fn)
        if not (np.allclose(d.x.values.astype(float), x, atol=1e-5)
                and np.allclose(d.y.values.astype(float), y, atol=1e-5)):
            raise ValueError('%s differs from the points in pcsrc_regions.py -- '
                             'update POLYS if the polygon was redrawn on purpose' % fn)
    return x, y


def cove_mask(Ldir, shape, gctag='wb1_pc1'):
    seg_fn = sorted((Ldir['LOo'] / 'extract' / 'tef2').glob('seg_info_dict_%s_*.p' % gctag))[0]
    seg = pickle.load(open(seg_fn, 'rb'))
    m = np.zeros(shape, dtype=bool)
    for s in COVE_SEGS:
        a = np.array(seg[s]['ji_list'])
        m[a[:, 0], a[:, 1]] = True
    return m


def regions(Ldir, lon, lat, wet):
    def inside(name):
        x, y = poly_xy(Ldir, name)
        p = MplPath(np.c_[x, y])
        return p.contains_points(np.c_[lon.ravel(), lat.ravel()]).reshape(lon.shape) & wet

    in_wb, in_wbn, in_sph = inside('wb'), inside('wb_north'), inside('sp_head')
    R = {}
    R['cove'] = cove_mask(Ldir, lon.shape) & wet
    R['spu'] = in_sph & ~R['cove']
    R['skb'] = in_wbn & ~in_sph & ~R['cove']
    R['south'] = in_wb & ~in_wbn & ~in_sph & ~R['cove']
    REGID = np.full(lon.shape, -1, dtype=int)
    for k, rn in enumerate(RNAMES):
        REGID[R[rn]] = k
    R['REGID'] = REGID
    R['in_wb'] = in_wb
    return R


def seed_mask(R, rn):
    """Cells of region rn that get particles (the horizontal stride)."""
    st = SEED[rn]['stride']
    jj, ii = np.indices(R[rn].shape)
    return R[rn] & (jj % st == 0) & (ii % st == 0)


def n_levels(h, rn):
    return np.maximum(1, np.round(h / SEED[rn]['DZ'])).astype(int)


def particle_volume(h, A, rn):
    """Volume (m3) each particle in a seeded cell of region rn stands for."""
    return h * A * SEED[rn]['stride'] ** 2 / n_levels(h, rn)


def release_tide_vol(P, t0, Vseg, h, A, R):
    """Particle volumes at the RELEASE tide (analysis only).

    The reduce stores vol at mean sea level (h * A * stride**2 / n). A release
    at low water holds less water than that (the E set starts at zeta down to
    -1.7 m), so labelled volume would exceed the target volume. Rescale by
    (h + zeta0) / h with zeta0 the segment-mean sea level at t0 from the tef2
    segment volumes Vseg (hourly DataFrame): the cove segments for the cove
    cohort, pc_lp_p for upper Saratoga. Same rule as 20261013_pcsrc_source.py.
    """
    segs = {'cove': COVE_SEGS, 'spu': ['pc_lp_p']}
    w = P.vol.values.copy()
    for rn in ['cove', 'spu']:
        m = (P.cohort == rn).values
        if not m.any():
            continue
        V0 = Vseg[segs[rn]].sum(axis=1).reindex([t0], method='nearest').values[0]
        zeta0 = (V0 - (h[R[rn]] * A[R[rn]]).sum()) / A[R[rn]].sum()
        hh = h[P.j0.values[m], P.i0.values[m]]
        w[m] *= np.maximum(hh + zeta0, 0) / hh
    return w
