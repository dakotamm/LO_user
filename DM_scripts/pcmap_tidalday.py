"""
Shared loading for the pcmap tidal-day-mean analyses (gyre / eddy trapping).

  from pcmap_tidalday import load, dist_to_mouth, landmarks, FAM, FAM_T
  REL, T = load(Ldir)            # list of per-release dicts, class table
  d = dist_to_mouth(cove, DX, DY, imouth)

load() reads every LO_output/DM_outs/20261013_pcmap_tidalday_reduce/<gtx>/*.p,
keeps only the releases in the class tables written by
20261006_pcmap_retention_bulk.py (the 218 of the every-3rd-lunar-day set; the
8 leftover early-January releases are dropped), and attaches each release's
classes (season, strat, springneap, wind, tideF) and, from
20261005_pcmap_reduce.py, the per-particle first_exit_h and censored flag
(same file name and particle order; checked on the starting cell).
"""
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

FAM = [('all', None, ['all']),
       ('season', 'season', ['Dec-Mar', 'Apr-Jul', 'Aug-Nov']),
       ('springneap', 'springneap', ['neap', 'mid', 'spring']),
       ('strat', 'strat', ['weak', 'mid', 'strong']),
       ('wind', 'wind', ['down-cove', 'mid', 'up-cove']),
       ('tideF', 'tideF', ['semidiurnal', 'mid', 'diurnal'])]
FAM_T = {'all': 'All releases', 'season': 'Season', 'springneap': 'Spring/neap (qprism terciles)',
         'strat': 'Stratification (cove drho terciles)', 'wind': 'Along-cove wind (terciles)',
         'tideF': 'Tide form (diurnal/semidiurnal terciles)'}


def classes(Ldir, gtx='wb1_t0_xn11abbur00'):
    cls_dir = Ldir['LOo'] / 'DM_outs' / '20261006_pcmap_retention_bulk' / gtx
    T = pd.read_csv(cls_dir / 'pcmap_retention_bulk_cove_strat_releases.csv')[['t0', 'set', 'season', 'strat']]
    for fn, col, new in [('tri_SN', 'springneap', 'springneap'), ('wind', 'class_year', 'wind'), ('tideF', 'class_year', 'tideF')]:
        d = pd.read_csv(cls_dir / ('pcmap_retention_bulk_cove_%s_releases.csv' % fn))
        T = T.merge(d[['t0', 'set', col]].rename(columns={col: new}), on=['t0', 'set'])
    T['t0'] = pd.to_datetime(T.t0)
    return T


def load(Ldir, gtx='wb1_t0_xn11abbur00'):
    T = classes(Ldir, gtx)
    key = {(r.t0, r.set): r for r in T.itertuples()}
    tdir = Ldir['LOo'] / 'DM_outs' / '20261013_pcmap_tidalday_reduce' / gtx
    rdir = Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_reduce' / gtx
    REL = []
    for fn in sorted(tdir.glob('pcmap_3d*.p')):
        D = pickle.load(open(fn, 'rb'))
        m = D['meta']
        if (m['t0'], m['set']) not in key:
            continue
        r = key[(m['t0'], m['set'])]
        D.update({c: getattr(r, c) for c in ['season', 'strat', 'springneap', 'wind', 'tideF']})
        P = pickle.load(open(rdir / fn.name, 'rb'))['P']
        assert (P.i0.values == D['i0']).all() and (P.j0.values == D['j0']).all(), fn.name
        D['first_exit_h'] = P.first_exit_h.values
        D['censored'] = P.censored.values
        D['name'] = m['dir']
        REL.append(D)
    assert len(REL) == len(T), '%d of %d releases found' % (len(REL), len(T))
    return REL, T


def dist_to_mouth(cove, DX, DY, imouth):
    """Along-water distance [m] from each cove cell to the u-face east of rho
    column imouth: 8-connected cells, no corner cutting across land."""
    NR, NC = cove.shape
    idx = -np.ones((NR, NC), dtype=int)
    N = cove.sum()
    idx[cove] = np.arange(N)
    r0, c0, w0 = [], [], []
    for dj, di in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        for j, i in zip(*np.nonzero(cove)):
            j2, i2 = j + dj, i + di
            if not (0 <= j2 < NR and 0 <= i2 < NC and cove[j2, i2]):
                continue
            if dj and di and not (cove[j, i2] and cove[j2, i]):
                continue
            r0.append(idx[j, i]); c0.append(idx[j2, i2])
            w0.append(np.hypot(0.5 * (DX[j, i] + DX[j2, i2]) * di, 0.5 * (DY[j, i] + DY[j2, i2]) * dj))
    jm = np.flatnonzero(cove[:, imouth])
    r0 += [N] * len(jm); c0 += list(idx[jm, imouth]); w0 += list(0.5 * DX[jm, imouth])
    G = coo_matrix((w0, (r0, c0)), shape=(N + 1, N + 1)).tocsr()
    d = np.full((NR, NC), np.nan)
    d[cove] = dijkstra(G, directed=False, indices=N)[:N]
    return d


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def landmarks(REL, taus, dt, dmin, lon_mouth, dist_f, lon_ax, lat_ax, quad_f=None, lat0=48.23):
    """Particles still in the cove (hourly first_exit_h > tau) at each landmark
    tau [d], described by their tidal-day-mean path up to tau:
      loops      signed turning / 2pi (headings from 12-h steps >= dmin m; + = CCW)
      d_tau_km   along-water distance to the mouth at the tidal-mean position at
                 tau (dist_f = distance field filled to every cell; 0 east of the mouth)
      spd_cm_s   residual path length / time over [first window, tau]
      lon/lat/cs_tau   that tidal-mean position; quad_tau if quad_f is given
      still      first_exit_h > tau + dt
    One row per particle x tau; 'rel' indexes REL, 'p' the particle."""
    M = 111.32e3
    NR, NC = dist_f.shape
    rows = []
    for r, D in enumerate(REL):
        t = D['t_h'].astype(float)
        X = (D['lon'].astype(float) - lon_mouth) * M * np.cos(np.deg2rad(lat0))
        Y = (D['lat'].astype(float) - lat0) * M
        dx, dy = np.diff(X, axis=0), np.diff(Y, axis=0)
        ds = np.hypot(dx, dy)
        hd = np.where(ds >= dmin, np.arctan2(dy, dx), np.nan)
        hprev = pd.DataFrame(hd).ffill().shift(1).values
        cturn = np.concatenate([np.zeros((1, X.shape[1])), np.cumsum(np.nan_to_num(_wrap(hd - hprev)), axis=0)])
        cpath = np.concatenate([np.zeros((1, X.shape[1])), np.cumsum(ds, axis=0)])
        fe = D['first_exit_h']
        for tau in taus:
            k = int(np.flatnonzero(t <= 24 * tau)[-1])
            p = np.flatnonzero(fe > 24 * tau)
            if p.size == 0:
                continue
            lo, la = D['lon'][k, p].astype(float), D['lat'][k, p].astype(float)
            i = np.clip(np.round((lo - lon_ax[0]) / (lon_ax[1] - lon_ax[0])).astype(int), 0, NC - 1)
            j = np.clip(np.round((la - lat_ax[0]) / (lat_ax[1] - lat_ax[0])).astype(int), 0, NR - 1)
            d = dict(rel=r, tau=tau, t_tau_h=t[k], p=p, surf0=D['cs0'][p] >= -0.5, quad0=D['quad0'][p],
                     loops=cturn[k, p] / (2 * np.pi), d_tau_km=np.where(lo > lon_mouth, 0.0, dist_f[j, i]) / 1e3,
                     spd_cm_s=100 * cpath[k, p] / ((t[k] - t[0]) * 3600), lon_tau=lo, lat_tau=la,
                     cs_tau=D['cs'][k, p].astype(float), still=fe[p] > 24 * (tau + dt))
            if quad_f is not None:
                d['quad_tau'] = quad_f[j, i]
            rows.append(pd.DataFrame(d))
    A = pd.concat(rows, ignore_index=True)
    A['looped'] = A.loops.abs() >= 1
    return A
