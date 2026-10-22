"""
Do Penn Cove particles wind around the southern-lobe eddy centre? Counts each
particle's winding number about a centre from its RAW HOURLY positions (no
tidal filter: tidal excursion in the lobe is ~100-200 m, about one cell, so the
tidal wobble adds no net winding unless it encircles the centre, which -rmin
excludes). Runs on apogee next to the pcmap tracks (reads lon, lat, cs only).

WINDING about a centre c(t): theta = atan2(y - yc, x - xc), increments
dtheta = wrap(theta[k+1] - theta[k]) summed hour by hour, W = sum / 2pi
(+ = counterclockwise). An hourly increment counts only if, at both ends, the
particle is inside the cove (pcmap_regions quadrants, as in the reduce), its
distance r to the centre is in [-rmin, -rmax], and the centre exists. Only the
segment before the particle FIRST leaves the cove is used (as in
20261013_pcmap_looping.py). A wall-following U-turn past the centre is ~0.5;
a closed loop is 1.

THREE CENTRES (the drift question):
  mov  hourly centroid of the Godin-filtered eddy core
       (20261021_pc_eddy_center.py, band -band, region -region), at the
       particle hour, ONLY in hours flagged present (an eddy event there);
       absent hours add nothing. Follows a drifting eddy.
  rel  the present-hour centre averaged over this release's tracking window
       (NaN, so W_rel = 0, if no present hours)
  fix  the present-hour centre averaged over the whole record (all hours)
If W_mov ~ W_fix the drift does not matter; if loops show up only about mov,
the eddy moves.

PLACEBO / CANDIDATE MAP: the same count about a FIXED centre at every wet cove
rho cell (W_fix at each candidate). Per release and candidate it stores
  n_elig  particles with >= -hmin counted hours about that candidate
  n_ccw   of those, W >= 1      n_cw   W <= -1     sumW  sum of W
A real eddy shows up as a peak in n_ccw / n_elig at its centre and not about
other points the same distance from the walls.

Per release (pickle) in LO_output/DM_outs/20261021_pc_eddy_winding/<region>/<band>/<gtx>/:
  meta   file, t0, set, band, region, rmin, rmax, hmin, circ3d (mean eddy circ
         over the first 3 d), pres3d / pres14 (share of present hours over the
         first 3 / 14 d), fix and rel centres
  P      per particle: j0, i0, quad0, surf0, cs0, first_exit_h, censored,
         W_mov, W_rel, W_fix, Wmax_mov (max |cumulative W_mov|),
         h_mov / h_fix (counted hours about mov / fix), frac_lower (share of those hours in
         the lower 75% of the column)
  cand   j, i, n_elig, n_ccw, n_cw, sumW over the candidate cells

  apogee: python 20261021_pc_eddy_winding.py
  mac:    python 20261021_pc_eddy_winding.py -dir_glob pcret_3d     (test)
  then:   python 20261021_pc_eddy_winding.py -agg                   (summary)
-agg writes, in .../<region>/<band>/:
  winding_releases.csv        per release: presence (pres3d, pres14), share of
                              particles (>= -hmin counted h) with W_fix >= 1, W_mov >= 1
  winding_candidate_maps.png  share winding >= 1 CCW about every fixed cove cell:
                              all / Nov-Dec / Aug-Oct releases (+ = eddy centre)
  winding_vs_presence.png     release share W_fix >= 1 vs share of present hours.
                              The NON-CIRCULAR test: presence comes from the
                              Eulerian field, W_fix about a fixed point ignores it.
"""
import argparse
import pickle

import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun
from pcmap_regions import regions

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-dir_glob', default='pcmap_3d*')
p.add_argument('-days', type=float, default=14.0)
p.add_argument('-band', default='lower', help='eddy-centre band from 20261021_pc_eddy_center.py')
p.add_argument('-region', default='pc_cp', help='eddy-centre search region (its output subfolder)')
p.add_argument('-rmin', type=float, default=200, help='min distance to the centre [m] (tidal excursion)')
p.add_argument('-rmax', type=float, default=1200, help='max distance to the centre [m]')
p.add_argument('-hmin', type=float, default=24, help='min counted hours for the candidate map')
p.add_argument('-no_cand', action='store_true', help='skip the candidate-centre map')
p.add_argument('-clobber', action='store_true')
p.add_argument('-agg', action='store_true', help='only combine the per-release pickles')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
trk = Ldir['LOo'] / 'tracks2' / args.gtx
out_dir = Ldir['LOo'] / 'DM_outs' / '20261021_pc_eddy_winding' / args.region / args.band / args.gtx
Lfun.make_dir(out_dir)
M_PER_DEG = 111.32e3
LAT0 = 48.23
LON_MOUTH = pd.read_pickle(Ldir['LOo'] / 'section_lines' / 'pc_lp.p').x.mean()
TWO_PI = 2 * np.pi


def to_xy(lo, la):
    return ((lo - LON_MOUTH) * M_PER_DEG * np.cos(np.deg2rad(LAT0)),
            (la - LAT0) * M_PER_DEG)


def wrap(a):
    return (a + np.pi) % TWO_PI - np.pi


# --------------------------------------------------------------- agg ---
if args.agg:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fns = sorted(out_dir.glob('*.p'))
    rows, rel, CAND = [], [], {}
    seas = lambda m: 'Nov-Dec' if m in (11, 12) else 'Aug-Oct' if m in (8, 9, 10) else 'other'
    for fn in fns:
        R = pickle.load(open(fn, 'rb'))
        M = R['meta']
        P = R['P'].copy()
        P['release'], P['t0'], P['set'] = fn.stem, M['t0'], M['set']
        P['circ3d'], P['pres3d'], P['pres14'] = M['circ3d'], M['pres3d'], M['pres14']
        rows.append(P)
        s_ = seas(pd.Timestamp(M['t0']).month)
        ef, em = P.h_fix >= args.hmin, P.h_mov >= args.hmin
        rel.append(dict(release=fn.stem, t0=M['t0'], season=s_, pres3d=M['pres3d'], pres14=M['pres14'],
                        circ3d=M['circ3d'], n_fix=int(ef.sum()), ccw_fix=(P.W_fix[ef] >= 1).mean(),
                        cw_fix=(P.W_fix[ef] <= -1).mean(), n_mov=int(em.sum()), ccw_mov=(P.W_mov[em] >= 1).mean()))
        c = R['cand'].set_index(['j', 'i'])[['n_elig', 'n_ccw', 'n_cw', 'sumW']]
        for k in ['all', s_]:
            CAND[k] = c if k not in CAND else CAND[k] + c
    A = pd.concat(rows, ignore_index=True)
    A.to_pickle(out_dir.parent / ('winding_particles_%s.p' % args.gtx))
    REL = pd.DataFrame(rel)
    REL.to_csv(out_dir.parent / 'winding_releases.csv', index=False, float_format='%.4g')
    for k, c in CAND.items():
        c['frac_ccw'], c['frac_cw'] = c.n_ccw / c.n_elig, c.n_cw / c.n_elig
    CAND['all'].reset_index().to_csv(out_dir.parent / ('winding_candidates_%s.csv' % args.gtx), index=False,
                                     float_format='%.5g')
    print('%d releases, %d particles (band %s, region %s)' % (len(fns), len(A), args.band, args.region))
    for lab in ['all', 'Nov-Dec', 'Aug-Oct']:
        r = REL if lab == 'all' else REL[REL.season == lab]
        print('%-8s %3d releases: release-mean share W>=1  fix %.3f  mov %.3f;  W<=-1 fix %.3f'
              % (lab, len(r), r.ccw_fix.mean(), r.ccw_mov.mean(), r.cw_fix.mean()))
    print('corr(release share W_fix>=1, pres14) = %.2f, (pres3d) = %.2f'
          % (REL.ccw_fix.corr(REL.pres14), REL.ccw_fix.corr(REL.pres3d)))

    g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
    glon, glat = g.lon_rho.values, g.lat_rho.values
    g.close()
    cfn = Ldir['LOo'] / 'DM_outs' / '20261021_pc_eddy_center' / args.region / ('pc_eddy_center_%s.csv' % args.band)
    C0 = pd.read_csv(cfn)
    C0 = C0[C0.present.astype(bool)]
    keys = [k for k in ['all', 'Nov-Dec', 'Aug-Oct'] if k in CAND]
    fig, axs = plt.subplots(1, len(keys), figsize=(5.5 * len(keys), 4.2), layout='constrained', squeeze=False)
    for ax, k in zip(axs[0], keys):
        c = CAND[k].reset_index()
        c = c[c.n_elig >= 50]
        F = np.full(glon.shape, np.nan)
        F[c.j.values, c.i.values] = c.frac_ccw.values
        h = ax.pcolormesh(glon, glat, F, cmap='magma_r', vmin=0, vmax=np.nanmax(F), shading='nearest')
        ax.plot(C0.lon_c.mean(), C0.lat_c.mean(), 'c+', ms=14, mew=2.5)
        ax.set_xlim(-122.735, -122.65)
        ax.set_ylim(48.212, 48.245)
        ax.set_aspect(1 / np.cos(np.deg2rad(LAT0)))
        ax.set_title('%s: share of particles winding >= 1 CCW\nabout each fixed cell (n_elig >= 50)' % k, fontsize=10)
        fig.colorbar(h, ax=ax, shrink=0.8)
    fn = out_dir.parent / 'winding_candidate_maps.png'
    fig.savefig(fn, dpi=200, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print('saved ' + str(fn))

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    col = {'Nov-Dec': 'tab:blue', 'Aug-Oct': 'tab:orange', 'other': '0.6'}
    for ax, xv in zip(axs, ['pres3d', 'pres14']):
        for s_, r in REL.groupby('season'):
            ax.scatter(r[xv], r.ccw_fix, s=14, color=col[s_], label=s_, alpha=0.8)
        b = pd.cut(REL[xv], [-0.01, 0.25, 0.5, 0.75, 1.0])
        m = REL.groupby(b, observed=True).ccw_fix.mean()
        ax.plot([iv.mid for iv in m.index], m.values, 'k-o', lw=1.5, ms=5)
        ax.set_xlabel('share of hours with the eddy present (%s)' % xv)
        ax.set_ylabel('share of particles winding >= 1 CCW\nabout the fixed centre (>= %g h counted)' % args.hmin)
        ax.set_title('r = %.2f' % REL.ccw_fix.corr(REL[xv]))
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.suptitle('Do particles wind more when the Eulerian eddy is present? (one dot per release; %s, %s)'
                 % (args.band, args.region))
    fn = out_dir.parent / 'winding_vs_presence.png'
    fig.savefig(fn, dpi=200, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print('saved ' + str(fn))
    raise SystemExit

# ------------------------------------------------------------ centre ---
cfn = Ldir['LOo'] / 'DM_outs' / '20261021_pc_eddy_center' / args.region / ('pc_eddy_center_%s.csv' % args.band)
if not cfn.is_file():
    raise SystemExit('missing %s: run 20261021_pc_eddy_center.py here first (or copy the csv)' % cfn)
CEN = pd.read_csv(cfn, parse_dates=['time']).set_index('time')
CEN.index = CEN.index.tz_localize(None) if CEN.index.tz is not None else CEN.index
CEN['present'] = CEN.present.astype(bool)
XF, YF = CEN.x_c[CEN.present].mean(), CEN.y_c[CEN.present].mean()
print('centre band %s: present-hour mean (%.0f, %.0f) m from the mouth; present %.0f%% of %d hours'
      % (args.band, XF, YF, 100 * CEN.present.mean(), len(CEN)))

# -------------------------------------------------------------- grid ---
g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat = g.lon_rho.values, g.lat_rho.values
g.close()
NR, NC = lon.shape
lon_ax, lat_ax = lon[0, :], lat[:, 0]
dlon, dlat = lon_ax[1] - lon_ax[0], lat_ax[1] - lat_ax[0]
REG = regions(Ldir, lon, lat)
cove, QUAD = REG['cove'], REG['QUAD']
cj, ci = np.nonzero(cove)
CX, CY = to_xy(lon[cj, ci], lat[cj, ci])
print('%d candidate centres (cove rho cells)' % len(cj))


def ji_of(plon, plat):
    ok = np.isfinite(plon) & np.isfinite(plat)
    i = np.zeros(plon.shape, dtype=int)
    j = np.zeros(plon.shape, dtype=int)
    i[ok] = np.clip(np.round((plon[ok] - lon_ax[0]) / dlon), 0, NC - 1).astype(int)
    j[ok] = np.clip(np.round((plat[ok] - lat_ax[0]) / dlat), 0, NR - 1).astype(int)
    return j, i, ok


def winding(X, Y, xc, yc, valid):
    """Hourly winding about (xc, yc) [scalars or (nt,1)]. valid (nt, np) =
    inside the cove and before first exit. Returns cumulative W (nt-1, np) and
    counted-hours mask (nt-1, np)."""
    dx, dy = X - xc, Y - yc
    r = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    ok = valid & (r >= args.rmin) & (r <= args.rmax) & np.isfinite(r)
    use = ok[1:] & ok[:-1]
    d = np.where(use, wrap(np.diff(th, axis=0)), 0.0)
    return np.cumsum(d, axis=0) / TWO_PI, use


# ------------------------------------------------------------- reduce ---
fns = sorted(f for dd in sorted(trk.glob(args.dir_glob)) if dd.is_dir() for f in dd.glob('release_*.nc'))
print('%d release files under %s/%s' % (len(fns), trk.name, args.dir_glob))
nf = int(round(args.days * 24)) + 1
for fn in fns:
    out_fn = out_dir / ('%s__%s.p' % (fn.parent.name, fn.stem))
    if out_fn.is_file() and not args.clobber:
        continue
    d = xr.open_dataset(fn)
    if d.sizes['Time'] < nf:
        print('SKIP %s: %d frames < %d' % (out_fn.name, d.sizes['Time'], nf))
        d.close()
        continue
    sl = slice(0, nf)
    plon, plat, cs = d.lon.values[sl], d.lat.values[sl], d.cs.values[sl]
    ot = pd.to_datetime(d.ot.values[sl])
    d.close()

    j, i, ok = ji_of(plon, plat)
    keep = ok[0] & cove[j[0], i[0]]
    plon, plat, cs, j, i, ok = plon[:, keep], plat[:, keep], cs[:, keep], j[:, keep], i[:, keep], ok[:, keep]
    q = np.where(ok, QUAD[j, i], -1)
    inside = q >= 0
    out_k = np.argmax(~inside[1:], axis=0) + 1
    out_k[inside[1:].all(axis=0)] = nf
    valid = np.arange(nf)[:, None] < out_k[None, :]          # before first exit
    X, Y = to_xy(plon, plat)

    c = CEN.reindex(ot, method='nearest', tolerance=pd.Timedelta('31min'))
    pres = c.present.fillna(False).values.astype(bool)
    xm = np.where(pres, c.x_c.values, np.nan)[:, None]
    ym = np.where(pres, c.y_c.values, np.nan)[:, None]
    xr_, yr_ = (np.nanmean(xm), np.nanmean(ym)) if pres.any() else (np.nan, np.nan)
    Wm, um = winding(X, Y, xm, ym, valid)
    Wr, _ = winding(X, Y, xr_, yr_, valid)
    Wf, uf = winding(X, Y, XF, YF, valid)
    lower = (cs[1:] < -0.25) & um
    P = pd.DataFrame(dict(j0=j[0], i0=i[0], quad0=q[0], surf0=cs[0] >= -0.5, cs0=cs[0],
                          first_exit_h=np.minimum(out_k, nf - 1).astype(float), censored=out_k >= nf,
                          W_mov=Wm[-1], W_rel=Wr[-1], W_fix=Wf[-1], Wmax_mov=np.abs(Wm).max(axis=0),
                          h_mov=um.sum(axis=0).astype(float), h_fix=uf.sum(axis=0).astype(float),
                          frac_lower=lower.sum(axis=0) / np.maximum(um.sum(axis=0), 1)))

    cand = None
    if not args.no_cand:
        rows = []
        for cx, cy, jj, ii in zip(CX, CY, cj, ci):
            W, u = winding(X, Y, cx, cy, valid)
            e = u.sum(axis=0) >= args.hmin
            w = W[-1][e]
            rows.append((jj, ii, int(e.sum()), int((w >= 1).sum()), int((w <= -1).sum()), float(w.sum())))
        cand = pd.DataFrame(rows, columns=['j', 'i', 'n_elig', 'n_ccw', 'n_cw', 'sumW'])

    t0 = ot[0]
    w3 = (CEN.index >= t0) & (CEN.index < t0 + pd.Timedelta('3D'))
    c3, p3 = CEN.circ[w3].mean(), CEN.present[w3].mean()
    meta = dict(file=str(fn), t0=t0, set=fn.parent.name.split('_')[2] if fn.parent.name.count('_') >= 2 else '',
                band=args.band, region=args.region, rmin=args.rmin, rmax=args.rmax, hmin=args.hmin, circ3d=c3,
                pres3d=p3, pres14=pres.mean(),
                fix=(XF, YF), rel=(xr_, yr_), centre_missing_h=int(np.isnan(c.x_c.values).sum()))
    pickle.dump(dict(meta=meta, P=P, cand=cand), open(out_fn, 'wb'))
    e = P.h_mov >= args.hmin
    print('%s: present %.0f%% of 14 d; %d particles, %d with >= %g h about mov; W>=1 mov %.3f rel %.3f fix %.3f; W<=-1 mov %.3f'
          % (out_fn.name, 100 * pres.mean(), len(P), e.sum(), args.hmin, (P.W_mov[e] >= 1).mean(), (P.W_rel[e] >= 1).mean(),
             (P.W_fix[e] >= 1).mean(), (P.W_mov[e] <= -1).mean()), flush=True)
