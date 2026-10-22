"""
DEPTH-MATCHED winding about the inner-cove eddy: for each depth band, a
particle's winding is counted ONLY in hours it is IN that band, about THAT
band's eddy centre, and compared with THAT band's eddy presence.
(20261021_pc_eddy_winding.py counted every hour whatever the particle's depth,
about one band's centre.) Runs on apogee next to the pcmap tracks.

BANDS (same definitions as 20261021_pc_eddy_center.py; -bands):
  z0-4, z4-10, z10-bed   depth below the moving surface, zeta - z [m]
  surf, mid, bot         fractional height above the bed, cs + 1 (top 25%,
                         25-75%, lower 25%)
An hourly increment of a particle's winding counts for band b only if, at both
ends of the hour, the particle is in band b, inside the cove before its first
exit, and -rmin to -rmax from band b's centre. Per particle and band:
  W_fix_<b>, h_fix_<b>   winding [turns] / counted hours about band b's FIXED
                         centre (its present-hour mean)
  W_mov_<b>, h_mov_<b>   the same about band b's MOVING centre, present hours only
  h_in_<b>               hours in band b inside the cove before first exit
Per release meta: pres3d_<b>, pres14_<b> (band b presence share, first 3 / 14 d).
Needs 20261021_pc_eddy_center.py's pc_eddy_center_<b>.csv for every band.

-agg (mac or apogee): per band, the selection-robust RATE (sum W / sum days
counted, all particles with any counted time) in releases with band b's eddy
absent (pres3d_b = 0), partial, present (= 1), present - absent, corr(pres3d_b,
release rate), and the within-start-cell (cell fixed effects) slope of rate on
pres3d_b; 95% block bootstrap over releases (14-d blocks). Figures:
  depth_rate.png    (a) binned release rate vs pres3d_b per band, (b) rate
                    absent / present per band, (c) present - absent per band
  depth_W_hist.png  W_fix_b of particles with >= -hmin h counted, eddy absent
                    vs present, one panel per band

Output: LO_output/DM_outs/20261022_pc_eddy_winding_depth/<region>/<gtx>/ (pickles),
        .../<region>/ (agg)

  apogee: python 20261022_pc_eddy_winding_depth.py
  mac:    python 20261022_pc_eddy_winding_depth.py -dir_glob pcret_3d    (test)
  then:   python 20261022_pc_eddy_winding_depth.py -agg
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
p.add_argument('-bands', default='z0-4,z4-10,z10-bed,surf,mid,bot')
p.add_argument('-region', default='pc_cp')
p.add_argument('-rmin', type=float, default=200)
p.add_argument('-rmax', type=float, default=1200)
p.add_argument('-hmin', type=float, default=24, help='agg histograms: min counted hours')
p.add_argument('-clobber', action='store_true')
p.add_argument('-agg', action='store_true')
p.add_argument('-all_rel', action='store_true', help='agg: keep releases outside the every-3rd-lunar-day set')
p.add_argument('-block_days', type=float, default=14)
p.add_argument('-nboot', type=int, default=500)
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
trk = Ldir['LOo'] / 'tracks2' / args.gtx
base = Ldir['LOo'] / 'DM_outs' / '20261022_pc_eddy_winding_depth' / args.region
out_dir = base / args.gtx
Lfun.make_dir(out_dir)
cdir = Ldir['LOo'] / 'DM_outs' / '20261021_pc_eddy_center' / args.region
M_PER_DEG = 111.32e3
LAT0 = 48.23
LON_MOUTH = pd.read_pickle(Ldir['LOo'] / 'section_lines' / 'pc_lp.p').x.mean()
TWO_PI = 2 * np.pi
BANDS = args.bands.split(',')


def in_band(b, dep, frac):
    """Particle in band b: dep = depth below surface [m], frac = height above bed / column."""
    if b.startswith('z'):
        a, z = b[1:].split('-')
        return (dep >= float(a)) & (dep < (np.inf if z == 'bed' else float(z)))
    return {'surf': frac >= 0.75, 'mid': (frac > 0.25) & (frac < 0.75), 'bot': frac <= 0.25}[b]


# --------------------------------------------------------------- agg ---
if args.agg:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fns = sorted(out_dir.glob('*.p'))
    lfn = Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_release_times' / 'pcmap_release_times_2025_every3.csv'
    if not args.all_rel and lfn.is_file():
        keep = set(pd.read_csv(lfn).out_name)
        fns = [f for f in fns if f.stem.split('__')[0] in keep]
    rows, rel = [], []
    for fn in fns:
        R_ = pickle.load(open(fn, 'rb'))
        P = R_['P'].copy()
        P['release'] = fn.stem
        rows.append(P)
        rel.append(dict(release=fn.stem, t0=R_['meta']['t0'], **{k: v for k, v in R_['meta'].items()
                                                                  if k.startswith('pres')}))
    A = pd.concat(rows, ignore_index=True)
    REL = pd.DataFrame(rel).sort_values('t0').reset_index(drop=True)
    REL['t0'] = pd.to_datetime(REL.t0)
    REL['block'] = ((REL.t0 - REL.t0.min()).dt.total_seconds() // (args.block_days * 86400)).astype(int)
    BLK = [g.index.values for _, g in REL.groupby('block')]
    rng = np.random.default_rng(0)
    print('%d releases, %d particles' % (len(REL), len(A)))

    def boot(stat):
        est = stat(np.arange(len(REL)))
        v = [stat(np.concatenate([BLK[k] for k in rng.integers(0, len(BLK), len(BLK))])) for _ in range(args.nboot)]
        return est, np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)

    out, RR, CC = [], {}, {}
    for b in BANDS:
        X = A[A['h_fix_' + b] > 0]
        g = X.groupby('release').agg(sumW=('W_fix_' + b, 'sum'), hrs=('h_fix_' + b, 'sum'))
        R = REL.set_index('release').join(g).reset_index()
        R['days'] = R.hrs / 24
        R['rate'] = R.sumW / R.days
        pc = 'pres3d_' + b
        RR[b] = R
        C = X.groupby(['release', 'j0', 'i0']).agg(sumW=('W_fix_' + b, 'sum'), hrs=('h_fix_' + b, 'sum')).reset_index()
        C = C.merge(R[['release', pc]], on='release')
        C['days'], C['rate'] = C.hrs / 24, C.sumW / (C.hrs / 24)
        Cr = {r: d for r, d in C.groupby('release')}

        def pooled(ix, lo, hi, R=R, pc=pc):
            s = R.iloc[ix]
            s = s[(s[pc] >= lo) & (s[pc] <= hi)]
            return s.sumW.sum() / s.days.sum() if s.days.sum() > 0 else np.nan

        def fe(ix, R=R, Cr=Cr, pc=pc):
            d = pd.concat([Cr[r] for r in R.release.values[ix] if r in Cr], ignore_index=True)
            k = d.j0 * 1000 + d.i0
            sw = d.days.groupby(k).transform('sum')
            xd = d[pc] - (d.days * d[pc]).groupby(k).transform('sum') / sw
            yd = d.rate - (d.days * d.rate).groupby(k).transform('sum') / sw
            den = (d.days * xd * xd).sum()
            return (d.days * xd * yd).sum() / den if den > 0 else np.nan

        ab, pr = boot(lambda ix: pooled(ix, 0, 0.001)), boot(lambda ix: pooled(ix, 0.999, 1))
        pa = boot(lambda ix: pooled(ix, 0.0011, 0.9989))
        df = boot(lambda ix: pooled(ix, 0.999, 1) - pooled(ix, 0, 0.001))
        rr = boot(lambda ix: R.iloc[ix][pc].corr(R.iloc[ix].rate))
        sl = boot(fe)
        out.append(dict(band=b, n_absent=int((R[pc] <= 0.001).sum()), n_present=int((R[pc] >= 0.999).sum()),
                        days_total=R.days.sum(), rate_absent=ab[0], ra_lo=ab[1], ra_hi=ab[2],
                        rate_partial=pa[0], rate_present=pr[0], rp_lo=pr[1], rp_hi=pr[2],
                        diff=df[0], diff_lo=df[1], diff_hi=df[2], r=rr[0], r_lo=rr[1], r_hi=rr[2],
                        fe_slope=sl[0], fe_lo=sl[1], fe_hi=sl[2]))
        print(out[-1]['band'], {k: round(v, 3) for k, v in out[-1].items() if k != 'band'}, flush=True)
    S = pd.DataFrame(out)
    S.to_csv(base / 'depth_rate_summary.csv', index=False, float_format='%.4g')
    txt = ['DEPTH-MATCHED WINDING RATE (turns/day; particles counted only while in the band, about that band\'s '
           'centre, vs that band\'s presence) -- %d releases; 95%% block bootstrap (%g-d blocks)'
           % (len(REL), args.block_days), '', S.to_string(index=False, float_format=lambda v: '%.3f' % v)]
    (base / 'depth_rate_summary.txt').write_text('\n'.join(txt) + '\n')
    print('\n'.join(txt))

    col = {'z0-4': '#9ecae1', 'z4-10': '#3182bd', 'z10-bed': '#08306b',
           'surf': '#fdae6b', 'mid': '#e6550d', 'bot': '#7f2704'}
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), layout='constrained')
    for b in BANDS:
        R = RR[b]
        bins = pd.cut(R['pres3d_' + b], [-0.01, 0.001, 0.25, 0.5, 0.75, 0.999, 1.0])
        m = R.groupby(bins, observed=True).rate.median()
        xs = [0 if iv.right <= 0.001 else 1 if iv.left >= 0.999 else iv.mid for iv in m.index]
        axs[0].plot(xs, m.values, '-o', color=col.get(b, 'k'), lw=2, ms=4,
                    ls='-' if b.startswith('z') else '--', label='%s (r = %.2f)' % (b, R['pres3d_' + b].corr(R.rate)))
    axs[0].set_xlabel("share of the first 3 d with the band's eddy present")
    axs[0].set_ylabel('release winding rate in the band, bin median [turns/day]')
    axs[0].set_title('(a) rate vs presence, band by band', fontsize=10, loc='left')
    axs[0].legend(fontsize=8, frameon=False)
    y = np.arange(len(S))[::-1]
    for dy, (k, lo, hi, c, lab) in zip([0.15, -0.15], [('rate_absent', 'ra_lo', 'ra_hi', '#e8590c', 'eddy absent'),
                                                       ('rate_present', 'rp_lo', 'rp_hi', '#2a6fdb', 'eddy present')]):
        axs[1].errorbar(S[k], y + dy, xerr=[S[k] - S[lo], S[hi] - S[k]], fmt='o', color=c, capsize=3, label=lab)
    axs[2].errorbar(S['diff'], y, xerr=[S['diff'] - S.diff_lo, S.diff_hi - S['diff']], fmt='o', color='k', capsize=3)
    for ax in axs[1:]:
        ax.set_yticks(y)
        ax.set_yticklabels(['%s (abs %d / pres %d)' % (b, a, q) for b, a, q in zip(S.band, S.n_absent, S.n_present)],
                           fontsize=8)
        ax.axvline(0, color='0.3', lw=0.8, ls=':')
    axs[1].set_xlabel('pooled winding rate [turns/day], 95% CI')
    axs[1].set_title('(b) rate with the band\'s eddy absent / present', fontsize=10, loc='left')
    axs[1].legend(fontsize=8, frameon=False)
    axs[2].set_xlabel('present - absent [turns/day], 95% CI')
    axs[2].set_title('(c) difference', fontsize=10, loc='left')
    for ax in axs:
        ax.grid(color='0.9', lw=0.6)
    fig.suptitle('Depth-matched winding about the inner-cove eddy (particles counted only while in each band, about '
                 'that band\'s centre)', fontsize=10)
    fn = base / 'depth_rate.png'
    fig.savefig(fn, dpi=200, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print('saved ' + str(fn))

    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5), layout='constrained', sharex=True)
    hb = np.arange(-2, 3.05, 0.1)
    for ax, b in zip(axs.ravel(), BANDS):
        P = A.merge(RR[b][['release', 'pres3d_' + b]], on='release')
        P = P[P['h_fix_' + b] >= args.hmin]
        for lo, hi, c, lab in [(0, 0.001, '#e8590c', 'absent'), (0.999, 1, '#2a6fdb', 'present')]:
            w = P['W_fix_' + b][(P['pres3d_' + b] >= lo) & (P['pres3d_' + b] <= hi)].values
            if len(w) == 0:
                continue
            ax.hist(np.clip(w, -2 + 1e-9, 3 - 1e-9), bins=hb, density=True, histtype='step', color=c, lw=2,
                    label='eddy %s: %d particles, median %+.2f' % (lab, len(w), np.median(w)))
            ax.axvline(np.median(w), color=c, lw=1)
        for x in [0, 1]:
            ax.axvline(x, color='0.3', lw=0.8, ls=':')
        ax.set_title(b, fontsize=10, loc='left')
        ax.legend(fontsize=7, frameon=False, loc='upper right')
        ax.grid(color='0.9', lw=0.6)
    for ax in axs[-1]:
        ax.set_xlabel('W in the band [turns]')
    fig.suptitle('Depth-matched W of particles with >= %g h counted in the band, by the band\'s eddy presence '
                 '(first 3 d)' % args.hmin, fontsize=10)
    fn = base / 'depth_W_hist.png'
    fig.savefig(fn, dpi=200, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print('saved ' + str(fn))
    raise SystemExit

# ------------------------------------------------------------ centres ---
CEN, FIX = {}, {}
for b in BANDS:
    C = pd.read_csv(cdir / ('pc_eddy_center_%s.csv' % b), usecols=['time', 'x_c', 'y_c', 'present'],
                    parse_dates=['time']).set_index('time')
    C['present'] = C.present.astype(bool)
    CEN[b] = C
    FIX[b] = (C.x_c[C.present].mean(), C.y_c[C.present].mean())
    print('band %-8s fixed centre (%.0f, %.0f) m; present %.0f%%' % (b, *FIX[b], 100 * C.present.mean()))

g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat = g.lon_rho.values, g.lat_rho.values
g.close()
NR, NC = lon.shape
lon_ax, lat_ax = lon[0, :], lat[:, 0]
dlon, dlat = lon_ax[1] - lon_ax[0], lat_ax[1] - lat_ax[0]
REG = regions(Ldir, lon, lat)
cove, QUAD = REG['cove'], REG['QUAD']


def ji_of(plon, plat):
    ok = np.isfinite(plon) & np.isfinite(plat)
    i = np.zeros(plon.shape, dtype=int)
    j = np.zeros(plon.shape, dtype=int)
    i[ok] = np.clip(np.round((plon[ok] - lon_ax[0]) / dlon), 0, NC - 1).astype(int)
    j[ok] = np.clip(np.round((plat[ok] - lat_ax[0]) / dlat), 0, NR - 1).astype(int)
    return j, i, ok


def winding(X, Y, xc, yc, valid):
    """Winding [turns] and counted hours about (xc, yc); valid (nt, np) = hour usable at that end."""
    dx, dy = X - xc, Y - yc
    r = np.hypot(dx, dy)
    ok = valid & (r >= args.rmin) & (r <= args.rmax) & np.isfinite(r)
    use = ok[1:] & ok[:-1]
    d = np.where(use, (np.diff(np.arctan2(dy, dx), axis=0) + np.pi) % TWO_PI - np.pi, 0.0)
    return d.sum(axis=0) / TWO_PI, use.sum(axis=0).astype(float)


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
    z, zeta = d.z.values[sl], d.zeta.values[sl]
    ot = pd.to_datetime(d.ot.values[sl])
    d.close()
    j, i, ok = ji_of(plon, plat)
    keep = ok[0] & cove[j[0], i[0]]
    plon, plat, cs, z, zeta = plon[:, keep], plat[:, keep], cs[:, keep], z[:, keep], zeta[:, keep]
    j, i, ok = j[:, keep], i[:, keep], ok[:, keep]
    inside = np.where(ok, QUAD[j, i], -1) >= 0
    out_k = np.argmax(~inside[1:], axis=0) + 1
    out_k[inside[1:].all(axis=0)] = nf
    before = np.arange(nf)[:, None] < out_k[None, :]
    X = (plon - LON_MOUTH) * M_PER_DEG * np.cos(np.deg2rad(LAT0))
    Y = (plat - LAT0) * M_PER_DEG
    dep, frac = zeta - z, cs + 1
    P = pd.DataFrame(dict(j0=j[0], i0=i[0], quad0=np.where(ok[0], QUAD[j[0], i[0]], -1), cs0=cs[0],
                          first_exit_h=np.minimum(out_k, nf - 1).astype(float)))
    meta = dict(file=str(fn), t0=ot[0], rmin=args.rmin, rmax=args.rmax, bands=BANDS)
    t0 = ot[0]
    msg = []
    for b in BANDS:
        inb = in_band(b, dep, frac) & before
        c = CEN[b].reindex(ot, method='nearest', tolerance=pd.Timedelta('31min'))
        pres = c.present.fillna(False).values.astype(bool)
        xm = np.where(pres, c.x_c.values, np.nan)[:, None]
        ym = np.where(pres, c.y_c.values, np.nan)[:, None]
        P['W_fix_' + b], P['h_fix_' + b] = winding(X, Y, FIX[b][0], FIX[b][1], inb)
        P['W_mov_' + b], P['h_mov_' + b] = winding(X, Y, xm, ym, inb)
        P['h_in_' + b] = inb.sum(axis=0).astype(float)
        w3 = (CEN[b].index >= t0) & (CEN[b].index < t0 + pd.Timedelta('3D'))
        meta['pres3d_' + b], meta['pres14_' + b] = CEN[b].present[w3].mean(), pres.mean()
        hd = P['h_fix_' + b].sum() / 24
        msg.append('%s %.2f/d p%.0f%%' % (b, P['W_fix_' + b].sum() / hd if hd > 0 else np.nan,
                                         100 * meta['pres3d_' + b]))
    pickle.dump(dict(meta=meta, P=P), open(out_fn, 'wb'))
    print('%s: %d particles | rate, pres3d: %s' % (out_fn.name, len(P), '; '.join(msg)), flush=True)
