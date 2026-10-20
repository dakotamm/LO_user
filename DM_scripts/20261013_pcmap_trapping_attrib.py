"""
Eddy trapping vs slow-flow trapping in Penn Cove: does looping in the gyre
keep a particle in the cove longer than its position and drift speed explain?

LANDMARK DESIGN (looping takes days, so comparing loopers with everyone would
just say "particles that stayed long enough to loop stayed long"):
  at each landmark tau (-taus, days), take only particles that have NOT left
  the cove yet (hourly first_exit_h > tau, from 20261005_pcmap_reduce.py), and
  describe each by what it did in [0, tau]:
    looped     |loops| >= 1 on its tidal-day-mean path up to tau (same turning
               rule as 20261013_pcmap_looping.py: headings from 12-h steps of
               >= -dmin m, + = counterclockwise)
    d_tau_km   along-water distance to pc_lp from its tidal-day-mean position
               at tau (nearest cove cell)
    spd_cm_s   residual drift speed over [0, tau] = residual path length / time
               (the slow-velocity process)
  outcome: still inside at tau + -dt days (never left: first_exit_h > that).

Output: LO_output/DM_outs/20261013_pcmap_trapping_attrib/
  pcmap_trapping_distance.png   P(still inside at tau + dt | inside at tau) vs
      d_tau, loopers vs non-loopers; rows = surface / bottom starters,
      columns = tau
  pcmap_trapping_speed.png      the same vs residual drift speed
  pcmap_trapping_logit.csv      logistic regression per tau x half:
      still ~ looped + d_tau_km + spd_cm_s, odds ratios with 95% CIs
      clustered by release
  pcmap_trapping_share.csv      per tau x half: n at risk, fraction loopers,
      P(still) loopers / non-loopers, and the share of the particles still
      inside at tau + dt that had looped

  python 20261013_pcmap_trapping_attrib.py
  python 20261013_pcmap_trapping_attrib.py -taus 2,4,6 -dt 2
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import statsmodels.api as sm

from lo_tools import Lfun
from pcmap_regions import regions
from pcmap_tidalday import load, dist_to_mouth, landmarks

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-taus', default='3,5,7', help='landmark times [d]')
p.add_argument('-dt', type=float, default=3, help='follow-up after the landmark [d]')
p.add_argument('-dmin', type=float, default=100, help='min 12-h residual step for a heading [m]')
p.add_argument('-nmin', type=int, default=30, help='min particles per plotted bin')
args = p.parse_args()
TAUS = [float(t) for t in args.taus.split(',')]

Ldir = Lfun.Lstart(gridname='wb1')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261013_pcmap_trapping_attrib'
Lfun.make_dir(out_dir)
M_PER_DEG = 111.32e3
LAT0 = 48.23

g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat = g.lon_rho.values, g.lat_rho.values
DX, DY = 1 / g.pm.values, 1 / g.pn.values
g.close()
lon_ax, lat_ax = lon[0], lat[:, 0]
NR, NC = lon.shape
cove = regions(Ldir, lon, lat)['cove']
dist = dist_to_mouth(cove, DX, DY, np.flatnonzero(cove.any(axis=0))[-1])
# every cell gets the distance of its nearest cove cell (tidal-mean positions can sit over land cells)
jn, in_ = distance_transform_edt(~cove, return_distances=False, return_indices=True)
dist_f = dist[jn, in_]
LON_MOUTH = pd.read_pickle(Ldir['LOo'] / 'section_lines' / 'pc_lp.p').x.mean()

REL, T = load(Ldir, args.gtx)
print('%d releases' % len(REL))


A = landmarks(REL, TAUS, args.dt, args.dmin, LON_MOUTH, dist_f, lon_ax, lat_ax, lat0=LAT0)
print('%d particle-landmarks' % len(A))

# ------------------------------------------------------- share + logistic ---
share, logit = [], []
for tau in TAUS:
    for half, hs in [('surf', True), ('bot', False)]:
        a = A[(A.tau == tau) & (A.surf0 == hs)]
        share.append(dict(tau_d=tau, half=half, n_at_risk=len(a), frac_looped=a.looped.mean(),
                          p_still_loopers=a.still[a.looped].mean(), p_still_nonloopers=a.still[~a.looped].mean(),
                          share_of_still_that_looped=a.looped[a.still].mean(),
                          med_d_tau_loopers=a.d_tau_km[a.looped].median(), med_d_tau_nonloopers=a.d_tau_km[~a.looped].median(),
                          med_spd_loopers=a.spd_cm_s[a.looped].median(), med_spd_nonloopers=a.spd_cm_s[~a.looped].median()))
        Xm = sm.add_constant(a[['looped', 'd_tau_km', 'spd_cm_s']].astype(float))
        fit = sm.GLM(a.still.astype(float), Xm, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': a.rel.values})
        ci = fit.conf_int()
        for v, lab in [('looped', 'looped (yes vs no)'), ('d_tau_km', 'distance to mouth (+1 km)'), ('spd_cm_s', 'residual speed (+1 cm/s)')]:
            logit.append(dict(tau_d=tau, half=half, term=lab, odds_ratio=np.exp(fit.params[v]),
                              ci_lo=np.exp(ci.loc[v, 0]), ci_hi=np.exp(ci.loc[v, 1]), p=fit.pvalues[v]))
S = pd.DataFrame(share)
L = pd.DataFrame(logit)
S.to_csv(out_dir / 'pcmap_trapping_share.csv', index=False, float_format='%.4g')
L.to_csv(out_dir / 'pcmap_trapping_logit.csv', index=False, float_format='%.4g')
pd.set_option('display.width', 220)
print(S.round(3).to_string(index=False))
print(L.round(3).to_string(index=False))

# ------------------------------------------------------------------ figures ---
BINS = {'d_tau_km': (np.arange(0, 6.01, 0.5), 'Distance to mouth at tau [km]', 'distance'),
        'spd_cm_s': (np.arange(0, 6.01, 0.5), 'Residual drift speed over [0, tau] [cm/s]', 'speed')}
for var, (edges, xlab, tag) in BINS.items():
    fig, axs = plt.subplots(2, len(TAUS), figsize=(5 * len(TAUS), 7.5), layout='constrained', sharey=True, squeeze=False)
    xc = 0.5 * (edges[:-1] + edges[1:])
    for rr, (hs, lab) in enumerate([(True, 'surface-half starters'), (False, 'bottom-half starters')]):
        for cc, tau in enumerate(TAUS):
            ax = axs[rr, cc]
            a = A[(A.tau == tau) & (A.surf0 == hs)]
            b = np.digitize(a[var], edges) - 1
            for lp, colr, nm in [(True, 'tab:blue', 'looped'), (False, '0.4', 'did not loop')]:
                m = a.looped.values == lp
                pr = np.array([a.still.values[m & (b == q)].mean() if (m & (b == q)).sum() >= args.nmin else np.nan for q in range(len(xc))])
                nn = np.array([(m & (b == q)).sum() for q in range(len(xc))])
                ax.plot(xc, pr, '-o', color=colr, ms=4, label='%s (n=%d)' % (nm, m.sum()))
            ax.grid(color='lightgray', linestyle='--', alpha=0.5)
            ax.set_ylim(0, 1)
            if rr == 0:
                ax.set_title('tau = %g d' % tau, fontsize=11)
            if rr == 1:
                ax.set_xlabel(xlab)
            if cc == 0:
                ax.set_ylabel('%s\nP(still inside at tau + %g d)' % (lab, args.dt))
            ax.legend(fontsize=8, loc='upper left' if var == 'd_tau_km' else 'upper right')
    fig.suptitle('Particles still inside the cove at tau: did looping in [0, tau] keep them longer? (bins with >= %d particles)' % args.nmin)
    fn = out_dir / ('pcmap_trapping_%s.png' % tag)
    fig.savefig(fn, dpi=200, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print('saved ' + str(fn))
