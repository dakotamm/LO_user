"""
Does Penn Cove have a BAROCLINIC LATERAL EXCHANGE?

THE QUESTION, MADE MEASURABLE
A textbook estuary exchanges VERTICALLY: in at depth, out at the surface. A
lateral exchange instead puts inflow on one side of the mouth and outflow on
the other. [[pc-lp-mouth-points]] already showed Penn Cove does the latter --
in on the north, out on the south. This script asks the next question: is that
lateral exchange BAROCLINIC, i.e. does it live in the density field, or is it
a depth-uniform slab that rotation, wind or bathymetry could equally produce?

Three tests, in order.

1. HOW LATERAL IS THE EXCHANGE AT ALL. The area-weighted velocity field
   u(t,z,p) = -q/(dd*DZ) (negated so + = INTO the cove) is split into

       u = U_net(t) + U_lat(t,p) + U_vert(t,z) + U_resid(t,z,p)

   a section mean, a depth-mean-per-face anomaly (the LATERAL mode), a
   width-mean-per-level anomaly (the VERTICAL mode) and what is left. The
   area-weighted variance of each mode is its share of the exchange. Run three
   ways -- on the 2-year mean field, on Godin-filtered hourly, and on raw
   hourly -- because the answer is different for the residual and the tide.

2. IS THE LATERAL MODE DEPTH-VARYING. The lateral field is split again into a
   depth-uniform part (each face's depth mean) and a depth-varying part

       u_bc(t,z,p) = u - <u>_z(t,p) - <u - <u>_z>_p(t,z)

   which has zero depth mean at every face AND zero width mean at every level,
   so it is purely the LATERAL structure of the vertical shear -- a lateral
   overturning signature and nothing else. Its transport is
   QLAT_BC = 0.5*sum|u_bc*A| / 2.

3. WHAT SUPPORTS IT. Two competing supports for a depth-varying lateral flow:
   (a) rotation, via thermal wind against the cross-mouth density gradient,
       du/dz = -(g/(f*rho0)) drho/dy   [sign for u = INTO the cove]
       tested face by face on a common z grid, on the lateral anomaly only, so
       the section-wide estuarine shear cannot inflate the fit; and
   (b) a non-rotating lateral gravitational circulation, whose fingerprint is
       that QLAT_BC scales with STRATIFICATION and NOT with tidal strength.
   Wind is the confounder -- it drives lateral flow directly -- so every
   density correlation is also reported as a partial correlation controlling
   for both wind components.

WHY THE h CONFOUND IS HANDLED. pc_lp is deeper in the middle (14-27 m), and
[[pc-alongchannel-wind-response]] found ~90% of a raw bottom ds/dy is just
that. So every density gradient here is taken on a COMMON z GRID (-14 to
-0.5 m, where all 12 faces are wet), and the single N-S contrast uses the
depth-matched pair p=2 / p=11 (h 19.8 / 20.0 m) from [[pc-lp-mouth-points]].

SIGNIFICANCE. Daily means with Bretherton n_eff, per
[[analyze-tides-on-anomalies]] -- hourly lag-1 autocorrelation of an already
lowpassed series is ~0.999 and makes n_eff meaningless.

CAVEAT, AND IT IS THE IMPORTANT ONE. tef2 stores only the SECTION-NORMAL
velocity. pc_lp is a N-S line of u faces, so this file has the east-west
component and nothing else. The cross-mouth (v) and vertical (w) velocities
that would close a lateral overturning CELL are not in it. Everything below is
therefore the lateral structure of the ALONG-channel exchange -- which is what
carries the transport -- not the secondary circulation itself. Seeing the cell
directly needs a new extraction carrying v and w.

Runs on the mac from the local extractions_avg.
run 20260818_pc_lateral_baroclinic.py
"""
import argparse
import warnings

import gsw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
p.add_argument('-gctag', default='wb1_pc1', type=str)
p.add_argument('-sect', default='pc_lp', type=str)
p.add_argument('-0', '--ds0', default='2024.01.01', type=str)
p.add_argument('-1', '--ds1', default='2025.12.31', type=str)
args = p.parse_args()

warnings.simplefilter('ignore')
Ldir = Lfun.Lstart(gridname=args.gctag.split('_')[0])
tef2 = Ldir['LOo'] / 'extract' / args.gtagex / 'tef2'
ex_fn = tef2 / ('extractions_avg_%s_%s' % (args.ds0, args.ds1)) / (args.sect + '.nc')
wind_fn = (Ldir['LOo'] / 'DM_outs' / '20260806_wind' /
           ('wind_hourly_atm00_%s_%s.p' % (args.ds0, args.ds1)))
out_dir = Ldir['LOo'] / 'DM_outs' / '20260818_pc_lateral_baroclinic'
Lfun.make_dir(out_dir)

CB = dict(blue='#0072B2', red='#CC0000', green='#009E73', orange='#D55E00',
          purple='#CC79A7', grey='#7f7f7f')
GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
FS = 13
LAT, LON = 48.235, -122.6534
DY = 200.5                      # face spacing, m (uniform, from grid.nc pn)
G, RHO0 = 9.81, 1023.0
F = gsw.f(LAT)


def godin(a):
    return zfun.lowpass(np.asarray(a, dtype=float), f='godin')


def neff_r(x, y):
    """Correlation with a Bretherton effective sample size."""
    r = np.corrcoef(x, y)[0, 1]
    ac = lambda a: np.corrcoef(a[:-1], a[1:])[0, 1]
    ne = max(len(x) * (1 - ac(x) * ac(y)) / (1 + ac(x) * ac(y)), 3.0)
    t = r * np.sqrt((ne - 2) / max(1 - r ** 2, 1e-12))
    return r, ne, 2 * stats.t.sf(abs(t), ne - 2)


def partial_r(x, y, ctrl):
    """Partial correlation of x and y with the columns of ctrl regressed out."""
    Z = np.column_stack([np.ones(len(x))] + list(ctrl))
    res = lambda a: a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]
    return neff_r(res(x), res(y))


def modes(u, A):
    """Split (t,z,p) velocity into net / lateral / vertical / residual modes."""
    At = A.sum((1, 2))
    Un = (u * A).sum((1, 2)) / At
    Ul = (u * A).sum(1) / A.sum(1) - Un[:, None]        # (t,p)
    Uv = (u * A).sum(2) / A.sum(2) - Un[:, None]        # (t,z)
    Ur = u - Un[:, None, None] - Ul[:, None, :] - Uv[:, :, None]
    var = dict(lat=(A.sum(1) * Ul ** 2).sum(1) / At,
               vert=(A.sum(2) * Uv ** 2).sum(1) / At,
               resid=(A * Ur ** 2).sum((1, 2)) / At)
    return Un, Ul, Uv, Ur, var


# ------------------------------------------------------------------ load
ds = xr.open_dataset(ex_fn)
DZ = ds.DZ.values
dd = ds.dd.values
h = ds.h.values
nt, nz, npf = DZ.shape
A = dd[None, None, :] * DZ                 # face-cell area, m2
u = -ds.q.values / A                       # + = INTO the cove
tt = pd.to_datetime(ds.time.values)

zw = np.zeros((nt, nz + 1, npf))
zw[:, 0, :] = -h[None, :]
zw[:, 1:, :] = -h[None, None, :] + np.cumsum(DZ, axis=1)
zr = 0.5 * (zw[:, :-1, :] + zw[:, 1:, :])
SA = gsw.SA_from_SP(ds.salt.values, -zr, LON, LAT)
rho = gsw.rho(SA, gsw.CT_from_pt(SA, ds.temp.values), 0.0)

# ------------------------------------------------- 1. how lateral is it
Am = A.mean(0)
um = (u * A).mean(0) / Am                                   # 2-year mean field
_, Ulm, Uvm, Urm, vmean = modes(um[None], Am[None])
Un, Ul, Uv, Ur, vinst = modes(u, A)
UlG = np.column_stack([godin(Ul[:, j]) for j in range(npf)])
UvG = np.column_stack([godin(Uv[:, k]) for k in range(nz)])
vsub = dict(lat=np.nanmean((Am.sum(0) * UlG ** 2).sum(1) / Am.sum()),
            vert=np.nanmean((Am.sum(1) * UvG ** 2).sum(1) / Am.sum()))

share = {}
s = sum(v[0] for v in vmean.values())
share['2-year mean'] = {k: 100 * v[0] / s for k, v in vmean.items()}
s = vsub['lat'] + vsub['vert']
share['subtidal'] = dict(lat=100 * vsub['lat'] / s, vert=100 * vsub['vert'] / s, resid=np.nan)
s = sum(v.mean() for v in vinst.values())
share['instantaneous'] = {k: 100 * v.mean() / s for k, v in vinst.items()}

# ---------------------------------- 2. depth-uniform vs depth-varying lateral
u_bc = u - ((u * A).sum(1) / A.sum(1))[:, None, :]          # kill each face's depth mean
u_bc = u_bc - ((u_bc * A).sum(2) / A.sum(2))[:, :, None]    # kill the width-mean profile
ubcm = (u_bc * A).mean(0) / Am

qp = (u * A).sum(1)
Qn = qp.sum(1)
Ap = A.sum(1)
QLAT = 0.5 * np.abs(qp - Qn[:, None] * Ap / Ap.sum(1, keepdims=True)).sum(1)
QLAT_BC = 0.25 * np.abs(u_bc * A).sum((1, 2))

# ------------------------------------------------- 3a. thermal wind on z grid
zt = np.arange(-14.0, -0.5 + 1e-9, 0.5)
U_z = np.empty((nt, len(zt), npf))
R_z = np.empty((nt, len(zt), npf))
for j in range(npf):
    for it in range(nt):
        U_z[it, :, j] = np.interp(zt, zr[it, :, j], u[it, :, j])
        R_z[it, :, j] = np.interp(zt, zr[it, :, j], rho[it, :, j])
U_z, R_z = U_z[:, :, ::-1], R_z[:, :, ::-1]                 # order faces south -> north
nd = nt // 24
Uz_d = U_z[:nd * 24].reshape(nd, 24, len(zt), npf).mean(1)
Rz_d = R_z[:nd * 24].reshape(nd, 24, len(zt), npf).mean(1)
Up = Uz_d - Uz_d.mean(2, keepdims=True)                     # lateral anomaly only
Rp = Rz_d - Rz_d.mean(2, keepdims=True)
sh_obs = np.gradient(Up, zt, axis=1)
sh_tw = -(G / (F * RHO0)) * np.gradient(Rp, DY, axis=2)
sl = slice(1, npf - 1)
o, pr = sh_obs[:, :, sl].ravel(), sh_tw[:, :, sl].ravel()
good = np.isfinite(o) & np.isfinite(pr)
r_tw = np.corrcoef(o[good], pr[good])[0, 1]
slope_tw = np.polyfit(pr[good], o[good], 1)[0]
amp_tw = np.sqrt((o[good] ** 2).mean()) / np.sqrt((pr[good] ** 2).mean())

Uscale = np.abs(Uz_d - Uz_d.mean(2, keepdims=True)).mean()
Wsec = npf * DY
drho_v = (Rz_d.max(1) - Rz_d.min(1)).mean()
c_i = np.sqrt(G * drho_v / RHO0 * 14.0)
Rd_i = c_i / F
Ro = Uscale / (F * Wsec)

# ------------------------------------------------------- 3b. drivers in time
rb = (rho * A).sum(2) / A.sum(2)
STRAT = rb[:, 0] - rb[:, -1]                                # bottom - top, kg/m3
dm = lambda j: (rho[:, :, j] * A[:, :, j]).sum(1) / A[:, :, j].sum(1)
DRHO_NS = dm(11) - dm(2)                                    # depth-matched south - north
W = pd.read_pickle(wind_fn)['W']
wa = W.u_pc.values * -0.966 + W.v_pc.values * -0.259        # + = toward the head
wc = W.u_pc.values * -0.259 + W.v_pc.values * 0.966         # + = toward the north shore
raw = dict(QLAT=QLAT, QLAT_BC=QLAT_BC, STRAT=STRAT, DRHO_NS=DRHO_NS,
           QPRISM=np.abs(Qn), wa=wa, wc=wc)
D = {k: v[:nd * 24].reshape(nd, 24).mean(1) for k, v in raw.items()}
td = tt[:nd * 24:24]

rows = []
for tgt in ['QLAT', 'QLAT_BC']:
    for prd in ['STRAT', 'DRHO_NS', 'QPRISM', 'wa', 'wc']:
        r, ne, pv = neff_r(D[tgt], D[prd])
        if prd in ('wa', 'wc'):
            rows.append((tgt, prd, r, pv, np.nan, np.nan, ne))
        else:
            pr_, pne, ppv = partial_r(D[tgt], D[prd], [D['wa'], D['wc']])
            rows.append((tgt, prd, r, pv, pr_, ppv, pne))
drv = pd.DataFrame(rows, columns=['target', 'predictor', 'r', 'p', 'r_partial_wind',
                                  'p_partial', 'n_eff'])

# ------------------------------------------------------------------ report
txt = []
add = txt.append
add('PENN COVE MOUTH (%s) -- LATERAL EXCHANGE, %s to %s' % (args.sect, args.ds0, args.ds1))
add('section width %.0f m, %d faces, depth %.1f-%.1f m' % (Wsec, npf, h.min(), h.max()))
add('')
add('1. SHARE OF THE EXCHANGE VARIANCE (%)')
add('   %-16s %8s %8s %8s' % ('', 'lateral', 'vertical', 'residual'))
for k, v in share.items():
    add('   %-16s %8.0f %8.0f %8.0f' % (k, v['lat'], v['vert'], v['resid']))
add('')
add('2. LATERAL EXCHANGE TRANSPORT (m3/s)')
add('   depth-uniform QLAT      %6.0f' % QLAT.mean())
add('   depth-varying QLAT_BC   %6.0f   (%.0f%% of the lateral exchange)'
    % (QLAT_BC.mean(), 100 * QLAT_BC.mean() / (QLAT.mean() + QLAT_BC.mean())))
add('')
add('3a. THERMAL WIND (lateral anomaly, interior faces, daily)')
add('    r = %+.3f   slope = %.2f   rms(obs)/rms(pred) = %.2f' % (r_tw, slope_tw, amp_tw))
add('    Ro = %.2f ; internal Rossby radius %.1f km ; W/Rd = %.2f' % (Ro, Rd_i / 1e3, Wsec / Rd_i))
add('')
add('3b. DRIVERS (daily, Bretherton n_eff)')
add(drv.to_string(index=False, float_format=lambda v: '%.3f' % v))
report = '\n'.join(txt)
print(report)
(out_dir / 'report.txt').write_text(report + '\n')
drv.to_csv(out_dir / 'drivers.csv', index=False)
pd.DataFrame(D, index=td).to_csv(out_dir / 'daily_series.csv')

# ------------------------------------------------------------------ figure
yc = (np.arange(npf)[::-1] - (npf - 1) / 2) * DY / 1e3      # km, + = north
zm = (zr * A).sum(2).mean(0) / A.sum(2).mean(0)             # mean depth of each level

fig, axs = plt.subplots(2, 3, figsize=(17, 9.5))
Y, Z = np.meshgrid(yc, zm)

ax = axs[0, 0]
lim = np.abs(100 * um).max()
pc = ax.pcolormesh(Y, Z, 100 * um, cmap='RdBu_r', vmin=-lim, vmax=lim, shading='gouraud')
ax.contour(Y, Z, um, [0], colors='k', linewidths=1.2)
plt.colorbar(pc, ax=ax, label='cm s$^{-1}$')
ax.set_title('a) 2-yr mean $u$ (+ = into cove)', fontsize=FS)
ax.set_xlabel('distance north of mouth centre (km)')
ax.set_ylabel('z (m)')

ax = axs[0, 1]
lim = np.abs(100 * ubcm).max()
pc = ax.pcolormesh(Y, Z, 100 * ubcm, cmap='PuOr_r', vmin=-lim, vmax=lim, shading='gouraud')
ax.contour(Y, Z, ubcm, [0], colors='k', linewidths=1.2)
plt.colorbar(pc, ax=ax, label='cm s$^{-1}$')
ax.set_title('b) depth-varying lateral mode $u_{bc}$', fontsize=FS)
ax.set_xlabel('distance north of mouth centre (km)')
ax.set_ylabel('z (m)')

ax = axs[0, 2]
ax.plot(yc, 100 * Ulm[0], 'o-', color=CB['blue'], label='depth-uniform lateral')
kt, kb = slice(24, nz), slice(0, 8)
ut = (um[kt] * Am[kt]).sum(0) / Am[kt].sum(0)
ub = (um[kb] * Am[kb]).sum(0) / Am[kb].sum(0)
ax.plot(yc, 100 * ut, 's--', color=CB['red'], label='near-surface (top ~2 m)')
ax.plot(yc, 100 * ub, '^--', color=CB['green'], label='near-bed (bottom ~8 m)')
ax.axhline(0, color='k', lw=0.8)
ax.grid(**GRID)
ax.legend(fontsize=10)
ax.set_title('c) lateral structure, surface vs bed', fontsize=FS)
ax.set_xlabel('distance north of mouth centre (km)')
ax.set_ylabel('u (cm s$^{-1}$)')

ax = axs[1, 0]
lab = list(share.keys())
xx = np.arange(len(lab))
ax.bar(xx - 0.2, [share[k]['lat'] for k in lab], 0.4, color=CB['blue'], label='lateral')
ax.bar(xx + 0.2, [share[k]['vert'] for k in lab], 0.4, color=CB['orange'], label='vertical')
ax.set_xticks(xx)
ax.set_xticklabels(lab, fontsize=10)
ax.set_ylabel('share of exchange variance (%)')
ax.grid(**GRID, axis='y')
ax.legend(fontsize=10)
ax.set_title('d) the residual is lateral, the tide is vertical', fontsize=FS)

ax = axs[1, 1]
sub = np.random.default_rng(0).choice(np.flatnonzero(good), 6000, replace=False)
ax.plot(1e3 * pr[sub], 1e3 * o[sub], '.', ms=2, color=CB['grey'], alpha=0.4)
ee = np.array([1e3 * pr[good].min(), 1e3 * pr[good].max()])
ax.plot(ee, ee, 'k--', lw=1, label='1:1 (thermal wind)')
ax.plot(ee, slope_tw * ee, color=CB['red'], lw=1.5, label='fit, slope %.2f' % slope_tw)
ax.set_xlabel(r'thermal-wind $\partial u/\partial z$ (10$^{-3}$ s$^{-1}$)')
ax.set_ylabel(r'observed $\partial u/\partial z$ (10$^{-3}$ s$^{-1}$)')
ax.grid(**GRID)
ax.legend(fontsize=10)
ax.set_title('e) not geostrophic (r = %+.2f, W/R$_d$ = %.2f)' % (r_tw, Wsec / Rd_i), fontsize=FS)

ax = axs[1, 2]
ax.plot(D['STRAT'], D['QLAT_BC'], '.', ms=3, color=CB['purple'], alpha=0.5)
b = np.polyfit(D['STRAT'], D['QLAT_BC'], 1)
xs = np.linspace(D['STRAT'].min(), D['STRAT'].max(), 20)
ax.plot(xs, np.polyval(b, xs), color=CB['red'], lw=2)
rs = drv.query('target=="QLAT_BC" and predictor=="STRAT"').iloc[0]
rq = drv.query('target=="QLAT_BC" and predictor=="QPRISM"').iloc[0]
ax.text(0.03, 0.96, 'r(stratification) = %+.2f\npartial | wind = %+.2f\nr(tidal strength) = %+.2f'
        % (rs.r, rs.r_partial_wind, rq.r), transform=ax.transAxes, va='top', fontsize=11)
ax.grid(**GRID)
ax.set_xlabel(r'stratification $\Delta\rho$ bottom-top (kg m$^{-3}$)')
ax.set_ylabel('QLAT$_{BC}$ (m$^3$ s$^{-1}$)')
ax.set_title('f) the lateral shear is density-controlled', fontsize=FS)

fig.suptitle('Penn Cove mouth (%s): baroclinic lateral exchange -- %s' % (args.sect, args.gtagex),
             fontsize=FS + 2)
fig.tight_layout()
fig.savefig(out_dir / 'pc_lateral_baroclinic.png', dpi=200, transparent=True)
plt.close(fig)
print('\nwrote %s' % out_dir)
