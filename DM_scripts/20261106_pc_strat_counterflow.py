"""
Why more stratification means more water moving against the net flow through the Penn Cove
sections (pc_cp, pc_lj, pc_lp), wb1_t0_xn11abbur00, 2024-2025, hourly ocean_his snapshots.

The net transport Q(t) is set by sea level (dV/dt), so stratification cannot change it; what it
changes is how much water moves IN and OUT at once. Compared at FIXED |Q| (hours binned by |Q|
quintile per section) across the record-wide thirds of Godin-filtered drho (pc_forcing_states.py):

PER HOUR, per section (u + into the cove, dA = dy dz on the face, as 20261103_pc_section_profiles.py)
    counterflow  Qc = transport in the cells flowing AGAINST that hour's net direction (>= 0)
    lateral      0.5 sum_p |(Ucol_p - Ubar) A_p|       column means about the section mean
    layered      0.5 sum_kp |(u - Ucol_p) dA|           each cell about its own column mean
                 (the names of [[exchange-terms-layered-lateral]]; each is that part's exchange transport)
    AKv          eddy viscosity (pc_cove_mix box, GLS, s_w at rho points, mean of the two rho cells
                 either side of the face) in the MIDDLE THIRD of each column's depth, width-weighted;
                 AKV_BAK = 5e-6 m2/s is the model's floor (BLANK.in)
    sigma        section-mean sigma0 (TEOS-10, area-weighted); along-cove difference
                 dsig_x = sigma(pc_lp) - sigma(pc_cp), + = denser at the mouth, then Godin-filtered
REGRESSION  standardized multiple regression of Qc, layered and lateral on |Q|, Godin log10 AKv,
    dsig_x and drho (all z-scored); drho and dsig_x are collinear (r ~0.86), so their split is weak.

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc, pc_cove_mix_[ds0]_[ds1].nc
OUTPUT LO_output/DM_outs/20261106_pc_strat_counterflow/
    series.npz                  hourly Q, Qc, layered, lateral, AKv, sigma per section (cache)
    strat_counterflow.png       counterflow, layered, lateral, AKv vs |Q| for the three strat thirds
    strat_forcing.png           along-cove vs vertical density difference; regression coefficients

run 20261106_pc_strat_counterflow.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import gsw

from lo_tools import Lfun, zfun
from pc_forcing_states import series, nearest

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SIGN = -1                                 # section-positive (east) is out of the cove
FACES = {                                 # u faces in the pc_cove box, north to south; west to east
    'pc_cp': dict(XI_U=15, ETA=np.arange(13, 5, -1)),
    'pc_lj': dict(XI_U=23, ETA=np.arange(14, 6, -1)),
    'pc_lp': dict(XI_U=30, ETA=np.arange(19, 7, -1)),
}
SECTS = list(FACES)
NAME = {'pc_lp': 'pc_lp (mouth)', 'pc_lj': 'pc_lj (Lovejoy)', 'pc_cp': 'pc_cp (Coupeville)'}
LON, LAT = -122.69, 48.23
AKV_BAK = 5.0e-6                          # m2/s, BLANK.in background viscosity
BLK = 744                                 # hours per read block (the box is chunked one hour x whole box)
VARS = ['Q', 'Qc', 'lay', 'lat', 'akv', 'sig']

Ldir = Lfun.Lstart(gridname='wb1')
box = Ldir['LOo'] / 'extract' / GTAGEX / 'box'
in_fn, mix_fn = box / ('pc_cove_%s_%s.nc' % (DS0, DS1)), box / ('pc_cove_mix_%s_%s.nc' % (DS0, DS1))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261106_pc_strat_counterflow'
out_dir.mkdir(parents=True, exist_ok=True)
cache_fn = out_dir / 'series.npz'

# ------------------------------------------------------------------ hourly section series (cached)
if cache_fn.exists():
    z = np.load(cache_fn)
    T = pd.DatetimeIndex(z['T'])
    out = {k: {v: z['%s_%s' % (k, v)] for v in VARS} for k in SECTS}
    print('loaded ' + str(cache_fn))
else:
    ds, mx = xr.open_dataset(in_fn), xr.open_dataset(mix_fn)
    T = pd.DatetimeIndex(ds.ocean_time.values)
    NT = len(T)
    assert (mx.ocean_time.values == ds.ocean_time.values).all()
    dy = {k: (1 / ds.pn.values[F['ETA']][:, [F['XI_U'], F['XI_U'] + 1]]).mean(-1) for k, F in FACES.items()}
    out = {k: {v: np.zeros(NT) for v in VARS} for k in SECTS}
    for t0 in range(0, NT, BLK):
        tt = slice(t0, min(t0 + BLK, NT))
        u_b = SIGN * ds.u.isel(ocean_time=tt).to_numpy().astype(float)
        zw_b = ds.z_w.isel(ocean_time=tt).to_numpy()
        S_b = ds.salt.isel(ocean_time=tt).to_numpy().astype(float)
        th_b = ds.temp.isel(ocean_time=tt).to_numpy().astype(float)
        ak_b = mx.AKv.isel(ocean_time=tt).to_numpy().astype(float)
        for k, F in FACES.items():
            X, E = F['XI_U'], F['ETA']
            face = lambda a: a[:, :, E][:, :, :, [X, X + 1]].mean(-1)          # rho field -> the face (mean of the two cells)
            o = out[k]
            u, zw = u_b[:, :, E, X], face(zw_b)
            dA = np.diff(zw, axis=1) * dy[k][None, None, :]
            q = u * dA
            Qn = q.sum((1, 2))
            s = np.sign(Qn)[:, None, None]
            A = dA.sum((1, 2))
            Ub = Qn / A
            Ac = dA.sum(1)
            Uc = q.sum(1) / Ac                                              # (t, p) column means
            o['Q'][tt] = Qn
            o['Qc'][tt] = -np.where(q * s < 0, q * s, 0).sum((1, 2))
            o['lat'][tt] = 0.5 * np.abs((Uc - Ub[:, None]) * Ac).sum(1)
            o['lay'][tt] = 0.5 * np.abs((u - Uc[:, None, :]) * dA).sum((1, 2))
            frac = (zw[:, -1:, :] - zw) / (zw[:, -1:, :] - zw[:, :1, :])     # 0 at the surface .. 1 at the bed
            w = ((frac > 1 / 3) & (frac < 2 / 3)) * dy[k][None, None, :]   # middle third of each column, width-weighted
            o['akv'][tt] = (face(ak_b) * w).sum((1, 2)) / w.sum((1, 2))
            SA = gsw.SA_from_SP(face(S_b), 0, LON, LAT)
            o['sig'][tt] = (gsw.sigma0(SA, gsw.CT_from_pt(SA, face(th_b))) * dA).sum((1, 2)) / A
        print('  hours %d-%d of %d' % (tt.start, tt.stop, NT), flush=True)
    ds.close()
    mx.close()
    np.savez(cache_fn, T=T.values, **{'%s_%s' % (k, v): out[k][v] for k in SECTS for v in VARS})
    print('saved ' + str(cache_fn))

# ------------------------------------------------------------------ classes: record-wide strat thirds, |Q| quintiles per section
X = series()
dr = X['drho_lp'][nearest(X['t_s'], T)]
dsx = zfun.lowpass(out['pc_lp']['sig'] - out['pc_cp']['sig'], f='godin')   # along-cove density difference, + denser at the mouth
ok = np.isfinite(dr) & np.isfinite(dsx)
q3 = np.nanpercentile(dr, [100 / 3, 200 / 3])
STRAT = {'low': dr <= q3[0], 'mid': (dr > q3[0]) & (dr <= q3[1]), 'high': dr > q3[1]}
SCOL = {'low': '#0072B2', 'mid': '0.5', 'high': '#D55E00'}
print('strat thirds of Godin drho: <= %.2f, %.2f-%.2f, > %.2f kg/m3' % (q3[0], *q3, q3[1]))
print('along-cove dsig_x median by strat third: ' + ', '.join('%s %.3f' % (c, np.median(dsx[ok & m])) for c, m in STRAT.items())
      + ' kg/m3; corr(dsig_x, drho) %.2f' % np.corrcoef(dsx[ok], dr[ok])[0, 1])

MED, QMID, COEF = {}, {}, {}
for k in SECTS:
    o = out[k]
    aQ = np.abs(o['Q'])
    edges = np.nanpercentile(aQ[ok], np.linspace(0, 100, 6))
    QMID[k] = 0.5 * (edges[1:] + edges[:-1])
    inq = [ok & (aQ >= edges[i]) & (aQ <= edges[i + 1]) for i in range(5)]
    MED[k] = {v: {c: np.array([np.median(o[v][m & STRAT[c]]) for m in inq]) for c in STRAT} for v in ['Qc', 'lay', 'lat', 'akv']}
    print('\n%s  medians averaged over the |Q| quintiles: low / mid / high strat' % k)
    for v, f, un in [('Qc', 1, 'counterflow [m3/s]'), ('lay', 1, 'layered exchange [m3/s]'),
                     ('lat', 1, 'lateral exchange [m3/s]'), ('akv', 1e6, 'mid-depth AKv [1e-6 m2/s]')]:
        mm = [MED[k][v][c].mean() * f for c in STRAT]
        print('   %-26s %8.1f %8.1f %8.1f   high/low x%.2f' % (un, *mm, mm[2] / mm[0]))
    # standardized multiple regression on |Q|, Godin log10 AKv, dsig_x, drho
    lak = zfun.lowpass(np.log10(np.maximum(o['akv'], 1e-8)), f='godin')
    m = ok & np.isfinite(lak)
    zs = lambda x: (x[m] - x[m].mean()) / x[m].std()
    Xm = np.c_[np.ones(m.sum()), zs(aQ), zs(lak), zs(dsx), zs(dr)]
    COEF[k] = {}
    for v in ['Qc', 'lay', 'lat']:
        y = zs(o[v])
        b = np.linalg.lstsq(Xm, y, rcond=None)[0]
        COEF[k][v] = (b[1:], 1 - np.sum((y - Xm @ b) ** 2) / np.sum(y ** 2))
        print('   %-4s standardized coefs: |Q| %+.2f, log AKv %+.2f, dsig_x %+.2f, drho %+.2f  (R2 %.2f)' % (v, *COEF[k][v][0], COEF[k][v][1]))

# ------------------------------------------------------------------ figure 1: the four measures vs |Q| by strat third
ROWS = [('Qc', 1, 'counterflow Q$_c$ [m$^3$ s$^{-1}$]\n(against the net direction)'),
        ('lay', 1, 'layered exchange [m$^3$ s$^{-1}$]\n0.5 $\\Sigma$|(u - U$_{col}$) dA|'),
        ('lat', 1, 'lateral exchange [m$^3$ s$^{-1}$]\n0.5 $\\Sigma$|(U$_{col}$ - U$_{bar}$) A$_{col}$|'),
        ('akv', 1e6, 'mid-depth AKv [10$^{-6}$ m$^2$ s$^{-1}$]\n(middle third of each column)')]
fig, axs = plt.subplots(4, 3, figsize=(14, 14), layout='constrained', sharex='col', sharey='row')
for r, (v, f, lab) in enumerate(ROWS):
    for c, k in enumerate(SECTS):
        ax = axs[r, c]
        for cl in STRAT:
            ax.plot(QMID[k], MED[k][v][cl] * f, 'o-', color=SCOL[cl], lw=2,
                    label='%s strat (drho %s)' % (cl, {'low': '<= %.1f' % q3[0], 'mid': '%.1f-%.1f' % tuple(q3), 'high': '> %.1f' % q3[1]}[cl]))
        if v == 'akv':
            ax.axhline(AKV_BAK * 1e6, color='k', lw=1, ls='--', label='model floor AKV_BAK')
            ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)
        if r == 0:
            ax.set_title(NAME[k], fontsize=11)
        if r == 3:
            ax.set_xlabel('|net Q| [m$^3$ s$^{-1}$] (quintile midpoints)')
        if c == 0:
            ax.set_ylabel(lab)
axs[0, 0].legend(fontsize=8, loc='upper right')
axs[3, 0].legend(fontsize=8, loc='lower right')
fig.suptitle('At the same net flow, more stratification = more water moving against it; the increase is LAYERED, and mid-depth\n'
             'viscosity sits at the model floor in every class. Medians, hourly snapshots, %s, %s to %s' % (GTAGEX, DS0, DS1), fontsize=11)
fig.savefig(out_dir / 'strat_counterflow.png', dpi=200, transparent=True)
plt.close(fig)

# ------------------------------------------------------------------ figure 2: forcing collinearity and regression coefficients
fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), layout='constrained', gridspec_kw={'width_ratios': [1, 1.6]})
ax = axs[0]
hb = ax.hexbin(dr[ok], dsx[ok], gridsize=50, mincnt=1, bins='log', cmap='Greys')
fig.colorbar(hb, ax=ax).set_label('hours')
for y in q3:
    ax.axvline(y, color='k', lw=0.8, ls='--')
ax.set_xlabel(r'vertical $\Delta\sigma_0$ bottom - top, Godin [kg m$^{-3}$]')
ax.set_ylabel(r'along-cove $\Delta\sigma_0$, pc_lp - pc_cp section means, Godin [kg m$^{-3}$]')
ax.set_title('vertical and along-cove density differences rise together (r %.2f)' % np.corrcoef(dsx[ok], dr[ok])[0, 1], fontsize=10)
ax.grid(True, alpha=0.3)
ax = axs[1]
pred = ['|Q|', 'log AKv', r'along-cove $\Delta\sigma$', r'vertical $\Delta\sigma$']
w = 0.08
for c, k in enumerate(SECTS):
    for j, (v, hatch) in enumerate([('Qc', ''), ('lay', '//'), ('lat', '..')]):
        b, r2 = COEF[k][v]
        x = np.arange(4) + (c * 3 + j - 4) * w
        ax.bar(x, b, w, color=['#009E73', '#CC79A7', '#E69F00'][c], hatch=hatch, edgecolor='k', lw=0.4,
               label='%s %s (R$^2$ %.2f)' % (k, {'Qc': 'counterflow', 'lay': 'layered', 'lat': 'lateral'}[v], r2))
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(np.arange(4))
ax.set_xticklabels(pred)
ax.set_ylabel('standardized coefficient')
ax.set_title('standardized multiple regression (hourly, z-scored)\nvertical and along-cove $\\Delta\\sigma$ are collinear, so their split is weak', fontsize=10)
ax.legend(fontsize=7, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.1))
ax.grid(True, axis='y', alpha=0.3)
fig.savefig(out_dir / 'strat_forcing.png', dpi=200, transparent=True)
plt.close(fig)
print('saved ' + str(out_dir / 'strat_counterflow.png') + ', strat_forcing.png')
