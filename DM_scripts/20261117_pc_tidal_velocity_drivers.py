"""
Penn Cove pc_lj (Lovejoy): which drivers go with the LARGEST flood / ebb velocities, at the surface and at the
bottom, and with their direction (with or against the tide)? Every 2024 flood and ebb at its peak hour,
greater / lesser of each lunar day (as 20261117_pc_every_flood_ebb.py). Unlike 20261117_pc_profile_drivers.py
(the SHAPE, u - Ubar), this is the raw velocity, so the size of the tide is part of the answer. Self-contained:
the tides and drivers are computed here with the same code as profile_drivers.

STEP 1  load pc_lj from the pc_cove box (u normal to the faces, + = into the cove; z_w, dy, h, top / bottom
        salt and temp at the faces); half-cycles from the section Q (runs of one sign, runs < MIN_RUN h absorbed,
        peak = max |Q| hour); lunar days = 4 consecutive half-cycles tiled from the baseline day's first flood
        (picks.csv); greater / lesser = the bigger / smaller flood (ebb) of each lunar day.
STEP 2  at every peak hour the cross-channel profile U_b = Q_b / A_b in 1 m bins below the moving surface (each
        cell's transport split among the bins it overlaps), bins wet (>= half a face) in EVERY tide of the row;
        Ubar = Q / A.
STEP 3  drivers at the peak hours (profile_drivers STEP 5 + STEP 9, the eddy left out):
            tide     |Q| at the peak; D2 and D1 envelopes of the pc_lj tidal transport (utide fits of
                     20261112_pc_utide, |hilbert|); |Q| of the half-cycle before and of the same-phase one before
            wind     applied cove-mean stress along / across the cove (Godin and the WIND_H h before the peak);
                     Saratoga Passage stress along / across the passage (20260806_wind cube, Godin and WIND_H h)
            density  drho = sigma0 bottom - top at pc_lj (Godin and at the peak); in-out sigma0 difference pc_lp - pc_cp
                     top and bottom (Godin); the pc_lj density profile at the peak and at the slack that starts the
                     half-cycle: half-density depth, depth of max dsigma/dz, shape PCs 1-NRHO
            other    surface heat flux (WIND_H h, Godin); Skagit flow lagged SKAGIT_LAG; subtidal sea level and
                     subtidal Q at pc_lj; season and hour of day (sin, cos)
STEP 4  responses per tide, velocities ALONG THE TIDE (x the tide's sign: + into the cove on a flood, + out on an
        ebb), so one signed number carries speed and direction: + = with the tide, - = against it:
            Ubar         section-mean velocity (the tide's strength; Q / A, so |Q| will explain it)
            u_surf       top bin (0-1 m)
            u_bot        deepest common bin
            u_max        largest with-tide velocity in the profile, and z_umax its depth [m]
            frac_against share of the profile area flowing against the tide [%]
            surf_excess  u_surf - Ubar: how much faster (+) or slower than the section mean the surface runs
            bot_excess   u_bot - Ubar, the same at the bottom
STEP 5  one driver at a time, per row x response x driver:
            r        Pearson correlation over the tides of the row
            sig      p < 0.05 with the effective n = n (1 - r1x r1y) / (1 + r1x r1y), r1 = lag-1 autocorrelation of
                     the tide sequence (Bretherton et al. 1999): neighbouring tides share the same subtidal state
            dhi_lo   composite: response mean on the driver's top-third tides minus its bottom-third tides, in the
                     response's own units
        Each driver alone, so correlated drivers (|Q| / D2, the density set, season / Skagit) look alike.
STEP 6  all drivers together, per row x response: leave-one-month-out CV R^2 of OLS and of gradient-boosted trees
        (sklearn HistGradientBoostingRegressor, max_depth GBT_DEPTH, GBT_ITER iterations, learning rate GBT_LR,
        >= GBT_LEAF tides per leaf), and the permutation importance of the trees by driver group (season and hour
        sin + cos, the density-profile PCs of each time as one group): the drop in CV R^2 when the group is
        shuffled within the held-out month, NPERM repeats.
STEP 7  figures and tables:
            correlation.png  r heat map, drivers x (row, response), significant cells marked
            composites.png   top-third minus bottom-third, coloured by the difference / response SD, labelled in units
            importance.png   permutation importance, driver groups x (row, response), CV R^2 in the column labels
            results.csv      STEP 5 (long format); importance.csv  STEP 6

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc, pc_cove_mix_[ds0]_[ds1].nc
       LO_output/DM_outs/20261112_pc_utide/fits.p
       LO_output/DM_outs/20261113_pc_candidate_days/picks.csv
       LO_output/DM_outs/20260806_river_hydrographs/daily_flow.csv
       LO_output/DM_outs/20260806_wind/wind_hourly_atm00_[ds0]_[ds1].p
OUTPUT LO_output/DM_outs/20261117_pc_tidal_velocity_drivers/

run 20261117_pc_tidal_velocity_drivers.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gsw
import numpy as np
import pandas as pd
import pickle
import utide
import xarray as xr
from scipy import stats
from scipy.linalg import svd
from scipy.signal import hilbert
from sklearn.ensemble import HistGradientBoostingRegressor

from lo_tools import Lfun, zfun

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'     # the box file
YEAR = 2024
PAD = pd.Timedelta(days=2)                # past the year end, for the Godin filter and the last lunar days
SIGN = -1                                 # section-positive (east) is out of the cove
XI_U, ETA = 23, np.arange(14, 6, -1)      # pc_lj u faces in the box, north to south (global i 60, j 223..216)
MIN_RUN = 3                               # half-cycles shorter than this [h] are slack jitter
EDGES = np.arange(0, 31, 1.0)             # profile bins, depth below the surface [m]
BAND = {'D2': ['M2', 'S2', 'N2', 'K2'],     # semidiurnal band
        'D1': ['K1', 'O1', 'P1', 'Q1']}     # diurnal band, S1 left out (mostly the daily wind)
EPAD = pd.Timedelta(days=30)              # reconstruction padding either side of the year, for hilbert
AXIS = (-0.9789, -0.2043)                 # cove axis, mouth -> head (pc_forcing_states.py)
WIND_H = 6                                # hours before the peak averaged for the short-term wind / heat flux
SKAGIT_LAG = pd.Timedelta(days=3)         # Skagit flow lag
SAR_BOX = (-122.62, -122.42, 48.02, 48.22)   # Saratoga Passage wind cells: lon0, lon1, lat0, lat1
SAR_AXIS = (-0.55, 0.83)                  # Saratoga Passage axis, + up-passage (toward the NW)
NRHO = 3                                  # density-profile EOFs used as drivers
GBT_DEPTH, GBT_ITER, GBT_LR, GBT_LEAF = 3, 200, 0.05, 20   # boosted trees (STEP 6)
NPERM = 5                                 # permutation repeats (STEP 6)
ROWS = [('flood', 'greater'), ('flood', 'lesser'), ('ebb', 'greater'), ('ebb', 'lesser')]
RESP = [('Ubar', 'section mean', 'cm/s'), ('u_surf', 'surface', 'cm/s'), ('u_bot', 'bottom', 'cm/s'),
        ('u_max', 'max with-tide', 'cm/s'), ('z_umax', 'depth of max', 'm'), ('frac_against', 'area against tide', '%'),
        ('surf_excess', 'surface - mean', 'cm/s'), ('bot_excess', 'bottom - mean', 'cm/s')]

gridname, tag, ex_name = GTAGEX.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
box_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_%s_%s.nc' % (DS0, DS1))
mix_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_mix_%s_%s.nc' % (DS0, DS1))
fit_fn = Ldir['LOo'] / 'DM_outs' / '20261112_pc_utide' / 'fits.p'
pick_fn = Ldir['LOo'] / 'DM_outs' / '20261113_pc_candidate_days' / 'picks.csv'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261117_pc_tidal_velocity_drivers'
Lfun.make_dir(out_dir)

# ------------------------------------------------------------------ STEP 1: load pc_lj, peak hours (as 20261117_pc_profile_drivers.py STEP 1)
t0 = pd.Timestamp('%d-01-01' % YEAR)
t1 = pd.Timestamp('%d-01-01' % (YEAR + 1))
ds = xr.open_dataset(box_fn).sel(ocean_time=slice(t0, t1 + PAD))
ot = pd.DatetimeIndex(ds.ocean_time.values)
u = SIGN * ds['u'].isel(xi_u=XI_U, eta_u=ETA).values.astype(float)                   # (t, k, p) + into the cove
zw = ds['z_w'].isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).values.mean(axis=-1)       # (t, w, p) at the face
dy = (1 / ds['pn'].isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).values).mean(axis=-1)  # (p,)
h_f = ds['h'].isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).values.mean(axis=-1)      # (p,) depth at the face
ts = {vn: ds[vn].isel(s_rho=[0, -1], eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).values.mean(axis=-1)
      for vn in ['salt', 'temp']}                                                    # (t, bot/top, p)
lon_f, lat_f = ds['lon_u'].isel(eta_u=ETA, xi_u=XI_U).values, ds['lat_u'].isel(eta_u=ETA, xi_u=XI_U).values
ds.close()
dz = np.diff(zw, axis=1)                                                             # (t, k, p)
Q = (u * dz * dy).sum(axis=(1, 2))                                                   # (t,) + = flood
A = (dz * dy).sum(axis=(1, 2))                                                       # (t,) section area
print('%s  %d hours  %s to %s' % (box_fn.name, len(ot), ot[0], ot[-1]))


def runs(x):
    """[(start, stop), ...] of constant x (stop exclusive)"""
    k = np.flatnonzero(np.diff(x) != 0) + 1
    return list(zip(np.r_[0, k], np.r_[k, len(x)]))


sg = np.sign(Q)
for a, b in runs(sg):                                    # short runs -> 0, then take the previous sign
    if b - a < MIN_RUN:
        sg[a:b] = 0
sg = pd.Series(sg).replace(0, np.nan).ffill().bfill().to_numpy()
HC = runs(sg)[1:-1]                                      # the first and last runs are partial
hc_sign = np.array([sg[a] for a, b in HC])
hc_pk = np.array([a + np.argmax(np.abs(Q[a:b])) for a, b in HC])    # peak hour of each half-cycle
picks = pd.read_csv(pick_fn, parse_dates=['t0', 't1']).set_index('case')
hc_t0 = ot[[a for a, b in HC]]
ia = np.flatnonzero((hc_t0 == picks.loc['baseline', 't0']) & (hc_sign == 1))
assert len(ia) == 1, 'baseline t0 is not a flood start of this Q'
hc_day = (np.arange(len(HC)) - ia[0]) // 4               # lunar day: F E F E from the baseline's first flood
PK = {}                                                  # {(phase, kind): peak hours in YEAR}
for phase, sgn in [('flood', 1), ('ebb', -1)]:
    m = np.flatnonzero(hc_sign == sgn)
    gr = np.zeros(len(m), dtype=bool)
    for dd in np.unique(hc_day[m]):                      # the larger |Q| of each lunar day is the greater
        i = np.flatnonzero(hc_day[m] == dd)
        gr[i[np.argmax(np.abs(Q[hc_pk[m[i]]]))]] = True
    for kind, sel in [('greater', gr), ('lesser', ~gr)]:
        k = hc_pk[m[sel]]
        PK[(phase, kind)] = k[(ot[k] >= t0) & (ot[k] < t1) & (k >= 1)]       # k - 1 must exist (STEP 4)


def binned_QA(u, zw, dy):
    """Q_b and A_b (t, b) in the EDGES bins of depth below the moving surface, top bin first: each cell's
    transport split among the bins it overlaps (sum_b Q_b = Q exactly). As 20261113_pc_baseline_profiles.py."""
    dz = np.diff(zw, axis=1)
    q = u * dz * dy
    dw = zw[:, -1:, :] - zw                                          # depth below the surface of each interface
    d_top, d_bot = dw[:, 1:, :], dw[:, :-1, :]
    assert d_bot.max() < EDGES[-1]
    nb = len(EDGES) - 1
    Qb, Ab = np.zeros((len(u), nb)), np.zeros((len(u), nb))
    for b in range(nb):
        ov = np.clip(np.minimum(d_bot, EDGES[b + 1]) - np.maximum(d_top, EDGES[b]), 0, None)
        Qb[:, b] = (q * ov / dz).sum(axis=(1, 2))
        Ab[:, b] = (ov * dy).sum(axis=(1, 2))
    assert np.allclose(Qb.sum(axis=1), q.sum(axis=(1, 2)))
    return Qb, Ab


# ------------------------------------------------------------------ STEP 2: binned profile at every peak hour
WET = 0.5 * dy.mean()                                                # a bin counts as wet with at least half a face
zcb = 0.5 * (EDGES[1:] + EDGES[:-1])                                 # bin centres, depth below the surface
D = {}                                                               # per row: hours, Ubar, profile, bin depths, weights
for row in ROWS:
    k = PK[row]
    Qb, Ab = binned_QA(u[k], zw[k], dy)                              # (n, b)
    wet = (Ab >= WET).all(axis=0)                                    # bins wet in every tide of the row
    D[row] = {'hrs': k, 'Ubar': Q[k] / A[k], 'profile': Qb[:, wet] / Ab[:, wet], 'z': zcb[wet], 'w': Ab[:, wet].mean(axis=0)}
    print('%-7s %-5s %3d tides; common bins 0-%.0f m (%d)' % (row[1], row[0], len(k), EDGES[1:][wet].max(), wet.sum()))

# ------------------------------------------------------------------ STEP 3: drivers (as 20261117_pc_profile_drivers.py STEP 5)
DRV = {}                                                 # {name: (t,) hourly series}
DRV['|Q| peak'] = np.abs(Q)

# tide: D2 and D1 envelopes of the pc_lj tidal transport (as 20261116 STEP 5)
F = pickle.load(open(fit_fn, 'rb'))
UV_OUT = ['Lsmaj', 'Lsmin', 'theta', 'g']
jt, it_ = 10, XI_U                                       # template coef: refit one cell (box rho j 10, i 23) as 20261112_pc_utide.py
kt = np.flatnonzero((F['jj'] == jt) & (F['ii'] == it_))[0]
dsf = xr.open_dataset(box_fn)
TF = dsf.ocean_time.values
ut = np.nan_to_num(dsf['ubar'].isel(eta_u=jt, xi_u=[it_ - 1, it_]).values, nan=0.0).mean(axis=1)
vt = np.nan_to_num(dsf['vbar'].isel(eta_v=[jt - 1, jt], xi_v=it_).values, nan=0.0).mean(axis=1)
dsf.close()
coef = utide.solve(TF, ut, vt, lat=F['lat'][kt], constit=F['cons'], **F['kw'])
io = [F['cons'].index(n) for n in coef.name]             # fits.p order -> coef order
for vn in UV_OUT:
    assert np.allclose(coef[vn], F['bar'][vn][kt, io]), vn
tr = pd.date_range(t0 - EPAD, t1 + EPAD, freq='h')
for band in ['D2', 'D1']:
    ub = np.zeros((len(tr), len(ETA)))                   # (t, p) east u of the band at the faces
    for p, j in enumerate(ETA):
        for i in [XI_U, XI_U + 1]:
            k = np.flatnonzero((F['jj'] == j) & (F['ii'] == i))[0]
            c = coef.copy()
            for vn in UV_OUT:
                c[vn] = F['bar'][vn][k, io]
            c['umean'], c['vmean'] = 0.0, 0.0            # reconstruct adds the fit's mean flow back: tide only
            ub[:, p] += 0.5 * utide.reconstruct(tr, c, constit=BAND[band], verbose=False).u
    env = pd.Series(np.abs(hilbert(SIGN * (ub * h_f * dy).sum(axis=1))), index=tr).reindex(ot)
    assert env.notna().all()
    DRV['%s env' % band] = env.to_numpy()

# wind: applied cove-mean stress, Godin and the mean of the WIND_H hours up to the peak
dm = xr.open_dataset(mix_fn).sel(ocean_time=slice(t0, t1 + PAD))
assert (pd.DatetimeIndex(dm.ocean_time.values) == ot).all()
mask = dm['mask_rho'].values == 1
mu = mask[:, :-1] & mask[:, 1:]
mu[:, 30:] = False                                       # cove u faces: west of the mouth face (box xi_u 30)
mv = mask[:-1, :] & mask[1:, :]
mv[:, 31:] = False                                       # cove v faces: rho columns <= 30
tx = np.nanmean(dm['sustr'].values[:, mu], axis=1)
ty = np.nanmean(dm['svstr'].values[:, mv], axis=1)
dm.close()
t_al, t_cr = tx * AXIS[0] + ty * AXIS[1], tx * AXIS[1] - ty * AXIS[0]   # + toward the head / the north shore
DRV['tau_al Godin'] = zfun.lowpass(t_al, f='godin')
DRV['tau_al %dh' % WIND_H] = pd.Series(t_al).rolling(WIND_H).mean().to_numpy()   # mean of the WIND_H h ending at the hour
DRV['tau_cr Godin'] = zfun.lowpass(t_cr, f='godin')
DRV['tau_cr %dh' % WIND_H] = pd.Series(t_cr).rolling(WIND_H).mean().to_numpy()

# density: drho = sigma0 bottom - top at pc_lj, width-weighted, Godin and at the hour
SA = gsw.SA_from_SP(ts['salt'], 0, lon_f, lat_f)
sig = gsw.sigma0(SA, gsw.CT_from_pt(SA, ts['temp']))      # (t, bot/top, p)
drho_h = ((sig[:, 0] - sig[:, 1]) * dy).sum(axis=1) / dy.sum()
DRV['drho Godin'] = zfun.lowpass(drho_h, f='godin')
DRV['drho peak'] = drho_h

# season: day of year as a circle
doy = 2 * np.pi * (ot.dayofyear.to_numpy() - 1) / 365.25
DRV['season sin'], DRV['season cos'] = np.sin(doy), np.cos(doy)

# hour of day: the solar day as a circle (UTC)
hod = 2 * np.pi * ot.hour.to_numpy() / 24
DRV['hour sin'], DRV['hour cos'] = np.sin(hod), np.cos(hod)
DNAME = list(DRV)
GROUPS = {n: [n] for n in DNAME if not n.startswith(('season', 'hour'))}   # dominance groups; sin + cos together
GROUPS['season'] = ['season sin', 'season cos']
GROUPS['hour'] = ['hour sin', 'hour cos']
# ------------------------------------------------------------------ STEP 3b: more drivers (as 20261117_pc_profile_drivers.py STEP 9)
NEW = {}
dsb = xr.open_dataset(box_fn).sel(ocean_time=slice(t0, t1 + PAD))
mask = dsb['mask_rho'].values == 1
sec = {}
for sn, X in [('pc_cp', 15), ('pc_lp', 30)]:             # wet faces: rho cells both sides wet
    jj = np.flatnonzero(mask[:, X] & mask[:, X + 1])
    sec[sn] = (jj, X)
sig_ts = {}
for sn, (jj, X) in sec.items():
    sl = dict(s_rho=[0, -1], eta_rho=jj, xi_rho=[X, X + 1])
    sa, te = dsb['salt'].isel(**sl).values, dsb['temp'].isel(**sl).values          # (t, bot/top, face, 2)
    lo, la = dsb['lon_rho'].isel(eta_rho=jj, xi_rho=[X, X + 1]).values, dsb['lat_rho'].isel(eta_rho=jj, xi_rho=[X, X + 1]).values
    SA_ = gsw.SA_from_SP(sa, 0, lo, la)
    sg_ = gsw.sigma0(SA_, gsw.CT_from_pt(SA_, te)).mean(axis=-1)                   # (t, bot/top, face)
    w_ = (1 / dsb['pn'].isel(eta_rho=jj, xi_rho=[X, X + 1]).values).mean(axis=-1)  # face width
    sig_ts[sn] = (sg_ * w_).sum(axis=-1) / w_.sum()                                  # (t, bot/top) width-mean
cove = mask.copy()
cove[:, 31:] = False                                     # cove water: rho columns <= 30
zeta_c = dsb['zeta'].values[:, cove].mean(axis=1)
dsb.close()
NEW['drhox top Godin'] = zfun.lowpass(sig_ts['pc_lp'][:, 1] - sig_ts['pc_cp'][:, 1], f='godin')
NEW['drhox bot Godin'] = zfun.lowpass(sig_ts['pc_lp'][:, 0] - sig_ts['pc_cp'][:, 0], f='godin')

dm = xr.open_dataset(mix_fn).sel(ocean_time=slice(t0, t1 + PAD))
shf = dm['shflux'].values[:, cove].mean(axis=1)
dm.close()
NEW['shflux %dh' % WIND_H] = pd.Series(shf).rolling(WIND_H).mean().to_numpy()
NEW['shflux Godin'] = zfun.lowpass(shf, f='godin')

prev1, prev2 = np.full(len(ot), np.nan), np.full(len(ot), np.nan)   # tidal history, defined at the peak hours
for i in range(2, len(hc_pk)):
    prev1[hc_pk[i]], prev2[hc_pk[i]] = np.abs(Q[hc_pk[i - 1]]), np.abs(Q[hc_pk[i - 2]])
NEW['prev half |Q|'], NEW['prev same |Q|'] = prev1, prev2
NEW['zeta Godin'] = zfun.lowpass(zeta_c, f='godin')
NEW['Q Godin'] = zfun.lowpass(Q, f='godin')

# Skagit flow, lagged
qf = pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20260806_river_hydrographs' / 'daily_flow.csv', parse_dates=['date']).set_index('date')['skagit']
NEW['Skagit %dd' % SKAGIT_LAG.days] = qf.reindex(ot.floor('D') - SKAGIT_LAG).to_numpy()

# wind stress over Saratoga Passage, from the hourly atm00 wind cube
WC_ = pickle.load(open(Ldir['LOo'] / 'DM_outs' / '20260806_wind' / ('wind_hourly_atm00_%s_%s.p' % (DS0, DS1)), 'rb'))['cube']
sel = WC_['water'] & (WC_['lon'] > SAR_BOX[0]) & (WC_['lon'] < SAR_BOX[1]) & (WC_['lat'] > SAR_BOX[2]) & (WC_['lat'] < SAR_BOX[3])
uw, vw = WC_['u'][:, sel], WC_['v'][:, sel]                         # (t, cells) [m/s]
sp = np.hypot(uw, vw)
cdl = np.where(sp < 10.0, 1.14e-3, (0.49 + 0.065 * sp) * 1e-3)     # Large & Pond (1981), as 20260806_wind_reduce.py
txs, tys = (1.22 * cdl * sp * uw).mean(axis=1), (1.22 * cdl * sp * vw).mean(axis=1)
tsa = pd.Series(txs * SAR_AXIS[0] + tys * SAR_AXIS[1], index=pd.DatetimeIndex(WC_['time'])).reindex(ot).to_numpy()
tsc = pd.Series(txs * SAR_AXIS[1] - tys * SAR_AXIS[0], index=pd.DatetimeIndex(WC_['time'])).reindex(ot).to_numpy()
print('Saratoga wind: %d cube cells; along-passage stress %.3f to %.3f Pa' % (sel.sum(), np.nanmin(tsa), np.nanmax(tsa)))
NEW['tau_sar al %dh' % WIND_H] = pd.Series(tsa).rolling(WIND_H).mean().to_numpy()
NEW['tau_sar al Godin'] = zfun.lowpass(tsa, f='godin')
NEW['tau_sar cr %dh' % WIND_H] = pd.Series(tsc).rolling(WIND_H).mean().to_numpy()
NEW['tau_sar cr Godin'] = zfun.lowpass(tsc, f='godin')


# density profile at pc_lj, at the peak hour and at the slack that starts the half-cycle
dsb = xr.open_dataset(box_fn).sel(ocean_time=slice(t0, t1 + PAD))
sl_ = dict(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1])
sa_, te_ = dsb['salt'].isel(**sl_).values, dsb['temp'].isel(**sl_).values      # (t, k, p, 2)
dsb.close()
SA_ = gsw.SA_from_SP(sa_, 0, lon_f.mean(), lat_f.mean())
sig_lj = gsw.sigma0(SA_, gsw.CT_from_pt(SA_, te_)).mean(axis=-1)                # (t, k, p) at the faces
del sa_, te_, SA_
start_of = dict(zip(hc_pk, [a for a, b in HC]))                                 # peak hour -> first hour of its half-cycle
pk_all = np.unique(np.concatenate([D[r]['hrs'] for r in ROWS]))
sl_all = np.array([start_of[k] for k in pk_all])


def binned_rho(hh):
    """cross-channel (area-weighted) sigma0 (n, b) in the EDGES bins below the moving surface; NaN where the
    bin is wetted by less than WET"""
    dw = zw[hh][:, -1:, :] - zw[hh]
    d_top, d_bot = dw[:, 1:, :], dw[:, :-1, :]
    out = np.full((len(hh), len(EDGES) - 1), np.nan)
    for b in range(len(EDGES) - 1):
        ov = np.clip(np.minimum(d_bot, EDGES[b + 1]) - np.maximum(d_top, EDGES[b]), 0, None) * dy
        a = ov.sum(axis=(1, 2))
        out[:, b] = np.where(a >= WET, (sig_lj[hh] * ov).sum(axis=(1, 2)) / np.maximum(a, 1e-12), np.nan)
    return out


rp, rs = binned_rho(pk_all), binned_rho(sl_all)
cb = np.isfinite(rp).all(axis=0) & np.isfinite(rs).all(axis=0)                  # bins wet at every peak and slack
rp, rs = rp[:, cb], rs[:, cb]
zb = zcb[cb]


def pycno(p):
    """z_half and z_maxN2 [m] of density profiles p (n, b)"""
    mid = 0.5 * (p[:, :1] + p[:, -1:])
    zh = np.array([np.interp(m, pp, zb) if np.all(np.diff(pp) >= 0) else zb[np.argmax(pp >= m)] for pp, m in zip(p, mid[:, 0])])
    zn = 0.5 * (zb[1:] + zb[:-1])[np.argmax(np.diff(p, axis=1), axis=1)]
    return zh, zn


rpa = rp - rp.mean(axis=1, keepdims=True)                                       # density shape
mu_r = rpa.mean(axis=0)
Ur, Sr, Vr = svd((rpa - mu_r).T, full_matrices=False)
ve_r = Sr ** 2 / (Sr ** 2).sum()
pcr_pk = Vr[:NRHO] * np.sqrt(len(pk_all))                                         # (NRHO, n) unit variance
pcr_sl = (Ur[:, :NRHO].T @ ((rs - rs.mean(axis=1, keepdims=True)) - mu_r).T) / Sr[:NRHO, None] * np.sqrt(len(pk_all))
print('pc_lj density profile: %d common bins (0-%.0f m); shape EOFs %s%% of the variance; first-to-last bin sigma0 at the peak: '
      'median %.2f kg/m3' % (cb.sum(), EDGES[1:][cb].max(), ' '.join('%.0f' % (100 * v) for v in ve_r[:NRHO]),
                             np.median(rp[:, -1] - rp[:, 0])))
for tag_, p_, pc_ in [('peak', rp, pcr_pk), ('slack', rs, pcr_sl)]:
    zh, zn = pycno(p_)
    for nm, v in [('pycno z_half %s' % tag_, zh), ('pycno z_maxN2 %s' % tag_, zn)] + \
                 [('rhoprof %s %d' % (tag_, i + 1), pc_[i]) for i in range(NRHO)]:
        NEW[nm] = np.full(len(ot), np.nan)
        NEW[nm][pk_all] = v                                                    # defined at the peak hours
    print('  %-5s z_half median %.1f m (5-95%% %.1f-%.1f), z_maxN2 median %.1f m' % (tag_, np.median(zh), *np.percentile(zh, [5, 95]), np.median(zn)))
NNAME = list(NEW)
for n in NNAME:
    v = NEW[n][np.concatenate([D[r]['hrs'] for r in ROWS])]
    print('%-16s at the peak hours: %8.3f to %8.3f (median %.3f)' % (n, np.nanmin(v), np.nanmax(v), np.nanmedian(v)))

# ------------------------------------------------------------------ STEP 4: responses along the tide
DNAMES = DNAME + NNAME
RS, XD = {}, {}
for row in ROWS:
    d = D[row]
    k = d['hrs']
    sgn = 1 if row[0] == 'flood' else -1                     # + = with the tide
    P, w, z = 100 * sgn * d['profile'], d['w'], d['z']       # (n, b) [cm/s]
    U = 100 * sgn * d['Ubar']
    im = np.argmax(P, axis=1)
    RS[row] = pd.DataFrame({'Ubar': U, 'u_surf': P[:, 0], 'u_bot': P[:, -1], 'u_max': P[np.arange(len(P)), im],
                            'z_umax': z[im], 'frac_against': 100 * ((P < 0) * w).sum(axis=1) / w.sum(),
                            'surf_excess': P[:, 0] - U, 'bot_excess': P[:, -1] - U}, index=ot[k])
    XD[row] = pd.DataFrame({n: (DRV[n] if n in DRV else NEW[n])[k] for n in DNAMES}, index=ot[k])
    q = RS[row].quantile([0.05, 0.5, 0.95])
    print('%-7s %-5s ' % (row[1], row[0]) + '; '.join('%s %.1f [%.1f, %.1f]' % (c, q.loc[0.5, c], q.loc[0.05, c], q.loc[0.95, c])
                                                     for c in RS[row].columns))
    print('              tides with the surface against the tide %.0f%%, the bottom against %.0f%%'
          % (100 * (RS[row]['u_surf'] < 0).mean(), 100 * (RS[row]['u_bot'] < 0).mean()))

# ------------------------------------------------------------------ STEP 5: one driver at a time
def r1(x):
    """lag-1 autocorrelation of a sequence"""
    return np.corrcoef(x[:-1], x[1:])[0, 1]


RES = []
for row in ROWS:
    X, Y = XD[row], RS[row]
    for rc, _, _ in RESP:
        for dn in DNAMES:
            ok = np.isfinite(X[dn].values) & np.isfinite(Y[rc].values)
            x, y = X[dn].values[ok], Y[rc].values[ok]
            if x.std() == 0:
                continue
            r = np.corrcoef(x, y)[0, 1]
            a, b = r1(x), r1(y)
            neff = len(x) * (1 - a * b) / (1 + a * b)
            tt = r * np.sqrt(max(neff - 2, 1) / max(1 - r ** 2, 1e-12))
            p = 2 * stats.t.sf(abs(tt), max(neff - 2, 1))
            lo, hi = np.percentile(x, [100 / 3, 200 / 3])
            RES.append({'row': '%s %s' % (row[1], row[0]), 'response': rc, 'driver': dn, 'r': r, 'n': len(x), 'n_eff': neff,
                        'p': p, 'sig': p < 0.05, 'dhi_lo': y[x >= hi].mean() - y[x <= lo].mean(), 'resp_sd': y.std()})
RES = pd.DataFrame(RES)
RES.to_csv(out_dir / 'results.csv', index=False, float_format='%.4f')
print('saved %s' % (out_dir / 'results.csv'))
for row in ROWS:
    rr = '%s %s' % (row[1], row[0])
    for rc, _, unit in RESP:
        sub = RES[(RES['row'] == rr) & (RES['response'] == rc)]
        sub = sub.loc[sub['r'].abs().sort_values(ascending=False).index[:4]]
        print('%-14s %-12s %s' % (rr, rc, ', '.join('%s %+.2f%s (%+.1f %s)' % (s['driver'], s['r'], '*' if s['sig'] else '', s['dhi_lo'], unit)
                                                  for _, s in sub.iterrows())))

# ------------------------------------------------------------------ STEP 6: all drivers together
def group_of(n):
    if n.startswith('rhoprof'):
        return n.rsplit(' ', 1)[0]
    if n.startswith(('season', 'hour')):
        return n.split(' ')[0]
    return n


GRP = {}
for i, n in enumerate(DNAMES):
    GRP.setdefault(group_of(n), []).append(i)


def gbt():
    return HistGradientBoostingRegressor(max_depth=GBT_DEPTH, max_iter=GBT_ITER, learning_rate=GBT_LR,
                                         min_samples_leaf=GBT_LEAF, random_state=0)


class OLS:
    def fit(self, X, y):
        self.b = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0]
        return self

    def predict(self, X):
        return np.column_stack([np.ones(len(X)), X]) @ self.b


def cv_pred(make, X, y, month, perm=False):
    """leave-one-month-out predictions; perm: also with each driver group shuffled in the held-out month"""
    rng = np.random.default_rng(0)
    yh = np.zeros_like(y)
    yp = {g: np.zeros((NPERM, len(y))) for g in GRP} if perm else {}
    for mo in np.unique(month):
        te = month == mo
        m = make().fit(X[~te], y[~te])
        yh[te] = m.predict(X[te])
        for g in yp:
            for k in range(NPERM):
                Xp = X[te].copy()
                Xp[:, GRP[g]] = Xp[rng.permutation(te.sum())][:, GRP[g]]
                yp[g][k, te] = m.predict(Xp)
    return yh, yp


R2 = lambda y, yh: 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()
IMP = []
for row in ROWS:
    X = XD[row].values
    ok = np.isfinite(X).all(axis=1)
    Xs = (X[ok] - X[ok].mean(axis=0)) / X[ok].std(axis=0)
    mo = XD[row].index[ok].month.to_numpy()
    for rc, _, _ in RESP:
        y = RS[row][rc].values[ok]
        r = {'row': '%s %s' % (row[1], row[0]), 'response': rc, 'n': ok.sum(), 'lin': R2(y, cv_pred(OLS, Xs, y, mo)[0])}
        yh, yp = cv_pred(gbt, Xs, y, mo, perm=True)
        r['gbt'] = R2(y, yh)
        for g in GRP:
            r[g] = r['gbt'] - np.mean([R2(y, yp[g][k]) for k in range(NPERM)])
        IMP.append(r)
        top = sorted(GRP, key=lambda g: -r[g])[:4]
        print('%-14s %-12s CV R2 lin %.2f gbt %.2f | %s' % (r['row'], rc, r['lin'], r['gbt'], ', '.join('%s %.2f' % (g, r[g]) for g in top)))
IMP = pd.DataFrame(IMP)
IMP.to_csv(out_dir / 'importance.csv', index=False, float_format='%.4f')
print('saved %s' % (out_dir / 'importance.csv'))

# ------------------------------------------------------------------ STEP 7: figures
plt.rcParams.update({'font.size': 10})
COLS = [('%s %s' % (r[1], r[0]), rc) for r in ROWS for rc, _, _ in RESP]
RLAB = dict((a, b) for a, b, _ in RESP)


def heat(M_, ann, clab, title, out_fn, cmap, vmin, vmax, ylab, cbar):
    fig, ax = plt.subplots(figsize=(22, 0.32 * M_.shape[0] + 3.5), layout='constrained')
    cs = ax.pcolormesh(M_, cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(M_.shape[0]):
        for j in range(M_.shape[1]):
            if ann[i][j]:
                ax.text(j + 0.5, i + 0.5, ann[i][j], ha='center', va='center', fontsize=6)
    for j in range(0, M_.shape[1], len(RESP)):
        ax.axvline(j, color='k', lw=1.5)
    ax.set_xticks(np.arange(M_.shape[1]) + 0.5, clab, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(M_.shape[0]) + 0.5, ylab, fontsize=8)
    ax.invert_yaxis()
    fig.colorbar(cs, ax=ax, label=cbar, shrink=0.6)
    ax.set_title(title, fontsize=11)
    fig.savefig(out_fn, dpi=200, transparent=True)
    plt.close(fig)
    print('saved %s' % out_fn)


Mr = np.full((len(DNAMES), len(COLS)), np.nan)
Mc = np.full_like(Mr, np.nan)
Ar, Ac = [[''] * len(COLS) for _ in DNAMES], [[''] * len(COLS) for _ in DNAMES]
for j, (rr, rc) in enumerate(COLS):
    sub = RES[(RES['row'] == rr) & (RES['response'] == rc)].set_index('driver')
    for i, dn in enumerate(DNAMES):
        if dn in sub.index:
            s = sub.loc[dn]
            Mr[i, j], Mc[i, j] = s['r'], s['dhi_lo'] / s['resp_sd']
            Ar[i][j] = ('%+.2f' % s['r']) + ('*' if s['sig'] else '')
            Ac[i][j] = '%+.1f' % s['dhi_lo'] if s['sig'] else ''
clab = ['%s\n%s' % (rr, RLAB[rc]) for rr, rc in COLS]
gtx = '%s %d' % (GTAGEX, YEAR)
heat(Mr, Ar, clab, 'pc_lj (Lovejoy), %s: Pearson r of each driver alone with each response, along the tide (+ = with the tide); '
     '* p < 0.05 with the effective n' % gtx, out_dir / 'correlation.png', 'RdBu_r', -1, 1, DNAMES, 'r')
heat(Mc, Ac, clab, 'pc_lj (Lovejoy), %s: response on the driver\'s top-third tides minus bottom-third, / response SD; '
     'labels in the response units, significant drivers only' % gtx, out_dir / 'composites.png', 'RdBu_r', -1.5, 1.5, DNAMES,
     '(top third - bottom third) / SD')
GN = list(GRP)
Mi = np.array([[IMP[(IMP['row'] == rr) & (IMP['response'] == rc)].iloc[0][g] for (rr, rc) in COLS] for g in GN])
Ai = [['%.2f' % v if v >= 0.02 else '' for v in Mi[i]] for i in range(len(GN))]
clab_i = ['%s\nCV R$^2$ %.2f' % (lab, IMP[(IMP['row'] == rr) & (IMP['response'] == rc)].iloc[0]['gbt']) for lab, (rr, rc) in zip(clab, COLS)]
heat(np.clip(Mi, 0, None), Ai, clab_i, 'pc_lj (Lovejoy), %s: all drivers together, boosted trees, leave-one-month-out; '
     'permutation importance (drop in CV R$^2$) by driver group' % gtx, out_dir / 'importance.png', 'viridis', 0,
     max(0.3, Mi.max()), GN, 'permutation importance')
