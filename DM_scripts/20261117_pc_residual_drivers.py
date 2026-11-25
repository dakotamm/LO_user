"""
Penn Cove pc_lj (Lovejoy): what sets the SHAPE of the RESIDUAL (tidally averaged) velocity profile?
The residual counterpart of 20261117_pc_profile_drivers.py (peak flood / ebb). Every 2024 averaging unit,
its residual split into strength + shape, the shapes reduced to a few EOFs, and each EOF's amplitude
regressed on all candidate drivers together; plus every residual drawn and coloured by class as
20261117_pc_every_flood_ebb.py.

-unit day    (default) explicit lunar day = 4 consecutive half-cycles F E F E, tiled from the baseline's
             first flood (picks.csv 2024-08-31 05:00), as the greater / lesser rule of
             20261117_pc_every_flood_ebb.py. Averages out D2, D1 and (24.84 h vs 24 h) nearly all of S1.
-unit cycle  one flood + the following ebb (~12.4 h, two per lunar day). Keeps the within-day variability:
             in a first check the two cycles of a lunar day differ by 71% of the per-cycle shape variance,
             only 1% of it systematic greater / lesser (diurnal inequality); the rest is short-term
             (wind, the solar day), so cycle mode adds the hour of day as a driver.

STEP 1  load pc_lj from the pc_cove box, half-cycles from the sign of the section Q (as
        20261117_pc_profile_drivers.py STEP 1), the units, and per unit the residual
            profile  U_b = sum_t Q_b / sum_t A_b, 1 m bins below the moving surface (as
                     20261113_pc_baseline_profiles.py: transport-weighted, includes the Stokes part)
            faces    U_p = sum_t sum_k u dz / sum_t sum_k dz, the depth average at each face
            Ubar     = sum_t Q / sum_t A, the residual section mean (net volume change; small)
        Bins: only those with a mean area of at least half a face in EVERY unit.
STEP 2  every residual drawn: profile | faces, thin lines + the mean, then coloured by class for the sets of
        20261117_pc_every_flood_ebb.py STEP 6 (springneap, tropeq, wind, strat, season; daynight in cycle
        mode), a unit taking the class of its middle hour; one figure per set and per class.
STEP 3  strength / shape split: shape = U - Ubar (profile: layered; faces: lateral).
STEP 4  EOFs of the shapes, one SVD per field, as 20261117_pc_profile_drivers.py STEP 3 (UW Lab 8-1, sqrt
        area weights, PC > 0 = upper / north water more INTO the cove than Ubar), modes to VAR_KEEP.
STEP 5  persistence. The +/- 1 h noise ceiling of the peaks does not exist for an average; instead
            persist  r(PC, PC of the next unit): how much of a mode carries over one unit (NOT a ceiling: a
                     driver that changes from unit to unit, like the unit-mean wind, can explain more)
            day      also printed: split-half r, the day's two F+E cycles projected on the day EOFs. It is
                     NEGATIVE for the profile modes (-0.4, -0.6): consecutive cycle residuals alternate (D1 /
                     S1 leaking into a 12.4 h average), so the F+E cycle is not a clean residual and a
                     Spearman-Brown ceiling from it is meaningless.
STEP 6  drivers per unit, z-scored, all continuous:
            tide     D2 and D1 envelopes of the pc_lj tidal transport (20261112_pc_utide fits), unit mean
            wind     tau_al, tau_cr, applied cove-mean stress (pc_cove_mix box): mean over the unit AND over
                     the LDAY h before it (the gradients answer the wind in 1-2 d)
            density  drho = sigma0 bottom - top at pc_lj, width-weighted, unit mean
            season   sin and cos of the day of year of the middle hour
            hour     (cycle only) sin and cos of the UTC hour of the middle hour
STEP 7  per field and kept PC (and Ubar): OLS on all drivers together, in-sample and leave-one-month-out CV
        R^2, general dominance analysis (sin + cos as one group), season added last.
STEP 8  RDA cross-check: the shape modes most explained by the drivers, against EOF 1.
STEP 9  figures and results.csv
STEP 10 candidate drivers added as STEP 9 of 20261117_pc_profile_drivers.py, as unit means here:
            drhox top / bot  in-out density difference, sigma0 (gsw) at pc_lp minus pc_cp, width-mean of the
                             top / bottom s-level of the wet faces' rho cells (+ = denser outside)
            shflux / prior   surface net heat flux (+ into the ocean), cove-mean (rho cols <= 30), over the
                             unit and over the LDAY h before it
            |Q| / prev |Q|   mean |Q| through pc_lj over the unit (its actual tide, wind and river incl.) and
                             over the previous unit (tidal history)
            zeta             cove-mean sea level (remote setup)
            Q net            net transport through pc_lj over the unit (the cove's volume change)
STEP 11 per field and PC, leave-one-month-out CV R^2 of four models, as STEP 10 of 20261117_pc_profile_drivers.py:
            lin old / lin all   OLS on the STEP 6 drivers / STEP 6 + STEP 10
            gbt old / gbt all   gradient-boosted trees (sklearn HistGradientBoostingRegressor, shallow), same sets
        Permutation importance of 'gbt all' (drop in CV R^2 with one driver group shuffled in the held-out month).
STEP 12 figure nonlinear.png: 2 x 2 (field x EOF1 | EOF2) permutation importance (STEP 10 drivers hatched),
        the four CV R^2 and the persistence; table nonlinear.csv (all PCs).

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc
       LO_output/extract/[gtagex]/box/pc_cove_mix_[ds0]_[ds1].nc
       LO_output/DM_outs/20261112_pc_utide/fits.p
       LO_output/DM_outs/20261113_pc_candidate_days/picks.csv
OUTPUT LO_output/DM_outs/20261117_pc_residual_drivers/[unit]/
       every.png, every_[set].png, every_[set]_[class].png    STEP 2
       eofs.png, pcs.png, dominance.png, rda.png, results.csv   STEP 9
       nonlinear.png, nonlinear.csv                             STEPS 11-12

run 20261117_pc_residual_drivers.py -unit day
"""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gsw
import numpy as np
import pandas as pd
import pickle
import utide
import xarray as xr
from scipy.linalg import svd
from scipy.signal import hilbert
from sklearn.ensemble import HistGradientBoostingRegressor

from lo_tools import Lfun, zfun

parser = argparse.ArgumentParser()
parser.add_argument('-unit', type=str, default='day', choices=['day', 'cycle'])
UNIT = parser.parse_args().unit

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'     # the box file
YEAR = 2024
PAD = pd.Timedelta(days=2)                # past the year end, so the last unit is whole
SIGN = -1                                 # section-positive (east) is out of the cove
XI_U, ETA = 23, np.arange(14, 6, -1)      # pc_lj u faces in the box, north to south (global i 60, j 223..216)
MIN_RUN = 3                               # half-cycles shorter than this [h] are slack jitter
EDGES = np.arange(0, 31, 1.0)             # profile bins, depth below the surface [m]
NHC = {'day': 4, 'cycle': 2}[UNIT]        # half-cycles per unit
BAND = {'D2': ['M2', 'S2', 'N2', 'K2'],     # semidiurnal band
        'D1': ['K1', 'O1', 'P1', 'Q1']}     # diurnal band, S1 left out (mostly the daily wind)
EPAD = pd.Timedelta(days=30)              # reconstruction padding either side of the year, for hilbert
AXIS = (-0.9789, -0.2043)                 # cove axis, mouth -> head (pc_forcing_states.py)
TTH = 0.01                                # wind stress threshold for the wind classes [Pa]
LDAY = 25                                 # hours of the 'prior' wind
NM = 6                                    # most EOF modes kept
VAR_KEEP = 0.9                            # keep modes until this share of the shape variance
GBT_DEPTH, GBT_ITER, GBT_LR, GBT_LEAF = 3, 200, 0.05, 20   # boosted trees (STEP 11), as 20261117_pc_profile_drivers.py
NPERM = 5                                 # permutation repeats (STEP 11)
FIELDS = ['profile', 'faces']
CCOL = {'spring': '#CC3311', 'neap': '#33BBEE', 'tropic': '#EE7733', 'equatorial': '#009988',
        'up-cove': '#EE3377', 'down-cove': '#0077BB', 'calm': 'k', 'high strat': '#AA3377', 'low strat': '#66CCEE',
        'DJF': '#4477AA', 'MAM': '#228833', 'JJA': '#EE6677', 'SON': '#CCBB44', 'day': '#DDAA33', 'night': '#004488',
        'middle third': '0.75', 'other': '0.75'}

gridname, tag, ex_name = GTAGEX.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
box_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_%s_%s.nc' % (DS0, DS1))
mix_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_mix_%s_%s.nc' % (DS0, DS1))
fit_fn = Ldir['LOo'] / 'DM_outs' / '20261112_pc_utide' / 'fits.p'
pick_fn = Ldir['LOo'] / 'DM_outs' / '20261113_pc_candidate_days' / 'picks.csv'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261117_pc_residual_drivers' / UNIT
Lfun.make_dir(out_dir)

# ------------------------------------------------------------------ STEP 1: load pc_lj, half-cycles, units, residuals
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
print('%s  %d hours  %s to %s;  -unit %s' % (box_fn.name, len(ot), ot[0], ot[-1], UNIT))


def runs(x):
    """[(start, stop), ...] of constant x (stop exclusive)"""
    k = np.flatnonzero(np.diff(x) != 0) + 1
    return list(zip(np.r_[0, k], np.r_[k, len(x)]))


sg = np.sign(Q)
for a, b in runs(sg):                                    # short runs -> 0, then take the previous sign
    if b - a < MIN_RUN:
        sg[a:b] = 0
sg = pd.Series(sg).replace(0, np.nan).ffill().bfill().to_numpy()
HC = np.array(runs(sg)[1:-1])                            # (half-cycle, start/stop); the first and last are partial
hc_sign = sg[HC[:, 0]]
picks = pd.read_csv(pick_fn, parse_dates=['t0', 't1']).set_index('case')
ia = np.flatnonzero((ot[HC[:, 0]] == picks.loc['baseline', 't0']) & (hc_sign == 1))
assert len(ia) == 1, 'baseline t0 is not a flood start of this Q'
rel = np.arange(len(HC)) - ia[0]                         # 0 = the baseline's first flood; even = flood


def unit_spans(n):
    """[(start hour, stop hour), ...] of the units of n half-cycles (F E ... from the baseline tiling) starting in YEAR"""
    out = []
    for g in np.unique(rel // n):
        i = np.flatnonzero(rel // n == g)
        if len(i) == n and t0 <= ot[HC[i[0], 0]] < t1:
            assert hc_sign[i[0]] == 1
            out.append((HC[i[0], 0], HC[i[-1], 1]))
    return np.array(out)


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


Qb_h, Ab_h = binned_QA(u, zw, dy)                                    # (t, b) every hour
UDZ_h, H_h = (u * dz).sum(axis=1), dz.sum(axis=1)                    # (t, p)


def residuals(spans):
    """per unit: profile (n, b) = sum Q_b / sum A_b, mean bin area (n, b), faces (n, p) = sum u dz / sum dz,
    Ubar (n,) = sum Q / sum A"""
    R = {k: [] for k in ['profile', 'Abar', 'faces', 'Ubar']}
    for a, b in spans:
        sQb, sAb = Qb_h[a:b].sum(axis=0), Ab_h[a:b].sum(axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            R['profile'].append(sQb / sAb)
        R['Abar'].append(sAb / (b - a))
        R['faces'].append(UDZ_h[a:b].sum(axis=0) / H_h[a:b].sum(axis=0))
        R['Ubar'].append(Q[a:b].sum() / A[a:b].sum())
    return {k: np.array(v) for k, v in R.items()}


SP = unit_spans(NHC)                                                 # the analysed units
SP_cyc = unit_spans(2)                                               # F+E cycles (STEP 5 split half)
R = residuals(SP)
RC = residuals(SP_cyc)
WET = 0.5 * dy.mean()                                                # a bin counts with a mean area of half a face
wet = (R['Abar'] >= WET).all(axis=0) & (RC['Abar'] >= WET).all(axis=0)
zcb = 0.5 * (EDGES[1:] + EDGES[:-1])                                 # bin centres, depth below the surface
xe = np.r_[0, np.cumsum(dy)] / 1000                                  # face edges [km] from the north end
xcf = 0.5 * (xe[1:] + xe[:-1])                                       # face centres
mid = ((SP[:, 0] + SP[:, 1]) // 2)                                   # middle hour of each unit
nh = SP[:, 1] - SP[:, 0]
print('%d %s units in %d, %d to %d h (median %.1f); profile bins kept 0-%.0f m (%d)'
      % (len(SP), UNIT, YEAR, nh.min(), nh.max(), np.median(nh), EDGES[1:][wet].max(), wet.sum()))
print('residual Ubar: median %+.2f cm/s, 5-95%% %+.2f to %+.2f cm/s'
      % tuple(100 * np.r_[np.median(R['Ubar']), np.percentile(R['Ubar'], [5, 95])]))

# ------------------------------------------------------------------ STEP 2: hourly forcing, classes, every residual drawn
# tide: D2 and D1 envelopes of the pc_lj tidal transport (as 20261117_pc_every_flood_ebb.py STEP 6)
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
ENV = {}
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
    ENV[band] = env.to_numpy()

# wind: applied cove-mean stress on the cove axis, hourly
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

# density: drho = sigma0 bottom - top at pc_lj, width-weighted, hourly
SA = gsw.SA_from_SP(ts['salt'], 0, lon_f, lat_f)
sig = gsw.sigma0(SA, gsw.CT_from_pt(SA, ts['temp']))      # (t, bot/top, p)
drho_h = ((sig[:, 0] - sig[:, 1]) * dy).sum(axis=1) / dy.sum()


def sun_elevation(t, lat, lon):
    """solar elevation [deg] at the UTC times t (DatetimeIndex), NOAA general solar position approximation"""
    hr = t.hour.to_numpy() + t.minute.to_numpy() / 60
    g = 2 * np.pi / 365 * (t.dayofyear.to_numpy() - 1 + (hr - 12) / 24)          # fractional year
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
           - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    ha = np.radians((hr * 60 + eqt + 4 * lon) / 4 - 180)                         # hour angle (true solar time)
    la = np.radians(lat)
    return 90 - np.degrees(np.arccos(np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)))


# classes of every hour, as 20261117_pc_every_flood_ebb.py STEP 6; each unit takes the class of its middle hour
yr = (ot >= t0) & (ot < t1)
CLS = {}                                                 # {set: (class of each unit (n,), [classes], subtitle)}
for sn, band, (c_hi, c_lo) in [('springneap', 'D2', ('spring', 'neap')), ('tropeq', 'D1', ('tropic', 'equatorial'))]:
    env = pd.Series(ENV[band], index=ot)
    cl = pd.Series('middle third', index=ot)
    for mo, e in env[yr].groupby(ot[yr].month):          # terciles within each month
        lo, hi = e.quantile([1 / 3, 2 / 3])
        cl[e.index[e >= hi]] = c_hi
        cl[e.index[e <= lo]] = c_lo
    CLS[sn] = (cl.to_numpy()[mid], [c_hi, c_lo, 'middle third'],
               '%s / %s = top / bottom third of hours by the %s envelope of the pc_lj tidal transport, within each month' % (c_hi, c_lo, band))
ta_g, tc_g = zfun.lowpass(t_al, f='godin'), zfun.lowpass(t_cr, f='godin')
WC = np.full(len(ot), 'other', dtype=object)
WC[ta_g > TTH] = 'up-cove'
WC[ta_g < -TTH] = 'down-cove'
WC[(np.abs(ta_g) <= TTH) & (np.abs(tc_g) <= TTH)] = 'calm'
CLS['wind'] = (WC[mid], ['up-cove', 'down-cove', 'calm', 'other'],
               'Godin cove-mean applied stress: up-cove / down-cove = along > +%.2f / < -%.2f Pa, calm = |along| and |cross| <= %.2f Pa, '
               'other = cross-cove only' % (TTH, TTH, TTH))
drho_g = zfun.lowpass(drho_h, f='godin')
lo, hi = np.nanpercentile(drho_g[yr], [100 / 3, 200 / 3])
SC = np.full(len(ot), 'middle third', dtype=object)
SC[drho_g >= hi] = 'high strat'
SC[drho_g <= lo] = 'low strat'
CLS['strat'] = (SC[mid], ['high strat', 'low strat', 'middle third'],
                'Godin drho = sigma0 bottom - top at pc_lj, width-weighted: high / low = top / bottom third of %d hours (> %.2f / < %.2f kg m$^{-3}$)'
                % (YEAR, hi, lo))
SEAS = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM', 6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON'}
CLS['season'] = (np.array([SEAS[m] for m in ot[mid].month], dtype=object), ['DJF', 'MAM', 'JJA', 'SON'],
                 'season of the middle hour: DJF = Jan, Feb, Dec %d, MAM, JJA, SON' % YEAR)
if UNIT == 'cycle':                                      # a lunar day spans the whole solar day
    elev = sun_elevation(ot[mid], lat_f.mean(), lon_f.mean())
    CLS['daynight'] = (np.where(elev > 0, 'day', 'night').astype(object), ['day', 'night'],
                       'day = sun above the horizon at pc_lj at the middle hour of the cycle (NOAA solar position), night = below')
for sn, (cl, order, _) in CLS.items():
    print('  %-10s %s' % (sn, ', '.join('%s %d' % (x, (cl == x).sum()) for x in order)))

# figures: profile | faces
plt.rcParams.update({'font.size': 12})
PROF = np.where(wet, R['profile'], np.nan)                           # bins not kept are not drawn
XL = [np.ceil(2 * 1.05 * 100 * np.nanmax(np.abs(x))) / 2 for x in [PROF, R['faces']]]
zbot = zcb[wet].max() + 1
UNAME = {'day': 'lunar day (F E F E)', 'cycle': 'flood + ebb cycle'}[UNIT]


def dress(axs, legend=True):
    for ax, xl in zip(axs, XL):
        ax.axvline(0, color='0.4', lw=0.8)
        ax.set_xlim(xl, -xl)                                         # + (into the cove) drawn LEFT
        ax.grid(color='lightgray', linestyle='--', alpha=0.5)
        ax.set_xlabel('residual u [cm s$^{-1}$]   ← IN | OUT →')
    axs[0].set_ylim(zbot, 0)
    axs[0].set_ylabel('depth below the surface [m]')
    axs[1].set_ylim(xe[-1], xe[0])                                   # north at the top
    axs[1].set_ylabel('distance across from the north end [km]\nS ← | → N')
    axs[0].set_title('Cross-channel averaged profile', fontweight='bold')
    axs[1].set_title('Depth average at each face', fontweight='bold')
    if legend:
        axs[1].legend(*axs[0].get_legend_handles_labels(), loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, frameon=False)


def title(fig, sub):
    fig.suptitle('pc_lj (Lovejoy), residual of every %s of %d (sum Q$_b$ / sum A$_b$), %s, hourly ocean_his\n%s'
                 % (UNAME, YEAR, GTAGEX, sub), fontsize=13)


def save(fig, fn):
    fig.savefig(out_dir / fn, dpi=200, transparent=True)
    plt.close(fig)
    print('saved %s' % (out_dir / fn))


fig, axs = plt.subplots(1, 2, figsize=(17, 7), layout='constrained')
axs[0].plot(100 * PROF.T, zcb, '-', color='0.5', lw=0.5, alpha=0.15)
axs[1].plot(100 * R['faces'].T, xcf, '-', color='0.5', lw=0.5, alpha=0.15)
axs[0].plot([], [], '-', color='0.5', lw=1.5, label='every %s (n = %d)' % (UNAME, len(SP)))
axs[0].plot(100 * np.where(wet, R['profile'].mean(axis=0), np.nan), zcb, '-', color='k', lw=3, label='average')
axs[1].plot(100 * R['faces'].mean(axis=0), xcf, '-', color='k', lw=3)
dress(axs)
title(fig, 'thin grey = each %s, black = their average' % UNAME)
save(fig, 'every.png')


def class_figure(cl, show, subtitle, fn, alpha, grey, bg=False):
    """every residual whose class is in show, coloured by class; bg: the other classes behind, light grey"""
    fig, axs = plt.subplots(1, 2, figsize=(17, 7), layout='constrained')
    if bg:
        nb = ~np.isin(cl, show)
        axs[0].plot(100 * PROF[nb].T, zcb, '-', color='0.8', lw=0.5, alpha=0.3)
        axs[1].plot(100 * R['faces'][nb].T, xcf, '-', color='0.8', lw=0.5, alpha=0.3)
        axs[0].plot([], [], '-', color='0.8', lw=2, label='other classes (n = %d)' % nb.sum())
    for j in np.flatnonzero(np.isin(cl, show)):                      # time order: no class painted over another
        c = grey if CCOL[cl[j]] == '0.75' else CCOL[cl[j]]
        axs[0].plot(100 * PROF[j], zcb, '-', color=c, lw=0.6, alpha=alpha)
        axs[1].plot(100 * R['faces'][j], xcf, '-', color=c, lw=0.6, alpha=alpha)
    for c in show:
        axs[0].plot([], [], '-', color=grey if CCOL[c] == '0.75' else CCOL[c], lw=2, label='%s (n = %d)' % (c, (cl == c).sum()))
    dress(axs)
    title(fig, subtitle)
    save(fig, fn)


for sn, (cl, order, subtitle) in CLS.items():
    class_figure(cl, order, subtitle, 'every_%s.png' % sn, 0.35, '0.75')
    for c in order:                                                  # one class per figure
        class_figure(cl, [c], subtitle + '\nthis figure: %s (other classes light grey)' % c,
                     'every_%s_%s.png' % (sn, c.replace(' ', '_')), 0.5, '0.4', bg=True)

# ------------------------------------------------------------------ STEP 3: strength / shape split
X = {'profile': R['profile'][:, wet] - R['Ubar'][:, None],          # (n, b) layered shape
     'faces': R['faces'] - R['Ubar'][:, None]}                      # (n, p) lateral shape
XC = {'profile': RC['profile'][:, wet] - RC['Ubar'][:, None], 'faces': RC['faces'] - RC['Ubar'][:, None]}
W = {'profile': R['Abar'][:, wet].mean(axis=0), 'faces': (H_h[SP[0, 0]:SP[-1, 1]].mean(axis=0) * dy)}
Y = {'profile': R['profile'][:, wet], 'faces': R['faces']}
for f in FIELDS:
    tot = ((Y[f] - Y[f].mean(axis=0)) ** 2 * W[f]).sum()           # area-weighted variance about the mean
    shp = ((X[f] - X[f].mean(axis=0)) ** 2 * W[f]).sum()
    print('%-7s: shape is %.0f%% of the unit-to-unit residual variance (strength the rest, cross term incl.)' % (f, 100 * shp / tot))


# ------------------------------------------------------------------ STEP 4: EOFs of the shapes, per field
def eof(x, w, upper):
    """EOFs as 20261117_pc_profile_drivers.py (UW Lab 8-1). x (n units, s space), w (s,) area weights,
    upper (s,) bool = the 'upper' points for the sign."""
    mean = x.mean(axis=0)
    Xw = (x - mean).T * np.sqrt(w)[:, None]                          # (space, unit), weighted
    U, S, V = svd(Xw, full_matrices=False)
    var_exp = S ** 2 / (S ** 2).sum()
    nk = min(NM, np.searchsorted(np.cumsum(var_exp), VAR_KEEP) + 1)
    n = x.shape[0]
    U, S, V = U[:, :nk].copy(), S[:nk], V[:nk].copy()
    for i in range(nk):                                              # PC > 0 = upper water more INTO the cove
        e = U[:, i] / np.sqrt(w)
        if (e[upper] * w[upper]).sum() < 0:
            U[:, i], V[i] = -U[:, i], -V[i]
    return {'mean': mean, 'eof': U * S / np.sqrt(n) / np.sqrt(w)[:, None],   # (s, nk) [m/s per 1 std of PC]
            'pc': V * np.sqrt(n), 'var_exp': var_exp, 'U': U, 'S': S, 'w': w, 'n': n}


def project(E, x):
    """PCs (nk, n) of shapes x (n, s) on the EOFs E, same scaling as E['pc']"""
    return (E['U'].T @ ((x - E['mean']).T * np.sqrt(E['w'])[:, None])) / E['S'][:, None] * np.sqrt(E['n'])


ms = X['profile'].mean(axis=0)                                       # mean profile shape
zc = np.flatnonzero(np.sign(ms) != np.sign(ms[0]))
UPPER = {'profile': np.arange(len(ms)) < (zc[0] if len(zc) else len(ms) // 2),   # above the first zero crossing
         'faces': np.arange(len(dy)) < len(dy) // 2}                 # north half
EOF = {f: eof(X[f], W[f], UPPER[f]) for f in FIELDS}

# ------------------------------------------------------------------ STEP 5: persistence (and the day's split half)
nx = np.flatnonzero(SP[1:, 0] == SP[:-1, 1])                         # units followed by the next one
for f in FIELDS:
    E = EOF[f]
    nk = len(E['S'])
    E['ceiling'] = np.array([np.corrcoef(E['pc'][i][nx], E['pc'][i][nx + 1])[0, 1] for i in range(nk)])   # persistence
    note = ''
    if UNIT == 'day':                                                # split half: the day's two F+E cycles
        ic = {a: i for i, a in enumerate(SP_cyc[:, 0])}
        i1 = np.array([ic.get(a, -1) for a in SP[:, 0]])
        i2 = np.array([ic.get(SP_cyc[i, 1], -1) if i >= 0 else -1 for i in i1])
        ok = (i1 >= 0) & (i2 >= 0)
        ok[ok] = SP_cyc[i2[ok], 1] == SP[ok, 1]                      # the two cycles tile the day exactly
        p1, p2 = project(E, XC[f][i1[ok]]), project(E, XC[f][i2[ok]])
        note = '; split-half r (cycle 1 vs 2 of the day) %s (%d days)' % (
            ' '.join('%.2f' % np.corrcoef(p1[i], p2[i])[0, 1] for i in range(nk)), ok.sum())
    print('%-7s %d modes for %.0f%%: var %s; persistence r(next unit) %s%s'
          % (f, nk, 100 * E['var_exp'][:nk].sum(), ' '.join('%.0f' % (100 * v) for v in E['var_exp'][:nk]),
             ' '.join('%.2f' % c for c in E['ceiling']), note))

# ------------------------------------------------------------------ STEP 6: drivers per unit
def umean(x):
    """mean of the hourly x over each unit"""
    return np.array([np.nanmean(x[a:b]) for a, b in SP])


def prior(x):
    """mean of the hourly x over the LDAY h before each unit (NaN if not all inside the record)"""
    return np.array([np.mean(x[a - LDAY:a]) if a >= LDAY else np.nan for a, b in SP])


DRV = {'D2 env': umean(ENV['D2']), 'D1 env': umean(ENV['D1']),
       'tau_al': umean(t_al), 'tau_al prior': prior(t_al),
       'tau_cr': umean(t_cr), 'tau_cr prior': prior(t_cr),
       'drho': umean(drho_h)}
doy = 2 * np.pi * (ot[mid].dayofyear.to_numpy() - 1) / 365.25
DRV['season sin'], DRV['season cos'] = np.sin(doy), np.cos(doy)
if UNIT == 'cycle':
    hod = 2 * np.pi * (ot[mid].hour.to_numpy() + 0.5 * (nh % 2)) / 24  # UTC hour of the unit's centre
    DRV['hour sin'], DRV['hour cos'] = np.sin(hod), np.cos(hod)
DNAME = list(DRV)
GROUPS = {n: [n] for n in DNAME if not n.startswith(('season', 'hour'))}   # dominance groups; sin + cos together
GROUPS['season'] = ['season sin', 'season cos']
if UNIT == 'cycle':
    GROUPS['hour'] = ['hour sin', 'hour cos']
GNAME = list(GROUPS)
Z = np.column_stack([DRV[n] for n in DNAME])
OK = np.isfinite(Z).all(axis=1)
Z = (Z - Z[OK].mean(axis=0)) / Z[OK].std(axis=0)                     # z-scores over the units used
MONTH = ot[mid].month
print('%d of %d units have every driver' % (OK.sum(), len(OK)))
pd.set_option('display.width', 220)
print('driver correlation:')
print(pd.DataFrame(Z[OK], columns=DNAME).corr().round(2).to_string())

# ------------------------------------------------------------------ STEP 7: OLS of each PC (and Ubar) on all drivers, dominance, CV
GCOL = [[DNAME.index(c) for c in GROUPS[g]] for g in GNAME]
NG = len(GNAME)
NOSEAS = sum(1 << g for g in range(NG) if GNAME[g] != 'season')     # subset mask without season


def r2(y, Z):
    """in-sample R^2 of OLS y ~ 1 + Z"""
    if Z.shape[1] == 0:
        return 0.0
    X1 = np.column_stack([np.ones(len(y)), Z])
    res = y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]
    return 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()


def cols(m):
    return [c for g in range(NG) if m >> g & 1 for c in GCOL[g]]


def dominance(y, Z):
    """general dominance of each driver group (sums to the full R^2), full R^2, R^2 without season"""
    R2 = np.array([r2(y, Z[:, cols(m)]) for m in range(1 << NG)])
    dom = np.zeros(NG)
    for g in range(NG):
        by_size = [[] for _ in range(NG)]
        for m in range(1 << NG):
            if not m >> g & 1:
                by_size[bin(m).count('1')].append(R2[m | 1 << g] - R2[m])
        dom[g] = np.mean([np.mean(v) for v in by_size])
    return dom, R2[-1], R2[NOSEAS]


def r2_cv(y, Z, month):
    """R^2 of OLS y ~ 1 + Z, each calendar month predicted from a fit to the other months"""
    X1 = np.column_stack([np.ones(len(y)), Z])
    yh = np.zeros_like(y)
    for mo in np.unique(month):
        te = month == mo
        yh[te] = X1[te] @ np.linalg.lstsq(X1[~te], y[~te], rcond=None)[0]
    return 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()


RES = []
Zo, mo = Z[OK], MONTH[OK]
resp = [('strength', 'Ubar', R['Ubar'][OK], np.nan, np.nan)]
for f in FIELDS:
    E = EOF[f]
    resp += [(f, 'EOF%d' % (i + 1), E['pc'][i][OK], E['var_exp'][i], E['ceiling'][i]) for i in range(len(E['S']))]
for f, mode, y, ve, ceil in resp:
    dom, R2, R2ns = dominance(y, Zo)
    r = {'unit': UNIT, 'field': f, 'mode': mode, 'var_exp': ve, 'persist': ceil, 'R2': R2,
         'R2_cv': r2_cv(y, Zo, mo), 'R2_cv_noseason': r2_cv(y, Zo[:, cols(NOSEAS)], mo), 'season_added': R2 - R2ns}
    r.update({'dom ' + g: dom[i] for i, g in enumerate(GNAME)})
    RES.append(r)
    top = np.argsort(-dom)[:3]
    print('%-8s %-5s var %3s persist %5s | R2 %.2f cv %.2f | season last +%.2f | %s'
          % (f, mode, '' if np.isnan(ve) else '%.0f%%' % (100 * ve), '' if np.isnan(ceil) else '%.2f' % ceil,
             R2, r['R2_cv'], r['season_added'], ', '.join('%s %.2f' % (GNAME[g], dom[g]) for g in top)))
RES = pd.DataFrame(RES)
RES.to_csv(out_dir / 'results.csv', index=False, float_format='%.4f')
print('saved %s' % (out_dir / 'results.csv'))

# ------------------------------------------------------------------ STEP 8: RDA, the shape modes most explained by the drivers
RDA = {}
X1 = np.column_stack([np.ones(OK.sum()), Zo])
for f in FIELDS:
    E = EOF[f]
    x, w = X[f][OK], E['w']
    Yw = (x - x.mean(axis=0)) * np.sqrt(w)                          # (n, s) weighted anomalies
    Yh = X1 @ np.linalg.lstsq(X1, Yw, rcond=None)[0]                 # fitted part
    R2 = (Yh ** 2).sum() / (Yw ** 2).sum()
    n, q = Yw.shape[0], Zo.shape[1]
    Uh, Sh, Vh = svd(Yh, full_matrices=False)
    e1, e0 = Vh[0] * Sh[0] / np.sqrt(n) / np.sqrt(w), E['eof'][:, 0]
    if np.sign((e1 * e0 * w).sum()) < 0:
        e1, Uh[:, 0] = -e1, -Uh[:, 0]
    cosang = (e1 * e0 * w).sum() / np.sqrt((e1 ** 2 * w).sum() * (e0 ** 2 * w).sum())
    load = np.array([np.corrcoef(Zo[:, j], Uh[:, 0])[0, 1] for j in range(q)])   # driver loadings on RDA 1
    RDA[f] = {'R2': R2, 'R2_adj': 1 - (1 - R2) * (n - 1) / (n - q - 1), 'mode1': e1,
              'mode1_share': Sh[0] ** 2 / (Sh ** 2).sum(), 'cos_eof1': cosang, 'load': load}
    top = np.argsort(-np.abs(load))[:3]
    print('%-7s RDA: drivers explain %.0f%% (adj %.0f%%) of the shape variance; mode 1 is %.0f%% of that, cos(RDA1, EOF1) %.2f; '
          'loadings %s' % (f, 100 * R2, 100 * RDA[f]['R2_adj'], 100 * RDA[f]['mode1_share'], cosang,
                           ', '.join('%s %+.2f' % (DNAME[j], load[j]) for j in top)))

# ------------------------------------------------------------------ STEP 9: figures
MC = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377']   # mode colours
GC = dict(zip(GNAME, list(plt.get_cmap('tab10').colors) + ['#882255']))   # driver group colours
YF = {'profile': zcb[wet], 'faces': xcf}
FT = {'profile': 'Profile shape (layered)', 'faces': 'Face shape (lateral)'}
CEIL = 'persist = r(PC, PC of the next %s)' % UNIT


def shape_axes(ax, f, xl):
    ax.axvline(0, color='0.4', lw=0.8)
    ax.set_xlim(xl, -xl)                                             # + (into the cove) drawn LEFT
    ax.grid(color='lightgray', linestyle='--', alpha=0.5)
    if f == 'profile':
        ax.set_ylim(zbot, 0)
        ax.set_ylabel('depth below the surface [m]')
    else:
        ax.set_ylim(xe[-1], xe[0])                                   # north at the top
        ax.set_ylabel('distance across from the north end [km]\nS ← | → N')
    ax.set_title(FT[f], fontweight='bold')


# eofs.png
fig, axs = plt.subplots(1, 2, figsize=(15, 7), layout='constrained')
for ax, f in zip(axs, FIELDS):
    E = EOF[f]
    ax.plot(100 * E['mean'], YF[f], '-', color='k', lw=3, label='mean shape')
    for i in range(len(E['S'])):
        ax.plot(100 * E['eof'][:, i], YF[f], '-', color=MC[i], lw=2.5,
                label='EOF%d  %.0f%%, persist %.2f' % (i + 1, 100 * E['var_exp'][i], E['ceiling'][i]))
    shape_axes(ax, f, np.ceil(1.15 * 100 * max(np.abs(E['mean']).max(), np.abs(E['eof']).max())))
    ax.legend(loc='lower left', fontsize=9)
    ax.set_xlabel('u - Ubar [cm s$^{-1}$]   ← IN | OUT →   (EOF: per 1 std of its PC)')
fig.suptitle('pc_lj (Lovejoy), %d, %s: EOFs of the residual shape, every %s (n = %d), section mean Ubar removed\n'
             'PC > 0 = upper (profile) / north (faces) water more INTO the cove; %s' % (YEAR, GTAGEX, UNAME, len(SP), CEIL), fontsize=13)
save(fig, 'eofs.png')

# pcs.png: the PCs through the year
nk = max(len(EOF[f]['S']) for f in FIELDS)
fig, axs = plt.subplots(2, 1, figsize=(15, 8), layout='constrained', sharex=True)
for ax, f in zip(axs, FIELDS):
    E = EOF[f]
    for i in range(len(E['S'])):
        ax.plot(ot[mid], E['pc'][i], '-', color=MC[i], lw=1.2, label='EOF%d (%.0f%%)' % (i + 1, 100 * E['var_exp'][i]))
    ax.axhline(0, color='0.4', lw=0.8)
    ax.grid(color='lightgray', linestyle='--', alpha=0.5)
    ax.set_ylabel('PC [std]')
    ax.set_title(FT[f], fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, ncols=nk)
fig.suptitle('pc_lj (Lovejoy), %d, %s: residual shape PCs, every %s' % (YEAR, GTAGEX, UNAME), fontsize=13)
save(fig, 'pcs.png')

# dominance.png
fig, axs = plt.subplots(1, 2, figsize=(15, 6), layout='constrained', sharex=True)
for c, f in enumerate(FIELDS):
    ax = axs[c]
    sub = RES[RES['field'].isin([f, 'strength'])].reset_index(drop=True)
    for i, s in sub.iterrows():
        left = 0
        for g in GNAME:
            v = max(s['dom ' + g], 0)
            ax.barh(i, v, left=left, color=GC[g], height=0.6, label=g if (c, i) == (0, 0) else None)
            left += v
        ax.plot(s['R2_cv'], i, 'k|', ms=22, mew=3, label='CV R$^2$ (leave one month out)' if (c, i) == (0, 0) else None)
        if np.isfinite(s['persist']):
            ax.plot(max(s['persist'], 0), i, 'kv', ms=10, label='persistence r(next unit)' if (c, i) == (0, 1) else None)
    ax.set_yticks(range(len(sub)), [s['mode'] if s['mode'] == 'Ubar' else '%s (%.0f%%)' % (s['mode'], 100 * s['var_exp'])
                                    for _, s in sub.iterrows()])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.grid(axis='x', color='lightgray', linestyle='--', alpha=0.5)
    ax.set_title(FT[f] + ' EOFs', fontweight='bold')
    ax.set_xlabel('R$^2$ of the PC: bars = each driver\'s dominance share (sum = in-sample R$^2$)')
fig.legend(loc='outside lower center', ncols=6, fontsize=10, frameon=False)
fig.suptitle('pc_lj (Lovejoy), %d, %s, residual of every %s: what explains each shape EOF? OLS on all drivers, general dominance'
             % (YEAR, GTAGEX, UNAME), fontsize=13)
save(fig, 'dominance.png')

# rda.png
fig, axs = plt.subplots(1, 2, figsize=(15, 7), layout='constrained')
for ax, f in zip(axs, FIELDS):
    E, Rd = EOF[f], RDA[f]
    ax.plot(100 * E['eof'][:, 0], YF[f], '--', color='k', lw=2, label='EOF1 (%.0f%% of the variance)' % (100 * E['var_exp'][0]))
    ax.plot(100 * Rd['mode1'], YF[f], '-', color='#CC3311', lw=3, label='RDA mode 1 (%.0f%% of the explained part)' % (100 * Rd['mode1_share']))
    shape_axes(ax, f, np.ceil(1.15 * 100 * max(np.abs(E['eof'][:, 0]).max(), np.abs(Rd['mode1']).max())))
    top = np.argsort(-np.abs(Rd['load']))[:4]
    ax.text(0.98, 0.02, 'drivers explain %.0f%% of the shape variance (adj. %.0f%%)\ncos(RDA1, EOF1) = %.2f\nloadings on RDA1:\n%s'
            % (100 * Rd['R2'], 100 * Rd['R2_adj'], Rd['cos_eof1'], '\n'.join('  %s %+.2f' % (DNAME[j], Rd['load'][j]) for j in top)),
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlabel('u - Ubar [cm s$^{-1}$]   ← IN | OUT →   (per 1 std)')
fig.suptitle('pc_lj (Lovejoy), %d, %s, residual of every %s: RDA, the shape most explained by the drivers, against EOF1'
             % (YEAR, GTAGEX, UNAME), fontsize=13)
save(fig, 'rda.png')

# ------------------------------------------------------------------ STEP 10: candidate drivers added in 20261117_pc_profile_drivers.py STEP 9
dsb = xr.open_dataset(box_fn).sel(ocean_time=slice(t0, t1 + PAD))
mask = dsb['mask_rho'].values == 1
sig_ts = {}
for sn, XS in [('pc_cp', 15), ('pc_lp', 30)]:            # wet faces: rho cells both sides wet
    jj = np.flatnonzero(mask[:, XS] & mask[:, XS + 1])
    sl = dict(s_rho=[0, -1], eta_rho=jj, xi_rho=[XS, XS + 1])
    sa, te = dsb['salt'].isel(**sl).values, dsb['temp'].isel(**sl).values          # (t, bot/top, face, 2)
    lo, la = dsb['lon_rho'].isel(eta_rho=jj, xi_rho=[XS, XS + 1]).values, dsb['lat_rho'].isel(eta_rho=jj, xi_rho=[XS, XS + 1]).values
    SA_ = gsw.SA_from_SP(sa, 0, lo, la)
    sg_ = gsw.sigma0(SA_, gsw.CT_from_pt(SA_, te)).mean(axis=-1)                   # (t, bot/top, face)
    w_ = (1 / dsb['pn'].isel(eta_rho=jj, xi_rho=[XS, XS + 1]).values).mean(axis=-1)  # face width
    sig_ts[sn] = (sg_ * w_).sum(axis=-1) / w_.sum()                                  # (t, bot/top) width-mean
cove = mask.copy()
cove[:, 31:] = False                                     # cove water: rho columns <= 30
zeta_c = dsb['zeta'].values[:, cove].mean(axis=1)
dsb.close()
dm = xr.open_dataset(mix_fn).sel(ocean_time=slice(t0, t1 + PAD))
shf = dm['shflux'].values[:, cove].mean(axis=1)
dm.close()

aQ = umean(np.abs(Q))
NEW = {'drhox top': umean(sig_ts['pc_lp'][:, 1] - sig_ts['pc_cp'][:, 1]),
       'drhox bot': umean(sig_ts['pc_lp'][:, 0] - sig_ts['pc_cp'][:, 0]),
       'shflux': umean(shf), 'shflux prior': prior(shf),
       '|Q|': aQ, 'prev |Q|': np.r_[np.nan, np.where(SP[1:, 0] == SP[:-1, 1], aQ[:-1], np.nan)],
       'zeta': umean(zeta_c), 'Q net': umean(Q)}
NNAME = list(NEW)
for n in NNAME:
    print('%-13s per unit: %9.3f to %9.3f (median %.3f)' % (n, np.nanmin(NEW[n]), np.nanmax(NEW[n]), np.nanmedian(NEW[n])))

# ------------------------------------------------------------------ STEP 11: linear vs boosted trees, old vs all drivers
GROUPS_ALL = dict(GROUPS, **{n: [n] for n in NNAME})
ALLNAME = DNAME + NNAME


def gbt():
    return HistGradientBoostingRegressor(max_depth=GBT_DEPTH, max_iter=GBT_ITER, learning_rate=GBT_LR,
                                         min_samples_leaf=GBT_LEAF, random_state=0)


def cv_pred(make, Xm, y, month, perm_groups=None):
    """leave-one-month-out predictions; with perm_groups {g: cols}, also the predictions with g shuffled in the
    held-out month (NPERM repeats): returns yh, {g: (NPERM, n)}"""
    rng = np.random.default_rng(0)
    yh = np.zeros_like(y)
    yp = {g: np.zeros((NPERM, len(y))) for g in (perm_groups or {})}
    for mo_ in np.unique(month):
        te = month == mo_
        m = make().fit(Xm[~te], y[~te])
        yh[te] = m.predict(Xm[te])
        for g, cc in (perm_groups or {}).items():
            for k in range(NPERM):
                Xp = Xm[te].copy()
                Xp[:, cc] = Xp[rng.permutation(te.sum())][:, cc]  # one permutation for the whole group
                yp[g][k, te] = m.predict(Xp)
    return yh, yp


class OLS:
    def fit(self, Xm, y):
        self.b = np.linalg.lstsq(np.column_stack([np.ones(len(y)), Xm]), y, rcond=None)[0]
        return self

    def predict(self, Xm):
        return np.column_stack([np.ones(len(Xm)), Xm]) @ self.b


R2f = lambda y, yh: 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()
Zn = np.column_stack([NEW[n] for n in NNAME])
ok = OK & np.isfinite(Zn).all(axis=1)
Zn = (Zn - Zn[ok].mean(axis=0)) / Zn[ok].std(axis=0)
Xo, Xa, mo = Z[ok], np.column_stack([Z, Zn])[ok], MONTH[ok]
pg = {g: [ALLNAME.index(c) for c in cc] for g, cc in GROUPS_ALL.items()}
print('%d of %d units have every driver (old + new)' % (ok.sum(), len(ok)))
NL = []
for f in FIELDS:
    E = EOF[f]
    for i in range(len(E['S'])):
        y = E['pc'][i][ok]
        r = {'unit': UNIT, 'field': f, 'mode': 'EOF%d' % (i + 1), 'var_exp': E['var_exp'][i], 'persist': E['ceiling'][i], 'n': ok.sum()}
        r['lin old'] = R2f(y, cv_pred(OLS, Xo, y, mo)[0])
        r['lin all'] = R2f(y, cv_pred(OLS, Xa, y, mo)[0])
        r['gbt old'] = R2f(y, cv_pred(gbt, Xo, y, mo)[0])
        yh, yp = cv_pred(gbt, Xa, y, mo, pg)
        r['gbt all'] = R2f(y, yh)
        for g in GROUPS_ALL:
            r['perm ' + g] = r['gbt all'] - np.mean([R2f(y, yp[g][k]) for k in range(NPERM)])
        NL.append(r)
        top = sorted(GROUPS_ALL, key=lambda g: -r['perm ' + g])[:4]
        print('%-7s %-4s persist %5.2f | CV R2: lin old %.2f, lin all %.2f, gbt old %.2f, gbt all %.2f | %s'
              % (f, r['mode'], r['persist'], r['lin old'], r['lin all'], r['gbt old'], r['gbt all'],
                 ', '.join('%s %.2f' % (g, r['perm ' + g]) for g in top)))
NL = pd.DataFrame(NL)
NL.to_csv(out_dir / 'nonlinear.csv', index=False, float_format='%.4f')
print('saved %s' % (out_dir / 'nonlinear.csv'))

# ------------------------------------------------------------------ STEP 12: figure
GCA = dict(GC, **dict(zip(NNAME, plt.get_cmap('Dark2').colors)))
fig, axs = plt.subplots(2, 2, figsize=(16, 11), layout='constrained', sharex=True)
for r_, f in enumerate(FIELDS):
    for c, mode in enumerate(['EOF1', 'EOF2']):
        ax = axs[r_, c]
        sel = NL[(NL['field'] == f) & (NL['mode'] == mode)]
        if len(sel) == 0:
            ax.set_visible(False)
            continue
        s_ = sel.iloc[0]
        gs = sorted(GROUPS_ALL, key=lambda g: s_['perm ' + g])
        ax.barh(range(len(gs)), [s_['perm ' + g] for g in gs], color=[GCA[g] for g in gs],
                hatch=['//' if g in NNAME else '' for g in gs], edgecolor='k', lw=0.5)
        ax.set_yticks(range(len(gs)), gs, fontsize=9)
        ax.axvline(0, color='0.4', lw=0.8)
        ax.grid(axis='x', color='lightgray', linestyle='--', alpha=0.5)
        ax.text(0.98, 0.04, 'CV R$^2$\nlin old %.2f\nlin all %.2f\ngbt old %.2f\ngbt all %.2f\npersist %.2f'
                % (s_['lin old'], s_['lin all'], s_['gbt old'], s_['gbt all'], s_['persist']),
                transform=ax.transAxes, ha='right', va='bottom', fontsize=10, bbox=dict(fc='w', ec='0.7'))
        ax.set_title('%s %s (%.0f%%)' % (FT[f], mode, 100 * s_['var_exp']), fontweight='bold')
for ax in axs[-1]:
    ax.set_xlabel('permutation importance (drop in CV R$^2$ of gbt all when the driver is shuffled)')
fig.suptitle('pc_lj (Lovejoy), %d, %s, residual of every %s: boosted trees on all drivers, leave-one-month-out; hatched = new (STEP 10) drivers'
             % (YEAR, GTAGEX, UNAME), fontsize=13)
save(fig, 'nonlinear.png')
