"""
Penn Cove pc_lj (Lovejoy): what modifies the SHAPE of the flood / ebb velocity profile most?
Every 2024 flood and ebb at its peak hour (greater / lesser of each lunar day, as
20261117_pc_every_flood_ebb.py), split into strength + shape, the shapes reduced to a few EOFs,
and each EOF's amplitude regressed on all candidate drivers together.

STEP 1  load pc_lj from the pc_cove box and the peak hours (code of 20261117_pc_every_flood_ebb.py
        STEPS 1-3): per tide the cross-channel profile U_b = Q_b / A_b in 1 m bins below the
        moving surface, and the depth average at each face U_p = sum_k u dz / sum_k dz.
STEP 2  strength / shape split, per tide:
            strength  Ubar = Q / A, the section-mean u (the barotropic part; |Q| tracks it)
            shape     profile  U_b - Ubar  (layered: vertical shear / exchange)
                      faces    U_p - Ubar  (lateral: in-north / out-south)
        Without it the strongest tides dominate every test (|Q| explained 28-37% of the face
        variance in the first pass); the shape is what the drivers should change.
        Bins: only those wet in EVERY tide of the row at the peak and +/- 1 h (no gap filling), so
        each sample is the same vector.
STEP 3  EOFs of the shapes, one SVD per ROW (greater flood, lesser flood, greater ebb, lesser ebb;
        ~350 tides each, the user's choice: each class its own modes, as the 4 x 2 figures) and
        field (profile, faces). Done as 20261110_pc_velocity_eof.py (UW Lab 8-1):
        X (space, tide) = shape anomaly (tide mean removed) * sqrt(w), U, S, V = svd(X), var_exp =
        S**2 / sum S**2, w = time-mean bin area A_b (profile) or face area H dy (faces), so S**2 is
        area-integrated variance; physical scaling EOF_i = U S / sqrt(n) / sqrt(w), PC_i = V sqrt(n);
        sign: PC > 0 = upper water moving more INTO the cove than the section mean (profile: area-
        weighted EOF above the median zero crossing of the mean shape > 0; faces: north half > 0).
        Keep the modes that reach 90% of the variance (at most NM).
STEP 4  noise ceiling: the same tides' profiles 1 h before and 1 h after the peak, projected on
        the STEP 3 EOFs. corr(PC(peak - 1 h), PC(peak + 1 h)) = the share of each PC's variance
        that persists over 2 h; hour-to-hour eddies / internal motions are in the rest, and no
        subtidal driver can explain them. It is the target for STEP 6's R^2.
        CORRECTION (2026-11-17, STEP 12): the PCs are not white noise from hour to hour. r(peak, -/+1 h) is
        0.88-0.97 but r(-1 h, +1 h) only 0.62-0.88, so the shape decorrelates smoothly (red, ~10-15 h scale), and
        averaging +/-1 h barely changes the PCs (r 0.98-1.00) or the CV R^2. This 'ceiling' is a 2 h
        autocorrelation, NOT the predictable share; hourly noise is at most ~5-12% of the variance, so the
        unexplained ~0.4-0.5 is persistent over many hours.
STEP 5  drivers at each peak hour, standardised (z-scores), all continuous (no terciles):
            tide      |Q| at the peak (this tide's strength); D2 and D1 envelopes of the pc_lj
                      tidal transport (20261112_pc_utide fits, as 20261116 STEP 5)
            wind      tau_al, tau_cr, the applied cove-mean stress (pc_cove_mix box): Godin AND the
                      mean over the 6 h before the peak (the surface layer answers in hours)
            density   drho = sigma0 bottom - top at pc_lj, Godin AND at the peak hour
            season    sin and cos of the day of year (a proxy, not a mechanism: STEP 6 asks
                      whether it explains anything the physical drivers do not)
            hour      sin and cos of the hour of day of the peak (UTC; the phase is free in the fit).
                      Added 2026-11-17: the tides drift through the solar day over the year, and the
                      'season' effect on the floods and lesser ebbs was this (the S1 two-layer daily
                      exchange of 20261110_pc_cove_tide_maps.py); with it, season adds ~0 there.
        Tides with a NaN driver (Godin, first 35 h of 2024) are left out of STEPS 6-7.
        Printed: the driver correlation matrix (collinearity is the reason for STEP 6's method).
STEP 6  per row, field and kept PC: OLS of the PC on all drivers together.
            R^2 in-sample and cross-validated leaving one calendar month out (neighbouring tides
            share the same subtidal state, so they are not independent samples)
            dominance analysis (general dominance): each driver's added R^2 averaged over every
            subset of the other drivers, so correlated drivers share credit instead of the first
            one taking it; the shares sum to the full-model R^2. Drivers enter as GROUPS (season =
            sin + cos, hour = sin + cos), 11 groups, all 2^11 subsets fitted.
            the same with season left out, then season added last: its added R^2 = what the
            physical drivers miss
        Also for the strength Ubar (a scalar), to set against the shape.
STEP 7  RDA (redundancy analysis) cross-check, per row and field: regress the whole weighted
        shape matrix on the drivers, SVD of the fitted part = the shape modes MOST explained by
        the drivers (EOFs are the modes of most variance, which may be the noise). Total R^2 of
        the shapes and the first RDA mode against EOF 1.
STEP 8  figures:
            eofs.png       4 x 2 (rows x profile | faces): mean shape + EOF 1..k, var_exp and ceiling
            dominance.png  4 x 2: per PC (and the strength Ubar) each driver's dominance share as
                           stacked bars = in-sample R^2, CV R^2 and the ceiling marked
            rda.png        4 x 2: RDA mode 1 beside EOF 1, total R^2 and the leading driver loadings
            results.csv    the STEP 6 table
STEP 9  (2026-11-17) candidate drivers not in STEP 5, at the peak hours:
            drhox top / bot  in-out density difference, sigma0 (gsw) at pc_lp minus pc_cp, width-mean of
                             the top / bottom s-level of the wet faces' rho cells, Godin (+ = denser
                             outside: the horizontal gradient that drives the exchange, not the vertical
                             stratification drho)
            shflux 6h / Godin  surface net heat flux (+ into the ocean), cove-mean (rho cols <= 30), mean
                             of the WIND_H h up to the peak and Godin (night cooling vs day heating)
            prev |Q|         |Q| at the peak of the half-cycle just before (opposite phase) and of the
                             same-phase one before that (tidal history)
            zeta Godin       subtidal cove-mean sea level (remote setup)
            Q Godin          subtidal net transport through pc_lj
            (added after the first STEP 9 run, 2026-11-17)
            Skagit 3d        Skagit River flow in the model forcing (20260806_river_hydrographs/daily_flow.csv,
                             daily, held through the day), lagged SKAGIT_LAG (lag of best r with pc_lj
                             stratification, 20261110_pc_skagit_scenarios.py)
            tau_sar al / cr  wind stress over Saratoga Passage: hourly atm00 wind cube of 20260806_wind
                             (3.2 km), water cells in SAR_BOX, Large & Pond stress, onto SAR_AXIS (+ up-
                             passage, toward the NW / Penn Cove) and across it; 6 h and Godin
            (eddy state, the 20261110_pc_metric_ts.py back-cove ring, was tried and REMOVED 2026-11-17: the
            eddy is itself set by the tide (tropic > equatorial) and stratification, so it is a step in the
            chain, not an independent driver; it took D1's credit for the greater flood)
            (added 2026-11-17 after STEP 12 showed the unexplained part persists for hours)
            density profile  sigma0 (gsw) at pc_lj in the same 1 m bins below the surface, cross-channel
                             (area-weighted) mean, at the PEAK hour and at the SLACK that starts the half-
                             cycle (the column the tide begins with; the peak one is partly the flow's own
                             straining, so partly circular):
                pycno z_half   depth where sigma0 crosses halfway between the top and the deepest common bin
                pycno z_maxN2  depth of the largest bin-to-bin increase of sigma0
                rhoprof 1-3    PCs of the first NRHO EOFs of the density shape (each profile minus its depth
                               mean; SVD over all peak hours, slack profiles projected on the same EOFs);
                               one dominance / permutation group per time (peak, slack)
STEP 10 per row, field and PC, leave-one-month-out CV R^2 of four models:
            lin old   OLS, STEP 5 drivers (as STEP 6)        lin all   OLS, STEP 5 + STEP 9 drivers
            gbt old   gradient-boosted trees, STEP 5         gbt all   gradient-boosted trees, STEP 5 + 9
        (sklearn HistGradientBoostingRegressor, shallow trees: max_depth GBT_DEPTH, GBT_ITER iterations,
        learning rate GBT_LR, >= GBT_LEAF tides per leaf). Reading: gbt old >> lin old = interactions /
        nonlinearity among the drivers we had; lin all or gbt all >> old = missing drivers; neither = the
        rest is internal (eddy) variability, below the ceiling. Permutation importance of 'gbt all': the
        drop in CV R^2 when one driver group is shuffled within the held-out month (NPERM repeats).
STEP 11 figure nonlinear.png: one heat map of the permutation importance of 'gbt all', columns = rows x
        (profile EOF1, EOF2, face EOF1, EOF2, EOF3) with each EOF's variance share and CV R^2, driver rows in
        plain names grouped tide / wind / density / other, only drivers reaching SHOW somewhere; the top
        driver of each EOF boxed. Table nonlinear.csv (all PCs, all four models).
STEP 12 (2026-11-17) is the gap below the ceiling hour-to-hour noise? Each tide's profile shape averaged over
        peak - 1 h, peak, peak + 1 h, new EOFs (as STEP 3), the PCs refitted on the same drivers (lin all and
        gbt all, leave-one-month-out). If the ceiling is right and the rest is noise independent from hour
        to hour (the STEP 4 assumption), averaging 3 hours cuts the noise variance to 1/3 and the CV R^2 should
        rise to R^2_1h / (c + (1 - c) / 3), c = the STEP 4 ceiling. Rises that much: the gap was short-lived
        noise. Barely moves: the noise persists over hours (eddies), or the ceiling counts the systematic
        change through the peak as signal and overstates what is predictable.
STEP 13 figure avg3.png: per row and profile EOF1 | EOF2, CV R^2 single hour vs 3-h average (gbt all), the
        white-noise expectation and the ceiling; table avg3.csv.
STEP 14 (2026-11-18) the REDUCED LINEAR model, the standard way to attribute EOFs (STEPS 9-11 kept as the
        exploratory, boosted-tree version): one driver per process, 14 groups / 16 columns:
            tide     this tide |Q|, D1 envelope, previous half-cycle |Q|
            wind     along-cove stress 6 h and subtidal, cross-cove stress subtidal
            density  in-out sigma0 difference surface and bottom (subtidal); top-bottom drho and the half-density
                     depth (pycnocline) at the SLACK that starts the half-cycle (the column the tide starts with)
                     AND their CHANGE from slack to peak (the half-cycle's own straining, so partly a response;
                     slack alone loses ~0.1-0.2 of CV R^2 on the ebbs). Entered as slack + change, not slack + peak:
                     slack and peak are nearly the same series (VIF 21-57, coefficients +-1.5-2 of opposite sign),
                     the change carries the same information without the collinearity
            other    hour of day (sin, cos), season (sin, cos)
        (dropped: D2 envelope and the earlier tide |Q| (tide size via this tide |Q|), the 6 h cross / Saratoga winds,
        Godin drho (r 0.98 with peak drho), Skagit (carried by the density), heat flux, sea level, subtidal Q, the
        density-shape PCs (no physical units)). Per row x field x kept PC, drivers z-scored, PC standardised:
            beta      OLS standardised coefficient (SD of the PC per SD of the driver), all drivers together, with a
                      95% CI from NBOOT bootstrap draws of whole calendar months (tides within a month share the
                      subtidal state, so per-tide standard errors would be too small)
            dominance general dominance of each group (sums to the in-sample R^2)
            R^2       in-sample, CV leaving one month out, and CV of boosted trees on the SAME drivers (the
                      robustness check: trees >> OLS would mean nonlinearity the linear model misses)
            VIF       printed per row (collinearity check)
        Table linear.csv.
STEP 15 figure linear.png: a coefficient plot, one panel each for profile EOF1, profile EOF2 and face EOF1 (face
        EOF2-3 are poorly explained, CV R^2 <= 0.27, so left out): for every driver the standardised coefficient
        of each row (dot, filled if its 95% CI excludes 0) with the CI (bar); the CV R^2 of each row in the panel
        title. Hour of day and season are in the model as controls but not drawn (sin / cos have no single sign).
        The dominance shares and the trees' CV R^2 are in linear.csv.

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc
       LO_output/extract/[gtagex]/box/pc_cove_mix_[ds0]_[ds1].nc
       LO_output/DM_outs/20261112_pc_utide/fits.p
       LO_output/DM_outs/20261113_pc_candidate_days/picks.csv
OUTPUT LO_output/DM_outs/20261117_pc_profile_drivers/

run 20261117_pc_profile_drivers.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gsw
import itertools
import numpy as np
import pandas as pd
import pickle
import utide
import xarray as xr
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
WIND_H = 6                                # hours before the peak averaged for the short-term wind
NM = 6                                    # most EOF modes kept
VAR_KEEP = 0.9                            # keep modes until this share of the shape variance
GBT_DEPTH, GBT_ITER, GBT_LR, GBT_LEAF = 3, 200, 0.05, 20   # boosted trees (STEP 10)
NPERM = 5                                 # permutation repeats (STEP 10)
SKAGIT_LAG = pd.Timedelta(days=3)         # Skagit flow lag (STEP 9)
SAR_BOX = (-122.62, -122.42, 48.02, 48.22)   # Saratoga Passage wind cells: lon0, lon1, lat0, lat1 (STEP 9)
SAR_AXIS = (-0.55, 0.83)                  # Saratoga Passage axis, + up-passage (toward the NW)
NRHO = 3                                  # density-profile EOFs used as drivers (STEP 9)
ROWS = [('flood', 'greater'), ('flood', 'lesser'), ('ebb', 'greater'), ('ebb', 'lesser')]
FIELDS = ['profile', 'faces']

gridname, tag, ex_name = GTAGEX.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
box_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_%s_%s.nc' % (DS0, DS1))
mix_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_mix_%s_%s.nc' % (DS0, DS1))
fit_fn = Ldir['LOo'] / 'DM_outs' / '20261112_pc_utide' / 'fits.p'
pick_fn = Ldir['LOo'] / 'DM_outs' / '20261113_pc_candidate_days' / 'picks.csv'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261117_pc_profile_drivers'
Lfun.make_dir(out_dir)

# ------------------------------------------------------------------ STEP 1: load pc_lj, peak hours, binned profiles
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


WET = 0.5 * dy.mean()                                                # a bin counts as wet with at least half a face
OFF = [-1, 0, 1]                                                     # hours relative to the peak
zcb = 0.5 * (EDGES[1:] + EDGES[:-1])                                 # bin centres, depth below the surface
xe = np.r_[0, np.cumsum(dy)] / 1000                                  # face edges [km] from the north end
xcf = 0.5 * (xe[1:] + xe[:-1])                                       # face centres
D = {}                                                               # per row: hours, Ubar, prof, faces, weights
for row in ROWS:
    k = PK[row]
    hh = np.stack([k + o for o in OFF])                              # (3, n) hours
    Qb, Ab = binned_QA(u[hh.ravel()], zw[hh.ravel()], dy)
    Qb, Ab = Qb.reshape(3, len(k), -1), Ab.reshape(3, len(k), -1)    # (3, n, b)
    wet = (Ab >= WET).all(axis=(0, 1))                               # bins wet in every tide at all 3 hours
    UDZ, H = (u[hh] * dz[hh]).sum(axis=2), dz[hh].sum(axis=2)        # (3, n, p)
    D[row] = {'hrs': k, 'Ubar': Q[hh] / A[hh],                       # (3, n) section-mean u
              'profile': Qb[:, :, wet] / Ab[:, :, wet], 'faces': UDZ / H,
              'w_profile': Ab[1][:, wet].mean(axis=0), 'w_faces': (H[1] * dy).mean(axis=0),
              'z': zcb[wet]}
    print('%-7s %-5s %3d tides; profile bins wet in all: 0-%.0f m (%d bins); Ubar at peak median %+5.1f cm/s'
          % (row[1], row[0], len(k), EDGES[1:][wet].max(), wet.sum(), 100 * np.median(D[row]['Ubar'][1])))

# ------------------------------------------------------------------ STEP 2: strength / shape split
for row in ROWS:
    d = D[row]
    d['shape_profile'] = d['profile'] - d['Ubar'][:, :, None]       # (3, n, b) layered shape
    d['shape_faces'] = d['faces'] - d['Ubar'][:, :, None]           # (3, n, p) lateral shape
    for f in FIELDS:
        x, w = d['shape_' + f][1], d['w_' + f]
        tot = ((d[f][1] - d[f][1].mean(axis=0)) ** 2 * w).sum()     # area-weighted variance about the row mean
        shp = ((x - x.mean(axis=0)) ** 2 * w).sum()
        print('  %-7s %-5s %-7s: shape is %2.0f%% of the tide-to-tide variance (strength the rest, cross term incl.)'
              % (row[1], row[0], f, 100 * shp / tot))

# ------------------------------------------------------------------ STEP 3: EOFs of the shapes, per row and field
def eof(x, w, upper):
    """EOFs as 20261110_pc_velocity_eof.py (UW Lab 8-1). x (n tides, s space) shape at the peak, w (s,) area
    weights, upper (s,) bool = the 'upper' points for the sign. Returns the dict of the row/field."""
    mean = x.mean(axis=0)
    X = (x - mean).T * np.sqrt(w)[:, None]                           # (space, tide), weighted
    U, S, V = svd(X, full_matrices=False)
    var_exp = S ** 2 / (S ** 2).sum()
    nk = min(NM, np.searchsorted(np.cumsum(var_exp), VAR_KEEP) + 1)  # modes to reach VAR_KEEP
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


EOF = {}
for row in ROWS:
    d = D[row]
    ms = d['shape_profile'][1].mean(axis=0)                          # mean profile shape
    zc = np.flatnonzero(np.sign(ms) != np.sign(ms[0]))
    up_prof = np.arange(len(ms)) < (zc[0] if len(zc) else len(ms) // 2)   # above the first zero crossing
    up_face = np.arange(len(dy)) < len(dy) // 2                      # north half
    for f, upper in [('profile', up_prof), ('faces', up_face)]:
        E = eof(d['shape_' + f][1], d['w_' + f], upper)
        # ------------------------------------------------------------------ STEP 4: noise ceiling from peak -/+ 1 h
        pm, pp = project(E, d['shape_' + f][0]), project(E, d['shape_' + f][2])
        E['ceiling'] = np.array([np.corrcoef(pm[i], pp[i])[0, 1] for i in range(len(E['S']))])
        E['r_pk_pm1'] = np.array([np.corrcoef(E['pc'][i], 0.5 * (pm[i] + pp[i]))[0, 1] for i in range(len(E['S']))])
        EOF[(row, f)] = E
        print('%-7s %-5s %-7s %d modes for %.0f%%: var %s; ceiling r(-1h, +1h) %s'
              % (row[1], row[0], f, len(E['S']), 100 * E['var_exp'][:len(E['S'])].sum(),
                 ' '.join('%.0f' % (100 * v) for v in E['var_exp'][:len(E['S'])]),
                 ' '.join('%.2f' % c for c in E['ceiling'])))

# ------------------------------------------------------------------ STEP 5: drivers at every hour, then at the peak hours
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
GNAME = list(GROUPS)

for row in ROWS:
    k = D[row]['hrs']
    Z = np.column_stack([DRV[n][k] for n in DNAME])               # (n, drivers)
    ok = np.isfinite(Z).all(axis=1)
    D[row]['ok'] = ok
    D[row]['Z'] = (Z - np.nanmean(Z[ok], axis=0)) / np.nanstd(Z[ok], axis=0)   # z-scores over the tides used
    D[row]['month'] = ot[k].month
    print('%-7s %-5s %3d of %d tides have every driver' % (row[1], row[0], ok.sum(), len(ok)))
kall = np.concatenate([D[row]['hrs'] for row in ROWS])
C = pd.DataFrame({n: DRV[n][kall] for n in DNAME}).corr()
pd.set_option('display.width', 220)
print('driver correlation, all peak hours:')
print(C.round(2).to_string())

# ------------------------------------------------------------------ STEP 6: OLS of each PC (and Ubar) on all drivers, dominance, CV
GCOL = [[DNAME.index(c) for c in GROUPS[g]] for g in GNAME]
NG = len(GNAME)
NOSEAS = sum(1 << g for g in range(NG) if GNAME[g] != 'season')   # subset mask without season


def r2(y, Z):
    """in-sample R^2 of OLS y ~ 1 + Z"""
    if Z.shape[1] == 0:
        return 0.0
    X = np.column_stack([np.ones(len(y)), Z])
    res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
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
    X = np.column_stack([np.ones(len(y)), Z])
    yh = np.zeros_like(y)
    for mo in np.unique(month):
        te = month == mo
        yh[te] = X[te] @ np.linalg.lstsq(X[~te], y[~te], rcond=None)[0]
    return 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()


RES = []
for row in ROWS:
    d = D[row]
    ok, Z, mo = d['ok'], d['Z'][d['ok']], d['month'][d['ok']]
    resp = [('strength', 'Ubar', d['Ubar'][1][ok], np.nan, np.nan)]
    for f in FIELDS:
        E = EOF[(row, f)]
        resp += [(f, 'EOF%d' % (i + 1), E['pc'][i][ok], E['var_exp'][i], E['ceiling'][i]) for i in range(len(E['S']))]
    for f, mode, y, ve, ceil in resp:
        dom, R2, R2ns = dominance(y, Z)
        r = {'row': '%s %s' % row, 'field': f, 'mode': mode, 'var_exp': ve, 'ceiling': ceil, 'R2': R2,
             'R2_cv': r2_cv(y, Z, mo), 'R2_cv_noseason': r2_cv(y, Z[:, cols(NOSEAS)], mo), 'season_added': R2 - R2ns}
        r.update({'dom ' + g: dom[i] for i, g in enumerate(GNAME)})
        RES.append(r)
        top = np.argsort(-dom)[:3]
        print('%-14s %-8s %-5s var %3s ceil %4s | R2 %.2f cv %.2f | season last +%.2f | %s'
              % (r['row'], f, mode, '' if np.isnan(ve) else '%.0f%%' % (100 * ve), '' if np.isnan(ceil) else '%.2f' % ceil,
                 R2, r['R2_cv'], r['season_added'], ', '.join('%s %.2f' % (GNAME[g], dom[g]) for g in top)))
RES = pd.DataFrame(RES)
RES.to_csv(out_dir / 'results.csv', index=False, float_format='%.4f')
print('saved %s' % (out_dir / 'results.csv'))

# ------------------------------------------------------------------ STEP 7: RDA, the shape modes most explained by the drivers
RDA = {}
for row in ROWS:
    d = D[row]
    ok, Z = d['ok'], d['Z'][d['ok']]
    X1 = np.column_stack([np.ones(ok.sum()), Z])
    for f in FIELDS:
        E = EOF[(row, f)]
        x, w = d['shape_' + f][1][ok], E['w']
        Y = (x - x.mean(axis=0)) * np.sqrt(w)                       # (n, s) weighted anomalies
        Yh = X1 @ np.linalg.lstsq(X1, Y, rcond=None)[0]              # fitted part
        R2 = (Yh ** 2).sum() / (Y ** 2).sum()
        n, q = Y.shape[0], Z.shape[1]
        Uh, Sh, Vh = svd(Yh, full_matrices=False)
        e1 = Vh[0] * Sh[0] / np.sqrt(n) / np.sqrt(w)                # RDA mode 1 [m/s per 1 std of its score]
        e0 = E['eof'][:, 0]
        if np.sign((e1 * e0 * w).sum()) < 0:
            e1, Uh[:, 0] = -e1, -Uh[:, 0]
        cosang = (e1 * e0 * w).sum() / np.sqrt((e1 ** 2 * w).sum() * (e0 ** 2 * w).sum())
        load = np.array([np.corrcoef(Z[:, j], Uh[:, 0])[0, 1] for j in range(q)])   # driver loadings on RDA 1
        RDA[(row, f)] = {'R2': R2, 'R2_adj': 1 - (1 - R2) * (n - 1) / (n - q - 1), 'mode1': e1,
                         'mode1_share': Sh[0] ** 2 / (Sh ** 2).sum(), 'cos_eof1': cosang, 'load': load}
        top = np.argsort(-np.abs(load))[:3]
        print('%-7s %-5s %-7s RDA: drivers explain %.0f%% (adj %.0f%%) of the shape variance; mode 1 is %.0f%% of that, '
              'cos(RDA1, EOF1) %.2f; loadings %s'
              % (row[1], row[0], f, 100 * R2, 100 * RDA[(row, f)]['R2_adj'], 100 * RDA[(row, f)]['mode1_share'], cosang,
                 ', '.join('%s %+.2f' % (DNAME[j], load[j]) for j in top)))

# ------------------------------------------------------------------ STEP 8: figures
plt.rcParams.update({'font.size': 12})
MC = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377']   # mode colours
GC = dict(zip(GNAME, list(plt.get_cmap('tab10').colors) + ['#882255']))     # driver group colours


def shape_axes(ax, f, row, xl):
    ax.axvline(0, color='0.4', lw=0.8)
    ax.set_xlim(xl, -xl)                                                 # + (into the cove) drawn LEFT
    ax.grid(color='lightgray', linestyle='--', alpha=0.5)
    if f == 'profile':
        ax.set_ylim(D[row]['z'].max() + 1, 0)
        ax.set_ylabel('%s %s\ndepth below the surface [m]' % (row[1].upper(), row[0].upper()), fontweight='bold')
    else:
        ax.set_ylim(xe[-1], xe[0])                                       # north at the top
        ax.set_ylabel('distance across from the north end [km]\nS ← | → N')


def yfield(row, f):
    return D[row]['z'] if f == 'profile' else xcf


# eofs.png
fig, axs = plt.subplots(len(ROWS), 2, figsize=(15, 20), layout='constrained')
for r, row in enumerate(ROWS):
    for c, f in enumerate(FIELDS):
        ax, E, y = axs[r, c], EOF[(row, f)], yfield(row, f)
        ax.plot(100 * E['mean'], y, '-', color='k', lw=3, label='mean shape')
        for i in range(len(E['S'])):
            ax.plot(100 * E['eof'][:, i], y, '-', color=MC[i], lw=2.5,
                    label='EOF%d  %.0f%%, ceiling %.2f' % (i + 1, 100 * E['var_exp'][i], E['ceiling'][i]))
        xl = np.ceil(1.15 * 100 * max(np.abs(E['mean']).max(), np.abs(E['eof']).max()))
        shape_axes(ax, f, row, xl)
        ax.legend(loc='lower left', fontsize=9)
for ax in axs[-1]:
    ax.set_xlabel('u - Ubar [cm s$^{-1}$]   ← IN | OUT →   (EOF: per 1 std of its PC)')
axs[0, 0].set_title('Profile shape (layered)', fontweight='bold')
axs[0, 1].set_title('Face shape (lateral)', fontweight='bold')
fig.suptitle('pc_lj (Lovejoy), %d, %s: EOFs of the flood / ebb shape at the peak hour (section mean Ubar removed), one SVD per row\n'
             'PC > 0 = upper (profile) / north (faces) water more INTO the cove; ceiling = r(PC at peak - 1 h, PC at peak + 1 h)'
             % (YEAR, GTAGEX), fontsize=13)
fig.savefig(out_dir / 'eofs.png', dpi=200, transparent=True)
plt.close(fig)
print('saved %s' % (out_dir / 'eofs.png'))

# dominance.png
fig, axs = plt.subplots(len(ROWS), 2, figsize=(15, 16), layout='constrained', sharex=True)
for r, row in enumerate(ROWS):
    for c, f in enumerate(FIELDS):
        ax = axs[r, c]
        sub = RES[(RES['row'] == '%s %s' % row) & (RES['field'] == f)].reset_index(drop=True)
        for i, s in sub.iterrows():
            left = 0
            for g in GNAME:
                v = max(s['dom ' + g], 0)
                ax.barh(i, v, left=left, color=GC[g], height=0.6, label=g if (r, c, i) == (0, 0, 0) else None)
                left += v
            ax.plot(s['R2_cv'], i, 'k|', ms=22, mew=3, label='CV R$^2$ (leave one month out)' if (r, c, i) == (0, 0, 0) else None)
            ax.plot(s['ceiling'], i, 'kv', ms=10, label='ceiling' if (r, c, i) == (0, 0, 0) else None)
        ax.set_yticks(range(len(sub)), ['%s (%.0f%%)' % (s['mode'], 100 * s['var_exp']) for _, s in sub.iterrows()])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.grid(axis='x', color='lightgray', linestyle='--', alpha=0.5)
        if c == 0:
            ax.set_ylabel('%s %s' % (row[1].upper(), row[0].upper()), fontweight='bold')
for ax in axs[-1]:
    ax.set_xlabel('R$^2$ of the PC: bars = each driver\'s dominance share (sum = in-sample R$^2$)')
axs[0, 0].set_title('Profile shape EOFs', fontweight='bold')
axs[0, 1].set_title('Face shape EOFs', fontweight='bold')
fig.legend(loc='outside lower center', ncols=6, fontsize=10, frameon=False)
fig.suptitle('pc_lj (Lovejoy), %d, %s: what explains each shape EOF? OLS on all drivers together, general dominance analysis'
             % (YEAR, GTAGEX), fontsize=13)
fig.savefig(out_dir / 'dominance.png', dpi=200, transparent=True)
plt.close(fig)
print('saved %s' % (out_dir / 'dominance.png'))

# rda.png
fig, axs = plt.subplots(len(ROWS), 2, figsize=(15, 20), layout='constrained')
for r, row in enumerate(ROWS):
    for c, f in enumerate(FIELDS):
        ax, E, R, y = axs[r, c], EOF[(row, f)], RDA[(row, f)], yfield(row, f)
        ax.plot(100 * E['eof'][:, 0], y, '--', color='k', lw=2, label='EOF1 (%.0f%% of the variance)' % (100 * E['var_exp'][0]))
        ax.plot(100 * R['mode1'], y, '-', color='#CC3311', lw=3,
                label='RDA mode 1 (%.0f%% of the explained part)' % (100 * R['mode1_share']))
        xl = np.ceil(1.15 * 100 * max(np.abs(E['eof'][:, 0]).max(), np.abs(R['mode1']).max()))
        shape_axes(ax, f, row, xl)
        top = np.argsort(-np.abs(R['load']))[:4]
        ax.text(0.98, 0.02, 'drivers explain %.0f%% of the shape variance (adj. %.0f%%)\ncos(RDA1, EOF1) = %.2f\nloadings on RDA1:\n%s'
                % (100 * R['R2'], 100 * R['R2_adj'], R['cos_eof1'], '\n'.join('  %s %+.2f' % (DNAME[j], R['load'][j]) for j in top)),
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9)
        ax.legend(loc='upper left', fontsize=9)
for ax in axs[-1]:
    ax.set_xlabel('u - Ubar [cm s$^{-1}$]   ← IN | OUT →   (per 1 std)')
axs[0, 0].set_title('Profile shape (layered)', fontweight='bold')
axs[0, 1].set_title('Face shape (lateral)', fontweight='bold')
fig.suptitle('pc_lj (Lovejoy), %d, %s: RDA, the shape most explained by the drivers, against EOF1' % (YEAR, GTAGEX), fontsize=13)
fig.savefig(out_dir / 'rda.png', dpi=200, transparent=True)
plt.close(fig)
print('saved %s' % (out_dir / 'rda.png'))

# ------------------------------------------------------------------ STEP 9: candidate drivers not in STEP 5
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

# ------------------------------------------------------------------ STEP 10: linear vs boosted trees, old vs all drivers
GROUPS_ALL = dict(GROUPS)
for n in NNAME:                                           # single drivers; the density-profile PCs one group per time
    GROUPS_ALL.setdefault(n.rsplit(' ', 1)[0] if n.startswith('rhoprof') else n, []).append(n)
ALLNAME = DNAME + NNAME


def gbt():
    return HistGradientBoostingRegressor(max_depth=GBT_DEPTH, max_iter=GBT_ITER, learning_rate=GBT_LR,
                                         min_samples_leaf=GBT_LEAF, random_state=0)


def cv_pred(make, X, y, month, perm_groups=None):
    """leave-one-month-out predictions; with perm_groups {g: cols}, also the predictions with g shuffled in the
    held-out month (NPERM repeats): returns yh, {g: (NPERM, n)}"""
    rng = np.random.default_rng(0)
    yh = np.zeros_like(y)
    yp = {g: np.zeros((NPERM, len(y))) for g in (perm_groups or {})}
    for mo in np.unique(month):
        te = month == mo
        m = make().fit(X[~te], y[~te])
        yh[te] = m.predict(X[te])
        for g, cc in (perm_groups or {}).items():
            for r in range(NPERM):
                Xp = X[te].copy()
                Xp[:, cc] = Xp[rng.permutation(te.sum())][:, cc]  # one permutation for the whole group
                yp[g][r, te] = m.predict(Xp)
    return yh, yp


class OLS:
    def fit(self, X, y):
        self.b = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0]
        return self

    def predict(self, X):
        return np.column_stack([np.ones(len(X)), X]) @ self.b


R2 = lambda y, yh: 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()
NL = []
for row in ROWS:
    d = D[row]
    Zn = np.column_stack([NEW[n][d['hrs']] for n in NNAME])
    okn = np.isfinite(Zn).all(axis=1)
    Zn = (Zn - np.nanmean(Zn[d['ok'] & okn], axis=0)) / np.nanstd(Zn[d['ok'] & okn], axis=0)
    ok = d['ok'] & okn
    Xo, Xa, mo = d['Z'][ok], np.column_stack([d['Z'], Zn])[ok], d['month'][ok]
    pg = {g: [ALLNAME.index(c) for c in cc] for g, cc in GROUPS_ALL.items()}
    for f in FIELDS:
        E = EOF[(row, f)]
        for i in range(len(E['S'])):
            y = E['pc'][i][ok]
            r = {'row': '%s %s' % row, 'field': f, 'mode': 'EOF%d' % (i + 1), 'var_exp': E['var_exp'][i], 'ceiling': E['ceiling'][i], 'n': ok.sum()}
            r['lin old'] = R2(y, cv_pred(OLS, Xo, y, mo)[0])
            r['lin all'] = R2(y, cv_pred(OLS, Xa, y, mo)[0])
            r['gbt old'] = R2(y, cv_pred(gbt, Xo, y, mo)[0])
            yh, yp = cv_pred(gbt, Xa, y, mo, pg)
            r['gbt all'] = R2(y, yh)
            for g in GROUPS_ALL:
                r['perm ' + g] = r['gbt all'] - np.mean([R2(y, yp[g][k]) for k in range(NPERM)])
            NL.append(r)
            top = sorted(GROUPS_ALL, key=lambda g: -r['perm ' + g])[:4]
            print('%-14s %-7s %-4s ceil %.2f | CV R2: lin old %.2f, lin all %.2f, gbt old %.2f, gbt all %.2f | %s'
                  % (r['row'], f, r['mode'], r['ceiling'], r['lin old'], r['lin all'], r['gbt old'], r['gbt all'],
                     ', '.join('%s %.2f' % (g, r['perm ' + g]) for g in top)))
NL = pd.DataFrame(NL)
NL.to_csv(out_dir / 'nonlinear.csv', index=False, float_format='%.4f')
print('saved %s' % (out_dir / 'nonlinear.csv'))

# ------------------------------------------------------------------ STEP 11: figure
LABEL = {'|Q| peak': 'this tide |Q|', 'D2 env': 'spring / neap (D2 envelope)', 'D1 env': 'tropic / equatorial (D1 envelope)',
         'prev half |Q|': 'previous half-cycle |Q|', 'prev same |Q|': 'same-phase tide before |Q|',
         'tau_al 6h': 'along-cove wind, 6 h', 'tau_al Godin': 'along-cove wind, subtidal', 'tau_cr 6h': 'cross-cove wind, 6 h',
         'tau_cr Godin': 'cross-cove wind, subtidal', 'tau_sar al 6h': 'Saratoga along-passage wind, 6 h',
         'tau_sar al Godin': 'Saratoga along-passage wind, subtidal', 'tau_sar cr 6h': 'Saratoga cross-passage wind, 6 h',
         'tau_sar cr Godin': 'Saratoga cross-passage wind, subtidal',
         'drho peak': 'top-bottom $\\Delta\\rho$, peak', 'drho Godin': 'top-bottom $\\Delta\\rho$, subtidal',
         'drhox top Godin': 'in-out $\\Delta\\rho$, surface', 'drhox bot Godin': 'in-out $\\Delta\\rho$, bottom',
         'rhoprof peak': 'density profile shape, peak', 'rhoprof slack': 'density profile shape, slack',
         'pycno z_half peak': 'half-density depth, peak', 'pycno z_half slack': 'half-density depth, slack',
         'pycno z_maxN2 peak': 'max-N$^2$ depth, peak', 'pycno z_maxN2 slack': 'max-N$^2$ depth, slack',
         'shflux 6h': 'surface heat flux, 6 h', 'shflux Godin': 'surface heat flux, subtidal', 'Skagit 3d': 'Skagit flow, 3 d lag',
         'zeta Godin': 'subtidal sea level', 'Q Godin': 'subtidal Q at pc_lj', 'season': 'season', 'hour': 'hour of day'}
CAT = [('tide', ['|Q| peak', 'D2 env', 'D1 env', 'prev half |Q|', 'prev same |Q|']),
       ('wind', ['tau_al 6h', 'tau_al Godin', 'tau_cr 6h', 'tau_cr Godin', 'tau_sar al 6h', 'tau_sar al Godin', 'tau_sar cr 6h', 'tau_sar cr Godin']),
       ('density', ['drho peak', 'drho Godin', 'drhox top Godin', 'drhox bot Godin', 'rhoprof peak', 'rhoprof slack', 'pycno z_half peak',
                    'pycno z_half slack', 'pycno z_maxN2 peak', 'pycno z_maxN2 slack']),
       ('other', ['shflux 6h', 'shflux Godin', 'Skagit 3d', 'zeta Godin', 'Q Godin', 'season', 'hour'])]
SHOW = 0.03                                               # a driver is shown if it reaches this importance in some EOF
COLS = [(row, f, m) for row in ROWS for f, m in [('profile', 'EOF1'), ('profile', 'EOF2'), ('faces', 'EOF1'), ('faces', 'EOF2'), ('faces', 'EOF3')]]
SUB = [NL[(NL['row'] == '%s %s' % r) & (NL['field'] == f) & (NL['mode'] == m)].iloc[0] for r, f, m in COLS]
Mfull = np.array([[s['perm ' + g] if 'perm ' + g in s else np.nan for s in SUB] for _, gs in CAT for g in gs])
keep, ylab, cat_of = [], [], []
i = 0
for c, gs in CAT:
    for g in gs:
        if np.nanmax(Mfull[i]) >= SHOW:
            keep.append(i)
            ylab.append(LABEL[g])
            cat_of.append(c)
        i += 1
M_ = np.clip(Mfull[keep], 0, None)
fig, ax = plt.subplots(figsize=(15, 0.42 * len(keep) + 3.2), layout='constrained')
cs = ax.pcolormesh(M_, cmap='Blues', vmin=0, vmax=0.25)
best = np.nanargmax(np.where(np.isfinite(M_), M_, -1), axis=0)
for j in range(M_.shape[1]):
    for i in range(M_.shape[0]):
        if M_[i, j] >= SHOW:
            ax.text(j + 0.5, i + 0.5, '%.2f' % M_[i, j], ha='center', va='center', fontsize=8,
                    color='w' if M_[i, j] > 0.15 else 'k', fontweight='bold' if i == best[j] else 'normal')
    ax.add_patch(plt.Rectangle((j, best[j]), 1, 1, fill=False, ec='k', lw=1.8))
for j in range(5, M_.shape[1], 5):
    ax.axvline(j, color='k', lw=2)
for j in range(2, M_.shape[1], 5):
    ax.axvline(j, color='0.5', lw=0.8, ls='--')
for i in range(1, len(cat_of)):
    if cat_of[i] != cat_of[i - 1]:
        ax.axhline(i, color='k', lw=1.2)
ax.set_xticks(np.arange(M_.shape[1]) + 0.5,
              ['%s %s\n%.0f%%\nR$^2$ %.2f' % ('P' if f == 'profile' else 'F', m[-1], 100 * s['var_exp'], s['gbt all'])
               for (r, f, m), s in zip(COLS, SUB)], fontsize=8)
ax.set_yticks(np.arange(len(keep)) + 0.5, ylab, fontsize=9)
ax.invert_yaxis()
axt = ax.secondary_xaxis('top')
axt.set_xticks([2.5 + 5 * k for k in range(len(ROWS))], ['%s %s' % (r[1], r[0]) for r in ROWS], fontsize=11, fontweight='bold')
axt.tick_params(length=0)
for i, c in enumerate(cat_of):
    if i == 0 or c != cat_of[i - 1]:
        ax.text(-0.08, i, c.upper(), transform=ax.get_yaxis_transform(), ha='right', va='top', fontsize=9, fontweight='bold', color='0.3')
fig.colorbar(cs, ax=ax, label='permutation importance (drop in CV R$^2$)', shrink=0.5, extend='max')
fig.suptitle('pc_lj (Lovejoy), %d, %s: what sets the shape of each flood / ebb row (P = profile EOF, F = face EOF; %% = share of the '
             'shape variance, R$^2$ = CV R$^2$ of the boosted trees)\nbox = top driver; only drivers reaching %.2f somewhere are shown'
             % (YEAR, GTAGEX, SHOW), fontsize=11)
fig.savefig(out_dir / 'nonlinear.png', dpi=200, transparent=True)
plt.close(fig)
print('saved %s' % (out_dir / 'nonlinear.png'))

# ------------------------------------------------------------------ STEP 12: 3-hour average around the peak
A3 = []
for row in ROWS:
    d = D[row]
    x3 = d['shape_profile'].mean(axis=0)                              # (n, b) mean of peak - 1, peak, peak + 1
    ms = x3.mean(axis=0)
    zc = np.flatnonzero(np.sign(ms) != np.sign(ms[0]))
    upper = np.arange(len(ms)) < (zc[0] if len(zc) else len(ms) // 2)
    E3, E1 = eof(x3, d['w_profile'], upper), EOF[(row, 'profile')]
    Zn = np.column_stack([NEW[n][d['hrs']] for n in NNAME])
    okn = np.isfinite(Zn).all(axis=1)
    Zn = (Zn - np.nanmean(Zn[d['ok'] & okn], axis=0)) / np.nanstd(Zn[d['ok'] & okn], axis=0)
    ok = d['ok'] & okn
    Xa, mo = np.column_stack([d['Z'], Zn])[ok], d['month'][ok]
    for i in range(2):
        y3 = E3['pc'][i][ok]
        w = d['w_profile']
        cosang = (E3['eof'][:, i] * E1['eof'][:, i] * w).sum() / np.sqrt((E3['eof'][:, i] ** 2 * w).sum() * (E1['eof'][:, i] ** 2 * w).sum())
        s1 = NL[(NL['row'] == '%s %s' % row) & (NL['field'] == 'profile') & (NL['mode'] == 'EOF%d' % (i + 1))].iloc[0]
        c = s1['ceiling']
        r = {'row': '%s %s' % row, 'mode': 'EOF%d' % (i + 1), 'var_exp_1h': s1['var_exp'], 'var_exp_3h': E3['var_exp'][i],
             'cos_eof': abs(cosang), 'ceiling': c, 'r_pc_1h_3h': abs(np.corrcoef(E3['pc'][i], E1['pc'][i])[0, 1]),
             'lin all 1h': s1['lin all'], 'gbt all 1h': s1['gbt all'],
             'lin all 3h': R2(y3, cv_pred(OLS, Xa, y3, mo)[0]), 'gbt all 3h': R2(y3, cv_pred(gbt, Xa, y3, mo)[0])}
        r['gbt expected if noise'] = min(s1['gbt all'] / (c + (1 - c) / 3), 1)
        A3.append(r)
        print('%-14s profile %s: var %.0f%% -> %.0f%%, cos(EOF 3h, 1h) %.2f, r(PC) %.2f | CV R2 gbt all 1h %.2f -> 3h %.2f '
              '(white-noise expectation %.2f, ceiling %.2f); lin all %.2f -> %.2f'
              % (r['row'], r['mode'], 100 * r['var_exp_1h'], 100 * r['var_exp_3h'], r['cos_eof'], r['r_pc_1h_3h'],
                 r['gbt all 1h'], r['gbt all 3h'], r['gbt expected if noise'], c, r['lin all 1h'], r['lin all 3h']))
A3 = pd.DataFrame(A3)
A3.to_csv(out_dir / 'avg3.csv', index=False, float_format='%.4f')
print('saved %s' % (out_dir / 'avg3.csv'))

# ------------------------------------------------------------------ STEP 13: figure
fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), layout='constrained', sharey=True)
xr_ = np.arange(len(ROWS))
for c, mode in enumerate(['EOF1', 'EOF2']):
    ax = axs[c]
    a = A3[A3['mode'] == mode].reset_index(drop=True)
    ax.bar(xr_ - 0.2, a['gbt all 1h'], 0.38, color='0.6', label='peak hour (gbt all)')
    ax.bar(xr_ + 0.2, a['gbt all 3h'], 0.38, color='#4477AA', label='peak $\\pm$1 h mean (gbt all)')
    ax.plot(xr_ + 0.2, a['gbt expected if noise'], 'k_', ms=30, mew=2.5, label='3-h expectation if the gap were hourly noise')
    ax.plot(xr_ - 0.2, a['ceiling'], 'kv', ms=10, label='ceiling (peak hour)')
    ax.set_xticks(xr_, [r_ for r_ in a['row']])
    ax.set_ylim(0, 1)
    ax.grid(axis='y', color='lightgray', linestyle='--', alpha=0.5)
    ax.set_title('Profile %s' % mode, fontweight='bold')
axs[0].set_ylabel('CV R$^2$ (leave one month out)')
axs[0].legend(loc='upper left', fontsize=9)
fig.suptitle('pc_lj (Lovejoy), %d, %s: does averaging out hour-to-hour noise raise the predictable share?' % (YEAR, GTAGEX), fontsize=12)
fig.savefig(out_dir / 'avg3.png', dpi=200, transparent=True)
plt.close(fig)
print('saved %s' % (out_dir / 'avg3.png'))

# ------------------------------------------------------------------ STEP 14: reduced linear model
LIN = [('this tide |Q|', ['|Q| peak']), ('tropic / equatorial (D1)', ['D1 env']), ('previous half-cycle |Q|', ['prev half |Q|']),
       ('along-cove wind, 6 h', ['tau_al 6h']), ('along-cove wind, subtidal', ['tau_al Godin']),
       ('cross-cove wind, subtidal', ['tau_cr Godin']),
       ('in-out $\\Delta\\rho$, surface', ['drhox top Godin']), ('in-out $\\Delta\\rho$, bottom', ['drhox bot Godin']),
       ('top-bottom $\\Delta\\rho$, slack', ['drho slack']), ('half-density depth, slack', ['pycno z_half slack']),
       ('top-bottom $\\Delta\\rho$, slack to peak change', ['drho change']),
       ('half-density depth, slack to peak change', ['z_half change']),
       ('hour of day', ['hour sin', 'hour cos']), ('season', ['season sin', 'season cos'])]
drho_slack = np.full(len(ot), np.nan)                    # top-bottom drho at the slack that starts the half-cycle
for k in pk_all:
    drho_slack[k] = drho_h[start_of[k]]
SRC = dict(DRV, **NEW, **{'drho slack': drho_slack, 'drho change': drho_h - drho_slack,      # change over the half-cycle
                            'z_half change': NEW['pycno z_half peak'] - NEW['pycno z_half slack']})
LCOL = [c for _, cc in LIN for c in cc]
LG = [[LCOL.index(c) for c in cc] for _, cc in LIN]
NBOOT = 1000                                             # month-block bootstrap draws for the coefficient CIs


def dominance_g(y, Z, gcols):
    """general dominance of each column group of Z (sums to the full R^2), and the full R^2"""
    ng = len(gcols)
    cl = lambda m: [c for g in range(ng) if m >> g & 1 for c in gcols[g]]
    R = np.array([r2(y, Z[:, cl(m)]) for m in range(1 << ng)])
    dom = np.zeros(ng)
    for g in range(ng):
        by = [[] for _ in range(ng)]
        for m in range(1 << ng):
            if not m >> g & 1:
                by[bin(m).count('1')].append(R[m | 1 << g] - R[m])
        dom[g] = np.mean([np.mean(v) for v in by])
    return dom, R[-1]


LR = []
rng = np.random.default_rng(0)
for row in ROWS:
    d = D[row]
    Z = np.column_stack([SRC[c][d['hrs']] for c in LCOL])
    ok = np.isfinite(Z).all(axis=1)
    Z = (Z[ok] - Z[ok].mean(axis=0)) / Z[ok].std(axis=0)
    mo = d['month'][ok]
    X1 = np.column_stack([np.ones(ok.sum()), Z])
    vif = [1 / max(1 - r2(Z[:, j], np.delete(Z, j, axis=1)), 1e-9) for j in range(Z.shape[1])]
    print('%-7s %-5s reduced model: %d tides, %d columns in %d groups; max VIF %.1f (%s)'
          % (row[1], row[0], ok.sum(), Z.shape[1], len(LIN), max(vif), LCOL[int(np.argmax(vif))]))
    for f in FIELDS:
        E = EOF[(row, f)]
        for i in range(len(E['S'])):
            y = E['pc'][i][ok]
            y = (y - y.mean()) / y.std()                     # standardised: beta = SD of the PC per SD of the driver
            b = np.linalg.lstsq(X1, y, rcond=None)[0][1:]
            months = np.unique(mo)
            bb = np.zeros((NBOOT, len(b)))
            for n in range(NBOOT):                           # resample whole months
                sel = np.concatenate([np.flatnonzero(mo == m) for m in rng.choice(months, len(months))])
                bb[n] = np.linalg.lstsq(X1[sel], y[sel], rcond=None)[0][1:]
            lo, hi = np.percentile(bb, [2.5, 97.5], axis=0)
            dom, R2f = dominance_g(y, Z, LG)
            r_ = {'row': '%s %s' % row, 'field': f, 'mode': 'EOF%d' % (i + 1), 'var_exp': E['var_exp'][i], 'n': ok.sum(),
                  'R2': R2f, 'R2_cv': r2_cv(y, Z, mo), 'R2_cv_trees': R2(y, cv_pred(gbt, Z, y, mo)[0])}
            for j, c in enumerate(LCOL):
                r_['beta ' + c], r_['lo ' + c], r_['hi ' + c] = b[j], lo[j], hi[j]
            for g, (gn, _) in enumerate(LIN):
                r_['dom ' + gn] = dom[g]
            LR.append(r_)
            top = np.argsort(-dom)[:3]
            print('  %-7s %s %3.0f%% | R2 %.2f, CV %.2f (trees %.2f) | %s'
                  % (f, r_['mode'], 100 * r_['var_exp'], R2f, r_['R2_cv'], r_['R2_cv_trees'],
                     ', '.join('%s %.2f (beta %s)' % (LIN[g][0], dom[g], ', '.join('%+.2f%s' % (b[j], '*' if lo[j] * hi[j] > 0 else '')
                                                                                   for j in LG[g])) for g in top)))
LR = pd.DataFrame(LR)
LR.to_csv(out_dir / 'linear.csv', index=False, float_format='%.4f')
print('saved %s' % (out_dir / 'linear.csv'))

# ------------------------------------------------------------------ STEP 15: coefficient plot of the reduced linear model
PANELS = [('profile', 'EOF1', 'Profile mode 1', '+ = surface more IN, deep more OUT'),
          ('profile', 'EOF2', 'Profile mode 2', '+ = mid-depth (~8 m) more IN, surface and bottom more OUT'),
          ('faces', 'EOF1', 'Face mode 1', '+ = north side more IN, centre / south more OUT')]
ABB = {('flood', 'greater'): 'GF', ('flood', 'lesser'): 'LF', ('ebb', 'greater'): 'GE', ('ebb', 'lesser'): 'LE'}
RCOL = {('flood', 'greater'): '#08519C', ('flood', 'lesser'): '#6BAED6', ('ebb', 'greater'): '#A50F15', ('ebb', 'lesser'): '#FB6A4A'}
PLOT = [(gn, cc[0]) for gn, cc in LIN if len(cc) == 1]   # hour and season (sin + cos) are controls, not plotted
NAME = {'this tide |Q|': 'size of this tide (|Q|)', 'tropic / equatorial (D1)': 'diurnal tide (tropic +)',
        'previous half-cycle |Q|': 'size of the previous half-cycle', 'along-cove wind, 6 h': 'up-cove wind, last 6 h',
        'along-cove wind, subtidal': 'up-cove wind, multi-day', 'cross-cove wind, subtidal': 'cross-cove wind, multi-day',
        'in-out $\\Delta\\rho$, surface': 'surface density, outside - inside', 'in-out $\\Delta\\rho$, bottom': 'bottom density, outside - inside',
        'top-bottom $\\Delta\\rho$, slack': 'stratification at slack', 'half-density depth, slack': 'pycnocline depth at slack',
        'top-bottom $\\Delta\\rho$, slack to peak change': 'stratification change, slack to peak',
        'half-density depth, slack to peak change': 'pycnocline deepening, slack to peak'}
fig, axs = plt.subplots(1, len(PANELS), figsize=(18, 9), layout='constrained', sharey=True)
yy = np.arange(len(PLOT))
off = np.linspace(-0.27, 0.27, len(ROWS))
for ax, (f, m, ttl, sgn) in zip(axs, PANELS):
    r2s = []
    for o, row in zip(off, ROWS):
        s_ = LR[(LR['row'] == '%s %s' % row) & (LR['field'] == f) & (LR['mode'] == m)].iloc[0]
        b = np.array([s_['beta ' + c] for _, c in PLOT])
        lo, hi = np.array([s_['lo ' + c] for _, c in PLOT]), np.array([s_['hi ' + c] for _, c in PLOT])
        sig = lo * hi > 0
        ax.errorbar(b, yy + o, xerr=[b - lo, hi - b], fmt='none', ecolor=RCOL[row], elinewidth=1.3, alpha=0.8)
        ax.scatter(b[sig], (yy + o)[sig], s=45, color=RCOL[row], zorder=3, label='%s %s' % (row[1], row[0]))
        r2s.append('%s %.2f' % (ABB[row], s_['R2_cv']))
        ax.scatter(b[~sig], (yy + o)[~sig], s=45, facecolors='w', edgecolors=RCOL[row], zorder=3)
    ax.axvline(0, color='k', lw=1)
    for k in [2.5, 5.5]:
        ax.axhline(k, color='0.6', lw=0.8)               # tide | wind | density
    ax.grid(axis='x', color='lightgray', linestyle='--', alpha=0.6)
    ax.set_xlim(-1, 1)
    ve = LR[(LR['field'] == f) & (LR['mode'] == m)]['var_exp']
    ax.set_title('%s (%.0f-%.0f%% of the shape variance)\n%s\nR$^2$ (cross-validated): %s' % (ttl, 100 * ve.min(), 100 * ve.max(), sgn, ', '.join(r2s)),
                 fontsize=10.5)
    ax.set_xlabel('standardised coefficient\n(change in the mode, in SDs, per 1 SD of the driver)')
axs[0].set_yticks(yy, [NAME.get(gn, gn) for gn, _ in PLOT], fontsize=10)
axs[0].invert_yaxis()
for k, lab in [(1.0, 'TIDE'), (4.0, 'WIND'), (8.5, 'DENSITY')]:
    axs[0].text(-0.62, k, lab, transform=axs[0].get_yaxis_transform(), rotation=90, ha='center', va='center', fontsize=10,
                fontweight='bold', color='0.35')
fig.legend(*axs[0].get_legend_handles_labels(), loc='outside lower center', ncols=4, fontsize=10, frameon=False)
fig.suptitle('pc_lj (Lovejoy), %d, %s: what changes the shape of the flood / ebb velocity profile? Linear regression on all drivers '
             'together (also controlling for hour of day and season).\nDots = standardised coefficient, bars = 95%% confidence '
             'interval (bootstrap over months); filled = significant, open = not.' % (YEAR, GTAGEX), fontsize=11)
fig.savefig(out_dir / 'linear.png', dpi=200, transparent=True)
plt.close(fig)
print('saved %s' % (out_dir / 'linear.png'))
