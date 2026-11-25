"""
Penn Cove: average flood, average ebb, and the Godin-filtered residual, at the daily LESSER
flood and lesser ebb hours of 2024. A copy of 20261116_pc_peak_flood_ebb.py (the daily greater
half-cycle); only STEP 2, the labels and the output folder differ.

STEP 1  load pc_lj (Lovejoy) from the pc_cove box (hourly ocean_his snapshots, hh:00) for 2024,
        straight from the extraction: u normal to the faces (+ = into the cove), z_w at the faces
        (mean of the two rho cells either side), face width dy = 1/pn, and salt / temp at the
        top and bottom s-levels (same two-cell mean, for STEP 9). The run continues into
        2025, so PAD days past the year end are kept for the Godin filter (71 h, so 35 h each
        side); the first 35 h of 2024 have no left side and will be NaN in the residual.
STEP 2  daily peak flood and peak ebb hours, from the pc_lj section-total Q (the model's own
        snapshot transport, + = flood). Lunar days, as in 20261112_pc_peak_scatter.py:
        half-cycles = runs of one sign of Q, runs shorter than MIN_RUN h (slack jitter)
        absorbed into the run before, first and last (partial) runs dropped; each
        half-cycle's peak = its max |Q| hour. Lunar days: a flood start to the second next flood
        start, i.e. 4 consecutive half-cycles (flood, ebb, flood, ebb; the runs alternate in sign),
        tiled in steps of 4 from the baseline day's first flood (20261113_pc_candidate_days/picks.csv,
        2024-08-31 05:00), so every day has 2 floods and 2 ebbs (as 20261117_pc_every_flood_ebb.py).
        In each day the flood (ebb) with the larger |Q| at its peak is the greater, the other the
        lesser; this script keeps the LESSER. Calendar days are not used: a peak near 00 UTC falls across the boundary.
        (Before 2026-11-17 the greater were picked largest |Q| first with an 18 h exclusion window.)
STEP 3  averages on the native section (s-level k x face p), dz-weighted because the s-levels
        stretch with the tide: U[k, p] = sum_t u dz / sum_t dz.
            flood     over the daily lesser peak flood hours
            ebb       over the daily lesser peak ebb hours
            residual  Godin(u dz) / Godin(dz) per element (the transport-weighted residual, as
                      u_res = Godin(q) / Godin(dA) in 20261110_pc_metric_ts.py), then averaged the
                      same way over the peak hours: res_flood at the flood hours, res_ebb at the
                      ebb hours (both are the same slow field, sampled at different hours).
                      Peak hours in the first 35 h of 2024 have no residual and are left out.
            residual  the plotted one: over all the peak hours, flood and ebb together.
STEP 4  figure: flood | ebb | residual sections, looking OUT of the cove (east), so north is at the
        LEFT; x = distance across from the north end of the section, each face drawn as its
        own column of native cells between its time-mean z_w over the same hours.
        Blue = into the cove, red = out. Flood and ebb share one colour scale; the residual
        (about 1/5 the size) has its own. Black line = u = 0, contoured on the cell centres
        (face centre, mid-height of each s-level cell).
        Symbols for flow through the section: looking out of the cove, inflow comes toward the
        viewer = \u2299 and outflow goes away = \u2297. One per region bounded by u = 0 (connected
        cells of one sign, 4-connected on the s-level x face grid), at the region's |q|-weighted
        centre, labelled with its transport Q = sum u dz dy [m3/s]. Symbol AREA proportional to
        |Q|, one scale for all three panels (DSYM pt diameter at the largest |Q|, at least DMIN pt).
STEP 5  spring / neap and tropic / equatorial class of every peak hour, from the 20261112_pc_utide fits (fits.p: utide
        ellipses of the depth-mean velocity at every cove rho cell, OLS, nodal at the 2024-25
        record centre, 11 tides + S1).
        Tidal transport through pc_lj in each band, D2 = M2 + S2 + N2 + K2 (semidiurnal) and
        D1 = K1 + O1 + P1 + Q1 (diurnal; S1 left out, it is mostly the daily wind), reconstructed at the
        16 rho cells either side of the 8 faces, east component, averaged to the faces,
        Q_band = sum_p u_band h dy (+ into the cove; h = mean of the two cells). Transport, not sea
        level, because the sea-level envelope wanders +-2-3 d off spring-neap
        ([[springneap-define-on-qprism]]). Envelope = |hilbert(Q_band)|, reconstructed with
        EPAD days either side of 2024 so the transform has no edge inside the year.
        fits.p keeps the amplitudes and phases but not utide's coef object, so one cell is
        refit (utide.solve, same options) as a template: its fit must reproduce fits.p, then
        each cell's fits.p values go into a copy of it for utide.reconstruct (its mean flow
        set to 0: reconstruct adds it back, and a mean under the tide puts an M2 ripple in the envelope).
        spring / neap = top / bottom third of hours by the D2 envelope, tropic / equatorial =
        top / bottom third by the D1 envelope, the terciles taken
        within each calendar month of 2024 (as pc_tidal_cycles.py: the envelope has a
        semiannual modulation, S2-K2 in D2 and K1-P1 in D1, so a year-wide threshold would compare seasons).
        A peak hour takes the class of its hour; the middle third is in neither.
STEP 6  figures: the STEP 4 sections in 2 rows, spring | neap and tropic | equatorial, each averaged over its own class's
        peak hours (residual: that class's flood and ebb peak hours together). One set of colour
        scales and one symbol scale for both rows.
STEP 7  wind class of every peak hour, from the surface stress ROMS applied (sustr, svstr, ocean_his),
        straight from the pc_cove_mix box: averaged over the cove's water faces (u faces west of
        the mouth face, box xi_u < 30; v faces in rho columns <= 30), onto the cove axis
        (ax, ay) = (-0.9789, -0.2043), mouth -> head:
            tau_al = sustr ax + svstr ay   + = toward the head (up-cove)
            tau_cr = sustr ay - svstr ax   + = toward the north shore
        hourly, THEN Godin-filtered (stress is quadratic in the wind). Classes as
        pc_forcing_states.py, by sign and a threshold TTH = 0.01 Pa (~3 m/s steady wind), not
        terciles: up-cove = tau_al > TTH, down-cove = tau_al < -TTH, calm = |tau_al| <= TTH AND
        |tau_cr| <= TTH. NOT balanced by month: the wind classes are seasonal.
STEP 8  figure: the sections in 3 rows, up-cove | down-cove | calm, as STEP 6.
STEP 9  stratification class of every peak hour, at pc_lj itself: per face sigma0 (TEOS-10, gsw)
        of the bottom and top s-levels, drho = sigma0_bot - sigma0_top, width-weighted across the
        faces (dy), hourly snapshots, then Godin-filtered so tidal straining does not flip the
        class within a tide. high = top third of the 2024 hours, low = bottom third, taken
        YEAR-WIDE as pc_forcing_states.py (not within month: stratification is the seasonal
        signal itself, so per-month terciles would remove what is being compared).
STEP 10 figure: the sections in 2 rows, high strat | low strat, as STEP 6.
STEP 11 alternate STEP 10 with calm wind only: peak hours that are high (low) strat (STEP 9) AND
        calm (STEP 7: |tau_al| and |tau_cr| <= TTH), so the strat contrast is not wind.
STEP 11b surface temperature and salinity classes, at pc_lj itself: the top s-level temp and
        salt of each face (STEP 1), width-weighted across the faces (dy), hourly, Godin-filtered.
        warm / cold surface = top / bottom third of the 2024 hours of temp; salty / fresh surface
        = top / bottom third of salt. YEAR-WIDE terciles, as strat (the user's choice): warm /
        cold is close to summer / winter, so the split carries the season. Section figures as
        STEP 6 (pc_lj_lesser_sections_temp.png, _salt.png), and both sets go into STEPS 12, 13, 15.
STEP 12 profile figures, one per set (all, springneap, tropeq, wind, strat, strat_calm, temp, salt), columns
        flood | ebb | residual, one line per class of the set:
        ROW 1  cross-channel averaged profile, U_b = sum_t Q_b / sum_t A_b in EDGES (1 m) bins of
               depth below the moving surface (as 20261113_pc_baseline_profiles.py: each cell's
               transport split among the bins it overlaps, sum_b Q_b = Q exactly). A bin is drawn
               only where it is wetted by at least half a face in every hour averaged.
        ROW 2  depth-averaged u at each face, sum_t sum_k u dz / sum_t sum_k dz, against distance
               across from the north end, north at the TOP (map orientation).
        residual: the same with Godin(Q_b) / Godin(A_b) and Godin(sum_k u dz) / Godin(sum_k dz).
        + = into the cove; u on x in both rows, reversed so + (into the cove, west) is drawn to the
        LEFT, as on a map. One symmetric u scale for every panel of a figure.
STEP 13 layer figures, one per set, rows surface 1/3 | bottom 1/3 | depth average, columns
        flood | ebb | residual, one line per class, u at each face against distance across
        (north at the TOP, + into the cove LEFT, as STEP 12 row 2). The thirds are of each face's
        own water depth H = sum_k dz, every hour (they move with the tide): surface = 0 to H/3
        below the surface, bottom = 2H/3 to H. Each s-level cell counts by its overlap with the
        third, ov_k: U = sum_t sum_k u ov_k / sum_t sum_k ov_k (residual: Godin of both sums, as
        STEP 3). Depth average = STEP 12 row 2.
STEP 14 the same three layers at EVERY cove cell (box rho columns <= 30), for map views. Read
        the box in blocks of BLK hours (it is chunked one hour x whole box). Velocity at rho
        points = mean of the two faces, wall (land) faces counted as 0 (20261110_pc_cove_layer_maps.py).
        Per cell and hour: Ux = sum_k u ov_k, Vy = sum_k v ov_k, Hl = sum_k ov_k for the
        surface third, the bottom third, and the whole column. Read from the 20261116 cache
        (20261116_pc_peak_flood_ebb/cove_layers_hourly.npz, same hours, independent of the peaks),
        else computed and cached there.
STEP 15 cove maps, one figure per class of every set: rows surface 1/3 | bottom 1/3 | depth
        average, columns flood | ebb | residual at the pc_lj daily lesser peak hours of the class.
        U = sum_t Ux / sum_t Hl (residual: Godin(Ux) / Godin(Hl) at the same hours, as STEP 3).
        Colour = along-cove component, axis (-0.9789, -0.2043) mouth -> head, blue = into the
        cove (as the sections); arrows = the full vector. Flood and ebb share one colour / arrow
        scale and the residual has its own, the same for every class of a set so they compare.
        Sections pc_cp, pc_lj (thick), pc_lp drawn.

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc
       LO_output/DM_outs/20261112_pc_utide/fits.p
       LO_output/DM_outs/20261113_pc_candidate_days/picks.csv
       LO_output/extract/[gtagex]/box/pc_cove_mix_[ds0]_[ds1].nc
OUTPUT LO_output/DM_outs/20261117_pc_lesser_flood_ebb/
       pc_lj_lesser_sections.png             STEP 4, all lesser peak hours
       pc_lj_lesser_sections_springneap.png  STEP 6, spring | neap rows
       pc_lj_lesser_sections_tropeq.png      STEP 6, tropic | equatorial rows
       pc_lj_lesser_sections_wind.png        STEP 8, up-cove | down-cove | calm rows
       pc_lj_lesser_sections_strat.png       STEP 10, high | low strat rows
       pc_lj_lesser_sections_strat_calm.png  STEP 11, high | low strat rows, calm wind only
       pc_lj_lesser_sections_temp.png        STEP 11b, warm | cold surface rows
       pc_lj_lesser_sections_salt.png        STEP 11b, salty | fresh surface rows
       pc_lj_lesser_profiles_[set].png       STEP 12 (8 sets), profile | depth-average rows, one line per class
       pc_lj_lesser_layers_[set].png         STEP 13, surface 1/3 | bottom 1/3 | depth-average rows
       cove_lesser_maps_[set]_[class].png    STEP 15, cove maps

run 20261117_pc_lesser_flood_ebb.py
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
from scipy import ndimage
from scipy.signal import hilbert

from lo_tools import Lfun, zfun

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'     # the box file
YEAR = 2024
PAD = pd.Timedelta(days=2)                # past the year end, for the Godin filter
SIGN = -1                                 # section-positive (east) is out of the cove
XI_U, ETA = 23, np.arange(14, 6, -1)      # pc_lj u faces in the box, north to south (global i 60, j 223..216)
MIN_RUN = 3                               # half-cycles shorter than this [h] are slack jitter
BAND = {'D2': ['M2', 'S2', 'N2', 'K2'],     # semidiurnal band
        'D1': ['K1', 'O1', 'P1', 'Q1']}     # diurnal band, S1 left out (mostly the daily wind)
CYCLES = {'springneap': ('D2', ('spring', 'neap')),         # band, (top third, bottom third)
          'tropeq': ('D1', ('tropic', 'equatorial'))}
EPAD = pd.Timedelta(days=30)              # reconstruction padding either side of the year, for hilbert
AXIS = (-0.9789, -0.2043)                 # cove axis, mouth -> head (pc_forcing_states.py)
TTH = 0.01                                # wind stress threshold [Pa]
EDGES = np.arange(0, 31, 1.0)             # profile bins, depth below the surface [m]
COL = {'all': 'k', 'spring': '#CC3311', 'neap': '#33BBEE', 'tropic': '#EE7733', 'equatorial': '#009988',
       'up-cove': '#EE3377', 'down-cove': '#0077BB', 'calm': 'k',
       'high strat': '#AA3377', 'low strat': '#66CCEE', 'high strat, calm': '#AA3377', 'low strat, calm': '#66CCEE',
       'warm surface': '#CC3311', 'cold surface': '#0077BB', 'salty surface': '#997700', 'fresh surface': '#33BBEE'}
BLK = 200                                 # box hours read per block (STEP 14)
SECT_XI_U = {'pc_cp': 15, 'pc_lj': 23, 'pc_lp': 30}
DSYM = 40                                 # symbol diameter at the largest region transport [pt]
DMIN = 8                                  # smallest symbol diameter, so the circle stays readable [pt]

gridname, tag, ex_name = GTAGEX.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
box_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_%s_%s.nc' % (DS0, DS1))
mix_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_mix_%s_%s.nc' % (DS0, DS1))
fit_fn = Ldir['LOo'] / 'DM_outs' / '20261112_pc_utide' / 'fits.p'
pick_fn = Ldir['LOo'] / 'DM_outs' / '20261113_pc_candidate_days' / 'picks.csv'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261117_pc_lesser_flood_ebb'
cache_dir = Ldir['LOo'] / 'DM_outs' / '20261116_pc_peak_flood_ebb'    # STEP 14 cache, shared with the greater script
Lfun.make_dir(out_dir)

# ------------------------------------------------------------------ STEP 1: load pc_lj from the box
t0 = pd.Timestamp('%d-01-01' % YEAR)
t1 = pd.Timestamp('%d-01-01' % (YEAR + 1)) + PAD
ds = xr.open_dataset(box_fn).sel(ocean_time=slice(t0, t1))
ot = pd.DatetimeIndex(ds.ocean_time.values)
u = SIGN * ds['u'].isel(xi_u=XI_U, eta_u=ETA).values.astype(float)                   # (t, k, p) + into the cove
zw = ds['z_w'].isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).values.mean(axis=-1)       # (t, w, p) at the face
dy = (1 / ds['pn'].isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).values).mean(axis=-1)  # (p,)
lat = ds['lat_u'].isel(eta_u=ETA, xi_u=XI_U).values                                  # (p,) north to south
ts = {vn: ds[vn].isel(s_rho=[0, -1], eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).values.mean(axis=-1)
      for vn in ['salt', 'temp']}                                                    # (t, bot/top, p)
lon_f, lat_f = ds['lon_u'].isel(eta_u=ETA, xi_u=XI_U).values, lat
h_f = ds['h'].isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).values.mean(axis=-1)      # (p,) depth at the face
ds.close()
dz = np.diff(zw, axis=1)                                                             # (t, k, p)
Q = (u * dz * dy).sum(axis=(1, 2))                                                   # (t,) + = flood

print('%s  %d hours  %s to %s' % (box_fn.name, len(ot), ot[0], ot[-1]))
print('pc_lj: %d faces, %d s-levels, width %.0f m; Q %+.0f to %+.0f m3/s, mean %+.1f'
      % (len(dy), u.shape[1], dy.sum(), Q.min(), Q.max(), Q.mean()))

# ------------------------------------------------------------------ STEP 2: daily (lunar-day) lesser peak flood / ebb hours
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

yr1 = pd.Timestamp('%d-01-01' % (YEAR + 1))
picks = pd.read_csv(pick_fn, parse_dates=['t0', 't1']).set_index('case')
hc_t0 = ot[[a for a, b in HC]]                           # start hour of each half-cycle
ia = np.flatnonzero((hc_t0 == picks.loc['baseline', 't0']) & (hc_sign == 1))
assert len(ia) == 1, 'baseline t0 is not a flood start of this Q'
hc_day = (np.arange(len(HC)) - ia[0]) // 4               # lunar day of each half-cycle: F E F E from the baseline's first flood

PK = {}
for phase, sgn in [('flood', 1), ('ebb', -1)]:
    m = np.flatnonzero(hc_sign == sgn)
    gr = np.zeros(len(m), dtype=bool)
    for dd in np.unique(hc_day[m]):                      # the larger |Q| of each lunar day is the greater
        i = np.flatnonzero(hc_day[m] == dd)
        gr[i[np.argmax(np.abs(Q[hc_pk[m[i]]]))]] = True
    k = hc_pk[m[~gr]]                                    # the lesser half-cycle of each lunar day
    PK[phase] = k[(ot[k] >= t0) & (ot[k] < yr1)]         # peaks in YEAR only
    dt = np.diff(ot[PK[phase]]) / pd.Timedelta(hours=1)
    print('%-5s %d half-cycles, %d lesser peaks in %d; spacing median %.1f h, 5-95%% %.1f-%.1f h; Q at peak median %+.0f m3/s'
          % (phase, len(m), len(PK[phase]), YEAR, np.median(dt), *np.percentile(dt, [5, 95]), np.median(Q[PK[phase]])))

# ------------------------------------------------------------------ STEP 3: flood / ebb / residual averages
udz_res = zfun.lowpass(u * dz, f='godin')                # (t, k, p), NaN in the first and last 35 h
dz_res = zfun.lowpass(dz, f='godin')
ok = np.isfinite(dz_res).all(axis=(1, 2))                # hours that have a residual

def dzmean(udz, dz, it):
    """sum_t u dz / sum_t dz over the hours it, per element (k, p)"""
    return udz[it].sum(axis=0) / dz[it].sum(axis=0)


def averages(pk, pr):
    """flood / ebb / residual averages over the peak hours pk = {'flood': it, 'ebb': it} (pr = those with a
    residual). Returns U (k, p) [m/s], DZ (k, p) mean cell height [m], HRS hours used, per class."""
    HRS = {'flood': pk['flood'], 'ebb': pk['ebb'], 'residual': np.r_[pr['flood'], pr['ebb']]}
    U = {'flood': dzmean(u * dz, dz, HRS['flood']), 'ebb': dzmean(u * dz, dz, HRS['ebb']),
         'residual': dzmean(udz_res, dz_res, HRS['residual'])}
    DZ = {'flood': dz[HRS['flood']].mean(axis=0), 'ebb': dz[HRS['ebb']].mean(axis=0),
          'residual': dz_res[HRS['residual']].mean(axis=0)}
    for C in U:
        assert np.isfinite(U[C]).all(), C
        Qc = (U[C] * DZ[C] * dy).sum()
        print('  %-9s n %3d  section mean %+5.1f cm/s; elements %+5.1f to %+5.1f cm/s; inflow %+5.0f, outflow %+5.0f m3/s'
              % (C, len(HRS[C]), 100 * Qc / (DZ[C] * dy).sum(), 100 * U[C].min(), 100 * U[C].max(),
                 (np.clip(U[C], 0, None) * DZ[C] * dy).sum(), (np.clip(U[C], None, 0) * DZ[C] * dy).sum()))
    return U, DZ, HRS


PR = {phase: k[ok[k]] for phase, k in PK.items()}       # peak hours that have a residual
for phase in PK:
    print('%-5s residual: %d of %d peak hours (first 35 h dropped)' % (phase, len(PR[phase]), len(PK[phase])))
print('all peak hours')
U, DZ, HRS = averages(PK, PR)
d = dzmean(udz_res, dz_res, PR['flood']) - dzmean(udz_res, dz_res, PR['ebb'])
print('residual at flood hours - at ebb hours: rms %.2f cm/s, max |.| %.2f cm/s (residual rms %.2f cm/s)'
      % (100 * np.sqrt((d ** 2).mean()), 100 * np.abs(d).max(), 100 * np.sqrt((U['residual'] ** 2).mean())))

# ------------------------------------------------------------------ STEP 4: figure, looking out of the cove
xe = np.r_[0, np.cumsum(dy)] / 1000                     # face edges [km] from the north end, north to south
CLASSES = ['flood', 'ebb', 'residual']

def scales(UU, DD):
    """colour limits (flood & ebb shared, residual own) and the symbol reference transport, over a list of U / DZ"""
    vmax = {'flood': np.ceil(100 * max(np.abs(U_[C]).max() for U_ in UU for C in ['flood', 'ebb'])),
            'residual': np.ceil(100 * max(np.abs(U_['residual']).max() for U_ in UU))}
    vmax['ebb'] = vmax['flood']
    qref = max(np.abs(np.clip(U_[C], lo, hi) * D_[C] * dy).sum() for U_, D_ in zip(UU, DD) for C in CLASSES
               for lo, hi in [(0, None), (None, 0)])      # largest one-sign transport, to scale the symbols
    return vmax, qref


def draw_row(axr, U, DZ, HRS, VMAX, QREF, tag):
    """flood | ebb | residual sections on the 3 axes axr; returns the last pcolormesh of each class"""
    CS = {}
    for ax, C in zip(axr, CLASSES):
        zwm = zw[HRS[C]].mean(axis=0)                    # (w, p) north to south
        uc = 100 * U[C]                                  # (k, p) [cm/s]
        for j in range(len(dy)):                         # one column of native cells per face
            CS[C] = ax.pcolormesh(xe[j:j + 2], zwm[:, j], uc[:, j:j + 1], cmap='RdBu', vmin=-VMAX[C], vmax=VMAX[C])
        xc = np.broadcast_to(0.5 * (xe[1:] + xe[:-1]), uc.shape)    # (k, p) cell centres
        zc = 0.5 * (zwm[1:] + zwm[:-1])
        ax.contour(xc, zc, uc, levels=[0], colors='k', linewidths=1.5)
        lab, nlab = ndimage.label(uc > 0)                 # inflow regions (4-connected cells)
        lab_o, nlab_o = ndimage.label(uc < 0)             # outflow regions
        q = U[C] * DZ[C] * dy                             # (k, p) cell transport [m3/s], north to south
        for L in [lab, np.where(lab_o > 0, lab_o + nlab, 0)]:
            for r in np.unique(L[L > 0]):
                m = L == r
                Qr, w = q[m].sum(), np.abs(q[m])
                x0, z0 = (xc[m] * w).sum() / w.sum(), (zc[m] * w).sum() / w.sum()   # |q|-weighted centre
                d = max(DSYM * np.sqrt(abs(Qr) / QREF), DMIN)
                ax.scatter(x0, z0, s=d ** 2, facecolors='w', edgecolors='k', linewidths=1.2, zorder=3)
                ax.scatter(x0, z0, s=(0.3 * d) ** 2 if Qr > 0 else (0.5 * d) ** 2, marker='.' if Qr > 0 else 'x',
                           c='k', linewidths=1.2, zorder=4)
                ax.annotate('%+.0f' % Qr, (x0, z0), xytext=(0, -d / 2 - 2), textcoords='offset points',
                            ha='center', va='top', fontsize=10, fontweight='bold', zorder=4)
                print('  %-6s %-8s region %s: %3d cells, Q %+6.0f m3/s at x %.2f km, z %.1f m'
                      % (tag, C, 'in ' if Qr > 0 else 'out', m.sum(), Qr, x0, z0))
        ax.set_xlim(xe[0], xe[-1])
        ax.set_title('%s (%s, n = %d)' % ({'flood': 'Flood', 'ebb': 'Ebb', 'residual': 'Godin residual'}[C],
                                          'lesser peak hours' if C != 'residual' else 'same hours', len(HRS[C])),
                     fontweight='bold')
    return CS


plt.rcParams.update({'font.size': 12})
VMAX, QREF = scales([U], [DZ])
fig, axs = plt.subplots(1, 3, figsize=(15, 5), layout='constrained', sharey=True)
CS = draw_row(axs, U, DZ, HRS, VMAX, QREF, 'all')
for ax in axs:
    ax.set_xlabel('distance across [km]   N \u2190 | \u2192 S')
fig.colorbar(CS['ebb'], ax=axs[:2], label='u [cm s$^{-1}$]   + into the cove', shrink=0.9)
fig.colorbar(CS['residual'], ax=axs[2], label='u [cm s$^{-1}$]   + into the cove', shrink=0.9)
axs[0].set_ylabel('z [m]')
axs[0].set_ylim(top=np.ceil(zw[:, -1].max()))
fig.suptitle('pc_lj (Lovejoy) looking out of the cove (east), %d, %s, hourly ocean_his; blue / \u2299 = into the cove, red / \u2297 = out'
             % (YEAR, GTAGEX), fontsize=12)
out_fn = out_dir / 'pc_lj_lesser_sections.png'
fig.savefig(out_fn, dpi=200, transparent=True)
plt.close(fig)
print('saved %s' % out_fn)

# ------------------------------------------------------------------ STEP 5: spring / neap, tropic / equatorial from the utide fits
F = pickle.load(open(fit_fn, 'rb'))
UV_OUT = ['Lsmaj', 'Lsmin', 'theta', 'g']

# template coef: refit one cell (box rho j 10, i 23) on the same 2024-25 record, the same way as 20261112_pc_utide.py
jt, it_ = 10, XI_U
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
print('template refit reproduces fits.p at box rho (%d, %d)' % (jt, it_))

tr = pd.date_range(t0 - EPAD, pd.Timestamp('%d-01-01' % (YEAR + 1)) + EPAD, freq='h')
yr = (ot >= t0) & (ot < pd.Timestamp('%d-01-01' % (YEAR + 1)))
ENV, PS, PRS = {}, {}, {}
for cyc, (band, (c_hi, c_lo)) in CYCLES.items():
    ub = np.zeros((len(tr), len(ETA)))                   # (t, p) east u of the band at the faces
    for p, j in enumerate(ETA):
        for i in [XI_U, XI_U + 1]:
            k = np.flatnonzero((F['jj'] == j) & (F['ii'] == i))[0]
            c = coef.copy()
            for vn in UV_OUT:
                c[vn] = F['bar'][vn][k, io]
            c['umean'], c['vmean'] = 0.0, 0.0            # reconstruct adds the fit's mean flow back: tide only
            ub[:, p] += 0.5 * utide.reconstruct(tr, c, constit=BAND[band], verbose=False).u
    Qb = SIGN * (ub * h_f * dy).sum(axis=1)              # + into the cove
    env = pd.Series(np.abs(hilbert(Qb)), index=tr).reindex(ot)
    assert env.notna().all()
    ENV[cyc] = env

    cls = pd.Series('', index=ot)
    for mo, e in env[yr].groupby(ot[yr].month):          # terciles within each month
        lo, hi = e.quantile([1 / 3, 2 / 3])
        cls[e.index[e >= hi]] = c_hi
        cls[e.index[e <= lo]] = c_lo
    PS[cyc] = {cs: {phase: PK[phase][cls.iloc[PK[phase]].values == cs] for phase in PK} for cs in [c_hi, c_lo]}
    PRS[cyc] = {cs: {phase: k[ok[k]] for phase, k in PS[cyc][cs].items()} for cs in PS[cyc]}   # ... that have a residual
    print('Q_%s envelope in %d: %.0f to %.0f m3/s (median %.0f)' % (band, YEAR, env[yr].min(), env[yr].max(), env[yr].median()))
    for phase in PK:
        for cs in [c_hi, c_lo]:
            k = PS[cyc][cs][phase]
            print('  %-5s %-10s %3d peak hours; envelope median %4.0f m3/s, peak Q median %+5.0f m3/s'
                  % (phase, cs, len(k), env.iloc[k].median(), np.median(Q[k])))
ra = np.corrcoef(ENV['springneap'][yr], ENV['tropeq'][yr])[0, 1]
print('r(D2 envelope, D1 envelope) in %d: %+.2f' % (YEAR, ra))
for cs in ['tropic', 'equatorial']:
    n = {c2: sum(np.isin(PS['tropeq'][cs][ph], PS['springneap'][c2][ph]).sum() for ph in PK) for c2 in ['spring', 'neap']}
    print('  %-10s peak hours that are also spring %d, neap %d' % (cs, n['spring'], n['neap']))

# ------------------------------------------------------------------ STEP 6: one figure per cycle, a row per class
def class_figure(PSd, PRSd, rows, subtitle, out_fn):
    """the sections in one row per class (rows), each averaged over that class's peak hours PSd[cs]
    (residual: PRSd[cs]); one set of colour and symbol scales for all rows"""
    R = {}
    for cs in rows:
        print(cs)
        R[cs] = averages(PSd[cs], PRSd[cs])
    VMAX, QREF = scales([R[cs][0] for cs in rows], [R[cs][1] for cs in rows])
    nr = len(rows)
    fig, axs = plt.subplots(nr, 3, figsize=(15, 0.5 + 4.5 * nr), layout='constrained', sharex=True, sharey=True)
    for i, cs in enumerate(rows):
        CS = draw_row(axs[i], *R[cs], VMAX, QREF, cs)
        axs[i, 0].set_ylabel('%s\nz [m]' % cs.upper(), fontweight='bold')
    for ax in axs[-1]:
        ax.set_xlabel('distance across [km]   N \u2190 | \u2192 S')
    fig.colorbar(CS['ebb'], ax=axs[:, :2], label='u [cm s$^{-1}$]   + into the cove', shrink=0.6)
    fig.colorbar(CS['residual'], ax=axs[:, 2], label='u [cm s$^{-1}$]   + into the cove', shrink=0.6)
    axs[0, 0].set_ylim(top=np.ceil(zw[:, -1].max()))
    fig.suptitle('pc_lj (Lovejoy) looking out of the cove (east), %d, %s, hourly ocean_his; blue / \u2299 = into the cove, red / \u2297 = out\n%s'
                 % (YEAR, GTAGEX, subtitle), fontsize=12)
    fig.savefig(out_fn, dpi=200, transparent=True)
    plt.close(fig)
    print('saved %s' % out_fn)


for cyc, (band, rows) in CYCLES.items():
    class_figure(PS[cyc], PRS[cyc], rows,
                 '%s / %s = top / bottom third of hours by the %s envelope of the pc_lj tidal transport (20261112_pc_utide), within each month'
                 % (rows[0], rows[1], band), out_dir / ('pc_lj_lesser_sections_%s.png' % cyc))

# ------------------------------------------------------------------ STEP 7: wind class from the applied surface stress
dm = xr.open_dataset(mix_fn).sel(ocean_time=slice(t0, t1))
assert (pd.DatetimeIndex(dm.ocean_time.values) == ot).all()
mask = dm['mask_rho'].values == 1
mu = mask[:, :-1] & mask[:, 1:]
mu[:, 30:] = False                                       # cove u faces: west of the mouth face (box xi_u 30)
mv = mask[:-1, :] & mask[1:, :]
mv[:, 31:] = False                                       # cove v faces: rho columns <= 30
tx = np.nanmean(dm['sustr'].values[:, mu], axis=1)
ty = np.nanmean(dm['svstr'].values[:, mv], axis=1)
dm.close()
tau_al = zfun.lowpass(tx * AXIS[0] + ty * AXIS[1], f='godin')   # + toward the head
tau_cr = zfun.lowpass(tx * AXIS[1] - ty * AXIS[0], f='godin')   # + toward the north shore

WC = np.full(len(ot), '', dtype=object)
WC[tau_al > TTH] = 'up-cove'
WC[tau_al < -TTH] = 'down-cove'
WC[(np.abs(tau_al) <= TTH) & (np.abs(tau_cr) <= TTH)] = 'calm'
WROWS = ['up-cove', 'down-cove', 'calm']
PW = {cs: {phase: PK[phase][WC[PK[phase]] == cs] for phase in PK} for cs in WROWS}
PRW = {cs: {phase: k[ok[k]] for phase, k in PW[cs].items()} for cs in PW}
print('Godin tau_al in %d: %.3f to %.3f Pa; tau_cr %.3f to %.3f Pa'
      % (YEAR, np.nanmin(tau_al[yr]), np.nanmax(tau_al[yr]), np.nanmin(tau_cr[yr]), np.nanmax(tau_cr[yr])))
SEAS = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM', 6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON'}
for cs in WROWS:
    for phase in PK:
        k = PW[cs][phase]
        sea = pd.Series([SEAS[m] for m in ot[k].month]).value_counts()
        print('  %-5s %-9s %3d peak hours; tau_al median %+.3f Pa, peak Q median %+5.0f m3/s;  %s'
              % (phase, cs, len(k), np.median(tau_al[k]), np.median(Q[k]),
                 ' '.join('%s %d' % (x, sea.get(x, 0)) for x in ['DJF', 'MAM', 'JJA', 'SON'])))

# ------------------------------------------------------------------ STEP 8: wind figure, 3 rows
class_figure(PW, PRW, WROWS,
             'wind: Godin cove-mean surface stress ROMS applied; up-cove / down-cove = along-cove > +%.2f / < -%.2f Pa, '
             'calm = |along| and |cross| <= %.2f Pa' % (TTH, TTH, TTH), out_dir / 'pc_lj_lesser_sections_wind.png')

# ------------------------------------------------------------------ STEP 9: stratification class at pc_lj
SA = gsw.SA_from_SP(ts['salt'], 0, lon_f, lat_f)
sig = gsw.sigma0(SA, gsw.CT_from_pt(SA, ts['temp']))      # (t, bot/top, p)
drho_h = ((sig[:, 0] - sig[:, 1]) * dy).sum(axis=1) / dy.sum()   # (t,) width-weighted
drho = zfun.lowpass(drho_h, f='godin')
lo, hi = np.nanpercentile(drho[yr], [100 / 3, 200 / 3])
SC = np.full(len(ot), '', dtype=object)
SC[drho >= hi] = 'high strat'
SC[drho <= lo] = 'low strat'
SROWS = ['high strat', 'low strat']
PSt = {cs: {phase: PK[phase][SC[PK[phase]] == cs] for phase in PK} for cs in SROWS}
PRSt = {cs: {phase: k[ok[k]] for phase, k in PSt[cs].items()} for cs in PSt}
print('Godin drho at pc_lj in %d: %.2f to %.2f kg/m3; terciles %.2f, %.2f' % (YEAR, np.nanmin(drho[yr]), np.nanmax(drho[yr]), lo, hi))
for cs in SROWS:
    for phase in PK:
        k = PSt[cs][phase]
        sea = pd.Series([SEAS[m] for m in ot[k].month]).value_counts()
        print('  %-5s %-10s %3d peak hours; drho median %.2f kg/m3, peak Q median %+5.0f m3/s;  %s'
              % (phase, cs, len(k), np.median(drho[k]), np.median(Q[k]),
                 ' '.join('%s %d' % (x, sea.get(x, 0)) for x in ['DJF', 'MAM', 'JJA', 'SON'])))

# ------------------------------------------------------------------ STEP 10: stratification figure, 2 rows
class_figure(PSt, PRSt, SROWS,
             'stratification: Godin drho = sigma0 bottom - top at pc_lj, width-weighted; high / low = top / bottom third of %d hours '
             '(> %.2f / < %.2f kg m$^{-3}$)' % (YEAR, hi, lo), out_dir / 'pc_lj_lesser_sections_strat.png')

# ------------------------------------------------------------------ STEP 11: high / low strat, calm wind only
CROWS = ['high strat, calm', 'low strat, calm']
PSc = {cs: {phase: PK[phase][(SC[PK[phase]] == cs.split(',')[0]) & (WC[PK[phase]] == 'calm')] for phase in PK} for cs in CROWS}
PRSc = {cs: {phase: k[ok[k]] for phase, k in PSc[cs].items()} for cs in PSc}
for cs in CROWS:
    for phase in PK:
        k = PSc[cs][phase]
        sea = pd.Series([SEAS[m] for m in ot[k].month]).value_counts()
        print('  %-5s %-16s %3d peak hours; drho median %.2f kg/m3, tau_al median %+.3f Pa;  %s'
              % (phase, cs, len(k), np.median(drho[k]), np.median(tau_al[k]),
                 ' '.join('%s %d' % (x, sea.get(x, 0)) for x in ['DJF', 'MAM', 'JJA', 'SON'])))
class_figure(PSc, PRSc, CROWS,
             'stratification as STEP 10 (high / low: drho > %.2f / < %.2f kg m$^{-3}$), calm wind only '
             '(Godin |along| and |cross| stress <= %.2f Pa)' % (hi, lo, TTH), out_dir / 'pc_lj_lesser_sections_strat_calm.png')

# ------------------------------------------------------------------ STEP 11b: warm / cold and salty / fresh surface at pc_lj
TS = {}
for vn, (c_hi, c_lo), unit in [('temp', ('warm surface', 'cold surface'), 'degC'), ('salt', ('salty surface', 'fresh surface'), 'g/kg')]:
    x = zfun.lowpass((ts[vn][:, 1] * dy).sum(axis=1) / dy.sum(), f='godin')    # top s-level, width-weighted
    q0, q1 = np.nanpercentile(x[yr], [100 / 3, 200 / 3])
    cl = np.full(len(ot), '', dtype=object)
    cl[x >= q1] = c_hi
    cl[x <= q0] = c_lo
    Pd = {cs: {phase: PK[phase][cl[PK[phase]] == cs] for phase in PK} for cs in [c_hi, c_lo]}
    Prd = {cs: {phase: k[ok[k]] for phase, k in Pd[cs].items()} for cs in Pd}
    TS[vn] = (Pd, Prd, [c_hi, c_lo], q0, q1)
    print('Godin surface %s at pc_lj in %d: %.2f to %.2f %s; terciles %.2f, %.2f' % (vn, YEAR, np.nanmin(x[yr]), np.nanmax(x[yr]), unit, q0, q1))
    for cs in [c_hi, c_lo]:
        for phase in PK:
            k = Pd[cs][phase]
            sea = pd.Series([SEAS[m] for m in ot[k].month]).value_counts()
            print('  %-5s %-13s %3d peak hours; %s median %.2f, drho median %.2f;  %s'
                  % (phase, cs, len(k), vn, np.median(x[k]), np.median(drho[k]),
                     ' '.join('%s %d' % (z_, sea.get(z_, 0)) for z_ in ['DJF', 'MAM', 'JJA', 'SON'])))
    class_figure(Pd, Prd, [c_hi, c_lo],
                 'surface %s: Godin top s-level %s at pc_lj, width-weighted; %s / %s = top / bottom third of %d hours (> %.2f / < %.2f %s)'
                 % (vn, vn, c_hi, c_lo, YEAR, q1, q0, unit), out_dir / ('pc_lj_lesser_sections_%s.png' % vn))

# ------------------------------------------------------------------ STEP 12: profile figures, one per set
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


Qb, Ab = binned_QA(u, zw, dy)                                        # (t, b)
UDZ, H = (u * dz).sum(axis=1), dz.sum(axis=1)                        # (t, p)
Qb_r, Ab_r = zfun.lowpass(Qb, f='godin'), zfun.lowpass(Ab, f='godin')
UDZ_r, H_r = zfun.lowpass(UDZ, f='godin'), zfun.lowpass(H, f='godin')
WET = 0.5 * dy.mean()                                                # a bin counts as wet with at least half a face


def profiles(pk, pr):
    """{class: (profile (b,), face depth average (p,))} [m/s] for flood / ebb over pk, residual over pr"""
    out = {}
    for C, (it, qb, ab, ud, hh) in {'flood': (pk['flood'], Qb, Ab, UDZ, H), 'ebb': (pk['ebb'], Qb, Ab, UDZ, H),
                                    'residual': (np.r_[pr['flood'], pr['ebb']], Qb_r, Ab_r, UDZ_r, H_r)}.items():
        prof = qb[it].sum(axis=0) / np.maximum(ab[it].sum(axis=0), 1e-12)
        prof = np.where((ab[it] >= WET).all(axis=0), prof, np.nan)
        out[C] = (prof, ud[it].sum(axis=0) / hh[it].sum(axis=0))
    return out


SETS = {'all': ({'all': PK}, {'all': PR}, ['all'], 'all daily lesser peak hours'),
        'springneap': (PS['springneap'], PRS['springneap'], ['spring', 'neap'],
                       'spring / neap = top / bottom third of hours by the D2 envelope of the pc_lj tidal transport, within each month'),
        'tropeq': (PS['tropeq'], PRS['tropeq'], ['tropic', 'equatorial'],
                   'tropic / equatorial = top / bottom third of hours by the D1 envelope of the pc_lj tidal transport, within each month'),
        'wind': (PW, PRW, WROWS, 'Godin cove-mean applied stress: up-cove / down-cove = along > +%.2f / < -%.2f Pa, calm = |along| and |cross| <= %.2f Pa'
                 % (TTH, TTH, TTH)),
        'strat': (PSt, PRSt, SROWS, 'Godin drho at pc_lj: high / low = top / bottom third of %d hours (> %.2f / < %.2f kg m$^{-3}$)' % (YEAR, hi, lo)),
        'strat_calm': (PSc, PRSc, CROWS, 'high / low strat (drho at pc_lj > %.2f / < %.2f kg m$^{-3}$), calm wind only (|along| and |cross| stress <= %.2f Pa)' % (hi, lo, TTH)),
        'temp': (TS['temp'][0], TS['temp'][1], TS['temp'][2], 'Godin top s-level temp at pc_lj: warm / cold = top / bottom third of %d hours (> %.2f / < %.2f degC)'
                 % (YEAR, TS['temp'][4], TS['temp'][3])),
        'salt': (TS['salt'][0], TS['salt'][1], TS['salt'][2], 'Godin top s-level salt at pc_lj: salty / fresh = top / bottom third of %d hours (> %.2f / < %.2f g/kg)'
                 % (YEAR, TS['salt'][4], TS['salt'][3]))}
zcb = 0.5 * (EDGES[1:] + EDGES[:-1])                                 # bin centres, depth below the surface
xcf = 0.5 * (xe[1:] + xe[:-1])                                       # face centres [km] from the north end
for sn, (PSd, PRSd, rows, subtitle) in SETS.items():
    P = {cs: profiles(PSd[cs], PRSd[cs]) for cs in rows}
    XL = np.ceil(2 * 1.1 * 100 * max(np.nanmax(np.abs(np.r_[P[cs][C][0], P[cs][C][1]])) for cs in rows for C in CLASSES)) / 2
    zbot = max(zcb[np.isfinite(P[cs][C][0])].max() for cs in rows for C in CLASSES) + 1
    fig, axs = plt.subplots(2, 3, figsize=(15, 9), layout='constrained')
    for j, C in enumerate(CLASSES):
        for cs in rows:
            n = len(np.r_[PRSd[cs]['flood'], PRSd[cs]['ebb']]) if C == 'residual' else len(PSd[cs][C])
            axs[0, j].plot(100 * P[cs][C][0], zcb, '-', color=COL[cs], lw=2.5, label='%s (n = %d)' % (cs, n))
            axs[1, j].plot(100 * P[cs][C][1], xcf, '-', color=COL[cs], lw=2.5, label='%s (n = %d)' % (cs, n))
            print('  %-10s %-16s %-8s profile %+5.1f to %+5.1f cm/s, faces %+5.1f to %+5.1f cm/s'
                  % (sn, cs, C, 100 * np.nanmin(P[cs][C][0]), 100 * np.nanmax(P[cs][C][0]),
                     100 * P[cs][C][1].min(), 100 * P[cs][C][1].max()))
        ax = axs[0, j]
        ax.axvline(0, color='0.4', lw=0.8)
        ax.set_xlim(XL, -XL)                                         # + (into the cove) drawn LEFT
        ax.set_ylim(zbot, 0)
        ax.set_xlabel('u [cm s$^{-1}$]   \u2190 IN | OUT \u2192')
        ax.set_title({'flood': 'Flood (lesser peak hours)', 'ebb': 'Ebb (lesser peak hours)', 'residual': 'Godin residual (same hours)'}[C],
                     fontweight='bold')
        ax = axs[1, j]
        ax.axvline(0, color='0.4', lw=0.8)
        ax.set_xlim(XL, -XL)                                         # + (into the cove) drawn LEFT
        ax.set_ylim(xe[-1], xe[0])                                   # north at the top
        ax.set_xlabel('depth-averaged u [cm s$^{-1}$]   \u2190 IN | OUT \u2192')
        for i in range(2):
            axs[i, j].grid(color='lightgray', linestyle='--', alpha=0.5)
            axs[i, j].legend(loc='lower left', fontsize=9)
    axs[0, 0].set_ylabel('depth below the surface [m]')
    axs[1, 0].set_ylabel('distance across from the north end [km]\nS \u2190 | \u2192 N')   # rotated: S bottom, N top
    fig.suptitle('pc_lj (Lovejoy), %d, %s, hourly ocean_his; row 1 cross-channel averaged profile, row 2 depth average per face\n%s'
                 % (YEAR, GTAGEX, subtitle), fontsize=12)
    out_fn = out_dir / ('pc_lj_lesser_profiles_%s.png' % sn)
    fig.savefig(out_fn, dpi=200, transparent=True)
    plt.close(fig)
    print('saved %s' % out_fn)

# ------------------------------------------------------------------ STEP 13: surface / bottom third and depth-average figures
dw = zw[:, -1:, :] - zw                                             # (t, w, p) depth below the surface of each interface
d_top, d_bot = dw[:, 1:, :], dw[:, :-1, :]                          # (t, k, p) top and bottom of each cell
LAY = {}
for L, (f0, f1) in {'surface 1/3': (0, 1 / 3), 'bottom 1/3': (2 / 3, 1)}.items():
    b0, b1 = f0 * H[:, None, :], f1 * H[:, None, :]                 # (t, 1, p) the third's limits
    ov = np.clip(np.minimum(d_bot, b1) - np.maximum(d_top, b0), 0, None)
    ul, hl = (u * ov).sum(axis=1), ov.sum(axis=1)                   # (t, p)
    assert np.allclose(hl, H / 3)
    LAY[L] = (ul, hl, zfun.lowpass(ul, f='godin'), zfun.lowpass(hl, f='godin'))
LAY['depth average'] = (UDZ, H, UDZ_r, H_r)
LROWS = list(LAY)


def layers(pk, pr):
    """{layer: {class: face u (p,)}} [m/s] for flood / ebb over pk, residual over pr"""
    out = {}
    for L, (ul, hl, ul_r, hl_r) in LAY.items():
        out[L] = {}
        for C, (it, a, b) in {'flood': (pk['flood'], ul, hl), 'ebb': (pk['ebb'], ul, hl),
                              'residual': (np.r_[pr['flood'], pr['ebb']], ul_r, hl_r)}.items():
            out[L][C] = a[it].sum(axis=0) / b[it].sum(axis=0)
    return out


for sn, (PSd, PRSd, rows, subtitle) in SETS.items():
    P = {cs: layers(PSd[cs], PRSd[cs]) for cs in rows}
    XL = np.ceil(2 * 1.1 * 100 * max(np.abs(P[cs][L][C]).max() for cs in rows for L in LROWS for C in CLASSES)) / 2
    fig, axs = plt.subplots(3, 3, figsize=(15, 13), layout='constrained', sharex=True, sharey=True)
    for i, L in enumerate(LROWS):
        for j, C in enumerate(CLASSES):
            ax = axs[i, j]
            for cs in rows:
                n = len(np.r_[PRSd[cs]['flood'], PRSd[cs]['ebb']]) if C == 'residual' else len(PSd[cs][C])
                ax.plot(100 * P[cs][L][C], xcf, '-', color=COL[cs], lw=2.5, label='%s (n = %d)' % (cs, n))
                print('  %-10s %-16s %-13s %-8s faces %+5.1f to %+5.1f cm/s'
                      % (sn, cs, L, C, 100 * P[cs][L][C].min(), 100 * P[cs][L][C].max()))
            ax.axvline(0, color='0.4', lw=0.8)
            ax.grid(color='lightgray', linestyle='--', alpha=0.5)
            if i == 0:
                ax.set_title({'flood': 'Flood (lesser peak hours)', 'ebb': 'Ebb (lesser peak hours)',
                              'residual': 'Godin residual (same hours)'}[C], fontweight='bold')
            if i == len(LROWS) - 1:
                ax.set_xlabel('u [cm s$^{-1}$]   \u2190 IN | OUT \u2192')
            if i == 0 and j == 0:
                ax.legend(loc='lower left', fontsize=9)
        axs[i, 0].set_ylabel('%s\ndistance across from the north end [km]\nS \u2190 | \u2192 N' % L.upper(), fontweight='bold')
    axs[0, 0].set_xlim(XL, -XL)                                      # + (into the cove) drawn LEFT
    axs[0, 0].set_ylim(xe[-1], xe[0])                                # north at the top
    fig.suptitle('pc_lj (Lovejoy), %d, %s, hourly ocean_his; u at each face in the surface / bottom third of the local water depth, '
                 'and the depth average\n%s' % (YEAR, GTAGEX, subtitle), fontsize=12)
    out_fn = out_dir / ('pc_lj_lesser_layers_%s.png' % sn)
    fig.savefig(out_fn, dpi=200, transparent=True)
    plt.close(fig)
    print('saved %s' % out_fn)

# ------------------------------------------------------------------ STEP 14: surface / bottom third and full column at every cove cell
cache = cache_dir / 'cove_layers_hourly.npz'
dsb = xr.open_dataset(box_fn).sel(ocean_time=slice(t0, t1))
assert (pd.DatetimeIndex(dsb.ocean_time.values) == ot).all()
MASK = dsb['mask_rho'].values == 1
LON, LAT = dsb['lon_rho'].values, dsb['lat_rho'].values
cove = MASK.copy()
cove[:, 31:] = False                                     # the cove: rho columns <= 30
JJ, II = np.nonzero(cove)
MLAY = ['surface 1/3', 'bottom 1/3', 'depth average']
FR = [(0, 1 / 3), (2 / 3, 1), (0, 1)]                    # fractions of the local water depth below the surface
if cache.exists():
    z = np.load(cache)
    Ux, Vy, Hl = z['Ux'], z['Vy'], z['Hl']
else:
    Ux, Vy, Hl = [np.full((len(ot), len(MLAY), len(JJ)), np.nan, np.float32) for _ in range(3)]
    for b0 in range(0, len(ot), BLK):
        tt = slice(b0, min(b0 + BLK, len(ot)))
        U_ = np.nan_to_num(dsb['u'].isel(ocean_time=tt).values.astype(float))   # (t, k, eta_u, xi_u)
        V_ = np.nan_to_num(dsb['v'].isel(ocean_time=tt).values.astype(float))   # (t, k, eta_v, xi_v)
        ZW = dsb['z_w'].isel(ocean_time=tt).values                              # (t, w, eta, xi)
        ur = np.zeros(U_.shape[:2] + ZW.shape[2:])
        vr = np.zeros(U_.shape[:2] + ZW.shape[2:])
        ur[..., 1:31] = 0.5 * (U_[..., 0:30] + U_[..., 1:31])            # rho i: faces xi_u i-1 (west) and i (east)
        vr[:, :, 1:20, :] = 0.5 * (V_[:, :, 0:19, :] + V_[:, :, 1:20, :])
        zwc = ZW[:, :, JJ, II]                                            # (t, w, cell)
        dwc = zwc[:, -1:, :] - zwc
        dt_, db_ = dwc[:, 1:, :], dwc[:, :-1, :]                          # (t, k, cell) top / bottom depth of each cell
        Hc = dwc[:, 0:1, :]                                               # (t, 1, cell) water depth
        uc, vc = ur[:, :, JJ, II], vr[:, :, JJ, II]
        for n, (f0, f1) in enumerate(FR):
            ov = np.clip(np.minimum(db_, f1 * Hc) - np.maximum(dt_, f0 * Hc), 0, None)
            Ux[tt, n], Vy[tt, n], Hl[tt, n] = (uc * ov).sum(axis=1), (vc * ov).sum(axis=1), ov.sum(axis=1)
        print('  STEP 14 hours %d-%d of %d' % (tt.start, tt.stop, len(ot)))
    np.savez(cache, Ux=Ux, Vy=Vy, Hl=Hl)
dsb.close()
Ux, Vy, Hl = Ux.astype(float), Vy.astype(float), Hl.astype(float)
assert np.allclose(Hl[:, 0], Hl[:, 2] / 3, rtol=1e-4) and np.allclose(Hl[:, 1], Hl[:, 2] / 3, rtol=1e-4)
Ux_r, Vy_r, Hl_r = [zfun.lowpass(a, f='godin') for a in (Ux, Vy, Hl)]
print('STEP 14: %d cove cells, %d hours' % (len(JJ), len(ot)))

# ------------------------------------------------------------------ STEP 15: cove maps, one figure per class
al = lambda u, v: u * AXIS[0] + v * AXIS[1]              # along-cove, + toward the head


def to2d(a):
    out = np.full(LON.shape, np.nan)
    out[JJ, II] = a
    return out


def basemap(ax):
    ax.pcolormesh(LON, LAT, np.where(MASK, np.nan, 1.0), cmap='Greys', vmin=0, vmax=2.5, shading='nearest')
    for SN, XI in SECT_XI_U.items():
        jw = np.flatnonzero(MASK[:, XI] & MASK[:, XI + 1])
        lo_ = 0.5 * (LON[0, XI] + LON[0, XI + 1])
        ax.plot([lo_, lo_], [LAT[jw.min(), 0] - 0.0009, LAT[jw.max(), 0] + 0.0009], color='0.2',
                lw=3 if SN == 'pc_lj' else 1.2)
    ax.set_xlim(LON[0, 0], LON[0, -1])
    ax.set_ylim(LAT[0, 0], LAT[-1, 0])
    ax.set_aspect(1 / np.cos(np.deg2rad(48.23)))
    ax.tick_params(labelsize=8)


def maps(pk, pr):
    """{layer: {phase: (u, v) per cove cell}} [m/s]"""
    out = {}
    for n, L in enumerate(MLAY):
        out[L] = {}
        for C, (it, a, b, c) in {'flood': (pk['flood'], Ux, Vy, Hl), 'ebb': (pk['ebb'], Ux, Vy, Hl),
                                 'residual': (np.r_[pr['flood'], pr['ebb']], Ux_r, Vy_r, Hl_r)}.items():
            hs = c[it, n].sum(axis=0)
            out[L][C] = (a[it, n].sum(axis=0) / hs, b[it, n].sum(axis=0) / hs)
    return out


for sn, (PSd, PRSd, rows, subtitle) in SETS.items():
    M = {cs: maps(PSd[cs], PRSd[cs]) for cs in rows}
    VM = {'flood': np.ceil(100 * max(np.hypot(*M[cs][L][C]).max() for cs in rows for L in MLAY for C in ['flood', 'ebb'])),
          'residual': np.ceil(100 * max(np.hypot(*M[cs][L]['residual']).max() for cs in rows for L in MLAY))}
    VM['ebb'] = VM['flood']
    for cs in rows:
        fig, axs = plt.subplots(3, 3, figsize=(16, 9.5), layout='constrained', sharex=True, sharey=True)
        for i, L in enumerate(MLAY):
            for j, C in enumerate(CLASSES):
                ax = axs[i, j]
                basemap(ax)
                u_, v_ = M[cs][L][C]
                CSm = ax.pcolormesh(LON, LAT, to2d(100 * al(u_, v_)), cmap='RdBu', vmin=-VM[C], vmax=VM[C], shading='nearest')
                ax.quiver(LON[JJ, II], LAT[JJ, II], u_, v_, scale=VM[C] / 100 * 12, scale_units='width', width=0.003, color='k')
                if i == 0:
                    n = len(np.r_[PRSd[cs]['flood'], PRSd[cs]['ebb']]) if C == 'residual' else len(PSd[cs][C])
                    ax.set_title('%s (n = %d)' % ({'flood': 'Flood', 'ebb': 'Ebb', 'residual': 'Godin residual'}[C], n), fontweight='bold')
                if j == 0:
                    ax.set_ylabel(L.upper(), fontweight='bold')
                if (i, C) in [(1, 'ebb'), (1, 'residual')]:
                    fig.colorbar(CSm, ax=axs[:, :2] if C == 'ebb' else axs[:, 2], shrink=0.5,
                                 label='along-cove u [cm s$^{-1}$], + into the cove' + (' (own scale)' if C == 'residual' else ''))
                print('  %-10s %-16s %-13s %-8s along-cove %+5.1f to %+5.1f cm/s'
                      % (sn, cs, L, C, 100 * al(u_, v_).min(), 100 * al(u_, v_).max()))
        fig.suptitle('Penn Cove, %s, %d, %s, hourly ocean_his, at the pc_lj (thick line) daily lesser peak flood / ebb hours (residual: Godin, same hours);\n'
                     'velocity in the surface / bottom third of the local water depth and the depth average\n%s' % (cs.upper(), YEAR, GTAGEX, subtitle), fontsize=12)
        out_fn = out_dir / ('cove_lesser_maps_%s_%s.png' % (sn, cs.replace(', ', '_').replace(' ', '_')))
        fig.savefig(out_fn, dpi=200, transparent=True)
        plt.close(fig)
        print('saved %s' % out_fn)
