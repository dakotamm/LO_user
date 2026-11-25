"""
Penn Cove pc_lj (Lovejoy): the most extreme greater / lesser flood and ebb profiles of 2024, and what
characterises them. Same tides as 20261117_pc_every_flood_ebb.py and 20261117_pc_profile_drivers.py (every
half-cycle at its peak hour, greater / lesser = the bigger / smaller of each lunar day).

STEP 1  load pc_lj from the pc_cove box: u normal to the faces (+ = into the cove), z_w and dy at the faces, and
        sigma0 (gsw) in every cell of the face columns (mean of the two rho cells either side) for STEP 3.
STEP 2  half-cycles, peak hours and lunar days, as 20261117_pc_profile_drivers.py STEP 1.
STEP 3  at every peak hour: profile U_b = Q_b / A_b in EDGES (1 m) bins below the moving surface, face depth
        average U_p, and the cross-channel area-mean sigma0 in the same bins. Profile bins: wet in every tide of
        the row at its peak hour (one more bin than profile_drivers, which also needed peak -/+ 1 h).
STEP 4  extremes, per row and field, on the SHAPE (u - Ubar, Ubar = Q / A): on raw u the extremes would just be the
        biggest tides. Two kinds of extreme:
            far       the NX tides with the largest area-weighted RMS distance from the row's mean shape,
                      d = sqrt(sum w (x - mean)^2 / sum w): unusual in any direction
            EOF1 + / -, EOF2 + / -   the NX tides at each tail of the shape PCs (EOFs as profile_drivers STEP 3,
                      sign: + = upper / north water more INTO the cove than the section mean): unusual in a named way
        and the most COMMON tides (added 2026-11-17), the same size:
            near      the NX tides with the smallest d: closest to the row's mean shape
            mode      the NX tides at the highest density in (PC1, PC2) (gaussian_kde, Scott bandwidth; the PCs have
                      unit variance, so one bandwidth fits both): where the tides pile up, which is not the mean
                      shape if the distribution is skewed
        NX = XFRAC of the row's tides. Events: extremes closer than EVENT_GAP in time are one event (printed), so a
        set of 18 tides from 3 storms reads as 3, not 18.
STEP 5  drivers at every hour (profile_drivers STEP 5 + STEP 9, the season / hour sin-cos left out: a percentile
        of them means nothing) plus Ubar, the sun elevation at pc_lj (day / night, which carries the hour of day
        through the year) and the D2 / D1 envelopes. Characteristic of a set = the median percentile of its tides
        within the row (50 = typical of the row). Null: the median percentile of NDRAW random sets of NX tides of
        the row, 2.5-97.5%; a set outside it is marked. The random sets treat tides as independent, while
        extremes cluster in events, so the band is narrower than an honest one.
STEP 6  figures and table:
            extremes_[field]_[set].png  4 x 3 (rows x profile u | face u | sigma0 profile): every tide of the row
                                        light grey, the row mean black, the extreme tides in the set colour and
                                        their mean dashed
            characteristics_[field].png heat map, drivers x (row, set): median percentile, outside the null in bold
            timeline_[field].png        4 x 3: PC1 and PC2 through 2024 with the extremes marked, PC1 vs PC2
            common_[field]_[near|mode].png  the same for the most common sets
            extremes.csv                one line per extreme (and common) tide: time, row, field, set, rank, score, PCs, drivers

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc
       LO_output/extract/[gtagex]/box/pc_cove_mix_[ds0]_[ds1].nc
       LO_output/DM_outs/20261112_pc_utide/fits.p
       LO_output/DM_outs/20261113_pc_candidate_days/picks.csv
       LO_output/DM_outs/20260806_river_hydrographs/daily_flow.csv
       LO_output/DM_outs/20260806_wind/wind_hourly_atm00_[ds0]_[ds1].p
       LO_output/DM_outs/20261110_pc_metric_ts/pc_metric_ts.nc
OUTPUT LO_output/DM_outs/20261117_pc_extreme_profiles/

run 20261117_pc_extreme_profiles.py
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
from scipy.linalg import svd
from scipy.signal import hilbert
from scipy.stats import gaussian_kde

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
NM = 2                                    # shape EOFs used for the directional extremes
XFRAC = 0.05                              # share of each row's tides that is 'extreme'
EVENT_GAP = pd.Timedelta(days=3)          # extremes closer than this are one event
NDRAW = 2000                              # random sets for the null (STEP 5)
SKAGIT_LAG = pd.Timedelta(days=3)         # Skagit flow lag
SAR_BOX = (-122.62, -122.42, 48.02, 48.22)   # Saratoga Passage wind cells: lon0, lon1, lat0, lat1
SAR_AXIS = (-0.55, 0.83)                  # Saratoga Passage axis, + up-passage (toward the NW)
ROWS = [('flood', 'greater'), ('flood', 'lesser'), ('ebb', 'greater'), ('ebb', 'lesser')]
FIELDS = ['profile', 'faces']
SETS = ['far', 'EOF1 +', 'EOF1 -', 'EOF2 +', 'EOF2 -', 'near', 'mode']
SCOL = {'far': '#AA3377', 'EOF1 +': '#CC3311', 'EOF1 -': '#0077BB', 'EOF2 +': '#EE7733', 'EOF2 -': '#009988',
        'near': '#228833', 'mode': '#BBBB00'}

gridname, tag, ex_name = GTAGEX.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
box_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_%s_%s.nc' % (DS0, DS1))
mix_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_mix_%s_%s.nc' % (DS0, DS1))
fit_fn = Ldir['LOo'] / 'DM_outs' / '20261112_pc_utide' / 'fits.p'
pick_fn = Ldir['LOo'] / 'DM_outs' / '20261113_pc_candidate_days' / 'picks.csv'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261117_pc_extreme_profiles'
Lfun.make_dir(out_dir)

# ------------------------------------------------------------------ STEP 1: load pc_lj from the box
t0 = pd.Timestamp('%d-01-01' % YEAR)
t1 = pd.Timestamp('%d-01-01' % (YEAR + 1))
ds = xr.open_dataset(box_fn).sel(ocean_time=slice(t0, t1 + PAD))
ot = pd.DatetimeIndex(ds.ocean_time.values)
u = SIGN * ds['u'].isel(xi_u=XI_U, eta_u=ETA).values.astype(float)                   # (t, k, p) + into the cove
zw = ds['z_w'].isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).values.mean(axis=-1)       # (t, w, p) at the face
dy = (1 / ds['pn'].isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).values).mean(axis=-1)  # (p,)
h_f = ds['h'].isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).values.mean(axis=-1)      # (p,) depth at the face
lon_c = ds['lon_rho'].isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).values               # (p, 2)
lat_c = ds['lat_rho'].isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).values
sl = dict(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1])
SA = gsw.SA_from_SP(ds['salt'].isel(**sl).values, 0, lon_c, lat_c)                  # (t, k, p, 2)
sig = gsw.sigma0(SA, gsw.CT_from_pt(SA, ds['temp'].isel(**sl).values)).mean(axis=-1)  # (t, k, p) at the face
lon_f, lat_f = ds['lon_u'].isel(eta_u=ETA, xi_u=XI_U).values, ds['lat_u'].isel(eta_u=ETA, xi_u=XI_U).values
ds.close()
dz = np.diff(zw, axis=1)                                                             # (t, k, p)
Q = (u * dz * dy).sum(axis=(1, 2))                                                   # (t,) + = flood
A = (dz * dy).sum(axis=(1, 2))                                                       # (t,) section area
print('%s  %d hours  %s to %s' % (box_fn.name, len(ot), ot[0], ot[-1]))

# ------------------------------------------------------------------ STEP 2: half-cycles, peak hours, lunar days
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
        PK[(phase, kind)] = k[(ot[k] >= t0) & (ot[k] < t1)]
        print('%-7s %-5s %3d tides' % (kind, phase, len(PK[(phase, kind)])))

# ------------------------------------------------------------------ STEP 3: profile, faces and sigma0 at every peak hour
def binned(f, zw, dy):
    """F_b = sum f ov dy and A_b = sum ov dy (t, b) in the EDGES bins of depth below the moving surface, top bin first,
    ov = each cell's overlap with the bin. f = u gives Q_b (sum_b Q_b = Q exactly, as 20261113_pc_baseline_profiles.py);
    F_b / A_b of any cell field is its area mean in the bin."""
    dz = np.diff(zw, axis=1)
    dw = zw[:, -1:, :] - zw                                          # depth below the surface of each interface
    d_top, d_bot = dw[:, 1:, :], dw[:, :-1, :]
    assert d_bot.max() < EDGES[-1]
    nb = len(EDGES) - 1
    Fb, Ab = np.zeros((len(f), nb)), np.zeros((len(f), nb))
    for b in range(nb):
        ov = np.clip(np.minimum(d_bot, EDGES[b + 1]) - np.maximum(d_top, EDGES[b]), 0, None)
        Fb[:, b] = (f * ov * dy).sum(axis=(1, 2))
        Ab[:, b] = (ov * dy).sum(axis=(1, 2))
    return Fb, Ab


WET = 0.5 * dy.mean()                                                # a bin counts as wet with at least half a face
zcb = 0.5 * (EDGES[1:] + EDGES[:-1])                                 # bin centres, depth below the surface
xe = np.r_[0, np.cumsum(dy)] / 1000                                  # face edges [km] from the north end
xcf = 0.5 * (xe[1:] + xe[:-1])                                       # face centres
D = {}                                                               # per row: hours, Ubar, profile, faces, sigma, weights
for row in ROWS:
    k = PK[row]
    Qb, Ab = binned(u[k], zw[k], dy)
    assert np.allclose(Qb.sum(axis=1), Q[k])
    Sb, _ = binned(sig[k], zw[k], dy)
    wet = (Ab >= WET).all(axis=0)                                    # bins wet in every tide of the row
    UDZ, H = (u[k] * dz[k]).sum(axis=1), dz[k].sum(axis=1)           # (n, p)
    D[row] = {'hrs': k, 'Ubar': Q[k] / A[k], 'profile': Qb[:, wet] / Ab[:, wet], 'faces': UDZ / H,
              'sigma': Sb[:, wet] / Ab[:, wet], 'w_profile': Ab[:, wet].mean(axis=0), 'w_faces': (H * dy).mean(axis=0),
              'z': zcb[wet]}
    print('%-7s %-5s %3d tides; profile bins wet in all: 0-%.0f m; Ubar median %+5.1f cm/s; sigma0 %.2f to %.2f'
          % (row[1], row[0], len(k), EDGES[1:][wet].max(), 100 * np.median(D[row]['Ubar']),
             np.nanmin(D[row]['sigma']), np.nanmax(D[row]['sigma'])))

# ------------------------------------------------------------------ STEP 4: shape, EOFs, extreme sets
def eof(x, w, upper):
    """EOFs as 20261117_pc_profile_drivers.py STEP 3 (UW Lab 8-1), first NM modes. x (n tides, s space) shape,
    w (s,) area weights, upper (s,) bool = the 'upper' points for the sign."""
    mean = x.mean(axis=0)
    X = (x - mean).T * np.sqrt(w)[:, None]                           # (space, tide), weighted
    U, S, V = svd(X, full_matrices=False)
    var_exp = S ** 2 / (S ** 2).sum()
    n = x.shape[0]
    U, S, V = U[:, :NM].copy(), S[:NM], V[:NM].copy()
    for i in range(NM):                                              # PC > 0 = upper water more INTO the cove
        e = U[:, i] / np.sqrt(w)
        if (e[upper] * w[upper]).sum() < 0:
            U[:, i], V[i] = -U[:, i], -V[i]
    return {'mean': mean, 'eof': U * S / np.sqrt(n) / np.sqrt(w)[:, None], 'pc': V * np.sqrt(n), 'var_exp': var_exp}


def events(tt):
    """[(first, last, n), ...] of the times tt grouped where consecutive gaps are < EVENT_GAP"""
    tt = pd.DatetimeIndex(np.sort(tt))
    br = np.flatnonzero(np.diff(tt) >= EVENT_GAP) + 1
    return [(g[0], g[-1], len(g)) for g in np.split(tt, br)]


EOF, XS = {}, {}                                                     # XS[(row, field)][set] = indices into the row's tides
for row in ROWS:
    d = D[row]
    n = len(d['hrs'])
    NX = int(round(XFRAC * n))
    for f in FIELDS:
        x, w = d[f] - d['Ubar'][:, None], d['w_' + f]                # shape (n, s)
        d['shape_' + f] = x
        if f == 'profile':
            ms = x.mean(axis=0)
            zc = np.flatnonzero(np.sign(ms) != np.sign(ms[0]))
            upper = np.arange(len(ms)) < (zc[0] if len(zc) else len(ms) // 2)   # above the first zero crossing
        else:
            upper = np.arange(len(dy)) < len(dy) // 2                # north half
        E = eof(x, w, upper)
        E['dist'] = np.sqrt(((x - E['mean']) ** 2 * w).sum(axis=1) / w.sum())   # area-weighted RMS distance [m/s]
        EOF[(row, f)] = E
        o1, o2 = np.argsort(E['pc'][0]), np.argsort(E['pc'][1])
        XS[(row, f)] = {'far': np.argsort(-E['dist'])[:NX], 'EOF1 +': o1[::-1][:NX], 'EOF1 -': o1[:NX],
                        'EOF2 +': o2[::-1][:NX], 'EOF2 -': o2[:NX], 'near': np.argsort(E['dist'])[:NX]}
        E['dens'] = gaussian_kde(E['pc'][:2])(E['pc'][:2])            # density in (PC1, PC2) at each tide
        XS[(row, f)]['mode'] = np.argsort(-E['dens'])[:NX]
        print('%-7s %-5s %-7s EOF1 %.0f%%, EOF2 %.0f%%; RMS distance median %.1f, extreme (top %d) >= %.1f cm/s'
              % (row[1], row[0], f, 100 * E['var_exp'][0], 100 * E['var_exp'][1], 100 * np.median(E['dist']), NX,
                 100 * np.sort(E['dist'])[-NX]))
        for s in SETS:
            ev = events(ot[d['hrs'][XS[(row, f)][s]]])
            print('    %-6s %2d tides in %2d events: %s' % (s, NX, len(ev), ', '.join(
                '%s%s (%d)' % (a.strftime('%m-%d'), '' if b - a < pd.Timedelta(days=1) else '..' + b.strftime('%m-%d'), m)
                for a, b, m in ev)))

# ------------------------------------------------------------------ STEP 5: drivers at every hour (profile_drivers STEP 5 + 9)
DRV = {}                                                 # {name: (t,) hourly series}
DRV['|Q| peak'] = np.abs(Q)
DRV['Ubar'] = np.abs(Q / A)

# tide: D2 and D1 envelopes of the pc_lj tidal transport
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
cove = mask.copy()
cove[:, 31:] = False                                     # cove water: rho columns <= 30
shf = dm['shflux'].values[:, cove].mean(axis=1)
dm.close()
t_al, t_cr = tx * AXIS[0] + ty * AXIS[1], tx * AXIS[1] - ty * AXIS[0]   # + toward the head / the north shore
DRV['tau_al Godin'] = zfun.lowpass(t_al, f='godin')
DRV['tau_al %dh' % WIND_H] = pd.Series(t_al).rolling(WIND_H).mean().to_numpy()   # mean of the WIND_H h ending at the hour
DRV['tau_cr Godin'] = zfun.lowpass(t_cr, f='godin')
DRV['tau_cr %dh' % WIND_H] = pd.Series(t_cr).rolling(WIND_H).mean().to_numpy()

# density: drho = sigma0 bottom - top at pc_lj (bottom / top s-level), width-weighted, Godin and at the hour
drho_h = ((sig[:, 0] - sig[:, -1]) * dy).sum(axis=1) / dy.sum()
DRV['drho Godin'] = zfun.lowpass(drho_h, f='godin')
DRV['drho peak'] = drho_h

# in-out density difference (pc_lp - pc_cp), top / bottom s-level, Godin
dsb = xr.open_dataset(box_fn).sel(ocean_time=slice(t0, t1 + PAD))
sig_ts = {}
for sn, X in [('pc_cp', 15), ('pc_lp', 30)]:             # wet faces: rho cells both sides wet
    jj = np.flatnonzero(mask[:, X] & mask[:, X + 1])
    sl = dict(s_rho=[0, -1], eta_rho=jj, xi_rho=[X, X + 1])
    sa, te = dsb['salt'].isel(**sl).values, dsb['temp'].isel(**sl).values          # (t, bot/top, face, 2)
    lo, la = dsb['lon_rho'].isel(eta_rho=jj, xi_rho=[X, X + 1]).values, dsb['lat_rho'].isel(eta_rho=jj, xi_rho=[X, X + 1]).values
    SA_ = gsw.SA_from_SP(sa, 0, lo, la)
    sg_ = gsw.sigma0(SA_, gsw.CT_from_pt(SA_, te)).mean(axis=-1)                   # (t, bot/top, face)
    w_ = (1 / dsb['pn'].isel(eta_rho=jj, xi_rho=[X, X + 1]).values).mean(axis=-1)  # face width
    sig_ts[sn] = (sg_ * w_).sum(axis=-1) / w_.sum()                                  # (t, bot/top) width-mean
zeta_c = dsb['zeta'].values[:, cove].mean(axis=1)
dsb.close()
DRV['drhox top Godin'] = zfun.lowpass(sig_ts['pc_lp'][:, 1] - sig_ts['pc_cp'][:, 1], f='godin')
DRV['drhox bot Godin'] = zfun.lowpass(sig_ts['pc_lp'][:, 0] - sig_ts['pc_cp'][:, 0], f='godin')

# heat flux, tidal history, subtidal sea level and net transport
DRV['shflux %dh' % WIND_H] = pd.Series(shf).rolling(WIND_H).mean().to_numpy()
DRV['shflux Godin'] = zfun.lowpass(shf, f='godin')
prev1, prev2 = np.full(len(ot), np.nan), np.full(len(ot), np.nan)   # defined at the peak hours
for i in range(2, len(hc_pk)):
    prev1[hc_pk[i]], prev2[hc_pk[i]] = np.abs(Q[hc_pk[i - 1]]), np.abs(Q[hc_pk[i - 2]])
DRV['prev half |Q|'], DRV['prev same |Q|'] = prev1, prev2
DRV['zeta Godin'] = zfun.lowpass(zeta_c, f='godin')
DRV['Q Godin'] = zfun.lowpass(Q, f='godin')

# Skagit flow, lagged
qf = pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20260806_river_hydrographs' / 'daily_flow.csv', parse_dates=['date']).set_index('date')['skagit']
DRV['Skagit %dd' % SKAGIT_LAG.days] = qf.reindex(ot.floor('D') - SKAGIT_LAG).to_numpy()

# wind stress over Saratoga Passage, from the hourly atm00 wind cube
WC_ = pickle.load(open(Ldir['LOo'] / 'DM_outs' / '20260806_wind' / ('wind_hourly_atm00_%s_%s.p' % (DS0, DS1)), 'rb'))['cube']
sel = WC_['water'] & (WC_['lon'] > SAR_BOX[0]) & (WC_['lon'] < SAR_BOX[1]) & (WC_['lat'] > SAR_BOX[2]) & (WC_['lat'] < SAR_BOX[3])
uw, vw = WC_['u'][:, sel], WC_['v'][:, sel]                         # (t, cells) [m/s]
sp = np.hypot(uw, vw)
cdl = np.where(sp < 10.0, 1.14e-3, (0.49 + 0.065 * sp) * 1e-3)     # Large & Pond (1981), as 20260806_wind_reduce.py
txs, tys = (1.22 * cdl * sp * uw).mean(axis=1), (1.22 * cdl * sp * vw).mean(axis=1)
tsa = pd.Series(txs * SAR_AXIS[0] + tys * SAR_AXIS[1], index=pd.DatetimeIndex(WC_['time'])).reindex(ot).to_numpy()
tsc = pd.Series(txs * SAR_AXIS[1] - tys * SAR_AXIS[0], index=pd.DatetimeIndex(WC_['time'])).reindex(ot).to_numpy()
DRV['tau_sar al %dh' % WIND_H] = pd.Series(tsa).rolling(WIND_H).mean().to_numpy()
DRV['tau_sar al Godin'] = zfun.lowpass(tsa, f='godin')
DRV['tau_sar cr %dh' % WIND_H] = pd.Series(tsc).rolling(WIND_H).mean().to_numpy()
DRV['tau_sar cr Godin'] = zfun.lowpass(tsc, f='godin')

# eddy state: the back-cove ring (subtidal)
dsr = xr.open_dataset(Ldir['LOo'] / 'DM_outs' / '20261110_pc_metric_ts' / 'pc_metric_ts.nc')
tt = pd.DatetimeIndex(dsr.time.values)
DRV['eddy rot_U'] = pd.Series(dsr['rot_U'].values, index=tt).reindex(ot).to_numpy()
DRV['eddy rot_frac'] = pd.Series(dsr['rot_frac'].values, index=tt).reindex(ot).to_numpy()
for b, nm in [('z0-4', 'eddy U_rot z0-4'), ('z10-bed', 'eddy U_rot z10-bed')]:
    DRV[nm] = pd.Series(dsr['U_rot_band'].sel(band=b).values, index=tt).reindex(ot).to_numpy()
dsr.close()


def sun_elevation(t, lat, lon):
    """solar elevation [deg] at the UTC times t (DatetimeIndex), NOAA general solar position approximation
    (as 20261117_pc_every_flood_ebb.py)"""
    hr = t.hour.to_numpy() + t.minute.to_numpy() / 60
    g = 2 * np.pi / 365 * (t.dayofyear.to_numpy() - 1 + (hr - 12) / 24)          # fractional year
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
           - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    ha = np.radians((hr * 60 + eqt + 4 * lon) / 4 - 180)                         # hour angle (true solar time)
    la = np.radians(lat)
    return 90 - np.degrees(np.arccos(np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)))


DRV['sun elevation'] = sun_elevation(ot, lat_f.mean(), lon_f.mean())
DNAME = list(DRV)

# characteristic of each set: median percentile of its tides within the row, against random sets of the same size
rng = np.random.default_rng(0)
CH = []
for row in ROWS:
    d = D[row]
    n = len(d['hrs'])
    P = np.full((n, len(DNAME)), np.nan)                 # percentile of each tide within the row, per driver
    for j, nm in enumerate(DNAME):
        v = DRV[nm][d['hrs']]
        ok = np.isfinite(v)
        P[ok, j] = 100 * (pd.Series(v[ok]).rank().to_numpy() - 0.5) / ok.sum()
    d['pct'] = P
    NX = len(XS[(row, FIELDS[0])]['far'])
    null = np.array([np.nanmedian(P[rng.choice(n, NX, replace=False)], axis=0) for _ in range(NDRAW)])
    lo, hi = np.percentile(null, [2.5, 97.5], axis=0)
    for f in FIELDS:
        for s in SETS:
            med = np.nanmedian(P[XS[(row, f)][s]], axis=0)
            for j, nm in enumerate(DNAME):
                CH.append({'row': '%s %s' % (row[1], row[0]), 'field': f, 'set': s, 'driver': nm, 'pct': med[j],
                           'null_lo': lo[j], 'null_hi': hi[j], 'outside': (med[j] < lo[j]) or (med[j] > hi[j])})
CH = pd.DataFrame(CH)
for (r_, f, s), g in CH.groupby(['row', 'field', 'set'], sort=False):
    g = g[g['outside']].assign(dev=lambda x: (x['pct'] - 50).abs()).sort_values('dev', ascending=False)
    print('%-14s %-7s %-6s %s' % (r_, f, s, ', '.join('%s %.0f' % (a, b) for a, b in zip(g['driver'][:5], g['pct'][:5])) or '-'))
CH.to_csv(out_dir / 'characteristics.csv', index=False, float_format='%.2f')
print('saved %s' % (out_dir / 'characteristics.csv'))

# ------------------------------------------------------------------ STEP 6: figures and table
plt.rcParams.update({'font.size': 12})
XL = [np.ceil(1.05 * 100 * max(np.abs(D[r][f]).max() for r in ROWS)) for f in FIELDS]   # u scale per column, every figure
SL = (np.floor(min(np.nanmin(D[r]['sigma']) for r in ROWS)), np.ceil(max(np.nanmax(D[r]['sigma']) for r in ROWS)))
zbot = max(D[r]['z'].max() for r in ROWS) + 1

for f in FIELDS:
    for s in SETS:
        fig, axs = plt.subplots(len(ROWS), 3, figsize=(20, 22), layout='constrained', sharex='col')
        for r, row in enumerate(ROWS):
            d, x = D[row], XS[(row, f)][s]
            for c, (vn, y) in enumerate([('profile', d['z']), ('faces', xcf), ('sigma', d['z'])]):
                ax, sc = axs[r, c], 1 if vn == 'sigma' else 100
                ax.plot(sc * d[vn].T, y, '-', color='0.6', lw=0.5, alpha=0.1)
                ax.plot(sc * d[vn][x].T, y, '-', color=SCOL[s], lw=0.8, alpha=0.6)
                ax.plot(sc * d[vn].mean(axis=0), y, '-', color='k', lw=3, label='row mean (n = %d)' % len(d['hrs']))
                ax.plot(sc * d[vn][x].mean(axis=0), y, '--', color=SCOL[s], lw=3, label='%s mean (n = %d)' % (s, len(x)))
                ax.grid(color='lightgray', linestyle='--', alpha=0.5)
                if vn == 'sigma':
                    ax.set_xlim(*SL)
                else:
                    ax.axvline(0, color='0.4', lw=0.8)
                    ax.set_xlim(XL[c], -XL[c])                       # + (into the cove) drawn LEFT
                if vn == 'faces':
                    ax.set_ylim(xe[-1], xe[0])                       # north at the top
                    ax.set_ylabel('distance across from the north end [km]\nS ← | → N')
                else:
                    ax.set_ylim(zbot, 0)
            axs[r, 0].set_ylabel('%s %s\ndepth below the surface [m]' % (row[1].upper(), row[0].upper()), fontweight='bold')
            ev = events(ot[d['hrs'][x]])
            axs[r, 2].text(0.02, 0.02, '%d events:\n%s' % (len(ev), '\n'.join('%s (%d)' % (a.strftime('%b %d'), m) for a, b, m in ev[:10])),
                           transform=axs[r, 2].transAxes, ha='left', va='bottom', fontsize=8)
            axs[r, 0].legend(loc='lower left', fontsize=9)
        for c, lab in enumerate(['u [cm s$^{-1}$]   ← IN | OUT →'] * 2 + ['$\\sigma_0$ [kg m$^{-3}$]']):
            axs[-1, c].set_xlabel(lab)
        axs[0, 0].set_title('Cross-channel averaged profile', fontweight='bold')
        axs[0, 1].set_title('Depth average at each face', fontweight='bold')
        axs[0, 2].set_title('Cross-channel mean $\\sigma_0$', fontweight='bold')
        how = {'far': 'largest area-weighted RMS distance of the %s shape (u - Ubar) from the row mean' % f,
               'near': 'smallest area-weighted RMS distance of the %s shape (u - Ubar) from the row mean (most common)' % f,
               'mode': 'highest density of the %s shape in (PC1, PC2) (most common)' % f}.get(
            s, '%s tail of the %s shape %s PC (+ = upper / north water more INTO the cove)' % (s[-1], f, s[:4]))
        fig.suptitle('pc_lj (Lovejoy), %d, %s: the %s %.0f%% of each row by %s\n%s'
                     % (YEAR, GTAGEX, 'most common' if s in ['near', 'mode'] else 'most extreme', 100 * XFRAC, f, how), fontsize=13)
        out_fn = out_dir / ('%s_%s_%s.png' % ('common' if s in ['near', 'mode'] else 'extremes', f, s.replace(' ', '').replace('+', 'pos').replace('-', 'neg')))
        fig.savefig(out_fn, dpi=200, transparent=True)
        plt.close(fig)
        print('saved %s' % out_fn)

# characteristics_[field].png: median percentile heat map
for f in FIELDS:
    sub = CH[CH['field'] == f]
    cols = [(r_, s) for r_ in sub['row'].unique() for s in SETS]
    M = np.array([[sub[(sub['row'] == r_) & (sub['set'] == s) & (sub['driver'] == nm)]['pct'].iloc[0] for r_, s in cols] for nm in DNAME])
    O = np.array([[sub[(sub['row'] == r_) & (sub['set'] == s) & (sub['driver'] == nm)]['outside'].iloc[0] for r_, s in cols] for nm in DNAME])
    fig, ax = plt.subplots(figsize=(24, 13), layout='constrained')
    im = ax.imshow(M, cmap='RdBu_r', vmin=0, vmax=100, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, '%.0f' % M[i, j], ha='center', va='center', fontsize=8,
                    fontweight='bold' if O[i, j] else 'normal', color='k' if O[i, j] else '0.55')
    ax.set_yticks(range(len(DNAME)), DNAME)
    ax.set_xticks(range(len(cols)), [s for _, s in cols], rotation=90)
    for j in range(len(SETS), len(cols), len(SETS)):
        ax.axvline(j - 0.5, color='k', lw=2)
    for k_, r_ in enumerate(sub['row'].unique()):
        ax.text(k_ * len(SETS) + (len(SETS) - 1) / 2, -0.8, r_.upper(), ha='center', va='bottom', fontweight='bold')
    fig.colorbar(im, ax=ax, label='median percentile of the set within its row (50 = typical)', shrink=0.6)
    ax.set_title('pc_lj (Lovejoy), %d, %s: what characterises the most extreme and most common %.0f%% of %s shapes?\n'
                 'bold = outside 2.5-97.5%% of %d random sets of the same size (tides treated as independent)\n\n'
                 % (YEAR, GTAGEX, 100 * XFRAC, f, NDRAW), fontsize=13)
    out_fn = out_dir / ('characteristics_%s.png' % f)
    fig.savefig(out_fn, dpi=200, transparent=True)
    plt.close(fig)
    print('saved %s' % out_fn)

# timeline_[field].png: PC1, PC2 through the year and PC1 vs PC2, extremes marked
for f in FIELDS:
    fig, axs = plt.subplots(len(ROWS), 3, figsize=(22, 18), layout='constrained', gridspec_kw={'width_ratios': [2, 2, 1]})
    for r, row in enumerate(ROWS):
        d, E, X = D[row], EOF[(row, f)], XS[(row, f)]
        tt = ot[d['hrs']]
        for c in range(2):
            ax = axs[r, c]
            ax.plot(tt, E['pc'][c], '.', color='0.6', ms=3)
            for s in ['EOF%d +' % (c + 1), 'EOF%d -' % (c + 1)]:
                ax.plot(tt[X[s]], E['pc'][c][X[s]], 'o', color=SCOL[s], ms=6, label=s)
            ax.plot(tt[X['far']], E['pc'][c][X['far']], 'o', mfc='none', mec=SCOL['far'], ms=10, mew=1.5, label='far')
            for s in ['near', 'mode']:
                ax.plot(tt[X[s]], E['pc'][c][X[s]], 's', color=SCOL[s], ms=5, label=s)
            ax.axhline(0, color='0.4', lw=0.8)
            ax.grid(color='lightgray', linestyle='--', alpha=0.5)
            ax.set_ylabel('%s %s\nPC%d (%.0f%%)' % (row[1].upper(), row[0].upper(), c + 1, 100 * E['var_exp'][c]), fontweight='bold' if c == 0 else 'normal')
            ax.legend(loc='upper left', fontsize=9, ncols=5)
        ax = axs[r, 2]
        ax.plot(E['pc'][0], E['pc'][1], '.', color='0.6', ms=3)
        for s in SETS[1:]:
            ax.plot(E['pc'][0][X[s]], E['pc'][1][X[s]], 's' if s in ['near', 'mode'] else 'o', color=SCOL[s], ms=5)
        ax.plot(E['pc'][0][X['far']], E['pc'][1][X['far']], 'o', mfc='none', mec=SCOL['far'], ms=9, mew=1.5)
        ax.axhline(0, color='0.4', lw=0.8)
        ax.axvline(0, color='0.4', lw=0.8)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(color='lightgray', linestyle='--', alpha=0.5)
    fig.suptitle('pc_lj (Lovejoy), %d, %s: %s shape PCs of every tide (grey), the extreme and the most common (squares) %.0f%% sets'
                 % (YEAR, GTAGEX, f, 100 * XFRAC), fontsize=13)
    out_fn = out_dir / ('timeline_%s.png' % f)
    fig.savefig(out_fn, dpi=200, transparent=True)
    plt.close(fig)
    print('saved %s' % out_fn)

# extremes.csv: one line per extreme tide
XT = []
for row in ROWS:
    d = D[row]
    for f in FIELDS:
        E = EOF[(row, f)]
        for s in SETS:
            for rank, i in enumerate(XS[(row, f)][s]):
                r_ = {'time': ot[d['hrs'][i]], 'row': '%s %s' % (row[1], row[0]), 'field': f, 'set': s, 'rank': rank + 1,
                      'dist': E['dist'][i], 'density': E['dens'][i], 'PC1': E['pc'][0][i], 'PC2': E['pc'][1][i], 'Q': Q[d['hrs'][i]]}
                r_.update({nm: DRV[nm][d['hrs'][i]] for nm in DNAME})
                XT.append(r_)
pd.DataFrame(XT).to_csv(out_dir / 'extremes.csv', index=False, float_format='%.4g')
print('saved %s' % (out_dir / 'extremes.csv'))
