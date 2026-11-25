"""
Daily table of the inner-cove eddy's aspects (from 20261117_pc_eddy_detect.py) and of candidate drivers, every driver
computed here from the model extractions. Fitted in 20261117_pc_eddy_drivers_fit.py.

UNIT: the day. The eddy fields are Godin-filtered and an event lasts days (median ~4), so hourly rows would be ~24
near-copies each; every driver is Godin-filtered (or is daily already) and then averaged over the same day.

STEP 1  RESPONSES, from eddy_hourly.nc, for the bins ZB (where the eddy is present >= ~7% of hours):
          pres_[z]   share of the day's hours present (in a kept event)
          Umax_[z]   daily mean U_max, ALL hours (strength is defined every hour, so no selection on presence)
          Ro_[z]     daily mean zeta_c / f, all hours
        WHERE (present hour-bins of the day pooled, days with >= MIN_PH of them): x_c, y_c [m, x from the pc_lp mouth,
          y from 48.23 N], r_max [m]
        VERTICAL (column): z_pres = presence-weighted mean depth of the bins, thick = sum of presence x 2 m [m],
          z_Umax = depth of the bin with the largest daily U_max
STEP 2  DRIVERS
          D2, D1        semidiurnal / diurnal tide: utide fit to the inner-cove mean sea level (box zeta), each band
                        reconstructed and its Hilbert envelope [m]
          drho          inner-cove mean sigma0 bottom - top s-level
          z_pyc         inner-cove mean depth of the max d(sigma0)/dz below the surface [m]
          drhox_top/bot sigma0 pc_lp - pc_cp, top / bottom s-level, width-mean (in-out density difference)
          tau_al/cr     cove-mean surface stress (pc_cove_mix sustr, svstr), + toward the head / the north shore;
                        and _prior = mean of the PRIOR_D days before
          shflux        cove-mean net surface heat flux
          skagit        Skagit flow, lagged SKAGIT_LAG days (wb1 river forcing table)
          U_tide_max    pc_lj (box xi_u 23): daily max of the hourly max |u| over the section cells (faces x s-levels)
          U_res_max     pc_lj: daily mean of the max |Godin u| over the section cells
          sin_doy, cos_doy  season
STEP 3  save eddy_drivers_daily.csv

INPUT  LO_output/DM_outs/20261117_pc_eddy_detect/eddy_hourly.nc
       LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc, pc_cove_mix_[ds0]_[ds1].nc
       LO_output/DM_outs/20260806_river_hydrographs/daily_flow.csv
OUTPUT LO_output/DM_outs/20261117_pc_eddy_drivers/eddy_drivers_daily.csv

run 20261117_pc_eddy_drivers.py
"""
import gsw
import numpy as np
import pandas as pd
import utide
import xarray as xr
from scipy.signal import hilbert

from lo_tools import Lfun, zfun

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
ZB = [1, 3, 5, 7, 9, 11]                  # bin centres used [m]
MIN_PH = 12                               # min present hour-bins in a day for the WHERE responses
AXIS = (-0.9789, -0.2043)                 # cove axis, mouth -> head
PRIOR_D = 2                               # days averaged before the day for the prior wind
SKAGIT_LAG = 3                            # days
CHUNK = 720

gridname, tag, ex_name = GTAGEX.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
box_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_%s_%s.nc' % (DS0, DS1))
mix_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_mix_%s_%s.nc' % (DS0, DS1))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261117_pc_eddy_drivers'
Lfun.make_dir(out_dir)
LON_CP = pd.read_pickle(Ldir['LOo'] / 'section_lines' / 'pc_cp.p').x.mean()

# ------------------------------------------------------------------ STEP 1: responses
de = xr.open_dataset(Ldir['LOo'] / 'DM_outs' / '20261117_pc_eddy_detect' / 'eddy_hourly.nc').sel(z=ZB)
th = pd.DatetimeIndex(de.time.values)
pe = de.present_ev.values == 1                                  # (t, z)
R = pd.DataFrame(index=th)
for j, z in enumerate(ZB):
    R['pres_%d' % z] = pe[:, j].astype(float)
    R['Umax_%d' % z] = 100 * de.U_max.values[:, j]              # cm/s
    R['Ro_%d' % z] = de.zeta_c_f.values[:, j]
for vn, sc in [('x_c', 1), ('y_c', 1), ('r_max', 1)]:
    R[vn] = np.where(pe, de[vn].values * sc, 0.0).sum(axis=1)        # summed here, divided by the count below
R['n_ph'] = pe.sum(axis=1)
D = R.resample('D').sum(min_count=1)
nh = R.resample('D').size()
for c in [c for c in R if c.startswith(('pres_', 'Umax_', 'Ro_'))]:
    D[c] = R[c].resample('D').mean()
for vn in ['x_c', 'y_c', 'r_max']:
    D[vn] = np.where(D['n_ph'] >= MIN_PH, D[vn] / D['n_ph'], np.nan)
P = D[['pres_%d' % z for z in ZB]].to_numpy()
zz = np.array(ZB, float)
D['z_pres'] = np.where(P.sum(axis=1) > 0, (P * zz).sum(axis=1) / np.where(P.sum(axis=1) > 0, P.sum(axis=1), 1), np.nan)
D['thick'] = 2.0 * P.sum(axis=1)
D['z_Umax'] = zz[np.argmax(D[['Umax_%d' % z for z in ZB]].to_numpy(), axis=1)]
D = D[nh == 24].drop(columns='n_ph')                            # whole days only
print('responses: %d days, WHERE on %d days' % (len(D), D['x_c'].notna().sum()))

# ------------------------------------------------------------------ STEP 2: drivers
ds = xr.open_dataset(box_fn)
ot = pd.DatetimeIndex(ds.ocean_time.values)
lon, lat = ds['lon_rho'].values, ds['lat_rho'].values
mask = ds['mask_rho'].values == 1
inner = mask & (lon < LON_CP)                                   # inner-cove rho cells (the eddy's search region)
ji, ii = np.nonzero(inner)
FACE = {}                                                       # pc_cp (box xi_u 15), pc_lp (30): wet faces, widths
for sn, X in [('pc_cp', 15), ('pc_lp', 30)]:
    jj = np.flatnonzero(mask[:, X] & mask[:, X + 1])
    FACE[sn] = (jj, X, (1 / ds['pn'].values[jj][:, [X, X + 1]]).mean(axis=1))
H = {k: np.full(len(ot), np.nan) for k in ['zeta', 'drho', 'z_pyc', 'drhox_top', 'drhox_bot']}


def sig0(salt, temp, lo, la):
    SA = gsw.SA_from_SP(salt, 0, lo, la)
    return gsw.sigma0(SA, gsw.CT_from_pt(SA, temp))


for i0 in range(0, len(ot), CHUNK):
    sl = slice(i0, min(i0 + CHUNK, len(ot)))
    salt, temp, zr = ds['salt'][sl].values, ds['temp'][sl].values, ds['z_rho'][sl].values
    zeta = ds['zeta'][sl].values
    s = sig0(salt[:, :, ji, ii], temp[:, :, ji, ii], lon[ji, ii], lat[ji, ii])    # (t, k, cells)
    z = zr[:, :, ji, ii]
    H['zeta'][sl] = zeta[:, ji, ii].mean(axis=1)
    H['drho'][sl] = (s[:, 0] - s[:, -1]).mean(axis=1)
    g = (s[:, :-1] - s[:, 1:]) / (z[:, 1:] - z[:, :-1])        # -d(sigma0)/dz between levels, > 0 stable
    zm = zeta[:, ji, ii][:, None] - 0.5 * (z[:, 1:] + z[:, :-1])  # depth below the surface of each level pair
    H['z_pyc'][sl] = np.take_along_axis(zm, np.argmax(g, axis=1)[:, None], axis=1)[:, 0].mean(axis=1)
    st = {}
    for sn, (jj, X, w) in FACE.items():
        lo, la = lon[jj][:, [X, X + 1]], lat[jj][:, [X, X + 1]]
        sg = sig0(salt[:, [0, -1]][:, :, jj][..., [X, X + 1]], temp[:, [0, -1]][:, :, jj][..., [X, X + 1]], lo, la).mean(axis=-1)
        st[sn] = (sg * w).sum(axis=-1) / w.sum()               # (t, bot/top)
    H['drhox_bot'][sl] = st['pc_lp'][:, 0] - st['pc_cp'][:, 0]
    H['drhox_top'][sl] = st['pc_lp'][:, 1] - st['pc_cp'][:, 1]
    print('  read %s to %s' % (ot[sl.start].date(), ot[sl.stop - 1].date()), flush=True)
ds.close()

# pc_lj: tidal max and residual max speed over the section cells
ul = xr.open_dataset(box_fn)['u'].isel(xi_u=23, eta_u=np.arange(14, 6, -1)).values.reshape(len(ot), -1)   # (t, k*p)
ul = ul[:, np.isfinite(ul).all(axis=0)]                         # wet cells
Ut = pd.Series(np.abs(ul).max(axis=1), index=ot).resample('D').max()
Ur = pd.Series(np.abs(zfun.lowpass(ul.astype(float), f='godin')).max(axis=1), index=ot).resample('D').mean()

# tide: utide on the inner-cove mean sea level, band reconstructions, Hilbert envelopes
coef = utide.solve(ot, H['zeta'], lat=lat[inner].mean(), method='ols', conf_int='none', trend=False, verbose=False)
coef['mean'] = 0.0                                              # reconstruct would add the mean back
for band, cons in [('D2', ['M2', 'S2', 'N2', 'K2']), ('D1', ['K1', 'O1', 'P1', 'Q1'])]:
    H[band] = np.abs(hilbert(utide.reconstruct(ot, coef, constit=cons, verbose=False).h))

# wind stress and heat flux, cove-mean (pc_cove_mix)
dm = xr.open_dataset(mix_fn)
assert (pd.DatetimeIndex(dm.ocean_time.values) == ot).all()
cove = dm['mask_rho'].values == 1
cove[:, 31:] = False                                            # cove water: rho columns <= 30
mu = cove[:, :-1] & cove[:, 1:]
mv = cove[:-1, :] & cove[1:, :]
tx = np.nanmean(dm['sustr'].values[:, mu], axis=1)
ty = np.nanmean(dm['svstr'].values[:, mv], axis=1)
H['shflux'] = dm['shflux'].values[:, cove].mean(axis=1)
dm.close()
H['tau_al'], H['tau_cr'] = tx * AXIS[0] + ty * AXIS[1], tx * AXIS[1] - ty * AXIS[0]

# Godin every hourly driver (except the tidal envelopes, which are already smooth), then daily means
X = pd.DataFrame({k: (v if k in ['D2', 'D1'] else zfun.lowpass(v, f='godin')) for k, v in H.items() if k != 'zeta'}, index=ot)
X = X.resample('D').mean()
for k in ['tau_al', 'tau_cr']:
    X[k + '_prior'] = X[k].shift(1).rolling(PRIOR_D).mean()      # mean of the PRIOR_D days before
qf = pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20260806_river_hydrographs' / 'daily_flow.csv', parse_dates=['date']).set_index('date')['skagit']
X['U_tide_max'], X['U_res_max'] = 100 * Ut, 100 * Ur              # cm/s
X['skagit'] = qf.reindex(X.index - pd.Timedelta(days=SKAGIT_LAG)).to_numpy()
doy = X.index.dayofyear.to_numpy()
X['sin_doy'], X['cos_doy'] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)

# ------------------------------------------------------------------ STEP 3: save
T = D.join(X, how='inner')
T.index.name = 'date'
T.to_csv(out_dir / 'eddy_drivers_daily.csv', float_format='%.6g')
print('saved %s: %d days' % (out_dir / 'eddy_drivers_daily.csv', len(T)))
print(T[list(X.columns)].describe().T[['mean', 'std', 'min', 'max']].to_string(float_format=lambda v: '%.4g' % v))
