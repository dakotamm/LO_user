"""
Builds the wb1_t0_xn11abbur00 entries for the two cas7 comparison plots from extractions already on the Mac
(no apogee run), in exactly the pickle formats those scripts write, then you re-plot them with -plot_only:
    20261116_pc_cas7_bottom_DO_cove.py   (cove bottom DO / bottom T / air + wind)  -> pc_cove_compare.png
    20261117_pc_saratoga_strat_DO.py     (pc_west and M5 drho + bottom DO)         -> compare.png
    python 20261117_wb1_pc_from_boxes.py

SOURCES (LO_output/extract/wb1_t0_xn11abbur00/)
  box/pc_cove_o2_2024.01.01_2025.12.31.nc        hourly oxygen, salt, temp, z_rho; east edge = lon -122.652
  box/pc_cove_air_surf_2024.01.01_2025.12.31.nc  hourly Tair, same box
  moor/pc4/M5_2024.01.02_2025.12.30.nc           daily lowpass (Godin) columns at M5, with z_rho
  DM_outs/20260806_wind/wind_hourly_atm00_*.p    hourly 'pc'-polygon u, v, |tau| (Large & Pond per cell, then
                                                 averaged; rho_air 1.22), from the same atm00 forcing the run used
DAILY   ocean: Godin 24-24-25 on the hourly box (zfun.lowpass), sampled at 12:00 UTC = what lowpassed.nc holds
        (cas7_t0_x4b is lowpassed.nc too). Days 2024.01.02-2025.12.30 (the filter needs the day either side).
        air: plain UTC-day mean of the hourly fields, as the cas7 cove runs average the hourly atm forcing.
COVE    same rule as the cove script: wet cells with lon < -122.652, lat 48.20-48.26, connected to the seed
        (-122.70878, 48.22710); h, pm, pn, mask from the box (= the run output). Bottom = s_rho index 0.
WIND    the wind pickle stores |tau| per region but not its components, so taux, tauy = |tau| * (u, v) / |(u, v)|
        with the region-mean wind direction. Region = the 'pc' polygon (pc.p), not the cove cells. uw_avg/vw_avg =
        daily-mean u_pc, v_pc. u10, v10, spd = daily means of u_pc, v_pc, spd_pc.
POINTS  pc_west = the box cell nearest (-122.70878, 48.22710) (wb1 j=217, i=47, h 17.2 m); its salt/temp/oxygen
        columns are Godin-filtered hourly. M5 = the pc4 mooring (wb1 j=227, i=94, h 37.4 m). drho as both scripts.
OUT     DM_outs/20261116_pc_cas7_bottom_DO_cove/wb1_t0_xn11abbur00_pc_cove_2024.01.02_2025.12.30.p
        DM_outs/20261117_pc_saratoga_strat_DO/wb1_t0_xn11abbur00_pc_west-M5_2024.01.02_2025.12.30.p
"""

import pickle
import numpy as np
import pandas as pd
import netCDF4 as nc
import gsw
from scipy.ndimage import label

from lo_tools import Lfun, zfun

GTX = 'wb1_t0_xn11abbur00'
LON_MOUTH = -122.652
SEED = (-122.70878, 48.22710)
BOX_LAT = (48.20, 48.26)
DO_MMOL_TO_MGL = 32.0 / 1000.0
BLOCK = 1000                       # hours read at a time (one chunk = one hour of the whole box)
AIR_COLS = ['uw_avg', 'vw_avg', 'atm_n', 'Tair', 'u10', 'v10', 'spd', 'taux', 'tauy', 'tau']

Ldir = Lfun.Lstart(gridname='wb1', tag='t0', ex_name='xn11abbur00')
ex_dir = Ldir['LOo'] / 'extract' / GTX
o2_fn = ex_dir / 'box' / 'pc_cove_o2_2024.01.01_2025.12.31.nc'
air_fn = ex_dir / 'box' / 'pc_cove_air_surf_2024.01.01_2025.12.31.nc'
m5_fn = ex_dir / 'moor' / 'pc4' / 'M5_2024.01.02_2025.12.30.nc'
wind_fn = Ldir['LOo'] / 'DM_outs' / '20260806_wind' / 'wind_hourly_atm00_2024.01.01_2025.12.31.p'
tag = '2024.01.02_2025.12.30'

# ---- cove cells and pc_west, on the box grid ----
ds = nc.Dataset(o2_fn)
lon, lat = ds['lon_rho'][:].data, ds['lat_rho'][:].data
mask, h, pm, pn = (ds[v][:].data for v in ['mask_rho', 'h', 'pm', 'pn'])
t_h = pd.to_datetime(nc.num2date(ds['ocean_time'][:], ds['ocean_time'].units,
                                 only_use_cftime_datetimes=False, only_use_python_datetimes=True))
wet = (mask == 1) & (lon < LON_MOUTH) & (lat > BOX_LAT[0]) & (lat < BOX_LAT[1])
lab, _ = label(wet)
js, is_ = np.unravel_index(np.argmin((lon - SEED[0])**2 + (lat - SEED[1])**2), lon.shape)
cove = lab == lab[js, is_]
J, I = np.where(cove)
k_seed = int(np.where((J == js) & (I == is_))[0][0])
area = 1 / (pm[J, I] * pn[J, I])
dpc = ((lon - SEED[0]) * np.cos(np.deg2rad(SEED[1])))**2 + (lat - SEED[1])**2
dpc[mask != 1] = np.inf
jp, ip = np.unravel_index(np.argmin(dpc), dpc.shape)
print(f'cove   : {len(J)} cells  h {h[J, I].min():.1f}-{h[J, I].max():.1f} m  area {area.sum() / 1e6:.2f} km2')
print(f'pc_west: box j={jp} i={ip}  lon {lon[jp, ip]:.5f}  lat {lat[jp, ip]:.5f}  h {h[jp, ip]:.1f} m')
# 317 cells = the count the cove script finds on the full wb1_t0 grid, so the box holds the whole cove

# ---- hourly reads, in blocks ----
NT, N = len(t_h), ds.dimensions['s_rho'].size
do_h, tb_h = np.full((NT, len(J)), np.nan), np.full((NT, len(J)), np.nan)
col = {v: np.full((NT, N), np.nan) for v in ['salt', 'temp', 'oxygen', 'z_rho']}
for t0 in range(0, NT, BLOCK):
    sl = slice(t0, min(t0 + BLOCK, NT))
    for v in ['oxygen', 'salt', 'temp', 'z_rho']:
        a = np.ma.filled(ds[v][sl].astype(float), np.nan)      # (t, N, eta, xi)
        col[v][sl] = a[:, :, jp, ip]
        if v == 'oxygen':
            do_h[sl] = a[:, 0, J, I]
        elif v == 'temp':
            tb_h[sl] = a[:, 0, J, I]
    print(f'  read {sl.stop} / {NT} hours', flush=True)
ds.close()

# ---- Godin, sampled at 12:00 UTC ----
noon = (t_h.hour == 12) & (t_h >= '2024-01-02') & (t_h <= '2025-12-30 12:00')
god = lambda a: zfun.lowpass(a, f='godin')[noon]
t_d = t_h[noon]
do_d, tb_d = god(do_h) * DO_MMOL_TO_MGL, god(tb_h)
col_d = {v: god(a) for v, a in col.items()}

# ---- air: Tair over cove cells (box), wind from the 'pc' polygon reduction; UTC-day means ----
with nc.Dataset(air_fn) as da:
    Tair_h = np.nanmean(np.ma.filled(da['Tair'][:].astype(float), np.nan)[:, J, I], axis=1)
W = pickle.load(open(wind_fn, 'rb'))['W']
w = pd.DataFrame({'u': W['u_pc'], 'v': W['v_pc'], 'spd': W['spd_pc'], 'tau': W['tau_pc']})
U = np.sqrt(w.u**2 + w.v**2).replace(0, np.nan)
w['taux'], w['tauy'] = w.tau * w.u / U, w.tau * w.v / U
w['Tair'] = pd.Series(Tair_h, index=t_h).reindex(w.index)
dd = w.groupby(w.index.normalize()).agg(['mean', 'count'])
days = t_d.normalize()
air = pd.DataFrame(index=t_d)
air['uw_avg'], air['vw_avg'] = dd[('u', 'mean')].reindex(days).values, dd[('v', 'mean')].reindex(days).values
air['atm_n'] = dd[('u', 'count')].reindex(days).values
for c in ['Tair', 'taux', 'tauy', 'tau', 'spd']:
    air[c] = dd[(c, 'mean')].reindex(days).values
air['u10'], air['v10'] = air['uw_avg'], air['vw_avg']
air['atm'] = 'atm00'
air = air[['atm'] + AIR_COLS]

# ---- cove pickle (format of 20261116_pc_cas7_bottom_DO_cove.py) ----
out_cove = Ldir['LOo'] / 'DM_outs' / '20261116_pc_cas7_bottom_DO_cove'
Pc = {'gtagex': GTX, 'time': t_d.values, 'do': do_d, 'tb': tb_d, 'air': air,
      'frc_root': str(wind_fn) + ' + ' + air_fn.name,
      'j': J, 'i': I, 'k_seed': k_seed, 'lon': lon[J, I], 'lat': lat[J, I], 'h': h[J, I], 'area': area,
      'box_lon': lon, 'box_lat': lat, 'box_mask': mask, 'cj': J, 'ci': I, 'lon_mouth': LON_MOUTH,
      'source': 'built from boxes by 20261117_wb1_pc_from_boxes.py'}
fn = out_cove / f'{GTX}_pc_cove_{tag}.p'
pickle.dump(Pc, open(fn, 'wb'))
print(f'saved  : {fn}')

# ---- point pickle (format of 20261117_pc_saratoga_strat_DO.py) ----
def station(salt, temp, oxy_mgL, z_rho, zeta, j, i, lo, la, hh):
    SA = gsw.SA_from_SP(salt, -z_rho, lo, la)
    return dict(j=int(j), i=int(i), lon=float(lo), lat=float(la), h=float(hh), zeta=zeta, z_rho=z_rho, salt=salt,
                temp=temp, oxygen=oxy_mgL, sigma0=gsw.sigma0(SA, gsw.CT_from_pt(SA, temp)))

st = {'pc_west': station(col_d['salt'], col_d['temp'], col_d['oxygen'] * DO_MMOL_TO_MGL, col_d['z_rho'],
                         np.full(len(t_d), np.nan), 217, 47, lon[jp, ip], lat[jp, ip], h[jp, ip])}
with nc.Dataset(m5_fn) as dm:
    t_m = pd.to_datetime(nc.num2date(dm['ocean_time'][:], dm['ocean_time'].units,
                                     only_use_cftime_datetimes=False, only_use_python_datetimes=True))
    if not (len(t_m) == len(t_d) and (t_m.normalize() == t_d.normalize()).all()):
        raise SystemExit('*** M5 mooring days do not match the box days')
    g = lambda v: np.ma.filled(dm[v][:].astype(float), np.nan)
    st['M5'] = station(g('salt'), g('temp'), g('oxygen') * DO_MMOL_TO_MGL, g('z_rho'), g('zeta'),
                       227, 94, float(dm['lon_rho'][:]), float(dm['lat_rho'][:]), float(dm['h'][:]))
out_pt = Ldir['LOo'] / 'DM_outs' / '20261117_pc_saratoga_strat_DO'
Pp = {'gtagex': GTX, 'lt': 'lowpass', 'time': t_d.values, 'st': st,
      'source': 'built from boxes + pc4 mooring by 20261117_wb1_pc_from_boxes.py'}
fn = out_pt / f'{GTX}_pc_west-M5_{tag}.p'
pickle.dump(Pp, open(fn, 'wb'))
print(f'saved  : {fn}')
for sn, s in st.items():
    drho = s['sigma0'][:, 0] - s['sigma0'][:, -1]
    print(f'{sn:7s}: h {s["h"]:.1f} m  bottom DO min {np.nanmin(s["oxygen"][:, 0]):.2f}  days<2 '
          f'{int((s["oxygen"][:, 0] < 2).sum())} of {len(t_d)}  drho mean {np.nanmean(drho):.2f}')
print(f'cove   : days with any cell < 2 mg/L {int((np.nanmin(do_d, axis=1) < 2).sum())} of {len(t_d)}  '
      f'Tair mean {air.Tair.mean():.1f}  tau mean {air.tau.mean():.3f}')
