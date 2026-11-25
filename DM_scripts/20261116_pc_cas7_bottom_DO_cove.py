"""
Bottom DO and bottom temperature in every Penn Cove cell of cas7_t1_x11ab (Parker's run on apogee /dat1), plus
cove air temperature and wind stress, daily, 2012.10.07-2025.06.12: is there any hypoxia, and what goes with it?
Companion to 20261116_pc_cas7_bottom_DO.py (one point; that cell, j=882 i=557, is one of the cove cells here).
Also run on cas7_t0_x4b (0% burial; x11ab buries 50%), which has hourly his + lowpassed.nc but no ocean_avg: -lt lowpass.

Check the forcing path first (Parker's atm forcing for cas7; the run's forcing_list.csv says atm02):
    ls /dat1/parker/LO_output/forcing/cas7/f2012.10.07 /dat1/parker/LO_output/forcing/cas7/f2025.06.12
Run on apogee (-ro 1 = Ldir['roms_out1'] = /dat1/parker/LO_roms on dm_apogee):
    nohup python 20261116_pc_cas7_bottom_DO_cove.py -gtx cas7_t1_x11ab -ro 1 -0 2012.10.07 -1 2025.06.12 -frc_root /dat1/parker/LO_output/forcing/cas7 > pc_cas7_bottom_DO_cove.log 2>&1 &
  Check on it with: tail pc_cas7_bottom_DO_cove.log
cas7_t0_x4b (output in Parker's /dat2 = Ldir['roms_out3'] on dm_apogee, through 2019.12.31; forcing_list.csv says atm00):
    nohup python 20261116_pc_cas7_bottom_DO_cove.py -gtx cas7_t0_x4b -ro 3 -0 2012.10.07 -1 2019.12.31 -lt lowpass -atm atm00 -frc_root /dat1/parker/LO_output/forcing/cas7 > pc_cas7_x4b_bottom_DO_cove.log 2>&1 &
wb1_t0_xn11abbur00 (0% burial nest; lowpassed.nc in our /dat2 = roms_out2; atm00 forcing in our own LO_output, the
-frc_root default). Same mask rule on the wb1 grid, so the cove has far more (smaller) cells than cas7's 38:
    nohup python 20261116_pc_cas7_bottom_DO_cove.py -gtx wb1_t0_xn11abbur00 -ro 2 -0 2024.01.01 -1 2025.12.31 -lt lowpass -atm atm00 > pc_cove_wb1.log 2>&1 &
RESUME  each day is appended to <out_name>_partial.csv (flushed every 100 files). If the run dies, rerun the same
  command: days already in the partial file are skipped (a truncated last line is dropped). The final pickle is
  written, and the partial removed, at the end.
Re-plot from the saved pickle (e.g. on the Mac after copying it over):
    python 20261116_pc_cas7_bottom_DO_cove.py -gtx cas7_t1_x11ab -0 2012.10.07 -1 2025.06.12 -plot_only True

COVE    wet cas7 rho cells west of LON_MOUTH = -122.652 (the wb1 pc_lp mouth line, u-face i=67/68), connected to the
        seed (-122.70878, 48.22710) = the single-point cell j=882, i=557. 38 cells, h 12.6-26.6 m.
        Grid (lon, lat, mask, h, pm, pn) from the first ocean file, else that day's ocean_his_0002.nc, else grid.nc.
OCEAN   -lt average: f<date>/ocean_avg_0001.nc (UTC-day average); -lt lowpass: f<date>/lowpassed.nc (one Godin-
        filtered record per day; stamp taken from the file). One hyperslab per variable over the cove box:
        oxygen, temp at s_rho index 0 (bottom) -> per cell; Uwind, Vwind -> cove mean (daily-mean wind, kept as a
        fallback: the stress of a daily-mean wind is not the daily-mean stress). oxygen mmol m-3 * 32/1000 = mg/L.
AIR     the avg file has no Tair or surface stress (Aout F in BLANK.in), so they come from the hourly atm forcing
        <frc_root>/f<date>/<atm>/Tair.nc (frc_root default: Ldir['LOo']/forcing/<gridname>), Uwind.nc, Vwind.nc (first of atm02, atm01, atm00 that exists, or only
        -atm if given; recorded).
        Each file's own 24 hours are kept (record 24 = hour 0 of the next day). Cove mean over the 38 cells.
        Stress per cell and hour with Large & Pond (1981) Cd, rho_air 1.22 (as 20260806_wind_reduce.py), then
        averaged: taux, tauy, |tau|. ROMS computes its own stress with COARE from the same winds, so this is a proxy
        with the right form and magnitude. Uwind/Vwind are already true east/north. Missing forcing -> NaN.
RIVER   Skagit from 20261116_skagit_flow.py (Mac; read here if its csv exists, for the stack plot only).
MISSING days with no ocean file are skipped and counted. Uwind/Vwind are NaN if the ocean file lacks them.
OUT     LO_output/DM_outs/20261116_pc_cas7_bottom_DO_cove/
        <out_name>.p    dict: time, do and tb (Nt x Ncell), daily air/wind DataFrame, cells, box
        <out_name>.csv  daily cove DO min/mean, hypoxic area %, bottom T mean, point-cell DO/T, air and wind
        <out_name>_ts.png, <out_name>_map.png, <out_name>_stack.png;  pc_cove_compare.png  every run found, overlaid
"""

import argparse
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import netCDF4 as nc
from scipy.ndimage import label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun, zfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', type=str, default='cas7_t1_x11ab')
parser.add_argument('-ro', '--roms_out_num', type=int, default=1)  # 1 = Ldir['roms_out1']
parser.add_argument('-0', '--ds0', type=str, default='2012.10.07')
parser.add_argument('-1', '--ds1', type=str, default='2025.06.12')
parser.add_argument('-lt', type=str, default='average')   # average (ocean_avg_0001) or lowpass (lowpassed.nc)
parser.add_argument('-atm', type=str, default=None)       # force one atm forcing version, e.g. atm00
parser.add_argument('-frc_root', type=str, default=None)   # default Ldir['LOo'] / 'forcing' / gridname
parser.add_argument('-plot_only', type=zfun.boolean_string, default=False)
args = parser.parse_args()

gridname, tag, ex_name = args.gtagex.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
if args.roms_out_num > 0:
    Ldir['roms_out'] = Ldir['roms_out' + str(args.roms_out_num)]

DO_MMOL_TO_MGL = 32.0 / 1000.0
HYP = 2.0                                  # mg/L
RUN_COLORS = {'cas7_t1_x11ab': 'tab:blue', 'cas7_t0_x4b': 'tab:red', 'wb1_t0_xn11abbur00': 'tab:green'}   # same in both compare plots
LON_MOUTH = -122.652                       # wb1 pc_lp line
SEED = (-122.70878, 48.22710)              # single-point cell
BOX_LAT = (48.20, 48.26)
AX = [-122.73, -122.64, 48.215, 48.245]
ATM_LIST = [args.atm] if args.atm else ['atm02', 'atm01', 'atm00']
RHO_A = 1.22
AIR_COLS = ['uw_avg', 'vw_avg', 'atm_n', 'Tair', 'u10', 'v10', 'spd', 'taux', 'tauy', 'tau']

out_dir = Ldir['LOo'] / 'DM_outs' / '20261116_pc_cas7_bottom_DO_cove'
Lfun.make_dir(out_dir)
out_name = args.gtagex + '_pc_cove_' + args.ds0 + '_' + args.ds1
p_fn = out_dir / (out_name + '.p')


def cd_lp(spd):
    """Large & Pond (1981) neutral drag coefficient."""
    return np.where(spd < 10.0, 1.14e-3, (0.49 + 0.065 * spd) * 1e-3)


def read_box(ds, vn, bj, bi, cj, ci, bottom):
    """Cove-cell values of ds[vn] at time 0 (and s_rho 0 if bottom); NaN if the variable is missing."""
    if vn not in ds.variables:
        return np.full(len(cj), np.nan)
    a = ds[vn][0, 0, bj, bi] if bottom else ds[vn][0, bj, bi]
    return np.ma.filled(a.astype(float), np.nan)[cj, ci]


def read_atm(day_dir, bj, bi, cj, ci):
    """Daily cove-mean air fields from the hourly atm forcing, or NaNs. Returns (atm name, dict)."""
    out = {k: np.nan for k in ['atm_n', 'Tair', 'u10', 'v10', 'spd', 'taux', 'tauy', 'tau']}
    for atm in ATM_LIST:
        d = day_dir / atm
        if all((d / (vn + '.nc')).exists() for vn in ['Tair', 'Uwind', 'Vwind']):
            break
    else:
        return '', out
    F = {}
    for vn in ['Tair', 'Uwind', 'Vwind']:
        with nc.Dataset(d / (vn + '.nc')) as ds:
            a = ds[vn][:24, bj, bi]                     # this day's own 24 hours
            F[vn] = np.ma.filled(a.astype(float), np.nan)[:, cj, ci]
    u, v = F['Uwind'], F['Vwind']
    spd = np.sqrt(u**2 + v**2)
    taux, tauy = RHO_A * cd_lp(spd) * spd * u, RHO_A * cd_lp(spd) * spd * v
    out.update(atm_n=u.shape[0], Tair=np.nanmean(F['Tair']), u10=np.nanmean(u), v10=np.nanmean(v),
               spd=np.nanmean(spd), taux=np.nanmean(taux), tauy=np.nanmean(tauy),
               tau=np.nanmean(np.sqrt(taux**2 + tauy**2)))
    return atm, out


if not args.plot_only:
    fn_list = Lfun.get_fn_list(args.lt, Ldir, args.ds0, args.ds1)
    fn_have = [fn for fn in fn_list if fn.exists()]
    print(f'files  : {len(fn_have)} of {len(fn_list)} days have {fn_list[0].name}')
    if len(fn_have) == 0:
        print(f'*** no files under {Ldir["roms_out"] / args.gtagex}')
        sys.exit()
    frc_root = Path(args.frc_root) if args.frc_root else Ldir['LOo'] / 'forcing' / gridname
    print(f'forcing: {frc_root}  (exists: {frc_root.exists()})')

    # ---- cove mask ----
    vns = ['lon_rho', 'lat_rho', 'mask_rho', 'h', 'pm', 'pn']
    for gfn in [fn_have[0], fn_have[0].parent / 'ocean_his_0002.nc', Ldir['grid'] / 'grid.nc']:
        if gfn.exists():
            with nc.Dataset(gfn) as ds:
                if all(vn in ds.variables for vn in vns):
                    G = {vn: ds[vn][:].data for vn in vns}
                    break
    else:
        print('*** no grid variables in the ocean file, ocean_his_0002.nc or grid.nc')
        sys.exit()
    print(f'grid   : {gfn}')
    lon, lat = G['lon_rho'], G['lat_rho']
    wet = (G['mask_rho'] == 1) & (lon < LON_MOUTH) & (lat > BOX_LAT[0]) & (lat < BOX_LAT[1])
    lab, _ = label(wet)
    js, is_ = np.unravel_index(np.argmin((lon - SEED[0])**2 + (lat - SEED[1])**2), lon.shape)
    if lab[js, is_] == 0:
        print('*** seed is not on a wet cove cell')
        sys.exit()
    cove = lab == lab[js, is_]
    J, I = np.where(cove)
    k_seed = int(np.where((J == js) & (I == is_))[0][0])
    bj = slice(J.min() - 1, J.max() + 2)   # box with a one-cell margin, for the map
    bi = slice(I.min() - 1, I.max() + 2)
    cj, ci = J - bj.start, I - bi.start    # cove cells within the box
    area = 1 / (G['pm'][J, I] * G['pn'][J, I])
    print(f'cove   : {len(J)} cells  h {G["h"][J, I].min():.1f}-{G["h"][J, I].max():.1f} m  '
          f'area {area.sum() / 1e6:.2f} km2  box j {bj.start}-{bj.stop - 1} i {bi.start}-{bi.stop - 1}  '
          f'point cell j={js} i={is_} (k={k_seed})')

    # ---- one hyperslab per variable per file, appended to a partial csv so a killed run can resume ----
    part_fn = out_dir / (out_name + '_partial.csv')
    cell_line = '# v2 cells (j,i): ' + ' '.join(f'{j},{i}' for j, i in zip(J, I)) + '\n'
    dcols = [f'do{k}' for k in range(len(J))]
    tcols = [f'tb{k}' for k in range(len(J))]
    cols = ['fdir', 'time_utc'] + dcols + tcols + AIR_COLS + ['atm', 'ok']
    done = set()
    if part_fn.exists():
        with open(part_fn) as f:
            if f.readline() != cell_line:
                print(f'*** {part_fn.name} is for a different cove mask or script version: delete it')
                sys.exit()
        # keep only complete rows (a kill mid-write leaves a short last line, so no ok=1), rewrite, then append
        dfp = pd.read_csv(part_fn, comment='#', on_bad_lines='skip', keep_default_na=False, na_values=['nan'])
        dfp = dfp[pd.to_numeric(dfp['ok'], errors='coerce') == 1]
        with open(part_fn, 'w') as f:
            f.write(cell_line)
            dfp.to_csv(f, index=False)
        done = set(dfp['fdir'])
        print(f'resume : {len(done)} days already in {part_fn.name}')
    else:
        with open(part_fn, 'w') as f:
            f.write(cell_line + ','.join(cols) + '\n')
    fn_todo = [fn for fn in fn_have if fn.parent.name not in done]

    n_noatm = 0
    with open(part_fn, 'a') as f:
        for nn, fn in enumerate(fn_todo):
            with nc.Dataset(fn) as ds:
                t0 = nc.num2date(ds['ocean_time'][0], ds['ocean_time'].units,
                                 only_use_cftime_datetimes=False, only_use_python_datetimes=True)
                do0 = read_box(ds, 'oxygen', bj, bi, cj, ci, True) * DO_MMOL_TO_MGL
                tb0 = read_box(ds, 'temp', bj, bi, cj, ci, True)
                uw = np.nanmean(read_box(ds, 'Uwind', bj, bi, cj, ci, False))
                vw = np.nanmean(read_box(ds, 'Vwind', bj, bi, cj, ci, False))
            atm, A = read_atm(frc_root / fn.parent.name, bj, bi, cj, ci)
            n_noatm += (atm == '')
            vals = list(do0) + list(tb0) + [uw, vw] + [A[k] for k in AIR_COLS[2:]]
            f.write(f'{fn.parent.name},{t0:%Y-%m-%d %H:%M:%S},' + ','.join(f'{v:.4f}' for v in vals)
                    + f',{atm},1\n')
            if nn % 100 == 0:
                f.flush()
                print(f'  {nn} / {len(fn_todo)}  {fn.parent.name}  atm={atm or "MISSING"}', flush=True)
    print(f'days with no atm forcing this run: {n_noatm}')

    df = pd.read_csv(part_fn, comment='#', keep_default_na=False, na_values=['nan'])
    df = df.drop_duplicates('fdir').sort_values('time_utc').reset_index(drop=True)
    air = df[['atm'] + AIR_COLS].copy()
    air.index = pd.to_datetime(df['time_utc'])
    P = {'gtagex': args.gtagex, 'time': pd.to_datetime(df['time_utc']).values,
         'do': df[dcols].values, 'tb': df[tcols].values, 'air': air, 'frc_root': str(frc_root),
         'j': J, 'i': I, 'k_seed': k_seed, 'lon': lon[J, I], 'lat': lat[J, I], 'h': G['h'][J, I], 'area': area,
         'box_lon': lon[bj, bi], 'box_lat': lat[bj, bi], 'box_mask': G['mask_rho'][bj, bi], 'cj': cj, 'ci': ci,
         'lon_mouth': LON_MOUTH}
    pickle.dump(P, open(p_fn, 'wb'))
    part_fn.unlink()
    print(f'saved  : {p_fn}')

# ---- stats ----
P = pickle.load(open(p_fn, 'rb'))
if 'tb' not in P:   # pickle from the DO-only version of this script: no bottom T or air fields
    P['tb'] = np.full_like(P['do'], np.nan)
    P['air'] = pd.DataFrame(np.nan, index=pd.to_datetime(P['time']), columns=AIR_COLS)
    P['air']['atm'] = ''
    P['k_seed'] = int(np.argmin((P['lon'] - SEED[0])**2 + (P['lat'] - SEED[1])**2))
t, do, tb, area, air, ks = pd.to_datetime(P['time']), P['do'], P['tb'], P['area'], P['air'], P['k_seed']
ok = np.isfinite(do)
do_min = np.nanmin(do, axis=1)
do_mean = np.nansum(np.where(ok, do, 0) * area, axis=1) / np.sum(ok * area, axis=1)
hyp_pct = 100 * np.sum((do < HYP) * area, axis=1) / np.sum(ok * area, axis=1)
okt = np.isfinite(tb)
tb_mean = np.nansum(np.where(okt, tb, 0) * area, axis=1) / np.sum(okt * area, axis=1)
summ = pd.DataFrame({'time_utc': t, 'DO_bot_min_mgL': do_min, 'DO_bot_mean_mgL': do_mean, 'hyp_area_pct': hyp_pct,
                     'T_bot_mean_C': tb_mean, 'DO_bot_point_mgL': do[:, ks], 'T_bot_point_C': tb[:, ks]})
for c in ['Tair', 'taux', 'tauy', 'tau', 'u10', 'v10', 'spd', 'uw_avg', 'vw_avg', 'atm']:
    summ[c] = air[c].values
summ.to_csv(out_dir / (out_name + '.csv'), index=False)

print(f'cove bottom DO: min {np.nanmin(do):.2f} mg/L on {t[np.nanargmin(do_min)]:%Y-%m-%d};  '
      f'days with any cell < {HYP:g} mg/L: {int((do_min < HYP).sum())} of {len(t)}')
yr = pd.Series(do_min < HYP, index=t).groupby(t.year)
print('days with any hypoxia, by year:')
print(pd.DataFrame({'hyp_days': yr.sum(), 'days': yr.size(),
                    'min_DO': pd.Series(do_min, index=t).groupby(t.year).min().round(2)}).to_string())
print(f'atm forcing: {air["atm"].replace("", "MISSING").value_counts().to_dict()}')

# ---- time series ----
fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
ax = axs[0]
ax.plot(t, do_mean, '-', color='tab:blue', lw=0.8, label='cove mean (area-weighted)')
ax.plot(t, do_min, '-', color='tab:red', lw=0.8, label='cove min')
ax.axhline(HYP, color='gray', ls=':', lw=1)
ax.set_ylabel('Bottom DO [mg/L]')
ax.set_ylim(bottom=0)
ax.legend(loc='upper right', fontsize=9)
ax.set_title(f'{P["gtagex"]}  Penn Cove bottom DO, {len(P["j"])} cells west of {P["lon_mouth"]}', fontsize=10)
ax = axs[1]
ax.fill_between(t, hyp_pct, color='tab:red', lw=0)
ax.set_ylabel(f'cove area < {HYP:g} mg/L [%]')
ax.set_ylim(0, 100)
ax.set_xlim(t[0], t[-1])
for ax in axs:
    ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(out_dir / (out_name + '_ts.png'), dpi=200, transparent=True)
plt.close(fig)

# ---- maps: fraction of days < 2 mg/L, and min bottom DO, per cell ----
fhyp = np.sum(do < HYP, axis=0) / np.sum(ok, axis=0)
fields = [(fhyp, 'fraction of days < 2 mg/L', 'Reds', 0, max(fhyp.max(), 0.01)),
          (np.nanmin(do, axis=0), 'min bottom DO [mg/L]', 'viridis', 0, 8)]
fig, axs = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for ax, (val, lab_, cmap, v0, v1) in zip(axs, fields):
    F = np.full(P['box_lon'].shape, np.nan)
    F[P['cj'], P['ci']] = val
    ax.pcolormesh(P['box_lon'], P['box_lat'], np.where(P['box_mask'] == 1, np.nan, 1),
                  cmap='Greys', vmin=0, vmax=4, shading='nearest')   # land grey
    cs = ax.pcolormesh(P['box_lon'], P['box_lat'], F, cmap=cmap, vmin=v0, vmax=v1, shading='nearest')
    fig.colorbar(cs, ax=ax, label=lab_)
    ax.axis(AX)
    ax.set_aspect(1 / np.cos(np.deg2rad(48.23)))
    ax.set_xticks([-122.72, -122.69, -122.66])
    ax.set_xlabel('Longitude')
axs[0].set_ylabel('Latitude')
fig.suptitle(f'{P["gtagex"]}  {t[0]:%Y-%m-%d} to {t[-1]:%Y-%m-%d}', fontsize=10)
fig.tight_layout()
fig.savefig(out_dir / (out_name + '_map.png'), dpi=200, transparent=True)
plt.close(fig)

# ---- stack: bottom DO, bottom T, air T, wind stress, Skagit ----
sk_fn = Ldir['LOo'] / 'DM_outs' / '20261116_skagit_flow' / 'skagit_flow_2012.10.07_2025.06.12.csv'
sk = pd.read_csv(sk_fn, index_col=0, parse_dates=True)['skagit_m3s'] if sk_fn.exists() else None
fig, axs = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
ax = axs[0]
ax.plot(t, do_mean, '-', color='tab:blue', lw=0.7, label='cove mean')
ax.plot(t, do_min, '-', color='tab:red', lw=0.7, label='cove min')
ax.axhline(HYP, color='gray', ls=':', lw=1)
ax.set_ylabel('Bottom DO [mg/L]')
ax.set_ylim(bottom=0)
ax.legend(loc='upper right', fontsize=8)
ax.set_title(f'{P["gtagex"]}  Penn Cove ({len(P["j"])} cells), daily', fontsize=10)
ax = axs[1]
ax.plot(t, tb_mean, '-', color='tab:orange', lw=0.7)
ax.set_ylabel('Bottom T, cove mean [°C]')
ax = axs[2]
ax.plot(t, air['Tair'], '-', color='tab:purple', lw=0.7)
ax.set_ylabel('Air T, cove mean [°C]')
ax = axs[3]
ax.plot(t, air['taux'], '-', color='k', lw=0.6, label=r'$\tau_x$ (east)')
ax.plot(t, air['tauy'], '-', color='tab:green', lw=0.6, label=r'$\tau_y$ (north)')
ax.axhline(0, color='gray', lw=0.5)
ax.set_ylabel('Wind stress [Pa]')
ax.legend(loc='upper right', fontsize=8)
ax = axs[4]
if sk is not None:
    ax.plot(sk.index, sk.values, '-', color='tab:blue', lw=0.7)
else:
    ax.text(0.5, 0.5, f'no {sk_fn.name}: run 20261116_skagit_flow.py', transform=ax.transAxes, ha='center')
ax.set_ylabel('Skagit [m$^3$ s$^{-1}$]')
ax.set_xlim(t[0], t[-1])
for ax in axs:
    ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(out_dir / (out_name + '_stack.png'), dpi=200, transparent=True)
plt.close(fig)

# ---- every run in out_dir, overlaid: cove min bottom DO, % cove area < 2 mg/L, cove-mean bottom T ----
runs = sorted(out_dir.glob('*_pc_cove_*.p'))
if len(runs) > 1:
    fig, axs = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    runs = sorted([pickle.load(open(fn, 'rb')) for fn in runs],
                  key=lambda Pr: list(RUN_COLORS).index(Pr['gtagex']) if Pr['gtagex'] in RUN_COLORS else len(RUN_COLORS))   # drawing order: long x11ab first, shorter runs on top
    for Pr, c in zip(runs, ['tab:purple', 'tab:orange', 'tab:brown', 'tab:pink']):
        c = RUN_COLORS.get(Pr['gtagex'], c)
        tr, dor, ar = pd.to_datetime(Pr['time']), Pr['do'], Pr['area']
        okr = np.isfinite(dor)
        lab = f'{Pr["gtagex"]} ({len(Pr["j"])} cells)'
        axs[0].plot(tr, np.nanmin(dor, axis=1), '-', color=c, lw=0.7, label=lab)
        axs[1].plot(tr, 100 * np.sum((dor < HYP) * ar, axis=1) / np.sum(okr * ar, axis=1), '-', color=c, lw=0.7,
                    label=lab)
        if 'tb' in Pr:
            okt_r = np.isfinite(Pr['tb'])
            axs[2].plot(tr, np.nansum(np.where(okt_r, Pr['tb'], 0) * ar, axis=1) / np.sum(okt_r * ar, axis=1), '-',
                        color=c, lw=0.7, label=lab)
    axs[0].axhline(HYP, color='gray', ls=':', lw=1)
    axs[0].set_ylabel('Cove min bottom DO [mg/L]')
    axs[0].set_ylim(bottom=0)
    axs[0].legend(loc='upper right', fontsize=8)
    axs[0].set_title(f'Penn Cove bottom, wet cells west of {P["lon_mouth"]}, daily', fontsize=10)
    axs[1].set_ylabel(f'cove area < {HYP:g} mg/L [%]')
    axs[1].set_ylim(0, 100)
    axs[2].set_ylabel('Bottom T, cove mean [°C]')
    for ax in axs:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'pc_cove_compare.png', dpi=200, transparent=True)
    plt.close(fig)
print(f'saved  : {out_dir}')
