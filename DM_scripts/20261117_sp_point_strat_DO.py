"""
Daily stratification and bottom DO at one Saratoga Passage point (default M5, just outside Penn Cove), for the cas7
hindcasts. Each day reads the full salt / temp / oxygen column and zeta at the cell, so any other column metric can
be computed later from the pickle without another apogee run.

Run on apogee (both lines can run at the same time; resumable like 20261116_pc_cas7_bottom_DO_cove.py):
  cas7_t1_x11ab (50% burial; daily ocean_avg_0001.nc in Parker's /dat1 = roms_out1):
    nohup python 20261117_sp_point_strat_DO.py -gtx cas7_t1_x11ab -ro 1 -0 2012.10.07 -1 2025.06.12 -lt average > sp_x11ab.log 2>&1 &
  cas7_t0_x4b (0% burial; lowpassed.nc in Parker's /dat2 = roms_out3):
    nohup python 20261117_sp_point_strat_DO.py -gtx cas7_t0_x4b -ro 3 -0 2012.10.07 -1 2019.12.31 -lt lowpass > sp_x4b.log 2>&1 &
  Check on them with: tail sp_x11ab.log sp_x4b.log
RESUME  each day is appended to <out_name>_partial.csv (flushed every 100 files). If a run dies, rerun the same
  command: days already in the partial file are skipped (a truncated last line is dropped). The final pickle is
  written, and the partial removed, at the end.
Re-plot (e.g. on the Mac after copying the pickles over); also redraws the run comparison:
    python 20261117_sp_point_strat_DO.py -gtx cas7_t1_x11ab -0 2012.10.07 -1 2025.06.12 -plot_only True

POINT   M5 (-122.582462, 48.245981), the pc0/pc4 Saratoga Passage station (wb1 h 37.4 m). Snapped to the nearest wet
        rho cell: cas7 j=886, i=576, h = 39.2 m, 0.15 km away (printed). Grid from the first ocean file, else that
        day's ocean_his_0002.nc, else grid.nc.
READ    -lt average: f<date>/ocean_avg_0001.nc (UTC-day average); -lt lowpass: f<date>/lowpassed.nc (one Godin-
        filtered record per day). zeta, and salt, temp, oxygen at every s_rho level, at the one cell.
STRAT   drho = sigma0(bottom) - sigma0(top) [kg/m3], TEOS-10 as pc_forcing_states.py / 20261110_pc_column_metrics.py:
        SA = SA_from_SP(salt, -z_rho, lon, lat), CT = CT_from_pt(SA, temp). z_rho from zrfun.get_z with each day's
        zeta and the file's s-coordinate. Also dS and dT (bottom - top).
BOTTOM  DO at s_rho index 0; oxygen mmol m-3 * 32/1000 = mg/L.
OUT     LO_output/DM_outs/20261117_sp_point_strat_DO/
        <out_name>.p    dict: time, zeta, z_rho / salt / temp / oxygen / sigma0 profiles (Nt x N), cell info
        <out_name>.csv  daily bottom DO, drho, dS, dT, top/bottom S and T
        <out_name>.png  bottom DO + drho;  <sn>_compare.png  every run found for this station, overlaid
"""

import argparse
import sys
import pickle
import numpy as np
import pandas as pd
import netCDF4 as nc
import gsw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun, zfun, zrfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', type=str, default='cas7_t1_x11ab')
parser.add_argument('-ro', '--roms_out_num', type=int, default=1)  # 1 = Ldir['roms_out1']
parser.add_argument('-0', '--ds0', type=str, default='2012.10.07')
parser.add_argument('-1', '--ds1', type=str, default='2025.06.12')
parser.add_argument('-lt', type=str, default='average')   # average (ocean_avg_0001) or lowpass (lowpassed.nc)
parser.add_argument('-lon', type=float, default=-122.582462)
parser.add_argument('-lat', type=float, default=48.245981)
parser.add_argument('-sn', type=str, default='M5')        # station name, used in the output file names
parser.add_argument('-plot_only', type=zfun.boolean_string, default=False)
args = parser.parse_args()

gridname, tag, ex_name = args.gtagex.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
if args.roms_out_num > 0:
    Ldir['roms_out'] = Ldir['roms_out' + str(args.roms_out_num)]

DO_MMOL_TO_MGL = 32.0 / 1000.0
HYP = 2.0                                  # mg/L

out_dir = Ldir['LOo'] / 'DM_outs' / '20261117_sp_point_strat_DO'
Lfun.make_dir(out_dir)
out_name = args.gtagex + '_' + args.sn + '_' + args.ds0 + '_' + args.ds1
p_fn = out_dir / (out_name + '.p')

if not args.plot_only:
    fn_list = Lfun.get_fn_list(args.lt, Ldir, args.ds0, args.ds1)
    fn_have = [fn for fn in fn_list if fn.exists()]
    print(f'files  : {len(fn_have)} of {len(fn_list)} days have {fn_list[0].name}')
    if len(fn_have) == 0:
        print(f'*** no files under {Ldir["roms_out"] / args.gtagex}')
        sys.exit()

    # ---- nearest wet rho cell ----
    vns = ['lon_rho', 'lat_rho', 'mask_rho', 'h']
    for gfn in [fn_have[0], fn_have[0].parent / 'ocean_his_0002.nc', Ldir['grid'] / 'grid.nc']:
        if gfn.exists():
            with nc.Dataset(gfn) as ds:
                if all(vn in ds.variables for vn in vns):
                    G = {vn: ds[vn][:].data for vn in vns}
                    break
    else:
        print('*** no grid variables in the ocean file, ocean_his_0002.nc or grid.nc')
        sys.exit()
    dist2 = ((G['lon_rho'] - args.lon) * np.cos(np.deg2rad(args.lat)))**2 + (G['lat_rho'] - args.lat)**2
    dist2[G['mask_rho'] != 1] = np.inf
    j, i = np.unravel_index(np.argmin(dist2), dist2.shape)
    lon, lat, h = float(G['lon_rho'][j, i]), float(G['lat_rho'][j, i]), float(G['h'][j, i])
    S = zrfun.get_basic_info(fn_have[0], only_S=True)
    N = int(S['N'])
    print(f'grid   : {gfn}')
    print(f'cell   : j={j} i={i}  lon {lon:.5f}  lat {lat:.5f}  h {h:.1f} m  N {N}  '
          f'({np.sqrt(dist2[j, i]) * 111:.2f} km from the requested point)')

    # ---- one column per file, appended to a partial csv so a killed run can resume ----
    part_fn = out_dir / (out_name + '_partial.csv')
    cell_line = f'# v1 {args.gtagex} {args.lt} cell j={j} i={i} N={N}\n'
    scols, tcols, ocols = ([f'{c}{k}' for k in range(N)] for c in ['s', 't', 'o'])
    cols = ['fdir', 'time_utc', 'zeta'] + scols + tcols + ocols + ['ok']
    done = set()
    if part_fn.exists():
        with open(part_fn) as f:
            if f.readline() != cell_line:
                print(f'*** {part_fn.name} is for a different cell or file type: delete it')
                sys.exit()
        # keep only complete rows (a kill mid-write leaves a short last line, so no ok=1), rewrite, then append
        dfp = pd.read_csv(part_fn, comment='#', on_bad_lines='skip')
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

    with open(part_fn, 'a') as f:
        for nn, fn in enumerate(fn_todo):
            with nc.Dataset(fn) as ds:
                t0 = nc.num2date(ds['ocean_time'][0], ds['ocean_time'].units,
                                 only_use_cftime_datetimes=False, only_use_python_datetimes=True)
                vals = [float(ds['zeta'][0, j, i])]
                for vn in ['salt', 'temp', 'oxygen']:
                    vals += list(np.ma.filled(ds[vn][0, :, j, i].astype(float), np.nan))
            f.write(f'{fn.parent.name},{t0:%Y-%m-%d %H:%M:%S},' + ','.join(f'{v:.4f}' for v in vals) + ',1\n')
            if nn % 100 == 0:
                f.flush()
                print(f'  {nn} / {len(fn_todo)}  {fn.parent.name}', flush=True)

    df = pd.read_csv(part_fn, comment='#').drop_duplicates('fdir').sort_values('time_utc').reset_index(drop=True)
    zeta = df['zeta'].values
    z_rho, _ = zrfun.get_z(h * np.ones(len(zeta)), zeta, S)     # (N, Nt), bottom first
    z_rho = z_rho.T
    salt, temp = df[scols].values, df[tcols].values
    SA = gsw.SA_from_SP(salt, -z_rho, lon, lat)
    sig = gsw.sigma0(SA, gsw.CT_from_pt(SA, temp))
    P = {'gtagex': args.gtagex, 'lt': args.lt, 'sn': args.sn, 'time': pd.to_datetime(df['time_utc']).values,
         'zeta': zeta, 'z_rho': z_rho, 'salt': salt, 'temp': temp, 'oxygen': df[ocols].values * DO_MMOL_TO_MGL,
         'sigma0': sig, 'j': j, 'i': i, 'lon': lon, 'lat': lat, 'h': h}
    pickle.dump(P, open(p_fn, 'wb'))
    part_fn.unlink()
    print(f'saved  : {p_fn}')

# ---- daily series ----
def series(P):
    return pd.DataFrame({'time_utc': pd.to_datetime(P['time']), 'DO_bot_mgL': P['oxygen'][:, 0],
                         'drho': P['sigma0'][:, 0] - P['sigma0'][:, -1],
                         'dS': P['salt'][:, 0] - P['salt'][:, -1], 'dT': P['temp'][:, 0] - P['temp'][:, -1],
                         'S_top': P['salt'][:, -1], 'S_bot': P['salt'][:, 0],
                         'T_top': P['temp'][:, -1], 'T_bot': P['temp'][:, 0]})

P = pickle.load(open(p_fn, 'rb'))
D = series(P)
D.to_csv(out_dir / (out_name + '.csv'), index=False)
t = D['time_utc']
print(f'{P["gtagex"]} {P["sn"]} (j={P["j"]} i={P["i"]}, h {P["h"]:.1f} m): bottom DO min {D.DO_bot_mgL.min():.2f} mg/L '
      f'on {t[D.DO_bot_mgL.idxmin()]:%Y-%m-%d}, days < {HYP:g} mg/L: {int((D.DO_bot_mgL < HYP).sum())} of {len(D)};  '
      f'drho mean {D.drho.mean():.2f}, max {D.drho.max():.2f} kg/m3')

title = f'{P["gtagex"]}  {P["sn"]} ({P["lon"]:.4f}, {P["lat"]:.4f}, h = {P["h"]:.1f} m), {P["lt"]}'
fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
axs[0].plot(t, D.DO_bot_mgL, '-', color='tab:blue', lw=0.8)
axs[0].axhline(HYP, color='gray', ls=':', lw=1)
axs[0].set_ylabel('Bottom DO [mg/L]')
axs[0].set_ylim(bottom=0)
axs[0].set_title(title, fontsize=10)
axs[1].plot(t, D.drho, '-', color='k', lw=0.8)
axs[1].set_ylabel(r'$\Delta\sigma_0$ bottom - top [kg m$^{-3}$]')
axs[1].set_ylim(bottom=0)
axs[1].set_xlim(t.iloc[0], t.iloc[-1])
for ax in axs:
    ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(out_dir / (out_name + '.png'), dpi=200, transparent=True)
plt.close(fig)

# ---- every run for this station, overlaid ----
runs = sorted(out_dir.glob(f'*_{P["sn"]}_*.p'))
if len(runs) > 1:
    fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    for fn, c in zip(runs, ['tab:blue', 'tab:red', 'tab:green', 'tab:purple']):
        Pr = pickle.load(open(fn, 'rb'))
        Dr = series(Pr)
        lab = f'{Pr["gtagex"]} ({Pr["lt"]})'
        axs[0].plot(Dr.time_utc, Dr.DO_bot_mgL, '-', color=c, lw=0.7, label=lab)
        axs[1].plot(Dr.time_utc, Dr.drho, '-', color=c, lw=0.7, label=lab)
    axs[0].axhline(HYP, color='gray', ls=':', lw=1)
    axs[0].set_ylabel('Bottom DO [mg/L]')
    axs[0].set_ylim(bottom=0)
    axs[0].legend(loc='upper right', fontsize=8)
    axs[0].set_title(f'{P["sn"]} ({P["lon"]:.4f}, {P["lat"]:.4f}, h = {P["h"]:.1f} m)', fontsize=10)
    axs[1].set_ylabel(r'$\Delta\sigma_0$ bottom - top [kg m$^{-3}$]')
    axs[1].set_ylim(bottom=0)
    for ax in axs:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / (P['sn'] + '_compare.png'), dpi=200, transparent=True)
    plt.close(fig)
print(f'saved  : {out_dir}')
