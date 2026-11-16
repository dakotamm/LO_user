"""
Bottom DO in every Penn Cove cell of cas7_t1_x11ab (Parker's run on apogee /dat1), from the daily averages
f<YYYY.MM.DD>/ocean_avg_0001.nc: is there any hypoxia? Companion to 20261116_pc_cas7_bottom_DO.py (one point).

Run on apogee (-ro 1 = Ldir['roms_out1'] = /dat1/parker/LO_roms on dm_apogee):
    nohup python 20261116_pc_cas7_bottom_DO_cove.py -gtx cas7_t1_x11ab -ro 1 -0 2012.10.07 -1 2025.06.12 > pc_cas7_bottom_DO_cove.log 2>&1 &
  Check on it with: tail pc_cas7_bottom_DO_cove.log
RESUME  each day is appended to <out_name>_partial.csv (flushed every 100 files). If the run dies, rerun the same
  command: days already in the partial file are skipped (a truncated last line is dropped). The final pickle is
  written, and the partial removed, at the end.
Re-plot from the saved pickle (e.g. on the Mac after copying it over):
    python 20261116_pc_cas7_bottom_DO_cove.py -gtx cas7_t1_x11ab -0 2012.10.07 -1 2025.06.12 -plot_only True

COVE    wet cas7 rho cells west of LON_MOUTH = -122.652 (the wb1 pc_lp mouth line, u-face i=67/68), connected to the
        seed (-122.70878, 48.22710) = the single-point cell j=882, i=557. 38 cells, h 12.6-26.6 m.
        Grid (lon, lat, mask, h, pm, pn) from the first avg file, or grid.nc if the avg file has no grid variables.
READ    one hyperslab per file, oxygen[0, 0, box] (bottom layer, s_rho index 0), then the cove cells.
        oxygen is mmol m-3; mg/L = mmol m-3 * 32/1000.
STATS   per day: cove min, area-weighted mean, % of cove area < 2 mg/L. Per cell: fraction of days < 2 mg/L, min.
MISSING days with no ocean_avg_0001.nc are skipped and counted.
OUT     LO_output/DM_outs/20261116_pc_cas7_bottom_DO_cove/
        <out_name>.p    dict: time, do (Nt x Ncell, mg/L), cell j/i/lon/lat/h/area, box lon/lat/mask
        <out_name>.csv  daily cove min, mean, hypoxic area %
        <out_name>_ts.png, <out_name>_map.png
"""

import argparse
import sys
import pickle
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
parser.add_argument('-plot_only', type=zfun.boolean_string, default=False)
args = parser.parse_args()

gridname, tag, ex_name = args.gtagex.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
if args.roms_out_num > 0:
    Ldir['roms_out'] = Ldir['roms_out' + str(args.roms_out_num)]

DO_MMOL_TO_MGL = 32.0 / 1000.0
HYP = 2.0                                  # mg/L
LON_MOUTH = -122.652                       # wb1 pc_lp line
SEED = (-122.70878, 48.22710)              # single-point cell
BOX_LAT = (48.20, 48.26)
AX = [-122.73, -122.64, 48.215, 48.245]

out_dir = Ldir['LOo'] / 'DM_outs' / '20261116_pc_cas7_bottom_DO_cove'
Lfun.make_dir(out_dir)
out_name = args.gtagex + '_pc_cove_' + args.ds0 + '_' + args.ds1
p_fn = out_dir / (out_name + '.p')

if not args.plot_only:
    fn_list = Lfun.get_fn_list('average', Ldir, args.ds0, args.ds1)
    fn_have = [fn for fn in fn_list if fn.exists()]
    print(f'files  : {len(fn_have)} of {len(fn_list)} days have ocean_avg_0001.nc')
    if len(fn_have) == 0:
        print(f'*** no files under {Ldir["roms_out"] / args.gtagex}')
        sys.exit()

    # ---- cove mask ----
    vns = ['lon_rho', 'lat_rho', 'mask_rho', 'h', 'pm', 'pn']
    with nc.Dataset(fn_have[0]) as ds:
        if 'mask_rho' in ds.variables:
            G = {vn: ds[vn][:].data for vn in vns}
        else:
            with nc.Dataset(Ldir['grid'] / 'grid.nc') as dg:
                G = {vn: dg[vn][:].data for vn in vns}
    lon, lat = G['lon_rho'], G['lat_rho']
    wet = (G['mask_rho'] == 1) & (lon < LON_MOUTH) & (lat > BOX_LAT[0]) & (lat < BOX_LAT[1])
    lab, _ = label(wet)
    js, is_ = np.unravel_index(np.argmin((lon - SEED[0])**2 + (lat - SEED[1])**2), lon.shape)
    if lab[js, is_] == 0:
        print('*** seed is not on a wet cove cell')
        sys.exit()
    cove = lab == lab[js, is_]
    J, I = np.where(cove)
    bj = slice(J.min() - 1, J.max() + 2)   # box with a one-cell margin, for the map
    bi = slice(I.min() - 1, I.max() + 2)
    cj, ci = J - bj.start, I - bi.start    # cove cells within the box
    area = 1 / (G['pm'][J, I] * G['pn'][J, I])
    print(f'cove   : {len(J)} cells  h {G["h"][J, I].min():.1f}-{G["h"][J, I].max():.1f} m  '
          f'area {area.sum() / 1e6:.2f} km2  box j {bj.start}-{bj.stop - 1} i {bi.start}-{bi.stop - 1}')

    # ---- one hyperslab per file, appended to a partial csv so a killed run can resume ----
    part_fn = out_dir / (out_name + '_partial.csv')
    cell_line = '# cells (j,i): ' + ' '.join(f'{j},{i}' for j, i in zip(J, I)) + '\n'
    ccols = [f'c{k}' for k in range(len(J))]
    done = set()
    if part_fn.exists():
        with open(part_fn) as f:
            if f.readline() != cell_line:
                print(f'*** {part_fn.name} is for a different cove mask: delete it')
                sys.exit()
        # keep only complete rows (a kill mid-write can leave a truncated last line), rewrite, then append
        dfp = pd.read_csv(part_fn, comment='#', on_bad_lines='skip')
        dfp = dfp[dfp['fdir'].astype(str).str.match(r'^f\d{4}\.\d{2}\.\d{2}$')]
        dfp = dfp[dfp[ccols].apply(pd.to_numeric, errors='coerce').notna().all(axis=1)
                  & pd.to_datetime(dfp['time_utc'], errors='coerce').notna()]
        with open(part_fn, 'w') as f:
            f.write(cell_line)
            dfp.to_csv(f, index=False)
        done = set(dfp['fdir'])
        print(f'resume : {len(done)} days already in {part_fn.name}')
    else:
        with open(part_fn, 'w') as f:
            f.write(cell_line + 'fdir,time_utc,' + ','.join(ccols) + '\n')
    fn_todo = [fn for fn in fn_have if fn.parent.name not in done]

    with open(part_fn, 'a') as f:
        for nn, fn in enumerate(fn_todo):
            with nc.Dataset(fn) as ds:
                t0 = nc.num2date(ds['ocean_time'][0], ds['ocean_time'].units,
                                 only_use_cftime_datetimes=False, only_use_python_datetimes=True)
                ob = ds['oxygen'][0, 0, bj, bi]
            o0 = np.ma.filled(ob.astype(float), np.nan)[cj, ci] * DO_MMOL_TO_MGL
            f.write(f'{fn.parent.name},{t0:%Y-%m-%d %H:%M:%S},' + ','.join(f'{v:.4f}' for v in o0) + '\n')
            if nn % 100 == 0:
                f.flush()
                print(f'  {nn} / {len(fn_todo)}  {fn.parent.name}', flush=True)

    df = pd.read_csv(part_fn, comment='#').drop_duplicates('fdir').sort_values('time_utc')
    P = {'gtagex': args.gtagex, 'time': pd.to_datetime(df['time_utc']).values, 'do': df[ccols].values,
         'j': J, 'i': I, 'lon': lon[J, I], 'lat': lat[J, I], 'h': G['h'][J, I], 'area': area,
         'box_lon': lon[bj, bi], 'box_lat': lat[bj, bi], 'box_mask': G['mask_rho'][bj, bi], 'cj': cj, 'ci': ci,
         'lon_mouth': LON_MOUTH}
    pickle.dump(P, open(p_fn, 'wb'))
    part_fn.unlink()
    print(f'saved  : {p_fn}')

# ---- stats ----
P = pickle.load(open(p_fn, 'rb'))
t, do, area = pd.to_datetime(P['time']), P['do'], P['area']
ok = np.isfinite(do)
do_min = np.nanmin(do, axis=1)
do_mean = np.nansum(np.where(ok, do, 0) * area, axis=1) / np.sum(ok * area, axis=1)
hyp_pct = 100 * np.sum((do < HYP) * area, axis=1) / np.sum(ok * area, axis=1)
pd.DataFrame({'time_utc': t, 'DO_bot_min_mgL': do_min, 'DO_bot_mean_mgL': do_mean, 'hyp_area_pct': hyp_pct}
             ).to_csv(out_dir / (out_name + '.csv'), index=False)

print(f'cove bottom DO: min {np.nanmin(do):.2f} mg/L on {t[np.nanargmin(do_min)]:%Y-%m-%d};  '
      f'days with any cell < {HYP:g} mg/L: {int((do_min < HYP).sum())} of {len(t)}')
yr = pd.Series(do_min < HYP, index=t).groupby(t.year)
print('days with any hypoxia, by year:')
print(pd.DataFrame({'hyp_days': yr.sum(), 'days': yr.size(),
                    'min_DO': pd.Series(do_min, index=t).groupby(t.year).min().round(2)}).to_string())

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
print(f'saved  : {out_dir}')
