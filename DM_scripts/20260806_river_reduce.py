"""
Reduce the wb1_t0_xn11abbur00 river forcing to one small pickle, on apogee.

The forcing is ~730 daily rivers.nc files spread over ~730 directories, but
the part that matters is one number per source per day. This walks the run,
pulls that out, and writes a single file of a few MB that can be brought home
and analysed offline by 20260806_river_hydrographs.py.

It writes into the same DM_outs folder the analysis script reads from, so the
path is identical on apogee and at home and the file copies straight across:

  scp apogee:LO_output/DM_outs/20260806_river_hydrographs/river_daily_*.p \\
      ~/LO_output/DM_outs/20260806_river_hydrographs/

Three things get captured that cannot be recovered later from the pickle
alone, so they are all stored now rather than on a second trip:

  - flow AND the tracer concentrations, so nutrient/oxygen loading questions
    do not need another pull. They are tiny next to the flow.
  - river_direction / X / Epos, which is how sources get classified into
    rivers vs WWTPs (see the analysis script).
  - the per-source fill method from Info/screen_output.txt, which is the
    model's own statement of where each source's data came from.

Each daily forcing file holds a short window centred on its own date, so the
files overlap heavily; taking every file's whole record would multiply-count.
Only the entry at the file's own date is kept -- the one ROMS was reading
that day.

run 20260806_river_reduce.py
"""
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from pathlib import Path

from lo_tools import Lfun

Ldir = Lfun.Lstart(gridname='wb1')
gtx = 'wb1_t0_xn11abbur00'
FRC = 'trapsN00'
DATE0, DATE1 = '2024.01.01', '2025.12.31'

# s_rho-mean tracer concentrations to carry along with the flow
TRACERS = ['temp', 'salt', 'NO3', 'NH4', 'Oxyg', 'TIC', 'TAlk']

out_dir = Ldir['LOo'] / 'DM_outs' / '20260806_river_hydrographs'
Lfun.make_dir(out_dir)
out_fn = out_dir / ('river_daily_%s_%s_%s.p' % (FRC, DATE0, DATE1))

frc_dir = Ldir['LOo'] / 'forcing' / Ldir['gridname']
dates = pd.date_range(DATE0.replace('.', '-'), DATE1.replace('.', '-'),
                      freq='D')
print('reducing %d days of %s forcing from %s' % (len(dates), FRC, frc_dir))

Qd = {}
Vd = {v: {} for v in TRACERS}
meta, prov, missing = None, {}, []

for k, dt in enumerate(dates):
    day_dir = frc_dir / ('f' + dt.strftime('%Y.%m.%d')) / FRC
    fn = day_dir / 'rivers.nc'
    if not fn.is_file():
        missing.append(dt)
        continue

    ds = xr.open_dataset(fn)
    names = [str(s) for s in ds.river_name.values]
    want = dt + pd.Timedelta('12h')
    rt = pd.to_datetime(ds.river_time.values)
    i = int(np.argmin(np.abs(rt - want)))
    if abs(rt[i] - want) > pd.Timedelta('12h'):
        # the file does not actually cover its own date -- skip it rather than
        # silently substitute a neighbouring day
        missing.append(dt)
        ds.close()
        continue

    # sign is the source's isign, i.e. which way the flux points on the grid,
    # not source vs sink. Magnitude is what a hydrograph wants.
    Qd[dt] = pd.Series(np.abs(ds.river_transport.values[i, :]), index=names)
    for v in TRACERS:
        vn = 'river_' + v
        if vn in ds:
            Vd[v][dt] = pd.Series(ds[vn].values[i, :, :].mean(axis=0),
                                  index=names)
    if meta is None:
        meta = pd.DataFrame(dict(direction=ds.river_direction.values,
                                 Xpos=ds.river_Xposition.values,
                                 Epos=ds.river_Eposition.values), index=names)
    ds.close()

    # the fill method is the same statement every day, so read it once
    if not prov:
        log_fn = day_dir / 'Info' / 'screen_output.txt'
        if log_fn.is_file():
            for line in log_fn.read_text().splitlines():
                if line.startswith('-- ') and ': filled from' in line:
                    nm, how = line[3:].split(': filled from', 1)
                    prov[nm.strip()] = how.strip()   # last statement wins

    if (k + 1) % 100 == 0:
        print('  %d / %d' % (k + 1, len(dates)))

if not Qd:
    raise SystemExit('no %s forcing found under %s' % (FRC, frc_dir))

Q = pd.DataFrame(Qd).T.sort_index()
V = {v: pd.DataFrame(d).T.sort_index() for v, d in Vd.items() if d}

pd.to_pickle(dict(Q=Q, V=V, meta=meta, prov=prov, missing=missing,
                  info=dict(gtx=gtx, frc=FRC, date0=DATE0, date1=DATE1,
                            made=datetime.now().isoformat(timespec='seconds'))),
             out_fn)

print('\n%d days x %d sources, tracers: %s'
      % (Q.shape[0], Q.shape[1], ', '.join(V)))
if missing:
    print('** %d model days had no usable %s forcing file, e.g. %s'
          % (len(missing), FRC,
             ', '.join(d.strftime('%Y.%m.%d') for d in missing[:5])))
print('mean total flow %.1f m3/s over %s to %s'
      % (Q.sum(axis=1).mean(), Q.index[0].date(), Q.index[-1].date()))
print('\nsaved %s  (%.2f MB)' % (out_fn, out_fn.stat().st_size / 1e6))
# The destination is built from LOo's own last component rather than from
# Path.home(): on apogee LOo is /dat2/dakotamm/LO_output while $HOME is
# /home/dakotamm, so relative_to(home) has nothing to strip and raises.
print('\nbring it home with:\n  scp apogee:%s \\\n      ~/%s/'
      % (out_fn, Path(Ldir['LOo'].name) / out_dir.relative_to(Ldir['LOo'])))
