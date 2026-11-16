"""
Daily Skagit River flow 2012.10.07-2025.06.12, as the LO river forcing builds it (runs on the Mac, no apogee):
    2012.10.07-2023.12.31  LO_output/pre/river1/lo_base/Data_historical/ALL_flow.p['skagit'] (ends 2023-12-31)
    2024.01.01-2025.06.12  USGS 12200500 daily mean (cfs -> m3/s) * ratio 1.0096, via river_functions.get_usgs_data
river_info.p gives the gauge and ratio. Verified to 0.0000 m3/s against the wb1 2024-25 rivers.nc (memory:
wb1_river_forcing_provenance); assumed the same for cas7 trapsN00, which uses the same lo_base river1 tables.
OUT  LO_output/DM_outs/20261116_skagit_flow/skagit_flow_<ds0>_<ds1>.csv + .png
    python 20261116_skagit_flow.py
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

from lo_tools import Lfun
from lo_tools import river_functions as rivf

DS0, DS1 = '2012.10.07', '2025.06.12'

Ldir = Lfun.Lstart()
out_dir = Ldir['LOo'] / 'DM_outs' / '20261116_skagit_flow'
Lfun.make_dir(out_dir)
riv_dir = Ldir['LOo'] / 'pre' / 'river1' / 'lo_base'

t0, t1 = datetime.strptime(DS0, Lfun.ds_fmt), datetime.strptime(DS1, Lfun.ds_fmt)
q_hist = pd.read_pickle(riv_dir / 'Data_historical' / 'ALL_flow.p')['skagit']
q_hist = q_hist[(q_hist.index >= t0) & (q_hist.index <= datetime(2023, 12, 31, 23))]

rs = pd.read_pickle(riv_dir / 'river_info.p').loc['skagit'].copy()
rs.name = 'skagit'
rs, q_usgs = rivf.get_usgs_data(rs, (datetime(2024, 1, 1), t1))
if not rs.got_data:
    raise SystemExit('*** USGS fetch failed')

q = pd.concat([q_hist, q_usgs]).sort_index()
q.index = q.index.normalize() + pd.Timedelta(hours=12)       # noon stamp, one per day
q = q[~q.index.duplicated()]
full = pd.date_range(t0 + pd.Timedelta(hours=12), t1 + pd.Timedelta(hours=12), freq='D')
print(f'skagit: {q.notna().sum()} of {len(full)} days  (hist {len(q_hist)}, usgs {len(q_usgs)})  '
      f'missing {len(full.difference(q.index))}')
print(f'        min {q.min():.0f}  mean {q.mean():.0f}  max {q.max():.0f} m3/s')

fn = f'skagit_flow_{DS0}_{DS1}'
q.rename('skagit_m3s').rename_axis('time').to_csv(out_dir / (fn + '.csv'))

fig, ax = plt.subplots(figsize=(14, 3.5))
ax.plot(q.index, q.values, '-', color='tab:blue', lw=0.7)
ax.axvline(datetime(2024, 1, 1), color='gray', ls=':', lw=1)
ax.text(datetime(2024, 1, 15), 0.95, 'USGS from here', transform=ax.get_xaxis_transform(), va='top', fontsize=8)
ax.set_ylabel('Skagit flow [m$^3$ s$^{-1}$]')
ax.set_xlim(full[0], full[-1])
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)
ax.set_title('Skagit River (LO river forcing: USGS 12200500 x 1.0096)', fontsize=10)
fig.tight_layout()
fig.savefig(out_dir / (fn + '.png'), dpi=200, transparent=True)
plt.close(fig)
print(f'saved: {out_dir / fn}.csv / .png')
