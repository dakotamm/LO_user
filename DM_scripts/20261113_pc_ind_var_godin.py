"""
Godin-filtered versions of the Penn Cove independent variable time series.

STEP 1: load the hourly series from 20261112_pc_ind_var_ts.py
    LO_output/DM_outs/20261112_pc_ind_var_ts/ind_var_ts.p, a DataFrame on the
    hourly hh:00 time index (2024-01-01 00:00 to 2026-01-01 00:00, 17545 hours).
    The Godin filter needs evenly spaced hourly data with no gaps, so check that.

STEP 2: Godin filter every column (zfun.lowpass, f='godin': the 24-24-25 hour
    filter, 71 hourly weights). The first and last 35 hours have no full window
    and are NaN. Same column names, same time index.
    eta_tide and Q_tide are left out: they are pure tides, so the filter takes
    them to ~0.
    Saved to LO_output/DM_outs/20261113_pc_ind_var_godin/ind_var_godin.p.

run 20261113_pc_ind_var_godin.py
"""
import numpy as np
import pandas as pd

from lo_tools import Lfun, zfun

Ldir = Lfun.Lstart(gridname='wb1')
in_fn = Ldir['LOo'] / 'DM_outs' / '20261112_pc_ind_var_ts' / 'ind_var_ts.p'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261113_pc_ind_var_godin'
Lfun.make_dir(out_dir)
out_fn = out_dir / 'ind_var_godin.p'

df = pd.read_pickle(in_fn)
df = df.drop(columns=['eta_tide', 'Q_tide'])

dt = np.diff(df.index.values).astype('timedelta64[s]').astype(int)
assert (dt == 3600).all()
assert df.notna().all().all()

print('%d hours, %s to %s' % (len(df), str(df.index[0])[:16], str(df.index[-1])[:16]))
print('columns: ' + ', '.join(df.columns))

"""
STEP 2: Godin filter.
"""
dg = pd.DataFrame(index=df.index)
for vn in df.columns:
    dg[vn] = zfun.lowpass(df[vn].values.astype(float), f='godin')

print('%-12s %12s %12s' % ('', 'hourly std', 'Godin std'))
for vn in df.columns:
    print('%-12s %12.4g %12.4g' % (vn, df[vn].std(), dg[vn].std()))

dg.to_pickle(out_fn)
print('saved %s (%d hours, %d columns; %d NaN hours at each end)'
      % (out_fn, len(dg), dg.shape[1], dg.iloc[:, 0].isna().sum() // 2))
