"""
Utility functions for the tef2 _avg scripts.
"""

from pathlib import Path
from datetime import datetime, timedelta

def get_avg_fn_list(Ldir, ds0, ds1):
    """
    Get a list of all hourly average files (ocean_avg_0001.nc through
    ocean_avg_0024.nc) for each day in the date range [ds0, ds1].
    
    This replaces Lfun.get_fn_list('average', ...) which only returns
    ocean_avg_0001.nc per day.
    """
    ds_fmt = Lfun_ds_fmt = '%Y.%m.%d'
    dt0 = datetime.strptime(ds0, ds_fmt)
    dt1 = datetime.strptime(ds1, ds_fmt)
    dir0 = Ldir['roms_out'] / Ldir['gtagex']
    
    fn_list = []
    dt = dt0
    while dt <= dt1:
        f_string = 'f' + dt.strftime(ds_fmt)
        for nhis in range(1, 25):
            nhiss = ('0000' + str(nhis))[-4:]
            fn = dir0 / f_string / ('ocean_avg_' + nhiss + '.nc')
            fn_list.append(fn)
        dt = dt + timedelta(days=1)
    return fn_list
