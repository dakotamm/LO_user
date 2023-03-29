"""
This is code for doing cast extractions. It will create cast field using specified obs locations and selected LO history files.

Created 2023/03/23.

Test on mac:
run extract_casts_DM -gtx cas6_v0_live -source dfo -otype ctd -year 2019 -test False

"""

import sys
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime

from lo_tools import Lfun, zfun, zrfun
from lo_tools import extract_argfun as exfun
import cast_functions as cfun
import tef_fun as tfun
import pickle

from time import time
from subprocess import Popen as Po
from subprocess import PIPE as Pi

import VFC_functions as vfun

import itertools

Ldir = exfun.intro() # this handles the argument passing

year_str = str(Ldir['year'])

month_num = ['01','02','03','04','05','06','07','08','09','10','11','12']

month_str = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

#month_num = ['07','08','09','10','11','12']

#month_str = ['Jul','Aug','Sep','Oct','Nov','Dec']

cast_start = datetime(2019,6,1)
cast_end = datetime(2019,6,30)

segments = ['G6']

for (mon_num, mon_str) in zip(month_num, month_str):

    dt = pd.Timestamp('2022-' + mon_num +'-01 01:30:00')
    fn = cfun.get_his_fn_from_dt(Ldir, dt)
    
    fn_mon_str = str(dt.month)
    fn_year_str = str(dt.year)
    
    for segment in segments:
    
        vfun.extractCastsPerSegments(Ldir, fn, cast_start, cast_end, year_str, fn_mon_str, fn_year_str, segment)