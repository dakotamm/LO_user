#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 10:49:01 2022

@author: dakotamascarenas
"""

import numpy as np
import xarray as xr
import pickle
from datetime import datetime, timedelta
import pandas as pd
from cmocean import cm
import sys

from lo_tools import Lfun, zfun, zrfun
from lo_tools import plotting_functions as pfun
#import pinfo
from importlib import reload
#reload(pfun)
#reload(pinfo)

Ldir = Lfun.Lstart()
if '_mac' in Ldir['lo_env']: # mac version
    pass
else: # remote linux version
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt

fn0 = '/Users/dakotamascarenas/LO_output/extract/cas6_v0_live/segment_temp_2022.11.30-2022.11.30/A_2022.11.30_0001.p'

fn1 = '/Users/dakotamascarenas/LO_output/extract/tef/volumes_cas6/bathy_dict.p'

fn2 = '/Users/dakotamascarenas/LO_output/extract/tef/volumes_cas6/i_dict.p'

fn3 = '/Users/dakotamascarenas/LO_output/extract/tef/volumes_cas6/j_dict.p'

fn4 = '/Users/dakotamascarenas/LO_output/extract/tef/volumes_cas6/volumes.p'

segments = pd.read_pickle(fn0)

bathy = pd.read_pickle(fn1)

i_dict = pd.read_pickle(fn2)

j_dict = pd.read_pickle(fn3)

volumes = pd.read_pickle(fn4)


