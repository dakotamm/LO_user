"""
IDK YET

Test on mac in ipython:
run create_obs_data_structures -gtx cas6_v0_live -source ecology -otype ctd -year 2017 -test False

"""

import sys
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime

from lo_tools import Lfun, zfun, zrfun
from lo_tools import extract_argfun as exfun
import cast_functions as cfun
from lo_tools import plotting_functions as pfun
import tef_fun as tfun
import pickle

#import VFC_functions as vfun

from time import time
from subprocess import Popen as Po
from subprocess import PIPE as Pi

from scipy.spatial import KDTree

import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import itertools

from collections import defaultdict

import os

import VFC_functions_2 as vfun

from pathlib import Path

# %%


Ldir = exfun.intro() # this handles the argument passing

# %%


info_df_dir = (Ldir['LOo'] / 'obs' / 'vfc')

df_dir = (Ldir['LOo'] / 'obs' / 'vfc' )

Lfun.make_dir(info_df_dir, clean=False)

Lfun.make_dir(df_dir, clean=False)


# %%

info_fn_temp = Ldir['LOo'] / 'obs' / Ldir['source'] / Ldir['otype'] / ('info_' + str(Ldir['year']) + '.p')

info_fn = (info_df_dir / ('info_' + str(Ldir['year']) + '.p'))


info_df_temp, info_df = vfun.buildInfoDF(Ldir, info_fn_temp, info_fn)

# %%

fn_temp = Ldir['LOo'] / 'obs' / Ldir['source'] / Ldir['otype'] / (str(Ldir['year']) + '.p')

fn = (df_dir / (str(Ldir['year']) + '.p'))


df_temp, df = vfun.buildDF(Ldir, fn_temp, fn, info_df)
            
