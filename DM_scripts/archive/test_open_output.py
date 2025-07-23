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
reload(pfun)
#reload(pinfo)

import matplotlib.pyplot as plt

fn = '/Users/dakotamascarenas/LO_roms/cas6_v0_live/f2022.11.30/ocean_his_0019.nc'

ds = xr.open_dataset(fn)
