#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 16:13:50 2025

@author: dakotamascarenas
"""

from lo_tools import Lfun, zfun, zrfun
from lo_tools import plotting_functions as pfun
import matplotlib.pyplot as plt
import matplotlib.path as mpth
import xarray as xr
import numpy as np
import pandas as pd
import datetime

from warnings import filterwarnings
filterwarnings('ignore') # skip some warning messages

import seaborn as sns

import scipy.stats as stats

import D_functions as dfun

import pickle

import math

from scipy.interpolate import interp1d

import gsw

import matplotlib.path as mpth

import matplotlib.patches as patches

import cmocean


# %%

his = xr.open_dataset('/Users/dakotamascarenas/Desktop/roms_his.nc')

x_rho = his.x_rho.values
y_rho = his.y_rho.values
temp = his.temp.values

# %%

fig,ax = plt.subplots(nrows=1, ncols=1)


pcm = ax.pcolormesh(x_rho, y_rho, temp[-1,-1,:,:], cmap='Spectral_r')


plt.savefig('/Users/dakotamascarenas/Desktop/pltz/test_upwelling_temp.png', dpi=500)
