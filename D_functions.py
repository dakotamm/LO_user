"""
Functions used by Dakota!

Created 2023/12/14.

"""

import sys
import pandas as pd
import xarray as xr
import numpy as np

from lo_tools import Lfun, zfun, zrfun #original LO tools
import extract_argfun_DM as exfun
import cast_functions_DM as cfun
from lo_tools import plotting_functions as pfun #original LO tools
import tef_fun_DM as tfun
import pickle

from time import time as Time
from subprocess import Popen as Po
from subprocess import PIPE as Pi

from scipy.spatial import KDTree

from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

import matplotlib.colors as cm

import math


import itertools

import copy

from pathlib import PosixPath

from datetime import timedelta


from dateutil.relativedelta import relativedelta

import matplotlib.path as mpth
import datetime

from warnings import filterwarnings
filterwarnings('ignore') # skip some warning messages

import seaborn as sns

# %%

def getPolyData(Ldir, poly_list, source_list=['ecology', 'nceiSalish', 'dfo1', 'collias'], otype_list=['ctd', 'bottle'], year_list=np.arange(1930, 2022)):

    fng = Ldir['grid'] / 'grid.nc'
    dsg = xr.open_dataset(fng)
    x = dsg.lon_rho.values
    y = dsg.lat_rho.values
    m = dsg.mask_rho.values
    xp, yp = pfun.get_plon_plat(x,y)
    h = dsg.h.values
    h[m==0] = np.nan
    
    path_dict = dict()
    xxyy_dict = dict()
    for poly in poly_list:
        # polygon
        fnp = Ldir['LOo'] / 'section_lines' / (poly+'.p')
        p = pd.read_pickle(fnp)
        xx = p.x.to_numpy()
        yy = p.y.to_numpy()
        xxyy = np.concatenate((xx.reshape(-1,1),yy.reshape(-1,1)), axis=1)
        path = mpth.Path(xxyy)
        # store in dicts
        path_dict[poly] = path
        xxyy_dict[poly] = xxyy

    # observations
    ii = 0
    for year in year_list:
        for source in source_list:
            for otype in otype_list:
                odir = Ldir['LOo'] / 'obs' / source / otype
                try:
                    if ii == 0:
                        odf = pd.read_pickle( odir / (str(year) + '.p'))
                        if 'ecology' in source_list:
                            if source == 'ecology' and otype == 'bottle':
                                odf['DO (uM)'] == np.nan
                        odf['source'] = source
                        odf['otype'] = otype
                        # print(odf.columns)
                    else:
                        this_odf = pd.read_pickle( odir / (str(year) + '.p'))
                        if 'ecology' in source_list:
                            if source == 'ecology' and otype == 'bottle':
                                this_odf['DO (uM)'] == np.nan
                        this_odf['cid'] = this_odf['cid'] + odf['cid'].max() + 1
                        this_odf['source'] = source
                        this_odf['otype'] = otype
                        # print(this_odf.columns)
                        odf = pd.concat((odf,this_odf),ignore_index=True)
                    ii += 1
                except FileNotFoundError:
                    pass
                
        print(str(year))

    # if True:
    #     # limit time range
    #     ti = pd.DatetimeIndex(odf.time)
    #     mo = ti.month
    #     mo_mask = mo==0 # initialize all false
    #     for imo in [9,10,11]:
    #         mo_mask = mo_mask | (mo==imo)
    #     odf = odf.loc[mo_mask,:]
        
    # get lon lat of (remaining) obs
    ox = odf.lon.to_numpy()
    oy = odf.lat.to_numpy()
    oxoy = np.concatenate((ox.reshape(-1,1),oy.reshape(-1,1)), axis=1)


    # get all profiles inside each polygon
    odf_dict = dict()
    for poly in poly_list:
        path = path_dict[poly]
        oisin = path.contains_points(oxoy)
        odfin = odf.loc[oisin,:]
        odf_dict[poly] = odfin.copy()
        
    return odf_dict

# %%

#def cleanODFDict(odf_dict):
    
# %%

#def getCounts(frequency):
    
    
    