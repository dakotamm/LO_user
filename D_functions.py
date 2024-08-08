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

import scipy.stats as stats


# %%

def getPolyData(Ldir, poly_list, source_list=['ecology_nc', 'nceiSalish', 'dfo1', 'collias'], otype_list=['ctd', 'bottle'], year_list=np.arange(1930, 2022)):

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
                            if source == 'ecology' and otype == 'bottle': #keep an eye on this for calculating confidence intervals!!!
                                odf['DO (uM)'] == np.nan
                        if 'kc_point_jefferson' in source_list:
                            if source == 'kc_point_jefferson' and otype == 'bottle': #keep an eye on this for calculating confidence intervals!!!
                                odf['CT'] == np.nan                        
                        odf['source'] = source
                        odf['otype'] = otype
                        # print(odf.columns)
                    else:
                        this_odf = pd.read_pickle( odir / (str(year) + '.p'))
                        if 'ecology' in source_list:
                            if source == 'ecology' and otype == 'bottle':
                                this_odf['DO (uM)'] == np.nan
                        if 'kc_point_jefferson' in source_list:
                            if source == 'kc_point_jefferson' and otype == 'bottle': #keep an eye on this for calculating confidence intervals!!!
                                odf['CT'] == np.nan                          
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
        
    return odf_dict, path_dict

# %%

def mann_kendall(V, alpha=0.05):
    '''Mann Kendall Test (adapted from original Matlab function)
       Performs original Mann-Kendall test of the null hypothesis of trend absence in the vector V, against the alternative of trend.
       The result of the test is returned in reject_null:
       reject_null = True indicates a rejection of the null hypothesis at the alpha significance level. 
       reject_null = False indicates a failure to reject the null hypothesis at the alpha significance level.

       INPUTS:
       V = time series [vector]
       alpha =  significance level of the test [scalar] (i.e. for 95% confidence, alpha=0.05)
       OUTPUTS:
       reject_null = True/False (True: reject the null hypothesis) (False: insufficient evidence to reject the null hypothesis)
       p_value = p-value of the test
       
       From Original Matlab Help Documentation:
       The significance level of a test is a threshold of probability a agreed to before the test is conducted. 
       A typical value of alpha is 0.05. If the p-value of a test is less than alpha,        
       the test rejects the null hypothesis. If the p-value is greater than alpha, there is insufficient evidence 
       to reject the null hypothesis. 
       The p-value of a test is the probability, under the null hypothesis, of obtaining a value
       of the test statistic as extreme or more extreme than the value computed from
       the sample.
       
       References 
       Mann, H. B. (1945), Nonparametric tests against trend, Econometrica, 13, 245-259.
       Kendall, M. G. (1975), Rank Correlation Methods, Griffin, London.
       
       Original written by Simone Fatichi - simonef@dicea.unifi.it
       Copyright 2009
       Date: 2009/10/03
       modified: E.I. (1/12/2012)
       modified and converted to python: Steven Pestana - spestana@uw.edu (10/17/2019)
       '''

    V = np.reshape(V, (len(V), 1))
    alpha = alpha/2
    n = len(V)
    S = 0

    for i in range(0, n-1):
        for j in range(i+1, n):
            if V[j]>V[i]:
                S = S+1
            if V[j]<V[i]:
                S = S-1

    VarS = (n*(n-1)*(2*n+5))/18
    StdS = np.sqrt(VarS)
    # Ties are not considered

    # Kendall tau correction coefficient
    Kendall_Tau = S/(n*(n-1)/2)
    if S>=0:
        if S==0:
             Z = 0
        else:
            Z = ((S-1)/StdS)
    else:
        Z = (S+1)/StdS

    Zalpha = stats.norm.ppf(1-alpha,0,1)
    p_value = 2*(1-stats.norm.cdf(abs(Z), 0, 1)) #Two-tailed test p-value

    reject_null = abs(Z) > Zalpha # reject null hypothesis only if abs(Z) > Zalpha
    
    return reject_null, p_value, Z
    
# %%
    
# THEIL SEN #left off 7/17/2024 resume!
    