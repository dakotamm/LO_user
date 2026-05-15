#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 14:50:38 2026

@author: dakotamascarenas
"""

from lo_tools import Lfun, zfun, zrfun
from lo_tools import plotting_functions as pfun

import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from matplotlib.path import Path
import gsw
from cmcrameri import cm as cmc

from cmocean import cm as cmo# have to import after matplotlib to work on remote machine
# %%

var_list = ['CT', 'SA', 'DO (uM)', 'PH', 'Chl (mg m-3)']

for year in range(2014,2020):
    
    df = pd.read_pickle('/Users/dakotamascarenas/LO_output/obs/pcRaft/sonde/' + str(year) + '.p')
    
    
    fig, axd = plt.subplot_mosaic([var_list], sharex=True, figsize=(20,5), layout='constrained', gridspec_kw=dict(wspace=0.1))
    
    for var in var_list:
        
        if var == 'DO':
            
            mult = 32/1000
        else:
            mult = 1
            
        ax = axd[var]
        
        ax.scatter(df[df['z'] == -1]['time'], df[df['z'] == -1][var]*mult, color = 'blue', label = '1m')
        
        ax.scatter(df[df['z'] == -7]['time'], df[df['z'] == -7][var]*mult, color = 'red', label = '7m')
        
        ax.scatter(df[df['z'] == -0.5]['time'], df[df['z'] == -0.5][var]*mult, label = '0.5m')
        
        ax.scatter(df[df['z'] == -2]['time'], df[df['z'] == -2][var]*mult, label = '2m')
        
        ax.scatter(df[df['z'] == -3]['time'], df[df['z'] == -3][var]*mult, label = '3m')
        
        ax.scatter(df[df['z'] == -4]['time'], df[df['z'] == -4][var]*mult, label = '4m')
        
        ax.scatter(df[df['z'] == -5]['time'], df[df['z'] == -5][var]*mult, label = '5m')
        
        ax.scatter(df[df['z'] == -6]['time'], df[df['z'] == -6][var]*mult, label = '6m')

        
        ax.set_title(var)

    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/carrington_data_' + str(year) + '.png', dpi=500, transparent=False, bbox_inches='tight')


                                                 
                                                 
    
    
