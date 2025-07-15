#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 16:34:15 2025

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
import pickle

collias_hc_df = pd.read_pickle('/Users/dakotamascarenas/Desktop/ps_longterm/ptools_output/collias/region_5.p')

collias_hc_df['Data Source'] = 'Collias'

collias_hc_df['Station'] = 'n/a'

collias_hc_df['Region'] = 'Hood Canal'


collias_mb_df = pd.read_pickle('/Users/dakotamascarenas/Desktop/ps_longterm/ptools_output/collias/region_3.p')

collias_mb_df['Data Source'] = 'Collias'

collias_mb_df['Station'] = 'n/a'

collias_mb_df['Region'] = 'Main Basin'


# note I created the region_5 file for ecology from parker's ps_longterm that i have saved locally

ecology_hc_df = pd.read_pickle('/Users/dakotamascarenas/Desktop/ps_longterm/ptools_output/ecology/region_5.p')

ecology_hc_df['Data Source'] = 'Ecology'

ecology_hc_df['Region'] = 'Hood Canal'


ecology_mb_df = pd.read_pickle('/Users/dakotamascarenas/Desktop/ps_longterm/ptools_output/ecology/region_3.p')

ecology_mb_df['Data Source'] = 'Ecology'

ecology_mb_df['Region'] = 'Main Basin'

# %%


df = pd.concat([collias_hc_df, collias_mb_df, ecology_hc_df, ecology_mb_df])

# %%

df.to_csv('/Users/dakotamascarenas/Desktop/collias_ecology_hood_canal_DO_CT_SA_NO3_depth-binned.csv', index=True)