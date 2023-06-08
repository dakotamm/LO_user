"""
Loop through to create data structures - manual and ANNOYING do better next time :)

@author: dakotamascarenas
"""

from lo_tools import Lfun
from lo_tools import extract_argfun as exfun

import VFC_functions as vfun

from pathlib import Path, PosixPath

import os


years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]

sources = ['ecology', 'dfo1', 'nceiSalish']

types = ['ctd', 'bottle']


for src in sources:
    
    for tp in types:
        
        for yr in years:

            os.system('run create_obs_data_structures -gtx cas6_v0_live -source ecology -otype ctd -year 2017 -test False')
