#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:57:07 2023

@author: dakotamascarenas
"""

import os

outdir = '/Users/dakotamascarenas/Desktop/pltz/'

ff_str = ("ffmpeg -r 8 -i " + outdir + "sub_vol_pct_diff_5_mg_L_DO_errors_2017_%04d.png -vcodec libx264 -pix_fmt yuv420p -crf 25 " + outdir + "sub_vol_pct_diff_5_mg_L_DO_errors_2017_temp.mp4")
os.system(ff_str)

ff_str = ("ffmpeg -i " + outdir + "sub_vol_pct_diff_5_mg_L_DO_errors_2017_temp.mp4 -vf 'setpts=5*PTS' " + outdir + "sub_vol_pct_diff_5_mg_L_DO_errors_2017.mp4")
os.system(ff_str)