#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:57:07 2023

@author: dakotamascarenas
"""

import os

outdir = '/Users/dakotamascarenas/Desktop/pltz/'

fileprefix = 'sub_vol_pct_diff_5_mg_L_DO_errors_2017'

ff_str = ("ffmpeg -r 8 -i " + outdir + "'" + fileprefix + "_%04d.png' -vf 'crop=trunc(iw/2)*2:trunc(ih/2)*2' -vcodec libx264 -pix_fmt yuv420p -crf 25 " + outdir + "'" + fileprefix + "_temp.mp4'")
os.system(ff_str)

ff_str = ("ffmpeg -i " + outdir + "'" + fileprefix + "_temp.mp4' -vf 'setpts=5*PTS' " + outdir + "'" + fileprefix + ".mp4'")
os.system(ff_str)