#!/bin/bash
#Loop through to create data structures - manual and ANNOYING do better next time :)

#@author: dakotamascarenas



# note this doesn't handle different segmentations - for a later time

for yr in {1930..2022}

do
	python ./make_vol_dfs.py -gtx cas6_v0_live -year $yr -test False &
	
done
	

