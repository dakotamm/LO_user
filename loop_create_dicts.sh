#!/bin/bash
#Loop through to create data structures - manual and ANNOYING do better next time :)

#@author: dakotamascarenas



# note this doesn't handle different segmentations - for a later time

#for yr in 2000 2001 2002 2003 2004 2005 2006 2007 2018 2019 2020 2021

for yr in 2021

do
	python ./get_extracted_dicts.py -gtx cas6_v0_live -year $yr -test False &
done
	

