#!/bin/bash
#Loop through to create data structures - manual and ANNOYING do better next time :)

#@author: dakotamascarenas



# note this doesn't handle different segmentations - for a later time

#for yr in {1930..2022}

for yr in 1930 1933 1941 1959 1950 1951 1955 1961 1962 1974 1975 1978 1988 1992 1993 1995 1996 1997 2001 2002 2004 2006 2015 2022

do
	python ./create_dicts_temp.py -gtx cas6_v0_live -year $yr -test False &
done
	

