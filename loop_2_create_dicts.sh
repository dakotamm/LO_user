#!/bin/bash
#Loop through to create data structures - manual and ANNOYING do better next time :)

#@author: dakotamascarenas



# note this doesn't handle different segmentations - for a later time

#for yr in {1930..2022}

for yr in 1931 1936 1949 1950 1955 1960 1961 1962 1967 1975 1976 1992 1993 1995 2001 2002 2005 2006 2007 2008 2009 2011 2012 2013 2014 2015 2016 2021 2022

do
	python ./create_dicts.py -gtx cas6_v0_live -year $yr -test False &
done
	

