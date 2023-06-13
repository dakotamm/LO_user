#!/bin/bash


for yr in 2000 2001 2002 2003 2004 2005 2006 2007 2018 2019 2020 2021 2022
do
	python ./get_extracted_dicts.py -gtx cas6_v0_live -year $yr -test False &
	python ./make_vol_dfs.py -gtx cas6_v0_live -year $yr -test False &
done
	