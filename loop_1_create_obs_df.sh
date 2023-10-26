#!/bin/bash
#Loop through to create data structures - manual and ANNOYING do better next time :)

#@author: dakotamascarenas




for src in ecology dfo1 nceiSalish collias
do
    
    for tp in ctd bottle
	do
        
        for yr in {1930..2022}
		
		do
		
            python ./create_obs_df.py -gtx cas6_v0_live -source $src -otype $tp -year $yr -test False &
			
		done
		
	done
	
done
