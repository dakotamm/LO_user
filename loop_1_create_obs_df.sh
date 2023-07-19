#!/bin/bash
#Loop through to create data structures - manual and ANNOYING do better next time :)

#@author: dakotamascarenas




for src in ecology dfo1 nceiSalish
do
    
    for tp in ctd bottle
	do
        
        for yr in {1930..2022}
		
		do
		
            python ./create_obs_data_structures.py -gtx cas6_v0_live -source $src -otype $tp -year $yr -test False &
			
		done
		
	done
	
done
