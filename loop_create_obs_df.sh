#!/bin/bash
#Loop through to create data structures - manual and ANNOYING do better next time :)

#@author: dakotamascarenas




for src in ecology dfo1 nceiSalish
do
    
    for tp in ctd bottle
	do
        
        for yr in 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017
		
		do
		
            run create_obs_data_structures -gtx cas6_v0_live -source $src -otype $tp -year $yr -test False
			
		done
		
	done
	
done
