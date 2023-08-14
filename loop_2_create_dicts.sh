#!/bin/bash
#Loop through to create data structures - manual and ANNOYING do better next time :)

#@author: dakotamascarenas



# note this doesn't handle different segmentations - for a later time

#for yr in {1930..2022}

#for yr in 1930 1932 1933 1934 1937 1938 1939 1941 1948 1951 1952 1954 1956 1957 1958 1959 1963 1964 1965 1966 1968 1969 1970 1971 1972 1973 1974 1977 1978 1979 1980 1981 1982 1983 1984 1985 1986 1987 1988 1989 1990 1991 1994 1996 1997 1998 1999 2000 2003 2004 2010 2017 2018 2019 2020

for yr in {1930..2022}

do
	python ./create_dicts.py -gtx cas6_v0_live -year $yr -test False &
done
	

