# README for kc_his

Author: Dakota Mascarenas

Last updated: 2026/06/23

These files are from King County's monitoring of the water column near Point Jefferson, both bottle samples at now-defunct historical stations.

Data was received via email from Taylor Martin, King County, to Dakota Mascarenas on 2024/03/27.

Email correspondences is included in the corresponding data folder for methods information: "his_methods_email - TM to DM 20250117.pdf"

NOTE: "field" data and temperature are from CTD and others are from bottle. I use temperature from CTD for bottles for ease of use in observation/model comparison. This is all included in "old_do_data.csv" despite the name.

NOTE: Quality control voids measurements flagged rejected ('R'), questionable ('q'/'Q'), or estimated ('E') via the raw data's alphanumeric QUALIFIER column (see DataReadMeFile_WQ.docx). Detection-limit qualifiers (<MDL, <RDL, RDL) mark valid censored low values and are retained. For the oceanographic variables processed here this currently voids nothing -- all R/E flags in the raw file fall on bacteria/organics parameters that are not used -- but the filter is applied for consistency with other King County sources.

NOTE: Timestamps in the raw data are mostly date-only (midnight), assumed to be in PST. The processing scripts convert these to timezone-aware UTC (+8 hours) for consistency with other LO observation sources.

CTD data availability:
* Chl: 1998-2000
* DO: 1974-1986, 1998-2000
* CT: 1965-2000 - this seems really early for CTDs but this is what appears here.
* SA: 1998-2000

Bottle data availability
* Chl: 1966-1970, 1972-1975, 1997-2000
* DO: 1965-1973, 1997-2000
* NH4: 1996-2000
* NO3: 1970-1975, 1997-2000
* PO4: 1997-2000
* SA: 1965-1986, 1997-2000
* SiO4: 1970-1975, 1997-2000