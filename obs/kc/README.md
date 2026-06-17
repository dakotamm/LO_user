# README for kc

Author: Dakota Mascarenas

Last updated: 2026/06/15

These files are from King County's marine offshore monitoring water column samples, both bottle and CTD.

Bottle data is publically-accessible and was downloaded with data up to April 2026 by Dakota Mascarenas from https://data.kingcounty.gov/Environment-Waste-Management/Water-Quality/vwmt-pvjw.

CTD data was received by Dakota Mascarenas from Greg Ikeda, King County, via file transfer on 2024/04/05 and updated in the same way on 2026/05/01.

Station information downloaded in April 2026 by Dakota Mascarenas from: https://data.kingcounty.gov/Environment-Waste-Management/WLRD-Sites/wbhs-bbzf

Sampling analysis plans and methods are included in the corresponding data folder.

Two email correspondences are attached for further metadata:
* "metadata_questions_email - GI to DM 20230726.pdf" - answers to specific unit/processing questions
* "his_methods_email - TM to DM 20250117.pdf" - explains sampling method change overtime especially for dissolved oxygen

NOTE: Timestamps in the raw data are assumed to be in PST. The processing scripts convert these to timezone-aware UTC (+8 hours) for consistency with other LO observation sources.

NOTE: This includes bottle data replicated in the folder kc_whidbeyBasin bottle data, but is included here since this is the format that King County maintains.

NOTE: Light transmission is currently not considered in this data processing. However, for future use, reference https://green2.kingcounty.gov/marine/Monitoring/OffshoreCTD: "Light Transmission data prior to May 19, 2014 were referenced to air. After this date, all Light Transmission data are referenced to water. To convert the pre-May 19, 2014 data to ‘referenced to water’, multiply the values by 1.095."

NOTE: The CTD data are quality-controlled in process_ctd.py using King County's quality flags. Each measurement column has a paired "*_Qual" flag column (e.g. DO_Qual, SA_Qual); see DataReadMeFile_WQ.docx for the full flag vocabulary. Individual measurements are voided (set to NaN) when their flag indicates the value was rejected ("R"/"Rej"/"rej"/"REJ"/"rj"), questionable ("q"/"Q"), or estimated ("E"). Blank flags (passed QC) and "TA" (text-available note only, not a quality issue) are retained. Flags are masked per-variable, so a cast can keep good salinity/temperature while its flagged DO is voided. Note: King County marks some downcasts rejected with a "[USE_UPCAST]" note directing use of the upcast; since only downcasts are processed here, those values become gaps rather than upcast substitutions.

CTD data availability (years with at least some data retained after QC):
* CT: 1998-2026
* Chl: 1998-2026
* DO: 1998-2026
* NO3: 2017-2026
* SA: 1998-2026

Bottle data availability
* CT: 1965-2026
* Chl: 1997-2026
* DIC: 2015
* DO: 1965-2026
* NH4: 1996-2026
* NO3: 1997-2026
* PO4: 1997-2010
* SA: 1965-2026
* SiO4: 1997-2026
* TA: 2015
