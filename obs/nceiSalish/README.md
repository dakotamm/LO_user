# README for nceiSalish

Author: Dakota Mascarenas

Last updated: 2026/06/15

These files are from the compiled Salish cruise data product produced by NOAA and the Washington Ocean Acidification Center (WOAC): a compilation of 61 individual cruise data sets of sensor profile and discrete physical and biogeochemical measurements collected from a variety of research vessels in the southern Salish Sea and northern California Current System (Washington state marine waters) from 2008-02-04 to 2024-10-22.

Data package version v2025 (data file dated 09/07/2025) downloaded by Dakota Mascarenas from https://doi.org/10.25921/jgrz-v584.

Each row is a Niskin bottle firing on the CTD rosette upcast, pairing CTD-sensor measurements (temperature, salinity, oxygen) at the firing depth with discrete bottle samples (DIC, total alkalinity, nutrients, Winkler oxygen) analyzed in the lab. There is no separate bottle/CTD file or row-level flag; the two are distinguished only by column naming (`CTD*` = sensor, everything else measured = discrete bottle).

NOTE: Original processing script for earlier data versions authored by Parker MacCready.

NOTE: Data type is `bottle_ctd`, reflecting that each record combines CTD-sensor T/S/O2 with discrete bottle chemistry from the same rosette firing.

NOTE: Dissolved oxygen is taken from the `RECOMMENDED_OXYGEN_UMOL_KG` column, the data producers' merged best-estimate following Jiang et al. (2021): discrete Winkler oxygen is preferred where acceptable, with bottle-adjusted CTD-sensor oxygen used as fallback. The raw and bottle-adjusted CTD oxygen columns (`CTDOXY_UMOL_KG`, `CTDOXY_UMOL_KG_ADJ`) are read but not retained.

NOTE: WOCE quality-flag screening is applied before processing. Measured values are kept only when their `*_FLAG_W` flag is acceptable-grade and otherwise masked to NaN: T/S keep 2; oxygen keeps 2, 6, 7, 8 (6 = replicate mean; 7/8 = near-surface codes the producers define as analogous to 2/6); TA/DIC keep 2, 6; nutrients keep 2. Flags 3 (questionable), 4 (bad), 5 (not reported), and 9 (not sampled) are dropped.

NOTE: Timestamps are UTC (from the data product's `DATE_UTC`/`TIME_UTC` fields) and are tz-naive in the output. Hour=24 midnight-rollover entries are rolled forward one day during parsing.

NOTE: GSW conversions applied are from practical salinity (SP) to absolute salinity (SA) and in-situ temperature to conservative temperature (CT) using the `gsw` library.

NOTE: DO, TA, and DIC are converted from umol/kg to uM using in-situ density (x rho/1000). Nutrients are reported as the per-liter (umol/L) variants from the data product.

NOTE: This data product is the same WOAC/Salish cruise program processed under `prism`. The `prism` source holds the downcast continuous CTD profiles (1998-2018, T/S/sigma-t/O2 only); `nceiSalish` holds the upcast bottle-summary with discrete chemistry (2008-2024). In the overlap years 2008-2018 they share the same cruises and stations, so their CT/SA/DO are duplicate profiles and should not both be used for the shared physics.

bottle_ctd data availability:
* CT: 2008-2024
* SA: 2008-2024
* DO: 2008-2024
* DIC: 2008-2024
* TA: 2008-2024
* NO3: 2008-2024
* NO2: 2008-2024
* NH4: 2008-2024
* PO4: 2008-2024
* SiO4: 2008-2024
