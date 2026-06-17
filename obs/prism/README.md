# README for prism

Author: Dakota Mascarenas

Last updated: 2026/06/15

These files are the downcast CTD subset of the full Salish Cruise CTD data set, compiled by Dakota Mascarenas from the NANOOS NVS (Northwest Association of Networked Ocean Observing Systems, Visualization System) data archive. These are the same southern Salish Sea / Washington Ocean Acidification Center (WOAC) research-vessel cruises archived in the `nceiSalish` source, here as continuous downcast CTD profiles.

Source file: `SalishCruise_downcast_CTDdata_121998to092018_TSstO2subset.xlsx` (sheet: `all data`), spanning December 1998 to September 2018. Received via email on 2026/06/01 by Dakota Mascarenas from Simone Alin.

Each cast is a continuous downcast CTD profile. Station names are built as `CRUISE_STN` (e.g., `CAB1045_4`); cast IDs are assigned per unique profile timestamp, which is robust to rows missing a station number.

NOTE: Data type is `ctd` (downcast continuous profiles), as opposed to the upcast bottle-summary in `nceiSalish`.

NOTE: Although the source file is a T / S / sigma-t / CTD-oxygen subset, only temperature and salinity are processed and retained here (output variables CT and SA). Sigma-t and CTD oxygen are not carried through.

NOTE: Timestamps are research-vessel logs taken to be in UTC and are timezone-aware UTC in the output.

NOTE: GSW conversions applied are from practical salinity (SP) to absolute salinity (SA) and in-situ temperature to conservative temperature (CT) using the `gsw` library. Pressure is computed from depth (`DEPTH`, converted to negative z) via `gsw.p_from_z`.

NOTE: This data product is the same WOAC/Salish cruise program processed under `nceiSalish`. `prism` holds the downcast continuous CTD profiles (1998-2018, T/S only as processed); `nceiSalish` holds the upcast bottle-summary with discrete chemistry (2008-2024). In the overlap years 2008-2018 they share the same cruises and stations, so their CT/SA are duplicate profiles and should not both be used for the shared physics. Non-redundant coverage: `prism` alone spans 1998-2007; `nceiSalish` alone spans 2019-2024 and adds all DO and discrete chemistry.

ctd data availability:
* CT: 1998-2018
* SA: 1998-2018
