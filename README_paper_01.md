# Data & Scripts for: Century-Scale Changes in Temperature, Salinity, and Dissolved Oxygen in Puget Sound

Dataset DOI: [10.5061/dryad.612jm64g7](10.5061/dryad.612jm64g7)

## Description of the data and file structure

These data accompany a paper submitted to Geophysical Research Letters. The abstract is below.

Over the last century, coastal oceans are losing ecosystem-critical dissolved oxygen (DO). Changing coastal ocean conditions influence water properties in connected estuaries, which are also influenced by anthropogenic processes. We analyze nearly 100 years of temperature, salinity, and DO data in Puget Sound - a temperate, fjordal, urbanized estuary in Washington State, USA and the southernmost region of the Salish Sea - during fall at the bottom of the water column. We observe warming of about 1.4°C/century, consistent with regional trends and similar to coastal ocean and local atmospheric warming. Salinity is generally increasing, possibly due to shifting regional freshwater flow timing. Finally, central Puget Sound is losing DO at a rate of 0.3-0.9 mg/L/century. Warming-driven changes in DO surface saturation account for approximately 40-100% of this DO loss. Distal inlets in Puget Sound are warming similarly, but variability exceeds the measured DO trends.

Data from various agencies were used in the creation of this manuscript. Data were compiled and collected by Eugene Collias at the University of Washington (Collias, 1970; Collias & Lincoln, 1977). The Washington State Department of Ecology (2024) and King County (2024a, 2024b, 2024c) historical and modern monitoring datasets were used. NOAA National Center for Environmental Information (NCEI)’s Salish cruise datasets were also used, as part of the Ocean Carbon and Acidification Data System (OCADS) (Alin et al., 2021). Further, air temperature data from NOAA Global Historical Climate Network (GHCN) (2018) and river discharge data from USGS (2024) were used. (Please see manuscript for formal citations.)

## Files and variables

### Scripts - RUN THESE!

#### File: figure\_1.py

**Description:** Processes dataframes and plots manuscript Figure 1.

#### File: figure\_2.py

**Description:** Processes dataframes and plots manuscript Figure 2.

#### File: figure\_3.py

**Description:** Processes data and dataframes and plots manuscript Figure 3.

#### File: figure\_4.py

**Description:** Processes data and dataframes and plots manuscript Figure 4.

#### File: figure\_functions.py

**Description:** Functions for data and dataframe processing and plotting.

####

### LiveOcean grid attributes (used for plotting and bathymetry)

(MacCready et al., 2021 - see manuscript for full citation; see [https://github.com/parkermac/LO.git](https://github.com/parkermac/LO.git) for public repository)

#### File: X.p

**Description:** 2D longitude grid (ndarray) for LiveOcean cas7 domain

#### File: Y.p

**Description:** 2D latitude grid (ndarray) for LiveOcean cas7 domain

#### File: plon.p

**Description:** plaid 2D longitude grid (ndarray) for LiveOcean cas7 domain, used for Matplotlib pcolormesh plotting

#### File: plat.p

**Description:** plaid 2D latitude grid (ndarray) for LiveOcean cas7 domain, used for Matplotlib pcolormesh plotting

#### File: coast\_pnw\.p

**Description:** 2D points for graphical LiveOcean domain coastline, used for graphics

#### File: zm\_inverse.p

**Description:** 2D grid (ndarray) masking out "water" cells in the LiveOcean cas7 domain, used for graphics

###

### Air Temperature & River Data

#### File: 1527887.csv

**Description:** Daily minimum and maximum temperatures at Seattle-Tacoma International Airport (GHCND:USW00024233) (NOAA GHCN, 2018; see text for more details and citations), used in figure_3.py. Only variables used in figure_3.py are discussed below.

##### Variables

* DATE = measurement date
* TMAX = daily maximum temperature (degrees C)
* TMIN = daily minimum temperature (degrees C)
* STATION = agency-specified site identifier
* NAME = station name
* PRCP = daily precipitation (mm)
* TSUN = daily total sunshine (minutes)

#### File: skagit\_monthly.txt

**Description:** Monthly average discharge at Skagit River near Mount Vernon (USGS 12200500) (USGS, 2024; see text for more details and citations), used in figure_4.py. Original header metadata is included in this file. Only variables used in figure_4.py are discussed below.

##### Variables

* year_nu = year
* month_nu = month number
* mean_va = mean monthly discharge in cubic feet per second (cfs)
* agency_cd = code for sampling agency
* site_no = agency-specified site number
* parameter_cd = agency-specified code for measured variable (in this case, mean monthly discharge in cfs)
* ts_id = agency-specified time series identifier

### Dataframes+

Dataframes are "pickled" dataframes, meaning they are easily readable using Python Pandas' inherent functionality using "pickled" storage. Please see manuscript for data sources used to create dataframes.

#### File: ps\_casts\_DF.p

**Description:** This dataframe includes all shipboard profiles within Puget Sound considered in this work. Individual profiles are called "casts" and may have multiple data points at different depths along the profile. Data columns are discussed below.

##### Variables

* cid - unique cast identifier
* lon - longitude at which cast was recorded
* lat - latitude at which cast was recorded
* time - recorded cast time
* datetime - recorded cast time in DateTime format
* date_ordinal - recorded cast time in DateOrdinal format
* decade - recorded decade of cast (e.g., 1940s)
* year - recorded calendar year of cast
* season - recorded season of cast; seasons are defined in three-month trimesters and use the following shorthand (NOTE: shorthand is purely for coding readability and does not necessarily indicate scientific distinctions between seasons):
  * "loDO" = August-November
  * "winter" = December-March
  * "grow" = April-July
* month - recorded calendar month of cast
* yearday - recorded yearday of cast
* z - recorded data point depth [m]
* var - variable name, one of (see text for more description of conversions):
  * CT = conservative temperature in degrees C
  * SA = absolute salinity in g/kg
  * DO_mg_L = dissolved oxygen concentration [DO] in mg/L
* val - value (specified datapoint for the indicated variable)
* ix - x-index corresponding to longitude on the LiveOcean grid (see manuscript for citations and description)
* iy - y-index corresponding to latitude on the LiveOcean grid (see manuscript for citations and description)
* h - water column depth at specified index on the LiveOcean grid (see manuscript for citations and description)

#### File: site\_depth\_avg\_var\_DF.p

**Description:** This dataframe includes all depth-averaged cast data for the (5) selected sites with sufficient data for century-scale trend analyses considered in this work. Column nomenclature is the same as above; new columns are discussed below.

##### New Variables

* site - name of selected site
* surf_deep - depth bin
  * surf - depth-average using top 5m of water column
  * deep - depth-average using site-specific bottom percentage of water column (see manuscript for details)
* min_segment_h - greatest water column depth within site using the LiveOcean grid (see manuscript for citations and description)

#### File: site\_polygon\_dict.p

**Description:** Not really a dataframe, rather a "pickled" dictionary of Matplotlib Path objects that bound each site using polygons; this is used for plotting in figure_1.py.

## Code/software

Data processing and figure creation was conducted using Python 3.11.11 and the following packages: Matplotlib 3.10.1 (Hunter, 2007; The Matplotlib Development Team, 2025), Numpy 2.1.3 (Harris et al., 2020), Scipy 1.15.25 (Virtanen et al., 2020), Pandas 2.2.3 (The pandas development team, 2020), Pickle 4.0 (Van Rossum, 2020), and TEOS 10/Gibbs Seawater (GSW) Oceanographic Toolbox (McDougall & Barker, 2011). Grid and plotting tools adapted in this work from LiveOcean (MacCready et al., 2021) can be found [https://github.com/parkermac/LO.git](https://github.com/parkermac/LO.git). (Please see manuscript for formal citations.)
