This file explains the dataframes used to produce figures for XXXXXXX. Dataframes are "pickled" dataframes, meaning they are easily readable using Python Pandas' inherent functionality using "pickled" storage. Please see paper references for data sources.

1. puget_sound_casts_DF.p - This dataframe includes all shipboard profiles within Puget Sound considered in this work. Individual profiles are called "casts" and may have multiple data points at different depths along the profile. Data columns are as follows:
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
   * ix - x-index corresponding to longitude on the LiveOcean grid (see text for citations and description)
   * iy - y-index corresponding to latitude on the LiveOcean grid (see text for citations and description)
   * h - water column depth at specified index on the LiveOcean grid (see text for citations and description)


