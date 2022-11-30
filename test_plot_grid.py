import matplotlib.pyplot as plt
import pandas as pd
import netCDF4 as ncdf

file = '/Users/dakotamascarenas/LO_user/grids/cas6/grid.nc'

nc = ncdf.Dataset(file)

print(nc.variables.keys())
