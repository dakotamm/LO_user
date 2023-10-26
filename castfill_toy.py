"""
An example of filling in missing parts of a cast using nearest neighbor.

For Dakota, October 2023.

We use two methods, and they give identical results.
Using scipy.interpolate.interp1d is probably easiest.

"""

import numpy as np
from lo_tools import zfun

# create some z bins and some gappy data
z_edges = np.linspace(-100,0,12)
z = z_edges[:-1] + np.diff(z_edges)/2
d = np.nan * np.ones(len(z))
d[np.array([2,3,7,8])] = np.array([40,35,20,10])
# because of the data gaps we are going to need to fill in 
# missing internal values, and extrapolate top and bottom values.

# indices of bad and good data
ibad = np.argwhere(np.isnan(d) == True)[:,0]
igood = np.argwhere(np.isnan(d) == False)[:,0]
# Note that argwhere returns an array of shape (7,1) for ibad in this
# example, so we use [:,0] to just get a vector of length 7,
# which is needed for theindexing below.

# z vectors of bad and good data
zbad = z[ibad]
zgood = z[igood]

# z vector of good data
dgood = d[igood]

# Method 1: do it by hand
dd1 = d.copy() # array we are going to fill
for zz in zbad:
    # filling in bad data with good data
    iii = zfun.find_nearest_ind(z,zz)
    ii = zfun.find_nearest_ind(zgood,zz)
    dd1[iii] = dgood[ii]
    
# Method 2: use interp1d
from scipy.interpolate import interp1d
f = interp1d(zgood, dgood, kind='nearest', fill_value='extrapolate')
dd2 = f(z)

# RESULT: dd1 and dd2 were identical
print(d)
print(dd1)
print(dd2)