"""
IDK YET

Test on mac in ipython:
run create_obs_data_structures -gtx cas6_v0_live -source ecology -otype ctd -year 2017 -test False

"""

from lo_tools import Lfun
from lo_tools import extract_argfun as exfun

import VFC_functions as vfun


# %%


Ldir = exfun.intro() # this handles the argument passing

# %%


info_df_dir = (Ldir['LOo'] / 'obs' / 'vfc')

df_dir = (Ldir['LOo'] / 'obs' / 'vfc' )

Lfun.make_dir(info_df_dir, clean=False)

Lfun.make_dir(df_dir, clean=False)


# %%

info_fn_temp = Ldir['LOo'] / 'obs' / Ldir['source'] / Ldir['otype'] / ('info_' + str(Ldir['year']) + '.p')

info_fn = (info_df_dir / ('info_' + str(Ldir['year']) + '.p'))


info_df_temp, info_df = vfun.buildInfoDF(Ldir, info_fn_temp, info_fn)

# %%

fn_temp = Ldir['LOo'] / 'obs' / Ldir['source'] / Ldir['otype'] / (str(Ldir['year']) + '.p')

fn = (df_dir / (str(Ldir['year']) + '.p'))


df_temp, df = vfun.buildDF(Ldir, fn_temp, fn, info_df)

            
