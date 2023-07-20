"""
IDK YET

Test on mac in ipython:
run create_obs_df -gtx cas6_v0_live -source ecology -otype ctd -year 2002 -test False

"""

from lo_tools import Lfun
from lo_tools import extract_argfun as exfun

import VFC_functions as vfun

from pathlib import PosixPath


# %%


Ldir = exfun.intro() # this handles the argument passing


# %%


info_df_dir = (Ldir['LOo'] / 'obs' / 'vfc')

df_dir = (Ldir['LOo'] / 'obs' / 'vfc' )

Lfun.make_dir(info_df_dir, clean=False)

Lfun.make_dir(df_dir, clean=False)


# %%

# if Ldir['lo_env'] == 'dm_mac':

info_fn_in = Ldir['LOo'] / 'obs' / Ldir['source'] / Ldir['otype'] / ('info_' + str(Ldir['year']) + '.p')

fn_in = Ldir['LOo'] / 'obs' / Ldir['source'] / Ldir['otype'] / (str(Ldir['year']) + '.p')
    
# elif Ldir['lo_env'] == 'dm_perigee':
    
#     info_fn_in = PosixPath('/data1/parker/LO_output/obs/' + Ldir['source'] + '/' + Ldir['otype'] + '/info_' + str(Ldir['year']) + '.p')
    
#     fn_in = PosixPath('/data1/parker/LO_output/obs/' + Ldir['source'] + '/' + Ldir['otype'] + '/' + str(Ldir['year']) + '.p')
    
    # UNLESS YOU MAKE YOUR OWN - CHECK THIS
    
if info_fn_in.exists() & fn_in.exists():

    info_fn = (info_df_dir / ('info_' + str(Ldir['year']) + '.p'))

    info_df = vfun.buildInfoDF(Ldir, info_fn_in, info_fn)
    
    fn = (df_dir / (str(Ldir['year']) + '.p'))

    df = vfun.buildDF(Ldir, fn_in, fn, info_df)

            
