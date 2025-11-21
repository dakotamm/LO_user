

from lo_tools import Lfun
from lo_tools import extract_argfun as exfun

import pandas as pd

from pathlib import PosixPath



Ldir = Lfun.Lstart() # this handles the argument passing


info_fn_in = Ldir['LOo'] / 'obs' / 'kc_whidbeyBasin' / 'ctd' / 'info_2022.p'

fn_in = Ldir['LOo'] / 'obs' / 'kc_whidbeyBasin' / 'ctd' / '2022.p'


info_df = pd.read_pickle(info_fn_in)

df = pd.read_pickle(fn_in)



            
