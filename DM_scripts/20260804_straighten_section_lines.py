"""
Straighten hand-drawn lines in LO_output/section_lines/ to exactly N-S or E-W,
in place.

A line drawn by hand sits a fraction of a grid cell off axis. That is enough
for the stairstep in create_sect_df.py to drop one stray face onto the other
grid -- a lone v-face in an otherwise all-u section, say. The stray face
carries a different pm from the rest of the section, so the section stops being
a clean "flow through a constant-longitude wall" and the sign bookkeeping gets
harder to reason about for no benefit.

Snapping to the mean longitude (or latitude) moves each endpoint by less than
half a grid cell, so the section stays where it was drawn.

Only the sections named in -sections are touched. Genuinely diagonal lines
(skagit_sp, to_mb) must NOT be passed here -- snapping those would distort the
line rather than tidy it.

Every file that changes is backed up alongside itself as
<name>_prestraighten_backup_<date>.p first. Files with "backup" in the name are
ignored by 20260804_make_tef2_collection.py, so the backups will not turn into
sections.

run 20260804_straighten_section_lines.py -test True     # show, change nothing
run 20260804_straighten_section_lines.py
"""
import argparse
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

from lo_tools import Lfun

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gridname', default='wb1', type=str)
parser.add_argument('-sections', default='pc_lp,pc_lj,pc_cp,sp_mid', type=str,
                    help='comma-separated; only these are straightened')
parser.add_argument('-test', '--testing', default=False, type=Lfun.boolean_string,
                    help='True to report what would change without writing')
args = parser.parse_args()

Ldir = Lfun.Lstart(gridname=args.gridname)
sect_dir = Ldir['LOo'] / 'section_lines'
sn_list = [s for s in args.sections.split(',') if s]
dstr = datetime.now().strftime('%Y%m%d')

for sn in sn_list:
    fn = sect_dir / (sn + '.p')
    if not fn.is_file():
        print('%-10s MISSING %s' % (sn, fn))
        continue

    df = pd.read_pickle(fn)

    # decide orientation in metres, not degrees -- at 48 N a degree of
    # longitude is about two thirds of a degree of latitude
    dx = (df.x.max() - df.x.min()) * np.cos(np.deg2rad(df.y.mean()))
    dy = df.y.max() - df.y.min()

    # refuse to touch anything that is not clearly axis-aligned already
    if min(dx, dy) > 0.25 * max(dx, dy):
        print('%-10s SKIPPED -- too diagonal to snap (dx %.4f, dy %.4f in deg-equiv)'
              % (sn, dx, dy))
        continue

    new = df.copy()
    if dx < dy:
        spread = df.x.max() - df.x.min()
        new['x'] = df.x.mean()
        how = 'N-S, lon -> %.6f' % new.x.iloc[0]
    else:
        spread = df.y.max() - df.y.min()
        new['y'] = df.y.mean()
        how = 'E-W, lat -> %.6f' % new.y.iloc[0]

    print('%-10s %s  (endpoints moved <= %.6f deg)' % (sn, how, spread / 2))
    for a, b in zip(df.itertuples(), new.itertuples()):
        print('             (%.6f, %.6f) -> (%.6f, %.6f)' % (a.x, a.y, b.x, b.y))

    if args.testing:
        continue

    bak = sect_dir / ('%s_prestraighten_backup_%s.p' % (sn, dstr))
    if not bak.is_file():
        shutil.copyfile(fn, bak)
        print('             backed up -> %s' % bak.name)
    new.to_pickle(fn)

if args.testing:
    print('\n-test True: nothing written')
else:
    print('\ndone -- now rebuild the collection:')
    print('  run 20260804_make_tef2_collection.py -ctag pc1')
    print('  run create_sect_df.py -gctag wb1_pc1 -small True   (from LO/extract/tef2)')
