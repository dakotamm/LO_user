"""
Turn the hand-drawn lines in LO_output/section_lines/*.p into a tef2 section
collection, so the standard tef2 workflow can be run on them.

The hand-drawn files and the tef2 collection files are the same thing -- a
pickled DataFrame with columns ['x','y'] -- so this is really just a filtered
copy plus the bounding_sections.txt that create_seg_info_dict.py needs.

Which lines get copied:
  - open polylines only. Closed ones (first point == last point) are regions
    (wb, pc, wb_north, skagit_delta), not sections.
  - *_backup_* files are skipped.
  - anything in SKIP below is skipped.

Bounding sections are the outer ends of the region of interest. No segment is
seeded from them, but they still act as walls for the fill, so they are what
keeps it from escaping to the open N, S and W edges of the wb1 grid.

For pc1 the region of interest is Penn Cove plus the reach of Saratoga Passage
it opens onto, so the bounds are skagit_sp (north) and sp_mid (south).

deception_pass and to_mb are NOT used. They do not seal Whidbey Basin on this
grid: Skagit Bay has a second northern exit through the Swinomish Channel at
about lon -122.51, and the fill also gets around to_mb, so a collection bounded
by those two leaks out the north edge of the grid (checked 2026.08.04 with
20260804_check_seg_closure.py). Bounding on skagit_sp/sp_mid seals.

run 20260804_make_tef2_collection.py
run 20260804_make_tef2_collection.py -ctag pc1 -skip pc_ew,deception_pass,to_mb
"""
import argparse
import shutil

import numpy as np
import pandas as pd

from lo_tools import Lfun

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gridname', default='wb1', type=str)
parser.add_argument('-ctag', default='pc1', type=str)
# lines to leave out of the collection
parser.add_argument('-skip', default='pc_ew,deception_pass,to_mb', type=str,
                    help='comma-separated section names to skip')
parser.add_argument('-bounding', default='skagit_sp,sp_mid', type=str,
                    help='comma-separated outer sections, written to '
                         'bounding_sections.txt')
parser.add_argument('-keep_diagonal', default='skagit_sp', type=str,
                    help='comma-separated sections to leave as drawn; every '
                         'other section is snapped to pure N-S or E-W')
args = parser.parse_args()

Ldir = Lfun.Lstart(gridname=args.gridname)

SKIP = [s for s in args.skip.split(',') if s]
KEEP_DIAGONAL = [s for s in args.keep_diagonal.split(',') if s]


def straighten(df):
    """Snap a nearly-axis-aligned line to exactly N-S or E-W.

    A line drawn by hand is off-axis by a fraction of a grid cell, which is
    enough for the stairstep in create_sect_df.py to put one stray face on the
    other grid (a lone v-face in an otherwise all-u section, say). That face
    carries a different pm from the rest, so the section stops being a clean
    "flow through a constant-longitude wall" and the sign bookkeeping gets
    harder to reason about for no benefit.

    Orientation is decided by comparing the spans in metres, not degrees --
    at 48 N a degree of longitude is about two thirds of a degree of latitude.
    Returns (df, what_changed).
    """
    dx = (df.x.max() - df.x.min()) * np.cos(np.deg2rad(df.y.mean()))
    dy = df.y.max() - df.y.min()
    df = df.copy()
    if dx < dy:  # runs mostly north-south -> constant longitude
        old = df.x.max() - df.x.min()
        df['x'] = df.x.mean()
        return df, 'N-S, lon -> %.6f (was spread over %.6f deg)' % (df.x.iloc[0], old)
    else:        # runs mostly east-west -> constant latitude
        old = df.y.max() - df.y.min()
        df['y'] = df.y.mean()
        return df, 'E-W, lat -> %.6f (was spread over %.6f deg)' % (df.y.iloc[0], old)

# outer ends of the region -- the fill stops here. NOTE these are still
# extracted and still used in the analysis; "bounding" only means that no
# segment is seeded from them in create_seg_info_dict.py.
BOUNDING = [s for s in args.bounding.split(',') if s]

in_dir = Ldir['LOo'] / 'section_lines'
gctag = args.gridname + '_' + args.ctag
out_dir = Ldir['LOo'] / 'extract' / 'tef2' / ('sections_' + gctag)
Lfun.make_dir(out_dir, clean=True)

sn_list = []
for fn in sorted(in_dir.glob('*.p')):
    sn = fn.stem
    if 'backup' in sn or sn in SKIP:
        continue
    df = pd.read_pickle(fn)
    closed = (abs(df.x.iloc[0] - df.x.iloc[-1]) < 1e-9 and
              abs(df.y.iloc[0] - df.y.iloc[-1]) < 1e-9)
    if closed:
        print('%-16s closed -- region, skipping' % sn)
        continue
    if sn in KEEP_DIAGONAL:
        shutil.copyfile(fn, out_dir / (sn + '.p'))
        print('%-16s %d points -> collection, left as drawn' % (sn, len(df)))
    else:
        df, how = straighten(df)
        df.to_pickle(out_dir / (sn + '.p'))
        print('%-16s %d points -> collection, straightened %s' % (sn, len(df), how))
    sn_list.append(sn)

missing = [sn for sn in BOUNDING if sn not in sn_list]
if len(missing) > 0:
    raise SystemExit('bounding sections missing from collection: %s' % missing)

with open(out_dir / 'bounding_sections.txt', 'w') as f:
    f.write('\n'.join(BOUNDING))

print('\n%d sections in %s' % (len(sn_list), out_dir))
print('bounding: ' + ', '.join(BOUNDING))
print('interior: ' + ', '.join([s for s in sn_list if s not in BOUNDING]))
