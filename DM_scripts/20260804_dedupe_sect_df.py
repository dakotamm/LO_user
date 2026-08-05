"""
Drop duplicate u/v faces from a tef2 sect_df.

Where two sections meet in a T junction, create_sect_df.py can hand the same
u- or v-face to both of them -- the stairstep of the line that ends at the
junction overshoots by one face onto the line it ends against. For wb1_pc1
that happens twice, where the along-cove line pc_ew meets pc_lp (at the Penn
Cove mouth) and pc_cp (off Coupeville).

A shared face is double counted: its transport shows up in both sections, and
part of it is not even normal to one of them. So keep the face for exactly one
section and drop it from the others.

Which section keeps it is set by PRIORITY: the cross-channel sections are the
primary ones and win, so the along-axis line pc_ew gives up the overlap. This
does not open a hole in the segment fill -- the face is still blocked, just by
the other section.

The original is backed up next to the output as sect_df_<gctag>_predupe.p.

run 20260804_dedupe_sect_df.py -gctag wb1_pc1
"""
import argparse
import shutil

import pandas as pd

from lo_tools import Lfun

parser = argparse.ArgumentParser()
parser.add_argument('-gctag', default='wb1_pc1', type=str)
# lowest priority last: these give up a face they share with anything above
parser.add_argument('-priority', default='pc_ew', type=str,
                    help='comma-separated, lowest-priority sections')
args = parser.parse_args()

gridname = args.gctag.split('_')[0]
Ldir = Lfun.Lstart(gridname=gridname)

tef2_dir = Ldir['LOo'] / 'extract' / 'tef2'
fn = tef2_dir / ('sect_df_' + args.gctag + '.p')
bak_fn = tef2_dir / ('sect_df_' + args.gctag + '_predupe.p')

loser_list = [s for s in args.priority.split(',') if s]

df = pd.read_pickle(fn)
print('%d faces before' % len(df))

face = ['uv', 'i', 'j']
dup = df.duplicated(subset=face, keep=False)
if not dup.any():
    print('no shared faces -- nothing to do')
    raise SystemExit

print('\nshared faces:')
print(df[dup].sort_values(face).to_string())

# a row is dropped if it is a loser AND some other section claims the same face
claimed_by_other = pd.Series(False, index=df.index)
for sn in loser_list:
    is_sn = df.sn == sn
    other_faces = set(map(tuple, df.loc[~is_sn, face].to_numpy()))
    mine = pd.Series(
        [tuple(r) in other_faces for r in df.loc[is_sn, face].to_numpy()],
        index=df.index[is_sn])
    claimed_by_other.loc[mine.index] = mine

if claimed_by_other.sum() == 0:
    raise SystemExit('shared faces exist but none belong to %s -- '
                     'check -priority' % loser_list)

print('\ndropping %d face(s) from %s' % (int(claimed_by_other.sum()), loser_list))

if not bak_fn.is_file():
    shutil.copyfile(fn, bak_fn)
    print('backed up original -> %s' % bak_fn.name)

df = df[~claimed_by_other].reset_index(drop=True)
df.to_pickle(fn)

still = df.duplicated(subset=face, keep=False)
print('%d faces after, %d still shared' % (len(df), int(still.sum())))
print(df.groupby('sn').size().to_string())
