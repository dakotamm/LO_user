"""
Check LO_output/obs/ecology_nc bottle and ctd data for 2024 to see if/where
nitrogen variables are present.
"""
import pickle
from pathlib import Path

base = Path('/Users/dakotamascarenas/LO_output/obs/ecology_nc')

for source in ['bottle', 'ctd']:
    fn = base / source / '2024.p'
    print(f'\n=== {source} 2024 ({fn}) ===')
    with open(fn, 'rb') as f:
        df = pickle.load(f)
    print(f'type: {type(df).__name__}, shape: {getattr(df, "shape", "n/a")}')
    cols = list(df.columns)
    print(f'all columns: {cols}')
    n_cols = [c for c in cols if 'no3' in c.lower()]
    print(f'NO3 columns: {n_cols}')
    for c in n_cols:
        s = df[c]
        nn = s.notna().sum()
        print(f'  {c}: non-null={nn}/{len(s)}, min={s.min()}, max={s.max()}, mean={s.mean()}')
