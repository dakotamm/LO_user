"""
Synthetic test for track_eddies.py.

Generates a fake per-snapshot vortex CSV with KNOWN tracks, then runs
the tracker and verifies the recovered tracks match expectations.

The synthetic scenario (10 snapshots):
  Track A: CCW eddy drifting NE, persists snapshots 0-9 (10 detections)
  Track B: CW eddy near a fixed location, persists 2-7 (6 detections)
  Track C: CCW short-lived, snapshots 4-5 only (2 detections)
  Track D: CCW with a 1-snap gap: snapshots 0,1,3,4 (gap at 2)
  Noise:   one stray detection far from anything at snapshot 6

Expected with -max_dist_km 1.0 -max_gap 1 -min_persistence 2:
  - 4 tracks survive (A, B, C, D)
  - noise singleton dropped
  - Track A duration = 10 snapshots, Track D should bridge the gap

Usage
-----
    cd /Users/dakotamascarenas/LO_user/swirl
    python test_track_eddies.py
"""

import sys
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd


HERE = Path(__file__).parent
TEST_DIR = HERE / '_test_tracking'
TEST_DIR.mkdir(exist_ok=True)


def make_synthetic_csv():
    """Build a CSV that mimics run_swirl_roms.py output."""
    rng = np.random.default_rng(42)
    rows = []

    # Track A: CCW, drifting NE  ~0.05 km per snapshot
    A0_lon, A0_lat = -122.70, 48.22
    for s in range(10):
        rows.append(dict(
            date='2024.01.01', file_type='his', file_num=s,
            vel_type='surface', s_level=np.nan, vortex_id=0,
            center_eta_idx=10, center_xi_idx=20,
            center_lon=A0_lon + s * 0.0006,
            center_lat=A0_lat + s * 0.0004,
            radius_grid=3.0, radius_m=600.0,
            orientation=0.0, rotation=1,  # CCW
            mean_vorticity=1e-4, max_vorticity=2e-4, n_cells=20,
        ))

    # Track B: CW, fixed at another location, snapshots 2-7
    for s in range(2, 8):
        rows.append(dict(
            date='2024.01.01', file_type='his', file_num=s,
            vel_type='surface', s_level=np.nan, vortex_id=1,
            center_eta_idx=15, center_xi_idx=40,
            center_lon=-122.65 + rng.normal(0, 0.0001),
            center_lat=48.24 + rng.normal(0, 0.0001),
            radius_grid=2.5, radius_m=500.0,
            orientation=0.0, rotation=-1,  # CW
            mean_vorticity=-1e-4, max_vorticity=-2e-4, n_cells=15,
        ))

    # Track C: CCW, short-lived, snapshots 4-5
    for s in (4, 5):
        rows.append(dict(
            date='2024.01.01', file_type='his', file_num=s,
            vel_type='surface', s_level=np.nan, vortex_id=2,
            center_eta_idx=8, center_xi_idx=60,
            center_lon=-122.62, center_lat=48.21,
            radius_grid=2.0, radius_m=400.0,
            orientation=0.0, rotation=1,
            mean_vorticity=1e-4, max_vorticity=2e-4, n_cells=12,
        ))

    # Track D: CCW with a gap at snapshot 2 (present 0,1,3,4)
    D_lon, D_lat = -122.72, 48.25
    for s in (0, 1, 3, 4):
        rows.append(dict(
            date='2024.01.01', file_type='his', file_num=s,
            vel_type='surface', s_level=np.nan, vortex_id=3,
            center_eta_idx=20, center_xi_idx=10,
            center_lon=D_lon, center_lat=D_lat,
            radius_grid=2.0, radius_m=400.0,
            orientation=0.0, rotation=1,
            mean_vorticity=1e-4, max_vorticity=2e-4, n_cells=10,
        ))

    # Noise: stray singleton far away
    rows.append(dict(
        date='2024.01.01', file_type='his', file_num=6,
        vel_type='surface', s_level=np.nan, vortex_id=4,
        center_eta_idx=2, center_xi_idx=2,
        center_lon=-122.50, center_lat=48.10,  # far from everyone
        radius_grid=1.5, radius_m=300.0,
        orientation=0.0, rotation=1,
        mean_vorticity=1e-4, max_vorticity=2e-4, n_cells=8,
    ))

    df = pd.DataFrame(rows)
    csv_path = TEST_DIR / 'ow_vortices_test.csv'
    df.to_csv(csv_path, index=False)
    print(f'Wrote synthetic CSV: {csv_path} ({len(df)} detections)')
    return csv_path


def run_tracker(csv_path):
    """Invoke track_eddies.py as a subprocess."""
    cmd = [
        sys.executable, str(HERE / 'track_eddies.py'),
        '-csv', str(csv_path),
        '-max_dist_km', '1.0',
        '-max_gap', '1',
        '-min_persistence', '2',
    ]
    print('\n$ ' + ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print('STDERR:', result.stderr)
        sys.exit(1)


def verify(csv_path):
    """Check the tracker output against expectations."""
    stem = csv_path.stem
    tracks = pd.read_csv(csv_path.parent / f'{stem}_tracks.csv')
    tracked = pd.read_csv(csv_path.parent / f'{stem}_tracked.csv')

    print('\n=== Tracks summary ===')
    print(tracks[['track_id', 'n_detections', 'span_snapshots',
                  'rotation', 'mean_lon', 'mean_lat']].to_string(index=False))

    failures = []

    # Expectation 1: exactly 4 tracks survive
    if len(tracks) != 4:
        failures.append(f'Expected 4 tracks, got {len(tracks)}')

    # Expectation 2: longest track has 10 detections (Track A)
    if tracks['n_detections'].max() != 10:
        failures.append(
            f'Expected max persistence 10, got {tracks["n_detections"].max()}')

    # Expectation 3: Track D bridged its gap -> span_snapshots == 5, n == 4
    d_match = tracks[(tracks['n_detections'] == 4)
                     & (tracks['span_snapshots'] == 5)]
    if len(d_match) != 1:
        failures.append('Track D (4 detections spanning 5 snapshots) not found')

    # Expectation 4: noise singleton dropped (no track with 1 detection)
    if (tracks['n_detections'] == 1).any():
        failures.append('Singleton track was not filtered out')

    # Expectation 5: rotation signs preserved
    if not set(tracks['rotation'].unique()).issubset({-1, 1}):
        failures.append('Unexpected rotation values in tracks')

    # Expectation 6: total tracked detections = 10 + 6 + 2 + 4 = 22
    if len(tracked) != 22:
        failures.append(f'Expected 22 tracked detections, got {len(tracked)}')

    if failures:
        print('\nFAIL:')
        for f in failures:
            print('  -', f)
        sys.exit(1)
    else:
        print('\nAll checks PASSED.')


if __name__ == '__main__':
    csv_path = make_synthetic_csv()
    run_tracker(csv_path)
    verify(csv_path)
