#!/usr/bin/env python3
"""Build and run the isolated HCZ Quad2D OCCA raw troubled-cell detector comparison.

Usage: python3 test/testHczDetector.py [--modes Serial OpenMP CUDA]
Uses the repository's built OCCA library; no MPI or Python HCZ files needed.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modes', nargs='+', choices=('Serial', 'OpenMP', 'CUDA'),
                        default=('Serial', 'OpenMP'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='hcz_detector_') as directory:
        temp = Path(directory)
        binary = temp / 'hczDetectorQuad2D'
        subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra',
                        '-isystem', str(root / 'occa/include'), str(root / 'test/hczDetectorQuad2D.cpp'),
                        '-L' + str(root / 'occa/lib'), '-Wl,-rpath,' + str(root / 'occa/lib'),
                        '-locca', '-o', str(binary)], check=True)
        env = dict(os.environ, OCCA_CACHE_DIR=str(temp / 'occa'), OMP_NUM_THREADS='2')
        for mode in args.modes:
            subprocess.run([str(binary), str(root), mode], env=env, check=True)
    print('Isolated detector-kernel checks passed.')


if __name__ == '__main__':
    main()
