#!/usr/bin/env python3
"""Build HCZ detector/buffer and subcell-source tests using real libp OGS MPI halos.

Requires built libp libraries. Defaults to Serial/OpenMP, MPI 1/2/4 ranks.
Use --modes CUDA --ranks 1 2 to exercise ranks sharing the available GPU.
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
    parser.add_argument('--ranks', nargs='+', type=int, choices=(1, 2, 4), default=(1, 2, 4))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='hcz_mask_halo_') as directory:
        temp = Path(directory)
        binary = temp / 'hczMaskHaloQuad2D'
        subprocess.run(['mpic++', '-std=c++17', '-O2', '-fopenmp', '-Wall', '-Wextra',
                        '-isystem', str(root / 'include'), '-isystem', str(root / 'occa/include'),
                        str(root / 'test/hczMaskHaloQuad2D.cpp'),
                        '-L' + str(root / 'libs'), '-Wl,--start-group', '-logs', '-llinAlg', '-lcore',
                        '-Wl,--end-group', '-L' + str(root / 'occa/lib'),
                        '-Wl,-rpath,' + str(root / 'occa/lib'), '-locca', '-lopenblas',
                        '-o', str(binary)], check=True)
        env = dict(os.environ, OCCA_CACHE_DIR=str(temp / 'occa'),
                   LIBP_CACHE_DIR=str(temp / 'occa'),
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
        for mode in args.modes:
            for ranks in args.ranks:
                command = ['mpirun', '--oversubscribe', '-n', str(ranks), str(binary),
                           str(root), mode, str(temp / 'occa')]
                result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, timeout=240)
                if result.returncode:
                    raise RuntimeError(result.stdout)
                for line in result.stdout.splitlines():
                    if 'checks passed' in line or 'comparisons passed' in line:
                        print(line, flush=True)
    print('HCZ detector/buffer/subcell-source OGS halo tests passed.')


if __name__ == '__main__':
    main()
