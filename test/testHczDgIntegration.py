#!/usr/bin/env python3
"""Actual-mesh OCCA pure-DG checks after make -C solvers/bns.

Uses Serial/OpenMP and MPI 1/2/4 ranks. VERIFY compares all twelve RHS
moments at every evaluation; final output fields also compare independent
HOST/OCCA trajectories. Temporary setup/output files and cache are isolated.
"""
import argparse
import csv
import os
from pathlib import Path
import re
import subprocess
import tempfile
import numpy as np
from testHczIntegration import elements

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda', action='store_true', help='also run short CUDA trajectories')
    args = parser.parse_args()
    base = dict(re.findall(r'\[([^]]+)\]\s*\n([^\n]+)',
                          (ROOT / 'solvers/bns/setups/setupHczRTSmokeQuad2D.rc').read_text()))
    with tempfile.TemporaryDirectory(prefix='hcz_dg_integration_') as directory:
        temp = Path(directory)
        env = dict(os.environ, OCCA_CACHE_DIR=str(temp / 'occa'),
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2')

        def run(name, changes=None, ranks=1, failure=None):
            opts = dict(base, **{'HCZ DG BACKEND': 'VERIFY', 'HCZ SUBCELL WENO': 'FALSE',
                                'HCZ KAPPA': '0', 'BOX GLOBAL NX': '4', 'BOX GLOBAL NY': '8',
                                'HCZ INTERFACE WIDTH': '.75', 'FINAL TIME': '.002',
                                'OUTPUT INTERVAL': '.001', 'OUTPUT TO FILE': 'TRUE',
                                'OUTPUT FILE NAME': str(temp / name)})
            opts.update(changes or {})
            setup = temp / (name + '.rc')
            setup.write_text(''.join(f'[{key}]\n{value}\n' for key, value in opts.items()))
            command = [str(ROOT / 'solvers/bns/bnsMain'), str(setup)]
            if ranks > 1:
                command = ['mpirun', '--oversubscribe', '-n', str(ranks)] + command
            result = subprocess.run(command, cwd=temp, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
            (temp / (name + '.log')).write_text(result.stdout)
            if failure:
                assert result.returncode != 0 and failure in result.stdout, result.stdout
                print(name + ': rejected as expected', flush=True)
                return
            assert result.returncode == 0, result.stdout
            assert 'HCZ spatial backend: ' + opts['HCZ DG BACKEND'] in result.stdout
            if opts['HCZ DG BACKEND'] == 'OCCA':
                transfers = re.search(r'full_array_downloads=(\d+) full_array_uploads=(\d+)', result.stdout)
                assert transfers and int(transfers[1]) == 0 and int(transfers[2]) == 0, result.stdout
            if opts['HCZ DG BACKEND'] == 'VERIFY':
                match = re.search(r'HCZ OCCA verification: rhs_checks=(\d+) max_error=(\S+) max_scaled_error=(\S+)', result.stdout)
                assert match and int(match[1]) > 0, result.stdout
                assert float(match[3]) < 2e-11, match.groups()
                checks = re.search(r'admissibility_checks=(\d+)', result.stdout)
                assert checks and int(checks[1]) == 2 * int(match[1]), result.stdout
                detector = re.search(r'detector_checks=(\d+)', result.stdout)
                assert detector and int(detector[1]) == 0, result.stdout
                print(f'{name}: {match[1]} RHS checks; max error={float(match[2]):.3e}', flush=True)
            else:
                assert 'HCZ OCCA verification:' not in result.stdout
                print(name + ': completed', flush=True)
            with (temp / (name + '_diagnostics.csv')).open() as file:
                rows = list(csv.DictReader(file))
            assert abs(float(rows[-1]['mass_error'])) < 2e-11, rows[-1]
            assert int(rows[-1]['candidate_checks']) == 0 and int(rows[-1]['recomputations']) == 0
            assert float(rows[-1]['phi_min']) > 0
            files = sorted(temp.glob(name + '_0000_*.vtu'))
            last = files[-1].stem.rsplit('_', 1)[-1]
            fields = elements(sorted(temp.glob(name + '_*_' + last + '.vtu')),
                              n=int(opts['POLYNOMIAL DEGREE']) + 1)
            return rows, fields

        def agree(reference, result, label):
            assert reference[1].keys() == result[1].keys(), label
            error = max(np.max(np.abs(reference[1][key] - result[1][key])) for key in reference[1])
            assert error < 5e-12, (label, error)
            print(f'  {label}: final field difference={error:.3e}', flush=True)

        # Deliberately use a periodic BOX mesh with HCZ y walls; the HCZ
        # topology, rather than BOX's boundary tags, must control the kernels.
        wall = {'BOX BOUNDARY FLAG': '-1', 'SPEED OF SOUND': '1.3',
                'VISCOSITY': '.003', 'HCZ TAU PHI': '.0025'}
        host = run('wall_host', dict(wall, **{'HCZ DG BACKEND': 'HOST'}))
        for ranks in (1, 2, 4):
            agree(host, run(f'wall_verify_{ranks}', wall, ranks), f'wall VERIFY MPI{ranks}')
        agree(host, run('wall_occa', dict(wall, **{'HCZ DG BACKEND': 'OCCA'})), 'wall OCCA')
        agree(host, run('wall_openmp', dict(wall, **{'HCZ DG BACKEND': 'OCCA',
                                                  'THREAD MODEL': 'OpenMP'}), 2), 'wall OpenMP MPI2')

        periodic = {'BOX BOUNDARY FLAG': '1', 'HCZ Y BOUNDARY': 'PERIODIC',
                    'HCZ INITIAL CONDITION': 'SLAB', 'HCZ GRAVITY': '0',
                    'BOX GLOBAL NY': '4', 'BOX DIMY': '1', 'HCZ RHO HEAVY': '1.2',
                    'HCZ INTERFACE WIDTH': '.3', 'SPEED OF SOUND': '.7',
                    'VISCOSITY': '.002', 'HCZ TAU PHI': '.003',
                    'HCZ EOS B': '3.2', 'HCZ EOS A OVER B C2': '2.1'}
        ph = run('periodic_host', dict(periodic, **{'HCZ DG BACKEND': 'HOST'}))
        for ranks in (1, 2, 4):
            agree(ph, run(f'periodic_verify_{ranks}', periodic, ranks), f'periodic VERIFY MPI{ranks}')
        agree(ph, run('periodic_openmp', dict(periodic, **{'THREAD MODEL': 'OpenMP'}), 2), 'periodic OpenMP MPI2')
        if args.cuda:
            for ranks in (1, 2):
                agree(host, run(f'wall_cuda_{ranks}', dict(wall, **{'THREAD MODEL': 'CUDA'}), ranks),
                      f'wall CUDA MPI{ranks}')
            agree(ph, run('periodic_cuda', dict(periodic, **{'THREAD MODEL': 'CUDA'})), 'periodic CUDA')
            agree(host, run('wall_cuda_occa', dict(wall, **{'THREAD MODEL': 'CUDA',
                                                         'HCZ DG BACKEND': 'OCCA'})), 'wall CUDA OCCA')
        for degree in (1, 3):
            changes = dict(wall, **{'POLYNOMIAL DEGREE': str(degree)})
            dh = run(f'degree{degree}_host', dict(changes, **{'HCZ DG BACKEND': 'HOST'}))
            agree(dh, run(f'degree{degree}_verify', changes, 2), f'degree {degree} MPI2')

        uniform = run('uniform', {'HCZ INITIAL CONDITION': 'UNIFORM', 'HCZ GRAVITY': '0',
                                  'HCZ Y BOUNDARY': 'PERIODIC'}, 2)
        assert float(uniform[0][-1]['max_speed']) < 1e-12
        hydrostatic = run('hydrostatic', {'HCZ INITIAL CONDITION': 'UNIFORM'}, 2)
        assert float(hydrostatic[0][-1]['max_speed']) < 1e-12
        # An intentionally excessive step must retain the original physical
        # rejection, including the collective device decision across ranks.
        for backend in ('HOST', 'OCCA', 'VERIFY'):
            run('reject_candidate_' + backend.lower(),
                dict(wall, **{'HCZ DG BACKEND': backend, 'HCZ GRAVITY': '10000',
                              'CFL NUMBER': '100', 'HCZ COLLISION CFL': '100',
                              'FINAL TIME': '.1', 'OUTPUT INTERVAL': '1'}), 2,
                failure='HCZ LSERK4 candidate remains nonphysical; reduce CFL')
        run('reject_kappa', {'HCZ KAPPA': '.0001'}, failure='requires HCZ KAPPA = 0')
        run('reject_pml', {'MESH FILE': 'PMLBOX'}, failure='no PML')
        run('reject_stepper', {'TIME INTEGRATOR': 'SARK5'}, failure='requires TIME INTEGRATOR = LSERK4 or SARK4')
        print('HCZ OCCA pure-DG integration checks passed.', flush=True)


if __name__ == '__main__':
    main()
