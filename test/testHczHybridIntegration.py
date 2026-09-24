#!/usr/bin/env python3
"""Actual LSERK4 hybrid spatial integration against HOST, including MPI halos.

Run after make -C solvers/bns. --cuda adds CUDA trajectories on one/two ranks.
Device detector/buffering/recompute decisions are checked against HOST.
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
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--flux', choices=('CHARACTERISTIC', 'LF'), default='CHARACTERISTIC')
    parser.add_argument('--extended', action='store_true', help='also test wider buffers and RT to t=0.1')
    args = parser.parse_args()
    base = dict(re.findall(r'\[([^]]+)\]\s*\n([^\n]+)',
                          (ROOT / 'solvers/bns/setups/setupHczRTQuad2D.rc').read_text()))
    base['HCZ WENO FLUX'] = args.flux
    with tempfile.TemporaryDirectory(prefix='hcz_hybrid_integration_') as directory:
        temp = Path(directory)
        env = dict(os.environ, OCCA_CACHE_DIR=str(temp / 'occa'),
                   LIBP_CACHE_DIR=str(temp / 'occa'),
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2')

        def run(name, changes=None, ranks=1, active=True, failure=None):
            opts = dict(base, **{'THREAD MODEL': 'Serial', 'HCZ DG BACKEND': 'VERIFY', 'HCZ SUBCELL WENO': 'TRUE',
                                'HCZ KAPPA': '0', 'BOX GLOBAL NX': '4', 'BOX GLOBAL NY': '16',
                                'HCZ BOUND TOLERANCE': '1e-8', 'FINAL TIME': '.002',
                                'OUTPUT INTERVAL': '.001', 'OUTPUT TO FILE': 'TRUE',
                                'OUTPUT FILE NAME': str(temp / name)})
            opts.update(changes or {})
            setup = temp / (name + '.rc')
            setup.write_text(''.join(f'[{key}]\n{value}\n' for key, value in opts.items()))
            command = [str(ROOT / 'solvers/bns/bnsMain'), str(setup)]
            if ranks > 1:
                command = ['mpirun', '--oversubscribe', '-n', str(ranks)] + command
            result = subprocess.run(command, cwd=temp, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
            if failure:
                assert result.returncode != 0 and failure in result.stdout, result.stdout
                print(name + ': rejected as expected', flush=True)
                return
            assert result.returncode == 0, result.stdout
            rows = list(csv.DictReader((temp / (name + '_diagnostics.csv')).open()))
            last = rows[-1]
            recomputations = int(last['recomputations'])
            assert (recomputations > 0) == active, last
            assert int(last['candidate_checks']) > 0, last
            assert abs(float(last['mass_error'])) < 2e-11, last
            assert abs(float(last['index_error'])) < 2e-11, last
            assert float(last['phi_min']) > 0, last
            if opts['HCZ DG BACKEND'] != 'HOST':
                transfers = re.search(r'full_array_downloads=(\d+) full_array_uploads=(\d+)', result.stdout)
                assert transfers and int(transfers[2]) == 0, result.stdout
                if opts['HCZ DG BACKEND'] == 'OCCA':
                    assert int(transfers[1]) == 0, result.stdout
            if opts['HCZ DG BACKEND'] == 'VERIFY':
                dg = re.search(r'rhs_checks=(\d+) max_error=(\S+) max_scaled_error=(\S+)', result.stdout)
                hybrid = re.search(r'hybrid_checks=(\d+) max_hybrid_error=(\S+) max_scaled_hybrid_error=(\S+)', result.stdout)
                checks = re.search(r'admissibility_checks=(\d+)', result.stdout)
                detector = re.search(r'detector_checks=(\d+)', result.stdout)
                buffers = re.search(r'buffer_checks=(\d+)', result.stdout)
                updates = re.search(r'mask_update_checks=(\d+)', result.stdout)
                assert dg and hybrid and checks and detector, result.stdout
                assert int(dg[1]) == int(last['candidate_checks']), result.stdout
                assert int(checks[1]) == 2 * int(dg[1]), result.stdout
                assert int(hybrid[1]) == recomputations, result.stdout
                assert int(detector[1]) == int(dg[1]) + recomputations, result.stdout
                assert buffers and int(buffers[1]) >= int(dg[1]), result.stdout
                assert updates and int(updates[1]) >= recomputations, result.stdout
                assert float(dg[3]) < 2e-11 and float(hybrid[3]) < 2e-11, result.stdout
                print(f'{name}: {detector[1]} matching detector masks; {hybrid[1]} hybrid checks, '
                      f'max error={float(hybrid[2]):.3e}', flush=True)
            else:
                assert 'HCZ OCCA verification:' not in result.stdout
                print(f'{name}: {recomputations} recomputations', flush=True)
            files = sorted(temp.glob(name + '_0000_*.vtu'))
            frame = files[-1].stem.rsplit('_', 1)[-1]
            fields = elements(sorted(temp.glob(name + '_*_' + frame + '.vtu')),
                              n=int(opts['POLYNOMIAL DEGREE']) + 1)
            return last, fields

        def agree(reference, actual):
            assert reference[1].keys() == actual[1].keys()
            error = max(np.max(np.abs(reference[1][key] - actual[1][key])) for key in reference[1])
            assert error < 5e-12, error
            for key in ('candidate_checks', 'recomputations', 'troubled', 'max_recompute_passes'):
                assert reference[0][key] == actual[0][key], (key, reference[0], actual[0])
            print(f'  HOST final field difference={error:.3e}; stage decisions agree', flush=True)

        wall = {'BOX BOUNDARY FLAG': '-1'}  # HCZ walls override BOX tags.
        host = run('wall_host', dict(wall, **{'HCZ DG BACKEND': 'HOST'}))
        for ranks in (1, 2, 4):
            agree(host, run(f'wall_verify_{ranks}', wall, ranks))
        agree(host, run('wall_occa', dict(wall, **{'HCZ DG BACKEND': 'OCCA'})))
        agree(host, run('wall_openmp', dict(wall, **{'THREAD MODEL': 'OpenMP',
                                                   'HCZ DG BACKEND': 'VERIFY'}), 2))

        # No buffer leaves narrower marked regions and exercises mixed faces.
        narrow = dict(wall, **{'HCZ BUFFER LAYERS': '0', 'HCZ RECOMPUTE PASSES': '4'})
        nh = run('narrow_host', dict(narrow, **{'HCZ DG BACKEND': 'HOST'}))
        assert int(nh[0]['max_recompute_passes']) > 1, 'must exercise multiple retries in one stage'
        agree(nh, run('narrow_verify', narrow, 4))
        for backend in ('HOST', 'OCCA', 'VERIFY'):
            run('reject_pass_limit_' + backend.lower(),
                dict(narrow, **{'HCZ DG BACKEND': backend, 'HCZ RECOMPUTE PASSES': '1'}), 2,
                failure='HCZ LSERK4 detector needs more recompute passes or a smaller CFL')

        periodic = {'HCZ Y BOUNDARY': 'PERIODIC', 'HCZ GRAVITY': '0',
                    'HCZ INITIAL CONDITION': 'SLAB', 'BOX BOUNDARY FLAG': '1',
                    'BOX GLOBAL NX': '8', 'BOX GLOBAL NY': '4', 'BOX DIMY': '1',
                    'HCZ INTERFACE WIDTH': '.03125', 'SPEED OF SOUND': '.7',
                    'VISCOSITY': '.002', 'HCZ TAU PHI': '.003'}
        ph = run('periodic_host', dict(periodic, **{'HCZ DG BACKEND': 'HOST'}))
        agree(ph, run('periodic_verify', periodic, 4))
        for degree in (1, 3):
            changes = dict(wall, **{'POLYNOMIAL DEGREE': str(degree)})
            dh = run(f'degree{degree}_host', dict(changes, **{'HCZ DG BACKEND': 'HOST'}))
            agree(dh, run(f'degree{degree}_verify', changes, 2))

        uniform = {'HCZ INITIAL CONDITION': 'UNIFORM', 'HCZ GRAVITY': '0',
                   'HCZ Y BOUNDARY': 'PERIODIC'}
        uh = run('uniform_host', dict(uniform, **{'HCZ DG BACKEND': 'HOST'}), active=False)
        uv = run('uniform_verify', uniform, 2, active=False)
        agree(uh, uv)
        assert float(uv[0]['max_speed']) < 1e-12
        if args.cuda:
            for ranks in (1, 2):
                agree(host, run(f'wall_cuda_{ranks}', dict(wall, **{'THREAD MODEL': 'CUDA'}), ranks))
            agree(nh, run('narrow_cuda', dict(narrow, **{'THREAD MODEL': 'CUDA',
                                                       'HCZ DG BACKEND': 'OCCA'}), 2))
            agree(ph, run('periodic_cuda', dict(periodic, **{'THREAD MODEL': 'CUDA'}), 2))
            agree(uh, run('uniform_cuda', dict(uniform, **{'THREAD MODEL': 'CUDA'}), 2, active=False))
            run('reject_pass_limit_cuda', dict(narrow, **{'HCZ RECOMPUTE PASSES': '1',
                                                        'THREAD MODEL': 'CUDA'}), 2,
                failure='HCZ LSERK4 detector needs more recompute passes or a smaller CFL')
        if args.extended:
            large = dict(wall, **{'BOX GLOBAL NX': '20'})
            gh = run('large_host', dict(large, **{'HCZ DG BACKEND': 'HOST'}))
            agree(gh, run('large_verify', large))
            if args.cuda:
                agree(gh, run('large_cuda', dict(large, **{'THREAD MODEL': 'CUDA'})))
            for layers in (2, 3):
                changes = dict(wall, **{'HCZ BUFFER LAYERS': str(layers)})
                bh = run(f'buffer{layers}_host', dict(changes, **{'HCZ DG BACKEND': 'HOST'}))
                agree(bh, run(f'buffer{layers}_verify', changes, 4))
                if args.cuda:
                    agree(bh, run(f'buffer{layers}_cuda', dict(changes, **{'THREAD MODEL': 'CUDA'}), 2))
            longer = dict(wall, **{'FINAL TIME': '.1', 'OUTPUT INTERVAL': '.05'})
            lh = run('long_host', dict(longer, **{'HCZ DG BACKEND': 'HOST'}))
            agree(lh, run('long_verify', longer, 4))
            agree(lh, run('long_openmp', dict(longer, **{'THREAD MODEL': 'OpenMP',
                                                       'HCZ DG BACKEND': 'OCCA'}), 2))
            if args.cuda:
                agree(lh, run('long_cuda', dict(longer, **{'THREAD MODEL': 'CUDA',
                                                         'HCZ DG BACKEND': 'OCCA'}), 2))
        run('reject_kappa', {'HCZ KAPPA': '.0001'}, failure='requires HCZ KAPPA = 0')
        print('HCZ hybrid spatial integration checks passed.', flush=True)


if __name__ == '__main__':
    main()
