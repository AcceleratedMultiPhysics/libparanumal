#!/usr/bin/env python3
"""HCZ SARK4/WENO full-trial replay, CFL caps, HOST/device agreement and conservation."""
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
    args = parser.parse_args()
    base = dict(re.findall(r'\[([^]]+)\]\s*\n([^\n]+)',
                          (ROOT / 'solvers/bns/setups/setupHczSark4WenoQuad2D.rc').read_text()))
    base['HCZ WENO FLUX'] = args.flux
    with tempfile.TemporaryDirectory(prefix='hcz_sark_integration_') as directory:
        temp = Path(directory)
        env = dict(os.environ, OCCA_CACHE_DIR=str(temp / 'occa'), LIBP_CACHE_DIR=str(temp / 'occa'),
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')

        def run(name, changes=None, ranks=1, failure=None):
            opts = dict(base, **{'OUTPUT FILE NAME': str(temp / name), 'FINAL TIME': '.0017',
                                'OUTPUT INTERVAL': '.0006'})
            opts.update(changes or {})
            setup = temp / (name + '.rc')
            setup.write_text(''.join(f'[{k}]\n{v}\n' for k, v in opts.items()))
            command = [str(ROOT / 'solvers/bns/bnsMain'), str(setup)]
            if ranks > 1:
                command = ['mpirun', '--oversubscribe', '-n', str(ranks)] + command
            result = subprocess.run(command, cwd=temp, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
            if failure:
                assert result.returncode and failure in result.stdout, result.stdout
                if name.startswith('limited_'):
                    assert 'HCZ SARK4 recomputation budget exhausted' in result.stdout, result.stdout
                print(name + ': rejected as expected', flush=True)
                return
            assert result.returncode == 0, result.stdout
            if opts['TIME INTEGRATOR'] == 'SARK4':
                cap = re.search(r'HCZ SARK4: max_dt=(\S+)', result.stdout)
                stats = re.search(r'SARK4: accepted_steps=(\d+) error_rejections=(\d+) stage_rejections=(\d+) min_dt=(\S+) max_dt=(\S+)', result.stdout)
                assert cap and stats and int(stats[1]) > 0, result.stdout
                n = int(opts['POLYNOMIAL DEGREE']) + 1
                c = float(opts['SPEED OF SOUND'])
                invh = int(opts['BOX GLOBAL NX']) / float(opts['BOX DIMX']) + int(opts['BOX GLOBAL NY']) / float(opts['BOX DIMY'])
                expected_cap = float(opts['CFL NUMBER']) / (np.sqrt(3) * c * n*n * invh)
                if opts['HCZ SUBCELL WENO'] == 'TRUE':
                    expected_cap = min(expected_cap, float(opts.get('HCZ SUBCELL CFL', '.18')) / (float(opts.get('HCZ WENO ALPHA', '1.75')) * c * (2*n-1) * invh))
                assert abs(float(cap[1]) / expected_cap - 1) < 1e-12, result.stdout
                # The final summary prints only 12 significant digits.
                assert 0 < float(stats[5]) <= expected_cap * (1 + 5e-12), result.stdout
                print(f'{name}: {stats[1]} accepted, {stats[2]} error retries, {stats[3]} stage retries; max dt={stats[5]}', flush=True)
            if opts['HCZ DG BACKEND'] == 'VERIFY':
                rhs = re.search(r'rhs_checks=(\d+)', result.stdout)
                split = re.search(r'split_checks=(\d+) max_split_error=(\S+) max_scaled_split_error=(\S+)', result.stdout)
                assert rhs and split, result.stdout
                if opts['TIME INTEGRATOR'] == 'SARK4':
                    assert int(split[1]) == int(rhs[1]) > 0, result.stdout
                    assert float(split[3]) < 2e-11, result.stdout
                else:
                    assert int(split[1]) == 0, result.stdout
            if opts['HCZ DG BACKEND'] == 'OCCA':
                transfers = re.search(r'full_array_downloads=(\d+) full_array_uploads=(\d+)', result.stdout)
                assert transfers and int(transfers[1]) == int(transfers[2]) == 0, result.stdout
            last = list(csv.DictReader((temp / (name + '_diagnostics.csv')).open()))[-1]
            assert abs(float(last['time']) - float(opts['FINAL TIME'])) < 1e-14, last
            assert abs(float(last['mass_error'])) < 2e-11 and abs(float(last['index_error'])) < 2e-11, last
            if opts['HCZ SUBCELL WENO'] == 'TRUE':
                hybrid = re.search(r'HCZ SARK4 hybrid: attempts=(\d+) replays=(\d+) budget_rejections=(\d+) mean_rejections=(\d+)', result.stdout)
                assert hybrid and int(hybrid[1]) > 0 and int(hybrid[2]) == int(last['recomputations']), result.stdout
                assert int(last['candidate_checks']) > 0, last
                if opts['HCZ INITIAL CONDITION'] != 'UNIFORM':
                    assert int(hybrid[2]) > 0, last
                if opts['HCZ DG BACKEND'] == 'VERIFY':
                    detector = re.search(r'detector_checks=(\d+)', result.stdout)
                    buffers = re.search(r'buffer_checks=(\d+)', result.stdout)
                    updates = re.search(r'mask_update_checks=(\d+)', result.stdout)
                    assert detector and int(detector[1]) == int(last['candidate_checks']), result.stdout
                    assert buffers and int(buffers[1]) == int(hybrid[2]), result.stdout
                    assert updates and int(updates[1]) == int(detector[1]) + int(hybrid[2]), result.stdout
                last['budget_rejections'] = int(hybrid[3])
                print(f"  replays={hybrid[2]}, max passes={last['max_recompute_passes']}, budget rejections={hybrid[3]}", flush=True)
            else:
                assert int(last['recomputations']) == 0 and int(last['candidate_checks']) == 0, last
            files = sorted(temp.glob(name + '_0000_*.vtu'))
            frame = files[-1].stem.rsplit('_', 1)[-1]
            fields = elements(sorted(temp.glob(name + '_*_' + frame + '.vtu')),
                              n=int(opts['POLYNOMIAL DEGREE']) + 1)
            return last, fields

        def agree(ref, actual, tolerance=5e-11):
            assert ref[1].keys() == actual[1].keys()
            for counter in ('recomputations', 'candidate_checks', 'max_recompute_passes', 'budget_rejections'):
                assert ref[0][counter] == actual[0][counter], (counter, ref[0], actual[0])
            err = max(np.max(np.abs(ref[1][k] - actual[1][k])) for k in ref[1])
            assert err < tolerance, err
            print(f'  final-field difference={err:.3e}', flush=True)

        host = run('wall_host', {'HCZ DG BACKEND': 'HOST'})
        for ranks in (1, 2, 4):
            agree(host, run(f'wall_verify_{ranks}', ranks=ranks))
        agree(host, run('wall_occa', {'HCZ DG BACKEND': 'OCCA'}))
        agree(host, run('wall_openmp', {'THREAD MODEL': 'OpenMP'}, 2))
        # A new mask must cause replay from stage zero; the fixture must
        # actually require more than the first hybrid pass.
        assert int(host[0]['max_recompute_passes']) > 1, host[0]
        limited = {'HCZ RECOMPUTE PASSES': '1'}
        for backend in ('HOST', 'OCCA', 'VERIFY'):
            run('limited_' + backend.lower(), dict(limited, **{'HCZ DG BACKEND': backend}), 2,
                failure='Time step became too small')
        periodic = {'HCZ INITIAL CONDITION': 'SLAB', 'HCZ Y BOUNDARY': 'PERIODIC',
                    'HCZ GRAVITY': '0', 'BOX GLOBAL NX': '8', 'BOX GLOBAL NY': '4', 'BOX DIMY': '1',
                    'HCZ INTERFACE WIDTH': '.03125', 'SPEED OF SOUND': '.7',
                    'HCZ RHO HEAVY': '1.2', 'HCZ EOS B': '3.2', 'HCZ EOS A OVER B C2': '2.1'}
        ph = run('periodic_host', dict(periodic, **{'HCZ DG BACKEND': 'HOST'}))
        agree(ph, run('periodic_verify', periodic, 4))
        for degree in (1, 3):
            changes = {'POLYNOMIAL DEGREE': str(degree)}
            dh = run(f'degree{degree}_host', dict(changes, **{'HCZ DG BACKEND': 'HOST'}))
            agree(dh, run(f'degree{degree}_verify', changes, 2))
        uniform = run('uniform', {'HCZ INITIAL CONDITION': 'UNIFORM', 'HCZ GRAVITY': '0',
                                  'HCZ Y BOUNDARY': 'PERIODIC'}, 2)
        assert float(uniform[0]['max_speed']) < 1e-12
        hydrostatic = run('hydrostatic', {'HCZ INITIAL CONDITION': 'UNIFORM'}, 2)
        assert float(hydrostatic[0]['max_speed']) < 1e-12
        refined = {'POLYNOMIAL DEGREE': '4', 'BOX GLOBAL NX': '8', 'BOX GLOBAL NY': '32',
                   'CFL NUMBER': '.2', 'FINAL TIME': '.0047', 'OUTPUT INTERVAL': '.0013'}
        rh = run('refined_host', dict(refined, **{'HCZ DG BACKEND': 'HOST'}))
        agree(rh, run('refined_verify', refined, 4))
        if args.cuda:
            for ranks in (1, 2):
                agree(host, run(f'wall_cuda_{ranks}', {'THREAD MODEL': 'CUDA'}, ranks))
            agree(host, run('wall_cuda_occa', {'THREAD MODEL': 'CUDA', 'HCZ DG BACKEND': 'OCCA'}, 2))
            agree(ph, run('periodic_cuda', dict(periodic, **{'THREAD MODEL': 'CUDA'}), 2))
            agree(rh, run('refined_cuda', dict(refined, **{'THREAD MODEL': 'CUDA'}), 2))
        for layers in (0, 2):
            changes = {'HCZ BUFFER LAYERS': str(layers)}
            bh = run(f'buffer{layers}_host', dict(changes, **{'HCZ DG BACKEND': 'HOST'}))
            agree(bh, run(f'buffer{layers}_verify', changes, 4))
        # Force the subcell bound to be tighter than the transport bound.
        subcap = {'HCZ SUBCELL CFL': '.002'}
        sh = run('subcap_host', dict(subcap, **{'HCZ DG BACKEND': 'HOST'}))
        agree(sh, run('subcap_verify', subcap, 2))
        if args.cuda:
            run('limited_cuda', dict(limited, **{'THREAD MODEL': 'CUDA'}), 2, failure='Time step became too small')
            agree(sh, run('subcap_cuda', dict(subcap, **{'THREAD MODEL': 'CUDA'}), 2))
        run('reject_kappa', {'HCZ DG BACKEND': 'HOST', 'HCZ KAPPA': '.0001'}, failure='SARK4 requires HCZ KAPPA = 0')
        run('reject_atol', {'HCZ SARK4 ATOL': '0'}, failure='Invalid HCZ SARK4 tolerances')
        run('reject_rtol', {'HCZ SARK4 RTOL': '-1'}, failure='Invalid HCZ SARK4 tolerances')
        print('HCZ SARK4/WENO integration checks passed.', flush=True)


if __name__ == '__main__':
    main()
