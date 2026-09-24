#!/usr/bin/env python3
"""Check reported HCZ subcell masks against CSV counts and HOST/MPI/device results."""
import argparse
import csv
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
from testHczIntegration import elements

ROOT = Path(__file__).resolve().parents[1]
PLOT = ROOT / 'solvers/bns/plotHczSubcells.py'
spec = importlib.util.spec_from_file_location('plot_hcz', PLOT)
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--flux', choices=('CHARACTERISTIC', 'LF'), default='CHARACTERISTIC')
    parser.add_argument('--interface', action='store_true', help='test proactive interface seeding')
    parser.add_argument('--first-step-only', action='store_true', help='only verify immediate activation of all first-step RHS calls')
    parser.add_argument('--artifacts', type=Path, help='save example PNG and VTU pieces here')
    args = parser.parse_args()
    if args.first_step_only:
        args.interface = True
    base = dict(re.findall(r'\[([^]]+)\]\s*\n([^\n]+)',
                          (ROOT / 'solvers/bns/setups/setupHczSark4WenoQuad2D.rc').read_text()))
    base['HCZ INTERFACE SUBCELLS'] = 'TRUE' if args.interface else 'FALSE'
    base['HCZ WENO FLUX'] = args.flux
    with tempfile.TemporaryDirectory(prefix='hcz_subcell_output_') as directory:
        temp = Path(directory)
        env = dict(os.environ, OCCA_CACHE_DIR=str(temp / 'occa'), LIBP_CACHE_DIR=str(temp / 'occa'),
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MPLCONFIGDIR=str(temp / 'mpl'))

        def run(name, changes=None, ranks=1, expect_active=True, failure=None):
            opts = dict(base, **{'OUTPUT FILE NAME': str(temp / name), 'FINAL TIME': '.0017',
                                'OUTPUT INTERVAL': '.0016'})
            opts.update(changes or {})
            rc = temp / (name + '.rc')
            rc.write_text(''.join(f'[{k}]\n{v}\n' for k, v in opts.items()))
            command = [str(ROOT / 'solvers/bns/bnsMain'), str(rc)]
            if ranks > 1:
                command = ['mpirun', '--oversubscribe', '-n', str(ranks)] + command
            result = subprocess.run(command, cwd=temp, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
            if failure:
                assert result.returncode and failure in result.stdout, result.stdout
                print(name + ': rejected as expected', flush=True)
                return
            assert result.returncode == 0, result.stdout
            rows = list(csv.DictReader((temp / (name + '_diagnostics.csv')).open()))
            masks, seeds = [], []
            proactive = opts['HCZ INTERFACE SUBCELLS'] == 'TRUE'
            zero = (opts['HCZ SUBCELL WENO'] == 'FALSE' or opts['HCZ INITIAL CONDITION'] == 'UNIFORM'
                    or float(opts['FINAL TIME']) == 0)
            for frame, row in enumerate(rows):
                files = sorted(temp.glob(f'{name}_*_{frame:04d}.vtu'))
                assert len(files) == ranks, files
                mask, seed = {}, {}
                for file in files:
                    p = plot.read_piece(file)
                    assert abs(p['time'] - float(row['time'])) < 1e-14
                    assert len(p['mask']) == len(p['cells'])
                    for gid in np.unique(p['ids']):
                        values = p['mask'][p['ids'] == gid]
                        assert len(values) == int(opts['POLYNOMIAL DEGREE'])**2
                        assert np.all(values == values[0]) and gid not in mask
                        mask[gid] = int(values[0])
                        seed_values = p['interface'][p['ids'] == gid]
                        assert np.all(seed_values == seed_values[0])
                        seed[gid] = int(seed_values[0])
                        if frame == 0:
                            nodes = np.unique(p['cells'][p['ids'] == gid])
                            phi = p['phi'][nodes]
                            low, high = float(opts.get('HCZ PHI LIGHT', '.02381')), float(opts.get('HCZ PHI HEAVY', '.2508'))
                            eps = float(opts.get('HCZ INTERFACE THRESHOLD', '.01'))
                            expected = proactive and phi.max() > low + eps*(high-low) and phi.min() < high - eps*(high-low)
                            assert seed[gid] == expected, (name, gid, seed[gid], expected)
                        if not proactive:
                            assert seed[gid] == 0
                        if float(row['time']) > 0:
                            assert seed[gid] <= mask[gid], (name, frame, gid)
                assert len(mask) == int(opts['BOX GLOBAL NX']) * int(opts['BOX GLOBAL NY'])
                assert sum(mask.values()) == int(row['troubled']), (name, frame, row, mask)
                if frame == 0 or zero:
                    assert not any(mask.values()), (name, frame, mask)
                masks.append(mask)
                seeds.append(seed)
                assert abs(float(row['mass_error'])) < 2e-11 and abs(float(row['index_error'])) < 2e-11
            if not zero and expect_active:
                assert any(any(m.values()) for m in masks), name
            if not expect_active:
                assert not any(any(m.values()) for m in masks), name
            if proactive and opts['HCZ INITIAL CONDITION'] != 'UNIFORM':
                assert any(seeds[0].values()), name
            if proactive and opts['HCZ DG BACKEND'] == 'VERIFY' and float(opts['FINAL TIME']) > 0:
                checks = re.search(r'interface_checks=(\d+)', result.stdout)
                assert checks and int(checks[1]) > 0, result.stdout
                if '_first_' in name:
                    rhs = re.search(r'rhs_checks=(\d+)', result.stdout)
                    hybrid = re.search(r'hybrid_checks=(\d+)', result.stdout)
                    assert rhs and hybrid and int(rhs[1]) == int(hybrid[1]) == 5, result.stdout
                    if opts['TIME INTEGRATOR'] == 'SARK4':
                        replays = re.search(r'HCZ SARK4 hybrid: attempts=(\d+) replays=(\d+)', result.stdout)
                        assert replays and int(replays[1]) == 1 and int(replays[2]) == 0, result.stdout
                    print(name + ': all five first-step RHS calls used the seeded hybrid operator', flush=True)
            if opts['HCZ DG BACKEND'] == 'OCCA':
                transfer = re.search(r'full_array_downloads=(\d+) full_array_uploads=(\d+)', result.stdout)
                assert transfer and transfer[1] == transfer[2] == '0', result.stdout
            fields = elements(files, int(opts['POLYNOMIAL DEGREE']) + 1)
            print(f'{name}: {len(rows)} frames, correct mask/count/time; final marked={sum(masks[-1].values())}', flush=True)
            return masks, fields, files, seeds

        def agree(a, b, all_frames=True):
            assert (a[0] == b[0] if all_frames else a[0][-1] == b[0][-1])
            assert (a[3] == b[3] if all_frames else a[3][-1] == b[3][-1])
            assert a[1].keys() == b[1].keys()
            assert max(np.max(np.abs(a[1][k] - b[1][k])) for k in a[1]) < 5e-11

        if args.first_step_only:
            for stepper in ('SARK4', 'LSERK4'):
                first = {'TIME INTEGRATOR': stepper, 'FINAL TIME': '.000001', 'OUTPUT INTERVAL': '1',
                         'HCZ DMP DELTA': '10', 'HCZ BOUND TOLERANCE': '10'}
                run(stepper + '_first_off', dict(first, **{'HCZ INTERFACE SUBCELLS': 'FALSE'}), expect_active=False)
                host = run(stepper + '_first_host', dict(first, **{'HCZ DG BACKEND': 'HOST'}))
                agree(host, run(stepper + '_first_on', first, 2))
                if args.cuda:
                    agree(host, run(stepper + '_first_cuda', dict(first, **{'THREAD MODEL': 'CUDA'}), 2))
            print('HCZ first-step seeding checks passed.', flush=True)
            return

        for stepper in ('SARK4', 'LSERK4'):
            changes = {'TIME INTEGRATOR': stepper}
            host = run(stepper + '_host', dict(changes, **{'HCZ DG BACKEND': 'HOST'}))
            agree(host, run(stepper + '_verify', changes, 2))
            agree(host, run(stepper + '_occa', dict(changes, **{'HCZ DG BACKEND': 'OCCA'}), 2))
            agree(host, run(stepper + '_openmp', dict(changes, **{'THREAD MODEL': 'OpenMP'}), 2))
            # An output mini-step must not leave its mask attached to the final state.
            agree(host, run(stepper + '_no_sampling', dict(changes, **{'OUTPUT INTERVAL': '1'})), False)
            if args.cuda:
                gpu = run(stepper + '_cuda', dict(changes, **{'THREAD MODEL': 'CUDA', 'HCZ DG BACKEND': 'OCCA'}), 2)
                agree(host, gpu)
            run(stepper + '_uniform', dict(changes, **{'HCZ INITIAL CONDITION': 'UNIFORM'}))
            run(stepper + '_dg', dict(changes, **{'HCZ SUBCELL WENO': 'FALSE', 'HCZ INTERFACE SUBCELLS': 'FALSE', 'HCZ INTERFACE WIDTH': '.5'}))
            if args.interface:
                # Disable numerical overshoot criteria for a tiny first step.
                # Any hybrid evaluation must therefore come from the seed.
                first = dict(changes, **{'FINAL TIME': '.000001', 'OUTPUT INTERVAL': '1',
                                        'HCZ DMP DELTA': '10', 'HCZ BOUND TOLERANCE': '10'})
                run(stepper + '_first_off', dict(first, **{'HCZ INTERFACE SUBCELLS': 'FALSE'}), expect_active=False)
                first_host = run(stepper + '_first_host', dict(first, **{'HCZ DG BACKEND': 'HOST'}))
                agree(first_host, run(stepper + '_first_on', first, 2))
                if args.cuda:
                    agree(first_host, run(stepper + '_first_cuda', dict(first, **{'THREAD MODEL': 'CUDA'}), 2))
                for layers in (0, 2):
                    buffered = dict(changes, **{'HCZ BUFFER LAYERS': str(layers)})
                    bh = run(stepper + f'_buffer{layers}_host', dict(buffered, **{'HCZ DG BACKEND': 'HOST'}))
                    agree(bh, run(stepper + f'_buffer{layers}_verify', buffered, 4))
            if stepper == 'SARK4':
                example = gpu if args.cuda else host
        run('zero_duration', {'FINAL TIME': '0'})
        if args.interface:
            run('reject_no_weno', {'HCZ SUBCELL WENO': 'FALSE'}, failure='requires HCZ SUBCELL WENO')
            for value in ('0', '.5', 'nan'):
                run('reject_threshold_' + value, {'HCZ INTERFACE THRESHOLD': value}, failure='HCZ INTERFACE THRESHOLD must')
        image = temp / 'subcells.png'
        subprocess.run(['python3', str(PLOT), *map(str, example[2]), '--interface-value', '.137305',
                        '-o', str(image)], env=env, check=True)
        assert image.stat().st_size > 1000
        if args.artifacts:
            import shutil
            args.artifacts.mkdir(parents=True, exist_ok=True)
            for file in [image, *example[2]]:
                shutil.copy2(file, args.artifacts / file.name)
        print('HCZ subcell-output checks passed.', flush=True)


if __name__ == '__main__':
    main()
