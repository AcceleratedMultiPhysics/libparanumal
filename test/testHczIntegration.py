#!/usr/bin/env python3
"""BNS HCZ runtime checks: RT, MPI equivalence, uniform equilibrium, guards.
Run from any directory after make -C solvers/bns. Needs local MPI sockets.
"""
import csv
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def elements(files,n=3):
    result={}
    for path in files:
        piece=ET.parse(path).find('.//Piece')
        points=np.fromstring(piece.find('./Points/DataArray').text,sep=' ').reshape(-1,n*n,3)
        fields=[np.fromstring(a.text,sep=' ').reshape(len(points),n*n,-1) for a in piece.findall('./PointData/DataArray')]
        for e,xyz in enumerate(points):
            result[tuple(np.round(xyz.mean(axis=0),12))]=np.concatenate([f[e].ravel() for f in fields])
    return result


def main():
    base=dict(re.findall(r'\[([^]]+)\]\s*\n([^\n]+)',(ROOT/'solvers/bns/setups/setupHczRTQuad2D.rc').read_text()))
    with tempfile.TemporaryDirectory(prefix='bns_hcz_runtime_') as directory:
        temp=Path(directory)
        env=dict(os.environ,OCCA_CACHE_DIR='/tmp/hcz_occa',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        def run(name,changes=None,ranks=1,fail=None):
            opts=dict(base,**{'BOX GLOBAL NX':'4','BOX GLOBAL NY':'16','FINAL TIME':'.02','OUTPUT INTERVAL':'.01',
                             'THREAD MODEL':'Serial','HCZ DG BACKEND':'HOST',
                             'HCZ BOUND TOLERANCE':'1e-8','OUTPUT FILE NAME':str(temp/name)})
            opts.update(changes or {})
            rc=temp/(name+'.rc');rc.write_text(''.join(f'[{k}]\n{v}\n' for k,v in opts.items()))
            command=[str(ROOT/'solvers/bns/bnsMain'),str(rc)]
            if ranks>1:command=['mpirun','--oversubscribe','-n',str(ranks)]+command
            result=subprocess.run(command,env=env,cwd=temp,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
            (temp/(name+'.log')).write_text(result.stdout)
            if fail:
                assert result.returncode!=0 and fail in result.stdout,result.stdout
                print(name+': rejected as expected',flush=True);return
            assert result.returncode==0,result.stdout
            print(name+': completed',flush=True)
            if opts['BNS MODEL']=='HCZ':
                rows=list(csv.DictReader((temp/(name+'_diagnostics.csv')).open()))
                assert abs(float(rows[-1]['mass_error']))<1e-11,rows[-1]
                assert float(rows[-1]['phi_min'])>0,rows[-1]
                print(f"  mass error={float(rows[-1]['mass_error']):.3e}, recomputations={rows[-1]['recomputations']}",flush=True)
                files=sorted(temp.glob(name+'_0000_*.vtu'));last=files[-1].stem.rsplit('_',1)[-1]
                return rows,elements(sorted(temp.glob(name+'_*_'+last+'.vtu')),n=int(opts['POLYNOMIAL DEGREE'])+1)
            return result.stdout
        one,one_fields=run('rt_serial')
        assert int(one[-1]['recomputations'])>0,'RT test must exercise stage rejection'
        two,two_fields=run('rt_mpi2',ranks=2)
        four,four_fields=run('rt_mpi4',ranks=4)
        for label,fields in [('MPI2',two_fields),('MPI4',four_fields)]:
            assert fields.keys()==one_fields.keys()
            error=max(np.max(np.abs(fields[key]-one_fields[key])) for key in fields)
            assert error<3e-12,error
            print(f'{label} agrees with serial: max field difference {error:.3e}',flush=True)
        uniform,uf=run('uniform',{'HCZ INITIAL CONDITION':'UNIFORM','HCZ GRAVITY':'0','HCZ Y BOUNDARY':'PERIODIC'})
        assert float(uniform[-1]['max_speed'])<1e-12
        assert int(uniform[-1]['recomputations'])==0
        run('rt_dg',{'HCZ SUBCELL WENO':'FALSE','HCZ INTERFACE WIDTH':'.5','FINAL TIME':'.002'})
        run('rt16',{'BOX GLOBAL NX':'16','BOX GLOBAL NY':'64','FINAL TIME':'.002','OUTPUT INTERVAL':'.002','HCZ BOUND TOLERANCE':'.01'})
        single=run('single_phase',{'BNS MODEL':'BNS','DATA FILE':str(ROOT/'solvers/bns/data/bnsUniform2D.h'),
                   'BOX BOUNDARY FLAG':'-1','VISCOSITY':'.01','FINAL TIME':'.002','OUTPUT TO FILE':'FALSE'})
        norm=float(re.search(r'Solution norm =\s*(\S+)',single).group(1))
        assert abs(norm-math.sqrt(10))<1e-11,norm
        single_sark=run('single_phase_sark4',{'BNS MODEL':'BNS','DATA FILE':str(ROOT/'solvers/bns/data/bnsUniform2D.h'),
                       'BOX BOUNDARY FLAG':'-1','VISCOSITY':'.01','FINAL TIME':'.0023','OUTPUT INTERVAL':'.0007',
                       'TIME INTEGRATOR':'SARK4','OUTPUT TO FILE':'FALSE'})
        norm_sark=float(re.search(r'Solution norm =\s*(\S+)',single_sark).group(1))
        assert abs(norm_sark-math.sqrt(10))<1e-11,norm_sark
        run('reject_pml',{'MESH FILE':'PMLBOX'},fail='no PML')
        run('reject_triangle',{'ELEMENT TYPE':'3'},fail='2D Cartesian quadrilateral')
        run('reject_stepper',{'TIME INTEGRATOR':'SARK5'},fail='LSERK4')
        print('HCZ runtime checks passed.',flush=True)


if __name__=='__main__':
    main()
