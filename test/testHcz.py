#!/usr/bin/env python3
"""Compare C++ HCZ operators with the supplied Python implementation.

Usage: python3 test/testHcz.py --reference-dir /path/to/HCZ_DG
Uses temporary files; does not run the full reference benchmark or edit it.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from dataclasses import replace


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    core = load(args.reference_dir / 'hcz_gb_2d_quad_dg.py', 'hcz_core_reference')
    rt = load(args.reference_dir / 'hcz_index_dg_rayleigh_taylor_re256_subcell_weno3z.py', 'hcz_rt_reference')
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='bns_hcz_test_') as temp:
        temp = Path(temp)
        probe = temp / 'probe'
        subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', str(root / 'test/hczQuadReference.cpp'), '-o', str(probe)], check=True)
        for degree in (1, 2, 3):
            test = replace(rt.RayleighTaylorIndexConfig(), num_elements_x=4, num_elements_y=8,
                           polynomial_degree=degree, subcell_count=2*degree+1,
                           initial_interface_width_10_90=.5)
            scales = rt.derived_scales(test)
            cfg = replace(rt.SolverConfig(), length_x=1., length_y=4., num_elements_x=4, num_elements_y=8,
                          polynomial_degree=degree, rho_gas=1., rho_liquid=3.,
                          tau_phi=scales['tau'], tau_g=scales['tau'], initial_interface_width_10_90=.5)
            solver = rt.HCZIndexRayleighTaylorSubcellWENO(cfg, test, scales)
            # A smooth non-equilibrium state tests all 12 equations and sources.
            x,y=solver.x,solver.y
            phi=.13+.04*np.sin(2*np.pi*x)*np.cos(.5*np.pi*y)
            rho=solver.density_from_phi(phi)
            u=.003*np.sin(2*np.pi*x);v=.004*np.sin(.5*np.pi*y)
            q=np.concatenate((solver.equilibrium_phi(phi,u,v),solver.equilibrium_g(rho,.12+.002*y,u,v)),axis=-1)
            rng=np.random.default_rng(304+degree)
            q[...,1:6]+=rng.normal(0,1e-5,q[...,1:6].shape)
            q[...,9:12]+=rng.normal(0,1e-5,q[...,9:12].shape)
            mask=np.zeros((8,4),dtype=bool);mask[0,0]=mask[3,1]=mask[-1,-1]=True
            # Nonzero history catches accidental loss/double application of the
            # LSERK low-storage residual during stage recomputation.
            res=rng.normal(0,1e-6,q.shape)
            res[3,1,:,:,0]=-.5  # Force a rejected candidate with nonzero RK history.
            dt,a,b=1e-6,-.5,.3
            dg=solver._dg_rhs_only(q,0.)[0]
            hybrid=solver._hybrid_rhs_for_mask(q,0.,mask)[0]
            stage=dg.copy()
            trial=q+b*(a*res+dt*stage)
            detected,_=solver.detect_troubled_candidate((q,),trial)
            detected=solver._expand_troubled_mask(detected,test.subcell_buffer_layers)
            assert np.any(detected), 'The stage comparison must exercise WENO recomputation'
            for _ in range(test.subcell_max_recompute_passes):
                stage=solver._hybrid_rhs_for_mask(q,0.,detected)[0]
                trial=q+b*(a*res+dt*stage)
                added,_=solver.detect_troubled_candidate((q,),trial)
                added &= ~detected
                if not np.any(added):break
                detected |= solver._expand_troubled_mask(added,test.subcell_buffer_layers)
            qr,qw=np.polynomial.legendre.leggauss(degree+2)
            params=[cfg.c,cfg.tau_phi,cfg.tau_g,1,3,.02381,.2508,4,3,scales['gravity'],0]
            values=np.concatenate(([degree,4,8,0,solver.hx,solver.hy],solver.r,solver.weights,qr,qw,params,[dt,a,b],q.ravel(),res.ravel(),mask.astype(float).ravel()))
            np.savetxt(temp/'input',values,fmt='%.17g')
            subprocess.run([str(probe),str(temp/'input'),str(temp/'output')],check=True)
            actual=np.loadtxt(temp/'output');offset=0
            expected=[dg,hybrid,solver.dg_to_subcells(q),q,stage]
            for label,want in zip(('DG RHS','hybrid RHS','subcell averages','round trip','LSERK stage RHS'),expected):
                got=actual[offset:offset+want.size].reshape(want.shape);offset+=want.size
                np.testing.assert_allclose(got,want,atol=2e-11,rtol=2e-10,err_msg=f'N={degree} {label}')
                print(f'N={degree} {label}: max error {np.max(np.abs(got-want)):.3e}')
            # Global index transport cancels including periodic and mixed faces.
            mass_rate=solver.integral(hybrid[...,0])
            assert abs(mass_rate)<2e-13,mass_rate
            print(f'N={degree} mixed DG/FV index integral rate: {mass_rate:.3e}')
        # Different sound speed and relaxation times expose normalization
        # errors hidden by c=1. Periodic capillarity exercises repeated scalar
        # derivative exchanges and the optional third-derivative source.
        degree=2
        cfg=replace(core.Config(),length_x=1.,length_y=1.,num_elements_x=4,num_elements_y=4,
                    polynomial_degree=degree,c=1.3,tau_phi=.0012,tau_g=.0025,
                    rho_gas=1.,rho_liquid=3.,pressure_gradient_kappa=1e-5)
        solver=core.HeChenZhangGalerkinBoltzmann2D(cfg)
        x,y=solver.x,solver.y
        phi=.13+.02*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)
        rho=solver.density_from_phi(phi)
        u=.003*np.sin(2*np.pi*x);v=.004*np.sin(2*np.pi*y)
        q=np.concatenate((solver.equilibrium_phi(phi,u,v),solver.equilibrium_g(rho,.16+.002*np.cos(2*np.pi*y),u,v)),axis=-1)
        res=np.zeros_like(q);mask=np.zeros((4,4));dt,a,b=1e-6,0.,.3
        dg=solver.rhs(q,0.)[0]
        qr,qw=np.polynomial.legendre.leggauss(degree+2)
        params=[cfg.c,cfg.tau_phi,cfg.tau_g,1,3,.02381,.2508,4,3,0,cfg.pressure_gradient_kappa]
        values=np.concatenate(([degree,4,4,1,solver.hx,solver.hy],solver.r,solver.weights,qr,qw,params,[dt,a,b],q.ravel(),res.ravel(),mask.ravel()))
        np.savetxt(temp/'input',values,fmt='%.17g')
        subprocess.run([str(probe),str(temp/'input'),str(temp/'output')],check=True)
        actual=np.loadtxt(temp/'output')
        np.testing.assert_allclose(actual[:q.size].reshape(q.shape),dg,atol=2e-11,rtol=2e-10)
        np.testing.assert_allclose(actual[-q.size:].reshape(q.shape),dg,atol=2e-11,rtol=2e-10)
        print(f'Periodic capillarity, c=1.3, unequal tau: max DG error {np.max(np.abs(actual[:q.size].reshape(q.shape)-dg)):.3e}')
    print('HCZ operator comparisons passed.')


if __name__ == '__main__':
    main()
