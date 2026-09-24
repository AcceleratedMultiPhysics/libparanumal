/*

The MIT License (MIT)

Copyright (c) 2017-2022 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

// ================================================================
// Rayleigh-Taylor instability
//
// Domain:
//   0 <= x <= 0.5
//   0 <= y <= 0.5
//
// Heavy fluid: rho = 1.2, above the interface
// Light fluid: rho = 1.0, below the interface
//
// Perturbed sharp interface:
//   yInterface = y0 + amplitude*cos(2*pi*x/Lx)
//
// Recommended boundaries:
//   Left/right: periodic
//   Bottom/top: free-slip or no-slip
// ================================================================

// Fluid densities
#define RHO_LIGHT 1.0
#define RHO_HEAVY 1.2

// Reference density used for zero-mean gravitational forcing
#define RHO_REF 1.1

// Mean flow
#define RBAR RHO_REF
#define UBAR 0.0
#define VBAR 0.0

// Domain dimensions
#define DOMAIN_LX 10
#define DOMAIN_LY 10

// Interface definition
#define INTERFACE_Y0 0
#define PERTURBATION_AMPLITUDE 0.01
#define TWO_PI 6.28318530717958647692

// Downward gravitational acceleration
#define GRAVITY 0.0


// ================================================================
// Initial conditions
// ================================================================

#define bnsInitialConditions2D(c, nu, t, x, y, r, u, v,             \
                               s11, s12, s22)                         \
{                                                                     \
  const dfloat interfaceY =                                           \
      INTERFACE_Y0                                                     \
      + PERTURBATION_AMPLITUDE                                        \
      * cos(TWO_PI * (x) / DOMAIN_LX);                                \
                                                                      \
  /* Heavy fluid above, light fluid below */                           \
  *(r) = ((y) >= interfaceY) ? RHO_HEAVY : RHO_LIGHT;                 \
                                                                      \
  /* Initially stationary */                                          \
  *(u) = 0.0;                                                         \
  *(v) = 0.0;                                                         \
                                                                      \
  /* Initially equilibrium stress state */                            \
  *(s11) = 0.0;                                                       \
  *(s12) = 0.0;                                                       \
  *(s22) = 0.0;                                                       \
}


// ================================================================
// Body force
// ================================================================
//
// A zero-mean buoyancy force is used:
//
//   fy = -(rho-rhoRef)*g
//
// Therefore:
//   heavy fluid receives a downward force,
//   light fluid receives an upward force.
//
// This avoids accelerating the whole domain and does not require an
// initially hydrostatic pressure distribution.
// ================================================================

#define bnsBodyForce2D(c, nu, t, x, y, r, u, v, fx, fy) \
{                                                        \
  *(fx) = 0.0;                                           \
  *(fy) = -GRAVITY * ((r) - RHO_REF);                    \
}


// ================================================================
// Boundary conditions
//
// wall 1, inflow 2, outflow 3, x-slip 4, y-slip 5
// ================================================================

#define bnsBoundaryConditions2D(bc, c, nu,                           \
                                t, x, y, nx, ny,                     \
                                rM, uM, vM,                          \
                                s11M, s12M, s22M,                    \
                                rB, uB, vB,                          \
                                s11B, s12B, s22B)                    \
{                                                                    \
  if (bc == 1) {                                                     \
    /* No-slip wall */                                                \
    *(rB) = rM;                                                      \
    *(uB) = 0.0;                                                     \
    *(vB) = 0.0;                                                     \
    *(s11B) = s11M;                                                  \
    *(s12B) = s12M;                                                  \
    *(s22B) = s22M;                                                  \
                                                                     \
  } else if (bc == 4 || bc == 5) {                                   \
    /* Free-slip wall: remove the normal velocity component */       \
    const dfloat un = nx * uM + ny * vM;                             \
                                                                     \
    *(rB) = rM;                                                      \
    *(uB) = uM - un * nx;                                            \
    *(vB) = vM - un * ny;                                            \
    *(s11B) = s11M;                                                  \
    *(s12B) = s12M;                                                  \
    *(s22B) = s22M;                                                  \
                                                                     \
  } else {                                                           \
    /* Extrapolation fallback; periodicity is handled by mesh maps */\
    *(rB) = rM;                                                      \
    *(uB) = uM;                                                      \
    *(vB) = vM;                                                      \
    *(s11B) = s11M;                                                  \
    *(s12B) = s12M;                                                  \
    *(s22B) = s22M;                                                  \
  }                                                                  \
}