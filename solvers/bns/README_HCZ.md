# HCZ multiphase BNS on 2D quadrilaterals

Select `[BNS MODEL] HCZ` to evolve the independent-index He–Chen–Zhang
formulation ported from `hcz_gb_2d_quad_dg.py` and
`hcz_index_dg_rayleigh_taylor_re256_subcell_weno3z.py`. The default model
`BNS` continues to use the existing single-distribution solver.

The HCZ mode supports uniform, axis-aligned Cartesian `BOX` quadrilaterals
in 2D. Pure DG and hybrid DG/WENO support the existing `LSERK4` and
`SARK4` steppers. SARK4 requires `HCZ KAPPA=0`. Other
elements, imported or curved meshes, 3D, PML and other time integrators
are rejected.

## Run

From the repository root:

```sh
make -C solvers/bns -j4
cd solvers/bns
./bnsMain setups/setupHczRTSmokeQuad2D.rc
```

`setupHczRTQuad2D.rc` contains the configurable RT case; check its mesh,
degree and runtime settings before launching. `setupHczSlabQuad2D.rc` contains a periodic slab case using
pure DG. The scripts are reference inputs to the validation test; they
are not runtime dependencies of BNS.

For MPI, use e.g. `mpirun -n 4 ./bnsMain setups/setupHczRTSmokeQuad2D.rc`.
`BOX GLOBAL NX/NY` preserve the same resolution across process counts.
HCZ always uses periodic x boundaries. `HCZ Y BOUNDARY` selects `WALL` or
`PERIODIC`; these kinetic conditions and the HCZ neighbor topology govern
the spatial operator independently of the base mesh's `BOX BOUNDARY FLAG`.
Output coordinates translate the centered BOX mesh to `[0,Lx] × [0,Ly]`.

**Execution backend:** `HCZ DG BACKEND` selects `HOST` (default), `OCCA`,
or `VERIFY`. `HOST` retains the reference DG/subcell implementation.
`OCCA` runs DG/WENO spatial kernels, detection, mask buffering and updates,
OGS halo exchanges, and LSERK4/SARK4 admissibility checks on the
configured device. Both OCCA and VERIFY require `HCZ KAPPA=0` and support
`HCZ SUBCELL WENO=TRUE`.

**Hybrid stages keep the solution, residual, RHS and masks on the device.**
The host launches the kernels and controls the retry loop using scalar
counts/flags and MPI reductions. There are no routine full solution-array
downloads or uploads during ordinary OCCA stages. Initialization and
reporting still use host data. `VERIFY` intentionally downloads arrays for
reference comparisons: every DG/hybrid RHS, raw/buffered/updated mask and
admissibility decision is checked against the host implementation. Device
masks now control the hybrid stages in both OCCA and VERIFY modes.
Only neighboring complete elements (including corners) are communicated;
the solution is never gathered globally. A global ownership map is built
at setup. Runtime validation covers Serial/OpenMP with MPI 1/2/4 ranks
and short CUDA runs with MPI 1/2 ranks on the available GPU.

For the short OCCA pure-DG example, run from `solvers/bns`:

```sh
./bnsMain setups/setupHczDgOccaQuad2D.rc
mpirun -n 4 ./bnsMain setups/setupHczDgOccaQuad2D.rc
```

Use `HCZ DG BACKEND=VERIFY` to enable all-moment RHS and admissibility comparisons. The
backend and thread-model entries in each setup determine its execution mode.

To exercise the hybrid spatial path, use a copy of an RT setup with:

```text
[HCZ DG BACKEND]
VERIFY
[HCZ SUBCELL WENO]
TRUE
[HCZ KAPPA]
0
[THREAD MODEL]
CUDA
```

`Serial` and `OpenMP` are also supported. After checking with `VERIFY`,
select `OCCA` to disable reference RHS comparisons. Only scalar stage decisions and kernel/MPI orchestration
remain on the host in OCCA mode. The incremental development sections below record
the state at each milestone; this execution-backend section describes the
current runtime behavior.

## Evolved equations

Each node has twelve moments: six `q_phi` moments followed by six `q_g`
moments. The existing BNS storage remains element/field/node ordered; the
host reference packs to element/node/field order, while the OCCA DG kernels
operate directly in the existing layout.

- `phi = q_phi[0]`
- `rho = rho_light + (phi-phi_light)*(rho_heavy-rho_light)/(phi_heavy-phi_light)`
- `p = q_g[0]`
- `(u,v) = (q_g[1],q_g[2])/(rho*c)`

Both distributions use H2 Hermite transport and the existing BNS
characteristic upwind flux, specialized to the Cartesian coordinate axes.
DG BGK relaxation uses `N+2` Gauss points in each direction and GLL lumped
mass projection. Its conserved collision moments are explicitly zero.
`tau_g = VISCOSITY/c²`; `HCZ TAU PHI = 0` selects the same relaxation time,
and a positive value specifies the index relaxation time independently.

The thermodynamic corrections are kept distinct:

- index source: gradient of `p_CS(phi) - c²*phi`;
- hydrodynamic source: gradient of `p - c²*rho`.

The Carnahan–Starling EOS is evaluated only on the nodal/subcell index,
with `eta=b*phi/4`,
`p_CS=c²*phi*(1+eta+eta²-eta³)/(1-eta)³-a*phi²`, and
`a=(HCZ EOS A OVER B C2)*b*c²`. Defaults are `b=4`, `a/(b*c²)=3`, and
index endpoints `0.02381`, `0.2508`. These differ from the physical density
endpoints, which default to `1`, `3`.

Central DG derivatives form the thermodynamic gradients and the optional
capillary force `kappa*phi*grad(laplacian(phi))`. Gravity is the full force
per volume `(0,-rho*g)`. Source normalization does not reuse the legacy
BNS body-acceleration macro. No additional density factor is applied.
`HCZ KAPPA=0` disables capillarity for RT.

## Subcell stabilization with LSERK4

For each of the five existing LSERK4 stages, the new optional solver
callback evaluates a DG RHS and constructs a candidate using the actual
low-storage recurrence:

```text
res_candidate = a*res_previous + dt*rhs
q_candidate   = q_stage + b*res_candidate
```

At the first stage `a=0`, so uninitialized residual memory is not read.
The callback leaves both the accepted stage state and its residual history
untouched. On acceptance, the existing OCCA update applies the selected RHS
exactly once. Other solvers inherit a callback that simply calls `rhsf`.
The existing PML time-stepping path is unchanged.

The detector checks the candidate index against the accepted stage's 3 × 3
neighborhood envelope, relaxed by
`max(DMP_DELTA*index_jump, DMP_EPSILON*local_range)`. It also checks phase
bounds, finite moments, positive nodal index/density, and positive finite
BGK cubature density. Cubature index positivity is not imposed: BGK uses
mapped density there and does not evaluate the EOS.

Rejected elements and a configurable neighbor buffer use `(2N+1)²` equal
subcells. All twelve moments are projected to subcell averages. Auxiliary
index positivity scaling preserves all six index moment averages. Subcell
transport uses characteristic WENO3-Z reconstruction with exact upwind fluxes
by default (`HCZ WENO FLUX=CHARACTERISTIC`); the original componentwise
Lax–Friedrichs splitting remains available with `HCZ WENO FLUX=LF`;
thermodynamic derivatives use centered differences (one-sided physical
pressure-correction derivatives at y walls), and BGK is pointwise.

The reconstructed residual has its element mean corrected explicitly.
At mixed DG/FV faces, the DG neighbor receives the same integrated WENO
flux as the subcell element. This preserves the global index integral and,
through the affine density map, total physical mass under closed/periodic
boundaries.

Newly rejected elements outside the current mask trigger another
recomputation. As in the Python prototype, already marked WENO elements
may retain relaxed-DMP/phase-bound excursions. Hard physical checks still
apply to the final candidate. Unresolved new rejections after the pass
limit, invalid element averages, or a nonphysical final candidate stop the
run with a diagnostic; the code does not silently clip the evolved state.
The optional native DG positivity limiter from Python is not enabled/ported.

LSERK4 is not SSPRK3 and carries no SSPRK3 SSP guarantee. This adaptation
preserves the LSERK4 residual history; it is not an SSPRK convex-combination
port. Stability of a full nonlinear RT evolution still needs a resolution
and time-step study. The reference test also compares a rejected stage
with nonzero low-storage residual history.

The fixed timestep is the minimum of

```text
CFL / (sqrt(3)*c*(N+1)²*(1/hx+1/hy))
HCZ_COLLISION_CFL * min(tau_phi,tau_g)
HCZ_SUBCELL_CFL / (HCZ_WENO_ALPHA*c*(2N+1)*(1/hx+1/hy))  [when WENO enabled]
```

The `(N+1)²` factor follows the existing BNS degree scaling; the Python
SSPRK driver uses `2N+1`, so equal CFL settings do not mean equal timesteps.
Nonzero kappa also applies an additional heuristic capillary timestep
restriction; it is not a stability proof.

## Setups and output

The full RT setup has domain 1 × 4, densities 1/3 (Atwood 0.5), `c=1`,
`g=0.04`, `nu=0.00078125`, and thus reference Mach 0.2 and Reynolds 256.
The interface is `y=2+0.1*cos(2*pi*x)`, initialized with the Python
slope-adjusted tanh profile of 10–90 width `0.03125`. The smoke setup
uses a wider interface on a coarse 4 × 16 grid.

Hydrodynamic pressure is initialized independently by columnwise trapezoid
integration of `dp/dy=-rho*g` on the unique global GLL y coordinates, with
the average index endpoint EOS pressure as the reference offset. It is
not initialized as `p_CS(rho)`. For the full setup, model time 25
corresponds to reference time `t*=5`.

Available initial states are `RT`, `SLAB`, `BUBBLE`, and `UNIFORM`.
Slab half-width is 0.2 Lx; bubble radius is 0.2 min(Lx,Ly); both are
centered in the box. Periodic y requires zero gravity.

VTU output includes `Index`, `Density`, `Pressure`, and `Velocity`.
Each rank writes a separate file. CSV diagnostics report index and mass
integrals/errors, index extrema, maximum speed, troubled elements, and
candidate/recomputation counts. Output uses GLL points and their subquads.
The final state is always reported (`step=-1`). Counts include temporary
output-sampling steps taken by the existing LSERK4 implementation.

### Plot the elements using subcells

New VTU output also includes cell arrays `SubcellActive` (0: DG, 1: WENO,
including buffer elements) and `ElementId` (global Cartesian element ID).
Each plotting subquad inherits its parent DG element's flag. These flags
are the actual buffered masks used by the update, not a detector rerun on
the reported state. `TimeValue` stores the physical output time.

For SARK4, the flag is the fixed mask of the successful full-tableau pass
that produced the frame. For LSERK4, it is the final stage's mask, not the
union of all stage masks. Initial frames and pure-DG runs have zero flags.
Scheduled output mini-steps report their own masks. SARK4 saves/restores
the full-step mask around that temporary calculation so the final state
is reported with its own mask. Mask downloads occur only during output;
ordinary OCCA stages still have no full-array host transfers.

In ParaView, color by the **cell** array `SubcellActive`, using discrete
colors at 0 and 1. Threshold at 1 to isolate subcell elements, or overlay
their edges on `Index`, `Density`, `Pressure` or `Velocity`. Do not convert
the flag to point data: that would interpolate the binary mask.

For a standalone plot, from `solvers/bns`:

```sh
python3 plotHczSubcells.py 'hcz_rt_sark4_*_0010.vtu' \
  --interface-value 0.137305 -o hcz_interface_0010.png
```

Pass all MPI pieces of **one** frame. The utility requires NumPy and
Matplotlib and creates index, pressure, speed and mask panels. Orange
outlines identify marked parent elements; the black line is the selected
index contour. The example contour value is the midpoint of the default
index endpoints. Without `--interface-value`, the observed index range's
midpoint is used. Field colors interpolate the saved GLL samples; they do
not evaluate the full DG polynomial between those samples.

Existing VTU files cannot provide the historical mask: rerun the desired
test with the rebuilt solver and `OUTPUT TO FILE=TRUE`. No physical model,
detector threshold or time-integration formula is changed by this output.

Validate output masks, MPI/backend agreement, zero-transfer stage counts,
output sampling and plotting with:

```sh
python3 test/testHczSubcellOutput.py --cuda --artifacts /tmp/hcz_subcell_plots
```

This check passed for both steppers on HOST, Serial/OCCA, VERIFY, OpenMP
and CUDA, including two-rank MPI. Every frame's parent-element flags agree
with the CSV count and across backends. Tests also cover initial/pure-DG/
uniform masks, a zero-duration run, and final-state agreement with and
without output mini-steps. The SARK4 stepper tests additionally check
auxiliary-state restoration after a rejected output mini-step.

### Optional interface-based subcell activation

The default remains candidate-triggered recomputation. To use subcells at
the interface from the first update, add:

```text
[HCZ SUBCELL WENO]
TRUE
[HCZ INTERFACE SUBCELLS]
TRUE
[HCZ INTERFACE THRESHOLD]
0.01
```

Let `alpha = (Index - phi_light)/(phi_heavy - phi_light)`. An element is
seeded when its **nodal range** intersects the open band
`threshold < alpha < 1-threshold`: `max(alpha)>threshold` and
`min(alpha)<1-threshold`. This also selects a cell whose nodes lie on
opposite sides of the band even if none lies inside it. This is an
element-local nodal sensor, not exact interface geometry or a polynomial
extrema calculation. The existing candidate detector remains active.
The threshold must be finite and strictly between 0 and 0.5; smaller
values generally mark a wider band. The option requires WENO to be enabled.

SARK4 builds this seed from the unchanged timestep input, expands it using
`HCZ BUFFER LAYERS`, and uses the resulting hybrid operator for stage zero
and all subsequent stages. It refreshes the seed for each new attempt,
including retries and output mini-steps. During a tableau pass the mask
stays fixed; new detector marks grow it and trigger the existing complete
replay. Initial seeding uses no recomputation pass because no RHS has yet
been evaluated. Cells can return to DG on a later step when they leave
the interface band and no detector/buffer criterion selects them.

LSERK4 refreshes the seed from each stage input, unions it with the raw
candidate detector mask, then buffers the union and runs the existing
hybrid recomputation loop. Thus its first accepted stage also uses WENO
at seeded elements. HOST/OCCA/VERIFY support both steppers. VERIFY checks
the raw seed and combined/buffered mask against the HOST implementation;
`interface_checks` counts raw seed comparisons.

VTU now includes **`InterfaceMarked`**, the unbuffered seed used for that
update, separately from **`SubcellActive`**, the actual computation mask
including buffers and additional detector marks. At **t=0**, the interface
seed is visible but `SubcellActive` is zero because no update has occurred.
It becomes active during the first timestep. Reported seeds follow the
same SARK4 output save/restore rules as computation masks. They are not
redetected on the output state. The plotting utility adds a separate
interface-seed panel when this array is present; older mask-only VTU
files retain the four-panel plot.

Two new setups enable the option while preserving the original setups:

```sh
cd solvers/bns
# Short Serial/VERIFY smoke test:
./bnsMain setups/setupHczSark4InterfaceQuad2D.rc
# Copy of the current 32 x 128, degree-two CUDA/OCCA RT setup, t_final=25:
./bnsMain setups/setupHczRTSark4InterfaceQuad2D.rc
python3 plotHczSubcells.py 'hcz_rt_sark4_interface_*_0010.vtu' \
  --interface-value 0.137305 -o interface_seed_0010.png
```

The full setup writes `hcz_rt_sark4_interface_*`, keeping the baseline
outputs separate. This policy changes the spatial operator near the
interface and may change diffusion, replay counts and runtime. Short
validation does not establish that it removes late-time RT wiggles.

Validation commands from the repository root:

```sh
python3 test/testHczDetector.py --modes Serial OpenMP CUDA
python3 test/testHczSubcellOutput.py --interface --cuda --artifacts /tmp/hcz_interface_plots
python3 test/testHczSubcellOutput.py --first-step-only --cuda
python3 test/testHczSark4WenoIntegration.py --cuda
python3 test/testHczHybridIntegration.py --cuda
```

The sensor tests cover phase endpoints, exact band edges, samples just
inside the edges, crossing the band between nodes, a shifted interface,
permuted/subset/empty element lists and untouched halo entries. Integration
checks cover first-step activation, CPU/GPU/MPI agreement, conservation,
zero ordinary OCCA stage transfers, 0/1/2 buffer layers, uniform/pure-DG
states and output save/restore. A focused first-step check uses permissive
detector tolerances: seeding off produces no subcell activity, whereas
seeding on makes all five RHS evaluations hybrid without a SARK4 replay.
All five commands above passed on the available CPU/GPU. In the focused
4 x 16 first-step fixture, the seed marks eight elements and one buffer
layer expands the computation mask to sixteen; the detector-only control
marks zero. Serial/OpenMP/CUDA and MPI agree, with conservation errors
below `2e-11`. The original SARK4 and LSERK4 hybrid regression suites pass
with interface seeding disabled.

## Validation

```sh
python3 test/testHcz.py --reference-dir /path/to/HCZ_DG
python3 test/testHczIntegration.py
```

The operator comparison requires NumPy, SciPy, Matplotlib, and the two
original Python scripts. It compares degrees 1–3, DG/BGK/source residuals,
mixed DG/FV residuals, projections, and LSERK candidate recomputation.
A periodic nonzero-capillarity case with `c=1.3` and unequal relaxation
times checks the independent pressure/velocity normalizations.

The integration suite checks RT with forced detector activity, 1/2/4-rank
agreement, mass conservation, uniform equilibrium, pure DG, a short
16 × 64 RT run, the legacy BNS uniform-flow norm, and unsupported-mode
rejection. Temporary outputs and OCCA kernels live under `/tmp`.
These are operator and short-run checks, not validation of the complete
`t*=5` Rayleigh–Taylor benchmark.

## OCCA surface migration: increment 1

The ordinary `bnsSurfaceQuad2D` kernel now has an isolated HCZ variant,
selected at kernel build time with `p_HCZ=1` and `p_Nfields=12`. The
default (`p_HCZ` absent or zero) retains the existing six-field BNS path.
HCZ requires twelve fields; its build excludes the legacy PML and multirate
helpers/kernels. The ordinary kernel name, signature, element blocking,
face ordering, shared accumulation, geometric scaling and additive RHS
update are unchanged.

`surfaceTerms()` applies the same H2 upwind correction separately to
fields 0–5 and 6–11. HCZ uses double-precision inverse-square-root
constants to agree with the CPU reference; the legacy branch retains its
original constants. No EOS or primitive-variable conversion is involved
in the HCZ surface calculation.

The HCZ input boundary contract is `EToB <= 0` for interior/periodic
traces and `EToB == 1` for stationary kinetic walls. A wall copies the
interior moments and reverses moments 1 and 2 within each six-field
distribution. Integration must validate unsupported boundary tags before
launching the kernel.

This increment does **not** connect the kernel to the HCZ solver. The
current host operator remains the active execution path. Matching the RT
periodic-x face maps and using the production trace halo belong to the
future integration increment; the isolated test supplies its own maps and halo values.

Run the isolated comparison from the repository root:

```sh
python3 test/testHczSurface.py
# On a machine with a working CUDA device:
python3 test/testHczSurface.py --modes CUDA
```

The test builds against the existing OCCA library, uses temporary files,
and requires neither MPI nor the Python HCZ scripts. It checks degrees
1–3 on Serial and OpenMP, comparing the surface-only contribution with
the CPU HCZ transport algebra. Cases cover unequal element dimensions,
both distributions independently, zero hydrodynamic pressure at walls,
matching/discontinuous traces, periodic and halo-indexed traces, corner
accumulation, permuted/subset element lists, partial blocks, repeated
additive updates, and index conservation after adding the CPU volume
transport. It also checks the default six-field BNS periodic surface.

Observed maximum HCZ differences were approximately `7.1e-15`, `3.6e-14`,
and `7.1e-14` for degrees 1, 2, and 3 on both CPU backends. The degree-two
HCZ kernel also passed OCCA CUDA translation and offline NVCC compilation.
CUDA execution is unverified because the NVIDIA driver/device was not
available in the test environment.

## OCCA volume migration: increment 2

The ordinary `bnsVolumeQuad2D` kernel now supports isolated HCZ **transport
only**, selected with `p_HCZ=1` and `p_Nfields=12`. It applies the existing
six-moment H2 transport equations independently to fields 0–5 and 6–11.
The signature, field layout, element lists, shared storage, thread/block
organization, differentiation matrix and geometric factors are retained.
The volume kernel overwrites the selected elements' RHS; the surface
kernel then adds its correction. Unselected RHS entries are preserved.

The HCZ build excludes legacy primitive conversion, body forcing and PML
kernels. HCZ central DG gradients and thermodynamic/body/capillary sources
require subsequent increments; isolated cubature BGK is covered below. In particular,
the legacy body-force normalization cannot be reused for HCZ. The default
BNS branch, including PML, is unchanged; its preprocessed volume source
was checked against the version before this increment.

Both HCZ transport kernels remain disconnected from the production HCZ
solver, which continues to use the host reference operator. This increment
does not change LSERK4, the detector, subcell recomputation or halo exchange.

```sh
python3 test/testHczVolume.py
# On a machine with a working CUDA device:
python3 test/testHczVolume.py --modes CUDA
```

The volume test and existing surface test share the fixture in
`test/hczQuad2DKernelTest.hpp`. No MPI or original Python scripts are
needed. Tests cover degrees 1–3 on unequal Cartesian element dimensions,
constant states, analytic tensor polynomials, discontinuous states, each
distribution independently, zero index/pressure moments, permuted and
subset element lists, partial blocks and input preservation. Repeated
volume-then-surface calls verify overwrite/additive assembly semantics.
The combined operator is compared against the CPU transport algebra with
periodic/halo traces and stationary walls, including global index mass
conservation. A default six-field BNS case checks zero-force transport.
The surface-only regression suite also passes after sharing the fixture.

On both Serial and OpenMP, maximum absolute volume differences were
approximately `5.3e-15`, `3.6e-14`, and `6.5e-14` for degrees 1, 2, and 3;
combined volume-plus-surface differences were `1.4e-14`, `4.3e-14`, and
`8.5e-14`. The degree-two HCZ volume kernel also passed OCCA CUDA
translation and offline NVCC compilation. GPU execution remains unverified.

## OCCA relaxation migration: increment 3

The ordinary `bnsRelaxationQuad2D` kernel now has an isolated explicit HCZ
BGK variant (`p_HCZ=1`, `p_Nfields=12`). It keeps the nodal loading,
two-stage tensor interpolation to `N+2` Gauss points per direction,
two-stage projection, geometry weights, element lists, blocking, and
additive RHS update. Inactive lanes of partial HCZ blocks are initialized
so intermediate interpolation and projection do not read undefined data.

At each cubature point, density is mapped from the interpolated index,
`rho = rhoL + (phi-phiL)*(rhoH-rhoL)/(phiH-phiL)`, and the shared velocity
is `u=q7/(rho*c)`, `v=q8/(rho*c)`. The independent pressure `q6` is not
used as density. The index equilibrium uses `phi*u/c`, `phi*v/c`,
`phi*u*v/c^2`, `phi*u^2/(sqrt(2)*c^2)` and
`phi*v^2/(sqrt(2)*c^2)`. Hydrodynamic stress equilibria are `rho*u*v`,
`rho*u^2/sqrt(2)` and `rho*v^2/sqrt(2)`.

Fields 1–5 receive `(equilibrium-q)/tauPhi`; fields 9–11 receive
`(equilibrium-q)/tauG`. Only these eight fields are projected and added
to the RHS. Fields 0, 6, 7 and 8 have exactly zero collision; their existing
RHS entries are left untouched. This preserves transport contributions
already assembled into those fields.

The HCZ call signature adds six scalars after `tauInv` and before `q`:

```text
Nelements, elementIds, vgeo, cubvgeo, cubInterp, cubProject,
semiAnalytic, tauInv, c, tauPhiInv, rhoL, rhoH, phiL, phiH, q, rhsq
```

Here `tauInv=1/tauG` and `tauPhiInv=1/tauPhi`. HCZ always evaluates the
full explicit collision. Its caller contract requires `semiAnalytic=0`,
positive finite `c` and relaxation times, a valid density map, finite
input moments, and positive mapped density at cubature points. No clipping
or density floor is introduced in this kernel. Host validation of this
contract belongs to the later integration increment, alongside the
existing HCZ admissibility checks.

Device `cubInterp` and `cubProject` use the existing mesh's column-major
storage. `cubProject` is the unweighted interpolation transpose; `JW` at
cubature points and nodal `invJW` supply the weights and inverse lumped
mass. Passing the CPU reference's already weighted `P` here would apply
weights twice. The resulting operator agrees with the CPU projection on
the supported Cartesian quads.

```sh
python3 test/testHczRelaxation.py
# On a machine with a working CUDA device:
python3 test/testHczRelaxation.py --modes CUDA
```

The isolated tests compare cubature BGK with the CPU tensor projection
and collision, and compare volume+surface+BGK with the CPU DG operator
after subtracting its separate sources. Degrees 1–3 cover uniform
equilibrium, constant and varying nonequilibrium states, zero pressure,
coupling through hydrodynamic momentum, unequal relaxation times, two
sound speeds and density maps, exact preservation of conserved RHS
entries, repeated additive updates, permuted/subset element lists,
partial blocks, unchanged input states, and assembled index conservation
with periodic traces and stationary walls. Default six-field BNS explicit
and semi-analytic relaxation are also checked. The default BNS
preprocessed source, including PML, is identical to the original version.

On Serial and OpenMP, maximum absolute differences after repeated collision
updates were approximately `2.8e-14`, `1.3e-13`, and `1.4e-13` for degrees
1, 2, and 3. The corresponding volume+surface+BGK comparisons differed by
`1.4e-14`, `5.7e-14`, and `6.9e-14`. The degree-two HCZ relaxation kernel
passed OCCA CUDA translation and offline NVCC compilation; GPU execution
remains unverified.

This increment remains isolated from production execution. The HCZ solver
continues to use the host operator; gradients, sources, solver integration,
and PML are not implemented by this increment.

## OCCA thermodynamics and gradients: increment 4

Three isolated HCZ Quad2D kernels prepare the nodal quantities needed by
the source calculation:

| Kernel | Operation |
| --- | --- |
| `bnsHczThermodynamicsQuad2D` | Compute mapped density, `psiPhi = pCS(phi)-c^2*phi`, and `psiRho = q6-c^2*rho` |
| `bnsHczGradVolumeQuad2D` | Initialize the local volume derivatives of both potentials |
| `bnsHczGradSurfaceQuad2D` | Add the central-trace face corrections to those derivatives |

Each kernel lives in its corresponding `.okl` file under `solvers/bns/okl`.
The gradient split follows the existing libp Quad2D gradient pattern:
shared nodal values and differentiation matrix, physical derivatives from
`vgeo`, and separate opposing-face stages for safe corner accumulation.
All three kernels accept selected element lists and support partial blocks.
The volume gradient overwrites selected output entries; the surface gradient
adds to them. Unselected entries and all input buffers remain untouched.

The buffer layouts are:

```text
q:          [element][12 moments][node]
rho:        [element][node]
potentials: [element][psiPhi, psiRho][node]
gradients:  [element][psiPhi_x, psiPhi_y, psiRho_x, psiRho_y][node]
```

Thermodynamics takes `c, rhoL, rhoH, phiL, phiH, eosB, eosRatio` as scalar
parameters. It evaluates the Carnahan–Starling expression at DG nodes,
using the index rather than the mapped density as the EOS argument.
Hydrodynamic pressure remains the independent moment `q6`. It performs
no primitive-velocity conversion, clipping, relaxation or RHS update.
The caller must provide finite admissible nodal data and valid density/EOS
parameters, including a nonsingular EOS denominator.

For either potential, the gradient surface contribution is
`0.5 * invWJ * sJ * normal * (potentialP-potentialM)`. Interior and periodic
traces (`EToB <= 0`) use the supplied maps. Stationary walls (`EToB == 1`)
copy the interior scalar, so their surface correction is zero and their
exterior map is not read. The local volume derivative remains present at
the wall. This matches `QuadOperator::derivative` in DG mode; the one-sided
finite-volume derivative used by the subcell operator is a separate task.
Integration must reject unsupported boundary tags.

Future production assembly must prepare the owned potentials and exchange
their traces before launching the gradient surface kernel. Volume gradients
can be computed while that exchange proceeds. The isolated tests supply
halo values explicitly; this increment does not implement MPI exchange.

```sh
python3 test/testHczGradients.py
# On a machine with a working CUDA device:
python3 test/testHczGradients.py --modes CUDA
```

The tests check degrees 1–3 on unequal Cartesian element dimensions using
Serial and OpenMP. They cover two sound speeds, density maps and EOS parameter
sets, uniform and varying states, zero pressure, constant and discontinuous
potentials, an independent potential, and analytic continuous tensor
polynomials. Volume, surface-only and complete central derivatives are
compared with the host reference, including a complete thermodynamics-to-
gradient comparison against independently prepared CPU potentials. Periodic
and supplied halo traces, walls with invalid unused exterior maps, corners,
permuted/subset element lists, partial blocks, repeated overwrite/additive
updates, input preservation and output halo sentinels are checked.

On both CPU backends, thermodynamic differences were at most `2.3e-16`.
Complete central-gradient differences, including preparation roundoff, were
approximately `7.2e-16`, `5.0e-15`, and `9.7e-15` for degrees 1, 2, and 3.
Separate volume and repeated surface-only comparisons differed by at most
`5.0e-14` and `1.0e-13`. All three degree-two kernels passed OCCA CUDA
translation and offline NVCC compilation; GPU execution remains unverified.

Source assembly, gravity, capillary derivatives, subcell operators and solver
integration remain subsequent increments. Production HCZ execution still
uses the host operator; the three new kernels do not modify its active path.

## OCCA source assembly: increment 5

`bnsHczSourceQuad2D` adds the local nodal HCZ sources for **kappa=0** to
the existing RHS. It reads the twelve moments and the four central DG
gradients prepared in increment 4. It recomputes mapped density from
`q0`, then uses `u=q7/(rho*c)` and `v=q8/(rho*c)`. Gravity is a force
density: `Fx=0`, `Fy=-rho*gravity`, with positive gravity directed downward.
The moment sources match `hcz::source` with zero capillary inputs.

The call signature is:

```text
Nelements, elementIds, c, gravity, rhoL, rhoH, phiL, phiH,
q, gradients, rhsq
```

`q` and `rhsq` have layout `[element][12 moments][node]`; `gradients`
has layout `[element][psiPhi_x, psiPhi_y, psiRho_x, psiRho_y][node]`.
Selected elements are processed with the existing `p_NblockV` organization,
including partial blocks. The kernel adds directly to nodal RHS entries:
there is no additional geometry scaling, cubature weighting or time-step
factor. All neighbor information is already contained in the gradients.

Fields 1–5 receive the index-potential sources, field 6 receives the
velocity contraction with the hydrodynamic-potential gradient, field 8
receives `-c*rho*gravity`, and fields 9–11 receive the stress sources.
Fields 0 and 7 have zero source for this scope and are left exactly
untouched, preserving prior transport/collision contributions. Unselected
elements and all inputs remain unchanged.

The caller must provide finite input moments and gradients, positive `c`,
a valid density map with positive nodal density, and `kappa=0`. No clipping
or altered force normalization is introduced. The kernel has no capillarity
parameter or capillary derivatives; later integration must validate that
this kernel is used only for its supported scope.

The isolated complete DG sequence now consists of:

1. Prepare thermodynamic quantities and supply potential halos.
2. Compute central gradients with volume and surface kernels.
3. Initialize the RHS with volume transport and add surface transport.
4. Add cubature BGK relaxation.
5. Add the nodal HCZ sources.

```sh
python3 test/testHczSource.py
# On a machine with a working CUDA device:
python3 test/testHczSource.py --modes CUDA
```

The tests check the source kernel with prescribed independent potential
gradients, then compare the complete preparation/gradient/source sequence
against `QuadOperator::addSources`. All seven OCCA kernels are also assembled
and compared directly against the complete `QuadOperator::dg` result, with
no source subtraction. Tests use degrees 1–3, unequal Cartesian dimensions,
two sound speeds and density/EOS parameter sets, unequal relaxation times,
zero and nonzero gravity, zero pressure, uniform and varying states, periodic
and stationary-wall boundaries, and supplied halo traces. Additional checks
cover repeated additive updates, exact preservation of zero-source fields,
unselected elements and output halos, input preservation, permuted/subset
element lists, partial blocks, and global index conservation.

A constant-density, stationary state with linear pressure
`p(y)=p0-rho*gravity*y` checks hydrostatic balance at walls, including gravity's
sign and normalization. Uniform rest with zero gravity also has zero DG RHS.

Serial and OpenMP checks passed. Maximum absolute isolated-source differences
were below `5.6e-17`, and the preparation/gradient/source comparison differed
by at most `2.2e-15`. Complete DG RHS differences were approximately
`1.1e-14`, `2.8e-14`, and `5.3e-14` for degrees 1, 2, and 3 on both backends.
The degree-two source kernel also passed OCCA CUDA translation and offline
NVCC compilation; GPU execution remains unverified.

This completes the isolated DG RHS for `kappa=0`. Production execution still
uses the host operator. Solver integration, MPI exchange, detector and subcell
migration, capillarity and PML are outside this increment.

## First solver integration: pure DG, kappa=0

The seven validated kernels are now connected to `bns_t` through the HCZ
RHS and LSERK4 callbacks. Select the path with:

```text
[HCZ DG BACKEND]
OCCA
[HCZ SUBCELL WENO]
FALSE
[HCZ KAPPA]
0
```

`HOST` is the default and retains the original implementation, including
subcell recomputation and capillarity. `VERIFY` executes the OCCA path and
compares all twelve RHS moments with the host DG operator on every call.
It aborts on nonfinite results or a scaled discrepancy above the precision-
dependent tolerance (currently `2e-11` with double precision). The final
`HCZ OCCA verification` line reports the number of comparisons, maximum
absolute error, and maximum error divided by `max(1, ||host RHS||_infinity)`.
OCCA/VERIFY reject subcell mode and nonzero kappa explicitly; they do not
fall back silently to another method.

Setup builds the kernels using mesh properties and allocates native
element/field/node buffers for the state with halos, density, potentials
with halos, and gradients. The real mesh supplies `D`, `vgeo`, `sgeo`, and
the `N+2` cubature data. The HCZ Cartesian topology supplies the selected
element list, face maps, and boundary tags, ensuring periodic x and the
chosen y treatment independently of `BOX BOUNDARY FLAG`. Setup checks the
quad face-node ordering used by the surface kernels.

Halo communication uses libp's `ogs::halo_t`, with signed global element
IDs, rather than converting device data to the host reference layout.
Complete state and potential blocks are exchanged separately. The existing
HCZ ghost slots, including corners, are retained for this increment; trace
compaction is not introduced. The state exchange surrounds volume transport
and local thermodynamic preparation. The potential exchange surrounds
surface transport, gradient volume terms, and BGK. Gradient surface terms
and sources run once potential halos are ready. An owned-state scratch
buffer keeps the LSERK4 vectors and its existing update unchanged.

At this first integration increment, input and prospective LSERK4 candidate
admissibility still ran on the host. The following increment moves these
checks to OCCA while preserving the pure-DG policy, including ignoring the
uninitialized residual when the first-stage coefficient is zero.

Run the actual-mesh integration suite after building:

```sh
make -C solvers/bns -j4
python3 test/testHczDgIntegration.py
python3 test/testHczIntegration.py
```

The new suite compares HOST/OCCA/VERIFY short LSERK4 trajectories on actual
BOX meshes. It covers wall RT and periodic slab cases, Serial and OpenMP,
MPI 1/2/4 ranks, degrees 1–3, nonunit sound speeds, unequal relaxation times,
two EOS/density parameter sets, uniform rest, hydrostatic balance, index
conservation, and rejection of unsupported modes. BOX boundary flags are
deliberately different from HCZ boundary choices to check the HCZ maps.
VERIFY compares every RHS before it is used in a stage, while output-field
comparisons also check independent HOST and OCCA trajectories.

Both integration suites passed. The largest all-moment RHS difference in
the new suite was approximately `1.8e-14`, and the largest final output-field
difference was `2.8e-17`. The original suite still exercises host subcell
recomputation on MPI 1/2/4 ranks and checks legacy BNS behavior. These are
short CPU runs, not full-time RT validation or GPU execution tests. The
shipped `setupHczDgOccaQuad2D.rc` uses the short wall-case settings.

The detector/subcell OCCA migration and capillarity remain separate work.
PML, other elements, and 3D remain outside the agreed scope.

## Device admissibility: second integration increment

`bnsHczAdmissibilityQuad2D.okl` contains a read-only element checker and
`bnsHczAdmissibilityReduce`, which reduces its flags to one integer per rank.
The solver combines rank flags with `MPI_Allreduce` and retains the original
input-state and candidate rejection messages. OCCA and VERIFY use these
checks automatically; no additional setting is required.

The element checker consumes native `[element][12 moments][node]` buffers.
Input mode checks the stage state. Candidate mode evaluates
`q + rkB*(dt*rhs + rkA*residual)` in shared memory without modifying any
input or storing a complete candidate globally. When `rkA=0`, it does not
read residual memory. The existing LSERK4 update still applies the accepted
RHS exactly once.

The criteria match `QuadOperator::admissible`:

- Nodes: all twelve moments finite, index strictly above the existing
  physical floor, positive mapped density, and `eosB*index < 3.999`.
- BGK cubature: all twelve interpolated moments finite and positive mapped
  density. Cubature imposes neither the nodal index floor nor the EOS bound.

Element flags distinguish nodal failure (bit 1) and cubature failure (bit 2).
These are physical admissibility checks only; the relaxed-DMP detector,
neighbor buffering and WENO subcell recomputation remain on the HOST path.
There is no clipping or change to CFL selection or rejection policy.

VERIFY additionally forms the host candidate and checks that the global
host and device decisions agree. Its final verification line now includes
`admissibility_checks`; a successful LSERK4 run performs two such checks
per RHS evaluation.

```sh
python3 test/testHczAdmissibility.py
python3 test/testHczAdmissibility.py --modes CUDA
python3 test/testHczDgIntegration.py --cuda
python3 test/testHczIntegration.py
```

Isolated tests passed at degrees 1–3 on Serial, OpenMP and CUDA. They cover
strict nodal thresholds, NaN/Inf moments, cubature-only density and finiteness
failures, accepted cubature index overshoots, first-stage residual handling,
later-stage residual contributions, selected/empty element lists, reductions
across 256-element boundaries, repeat calls and input preservation.

Actual-mesh tests passed on Serial/OpenMP with MPI 1/2/4 ranks and CUDA with
MPI 1/2 ranks sharing the available GPU. All checked host/device decisions
agreed. Deliberately excessive steps were rejected with the original
candidate error on HOST, OCCA and VERIFY. Maximum RHS differences were
`1.8e-14` across the CPU suite and `6.5e-15` in the short CUDA cases; maximum
final output-field differences were `2.8e-17` and `1.8e-18`, respectively.
The original host/subcell suite also passed, retaining 641 recomputations
and matching Serial/MPI fields, along with the legacy BNS regression.
These are short-run checks; no full-time RT or performance result is claimed.

## Isolated troubled-cell detector: next subcell increment

`bnsHczDetectorQuad2D.okl` adds two isolated kernels matching the raw mask
from `QuadOperator::detect`:

| Kernel | Operation |
| --- | --- |
| `bnsHczIndexBoundsQuad2D` | Compute the minimum and maximum nodal index in each selected reference element |
| `bnsHczDetectorQuad2D` | Form the prospective LSERK4 candidate locally and flag troubled elements using their 3 × 3 reference neighborhood |

The bounds kernel reads the accepted stage state in native
`[element][12 moments][node]` layout and writes `[element][min,max]`.
The detector consumes those bounds and a `[owned element][9]` neighbor map
ordered `(dy+1)*3+dx+1`, including self. Negative entries denote physical
walls and are skipped. Nonnegative entries can reference halo slots;
diagonal neighbors and periodic wraps are included exactly as in the host
HCZ topology. The caller must provide current bounds for every referenced
neighbor before invoking the detector. Bounds can be reused while changing
the RHS during recomputation of the same stage.

The candidate is `q + rkB*(dt*rhs + rkA*residual)`. The first stage does not
read residual storage. Each node checks all twelve moments for finiteness,
the existing physical index/density/EOS criteria, the relaxed neighborhood
envelope, and the relaxed phase interval. Neighborhood relaxation is
`max(dmpDelta*(phiH-phiL), dmpEps*(neighborMax-neighborMin))`; the phase
relaxation is `boundTol*(phiH-phiL)`. Equality at the relaxed DMP/phase
endpoints is accepted, while the physical floor and EOS cutoff remain
strict.

At cubature points, **the detector checks only finite, positive mapped
density**. This deliberately differs from the final admissibility checker,
which checks every interpolated moment for finiteness. Neither check
imposes a cubature index floor or EOS bound. The detector does not replace
the final hard admissibility check.

The output is an unbuffered binary `int` mask indexed by element, with
`1` for troubled elements. Both kernels accept unique selected element
lists; unselected outputs and all inputs remain untouched. No complete
candidate is stored globally. No mask expansion, global reduction, state
clipping, WENO reconstruction, or RHS update occurs in these kernels.

```sh
python3 test/testHczDetector.py
# Include GPU execution when CUDA is available:
python3 test/testHczDetector.py --modes Serial OpenMP CUDA
```

The isolated test compares every selected mask entry directly against
`QuadOperator::detect`, and checks bounds independently from the nodal
reference values. It covers degrees 1–3, wall and periodic y boundaries,
diagonal/supplied halo neighbors, exclusion of distant elements, both DMP
relaxation terms, threshold equality and excursions, relaxed phase bounds,
physical cutoffs, all twelve nonfinite nodal moments, cubature-only density
failures, allowed cubature index/EOS excursions, first/later-stage residual
handling, varying candidates, selected/empty lists, repeated calls, and
input/output preservation. A cubature stress-overflow case explicitly
checks the difference between the raw detector and final admissibility.

Serial, OpenMP and CUDA execution passed: 422 cases per backend, with every
selected raw mask matching the host reference. Each case repeats both
kernels and checks exact input preservation and unselected-output sentinels.

These kernels remain isolated from production execution. The existing
OCCA/VERIFY solver still requires `HCZ SUBCELL WENO=FALSE`; HOST retains
the complete detector/buffering/WENO algorithm. At this increment, halo
values were supplied by the isolated test. The following increment adds
buffer expansion and exercises actual OGS communication for masks and bounds.

## Neighbor buffering and OGS halos

`bnsHczExpandMaskQuad2D.okl` expands a binary troubled-element mask by one
3 × 3 neighborhood layer, including diagonal neighbors. It uses the same
element-indexed masks and neighbor map as the raw detector. Negative
neighbors are skipped at walls; periodic neighbors use the supplied map.
Selected output entries are overwritten from a separate input snapshot,
so a newly marked element cannot propagate further within the same layer.
Unselected output entries and the input mask are preserved.

`hczDeviceMask.hpp` provides `hczExpandDeviceMask`, a reusable libp helper
matching the host `QuadOperator::expand` sequence:

1. Exchange the current integer mask with `ogs::halo_t`.
2. Run the one-layer expansion kernel into an owned-element scratch buffer.
3. Copy owned results back to the original device allocation.
4. Repeat for the requested number of layers.

All ranks participate in each exchange, including ranks with no owned
elements. Zero layers leaves both buffers untouched and performs no exchange.
The helper requires a selected list containing every owned element exactly
once, a distinct scratch buffer, and the same owned/ghost slot mapping as
the halo. On return, owned results are current; ghosts contain the input to
the last layer, matching the host routine. A subsequent consumer requiring
current ghost masks must exchange again, as the host hybrid operator does.
No full mask is copied to the host during expansion; communication uses
the existing OGS device-buffer interface.

The integer OGS path exposed an OKL parser issue: the kernel type spelling
`int32_t` was unresolved. `ogsUtils.cpp` now spells that kernel type `int`
and statically verifies it matches `int32_t`. Floating-point kernel types,
halo operations, and communication methods are unchanged. Initialize the
`ogs::Int32` kernels before exchanging device masks.

```sh
make -C solvers/bns -j4
python3 test/testHczMaskHalo.py
python3 test/testHczMaskHalo.py --modes CUDA --ranks 1 2 4
```

The new harness uses actual libp platforms and OGS signed global element
IDs. It compares distributed results against the global host reference
in the test, without gathering masks in the device algorithm. Coverage
includes zero through three layers, empty/full/single-seed/distributed masks,
walls and periodic wraps, diagonal halos, reversed and subset element lists,
partial 256-thread blocks, empty ranks, self-periodic neighbors, and buffer
sentinels. The detector-to-buffer test computes index bounds on owned
elements, exchanges two bounds per element with OGS, evaluates the raw
device mask, and expands that actual output by two layers. It compares
both stages with `QuadOperator::detect` and `QuadOperator::expand` at
degrees 1–3.

Validation passed on Serial, OpenMP and CUDA with MPI 1/2/4 ranks. CUDA
ranks shared the available GPU; this does not establish multi-GPU behavior.
All owned masks and exchanged bounds matched the host reference, including
empty-rank cases. The existing pure-DG CPU/CUDA integration suite and the
host WENO/legacy BNS regression suite also passed after rebuilding OGS.

Production OCCA/VERIFY execution remains pure DG with `HCZ SUBCELL WENO=FALSE`.
The detector/buffer components are validated independently before connecting
them to the subcell recomputation loop. Remaining subcell migration work is:

1. DG-to-subcell projection and conservative index positivity scaling.
2. WENO3-Z face fluxes, subcell divergence, pointwise BGK and subcell source
   derivatives, including their boundary and halo handling.
3. Subcell-to-DG residual reconstruction with mean correction, plus
   conservative flux corrections at mixed DG/subcell faces.
4. Integrate detector, buffering, mask updates and repeated hybrid RHS
   evaluations into LSERK4, preserving rejection rules and diagnostics;
   then validate complete CPU/GPU/MPI trajectories and conservation.

The following increment implements item 1. Nonzero capillarity remains
separate. PML, other elements, and 3D remain outside the agreed scope.

## Subcell projection and conservative index scaling

`bnsHczSubcellQuad2D.okl` adds two isolated kernels for the preparation
at the start of the host hybrid operator:

| Kernel | Operation |
| --- | --- |
| `bnsHczProjectSubcellQuad2D` | Project all twelve DG moments to `(2N+1)²` equal subcell averages |
| `bnsHczScaleSubcellQuad2D` | Apply the existing conservative scaling to moments 0–5 when the subcell index reaches the physical floor |

Both use selected element lists and native element/field/node storage.
The subcell layout is `[element][12 moments][subcell]`, with x varying
fastest. The projection matrix is the host operator's subcell-average
matrix `A`, supplied column-major as `A[subcell + node*(2N+1)]`. The kernel
performs the same full tensor contraction and accumulation order as
`QuadOperator::project`. These values are averages, not point samples;
there is no additional Jacobian factor on uniform Cartesian elements.

The scaling kernel reads the subcell buffer in place and computes the
minimum index and the separate averages of the six index-family moments.
If the minimum is above `Parameters::floor()`, it leaves the element
unchanged. Otherwise it uses the existing host formula:

```text
theta = (meanIndex-floor)/(meanIndex-minIndex) * (1-1e-12)
sub[k] = mean[k] + theta*(sub[k]-mean[k]),  k=0,...,5
```

All six index moment averages are preserved to roundoff. Hydrodynamic
moments 6–11 are untouched. If the mean index is at or below the floor,
the element is left unchanged and `invalid[element]=1`; other selected
elements receive zero. This is an element-indexed failure flag, not a
troubled-cell mask. Unselected outputs, halo slots and DG input data are
preserved. The helper is not a replacement for state admissibility checks
and assumes finite subcell inputs, as does the original host block.

The original host scaling block is now `QuadOperator::scaleSubcellIndex`,
called by `hybrid()` with the same collective decision and the same
`HCZ subcell index mean cannot be repaired conservatively` error. This
allows tests to compare directly against the production host routine.
Its arithmetic and strict/weak inequalities are unchanged. At a mean
within roundoff of the floor, the scaled minimum can still round onto the
floor, as in the host version; no extra clipping or acceptance rule is added.

```sh
python3 test/testHczSubcell.py
python3 test/testHczSubcell.py --modes Serial OpenMP CUDA
make -C solvers/bns -j4
python3 test/testHczIntegration.py
```

The isolated tests passed on Serial, OpenMP and CUDA for degrees 1–3.
They compare host/device projection and scaling, analytic polynomial
subcell averages, all twelve element means, the projection-to-scaling
sequence, and failure flags. Cases include valid DG states whose projection
requires scaling, both index- and density-imposed physical floors,
threshold equality, repairable minima, invalid means, mixed elements,
selected/empty lists, repeated projection calls, and unchanged inputs,
hydrodynamic moments and halo sentinels. Maximum kernel differences were
`6.7e-16`; maximum element-mean differences were `7.4e-15`. The host WENO
MPI 1/2/4 and legacy BNS runtime regressions also passed after extraction
of the shared reference helper.

These kernels remain isolated from production GPU subcell execution.
Integration must project and prepare all owned elements needed by the
hybrid stencil, including DG neighbors, collectively check invalid means,
and exchange the prepared twelve-moment subcell data before evaluating
WENO fluxes. Those steps are not wired into the solver by this increment.
OCCA/VERIFY still requires `HCZ SUBCELL WENO=FALSE`.

The following increment adds WENO3-Z subcell face fluxes. Subcell
divergence/BGK/sources, reconstruction, mixed-face conservation corrections,
and LSERK4 recomputation integration remain afterward.

## Isolated WENO3-Z subcell face fluxes

`bnsHczWenoFaceQuad2D.okl` evaluates all x and y subcell face fluxes for
selected elements, matching `QuadOperator::wenoFace`. The signature is:

```text
Nelements, elementIds, neighbors, c, alpha, epsilon, sub, faces
```

`sub` contains the prepared twelve-moment subcell averages from the previous
increment in `[element][12 moments][y subcell][x subcell]` order. With
`ns=2N+1`, the output layout is
`[element][axis][12 moments][tangent subcell][normal face]`, where axis
0 is x, axis 1 is y, tangent runs from 0 to `ns-1`, and normal face runs
from 0 to `ns`. Each element stores `2*12*ns*(ns+1)` values. This keeps
moments in separate contiguous arrays, consistent with native solver
storage. Every flux is oriented in the positive coordinate direction;
opposite sides of a shared face therefore store the same flux, not its
negative. No cell-width, timestep or surface-integral scaling is applied.

At face `a`, the four subcell-average samples are `a-2,a-1,a,a+1`
along the normal direction. `HCZ WENO FLUX` selects:

- `CHARACTERISTIC` (default): transform each six-moment family into the
  orthonormal eigenbasis of the Cartesian transport matrix. Reconstruct
  positive-speed states from the left and negative-speed states from the
  right, then form `A+ q_L + A- q_R`. The speeds are
  `+sqrt(3)*c, -sqrt(3)*c, +c, -c, 0, 0`. The stationary modes have zero
  flux. This is the same exact upwind operator as DG, applied to WENO
  interface states instead of DG traces.
- `LF`: retain the Python prototype's componentwise splitting
  `F± = (F ± alpha*c*q)/2`, then reconstruct each split moment flux.

Both use WENO3-Z indicators, `tau=abs(beta0-beta1)`, exponent two,
left linear weights `(1/3,2/3)` and right weights `(2/3,1/3)`.
`HCZ WENO EPSILON` regularizes squared characteristic-state differences
in CHARACTERISTIC mode and squared split-flux differences in LF mode.
There is no normalization by index, density or pressure.

`HCZ WENO ALPHA>=sqrt(3)` still bounds the subcell transport speed for
**both** time integrators and both flux choices, preserving the existing
CFL cap for controlled comparisons. It affects the numerical flux only
in LF mode. Kernel selection is specialized at build time using
`p_HCZ_CHARACTERISTIC_WENO`; HOST and VERIFY use the same setting.
The kernel signature and face layout are unchanged.

The mixed DG/subcell correction consumes these same computed face fluxes,
including its existing mean-preserving projection to the DG face. Thus
changing reconstruction does not introduce a second, inconsistent flux at
mixed faces. Source, relaxation, detector, interface seeding and buffer
settings are unchanged. This change does not itself establish reduced RT
wiggles or long-time stability; compare both choices at the same mesh,
CFL, interface threshold and output times.

Example setup entry (omit it to use the new default):

```text
[HCZ WENO FLUX]
CHARACTERISTIC
```

Use a different output prefix for each comparison run. The original
Python reference test explicitly selects LF because its transport scheme
has not changed; characteristic validation uses eigenmode/upwind checks,
HOST/device comparisons and hybrid conservation tests.

The neighbor map uses the existing nine-entry HCZ ordering and may point
to complete halo subcells. x is periodic; y can be periodic or a stationary
kinetic wall. At y walls, both ghost layers reflect subcell positions and
reverse moments 1, 2, 7 and 8, exactly as in the host `sample()` routine.
For degree at least one, the stencil reaches only immediate neighboring
elements. The caller must supply current prepared subcells in every
referenced halo slot; the face kernel itself performs no communication.

Each workgroup processes one selected element with 128 threads and a
strided face loop. The output is overwritten for selected elements;
unselected face arrays, output halo slots and all input buffers are
untouched. The computed faces are retained for the later divergence and
conservative mixed DG/subcell interface corrections.

```sh
python3 test/testHczWenoFace.py
python3 test/testHczWenoFace.py --modes CUDA
```

Validation passed on Serial, OpenMP and CUDA at degrees 1–4, with wall
and periodic boundaries: 640 characteristic cases and 360 LF cases per
backend. Degree four exercises
more faces than threads. Tests cover all twelve moments independently,
uniform physical H2 fluxes, smooth and discontinuous fields, alternating
and small-amplitude data, two sound speeds/flux-splitting parameters/epsilon
values, sign symmetry, periodic and supplied halo traces, both wall ghost
layers, permuted/subset/empty element lists, repeated overwrite calls,
and input/output preservation. A complete device projection → conservative
scaling → face-flux sequence is compared with the corresponding host
sequence; the harness supplies halo values after scaling.

Additional characteristic checks cover orthonormal transform inversion,
diagonalization of the transport matrix, equality to DG exact upwinding,
all moving/stationary eigenmodes in both directions and families, alpha
independence, and third-order smooth sine cell-average reconstruction.
The maximum host/device flux difference was below `2e-15`. Both sides of
shared faces agreed exactly in these tests; index and pressure-family
transport balance over the closed/periodic domain was within `3e-16`.
These checks validate isolated fluxes and supplied-halo behavior, not
MPI exchange of subcell states or a complete WENO time evolution.

Production OCCA/VERIFY execution now includes the downstream subcell
divergence, source, reconstruction, mixed-face correction and timestep
control kernels described below. The characteristic change also passed
the mixed-surface and complete LSERK4/SARK4 integration suites on CPU
and CUDA, including CPU MPI ranks 1/2/4 and CUDA MPI ranks 1/2.
Capillarity remains separate; PML, other element types and 3D stay
outside the agreed scope.

## Isolated subcell divergence and pointwise BGK

`bnsHczSubcellRhsQuad2D.okl` initializes the subcell residual with the
same collision-minus-divergence calculation as `hybrid()` immediately
before `addSources(sub,fv,true)`:

```text
rhs[k] = collision[k]
       - (Fx[right,k]-Fx[left,k])/(hx/ns)
       - (Fy[top,k]-Fy[bottom,k])/(hy/ns)
```

Its signature is:

```text
Nelements, elementIds, c, tauPhi, tauG, rhoL, rhoH, phiL, phiH,
hx, hy, sub, faces, rhs
```

`sub` and `rhs` use `[element][12 moments][y subcell][x subcell]` storage.
`faces` uses the previous increment's
`[element][axis][12 moments][tangent subcell][normal face]` layout.
With `ns=2N+1`, the Cartesian subcell widths are `hx/ns` and `hy/ns`.
The kernel overwrites selected residual entries, ready for later source
addition; it does not accumulate onto an earlier DG or subcell residual.
Input subcells and stored faces remain unchanged, as do unselected output
entries and output halos. There are no timestep, DG quadrature or projection
factors in this operation.

Pointwise BGK uses the affine index-to-density map and
`u=q7/(rho*c)`, `v=q8/(rho*c)`, with the same equilibria as the host
`collision()` function. Moments 0–5 use `tauPhi`, moments 6–11 use
`tauG`; collision entries 0, 6, 7 and 8 are explicitly zero. Independent
hydrodynamic pressure `q6` is never used as a velocity denominator.
The caller supplies finite states/faces, positive mapped density,
positive sound speed/relaxation times/element widths, and valid map endpoints.
The kernel performs no state repair, force evaluation or halo exchange.

```sh
python3 test/testHczSubcellRhs.py
python3 test/testHczSubcellRhs.py --modes CUDA
```

Validation passed on Serial, OpenMP and CUDA for degrees 1–4, with wall
and periodic boundaries: 200 cases per backend. Constant face fluxes
isolate BGK; analytic linear x/y face fields independently check divergence
signs, indexing and unequal element widths. Other checks cover two density
maps and sound speeds, independently varied relaxation times, zero pressure,
rest and periodic moving equilibria, supplied halo subcells, selected/empty
element lists, repeated overwrite calls, unchanged inputs and output halos.
The actual WENO face kernel feeds the residual kernel and is compared
against the host `wenoFace`/`collision` sequence. Conserved-moment residual
integrals are checked against boundary fluxes, including momentum at walls.

Maximum host/device residual differences were `1.5e-14`; conserved-moment
flux-balance errors were below `5.5e-16`. These are isolated operator checks,
not validation of a full subcell time evolution. Production OCCA/VERIFY
execution remains pure DG, with `HCZ SUBCELL WENO=FALSE`.

Remaining migration steps, in order:

1. Subcell thermodynamic potentials, finite-difference derivatives and
   source forces at `kappa=0`, including wall stencils and halo handling.
2. Subcell-to-DG residual reconstruction with the host mean correction.
3. Conservative flux corrections at mixed DG/subcell interfaces.
4. Integrate all components into LSERK4's detector/recomputation loop,
   including subcell exchanges, global failure decisions, mask updates,
   pass limits and diagnostics; validate full CPU/GPU/MPI trajectories
   and conservation.

The following increment completes item 1 in isolation. Nonzero capillarity
remains separate; PML, other elements and 3D remain outside scope.

## Subcell thermodynamics, derivatives and sources at kappa=0

The subcell source sequence reuses the existing pointwise
`bnsHczThermodynamicsQuad2D` and `bnsHczSourceQuad2D` kernels. Their formulas
are independent of DG interpolation or geometry, so no duplicate EOS or
source implementation is introduced. Build these two kernels with
`p_Nq=ns`, `p_Np=ns*ns` and `p_NblockV=max(1,256/(ns*ns))`, where
`ns=2N+1`. Their native element/field/node arrays then address subcells.

`bnsHczSubcellGradientQuad2D.okl` supplies the new finite-difference stage.
It is built with the original DG `p_Nq=N+1` and derives `ns=2*p_Nq-1`
internally, as the other dedicated subcell kernels do. Its signature is:

```text
Nelements, elementIds, neighbors, hx, hy, potentials, gradients
```

Potentials have layout `[element][psiPhi,psiRho][subcell]`; gradients use
`[element][psiPhi_x,psiPhi_y,psiRho_x,psiRho_y][subcell]`. Subcell x indices
vary fastest. The neighbor map follows the same nine-entry Cartesian HCZ
ordering and can reference halo potential arrays. x is periodic and y is
periodic or a stationary wall.

Both potentials use centered differences in the interior, with denominators
`2*hx/ns` and `2*hy/ns`. At a y wall, the index potential uses the boundary
subcell's scalar value as its exterior value, matching `scalarSample()`.
The hydrodynamic pressure-correction potential instead uses the host's
three-point one-sided derivative: `(-3*v0+4*v1-v2)/(2*hy/ns)` at the bottom
and `(3*v0-4*vMinus1+vMinus2)/(2*hy/ns)` at the top. This is distinct from
the reflected kinetic ghost stencil used by WENO transport.

The complete isolated subcell RHS sequence is now:

1. Prepare `rho`, `psiPhi=pCS(phi)-c²*phi` and `psiRho=q6-c²*rho` on owned
   subcells using the existing thermodynamics kernel.
2. Exchange the two potential arrays through OGS (`k=2*ns*ns`).
3. Compute the four subcell derivatives with the new gradient kernel.
4. Evaluate WENO face fluxes and initialize the residual with divergence/BGK.
5. Add the existing HCZ source formula using those derivatives and gravity.

The source retains `u=q7/(rho*c)`, `v=q8/(rho*c)` and force density
`Fy=-rho*gravity`. It adds to the existing residual; fields 0 and 7 receive
no source at `kappa=0`. The gradient kernel overwrites selected gradients
and preserves all inputs and unselected outputs. Neither the gradient nor
the source kernel performs communication internally. Prepared subcell halos
for the WENO stage remain a separate input requirement.

```sh
python3 test/testHczSubcellSource.py
python3 test/testHczSubcellSource.py --modes CUDA
python3 test/testHczMaskHalo.py --modes Serial OpenMP CUDA
```

The isolated suite passed 192 cases per backend on Serial, OpenMP and CUDA,
covering degrees 1–4 with walls and periodic boundaries. Prescribed constant,
polynomial and independent potential fields check both derivative rules,
including analytic one-sided wall derivatives. Tests also cover two EOS and
density parameter sets, nonunit sound speeds, zero and nonzero gravity,
zero hydrodynamic pressure, hydrostatic source normalization, selected/empty
element lists, repeated additions, and input/output preservation. Source
contributions are compared directly with `addSources(sub,rhs,true)`; the
complete WENO/BGK/source residual is compared against the host sequence and
checked for index conservation. Hydrostatic source normalization is checked
separately from the WENO residual; exact discrete hydrostatic balance of
the full WENO scheme is not asserted.

Maximum derivative differences were `2.9e-15`, source differences `4.6e-15`,
and complete residual differences `2.2e-14`. The extended OGS harness also
passed on Serial/OpenMP/CUDA with MPI 1/2/4 ranks, including empty ranks,
at degrees 1–3. It computes potentials on owned subcells, exchanges actual
device buffers, then compares all twelve source moments with the global
host reference. CUDA ranks shared the available GPU; multiple GPUs were
not tested.

This completes the isolated subcell spatial RHS at `kappa=0`. Production
OCCA/VERIFY execution still requires `HCZ SUBCELL WENO=FALSE` until the
hybrid recomputation path is assembled. The following increment adds subcell-to-DG
residual reconstruction with the host mean correction. Conservative mixed
DG/subcell face corrections and LSERK4 recomputation integration follow.

## Conservative subcell-to-DG residual reconstruction

`bnsHczReconstructQuad2D.okl` reconstructs all twelve moments with the
same operation as `project(fv,ns,n,R,true)` in the host hybrid operator.
Its signature is:

```text
Nelements, elementIds, R, weights, subRhs, dgRhs
```

`subRhs` uses `[element][12 moments][subcell]` storage and `dgRhs` uses
`[element][12 moments][DG node]`, both with x varying fastest. Build with
the original DG `p_Nq=N+1`; the kernel derives `ns=2N+1`. Supply the host
reconstruction matrix in column-major form `R[i+a*p_Nq]`, where `i` is
the DG coordinate and `a` is the subcell coordinate, plus the one-dimensional
GLL weights.

For each selected element and moment, the kernel first forms the tensor
reconstruction `R * subRhs * R^T`. It then computes the arithmetic mean
of the subcell residual and the GLL-weighted mean of the provisional DG
residual, using the host summation order:

```text
subMean = sum(subRhs)/(ns*ns)
dgMean  = sum(0.25*w[j]*w[i]*provisionalDg[j,i])
dgRhs[j,i] = provisionalDg[j,i] + subMean - dgMean
```

The implementation adds the precomputed difference `(subMean-dgMean)`
to each node, matching the host expression. This preserves each of the
twelve element means to roundoff. Residuals can be positive or negative;
no positivity scaling, clipping, timestep or extra geometry factor is
applied. Each selected output element is overwritten; all inputs,
unselected outputs and output halo slots remain untouched. A caller can
therefore eventually use a troubled-element list to replace only those
DG residuals. The kernel does not construct that list or apply mixed-face
corrections.

```sh
python3 test/testHczReconstruct.py
python3 test/testHczReconstruct.py --modes CUDA
python3 test/testHczSubcellSource.py --modes Serial OpenMP CUDA
```

The isolated reconstruction tests compare directly against the host
`project(...,true)` for degrees 1–4. They cover zero/constant/smooth/jump
residuals, amplitudes from `1e-8` to `1e3`, separate impulses in all twelve
moments, permuted/subset/empty element lists, repeated overwrites and
unchanged inputs/halos. The DG-to-subcell projection kernel feeds a
polynomial round-trip test. A deliberately perturbed test matrix makes
the mean correction appreciable, ensuring the test would detect an
omitted correction even though the actual pseudoinverse nearly preserves
the mean on its own. Production matrix construction is unchanged.

The existing subcell-source suite now also feeds the actual complete
device WENO/BGK/source residual into reconstruction, comparing the resulting
DG residual against the full host sequence and checking every element's
twelve means. This validates the local reconstruction stage with the
previously implemented physics; it does not yet assemble a mixed DG/subcell
time evolution.

Validation passed on Serial, OpenMP and CUDA: 92 standalone reconstruction
cases and 192 source/pipeline cases per backend, at degrees 1–4. The maximum
standalone reconstruction difference was `1.2e-12` in tests with residual
amplitudes up to `1e3`; mean errors were below `2.3e-13`. Polynomial
round-trip differences were below `3.9e-15`. The full physics-to-reconstructed
DG residual comparison differed by at most `1.2e-13` across the tested
backends and wall/periodic cases.

Production OCCA/VERIFY execution still requires `HCZ SUBCELL WENO=FALSE`.
The following increment adds conservative flux corrections at mixed DG/subcell
faces. After that comes LSERK4 recomputation integration, including global
decisions, mask updates, halo exchanges and CPU/GPU/MPI trajectory checks.
Capillarity remains separate; PML, other elements and 3D remain outside scope.

## Conservative mixed DG/subcell face corrections

`bnsHczMixedSurfaceQuad2D.okl` applies the face correction from the host
`hybrid()` operator to selected DG elements adjacent to marked subcell
elements. Its signature is:

```text
Nelements, elementIds, neighbors, mask, R, weights, c, hx, hy, q, faces, rhs
```

`q` and `rhs` use native `[element][12 moments][DG node]` storage. `mask`
is a binary integer array with owned and ghost slots; `neighbors` uses
the existing nine-entry HCZ Cartesian ordering. `faces` contains each
owned element's WENO fluxes in
`[element][axis][12 moments][subcell tangent][normal face]` order. The
matrix `R[i+s*p_Nq]` and one-dimensional GLL weights match the reconstruction
kernel. Build with the original DG `p_Nq=N+1`.

For an unmarked element with a marked neighbor, the kernel reconstructs
that element's WENO face flux onto the DG face nodes. It computes its
arithmetic subcell-face mean and reconstructed GLL face mean, then applies
the same additive correction as the host:

```text
newFlux[t] = reconstructedFlux[t] + subcellFaceMean - dgFaceMean
rhs[boundaryNode] -= 2/h * sign/w[boundaryIndex] * (newFlux[t]-oldDgFlux[t])
```

`oldDgFlux` is the existing two-distribution H2 upwind flux between the
original DG traces, oriented in the positive coordinate direction.
This makes the integrated DG contribution agree with the neighboring
subcell flux. The actual implementation retains the host arithmetic
ordering, including separate face means. `h` is `hx` or `hy`, and
`sign` is negative on the lower face and positive on the upper face.

Marked elements are left untouched because their residual has already
been replaced by subcell-to-DG reconstruction. DG/DG faces and physical
wall faces receive no mixed correction. A thread owns each DG node and
applies x-, x+, y-, y+ corrections in host order; corner contributions
therefore accumulate without atomics or overlapping writes. All input
arrays, unselected residuals, inactive/interior nodes and output halos
remain unchanged. Repeated calls add the correction again, as expected
for this additive operation.

The caller must exchange the current DG state and mask before use and
compute WENO faces for the owned DG neighbors as well as marked elements,
matching the host hybrid operator. This kernel neither exchanges halos
nor decides which elements are troubled.

```sh
python3 test/testHczMixedSurface.py
python3 test/testHczMixedSurface.py --modes CUDA
```

The isolated suite checks degrees 1–4 with walls and periodic boundaries,
two sound speeds/density maps and unequal element widths. Masks include
all-DG, all-subcell, single marked elements, checkerboards and stripes.
Tests exercise supplied DG/mask halos, corners with two corrections,
selected/empty element lists, repeated additions and input preservation.
Unused WENO faces are filled with NaNs to verify that inactive faces and
marked elements are skipped. A perturbed test reconstruction matrix makes
the face-mean correction observable; production matrix construction is
unchanged.

An assembled spatial test uses the host DG RHS as its baseline and runs
device projection, conservative scaling, WENO faces, subcell BGK/divergence,
thermodynamics/derivatives/sources, reconstruction on marked elements, and
the new correction on DG neighbors. It compares all twelve result moments
directly with `QuadOperator::hybrid` and checks the closed/periodic index
integral. This validates the assembled subcell path and its coupling to
a supplied DG baseline; it does not integrate a hybrid trajectory. Halos
are supplied by the test, so real MPI exchange of this assembled hybrid
path remains part of solver integration.

Validation passed on Serial, OpenMP and CUDA: 256 cases per backend at
degrees 1–4. Maximum mixed-correction differences were `1.8e-14`; the
assembled hybrid spatial residual differed from the host by at most
`4.3e-14`. Closed/periodic index-balance errors were below `3.5e-17`,
including the perturbed-matrix conservation check.

Production OCCA/VERIFY execution still requires `HCZ SUBCELL WENO=FALSE`.
The remaining main task is solver integration: allocate/reuse the subcell
buffers, connect hybrid RHS evaluation and halo exchanges, implement the
LSERK4 detector/recomputation control flow and global failure decisions,
and validate CPU/GPU/MPI trajectories and conservation. No additional
spatial physics kernels are planned for the agreed `kappa=0` scope.
Capillarity remains separate; PML, other elements and 3D remain outside scope.

## Hybrid spatial integration with host stage control

The OCCA/VERIFY solver now accepts `HCZ SUBCELL WENO=TRUE` with `HCZ KAPPA=0`.
`bnsHcz_t` allocates the subcell buffers and projection/reconstruction matrices
once during setup and builds the previously validated spatial kernels. It
reuses the existing HCZ element topology and OGS halo object for integer
masks, twelve-moment subcell states and two-component force potentials.

Each LSERK4 stage first evaluates the device DG RHS and saves an unchanged
copy. The existing host detector, buffer expansion and recompute loop then
request a hybrid RHS when needed. Both HOST and OCCA call the same
`QuadOperator::stageWithRhs` control implementation, supplied with their
respective spatial callbacks. No numerical detector thresholds, acceptance
rules or recompute limits change.

Each device hybrid evaluation starts from the saved DG RHS, projects and
conservatively scales subcell states, and collectively rejects invalid
index means. After exchanging masks/subcells, it evaluates WENO faces,
BGK/divergence, force potentials and their halo-dependent derivatives, then
adds sources. It reconstructs the residual on marked elements and applies
conservative mixed-face corrections on adjacent DG elements. All owned
elements prepare subcell fluxes, including unmarked neighbors, as in the
reference operator. Repeated passes therefore replace the previous hybrid
result without accumulating corrections from an earlier mask.

`VERIFY` compares every hybrid RHS against `QuadOperator::hybrid` using the
same stage state, original DG baseline and current mask, across all twelve
moments and MPI ranks. The existing DG comparison separately validates
that baseline. The final verification line adds `hybrid_checks`,
`max_hybrid_error` and `max_scaled_hybrid_error`. The accepted RHS remains
on the device for the existing LSERK4 update; its final prospective
admissibility check also uses the existing device kernel.

This is a spatial integration milestone. State, residual and candidate RHS
arrays still travel to the host for subcell stage decisions, and masks and
marked-element lists return to the device. The next increment is connecting
the existing device detector and buffer kernels to that stage control and
removing these transfers. Initialization, output and optional VERIFY
comparisons continue to require host data.

```sh
make -C solvers/bns -j4
python3 test/testHczHybridIntegration.py --cuda
python3 test/testHczDgIntegration.py --cuda
python3 test/testHczIntegration.py
```

The hybrid suite integrates short RT/wall and periodic-slab trajectories,
checks all-moment hybrid RHS agreement in VERIFY, and compares final fields
and stage/recomputation counts with independent HOST trajectories. It covers
Serial on 1/2/4 MPI ranks, OpenMP on two ranks, degrees 1–3, zero/one buffer
layers, and uniform equilibrium with no marked elements. `--cuda` adds active
wall/periodic hybrid runs and the no-marked case on one/two GPU-sharing ranks,
including OCCA mode without reference comparisons. It checks index/mass
conservation and retains the nonzero-kappa guard. These are short integration
checks, not a full-time RT benchmark or a performance claim.

Validation passed on Serial, OpenMP and CUDA. The largest hybrid RHS
difference was `5.3e-14`; final fields differed from HOST by at most
`1.4e-15`. Candidate/recomputation counts and final troubled-element counts
agreed, with index and mass errors below the suite's `2e-11` tolerance.
The existing pure-DG CPU/CUDA and HOST/legacy BNS regression suites also
passed.

## Device detector integration: VERIFY shadow checks

With WENO enabled, `VERIFY` now runs the existing device bounds/detector
kernels alongside the host detector during actual LSERK4 stages. HOST and
ordinary OCCA retain their previous stage decisions and do not build or
launch these verification kernels. The device mask is not yet used for
buffering, mask merging or acceptance.

The stage input is unchanged across recompute passes, so its nodal index
bounds are computed once and exchanged with OGS as two values per element,
including diagonal neighbors. Each detector call forms the prospective
candidate from the device state, residual and current RHS. It ignores the
residual when the RK coefficient is zero, as in the host implementation.
The comparison uses the raw owned-element mask before buffer expansion or
clearing already marked elements. It runs on the initial DG candidate and
again after every hybrid recomputation, using the same cached reference
bounds throughout that stage.

The shared host stage has an optional read-only mask verification hook.
The hook compares every owned element and reduces a mismatch flag across
MPI ranks; any mismatch raises `HCZ device detector mask differs from host
reference`. It does not change the host mask. The final verification line
adds `detector_checks`; for completed hybrid runs this must equal
`rhs_checks + hybrid_checks`. With subcells disabled it must be zero.

Validation uses `test/testHczHybridIntegration.py --cuda`, which asserts
that relationship and covers Serial/OpenMP/CUDA, MPI boundaries, degrees
1–3, wall and periodic conditions, buffered/unbuffered masks, recomputed
candidates and uniform equilibrium. The pure-DG integration suite checks
that no detector calls occur when WENO is disabled. Geometry remains
uniform, axis-aligned Cartesian BOX quads; this increment does not extend
the HCZ operators to general distorted or curved quadrilaterals.

The next increment is device mask buffering and its MPI exchange, with
verification before switching acceptance to device-produced masks. Host
recompute bookkeeping and removal of routine full-array transfers follow.

Validation passed on Serial, OpenMP and CUDA with exact agreement for all
raw detector masks. Final fields and recomputation counts retained their
agreement with HOST. The pure-DG CPU/CUDA and HOST/legacy BNS regression
suites also passed.

## Device mask buffering, recomputation control and transfer removal

Increments 2–4 connect the existing device detector and buffer kernels to
LSERK4 acceptance. The device path follows `QuadOperator::stage`:

1. Save the original DG RHS and compute/exchange reference index bounds.
2. Detect the prospective DG candidate, then buffer its mask with one OGS
   integer halo exchange per layer. A zero-layer buffer is a no-op.
3. If any rank has marked elements, evaluate the hybrid RHS from the saved
   DG baseline. Reconstruct only marked outputs and conservatively correct
   the adjacent DG faces.
4. Detect again, clear already marked elements from the new mask and reduce
   the remaining count. If further passes are available, buffer the new
   mask and merge it into the current mask. Repeat from the original DG RHS.
5. Preserve the original pass-limit error and final device admissibility
   check. The existing LSERK4 updater commits only an accepted RHS.

`bnsHczMaskControlQuad2D.okl` provides owned-element mask difference, union
and scalar count kernels. The count reduction handles empty input and
counts larger than a single thread block. `hczExpandDeviceMask` preserves
the original layer-by-layer buffering behavior. Current and next masks
have separate storage; their halo slots are exchanged before use.

`bnsHczReconstructQuad2D` has an optional `p_HCZ_MASKED_RECONSTRUCT=1`
specialization. It computes the original reconstruction and writes only
marked elements, allowing the solver to reuse the owned-element list
without host compaction or uploads. The original unmasked kernel interface
remains available for selected-list calls. This increment prioritizes the
validated algorithm; skipping reconstruction arithmetic on unmarked
blocks can be considered during performance work.

Ordinary OCCA stages perform no host state/residual/RHS conversions or
mask/list uploads. The CPU retains the small retry loop, kernel launches,
scalar failure/count decisions and MPI orchestration. Verification copies
are conditional on VERIFY. The final `HCZ device stages` line reports
full-array downloads/uploads through the stage conversion helpers, summed
across ranks; both counts must be zero in ordinary OCCA mode. This excludes
initialization, output and device-to-device copies.

VERIFY compares each raw, buffered, newly troubled and merged mask against
host calculations, in addition to every spatial RHS and admissibility
decision. It reports `buffer_checks` and `mask_update_checks` alongside the
existing counters. Diagnostics add `max_recompute_passes`, the maximum
passes used by a stage (including temporary output-sampling steps), to
make multiple-pass coverage explicit.

```sh
make -C solvers/bns -j4
python3 test/testHczReconstruct.py --modes Serial OpenMP CUDA
python3 test/testHczHybridIntegration.py --cuda --extended
python3 test/testHczDgIntegration.py --cuda
python3 test/testHczIntegration.py
```

The reconstruction suite now tests masked output preservation and mask
control kernels at 0, 1, 255, 256, 257 and 1031 elements, with poisoned halo
slots. The hybrid suite checks multiple retries in one stage and deliberate
pass-limit failures on CPU/GPU/MPI, exact mask comparisons, conserved
index/mass, final-field and counter agreement, and zero ordinary OCCA stage
transfers. Extended cases add 320 owned elements on one rank, buffer widths
2/3 and RT evolution to t=0.1. These checks do not constitute the full
final-time RT benchmark or a performance measurement.

Validation passed on Serial/OpenMP/CUDA, with CPU MPI 1/2/4 ranks and GPU
MPI 1/2 ranks. All 8,985 raw-mask comparisons in the extended suite matched,
as did buffered and updated masks. The suite exercised four passes in one
stage and verified rejection when only one pass was allowed. The t=0.1
runs completed 3,206 hybrid recomputations with identical counters across
HOST/VERIFY/OCCA; the largest final-field difference was `4.0e-15`.
Across the suite the maximum hybrid RHS difference was `1.4e-13`, and
index/mass errors stayed below `2e-11`. Ordinary OCCA stage full-array
transfer counts were zero across all ranks. Pure-DG CPU/GPU and HOST/legacy
BNS regressions also passed. Full final-time RT validation and performance
profiling remain separate work.

## First SARK4 increment: pure DG with relaxation splitting

The first increment added `TIME INTEGRATOR=SARK4` on HOST/OCCA/VERIFY, with
`HCZ SUBCELL WENO=FALSE` and `HCZ KAPPA=0`. This uses libp's existing
five-stage fourth-order semi-analytic RK method with an embedded third-order
error estimate. The original LSERK4 and DG/WENO paths remain available.
The next increment, described below, extends this to subcells with full-trial replay.

The constant diagonal linear part has rates:

| Moments | Lambda |
| --- | --- |
| 0 (index), 6 (pressure), 7–8 (hydrodynamic momentum) | 0 |
| 1–5 (relaxing index moments) | -1/tauPhi |
| 9–11 (relaxing hydrodynamic moments) | -1/tauG |

The stepper evolves `dq/dt = Lambda*q + F(q)`. We retain the complete
existing DG residual, then form `F(q) = DG(q) - Lambda*q` by adding the
appropriate nodal damping rates back to the relaxing fields. The new
`bnsHczSarkSplitQuad2D` kernel performs that operation on the device.
This distinction matters: cubature projection followed by interpolation
is not the identity for the highest GLL modes. Dropping the cubature damping
would change the spatially discretized equations. The split retains the
projection correction in F, so treating Lambda analytically does not
eliminate every possible relaxation-related accuracy/stability restriction.

The initial step and maximum adaptive step use the DG transport bound:

```text
dt_max = CFL NUMBER / (sqrt(3)*c*(N+1)^2*(1/hx + 1/hy))
```

SARK4 does not apply `HCZ COLLISION CFL` to its timestep. Accuracy control
can reduce the step below the transport cap. `HCZ SARK4 ATOL` (default
`1e-8`) and `HCZ SARK4 RTOL` (default `1e-6`) configure the existing weighted
RMS error estimate over all twelve moments. These tolerances are not
per-field rescalings and should be checked for the chosen physical scales.

Optional SARK4 solver hooks provide the cap, tolerances and a collective
admissibility decision. Other solvers retain default controls. HCZ checks
all nodal and cubature values before each RHS and before accepting the
final trial. A nonphysical trial halves dt and retries without committing
state, including a failed output-sampling step. Error-estimate rejection
uses the existing controller. Exponential coefficients are recomputed after
every dt change, including final-time clipping and output sampling.
The final `SARK4:` line reports accepted steps, error/stage rejections,
and the minimum/maximum accepted dt (excluding output mini-steps).

VERIFY checks both the complete DG RHS and split F against HOST, plus
admissibility decisions. It adds `split_checks`, `max_split_error` and
`max_scaled_split_error` to the verification summary. Ordinary OCCA runs
keep solution arrays on the device and transfer only scalar checks for
stage control; initialization, reporting and VERIFY still use host data.

A short smooth RT setup is supplied, using tighter error tolerances:

```sh
make -C solvers/bns -j4
cd solvers/bns
./bnsMain setups/setupHczSark4Quad2D.rc
```

The example defaults to Serial/VERIFY. Set `THREAD MODEL=CUDA` for GPU,
and use OCCA instead of VERIFY to disable reference comparisons. It is a
pure-DG example, not the sharp-interface WENO RT setup or a validated
replacement for the full RT benchmark.

```sh
python3 test/testHczSark4.py --modes Serial OpenMP CUDA --ranks 1 2
python3 test/testHczSark4Integration.py --cuda
python3 test/testHczDgIntegration.py --cuda
python3 test/testHczHybridIntegration.py --cuda
python3 test/testHczIntegration.py
```

The relaxation suite uses the actual SARK4 stepper and split kernel. It
checks homogeneous exponential decay with unequal relaxation times,
highest-mode projected collision (including its nonidentity projection),
fourth-order temporal convergence, adaptive error rejection, CFL capping,
nonaligned final/output times and collective rollback when an internal or
output-sampling trial is rejected on one rank. Flow tests cover smooth RT,
periodic slabs, hydrostatic/uniform equilibrium, degrees 1–3, CPU MPI 1/2/4
and GPU MPI 1/2. They compare independent HOST/OCCA trajectories and a
finely stepped LSERK4 reference and check conservation, timestep caps,
zero ordinary OCCA stage transfers and unsupported combinations.

Validation passed on Serial, OpenMP and CUDA. Projected-relaxation error
ratios approached 16 under timestep halving, confirming fourth-order
convergence. HOST/device final fields differed by at most `4.5e-16` in the
SARK4 flow suite; comparisons with the fine LSERK4 reference differed by
less than `6e-11`. Conservation errors stayed below `2e-11`; timestep caps,
error/physical rejection and zero ordinary OCCA stage transfers passed.
Existing LSERK4 DG/subcell and HOST/legacy BNS regressions, including the
legacy SARK4 path, passed. These results validate the first pure-DG
increment; they do not measure a speedup or validate SARK4 with WENO.

## Second SARK4 increment: DG/WENO with full-trial replay

`SARK4` now supports `HCZ SUBCELL WENO=TRUE` on HOST/OCCA/VERIFY, for
the same Cartesian 2D quads and `HCZ KAPPA=0` scope. It reuses the existing
DG, WENO3-Z, reconstruction and conservative mixed-face kernels. The
semi-analytic split is applied to the complete hybrid residual:
`F(q) = hybrid(q, mask) - Lambda*q`. Cubature-projection corrections remain
in F, just as in the pure-DG increment.

Each timestep attempt starts with an empty mask and keeps its input `q_n`
unchanged. Intermediate candidates use the existing physical and DMP
detector against neighborhood bounds from `q_n`. Newly troubled elements
are buffered and merged into the mask. Every mask change restarts the
entire SARK4 tableau at stage zero. All residuals and the embedded error
estimate are rebuilt using the enlarged mask; residuals computed under
different masks are never combined in an accepted trial. Elements already
in the mask do not trigger another replay, but every candidate still must
pass the hard admissibility checks.

`HCZ RECOMPUTE PASSES` limits mask-growth replays per timestep attempt.
Exhausting that budget, encountering a nonphysical candidate, or failing
conservative subcell index-mean repair rejects the attempt and halves dt.
The solution is committed only after a complete admissible trial passes
the error estimate. Output mini-steps use the same replay/rejection policy.
Decisions are collective across MPI ranks. A smaller dt cannot necessarily
compensate for too few mask-growth passes: persistent budget failures need
a larger `HCZ RECOMPUTE PASSES` value. The solver prints an explanatory
message on the first such failure and stops if dt reaches its minimum.

The adaptive maximum is the smaller of the DG transport bound above and
the existing subcell bound:

```text
dt_subcell = HCZ SUBCELL CFL / (HCZ WENO ALPHA*c*(2*N+1)*(1/hx + 1/hy))
dt_max = min(dt_DG, dt_subcell)
```

`HCZ COLLISION CFL` remains unused by SARK4. Removing that explicit cap
does not establish a speedup: accuracy control and repeated hybrid trials
can still limit performance. This whole-trial detector policy differs
from the stage-local LSERK4/Python policy, so their trajectories need not
match exactly.

The `SARK4:` summary now uses `stage_rejections` for all non-error-estimate
trial failures. `HCZ SARK4 hybrid:` reports attempts, replays, budget
rejections and subcell-mean rejections. For SARK4, the diagnostics CSV's
`recomputations` counts full-trial replays and `max_recompute_passes` records
the largest number of replays in one attempt; these include output and
rejected attempts. They do not count individual hybrid RHS calls.

OCCA keeps the reference state, candidates, masks and stage residuals on
the configured device. Only scalar counts/flags are transferred for stage
control. VERIFY also checks each raw/buffered/merged mask, complete hybrid
RHS and split RHS against HOST.

A short sharp-interface RT example is supplied:

```sh
make -C solvers/bns -j4
cd solvers/bns
./bnsMain setups/setupHczSark4WenoQuad2D.rc
```

It defaults to Serial/VERIFY. Set `THREAD MODEL=CUDA` to run on the GPU;
select `HCZ DG BACKEND=OCCA` for execution without host reference checks.
For CPU MPI, use `THREAD MODEL=OpenMP` or `Serial`, for example:

```sh
mpirun -n 4 ./bnsMain setups/setupHczSark4WenoQuad2D.rc
```

Validation commands, from the repository root:

```sh
python3 test/testHczSark4.py --modes Serial OpenMP CUDA --ranks 1 2
python3 test/testHczDetector.py --modes Serial OpenMP CUDA
python3 test/testHczReconstruct.py --modes Serial OpenMP CUDA
python3 test/testHczSark4WenoIntegration.py --cuda
python3 test/testHczSark4Integration.py --cuda
python3 test/testHczDgIntegration.py --cuda
python3 test/testHczHybridIntegration.py --cuda
python3 test/testHczIntegration.py
```

The stepper test changes the spatial operator during a trial and checks
the replayed solution against its analytic solution, detecting stale
residual history. It also injects collective RHS failures during normal
and output-sampling trials to check rollback. Detector tests cover both
LSERK4 increments and already assembled SARK4 candidates. Mask-control
tests check owned-entry reset without modifying halos.

The flow suite compares independent HOST and device trajectories, replay
counts, conservation and timestep caps. It covers wall RT, periodic slabs,
uniform/hydrostatic equilibrium, buffering with zero/one/two layers,
deliberately insufficient replay budgets, a restrictive subcell timestep
cap, nonaligned output/final times, and degrees 1–4. CPU MPI uses up to four
ranks and CUDA uses one/two ranks. The degree-four case uses an 8 × 32 mesh
and CFL 0.2. These are short integration tests; a full-time RT validation
and matched-accuracy LSERK4/SARK4 performance comparison remain to be done.

All eight validation commands above passed on the available CPU/GPU.
The largest HOST/device final-field difference in the SARK4/WENO suite was
`2.23e-15`; mass and index conservation errors stayed below `2e-11`.
Replay/candidate counts agreed across backends and MPI partitions. The
8 × 32 degree-four case reached `t=0.0047` in 17 accepted steps and 21
replays, with at most two replays per attempt. Zero ordinary OCCA stage
array transfers, rollback, insufficient-budget diagnostics and both
timestep caps passed. Existing pure-DG SARK4, LSERK4 DG/WENO and HOST/legacy
BNS regressions also passed.
