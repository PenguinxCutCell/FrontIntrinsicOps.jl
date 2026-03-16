# FrontIntrinsicOps.jl — Documentation

**FrontIntrinsicOps.jl** is a static-surface PDE toolkit for triangulated front meshes.
It implements intrinsic geometry, discrete exterior calculus (DEC), and a full suite of
PDE solvers — all working directly on the surface without embedding into a bulk domain.

---

## Contents

### Mathematical Background

Detailed derivations and formulas for every module:

| Document | Topic |
|----------|-------|
| [Mesh types and data structures](math/01_mesh_types.md) | `CurveMesh`, `SurfaceMesh`, and the primal–dual picture |
| [Topology and incidence matrices](math/02_topology.md) | Edges, faces, orientation, $d_0$, $d_1$ |
| [Discrete geometry and dual areas](math/03_geometry.md) | Normals, areas, edge lengths, barycentric and mixed duals |
| [Discrete exterior calculus](math/04_dec.md) | Hodge stars, exterior derivatives, cochains |
| [Laplace–Beltrami operator](math/05_laplace_beltrami.md) | DEC and cotan formulations, sign convention |
| [Curvature](math/06_curvature.md) | Signed (curve), mean, Gaussian, Gauss–Bonnet |
| [Surface diffusion, Poisson, Helmholtz](math/07_surface_diffusion.md) | Implicit time integration, gauge treatment |
| [Scalar transport](math/08_transport.md) | Conservative DEC fluxes, SSP-RK2/3, CFL |
| [Advection–diffusion IMEX](math/09_advection_diffusion.md) | Splitting, stability, factory reuse |
| [Reaction–diffusion IMEX](math/10_reaction_diffusion.md) | θ-scheme, Fisher–KPP, bistable |
| [Tangential vector calculus](math/11_vector_calculus.md) | Projection, surface gradient/divergence, Whitney forms |
| [Hodge decomposition](math/12_hodge_decomposition.md) | Helmholtz–Hodge, harmonic forms, genus-$g$ |
| [High-resolution transport](math/13_highres_transport.md) | Minmod, van Leer, superbee, SSP-RK2 |
| [Open surfaces and boundary conditions](math/14_open_surfaces.md) | Boundary detection, Dirichlet, Neumann |
| [Caching and performance](math/15_caching.md) | `SurfacePDECache`, in-place buffers, zero-allocation kernels |
| [Mesh generators](math/16_generators.md) | UV-sphere, icosphere, torus, ellipsoid, perturbed sphere |

### API Reference

Concise function signatures and descriptions:

| Document | Module |
|----------|--------|
| [Types](api/types.md) | `CurveMesh`, `SurfaceMesh`, `SurfaceGeometry`, … |
| [Generators](api/generators.md) | `generate_icosphere`, `generate_torus`, `generate_ellipsoid`, … |
| [Geometry and DEC](api/geometry.md) | `compute_geometry`, `build_dec`, Hodge stars |
| [PDE solvers](api/pdes.md) | Diffusion, transport, reaction–diffusion, open surfaces |
| [Diagnostics](api/diagnostics.md) | `check_mesh`, `check_dec`, Gauss–Bonnet, `star1_sign_report` |

### Tutorials

Step-by-step worked examples:

| Document | Topic |
|----------|-------|
| [Getting started](tutorials/01_getting_started.md) | Sphere geometry in five lines |
| [Surface diffusion](tutorials/02_surface_diffusion.md) | Heat equation on the sphere |
| [Scalar transport](tutorials/03_transport.md) | Rotating a patch with SSP-RK3 |
| [Reaction–diffusion](tutorials/04_reaction_diffusion.md) | Fisher–KPP wave on the sphere |
| [Hodge decomposition](tutorials/05_hodge_decomposition.md) | Decomposing a 1-form on sphere and torus |
| [Open surfaces](tutorials/06_open_surfaces.md) | Poisson with Dirichlet BC on a cap |

---

## Quick reference

```julia
using FrontIntrinsicOps

# ── Build mesh ──────────────────────────────────────────────────────────────
mesh = generate_icosphere(1.0, 3)          # level-3 icosphere (~640 verts)
geom = compute_geometry(mesh)              # intrinsic geometry
dec  = build_dec(mesh, geom)               # DEC operators

# ── Geometry diagnostics ────────────────────────────────────────────────────
println("Area   = ", measure(mesh, geom))
println("χ      = ", euler_characteristic(mesh))
println("GB res = ", gauss_bonnet_residual(mesh, geom))

# ── Surface diffusion ───────────────────────────────────────────────────────
u, fac = step_surface_diffusion_backward_euler(mesh, geom, dec, u0, dt, μ)

# ── Reaction–diffusion (Fisher–KPP) ─────────────────────────────────────────
cache = build_pde_cache(mesh, geom, dec; μ=0.1, dt=1e-3, θ=1.0)
reaction = fisher_kpp_reaction(2.0)
for _ in 1:200; step_diffusion_cached!(u, cache, reaction, t); end

# ── Hodge decomposition ──────────────────────────────────────────────────────
α = dec.d0 * u
result = hodge_decompose_1form(mesh, geom, dec, α)
# result.exact  result.coexact  result.harmonic

# ── High-resolution transport ────────────────────────────────────────────────
T_new = step_surface_transport_limited(mesh, geom, dec, topo, T, vel, dt;
                                       limiter=:van_leer, method=:ssprk2)
```

---

## Design principles

1. **Discrete exterior calculus first** — all operators derive from the primal cochain
   complex $\Omega^0 \xrightarrow{d_0} \Omega^1 \xrightarrow{d_1} \Omega^2$.
2. **No bulk-solver coupling** — purely intrinsic; coupling lives in a separate package.
3. **Sparse matrices throughout** — every global operator is a `SparseMatrixCSC`.
4. **Reuse factorizations** — linear solves are the bottleneck; the cache layer
   amortizes factorization cost across many time steps.
5. **Float64 default** — parameterised on `T<:AbstractFloat`; `Float32` meshes work.

---

## License

MIT
