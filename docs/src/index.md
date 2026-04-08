# FrontIntrinsicOps.jl — Documentation

**FrontIntrinsicOps.jl** is a static-surface PDE toolkit for triangulated front meshes.
It implements intrinsic geometry, discrete exterior calculus (DEC), and a full suite of
PDE solvers — all working directly on the surface without embedding into a bulk domain.

Release summary for ambient signed distance is available in `CHANGELOG.md` (v0.5 section).

---

## Contents

### Mathematical Background

Detailed derivations and formulas for every module:

| Document | Topic |
|----------|-------|
| [Mesh types and data structures](01_mesh_types.md) | `CurveMesh`, `SurfaceMesh`, and the primal–dual picture |
| [Topology and incidence matrices](02_topology.md) | Edges, faces, orientation, $d_0$, $d_1$ |
| [Discrete geometry and dual areas](03_geometry.md) | Normals, areas, edge lengths, barycentric and mixed duals |
| [Discrete exterior calculus](04_dec.md) | Hodge stars, exterior derivatives, cochains |
| [Laplace–Beltrami operator](05_laplace_beltrami.md) | DEC and cotan formulations, sign convention |
| [Curvature](06_curvature.md) | Signed (curve), mean, Gaussian, Gauss–Bonnet |
| [Surface diffusion, Poisson, Helmholtz](07_surface_diffusion.md) | Implicit time integration, gauge treatment |
| [Scalar transport](08_transport.md) | Conservative DEC fluxes, SSP-RK2/3, CFL |
| [Advection–diffusion IMEX](09_advection_diffusion.md) | Splitting, stability, factory reuse |
| [Reaction–diffusion IMEX](10_reaction_diffusion.md) | θ-scheme, Fisher–KPP, bistable |
| [Tangential vector calculus](11_vector_calculus.md) | Projection, surface gradient/divergence, Whitney forms |
| [Hodge decomposition](12_hodge_decomposition.md) | Helmholtz–Hodge, harmonic forms, genus-$g$ |
| [Topology-aware DEC](topology_aware_dec.md) | Betti numbers, cycle basis, harmonic/cohomology operators |
| [Geodesics](geodesics.md) | Heat-method distance, shortest paths, intrinsic balls, geodesic FPS |
| [Parallel transport](parallel_transport.md) | Face/vertex tangent frames, connection angle, holonomy |
| [Exterior algebra extensions](exterior_algebra_extensions.md) | Wedge, interior product, Lie derivative (Cartan) |
| [High-resolution transport](13_highres_transport.md) | Minmod, van Leer, superbee, SSP-RK2 |
| [Open surfaces and boundary conditions](14_open_surfaces.md) | Boundary detection, Dirichlet, Neumann |
| [Caching and performance](15_caching.md) | `SurfacePDECache`, in-place buffers, zero-allocation kernels |
| [Mesh generators](16_generators.md) | UV-sphere, icosphere, torus, ellipsoid, perturbed sphere |
| [Ambient signed distance](17_signed_distance.md) | Exact point-to-front distance, pseudonormal vs winding sign |

### API Reference

Concise function signatures and descriptions:

| Document | Module |
|----------|--------|
| [Types](types.md) | `CurveMesh`, `SurfaceMesh`, `SurfaceGeometry`, … |
| [Generators](generators.md) | `generate_icosphere`, `generate_torus`, `generate_ellipsoid`, … |
| [Geometry and DEC](geometry.md) | `compute_geometry`, `build_dec`, Hodge stars |
| [PDE solvers](pdes.md) | Diffusion, transport, reaction–diffusion, open surfaces |
| [Diagnostics](diagnostics.md) | `check_mesh`, `check_dec`, Gauss–Bonnet, `star1_sign_report` |
| [Plotting with Makie](plotting.md) | Optional weak-extension plotting for `CurveMesh` / `SurfaceMesh` |

### Tutorials

Step-by-step worked examples:

| Document | Topic |
|----------|-------|
| [Getting started](01_getting_started.md) | Sphere geometry in five lines |
| [Surface diffusion](02_surface_diffusion.md) | Heat equation on the sphere |
| [Scalar transport](03_transport.md) | Rotating a patch with SSP-RK3 |
| [Reaction–diffusion](04_reaction_diffusion.md) | Fisher–KPP wave on the sphere |
| [Hodge decomposition](05_hodge_decomposition.md) | Decomposing a 1-form on sphere and torus |
| [Open surfaces](06_open_surfaces.md) | Poisson with Dirichlet BC on a cap |

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
