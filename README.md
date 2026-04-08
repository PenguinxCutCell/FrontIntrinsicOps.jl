# FrontIntrinsicOps.jl

[![In development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://PenguinxCutCell.github.io/FrontIntrinsicOps.jl/dev)
![CI](https://github.com/PenguinxCutCell/FrontIntrinsicOps.jl/actions/workflows/ci.yml/badge.svg)
![Coverage](https://codecov.io/gh/PenguinxCutCell/FrontIntrinsicOps.jl/branch/main/graph/badge.svg)


A **static interface-geometry / DDG / DEC package** for triangulated front meshes.

`FrontIntrinsicOps.jl` computes intrinsic geometric quantities and intrinsic
discrete exterior calculus (DEC) operators on front meshes.  The primary inputs
are triangulated 3-D surfaces and 2-D polygonal curves (open or closed,
depending on feature).

This package is an **intrinsic operator layer** for interface meshes.  It does
**not** implement front advection, remeshing, or coupling to bulk solvers.
Those belong in separate packages.

---

## Optional Makie support

Plotting is provided via a weak `Makie` extension (`ext/MakieExt.jl`).
Load any Makie backend (for example `CairoMakie`) to enable plotting helpers.

```julia
using CairoMakie
using FrontIntrinsicOps

set_makie_theme!()
mesh = sample_circle(1.0, 128)
fig, ax, p = plot_front(mesh; show_vertices=true)
save("curve.png", fig)
```

---

## Scope

### What is implemented (current)

| Feature | Status |
|---------|--------|
| `CurveMesh` / `SurfaceMesh` mesh types | ✓ |
| STL loading via MeshIO/GeometryBasics | ✓ |
| CSV / point-list curve loading | ✓ |
| **Mesh generators** (`sample_circle`, `generate_uvsphere`, `generate_icosphere`, `generate_torus`) | ✓ v0.2 |
| Topology (edges, adjacency, orientation, manifold) | ✓ |
| Incidence matrices d₀, d₁ | ✓ |
| Geometry: normals, areas, edge lengths | ✓ |
| **Barycentric dual areas** (default) | ✓ |
| **Mixed/Voronoi dual areas** (Meyer et al. 2003) | ✓ v0.2 |
| Hodge stars ⋆₀, ⋆₁, ⋆₂ | ✓ |
| **DEC Laplace–Beltrami** (`method=:dec`) | ✓ |
| **Direct cotan Laplace–Beltrami** (`method=:cotan`) | ✓ v0.2 |
| Signed curvature (curves) | ✓ |
| Mean curvature (surfaces, via L applied to embedding) | ✓ |
| Gaussian curvature (angle-defect formula) | ✓ |
| Length, area, enclosed area/volume | ✓ |
| **Integrated Gaussian curvature** | ✓ v0.2 |
| Field integration over front | ✓ |
| **Euler characteristic** `χ = V − E + F` | ✓ v0.2 |
| **Gauss–Bonnet diagnostic** `|∫K dA − 2πχ|` | ✓ v0.2 |
| **star1 sign report** | ✓ v0.2 |
| **Laplace method comparison** | ✓ v0.2 |
| **Scalar surface diffusion** (BE, CN) | ✓ v0.3 |
| **Scalar surface transport** (FE, SSP-RK2/3, upwind/centered) | ✓ v0.3 |
| **Surface advection–diffusion IMEX** | ✓ v0.3 |
| **Convergence scripts** | ✓ v0.3 |
| **Ambient exact signed-distance queries (curve/surface)** | ✓ v0.5 |

#### Ambient signed-distance queries

| Capability | Status |
|------------|--------|
| `CurveMesh` (closed) | ✓ |
| `CurveMesh` (open) | ✓ |
| `SurfaceMesh` (closed, oriented) | ✓ |
| `SurfaceMesh` (open, oriented) | ✓ |
| `sign_mode=:auto` | ✓ |
| `sign_mode=:pseudonormal` | ✓ |
| `sign_mode=:winding` (closed meshes only) | ✓ |
| `sign_mode=:unsigned` | ✓ |

### 1D point fronts

`FrontIntrinsicOps.jl` also provides a lightweight 1-D front primitive for
downstream front-tracking / Stefan coupling:

- one marker `xΓ` (half-line split),
- or two markers `(xL, xR)` defining a bounded interval,
- with explicit inside/outside semantics and `signed_distance` convention
  `φ < 0` inside, `φ > 0` outside.

This 1-D support is intentionally minimal:

- no 1-D DEC operators,
- no 1-D PDE operators,
- no general 1-D graph/topology framework.

### What is deliberately not implemented

- Front advection / marker motion
- Remeshing / topology changes
- Coupling to Eulerian bulk solvers
- Space-time operators
- Embedded-boundary / cut-cell logic
- Generic symbolic exterior algebra
- CSV/JSON/Markdown output from convergence scripts

---

## Topology-aware DEC

For closed orientable surfaces, `FrontIntrinsicOps.jl` now exposes explicit topology-aware 1-form operators:

- `betti_numbers`, `first_betti_number`
- `cycle_basis`, `cohomology_basis_1`
- `harmonic_basis`, `project_exact`, `project_coexact`, `project_harmonic`
- `hodge_decomposition_full` with potentials and residual diagnostics

The harmonic basis lives in edge-cochain space and is orthonormalized in the Hodge `⋆1` inner product.

---

## Geodesics and intrinsic transport

Intrinsic distance tools are provided via a heat-method pipeline:

- `geodesic_distance`, `geodesic_distance_to_vertex`, `geodesic_distance_to_vertices`
- `shortest_path_vertices`, `geodesic_gradient`
- `intrinsic_ball`, `farthest_point_sampling_geodesic`

Discrete connection/transport tools are also available:

- `face_tangent_frames`, `vertex_tangent_frames`
- `connection_angle_across_edge`, `transport_matrix_across_edge`
- `parallel_transport_face_vector`, `parallel_transport_along_face_path`
- `parallel_transport_vertex_vector`, `holonomy_along_cycle`

---

## Exterior algebra additions

Low-order practical DEC form products/derivatives are included:

- `wedge`, `wedge0k`, `wedge11`
- `interior_product`
- `lie_derivative`, `cartan_lie_derivative`

Supported degree combinations are documented in `docs/src/exterior_algebra_extensions.md`.

---

## FEEC layer (lowest order)

`FrontIntrinsicOps.jl` now also provides an additive lowest-order FEEC-compatible layer:

- Whitney space descriptors: `Whitney0Space`, `Whitney1Space`, `Whitney2Space`
- explicit complex builder: `build_whitney_complex`, `build_de_rham_sequence`
- local basis/reconstruction: `whitney*_basis_local`, `reconstruct_*form*`
- canonical interpolators: `interpolate_0form`, `interpolate_1form`, `interpolate_2form`
  (aliases `Π0`, `Π1`, `Π2`)
- commuting checks: `projection_commutator_01`, `projection_commutator_12`
- consistent FEEC assembly: `assemble_whitney_mass*`, `assemble_whitney_stiffness0`,
  `assemble_whitney_hodge_laplacian*`

This FEEC layer does **not** replace the existing DEC workflows; it complements
them for reconstruction and variational experiments.

---

## Quick start

### Generate and analyse a sphere

```julia
using FrontIntrinsicOps

R    = 1.0
mesh = generate_icosphere(R, 3)         # level-3 icosphere (~640 vertices)
geom = compute_geometry(mesh)           # barycentric duals (default)
dec  = build_dec(mesh, geom)

println("Area   = ", measure(mesh, geom), "  (exact: ", 4π*R^2, ")")
println("Volume = ", enclosed_measure(mesh), "  (exact: ", (4/3)π*R^3, ")")

χ   = euler_characteristic(mesh)
intK = integrated_gaussian_curvature(mesh, geom)
gb  = gauss_bonnet_residual(mesh, geom)
println("χ = $χ  (expect 2 for sphere)")
println("∫K dA = $intK  (expect $(4π))")
println("Gauss-Bonnet residual = $gb")
```

### Mixed/Voronoi dual areas

```julia
geom_mixed = compute_geometry(mesh; dual_area=:mixed)
println("Dual-area method: ", geom_mixed.dual_area_method)
println("All positive: ", all(geom_mixed.vertex_dual_areas .> 0))
```

### Compare DEC vs cotan Laplacians

```julia
rpt = compare_laplace_methods(mesh, geom)
println("‖L_dec − L_cotan‖_∞ = ", rpt.norm_inf)
println("L_dec nullspace: ", rpt.dec_nullspace)
println("L_cotan nullspace: ", rpt.cotan_nullspace)
```

### Load an STL surface

```julia
mesh = load_surface_stl("my_surface.stl")
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)

H = mean_curvature(mesh, geom, dec)
println("Mean curvature stats: min=", minimum(H), " max=", maximum(H))
```

### Construct a 2-D curve

```julia
using FrontIntrinsicOps

R    = 1.0
mesh = sample_circle(R, 128)
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)

println("Length         = ", measure(mesh, geom))
println("Enclosed area  = ", enclosed_measure(mesh))
```

### Surface diffusion (v0.3)

```julia
using FrontIntrinsicOps

mesh = generate_icosphere(1.0, 3)
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)

u   = Float64[p[3] for p in mesh.points]   # initial scalar field
dt  = 0.01;  μ = 0.1

# Backward-Euler step (reuse factorization for efficiency)
u, fac = step_surface_diffusion_backward_euler(mesh, geom, dec, u, dt, μ)
for _ in 2:100
    u, fac = step_surface_diffusion_backward_euler(mesh, geom, dec, u, dt, μ;
                                                    factorization=fac)
end
```

### Surface transport (v0.3)

```julia
using FrontIntrinsicOps, StaticArrays

mesh = generate_icosphere(1.0, 3)
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)

# Rigid-rotation velocity  v = (-y, x, 0)
vel = SVector{3,Float64}[SVector{3,Float64}(-p[2], p[1], 0.0) for p in mesh.points]
u   = Float64[p[3] for p in mesh.points]

dt = estimate_transport_dt(mesh, geom, vel; cfl=0.3)
A  = assemble_transport_operator(mesh, geom, vel; scheme=:upwind)

for _ in 1:10
    u = step_surface_transport_ssprk3(mesh, geom, A, u, dt)
end
```

### Surface advection–diffusion IMEX (v0.3)

```julia
using FrontIntrinsicOps, StaticArrays

mesh = generate_icosphere(1.0, 3)
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)

vel = SVector{3,Float64}[SVector{3,Float64}(-p[2], p[1], 0.0) for p in mesh.points]
u   = Float64[p[3] for p in mesh.points]
dt  = 0.01;  μ = 0.05

# Pre-assemble the (constant) transport operator for efficiency
A_up = assemble_transport_operator(mesh, geom, vel; scheme=:upwind)

u, fac = step_surface_advection_diffusion_imex(mesh, geom, dec, u, vel, dt, μ;
                                                scheme             = :upwind,
                                                transport_operator = A_up)
for _ in 2:100
    u, fac = step_surface_advection_diffusion_imex(mesh, geom, dec, u, vel, dt, μ;
                                                    scheme             = :upwind,
                                                    transport_operator = A_up,
                                                    factorization      = fac)
end
```

### Surface PDE with geodesics + Cartan transport (v0.5)

The new geodesic and Lie-derivative operators can be used directly in a PDE
time loop, for example:

1. initialize from intrinsic geodesic distance,
2. advect with `lie_derivative` (Cartan form),
3. diffuse with existing implicit DEC diffusion.

See:

- `examples/surface_pde_intrinsic_tools.jl`

### Ambient signed-distance queries (v0.5)

```julia
using FrontIntrinsicOps, StaticArrays

mesh  = generate_icosphere(1.0, 2)
cache = build_signed_distance_cache(mesh)

q = SVector(0.2, 0.1, 0.0)
r = signed_distance(q, cache; sign_mode=:auto)
println("distance = ", r.distance)
println("closest face id = ", r.primitive)
println("closest point = ", r.closest)

# batched query (Vector{SVector})
pts = [SVector(0.0, 0.0, 0.0), SVector(1.7, 0.0, 0.0)]
S, I, C, N = signed_distance(pts, cache; sign_mode=:winding)
```

Sign modes:
- `:unsigned` – always returns positive distance.
- `:pseudonormal` – valid for open and closed meshes (oriented side sign).
- `:winding` – valid for closed oriented meshes only (inside/outside sign).
- `:auto` – uses `:winding` on closed meshes, otherwise `:pseudonormal`.

Semantics:
- closed meshes with `:winding` return inside/outside signed distance.
- open meshes with `:pseudonormal` return oriented side-of-curve / side-of-sheet signed distance.
- open meshes do not define a global inside/outside region.

Pseudonormal edge cases:
- points exactly on the front return `0` distance.
- near numerically ambiguous sign configurations, the sign may evaluate to `0` when the signing dot product is within tolerance.

---

## Release notes

See [CHANGELOG.md](CHANGELOG.md) for the v0.5 ambient signed-distance release summary.

---

## Mathematical background

### Primal mesh

The **primal mesh** consists of:

- Vertices (0-simplices): indexed $i = 1, \ldots, N_V$.
- Edges (1-simplices): each edge $e_{ij}$ carries an orientation.
- Faces (2-simplices, triangles): each triangle carries an orientation.

### Dual measures (v0.2)

Two dual-area formulas are supported:

**Barycentric** (default, always valid):

$$A_i^{\text{bary}} = \frac{1}{3} \sum_{f \ni i} A_f$$

**Mixed/Voronoi** (Meyer et al. 2003, recommended for curvature-sensitive computations):

For each triangle $T = (a, b, c)$:
- If $T$ is **non-obtuse**: the Voronoi contribution at $a$ is
  $$A_a^{\text{Vor}} = \frac{1}{8}\bigl(|ab|^2 \cot\angle c + |ac|^2 \cot\angle b\bigr)$$
- If $T$ is **obtuse at $a$**: contribution at $a$ is $A_T / 2$, at $b$ and $c$ is $A_T / 4$.
- If $T$ is **obtuse at $b$ or $c$**: analogous fallback.

The mixed dual areas are always positive and sum to the total surface area.

Select the method via keyword argument:

```julia
geom = compute_geometry(mesh; dual_area=:barycentric)  # default
geom = compute_geometry(mesh; dual_area=:mixed)        # Meyer et al. 2003
geom = compute_geometry(mesh; dual_area=:voronoi)      # alias for :mixed
```

For curves, the dual length at vertex $i$ is

$$\ell_i^{\text{dual}} = \frac{1}{2}\left(\ell_{e_{i-1}} + \ell_{e_i}\right)$$

### Incidence matrices

The **vertex-to-edge incidence matrix** $d_0$ (size $N_E \times N_V$):

$$[d_0]_{e,i} = \begin{cases}+1 & \text{if edge } e \text{ starts at } i \\ -1 & \text{if edge } e \text{ ends at } i \\ 0 & \text{otherwise}\end{cases}$$

The **edge-to-face incidence matrix** $d_1$ (size $N_F \times N_E$):

$$[d_1]_{f,e} = \begin{cases}+1 & \text{if edge } e \text{ appears positively in face } f \\ -1 & \text{if edge } e \text{ appears negatively in face } f \\ 0 & \text{otherwise}\end{cases}$$

**Key identity:** $d_1 \, d_0 = 0$ (boundary of boundary is empty).

### Hodge stars

| Operator | Domain | Diagonal entry |
|----------|--------|----------------|
| $\star_0$ | vertices | $A_i^{\text{dual}}$ (dual area / dual length) |
| $\star_1$ | edges | cotan weight $w_e = \frac{1}{2}(\cot\alpha_e + \cot\beta_e)$ |
| $\star_2$ | faces | $A_f^{-1}$ (inverse face area) |

For curves, $\star_1 = \mathrm{diag}(\ell_e / \ell_e^{\text{dual}})$.

**Note on non-positive star1 entries:** For obtuse triangles, the cotan weight
$w_e$ can be negative.  This is a well-known issue with DEC on poor-quality
meshes.  The `:mixed` dual-area path does not fix the sign of $\star_1$ entries,
but mixed duals improve curvature accuracy on meshes with obtuse triangles.
Use `star1_sign_report(dec)` to check.

### Scalar Laplace–Beltrami (v0.2: two paths)

#### DEC factored path (`method=:dec`, default)

$$L = \star_0^{-1}\, d_0^\top\, \star_1\, d_0$$

#### Direct cotan path (`method=:cotan`)

$$(L u)_i = \frac{1}{A_i} \sum_{j \in N(i)} w_{ij}(u_i - u_j), \quad w_{ij} = \tfrac{1}{2}(\cot\alpha_{ij} + \cot\beta_{ij})$$

Both paths:
- return the same positive-semi-definite matrix $L = -\Delta_\Gamma$.
- use the same vertex ordering and dual areas.
- satisfy `L * ones ≈ 0` (constant nullspace).

On well-shaped meshes, `‖L_dec − L_cotan‖_∞ < 10⁻¹²`.

**Sign convention:** $L = -\Delta_\Gamma$ (positive semi-definite).

On a sphere of radius $R$:
$$L x = \frac{2}{R^2}\, x, \quad L y = \frac{2}{R^2}\, y, \quad L z = \frac{2}{R^2}\, z$$

### Scalar transport operator (v0.3)

The conservative advection operator uses the **DEC codifferential** formula:

$$(Au)_i = \sum_e [d_0]_{e,i} \cdot w_e \cdot v_e \cdot \ell_e \cdot u_e^{\text{scheme}}$$

where:
- $w_e = \tfrac{1}{2}(\cot\alpha_e + \cot\beta_e)$ is the ⋆₁ cotan weight,
- $\ell_e$ is the primal edge length,
- $v_e = \tfrac{v_i + v_j}{2} \cdot \hat{t}_e$ is the edge-flux velocity,
- $u_e^{\text{scheme}}$ is the donor-cell (`:upwind`) or centered (`:centered`) value.

The product $w_e \cdot \ell_e$ equals the dual-edge length on a Voronoi dual,
making this formula geometrically consistent with the Laplace–Beltrami operator.
The full semi-discrete equation is:

$$M \frac{du}{dt} + Au = 0, \quad M = \star_0 = \mathrm{diag}(A_i^{\text{dual}})$$

**Convergence rates** (rigid rotation on unit sphere, `z`-field):

| Scheme   | Temporal | Spatial rate |
|----------|----------|--------------|
| Upwind   | SSP-RK3  | O(h¹) |
| Centered | SSP-RK3  | O(h²) |

### Surface advection–diffusion IMEX (v0.3)

The combined PDE:

$$\frac{du}{dt} + M^{-1} A u + \mu L u = g$$

is split into:
- **Explicit**: transport term $M^{-1} A u^n$.
- **Implicit**: diffusion term $\mu L u^{n+1}$.

One IMEX step solves the linear system:

$$(I + dt\,\mu\,L)\, u^{n+1} = u^n - dt\, M^{-1} A u^n + dt\, g$$

The factorization of $(I + dt\,\mu\,L)$ is reused across steps when `dt` and
`μ` are constant (pass `factorization=fac`).  The transport operator `A` should
be pre-assembled and passed as `transport_operator=A` to avoid reassembly.

**Convergence rates** on the unit sphere (z-field, rotation-invariant):

| Scheme   | Spatial rate |
|----------|--------------|
| Upwind   | O(h¹) — first-order numerical diffusion ε_num ∼ |v|h/2 |
| Centered | O(h²) — matches Laplace–Beltrami accuracy |

### Angle-defect Gaussian curvature

The discrete Gaussian curvature at vertex $i$ is:

$$K(i) = \frac{2\pi - \sum_{f \ni i} \theta_{f,i}}{A_i^{\text{dual}}}$$

where $\theta_{f,i}$ is the interior angle of face $f$ at vertex $i$.

### Gauss–Bonnet theorem

This package uses the standard convention:

$$\int_\Gamma K \, dA = 2\pi\chi$$

where $\chi = V - E + F$ is the Euler characteristic.

| Surface | $\chi$ | $\int K \, dA$ |
|---------|--------|-----------------|
| Sphere  | 2      | $4\pi$          |
| Torus   | 0      | $0$             |
| Genus-$g$ closed surface | $2-2g$ | $2\pi(2-2g)$ |

Numerically, `gauss_bonnet_residual(mesh, geom)` returns
$|\int K \, dA - 2\pi\chi|$, which should be near machine precision.

---

## API Reference

### Types

```julia
CurveMesh{T}     # 2-D polygonal curve
SurfaceMesh{T}   # 3-D triangulated surface
CurveGeometry{T}
SurfaceGeometry{T}   # includes dual_area_method::Symbol (v0.2)
CurveDEC{T}
SurfaceDEC{T}
```

### IO

```julia
load_surface_stl(path)
load_curve_csv(path)
load_curve_points(pts; closed=true)
```

### Mesh generators (v0.2)

```julia
sample_circle(R, N)                  # closed N-gon approximation of a circle
generate_uvsphere(R, nphi, ntheta)   # UV (latitude-longitude) sphere
generate_icosphere(R, level)         # icosahedron + subdivision
generate_torus(R, r, ntheta, nphi)   # standard torus
```

### Core pipeline

```julia
compute_geometry(mesh; dual_area=:barycentric)  # dual_area: :barycentric, :mixed, :voronoi
build_dec(mesh, geom; laplace=:dec)             # laplace: :dec or :cotan
build_laplace_beltrami(mesh, geom; method=:dec)
laplace_beltrami(mesh, geom, dec, u)
```

### Curvature

```julia
curvature(mesh, geom)             # signed curvature on curves
mean_curvature(mesh, geom, dec)   # scalar mean curvature on surfaces
mean_curvature_normal(mesh, geom, dec)
gaussian_curvature(mesh, geom)    # angle-defect Gaussian curvature
```

### Integrals

```julia
measure(mesh, geom)                          # total length or area
enclosed_measure(mesh)                       # enclosed area or volume
integrate_vertex_field(mesh, geom, u)
integrate_face_field(mesh, geom, u)
integrated_gaussian_curvature(mesh, geom)   # ∫K dA  (v0.2)
```

### Diagnostics (v0.2)

```julia
check_mesh(mesh)                   # → NamedTuple (includes euler_characteristic)
check_dec(mesh, geom, dec; tol)    # → NamedTuple with DEC report
euler_characteristic(mesh)         # V - E + F
gauss_bonnet_residual(mesh, geom)  # |∫K dA - 2π χ|
star1_sign_report(dec)             # min entry, count non-positive, fraction
compare_laplace_methods(mesh, geom)  # ‖L_dec − L_cotan‖, nullspace checks
```

### Surface PDEs (v0.3)

```julia
# Diffusion
step_surface_diffusion_backward_euler(mesh, geom, dec, u, dt, μ; factorization)
step_surface_diffusion_crank_nicolson(mesh, geom, dec, u, dt, μ; factorization)

# Transport
assemble_transport_operator(mesh, geom, vel; scheme=:upwind)   # returns sparse A
estimate_transport_dt(mesh, geom, vel; cfl=0.5)                # CFL time step
step_surface_transport_forward_euler(mesh, geom, A, u, dt)
step_surface_transport_ssprk2(mesh, geom, A, u, dt)
step_surface_transport_ssprk3(mesh, geom, A, u, dt)

# Advection–diffusion IMEX
step_surface_advection_diffusion_imex(mesh, geom, dec, u, vel, dt, μ;
                                       scheme=:upwind,
                                       transport_operator=A,   # pass to reuse
                                       factorization=fac)      # pass to reuse
```

---

## Convergence scripts (v0.3)

The `convergence_pdes/` folder contains reproducible convergence studies for
the PDE solvers.

```bash
# Diffusion
julia --project=. convergence_pdes/diffusion_sphere_mesh.jl     # mesh refinement
julia --project=. convergence_pdes/diffusion_sphere_time.jl     # time refinement

# Transport
julia --project=. convergence_pdes/transport_sphere.jl          # mesh refinement

# Advection–diffusion
julia --project=. convergence_pdes/advection_diffusion_sphere.jl           # time-evolution diagnostic
julia --project=. convergence_pdes/advection_diffusion_sphere_mesh.jl      # mesh refinement
```

See `convergence_pdes/README.md` for details.

The `convergence/` folder contains the v0.2 geometry convergence studies:

```bash
julia --project=. convergence/circle_convergence.jl
julia --project=. convergence/sphere_convergence.jl
julia --project=. convergence/torus_convergence.jl
julia --project=. convergence/poisson_sphere_convergence.jl
julia --project=. convergence/run_all.jl   # run all
```

---

## Running tests

```bash
julia --project -e "using Pkg; Pkg.test()"
```

---

## Dependencies

| Package | Role |
|---------|------|
| `StaticArrays` | Point storage and local vector algebra |
| `SparseArrays` | Global operator matrices |
| `LinearAlgebra` | Norm, dot, cross, decompositions |
| `FileIO` + `MeshIO` | STL loading |
| `GeometryBasics` | Intermediate mesh representation from IO |

---

## Roadmap

- **v0.1**: Static geometry and operators — edge lengths, face normals,
  barycentric dual areas, incidence matrices, Hodge stars, scalar
  Laplace–Beltrami (DEC factored form), mean and Gaussian curvature,
  integral quantities, mesh diagnostics.

- **v0.2**: Mesh generators, mixed/Voronoi dual areas, cotan-vs-DEC
  Laplace comparison, Euler characteristic, Gauss–Bonnet diagnostic,
  convergence scripts.

- **v0.3** (current): Surface PDE solvers with verified convergence —
  scalar diffusion (BE/CN), conservative scalar transport with DEC-consistent
  cotan-weighted edge fluxes (upwind/centered, FE/SSP-RK2/RK3), and IMEX
  advection–diffusion.  Transport mesh convergence fixed (DEC ⋆₁ weights).
  Allocation reduction for IMEX reuse path.  New mesh convergence study for
  advection–diffusion.

- **Later (separate package)**: Moving front coupling, cut-cell / embedded
  boundary logic, time integration.

---

## Design principles

- **No bulk-solver coupling**: this package is intentionally agnostic to any
  Eulerian or Lagrangian bulk solver.
- **Explicit structs over metaprogramming**: small, readable containers.
- **Sparse matrices for all global operators**.
- **Barycentric duals first**: simpler and always valid, even for obtuse
  triangulations.  Mixed duals available for accuracy-sensitive computations.
- **Two Laplace paths**: DEC factored ($L = \star_0^{-1} d_0^\top \star_1 d_0$)
  and direct cotan; both sign-consistent and numerically close on good meshes.
- **Float64 default**: all internal computations in double precision unless
  the mesh is constructed with a different `T`.

---

## License

MIT
