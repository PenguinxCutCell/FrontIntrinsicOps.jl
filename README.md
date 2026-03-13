# FrontIntrinsicOps.jl

A **static interface-geometry / DDG / DEC package** for triangulated front meshes.

`FrontIntrinsicOps.jl` computes intrinsic geometric quantities and intrinsic
discrete exterior calculus (DEC) operators on front meshes.  The primary inputs
are triangulated 3-D closed surfaces (from STL files) and closed 2-D polygonal
curves.

This package is intended as an **intrinsic operator layer** for interface
meshes.  It does **not** implement front advection, remeshing, or coupling to
bulk solvers.  Those belong in separate packages.

---

## Scope

### What is implemented (v0.1)

| Feature | Status |
|---------|--------|
| `CurveMesh` / `SurfaceMesh` mesh types | ✓ |
| STL loading via MeshIO/GeometryBasics | ✓ |
| CSV / point-list curve loading | ✓ |
| Topology (edges, adjacency, orientation, manifold) | ✓ |
| Incidence matrices d₀, d₁ | ✓ |
| Geometry: normals, areas, dual areas, edge lengths | ✓ |
| Hodge stars ⋆₀, ⋆₁, ⋆₂ | ✓ |
| Scalar Laplace–Beltrami (curves and surfaces) | ✓ |
| Signed curvature (curves) | ✓ |
| Mean curvature (surfaces, via L applied to embedding) | ✓ |
| Gaussian curvature (angle-defect formula) | ✓ |
| Length, area, enclosed area/volume | ✓ |
| Field integration over front | ✓ |
| Mesh and DEC diagnostics | ✓ |

### What is deliberately not implemented

- Front advection / marker motion
- Remeshing / topology changes
- Coupling to Eulerian bulk solvers
- Space-time operators
- Embedded-boundary / cut-cell logic
- Generic symbolic exterior algebra

---

## Quick start

### Load an STL surface

```julia
using FrontIntrinsicOps

mesh = load_surface_stl("my_surface.stl")
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)

println("Area   = ", measure(mesh, geom))
println("Volume = ", enclosed_measure(mesh))

H = mean_curvature(mesh, geom, dec)
println("Mean curvature stats: min=", minimum(H), " max=", maximum(H))
```

### Construct a 2-D curve

```julia
using FrontIntrinsicOps, StaticArrays

N = 128; R = 1.0
pts = [R * SVector{2,Float64}(cos(2π*k/N), sin(2π*k/N)) for k in 0:N-1]
mesh = load_curve_points(pts; closed=true)
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)

println("Length         = ", measure(mesh, geom))
println("Enclosed area  = ", enclosed_measure(mesh))
println("Mean curvature = ", sum(curvature(mesh, geom))/N)
```

---

## Mathematical background

### Primal mesh

The **primal mesh** consists of:

- Vertices (0-simplices): indexed $i = 1, \ldots, N_V$.
- Edges (1-simplices): each edge $e_{ij}$ carries an orientation (from $i$ to $j$).
- Faces (2-simplices, triangles): each triangle carries an orientation.

### Dual measures

The **dual cell** of vertex $i$ is formed by connecting barycenters of
incident faces to midpoints of incident edges.  For v0.1, barycentric
dual areas are used:

$$A_i^{\text{dual}} = \frac{1}{3} \sum_{f \ni i} A_f$$

For curves, the dual length at vertex $i$ is

$$\ell_i^{\text{dual}} = \frac{1}{2}\left(\ell_{e_{i-1}} + \ell_{e_i}\right)$$

### Incidence matrices

The **vertex-to-edge incidence matrix** $d_0$ (size $N_E \times N_V$) encodes
edge orientation:

$$[d_0]_{e,i} = \begin{cases}+1 & \text{if edge } e \text{ starts at } i \\ -1 & \text{if edge } e \text{ ends at } i \\ 0 & \text{otherwise}\end{cases}$$

The **edge-to-face incidence matrix** $d_1$ (size $N_F \times N_E$) encodes
face orientation:

$$[d_1]_{f,e} = \begin{cases}+1 & \text{if edge } e \text{ appears positively in face } f \\ -1 & \text{if edge } e \text{ appears negatively in face } f \\ 0 & \text{otherwise}\end{cases}$$

**Key identity:** $d_1 \, d_0 = 0$ (boundary of boundary is empty).

### Hodge stars

| Operator | Domain | Diagonal entry |
|----------|--------|----------------|
| $\star_0$ | vertices | $A_i^{\text{dual}}$ (dual area / dual length) |
| $\star_1$ | edges | cotan weight $w_e = \frac{1}{2}(\cot\alpha_e + \cot\beta_e)$ |
| $\star_2$ | faces | $A_f^{-1}$ (inverse face area) |

For curves, $\star_1 = \mathrm{diag}(\ell_e / \ell_e^{\text{dual}})$.

### Scalar Laplace–Beltrami

The scalar Laplace–Beltrami operator is assembled as

$$L = \star_0^{-1}\, d_0^\top\, \star_1\, d_0$$

**Sign convention:**  $L = -\Delta_\Gamma$ (positive semi-definite).

On a sphere of radius $R$:

$$L x = \frac{2}{R^2}\, x, \quad L y = \frac{2}{R^2}\, y, \quad L z = \frac{2}{R^2}\, z$$

because $\Delta_\Gamma x = -(2/R^2) x$ and $L = -\Delta_\Gamma$.

### Mean-curvature normal from $\Delta_\Gamma \mathbf{x}$

The discrete mean-curvature normal vector at vertex $i$ is

$$\mathbf{H}_n(i) = \frac{1}{2} (L \mathbf{p})(i)$$

applied coordinate-by-coordinate (using $L = -\Delta_\Gamma$, so $(L \mathbf{p})(i) = -(\Delta_\Gamma \mathbf{p})(i)$).
The scalar mean curvature is $H(i) = \|\mathbf{H}_n(i)\|$ (positive for convex surfaces).

---

## API Reference

### Types

```julia
CurveMesh{T}     # 2-D polygonal curve
SurfaceMesh{T}   # 3-D triangulated surface
CurveGeometry{T}
SurfaceGeometry{T}
CurveDEC{T}
SurfaceDEC{T}
```

### IO

```julia
load_surface_stl(path)
load_curve_csv(path)
load_curve_points(pts; closed=true)
```

### Core pipeline

```julia
compute_geometry(mesh)           # → CurveGeometry or SurfaceGeometry
build_dec(mesh, geom)            # → CurveDEC or SurfaceDEC
laplace_beltrami(mesh, geom, dec, u)
```

### Curvature

```julia
curvature(mesh, geom)            # signed curvature on curves
mean_curvature(mesh, geom, dec)  # scalar mean curvature on surfaces
mean_curvature_normal(mesh, geom, dec)
gaussian_curvature(mesh, geom)   # angle-defect Gaussian curvature
```

### Integrals

```julia
measure(mesh, geom)              # total length or area
enclosed_measure(mesh)           # enclosed area or volume
integrate_vertex_field(mesh, geom, u)
integrate_face_field(mesh, geom, u)
```

### Diagnostics

```julia
check_mesh(mesh)   # → NamedTuple with topology report
check_dec(mesh, geom, dec; tol)  # → NamedTuple with DEC report
```

---

## Examples

See the `examples/` directory:

| File | Description |
|------|-------------|
| `sphere_from_stl.jl` | Load or generate sphere; report area, volume, curvature |
| `torus_from_stl.jl`  | Non-uniform curvature on a torus |
| `circle_curve.jl`    | 2-D circle convergence study |
| `surface_poisson.jl` | Solve a surface Poisson problem |

---

## Running examples

```bash
julia --project examples/sphere_from_stl.jl
julia --project examples/circle_curve.jl
julia --project examples/torus_from_stl.jl
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

- **v0.1** (current): Static geometry and operators — edge lengths, face
  normals, dual areas, incidence matrices, Hodge stars, scalar
  Laplace–Beltrami, mean and Gaussian curvature, integral quantities, mesh
  diagnostics.

- **v0.2**: Improved duals (mixed/Voronoi dual areas), better Gaussian
  curvature convergence, cotan-vs-DEC comparison, Euler characteristic and
  Gauss–Bonnet diagnostics.

- **v0.3**: Surface PDE examples (diffusion, transport of a scalar on the
  front), k-form operators.

- **Later (separate package)**: Moving front coupling, cut-cell / embedded
  boundary logic, time integration.

---

## Design principles

- **No bulk-solver coupling**: this package is intentionally agnostic to any
  Eulerian or Lagrangian bulk solver.
- **Explicit structs over metaprogramming**: small, readable containers.
- **Sparse matrices for all global operators**.
- **Barycentric duals first**: simpler and always valid, even for obtuse
  triangulations.
- **DEC factorisation**: $L = \star_0^{-1} d_0^\top \star_1 d_0$ is the
  canonical form; cotan weights enter via $\star_1$.
- **Float64 default**: all internal computations in double precision unless
  the mesh is constructed with a different `T`.

---

## License

MIT
