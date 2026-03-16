# Tutorial 1: Getting Started

This tutorial shows how to set up FrontIntrinsicOps.jl, generate a sphere mesh,
and compute basic geometric quantities — all in less than 20 lines of code.

## Prerequisites

```julia
using Pkg
Pkg.add("FrontIntrinsicOps")
```

## Step 1: Generate a mesh

```julia
using FrontIntrinsicOps

R    = 1.0
mesh = generate_icosphere(R, 3)   # level-3 icosphere: 642 vertices, 1280 faces
```

The level-3 icosphere provides mesh size $h \approx 0.14$.  Use level 4 or 5
for convergence studies.

## Step 2: Compute geometry

```julia
geom = compute_geometry(mesh)   # barycentric dual areas (default)
```

This fills `geom.face_normals`, `geom.face_areas`, `geom.edge_lengths`,
`geom.vertex_dual_areas`, and `geom.vertex_normals`.

## Step 3: Build DEC operators

```julia
dec = build_dec(mesh, geom)
```

This assembles:
- `dec.d0` — vertex-to-edge coboundary $d_0$
- `dec.d1` — edge-to-face coboundary $d_1$
- `dec.star0` — Hodge star $\star_0$ (dual areas)
- `dec.star1` — Hodge star $\star_1$ (cotan weights)
- `dec.laplacian` — Laplace–Beltrami $L = \star_0^{-1} d_0^\top \star_1 d_0$

## Step 4: Geometric checks

```julia
println("Area           = ", measure(mesh, geom))        # ≈ 4π ≈ 12.566
println("Enclosed vol.  = ", enclosed_measure(mesh))     # ≈ (4/3)π ≈ 4.189
println("Euler χ        = ", euler_characteristic(mesh)) # = 2 for sphere
println("Gauss–Bonnet   = ", gauss_bonnet_residual(mesh, geom))  # ≈ 0
```

For a sphere of radius $R = 1$:

| Quantity | Exact | Numerical (level 3) |
|----------|-------|---------------------|
| Surface area | $4\pi \approx 12.566$ | $12.56$ |
| Enclosed volume | $\frac{4}{3}\pi \approx 4.189$ | $4.17$ |
| Euler char. | 2 | 2 (exact) |
| Gauss–Bonnet residual | 0 | $< 10^{-12}$ |

## Step 5: Curvature

```julia
H = mean_curvature(mesh, geom, dec)
K = gaussian_curvature(mesh, geom)

println("Mean curvature: min=", minimum(H), "  max=", maximum(H))
# ≈ [1.0, 1.0] for sphere of radius 1

println("Gaussian curvature: min=", minimum(K), "  max=", maximum(K))
# ≈ [1.0, 1.0] for sphere of radius 1

println("∫K dA = ", integrated_gaussian_curvature(mesh, geom))
# ≈ 4π ≈ 12.566
```

## Step 6: DEC diagnostics

```julia
report = check_dec(mesh, geom, dec; tol=1e-10)
println(report.d1_d0_zero)          # true: d₁ d₀ = 0
println(report.lap_constant_nullspace) # true: L 1 ≈ 0
println(report.star0_positive)      # true: all dual areas > 0
```

## Step 7: Apply the Laplacian

For a smooth function $u = z$ on the sphere, $-\Delta_\Gamma z = (2/R^2) z$:

```julia
u  = [p[3] for p in mesh.points]   # z-coordinate at each vertex
Lu = dec.laplacian * u             # L u ≈ (2/R²) u

err = maximum(abs, Lu .- 2.0 .* u)
println("Eigenvalue error: $err")   # → ~1e-2 at level 3, → 0 as level → ∞
```

## Complete script

```julia
using FrontIntrinsicOps

R    = 1.0
mesh = generate_icosphere(R, 3)
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)

println("=== Geometric quantities ===")
println("Area       = ", measure(mesh, geom))
println("Volume     = ", enclosed_measure(mesh))
println("χ          = ", euler_characteristic(mesh))
println("GB residual = ", gauss_bonnet_residual(mesh, geom))

H = mean_curvature(mesh, geom, dec)
println("\n=== Curvature ===")
println("H range = [", minimum(H), ", ", maximum(H), "]   (exact: 1.0)")

println("\n=== DEC check ===")
rpt = check_dec(mesh, geom, dec)
println("d1∘d0 = 0 : ", rpt.d1_d0_zero)
println("L nullspace: ", rpt.lap_constant_nullspace)
```

## Next steps

- [Surface diffusion tutorial](02_surface_diffusion.md)
- [Scalar transport tutorial](03_transport.md)
- [Math: Discrete geometry](../math/03_geometry.md)
- [API: Geometry and DEC](../api/geometry.md)
