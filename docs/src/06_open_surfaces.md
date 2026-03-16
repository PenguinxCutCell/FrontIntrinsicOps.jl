# Tutorial 6: Open Surfaces and Dirichlet Boundary Conditions

This tutorial solves the **Poisson equation** on an open surface (spherical cap)
with Dirichlet boundary conditions.

## Problem formulation

Let $\Gamma_h$ be a flat square patch $[0,1]^2$ triangulated uniformly.
Solve:

$$-\Delta_\Gamma u = f \quad \text{in } \Gamma_h$$
$$u = g \quad \text{on } \partial\Gamma_h$$

**Test case 1:** $u(x,y) = x$ (linear, $f = 0$).  The cotan Laplacian is exact
for P1 functions so the error should be at machine precision.

**Test case 2:** $u(x,y) = \sin(\pi x)\sin(\pi y)$ (manufactured solution,
$f = 2\pi^2 \sin(\pi x)\sin(\pi y)$).  Expected $O(h^2)$ convergence in $L^\infty$.

---

## Generating an open surface

The built-in generators produce closed surfaces.  For this tutorial we
construct a flat patch by hand:

```julia
using FrontIntrinsicOps, StaticArrays

function make_square_patch(n)
    # n×n grid of vertices, 2(n-1)² triangles
    pts = SVector{3,Float64}[]
    for j in 0:(n-1), i in 0:(n-1)
        push!(pts, SVector(i/(n-1), j/(n-1), 0.0))
    end
    idx(i,j) = j*n + i + 1
    faces = Tuple{Int,Int,Int}[]
    for j in 0:(n-2), i in 0:(n-2)
        a, b, c, d = idx(i,j), idx(i+1,j), idx(i+1,j+1), idx(i,j+1)
        push!(faces, (a, b, c))
        push!(faces, (a, c, d))
    end
    return SurfaceMesh(pts, faces)
end

mesh = make_square_patch(20)
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)
topo = build_topology(mesh)

println("Open surface: ", is_open_surface(topo))         # true
println("Boundary verts: ", length(detect_boundary_vertices(mesh, topo)))
```

---

## Test case 1: Linear solution $u = x$

```julia
# Exact solution: u(x,y,z) = x
u_exact = [p[1] for p in mesh.points]

# Right-hand side: f = 0
f = zeros(length(mesh.points))

# Boundary values
bnd_verts = detect_boundary_vertices(mesh, topo)
bnd_vals  = [mesh.points[v][1] for v in bnd_verts]   # u = x on ∂Γ

# Solve
u = solve_open_surface_poisson(mesh, geom, dec, topo, f, bnd_verts, bnd_vals)

err = maximum(abs, u .- u_exact)
println("Linear test error: $err")   # ≈ machine precision (exact for P1)
```

---

## Test case 2: Manufactured solution $u = \sin(\pi x)\sin(\pi y)$

```julia
# Exact solution
u_exact_mfr = [sin(π * p[1]) * sin(π * p[2]) for p in mesh.points]

# Source term: -ΔΓ u = 2π² sin(πx)sin(πy)  (flat patch, Δ = ∂²/∂x² + ∂²/∂y²)
f_mfr = [2π^2 * sin(π * p[1]) * sin(π * p[2]) for p in mesh.points]

# Homogeneous Dirichlet BCs (u = 0 on ∂Γ since sin vanishes there)
bnd_vals_zero = zeros(length(bnd_verts))

u_mfr = solve_open_surface_poisson(mesh, geom, dec, topo, f_mfr,
                                    bnd_verts, bnd_vals_zero)

err_mfr = maximum(abs, u_mfr .- u_exact_mfr)
println("Manufactured solution error (n=20): $err_mfr")  # ≈ 5e-3
```

---

## Convergence study

```julia
ns = [5, 10, 20, 40]
errs = Float64[]

for n in ns
    m    = make_square_patch(n)
    g    = compute_geometry(m)
    d    = build_dec(m, g)
    tp   = build_topology(m)

    f_k  = [2π^2 * sin(π * p[1]) * sin(π * p[2]) for p in m.points]
    bv   = detect_boundary_vertices(m, tp)
    bval = zeros(length(bv))

    u_k  = solve_open_surface_poisson(m, g, d, tp, f_k, bv, bval)
    u_ex = [sin(π * p[1]) * sin(π * p[2]) for p in m.points]

    push!(errs, maximum(abs, u_k .- u_ex))
end

println("n    L∞ error   rate")
for i in 2:length(ns)
    rate = log(errs[i-1]/errs[i]) / log(ns[i]/ns[i-1])
    println("$(ns[i])    $(errs[i])   $(round(rate, digits=2))")
end
# Expected: rate ≈ 2.0 (second-order convergence)
```

---

## Applying Dirichlet conditions manually

For full control over the system assembly:

```julia
using SparseArrays, LinearAlgebra

# Assemble the raw Laplacian
L = copy(dec.laplacian)
f = [2π^2 * sin(π * p[1]) * sin(π * p[2]) for p in mesh.points]
b = mass_matrix(mesh, geom) * f   # apply mass matrix to get RHS

# Apply Dirichlet BCs (symmetric form — preserves sparsity pattern)
apply_dirichlet_symmetric!(L, b, bnd_verts, zeros(length(bnd_verts)))

# Solve
u = L \ b
```

---

## Open surface diffusion

Solving the heat equation on an open surface with fixed boundary:

```julia
u = [sin(π * p[1]) * sin(π * p[2]) for p in mesh.points]   # initial condition

dt = 0.001
μ  = 0.01

for n in 1:100
    # Assemble system with Dirichlet BCs
    A = copy(dec.laplacian)
    b = (mass_matrix(mesh, geom) ./ dt) * u
    A_impl = mass_matrix(mesh, geom) / dt + μ * dec.laplacian
    apply_dirichlet_symmetric!(A_impl, b, bnd_verts, zeros(length(bnd_verts)))
    u = A_impl \ b
end
```

---

## See also

- [Math: Open surfaces](14_open_surfaces.md)
- [Math: Surface diffusion](07_surface_diffusion.md)
- [API: PDE solvers](pdes.md)
