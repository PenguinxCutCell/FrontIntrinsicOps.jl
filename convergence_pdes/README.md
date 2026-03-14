# convergence_pdes/README.md
#
# Convergence studies for PDE solvers in FrontIntrinsicOps.jl

# PDE Convergence Tests

This directory contains standalone convergence studies for the time-dependent
PDE solvers in FrontIntrinsicOps.jl.  Each script is self-contained and can
be run directly with Julia from the repository root:

```
julia --project=.. convergence_pdes/<script>.jl
```

or from within the `convergence_pdes/` directory:

```
julia --project=.. diffusion_sphere_mesh.jl
```

## Scripts

| Script | Description |
|--------|-------------|
| `common.jl` | Shared utilities (mesh construction, exact solutions, table printing) |
| `diffusion_sphere_mesh.jl` | Mesh refinement convergence for transient diffusion on the sphere |
| `diffusion_sphere_time.jl` | Time refinement convergence comparing backward Euler vs Crank–Nicolson |
| `transport_sphere.jl` | Transport accuracy study: rigid rotation, centered vs upwind |
| `advection_diffusion_sphere.jl` | IMEX benchmark: rigid rotation + small diffusion |

## Physical setup

All tests use the **unit sphere** (R=1) with the **z-coordinate** as the
initial condition.  This is the first-degree spherical harmonic Y₁⁰, which is
an eigenfunction of the scalar Laplace–Beltrami operator:

    L z = (2/R²) z    →    λ₁ = 2  (for R=1)

### Diffusion exact solution

    u(x,y,z,t) = exp(-μ λ₁ t) · z

### Transport exact solution (rigid rotation about z-axis)

    v = (-y, x, 0),  ω = 1 rad/s
    u(x,y,z,t) = z    (z is invariant under rotation about z)

## Convergence rates expected

| Scheme | Spatial | Temporal |
|--------|---------|----------|
| Backward Euler | O(h²) | O(dt) |
| Crank–Nicolson | O(h²) | O(dt²) |
| Upwind transport (FE) | O(h) | O(dt) |
| Centered transport (SSP-RK3) | O(h²) | O(dt³) |
