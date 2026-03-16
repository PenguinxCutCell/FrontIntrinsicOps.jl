# Tutorial 2: Surface Diffusion

This tutorial solves the **heat equation on the sphere**:

$$\frac{\partial u}{\partial t} = \mu \Delta_\Gamma u$$

using backward Euler (first order) and Crank–Nicolson (second order) time integration.

## Setup

```julia
using FrontIntrinsicOps

R    = 1.0
mesh = generate_icosphere(R, 4)   # ~2562 vertices
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)

nv  = length(mesh.points)
μ   = 0.1    # diffusion coefficient
dt  = 0.01   # time step
T   = 1.0    # final time
```

## Initial condition

Use the first spherical harmonic $u_0 = z / R$ (eigenfunction of $\Delta_\Gamma$
with eigenvalue $\lambda_1 = 2/R^2$):

```julia
u = [p[3] / R for p in mesh.points]   # u₀ = z (ℓ=1 mode)
```

**Exact solution:** $u(t) = e^{-\mu \lambda_1 t} u_0 = e^{-2\mu t} u_0$.

## Backward Euler (first-order accurate)

```julia
# First step: builds and factorises the system matrix
u_be, fac = step_surface_diffusion_backward_euler(mesh, geom, dec, u, dt, μ)

# Subsequent steps: reuse factorisation (no re-assembly)
N_steps = round(Int, T / dt)
for _ in 2:N_steps
    u_be, _ = step_surface_diffusion_backward_euler(mesh, geom, dec, u_be, dt, μ;
                                                     factorization=fac)
end
```

## Crank–Nicolson (second-order accurate)

```julia
u_cn = copy(u)
u_cn, fac_cn = step_surface_diffusion_crank_nicolson(mesh, geom, dec, u_cn, dt, μ)

for _ in 2:N_steps
    u_cn, _ = step_surface_diffusion_crank_nicolson(mesh, geom, dec, u_cn, dt, μ;
                                                      factorization=fac_cn)
end
```

## Error analysis

```julia
# Exact solution at t = T
u_exact = exp(-2 * μ * T) .* [p[3] / R for p in mesh.points]

# L∞ errors
err_be = maximum(abs, u_be .- u_exact)
err_cn = maximum(abs, u_cn .- u_exact)

println("Backward Euler error:   $err_be")
println("Crank–Nicolson error:  $err_cn")
```

Expected results (level-4 icosphere, $\mu = 0.1$, $dt = 0.01$, $T = 1$):

| Method | $L^\infty$ error | Order |
|--------|-----------------|-------|
| Backward Euler | $\sim 5 \times 10^{-3}$ | $O(dt)$ |
| Crank–Nicolson | $\sim 5 \times 10^{-5}$ | $O(dt^2)$ |

## Using the cache for performance

For long simulations, the `SurfacePDECache` avoids any re-assembly overhead:

```julia
cache = build_pde_cache(mesh, geom, dec; μ=μ, dt=dt, θ=1.0)
u     = [p[3] / R for p in mesh.points]

for _ in 1:N_steps
    step_diffusion_cached!(u, cache)
end
```

The cache is built once (includes factorization) and reused at zero marginal
allocation cost per step.

## Poisson solve (steady state)

For the **Poisson equation** $-\Delta_\Gamma u = f$ on the sphere with a
compatible source $f$:

```julia
# Source: f = 6z (eigenfunction with eigenvalue 6/R² for ℓ=2)
f = [6 * p[3] for p in mesh.points]
enforce_compatibility!(f, mesh, geom)   # ensure ∫ f dA = 0

u_poisson = solve_surface_poisson(mesh, geom, dec, f)
println("Poisson residual: ", maximum(abs, dec.laplacian * u_poisson .- f))
```

## See also

- [Getting started](01_getting_started.md)
- [Reaction–diffusion tutorial](04_reaction_diffusion.md)
- [Math: Surface diffusion](../math/07_surface_diffusion.md)
- [API: PDE solvers](../api/pdes.md)
