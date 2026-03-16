# Caching and Performance

## Overview

The cache and performance modules reduce the cost of **repeated PDE solves**
by pre-assembling operators and pre-factorising linear systems.  The design
goal is **zero allocations per time step** once the cache is built.

---

## The bottleneck

Each implicit time step requires solving a linear system of size $N_V \times N_V$.
The typical workflow without caching:

1. Assemble $L$ (cheap, but non-trivial).
2. Form $(M + dt\,\mu\,L)$ (cheap).
3. Factorize the system (expensive, $O(N_V^{1.5})$ for sparse 2-D problems).
4. Solve with the factorization (cheap, $O(N_V)$ after factorization).

When $dt$ and $\mu$ are constant, steps 1–3 are **identical** at every time step.
The cache layer builds these once and reuses them.

---

## `SurfacePDECache`

```julia
cache = build_pde_cache(mesh, geom, dec;
                         μ         = 0.1,    # diffusion coefficient
                         dt        = 1e-3,   # time step
                         θ         = 1.0,    # implicit weight (1=BE, 0.5=CN)
                         α_helmholtz = nothing)  # Helmholtz shift (if needed)
```

The cache stores:
- `dec` — the full DEC struct
- `mass` — the mass matrix $M = \star_0$
- `laplacian` — the Laplacian $L$
- `system` — the preassembled implicit matrix $(M + dt\,\theta\,\mu\,L)$
- `factorization` — the sparse LU/Cholesky factorization of `system`
- `μ`, `dt`, `θ` — scalar parameters

**Invalidation:** The cache must be rebuilt whenever $dt$, $\mu$, or the mesh
changes.  For time-varying problems with adaptive $dt$, call `update_pde_cache`.

---

## Diffusion step (cached)

```julia
# Build once (expensive)
cache = build_pde_cache(mesh, geom, dec; μ=0.1, dt=1e-3)

# Step many times (cheap)
for _ in 1:1000
    step_diffusion_cached!(u, cache)
end
```

Internally: `u ← factorization \ (M * u)`.  The factorization is reused.

---

## Reaction–diffusion step (cached)

```julia
cache = build_pde_cache(mesh, geom, dec; μ=0.1, dt=1e-3, θ=1.0)
R = fisher_kpp_reaction(2.0)

for n in 1:1000
    step_diffusion_cached!(u, cache, R, n * cache.dt)
end
```

The reaction vector $R^n$ is evaluated and added to the RHS before each solve.
The matrix factorization is reused.

---

## Helmholtz solve (cached)

```julia
cache = build_pde_cache(mesh, geom, dec; α_helmholtz=0.01)
u = solve_helmholtz_cached(cache, f)
```

Useful when many RHS vectors $f$ must be solved with the same operator
$(L + \alpha M)$.

---

## Updating the cache

```julia
cache = update_pde_cache(cache, mesh, geom, dec; dt=5e-4)
```

This rebuilds the system matrix and refactorises.

---

## `CurvePDECache`

The analogous cache for `CurveMesh` problems:

```julia
cache = build_pde_cache(mesh, geom, dec; μ=0.05, dt=1e-2)
step_diffusion_cached!(u, cache)
```

---

## Performance buffers (zero-allocation kernels)

For the tightest inner loops, the `performance.jl` module provides pre-allocated
scratch buffers:

```julia
buf = alloc_diffusion_buffers(nv)   # SurfaceDiffusionBuffers
buf = alloc_rd_buffers(nv)          # SurfaceRDBuffers
```

**Diffusion step (in-place, no allocations):**

```julia
step_diffusion_inplace!(u, cache, buf)
```

**Reaction–diffusion step (in-place):**

```julia
step_rd_inplace!(u, cache, buf, mesh, geom, R, t)
```

**In-place matrix–vector products:**

```julia
apply_mass_inplace!(y, cache, x)     # y ← M x
apply_laplace_inplace!(y, cache, x)  # y ← L x
```

---

## Norms

```julia
l2 = l2_norm_cached(cache, u)       # ‖u‖_{L²} = √(u⊤ M u)
en  = energy_norm_cached(cache, u)  # ‖∇u‖_{H¹} = √(u⊤ L u)
```

These reuse the cached mass and Laplacian matrices.

---

## Memory usage

The dominant memory cost is the factorization, which for a sparse 2-D problem
of size $N_V$ requires $O(N_V \log N_V)$ memory (supernodal LU).  For a mesh
with 10,000 vertices this is typically a few MB.

| Level | $N_V$ approx | Factorization size | Step time |
|-------|-------------|-------------------|-----------|
| Icosphere level 3 | 642 | < 1 MB | < 1 ms |
| Icosphere level 4 | 2562 | ~ 5 MB | ~ 5 ms |
| Icosphere level 5 | 10242 | ~ 50 MB | ~ 20 ms |

---

## Benchmark pattern

```julia
using BenchmarkTools

mesh = generate_icosphere(1.0, 4)
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)

cache = build_pde_cache(mesh, geom, dec; μ=0.1, dt=1e-3)
buf   = alloc_diffusion_buffers(length(mesh.points))
u     = zeros(length(mesh.points))

@benchmark step_diffusion_inplace!($u, $cache, $buf)
# Target: 0 allocations, < 5 ms for level-4 icosphere
```

---

## See also

- [Surface diffusion](07_surface_diffusion.md)
- [Reaction–diffusion](10_reaction_diffusion.md)
