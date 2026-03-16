# Tutorial 4: Reaction–Diffusion (Fisher–KPP)

This tutorial solves the **Fisher–KPP equation** on the sphere:

$$\frac{\partial u}{\partial t} = \mu \Delta_\Gamma u + \alpha u(1 - u)$$

The Fisher–KPP equation models logistic population growth with diffusive spread.
Starting from a localized initial condition, the solution evolves into a
travelling wave that invades the sphere.

## Mathematical background

**Equilibria:** $u^* = 0$ (unstable) and $u^* = 1$ (stable attractor).

**Travelling wave speed (on flat $\mathbb{R}^2$):**
$$c^* = 2\sqrt{\mu \alpha}$$

On a sphere the behaviour is more complex because the curvature modulates the wave
and the geometry is bounded.

## Setup

```julia
using FrontIntrinsicOps

R    = 1.0
mesh = generate_icosphere(R, 4)
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)

μ   = 0.02       # diffusion coefficient
α   = 2.0        # reaction rate (logistic)
dt  = 1e-3       # time step
T   = 5.0        # final time
```

## Initial condition

Start with a small patch near the north pole:

```julia
u = zeros(length(mesh.points))
for (i, p) in enumerate(mesh.points)
    if p[3] > 0.8    # northern cap
        u[i] = 1.0
    end
end
```

## Integration with the cache

```julia
cache = build_pde_cache(mesh, geom, dec; μ=μ, dt=dt, θ=1.0)
R_fkpp = fisher_kpp_reaction(α)

N_steps = round(Int, T / dt)
t = 0.0
for n in 1:N_steps
    step_diffusion_cached!(u, cache, R_fkpp, t)
    t += dt
end

println("Final coverage: u > 0.5 at ", count(u .> 0.5), " / ", length(u), " vertices")
```

## Monitoring the solution

To record the solution at intermediate times:

```julia
u = zeros(length(mesh.points))
for (i, p) in enumerate(mesh.points)
    u[i] = p[3] > 0.8 ? 1.0 : 0.0
end

cache    = build_pde_cache(mesh, geom, dec; μ=μ, dt=dt, θ=1.0)
R_fkpp   = fisher_kpp_reaction(α)
snapshots = [copy(u)]
times     = [0.0]

N_steps = round(Int, T / dt)
for n in 1:N_steps
    step_diffusion_cached!(u, cache, R_fkpp, (n-1)*dt)
    if n % 500 == 0
        push!(snapshots, copy(u))
        push!(times, n * dt)
        println("t = ", times[end], "  min=", minimum(u), "  max=", maximum(u))
    end
end
```

## Checking the maximum principle

The Fisher–KPP equation has a maximum principle: if $u^0 \in [0,1]$, then
$u(t) \in [0,1]$ for all $t > 0$.  Check this numerically:

```julia
println("u range: [", minimum(u), ", ", maximum(u), "]")
# Should remain within [0, 1] (approximately — IMEX can slightly exceed bounds)
```

## Convergence test with exact solution

For **linear decay** $R(u) = -\alpha u$, there is an exact solution:

```julia
# Decay test: exact solution = e^{-(μ λ₁ + α) t} z
# λ₁ = 2/R² for ℓ=1 mode on unit sphere
λ1 = 2.0 / R^2
α_decay = 0.5
R_decay = linear_decay_reaction(α_decay)

u0 = [p[3] for p in mesh.points]   # ℓ=1 mode
u  = copy(u0)

cache_decay = build_pde_cache(mesh, geom, dec; μ=μ, dt=dt, θ=1.0)
T_final = 1.0
N = round(Int, T_final / dt)

for n in 1:N
    step_diffusion_cached!(u, cache_decay, R_decay, (n-1)*dt)
end

decay_exact = exp(-(μ * λ1 + α_decay) * T_final)
u_exact     = decay_exact .* u0
err         = maximum(abs, u .- u_exact)
println("Linear decay error: $err   (expected ~$(dt * (μ*λ1 + α_decay)))")
```

## Custom reaction term

Define your own reaction term as a pointwise callback:

```julia
# Bistable / Allen–Cahn reaction
bistable(u, x, t) = 5.0 * u * (1 - u) * (u - 0.5)

u = rand(length(mesh.points))   # random initial data in [0,1]
cache = build_pde_cache(mesh, geom, dec; μ=0.01, dt=1e-3, θ=1.0)

for n in 1:2000
    step_diffusion_cached!(u, cache, bistable, n*1e-3)
end

# Solution should approach a pattern of u ≈ 0 and u ≈ 1 domains
println("u < 0.2:  ", count(u .< 0.2), " vertices")
println("u > 0.8:  ", count(u .> 0.8), " vertices")
```

## See also

- [Surface diffusion tutorial](02_surface_diffusion.md)
- [Math: Reaction–diffusion](10_reaction_diffusion.md)
- [Math: Caching and performance](15_caching.md)
- [API: PDE solvers](pdes.md)
