# Tutorial 3: Scalar Transport

This tutorial transports a scalar field on the sphere by a rigid rotation velocity,
using the SSP-RK3 time integrator with upwind and centered spatial schemes.

## Problem setup

Rigid rotation on the unit sphere:

$$v(x,y,z) = (-y,\, x,\, 0)$$

This is tangential to the sphere and **divergence-free** (so the integral of $u$
is conserved).

Initial condition: $u_0 = z$ (the height function).

Exact solution after time $T$: $u(x,y,z,T) = z\cos T + \sqrt{x^2+y^2}\sin T$
(field rotated by angle $T$ around the $z$-axis).

## Code

```julia
using FrontIntrinsicOps, StaticArrays

R    = 1.0
mesh = generate_icosphere(R, 4)
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)

# Velocity field: rigid rotation v = (-y, x, 0)
vel = SVector{3,Float64}[SVector(-p[2], p[1], 0.0) for p in mesh.points]

# Initial condition
u  = Float64[p[3] for p in mesh.points]
u0 = copy(u)

# CFL time step
dt = estimate_transport_dt(mesh, geom, vel; cfl=0.5)
println("dt = $dt")

# Assemble transport operator (constant velocity)
A_up  = assemble_transport_operator(mesh, geom, vel; scheme=:upwind)
A_cen = assemble_transport_operator(mesh, geom, vel; scheme=:centered)
```

## Running the simulation

```julia
T_final = 0.5          # rotate by 0.5 radian
N = round(Int, T_final / dt)
dt = T_final / N       # adjust so we hit T_final exactly

# Upwind + SSP-RK3
u_up = copy(u)
for _ in 1:N
    u_up = step_surface_transport_ssprk3(mesh, geom, A_up, u_up, dt)
end

# Centered + SSP-RK3
u_cn = copy(u)
for _ in 1:N
    u_cn = step_surface_transport_ssprk3(mesh, geom, A_cen, u_cn, dt)
end
```

## Error analysis

```julia
# Exact solution: rotate u₀ by T_final around z-axis
# For the field u₀ = z: exact solution is still z (z is rotation-invariant!)
u_exact = Float64[p[3] for p in mesh.points]   # z is unchanged by z-rotation

err_up  = maximum(abs, u_up  .- u_exact)
err_cen = maximum(abs, u_cn  .- u_exact)

println("Upwind error:   $err_up")
println("Centered error: $err_cen")
```

**Note:** Since $u_0 = z$ is invariant under rotation around the $z$-axis,
the exact solution is $u(T) = z$ exactly.  Any error is purely numerical.

## Mass conservation

```julia
mass0  = sum(geom.vertex_dual_areas .* u0)
mass_up  = sum(geom.vertex_dual_areas .* u_up)
mass_cen = sum(geom.vertex_dual_areas .* u_cn)

println("Initial mass:         $mass0")
println("Upwind mass drift:    ", abs(mass_up  - mass0))
println("Centered mass drift:  ", abs(mass_cen - mass0))
# Both should be < 1e-12 (machine precision)
```

## Using van Leer limiter (v0.4)

```julia
topo = build_topology(mesh)
u_vl = copy(u)

for _ in 1:N
    u_vl = step_surface_transport_limited(mesh, geom, dec, topo, u_vl, vel, dt;
                                           limiter=:van_leer, method=:ssprk2)
end

err_vl = maximum(abs, u_vl .- u_exact)
println("Van Leer error: $err_vl")
```

## Convergence rate study

```julia
levels = [2, 3, 4, 5]
errors = Float64[]

for level in levels
    m  = generate_icosphere(1.0, level)
    g  = compute_geometry(m)
    v  = SVector{3,Float64}[SVector(-p[2], p[1], 0.0) for p in m.points]
    u0 = Float64[p[3] for p in m.points]

    A  = assemble_transport_operator(m, g, v; scheme=:upwind)
    dt_cfl = estimate_transport_dt(m, g, v; cfl=0.5)
    N_steps = round(Int, T_final / dt_cfl)
    dt_step = T_final / N_steps

    u_k = copy(u0)
    for _ in 1:N_steps
        u_k = step_surface_transport_ssprk3(m, g, A, u_k, dt_step)
    end

    u_ex = Float64[p[3] for p in m.points]
    push!(errors, maximum(abs, u_k .- u_ex))
end

# Print convergence table
println("Level  N_V    L∞ error")
for (i, level) in enumerate(levels)
    nv = 12 * 4^level - 2 * 2^level + 2 # rough formula
    println("$level      $nv    $(errors[i])")
end
```

Expected convergence: upwind scheme achieves approximately $O(h^1)$ in $L^\infty$.

## See also

- [Getting started](01_getting_started.md)
- [Math: Scalar transport](../math/08_transport.md)
- [Math: High-resolution transport](../math/13_highres_transport.md)
- [API: PDE solvers](../api/pdes.md)
