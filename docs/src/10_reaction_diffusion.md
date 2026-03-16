# Reaction–Diffusion on Surfaces

## Overview

The **reaction–diffusion equation** on a surface is:

$$\frac{\partial u}{\partial t} = \mu \Delta_\Gamma u + R(u, x, t)$$

where $\mu > 0$ is the diffusion coefficient and $R$ is the (generally
nonlinear) reaction term.  This equation appears in:

- Pattern formation (Turing instability, morphogenesis)
- Population dynamics (Fisher–KPP travelling waves on surfaces)
- Chemical reactions on catalytic surfaces
- Phase-field models

---

## IMEX $\theta$-scheme

The reaction term is treated **explicitly** (it is nonlinear and cheap to
evaluate), while the stiff diffusion operator is treated **implicitly**.

The $\theta$-scheme (where $\theta \in [0, 1]$ blends backward and forward Euler
for the diffusion):

$$(M + dt\,\theta\,\mu\,L) u^{n+1} = (M - dt\,(1-\theta)\,\mu\,L) u^n + dt\, M R^n$$

where $R^n_i = R(u^n_i, x_i, t^n)$ is evaluated pointwise.

| $\theta$ | Scheme | Order | Stability |
|----------|--------|-------|-----------|
| $1$ | Backward Euler | $O(dt)$ | L-stable |
| $0.5$ | Crank–Nicolson | $O(dt^2)$ | A-stable (not L-stable) |
| $0$ | Forward Euler (unstable) | $O(dt)$ | Conditionally stable only |

**Default:** $\theta = 1$ (backward Euler — recommended for stiff reactions).

```julia
u, fac = step_surface_reaction_diffusion_imex(mesh, geom, dec, u, dt, μ, reaction, t;
                                               θ=1.0)
```

---

## Reaction API

Three forms are supported:

### 1. Built-in factory functions

```julia
R = fisher_kpp_reaction(α)       # f(u) = α u(1-u)
R = linear_decay_reaction(α)     # f(u) = -α u
R = bistable_reaction(α)         # f(u) = α u(1-u)(u-0.5)
```

Each returns a **callable** that is called as `R(u_i, x_i, t)`.

### 2. Pointwise scalar callback

```julia
my_reaction(u, x, t) = sin(x[1]) * u - u^3
u, fac = step_surface_reaction_diffusion_imex(mesh, geom, dec, u, dt, μ,
                                               my_reaction, t)
```

### 3. In-place vector callback

For maximum performance (no allocations):

```julia
function my_reaction!(r, u, mesh, geom, t)
    for i in eachindex(u)
        r[i] = 2.0 * u[i] * (1 - u[i])
    end
end
```

---

## Built-in reaction models

### Fisher–KPP (logistic growth)

$$R(u) = \alpha\, u (1 - u)$$

- Equilibria: $u = 0$ (unstable), $u = 1$ (stable).
- Travelling wave speed: $c^* = 2\sqrt{\mu \alpha}$.
- Models logistic population growth with diffusive spread.

### Linear decay

$$R(u) = -\alpha\, u, \quad \alpha > 0$$

- Exact solution for decay from $u_0$: $u(t) = e^{-(\lambda + \alpha) t} u_0$
  where $\lambda = \mu \lambda_\ell$ is the diffusion eigenvalue.
- Useful for convergence testing (exact solutions known).

### Bistable (Allen–Cahn type)

$$R(u) = \alpha\, u(1-u)(u - 0.5)$$

- Three equilibria: $u = 0$ (stable), $u = 0.5$ (unstable), $u = 1$ (stable).
- Generates phase-field-like solutions with sharp interfaces.

---

## Stability: maximum principle

For the Fisher–KPP equation with $u^0 \in [0, 1]$, the continuous solution
satisfies $u \in [0, 1]$ for all $t > 0$.  The IMEX scheme does **not**
generally preserve this property exactly, but for small $dt$ the solution
remains close to $[0, 1]$.

For the **Backward Euler** ($\theta = 1$) scheme with linear reactions, the
discrete maximum principle holds if $dt \leq 1/\alpha$.

---

## Convergence study

**Test case:** Linear decay on the sphere with exact solution.

Let $u_0 = Y_\ell^m$ (spherical harmonic), $R(u) = -\alpha u$:

$$u(t) = e^{-(\mu\lambda_\ell + \alpha) t} Y_\ell^m$$

where $\lambda_\ell = \ell(\ell+1)/R^2$.  This yields an **exact solution** for
convergence testing.

---

## Time integration API

### Single step (IMEX)

```julia
u, fac = step_surface_reaction_diffusion_imex(mesh, geom, dec, u, dt, μ, reaction, t;
                                               θ=1.0)
```

### Full integration

```julia
u_final, t_final = solve_surface_reaction_diffusion(mesh, geom, dec, u0, T_end, dt,
                                                     μ, reaction;
                                                     θ=1.0, scheme=:imex)
```

### Explicit Euler (reference, unstable for large $dt$)

```julia
u_new = step_surface_reaction_diffusion_explicit(mesh, geom, dec, u, dt, μ,
                                                  reaction, t)
```

---

## Factorization reuse

For constant $dt$, $\mu$, $\theta$, the matrix $(M + dt\,\theta\,\mu\,L)$
is the same at every step.  Passing back the factorization avoids recomputation:

```julia
u, fac = step_surface_reaction_diffusion_imex(mesh, geom, dec, u, dt, μ, R, 0.0)
for n in 1:N
    global u, fac
    u, _ = step_surface_reaction_diffusion_imex(mesh, geom, dec, u, dt, μ, R,
                                                  n*dt; factorization=fac)
end
```

Or use the **cache layer** which handles this automatically:

```julia
cache = build_pde_cache(mesh, geom, dec; μ=0.1, dt=1e-3, θ=1.0)
R = fisher_kpp_reaction(2.0)
for n in 1:N
    step_diffusion_cached!(u, cache, R, n*dt)
end
```

---

## See also

- [Surface diffusion](07_surface_diffusion.md)
- [Advection–diffusion IMEX](09_advection_diffusion.md)
- [Caching and performance](15_caching.md)
