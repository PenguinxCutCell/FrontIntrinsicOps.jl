# Advection–Diffusion IMEX

## Overview

The **advection–diffusion equation** on a surface combines transport by a
prescribed velocity field with diffusive smoothing:

$$\frac{\partial u}{\partial t} + \nabla_\Gamma \cdot (v u) = \mu \Delta_\Gamma u + g$$

FrontIntrinsicOps.jl solves this with an **IMEX splitting** (IMplicit–EXplicit):
transport is treated **explicitly** (cheap, easy to upwind) while diffusion is
treated **implicitly** (unconditionally stable for any $\mu dt$).

---

## IMEX splitting

Let:
- $A$ = transport operator (see [Scalar transport](08_transport.md))
- $L$ = Laplace–Beltrami operator (see [Laplace–Beltrami](05_laplace_beltrami.md))
- $M = \star_0$ = mass matrix

The semi-discrete equation is:

$$M \frac{du}{dt} + A u = \mu\, L_M u + M g, \quad L_M = M L$$

Wait — more precisely the strong form after dividing by $M$:

$$\frac{du}{dt} = -M^{-1} A u - \mu L u + g$$

**One IMEX step** (explicit transport, implicit diffusion):

$$(M + dt\,\mu\,L) u^{n+1} = M u^n - dt\, A u^n + dt\, M g^n$$

which can be rewritten as:

$$(I + dt\,\mu\,M^{-1} L) u^{n+1} = u^n - dt\, M^{-1} A u^n + dt\, g^n$$

---

## Stability analysis

For purely diffusive ($A = 0$) the implicit step is unconditionally stable.

For purely transport ($\mu = 0$) the explicit step requires the CFL condition
(see [Transport](08_transport.md)).

For the combined system the **transport CFL** dominates:

$$dt \leq dt_{\text{CFL}} = C_{\text{CFL}} \cdot \min_i \frac{A_i^*}{\sum_{e \ni i} w_e |v_e| \ell_e}$$

The diffusion term is always stable for any $dt$ once it is treated implicitly.

---

## Factorization reuse

The matrix $(M + dt\,\mu\,L)$ depends only on the constant parameters
$dt$ and $\mu$.  Pre-factorising it once and reusing across time steps
gives a large speedup:

```julia
# First step: assembles and factorises (expensive)
u, fac = step_surface_advection_diffusion_imex(mesh, geom, dec, u, vel, dt, μ;
                                                scheme=:upwind)

# Subsequent steps: reuses factorisation (cheap)
A = assemble_transport_operator(mesh, geom, vel; scheme=:upwind)
for _ in 2:N
    u, _ = step_surface_advection_diffusion_imex(mesh, geom, dec, u, vel, dt, μ;
                                                  scheme=:upwind,
                                                  transport_operator = A,
                                                  factorization      = fac)
end
```

---

## Fully implicit (backward Euler) variant

An alternative fully-implicit step treats both transport and diffusion
implicitly:

$$(M + dt\,A + dt\,\mu\,L) u^{n+1} = M u^n + dt\, M g^n$$

This is more expensive per step (larger solve) and the transport operator $A$
may not be symmetric.  Use when the velocity is slow relative to diffusion.

```julia
u = step_surface_advection_diffusion_backward_euler(mesh, geom, dec, u, vel, dt, μ)
```

---

## Convergence rates

On the unit sphere, transporting $u_0 = z$ with $v = (-y, x, 0)$ and
$\mu = 0.01$:

| Transport scheme | Spatial rate |
|------------------|-------------|
| Upwind | $O(h^1)$ — first-order numerical diffusion |
| Centered | $O(h^2)$ — matches Laplace–Beltrami accuracy |

For fixed coarse mesh and varying $dt$ (backward Euler in time):

| Method | Time rate |
|--------|-----------|
| IMEX (transport explicit, diffusion implicit) | $O(dt^1)$ |

---

## Assembling the transport operator once

```julia
# Pre-assemble A (constant velocity)
A = assemble_transport_operator(mesh, geom, vel; scheme=:upwind)

# CFL time step
dt = estimate_transport_dt(mesh, geom, vel; cfl=0.3)

# Integrate
u, fac = step_surface_advection_diffusion_imex(mesh, geom, dec, u, vel, dt, μ;
                                                scheme=:upwind,
                                                transport_operator=A)
```

---

## See also

- [Scalar transport](08_transport.md)
- [Surface diffusion](07_surface_diffusion.md)
- [Reaction–diffusion IMEX](10_reaction_diffusion.md)
