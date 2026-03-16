# Scalar Transport on Surfaces

## Overview

FrontIntrinsicOps.jl implements **conservative scalar transport** on triangulated
surfaces.  The key design choices are:

- A **DEC-consistent edge-flux** formula using cotan weights.
- Three explicit time integrators: Forward Euler, SSP-RK2, SSP-RK3.
- Two spatial schemes: upwind (1st order) and centered (2nd order).
- A CFL-based time-step estimator.

---

## The transport PDE

$$\frac{\partial u}{\partial t} + \nabla_\Gamma \cdot (v u) = 0, \quad (x, t) \in \Gamma \times [0, T]$$

where $v : \Gamma \to T\Gamma$ is a prescribed tangential velocity field and
$\nabla_\Gamma \cdot$ is the surface divergence.

**Conservation:** On a closed surface $\int_\Gamma u \, dA$ is conserved
provided $\nabla_\Gamma \cdot v = 0$ (divergence-free velocity).

---

## Tangential projection

A velocity field $v : V \to \mathbb{R}^3$ given in ambient space is projected
to the tangent plane at each vertex before use:

$$v_i^\tau = v_i - (v_i \cdot \hat{n}_i) \hat{n}_i$$

```julia
vel_tang = tangential_projection(mesh, geom, vel)
```

---

## Edge-flux velocity

The scalar flux through each edge is computed from the vertex velocities.
For edge $e = (i, j)$ with unit tangent $\hat{t}_e$:

$$v_e = \frac{v_i + v_j}{2} \cdot \hat{t}_e$$

This is the **average** of the two vertex velocities projected onto the edge
tangent direction.

```julia
ve = edge_flux_velocity(mesh, geom, vel)   # Vector{T}, length N_E
```

---

## Transport operator — semi-discrete form

The semi-discrete equation is:

$$M \frac{du}{dt} + A u = 0$$

where $M = \star_0$ is the mass matrix and $A$ is the **transport operator**
assembled as a sparse $N_V \times N_V$ matrix.

The transport operator uses the DEC codifferential structure.  For each vertex $i$:

$$(Au)_i = \sum_{e \ni i} \sigma_{ei} \cdot w_e \cdot \ell_e \cdot v_e \cdot u_e^{\text{scheme}}$$

where:
- $\sigma_{ei} \in \{+1, -1\}$ is the incidence sign of $d_0$,
- $w_e = \frac{1}{2}(\cot\alpha_e + \cot\beta_e)$ is the ⋆₁ cotan weight,
- $\ell_e$ is the primal edge length,
- $v_e$ is the edge-flux velocity,
- $u_e^{\text{scheme}}$ is the upwinded or centered value on edge $e$.

The product $w_e \ell_e$ approximates the **dual-edge length** on a Voronoi
dual mesh.  This makes the formula geometrically consistent with the
Laplace–Beltrami operator.

### Upwind scheme

$$u_e^{\text{up}} = \begin{cases} u_i & \text{if } v_e > 0 \\ u_j & \text{if } v_e < 0 \end{cases}$$

(donor-cell value: upwind vertex supplies the flux)

- Order: 1st order in $h$.
- Numerical diffusion: $\varepsilon_{\text{num}} \approx |v| h / 2$.
- Monotone: no spurious oscillations.

### Centered scheme

$$u_e^{\text{centered}} = \frac{u_i + u_j}{2}$$

- Order: 2nd order in $h$.
- Not monotone: can produce small oscillations near steep gradients.

```julia
A = assemble_transport_operator(mesh, geom, vel; scheme=:upwind)   # or :centered
```

---

## CFL time-step estimate

The CFL condition for explicit time integration:

$$dt \leq C_{\text{CFL}} \cdot \min_i \frac{A_i^*}{\displaystyle\sum_{e \ni i} w_e |v_e| \ell_e}$$

```julia
dt = estimate_transport_dt(mesh, geom, vel; cfl=0.5)   # cfl ∈ (0, 1]
```

---

## Time integrators

### Forward Euler (reference, first order)

$$u^{n+1} = u^n - dt\, M^{-1} A u^n$$

Conditionally stable for $dt \leq dt_{\text{CFL}}$.  Not recommended for
production; use SSP-RK2 or SSP-RK3 instead.

```julia
u_new = step_surface_transport_forward_euler(mesh, geom, A, u, dt)
```

### SSP-RK2 (Shu–Osher, second order in time)

$$u^{(1)} = u^n - dt\, M^{-1} A u^n$$
$$u^{n+1} = \frac{1}{2} u^n + \frac{1}{2} u^{(1)} - \frac{dt}{2} M^{-1} A u^{(1)}$$

**Strong-stability-preserving:** All intermediate stages are convex combinations
of Euler steps, so any monotone property preserved by Forward Euler is also
preserved by SSP-RK2.

```julia
u_new = step_surface_transport_ssprk2(mesh, geom, A, u, dt)
```

### SSP-RK3 (Gottlieb–Shu, third order in time)

$$u^{(1)} = u^n - dt\, M^{-1} A u^n$$
$$u^{(2)} = \frac{3}{4} u^n + \frac{1}{4} u^{(1)} - \frac{dt}{4} M^{-1} A u^{(1)}$$
$$u^{n+1} = \frac{1}{3} u^n + \frac{2}{3} u^{(2)} - \frac{2dt}{3} M^{-1} A u^{(2)}$$

```julia
u_new = step_surface_transport_ssprk3(mesh, geom, A, u, dt)
```

---

## Convergence rates

On the unit sphere, transporting $u_0(x,y,z) = z$ by the rigid-rotation
velocity $v = (-y, x, 0)$ for time $T = 0.5$:

| Scheme | Temporal | Observed spatial $O(h^p)$ |
|--------|---------|---------------------------|
| Upwind + SSP-RK3 | $O(dt^3)$ | $p \approx 1$ |
| Centered + SSP-RK3 | $O(dt^3)$ | $p \approx 2$ |

The centered scheme matches the $O(h^2)$ accuracy of the Laplace–Beltrami
operator; the upwind scheme is limited to $O(h^1)$ due to numerical diffusion.

---

## Mass conservation

On a **closed surface** with a **divergence-free** velocity:

$$\frac{d}{dt} \int_\Gamma u \, dA = -\int_\Gamma \nabla_\Gamma \cdot (vu) \, dA = 0$$

Discrete conservation:

$$\frac{d}{dt} \mathbf{1}^\top M u = -\mathbf{1}^\top A u = 0$$

This holds because $\mathbf{1}^\top A = 0$ for a conservative, divergence-free
discrete operator.  Monitor conservation as:

```julia
mass = sum(geom.vertex_dual_areas .* u)   # should be constant
```

---

## See also

- [Advection–diffusion IMEX](09_advection_diffusion.md)
- [High-resolution transport](13_highres_transport.md)
- [Tangential vector calculus](11_vector_calculus.md)
