# High-Resolution Transport with Flux Limiters

## Overview

The standard upwind scheme for scalar transport is only **first-order accurate**
and introduces significant **numerical diffusion**.  High-resolution (HR) schemes
combine upwind stability with near-second-order accuracy by using **flux limiters**
that smoothly blend between first- and second-order approximations.

FrontIntrinsicOps.jl implements the **minmod**, **van Leer**, and **superbee**
limiters, together with an SSP-RK2 time integrator.

---

## Flux limiter theory

### The limited flux

For edge $e = (i, j)$ with upstream vertex $u$ and downstream vertex $d$:

$$F_e = v_e \left( u_{\text{up}} + \frac{1}{2} \phi(r) (u_{\text{down}} - u_{\text{up}}) \right)$$

where:
- $u_{\text{up}}$ is the value at the upwind vertex,
- $\phi(r)$ is the limiter function,
- $r$ is the **slope ratio**:

$$r = \frac{u_{\text{up}} - u_{\text{far-upwind}}}{u_{\text{down}} - u_{\text{up}} + \varepsilon}$$

Here $u_{\text{far-upwind}}$ is the second upwind neighbor (one step further
upstream).

When $\phi(r) = 0$ the flux is pure upwind (first order).
When $\phi(r) = 1$ the flux is centered (second order, potentially oscillatory).

**TVD condition:** A limiter is **total variation diminishing** if $\phi$
lies in the Sweby region:

$$0 \leq \phi(r) \leq \min(2r, 2), \quad 0 \leq \phi(r) \leq 2$$

---

## Minmod limiter

$$\phi_{\text{mm}}(a, b) = \mathrm{sign}(a) \cdot \max(0, \min(|a|, \, \mathrm{sign}(a) b))$$

Equivalently: if $a$ and $b$ have the same sign, return the smaller-magnitude
value; otherwise return 0.

$$\phi_{\text{mm}}(r) = \max(0, \min(1, r))$$

- Most diffusive of the three limiters.
- Exactly one-sided near extrema.
- Provably TVD.

```julia
φ = minmod(a, b)     # two-argument form
φ = minmod3(a, b, c) # three-argument form: same-sign minimum
```

---

## Van Leer limiter

$$\phi_{\text{vL}}(r) = \frac{r + |r|}{1 + |r|}$$

- **Smooth** at all $r$ (unlike minmod which has a kink at $r=1$).
- Second-order at smooth extrema (unlike minmod which drops to first-order).
- TVD.
- Recommended for smooth solutions.

```julia
φ = vanleer_limiter(r)
```

---

## Superbee limiter

$$\phi_{\text{sb}}(r) = \max(0,\; \min(2r, 1),\; \min(r, 2))$$

- Most compressive of the three (closest to second-order everywhere).
- Can over-sharpen near genuine extrema.
- TVD but at the upper boundary of the Sweby region.
- Recommended when the solution is expected to have sharp but smooth fronts.

```julia
φ = superbee_limiter(r)
```

---

## Assembling the limited transport operator

```julia
A_limited = assemble_transport_operator_limited(mesh, geom, topo, vel, u;
                                                 limiter=:minmod)    # :minmod, :van_leer, :superbee
```

Note that unlike the standard upwind operator, the limited operator **depends
on the current solution** $u$ (through the slope ratio $r$).  It must be
reassembled at each time step.

---

## SSP-RK2 time integration

The SSP-RK2 scheme is used for its **strong-stability-preserving** properties.
Since the limited operator must be reassembled at each stage, two operator
assemblies per step are required:

**Stage 1:**

$$u^{(1)} = u^n - dt\, M^{-1} A(u^n) u^n$$

**Stage 2 (reassemble with $u^{(1)}$):**

$$u^{n+1} = \frac{1}{2} u^n + \frac{1}{2} u^{(1)} - \frac{dt}{2} M^{-1} A(u^{(1)}) u^{(1)}$$

```julia
u_new = step_surface_transport_limited(mesh, geom, dec, topo, u, vel, dt;
                                        limiter=:van_leer, method=:ssprk2)
```

Set `method=:euler` for a single forward-Euler stage (cheaper but less accurate).

---

## Comparison of limiters

| Limiter | Accuracy | Sharpness | Recommended for |
|---------|----------|-----------|-----------------|
| None (upwind) | $O(h)$ | High diffusion | Robustness only |
| Minmod | $O(h^2)$ near smooth regions | Moderate | General use |
| Van Leer | $O(h^2)$ smooth | High | Smooth solutions |
| Superbee | $O(h^2)$ aggressive | Highest | Sharp fronts |
| None (centered) | $O(h^2)$ | Oscillatory | Combined with diffusion only |

---

## CFL condition

The CFL condition for the limited scheme is the same as for the standard
upwind scheme:

$$dt \leq C_{\text{CFL}} \cdot \min_i \frac{A_i^*}{\sum_{e \ni i} w_e |v_e| \ell_e}$$

For SSP-RK2, the effective CFL coefficient can be taken up to 1.0; typical
practice is $C_{\text{CFL}} = 0.5$.

---

## Conservation

The limited scheme is **locally conservative**: the flux through each edge is
antisymmetric (what leaves vertex $i$ enters vertex $j$).  Global conservation:

$$\sum_i A_i^* u_i^{n+1} = \sum_i A_i^* u_i^n$$

holds to machine precision for divergence-free velocity fields.

---

## See also

- [Scalar transport](08_transport.md) — standard upwind and centered schemes
- [Advection–diffusion IMEX](09_advection_diffusion.md)
