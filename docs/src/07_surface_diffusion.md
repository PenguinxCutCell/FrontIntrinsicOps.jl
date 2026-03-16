# Surface Diffusion, Poisson, and Helmholtz Solvers

## Overview

This module provides implicit time-integration schemes for the surface **heat
equation**, the **Poisson equation**, and the **Helmholtz equation** on
triangulated surfaces.  All solvers share the same Laplace–Beltrami matrix
$L$ and use sparse direct factorization (SuiteSparse / Julia's `\`).

---

## Mass matrix

The (lumped) mass matrix is the Hodge star $\star_0$:

$$M = \star_0 = \mathrm{diag}(A_1^*, A_2^*, \ldots, A_{N_V}^*)$$

```julia
M = mass_matrix(mesh, geom)         # diagonal sparse
m = lumped_mass_vector(mesh, geom)  # just the diagonal vector
```

---

## Weighted mean and compatibility

For the Poisson equation on a closed surface the right-hand side must satisfy
the **compatibility condition** $\int_\Gamma f \, dA = 0$.

**Weighted mean:**

$$\bar{u} = \frac{\sum_i u_i A_i^*}{\sum_i A_i^*}$$

**Zero-mean projection:**

$$u \leftarrow u - \bar{u}$$

```julia
enforce_compatibility!(f, mesh, geom)    # ensures ∫ f dA = 0
zero_mean_projection!(u, mesh, geom)     # subtracts mean
```

---

## Surface Poisson equation

$$-\Delta_\Gamma u = f \quad \Leftrightarrow \quad L u = f$$

on a closed surface with the constraint $\int_\Gamma u \, dA = 0$.

**Solvability:** $L$ is singular (constant nullspace) on a closed surface.
Two regularisation strategies:

1. **Tikhonov shift** (`gauge=:zero_mean`, default):
   Solve $(L + \varepsilon I) u = f$, then project $u \leftarrow u - \bar{u}$.

2. **Pin a DOF** (`gauge=:pin`): Set $u[1] = 0$ and solve the reduced system.

```julia
u = solve_surface_poisson(mesh, geom, dec, f; gauge=:zero_mean, reg=1e-10)
```

**Convergence:** Second-order in the mesh size $h$ for smooth $f$ and smooth
surfaces (consistent with cotan-Laplace accuracy).

---

## Surface Helmholtz equation

$$(L + \alpha M) u = f, \quad \alpha > 0$$

For $\alpha > 0$ the matrix is **positive definite** even on closed surfaces —
no gauge is needed.

```julia
u = solve_surface_helmholtz(mesh, geom, dec, f, α)
```

**Usage pattern:** Helmholtz solves appear as the linear step in implicit
diffusion, reaction–diffusion, and Hodge decomposition.

---

## Heat equation — Backward Euler (first order)

$$M \frac{du}{dt} + \mu L u = g(t)$$

One backward-Euler step (step size $dt$):

$$(M + dt\,\mu\,L) u^{n+1} = M u^n + dt\, g^n$$

which, after dividing by $M$ (diagonal):

$$(I + dt\,\mu\,M^{-1} L) u^{n+1} = u^n + dt\, M^{-1} g^n$$

In the implementation the unfactored form is used for better numerical
conditioning.

**Local truncation error:** $O(dt)$ in time.
**Stability:** Unconditionally stable for all $dt > 0$.
**Factorization reuse:** The matrix $(M + dt\,\mu\,L)$ depends only on the
constant parameters $dt$, $\mu$.  Pass the factorization back to avoid
repeated expensive factorizations:

```julia
u, fac = step_surface_diffusion_backward_euler(mesh, geom, dec, u, dt, μ)
for _ in 2:100
    u, _ = step_surface_diffusion_backward_euler(mesh, geom, dec, u, dt, μ;
                                                  factorization=fac)
end
```

---

## Heat equation — Crank–Nicolson (second order)

$$\left(M + \frac{dt\,\mu}{2} L\right) u^{n+1} = \left(M - \frac{dt\,\mu}{2} L\right) u^n + dt\,\bar{g}$$

where $\bar{g} = (g^n + g^{n+1})/2$.

**Local truncation error:** $O(dt^2)$ in time.
**Stability:** A-stable (stable for all $dt > 0$), but not L-stable
(high-frequency modes are not damped in one step).

```julia
u, fac = step_surface_diffusion_crank_nicolson(mesh, geom, dec, u, dt, μ)
```

---

## Convergence rates

On the unit sphere, decaying modes $u(t) = e^{-\lambda t} Y_\ell^m$ (spherical
harmonics of degree $\ell$):

| Method | Time order | Spatial order |
|--------|-----------|---------------|
| Backward Euler | $O(dt)$ | $O(h^2)$ |
| Crank–Nicolson | $O(dt^2)$ | $O(h^2)$ |

Both methods saturate at $O(h^2)$ spatial error, consistent with the cotan
Laplacian.

---

## Compatibility of the RHS

When solving the Poisson equation $Lu = f$ on a closed surface, the
compatibility condition $\int_\Gamma f \, dA = 0$ must hold.

```julia
enforce_compatibility!(f, mesh, geom)   # modifies f in-place
u = solve_surface_poisson(mesh, geom, dec, f)
```

If $f$ does not satisfy compatibility, the Tikhonov-regularised system gives a
solution polluted by a constant drift.

---

## See also

- [Laplace–Beltrami operator](05_laplace_beltrami.md)
- [Reaction–diffusion IMEX](10_reaction_diffusion.md)
- [Caching and performance](15_caching.md)
- [Open surfaces and BCs](14_open_surfaces.md)
