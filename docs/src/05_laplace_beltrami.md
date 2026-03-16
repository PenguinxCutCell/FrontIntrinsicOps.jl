# Laplace–Beltrami Operator

## Overview

The **Laplace–Beltrami operator** $\Delta_\Gamma$ is the intrinsic Laplacian
on a Riemannian manifold $\Gamma$.  In local coordinates it generalises the
flat Laplacian $\Delta = \sum_i \partial^2/\partial x_i^2$ to curved surfaces.

FrontIntrinsicOps.jl implements two numerically equivalent discretisations:
the **DEC factored form** (default) and the **direct cotan form**.

**Sign convention throughout this package:**

$$L = -\Delta_\Gamma \quad \text{(positive semi-definite)}$$

---

## Smooth definition

On a smooth surface $\Gamma \subset \mathbb{R}^3$ with metric $g$:

$$\Delta_\Gamma u = \frac{1}{\sqrt{\det g}} \sum_{i,j} \partial_i \!\left(\sqrt{\det g}\, g^{ij} \partial_j u\right)$$

The operator is self-adjoint with respect to the $L^2(\Gamma)$ inner product:

$$\int_\Gamma (-\Delta_\Gamma u) v \, dA = \int_\Gamma \nabla_\Gamma u \cdot \nabla_\Gamma v \, dA \geq 0$$

---

## DEC factored form (`method=:dec`, default)

$$L = \star_0^{-1} d_0^\top \star_1 d_0$$

Expanded component-wise, the $i$-th row reads:

$$(Lu)_i = \frac{1}{A_i^*} \sum_{e \ni i} \sigma_{ei} w_e (d_0 u)_e
= \frac{1}{A_i^*} \sum_{j \in N(i)} w_{ij} (u_i - u_j)$$

where:
- $N(i)$ is the set of vertices adjacent to $i$,
- $w_{ij} = \frac{1}{2}(\cot\alpha_{ij} + \cot\beta_{ij})$ is the cotan weight
  for edge $(i,j)$,
- $A_i^*$ is the dual area (from `geom.vertex_dual_areas`).

**Matrix form:**

$$L = \underbrace{\mathrm{diag}(1/A_i^*)}_{\star_0^{-1}} \underbrace{d_0^\top \star_1 d_0}_{\text{symmetric}}$$

This factored form is numerically stable and directly reflects the DEC
cochain structure.

---

## Direct cotan form (`method=:cotan`)

Assembled by looping over triangles and distributing cotan contributions to
edges:

$$(L u)_i = \frac{1}{A_i^*} \sum_{j \in N(i)} w_{ij} (u_i - u_j)$$

This is algebraically identical to the DEC form.  On well-shaped meshes
$\|L_{\text{dec}} - L_{\text{cotan}}\|_\infty < 10^{-12}$.

**Usage:**

```julia
L_dec   = build_laplace_beltrami(mesh, geom; method=:dec)    # default
L_cotan = build_laplace_beltrami(mesh, geom; method=:cotan)

report = compare_laplace_methods(mesh, geom)
println(report.norm_inf)   # should be ~1e-14 on good meshes
```

---

## Properties of $L$

| Property | Statement |
|----------|-----------|
| **Symmetry** | $L^\top = L$ |
| **Positive semi-definite** | $u^\top L u \geq 0$ for all $u$ |
| **Constant nullspace** | $L \mathbf{1} = 0$; $\ker L = \mathrm{span}\{\mathbf{1}\}$ on connected closed surfaces |
| **Row-sum zero** | $\sum_j L_{ij} = 0$ for all $i$ |
| **Sparse** | $\mathrm{nnz}(L) = N_V + 2 N_E$ (diagonal + off-diagonal pairs) |

---

## Spectrum on the sphere

On a sphere of radius $R$ the exact eigenvalues of $-\Delta_\Gamma$ are:

$$\lambda_\ell = \frac{\ell(\ell+1)}{R^2}, \quad \ell = 0, 1, 2, \ldots$$

with multiplicity $2\ell+1$.  In particular:
- $\ell=0$: $\lambda_0 = 0$ (constant functions, nullspace of $L$).
- $\ell=1$: $\lambda_1 = 2/R^2$.  The eigenfunctions are $x, y, z$
  (coordinate functions).

**Discrete verification:**

```julia
R    = 1.0
mesh = generate_icosphere(R, 4)
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)

x = [p[1] for p in mesh.points]
Lx = dec.laplacian * x
err = maximum(abs, Lx .- (2/R^2) .* x)
println("Eigenvalue error = $err")   # → ~1e-3 at level 4, → 0 as level → ∞
```

---

## Poisson equation

$$L u = f, \quad \text{with } \int_\Gamma f \, dA = 0 \text{ (compatibility)}$$

On a closed surface $L$ is singular (constant nullspace).  Two regularization
strategies are implemented in `solve_surface_poisson`:

1. **Tikhonov (`gauge=:zero_mean`):** Solve $(L + \varepsilon I) u = f$, then
   project $u \leftarrow u - \bar u$.  Default $\varepsilon = 10^{-10}$.

2. **Zero mean projection:** enforce $\mathbf{1}^\top \star_0 u = 0$.

```julia
u = solve_surface_poisson(mesh, geom, dec, f; gauge=:zero_mean, reg=1e-10)
```

---

## Helmholtz equation

$$(L + \alpha M) u = f, \quad \alpha > 0$$

where $M = \star_0$ is the mass matrix.  For $\alpha > 0$ the system is
**non-singular** even on closed surfaces.  The Helmholtz solve is the core
building block for implicit diffusion and reaction–diffusion.

```julia
u = solve_surface_helmholtz(mesh, geom, dec, f, α)
```

---

## Backward-Euler diffusion

$$\frac{du}{dt} = -\mu L u \implies (I + dt\,\mu\,L) u^{n+1} = u^n$$

```julia
u, fac = step_surface_diffusion_backward_euler(mesh, geom, dec, u, dt, μ)
# Reuse factorization on subsequent steps:
for _ in 2:100
    u, _ = step_surface_diffusion_backward_euler(mesh, geom, dec, u, dt, μ;
                                                  factorization=fac)
end
```

---

## Crank–Nicolson diffusion (2nd order)

$$\left(I + \frac{dt\,\mu}{2} L\right) u^{n+1} = \left(I - \frac{dt\,\mu}{2} L\right) u^n$$

```julia
u, fac = step_surface_diffusion_crank_nicolson(mesh, geom, dec, u, dt, μ)
```

---

## See also

- [Discrete exterior calculus](04_dec.md) — derivation of $L$ from DEC operators
- [Surface diffusion, Poisson, Helmholtz](07_surface_diffusion.md) — full solver details
- [Curvature](06_curvature.md) — mean curvature via $L$ applied to coordinate functions
