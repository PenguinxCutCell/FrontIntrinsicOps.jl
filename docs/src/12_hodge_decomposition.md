# Hodge Decomposition of 1-Forms

## Overview

The **Hodge–Helmholtz decomposition** (or Hodge decomposition) states that
every smooth 1-form $\alpha$ on a compact Riemannian manifold can be uniquely
decomposed as:

$$\alpha = d\phi + \delta\psi + h$$

where:
- $d\phi = \nabla_\Gamma \phi$ is the **exact component** (gradient of a potential $\phi$),
- $\delta\psi = \star d \star \psi$ is the **coexact component** (codifferential of a potential $\psi$),
- $h$ is the **harmonic component** ($dh = 0$, $\delta h = 0$).

The dimension of the harmonic space equals the **first Betti number** $b_1 = 2g$,
where $g$ is the genus of the surface:

| Surface | Genus $g$ | $b_1$ | Harmonic forms |
|---------|-----------|-------|----------------|
| Sphere | 0 | 0 | None (trivial harmonic space) |
| Torus | 1 | 2 | Two independent harmonic 1-forms |
| Genus-$g$ surface | $g$ | $2g$ | $2g$ independent harmonic 1-forms |

---

## Discrete Hodge decomposition

### Step 1: Exact component

Solve the scalar Poisson equation for $\phi$:

$$L \phi = \delta_1 \alpha = \star_0^{-1} d_0^\top \star_1 \alpha$$

(the codifferential of $\alpha$), then set $\alpha_{\text{exact}} = d_0 \phi$.

On a closed surface $L$ is singular; the Tikhonov regularisation
$(L + \varepsilon I) \phi = \delta_1 \alpha$ is used.

### Step 2: Coexact component

Solve the **dual Poisson equation** for $\psi$:

$$\tilde{L} \psi = d_1 \alpha$$

where $\tilde{L} = \star_2^{-1} d_1 \star_1^{-1} d_1^\top \star_2$ is the
Laplacian on 2-forms (equivalently, on dual 0-cochains).

Set $\alpha_{\text{coexact}} = \delta_2 (\star_2 \psi) = \star_1^{-1} d_1^\top \star_2 \psi$.

### Step 3: Harmonic residual

$$h = \alpha - \alpha_{\text{exact}} - \alpha_{\text{coexact}}$$

For a **genus-0** surface (sphere) $h \approx 0$ (up to numerical error).
For a **torus** $h$ captures the two independent topological cycles.

---

## API

```julia
# Full decomposition
result = hodge_decompose_1form(mesh, geom, dec, α; reg=1e-10)
# result.exact       Vector{T}  — α_exact  = d0 φ
# result.coexact     Vector{T}  — α_coexact = δ₂(⋆₂ ψ)
# result.harmonic    Vector{T}  — h = α - α_exact - α_coexact
# result.phi         Vector{T}  — scalar potential φ
# result.psi         Vector{T}  — stream potential ψ

# Individual components
α_exact,   φ = exact_component_1form(mesh, geom, dec, α)
α_coexact, ψ = coexact_component_1form(mesh, geom, dec, α)
h              = harmonic_component_1form(mesh, geom, dec, α)
```

---

## Diagnostics

### Reconstruction residual

$$r = \frac{\|\alpha - \alpha_{\text{exact}} - \alpha_{\text{coexact}} - h\|_{\star_1}}{\|\alpha\|_{\star_1}}$$

where the $\star_1$-norm is $\|\cdot\|_{\star_1}^2 = \alpha^\top \star_1 \alpha$.

```julia
r = hodge_decomposition_residual(mesh, geom, dec, α, result)
# r ≈ 0 (machine precision) by definition
```

### Orthogonality check

The three components should be mutually orthogonal in $L^2$:

$$\langle \alpha_{\text{exact}},\, \alpha_{\text{coexact}} \rangle_{\star_1} \approx 0$$
$$\langle \alpha_{\text{exact}},\, h \rangle_{\star_1} \approx 0$$
$$\langle \alpha_{\text{coexact}},\, h \rangle_{\star_1} \approx 0$$

```julia
ips = hodge_inner_products(mesh, geom, dec, result)
# ips.exact_coexact   ≈ 0
# ips.exact_harmonic  ≈ 0
# ips.coexact_harmonic ≈ 0
```

---

## Physical interpretation

| Component | Geometry | Physics |
|-----------|----------|---------|
| $\alpha_{\text{exact}} = d\phi$ | Gradient field, curl-free | Conservative force, electric field |
| $\alpha_{\text{coexact}} = \delta\psi$ | Divergence-free, no potential | Magnetic field, solenoidal flow |
| $h$ (harmonic) | Both curl-free and divergence-free | Topological obstruction, DC current on a ring |

---

## Example: sphere (genus 0)

For a **gradient 1-form** $\alpha = d_0 u$ (exact by construction):

- $\alpha_{\text{exact}} \approx \alpha$
- $\alpha_{\text{coexact}} \approx 0$
- $h \approx 0$

```julia
u = [p[3] for p in mesh.points]   # height function z
α = dec.d0 * u                    # exact 1-form

result = hodge_decompose_1form(mesh, geom, dec, α)
println(norm(result.coexact))    # → ~1e-12
println(norm(result.harmonic))   # → ~1e-12
```

---

## Example: torus (genus 1)

On the torus there exist **two linearly independent harmonic 1-forms** dual to
the two fundamental cycles (meridional circle and longitudinal circle).

For a non-exact 1-form on the torus:

```julia
# Build a 1-form that winds around the torus once (not exact)
α = build_winding_oneform(mesh)   # custom test form

result = hodge_decompose_1form(mesh, geom, dec, α)
println(norm(result.harmonic))   # → significant — the harmonic component
                                   # captures the topological winding
```

The harmonic part has the same $L^2$ norm independent of the particular
representative chosen in the cohomology class.

---

## Potential recovery

The scalar potential $\phi$ satisfies:

$$\nabla_\Gamma \phi = \alpha_{\text{exact}}$$

and the stream potential $\psi$ satisfies:

$$\mathrm{rot}_\Gamma \psi \approx \alpha_{\text{coexact}}$$

These potentials can be used to integrate trajectories, compute stream lines,
or reconstruct the original 1-form:

```julia
φ = result.phi   # scalar potential (vertex field)
ψ = result.psi   # stream function (vertex field)
```

---

## References

- Desbrun, M., Kanso, E., & Tong, Y. (2008). *Discrete differential forms for
  computational modeling.* In Discrete Differential Geometry, pp. 287–324.
- Gu, X., & Yau, S.T. (2003). *Global conformal surface parameterization.*
  Proc. Eurographics Symposium on Geometry Processing.
- Hirani, A.N. (2003). *Discrete exterior calculus.* Ph.D. thesis, Caltech.

---

## See also

- [Discrete exterior calculus](04_dec.md)
- [Tangential vector calculus](11_vector_calculus.md)
