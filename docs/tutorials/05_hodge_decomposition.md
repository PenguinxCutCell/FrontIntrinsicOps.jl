# Tutorial 5: Hodge Decomposition

This tutorial demonstrates the Hodge–Helmholtz decomposition of a 1-form
on the sphere (genus 0) and the torus (genus 1).

## Background

Every 1-form $\alpha$ on a closed Riemannian manifold decomposes uniquely as:

$$\alpha = d\phi + \delta\psi + h$$

- **Exact part** $d\phi$: gradient of a scalar potential $\phi$.
- **Coexact part** $\delta\psi$: codifferential of a stream potential $\psi$.
- **Harmonic part** $h$: topologically non-trivial, exists only on surfaces with genus $g \geq 1$.

---

## Example 1: Sphere (genus 0, no harmonic part)

On a simply-connected surface all 1-forms decompose into exact and coexact
components only; the harmonic part vanishes.

```julia
using FrontIntrinsicOps

R    = 1.0
mesh = generate_icosphere(R, 4)
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)

# Build an exact 1-form: α = d(z) = gradient of the height function
u = [p[3] for p in mesh.points]   # z coordinate
α = dec.d0 * u                     # α ∈ Ω¹ (exact by construction)

# Hodge decomposition
result = hodge_decompose_1form(mesh, geom, dec, α)

println("‖α_coexact‖ = ", norm(result.coexact))   # ≈ 0 (α is exact)
println("‖α_harmonic‖ = ", norm(result.harmonic))  # ≈ 0 (sphere, genus 0)
println("Residual = ", hodge_decomposition_residual(mesh, geom, dec, α, result))

# Verify α_exact ≈ α
err = norm(result.exact .- α) / norm(α)
println("Relative error ‖α_exact - α‖ / ‖α‖ = $err")   # ≈ 0
```

---

## Example 2: Non-exact 1-form on the sphere

Build a coexact 1-form: $\alpha = \delta(z_f)$ (the codifferential of a
face 2-form).

```julia
# Build a face 2-form: ψ_f = area_f (or any face scalar field)
z_face = [mean(mesh.points[v][3] for v in f) for f in mesh.faces]
ψ = z_face   # face 2-form

# Compute codifferential: δ₂ ψ = ⋆₁⁻¹ d₁ᵀ ⋆₂ ψ
star1_inv = spdiagm(0 => 1.0 ./ diag(dec.star1))
star2     = dec.star2
coexact_alpha = star1_inv * dec.d1' * star2 * ψ

result2 = hodge_decompose_1form(mesh, geom, dec, coexact_alpha)

println("‖α_exact‖   = ", norm(result2.exact))   # ≈ 0 (coexact form)
println("‖α_coexact‖ = ", norm(result2.coexact))  # ≈ ‖coexact_alpha‖
println("‖α_harmonic‖ = ", norm(result2.harmonic)) # ≈ 0
```

---

## Example 3: Torus (genus 1, non-trivial harmonic part)

The torus has genus 1 and first Betti number $b_1 = 2$.  There exist two
linearly independent harmonic 1-forms dual to the toroidal and poloidal cycles.

```julia
# Generate a torus
mesh_t = generate_torus(3.0, 1.0, 60, 30)   # R=3, r=1
geom_t = compute_geometry(mesh_t)
dec_t  = build_dec(mesh_t, geom_t)

println("χ = ", euler_characteristic(mesh_t))   # = 0 (torus)
```

### Build a 1-form that wraps around the torus

The toroidal angle $\theta$ satisfies $d\theta \neq 0$ globally but is not
exact (not the differential of a global single-valued function).

```julia
# Toroidal coordinate θ = atan(y, x): wraps around the torus once
θ_vertex = [atan(p[2], p[1]) for p in mesh_t.points]
# d(θ) as a 1-form (note: atan is multi-valued — treat it as approximate)
dθ = dec_t.d0 * θ_vertex   # discontinuous across the branch cut

# Decompose
result_t = hodge_decompose_1form(mesh_t, geom_t, dec_t, dθ)

println("‖α_harmonic‖ = ", norm(result_t.harmonic))  # > 0 on the torus
println("‖α_exact‖    = ", norm(result_t.exact))
println("‖α_coexact‖  = ", norm(result_t.coexact))
```

The harmonic component captures the topological winding of $d\theta$ around
the torus.

---

## Orthogonality check

The three components should be mutually orthogonal in the $L^2$ inner product
$\langle \alpha, \beta \rangle_{\star_1} = \alpha^\top \star_1 \beta$:

```julia
ips = hodge_inner_products(mesh, geom, dec, result)
println("⟨exact, coexact⟩  = ", ips.exact_coexact)     # ≈ 0
println("⟨exact, harmonic⟩ = ", ips.exact_harmonic)    # ≈ 0
println("⟨coexact, harmonic⟩ = ", ips.coexact_harmonic) # ≈ 0
```

---

## Recovering the potentials

The scalar potential $\phi$ solves $L\phi = \delta\alpha$ and the stream
function $\psi$ solves $\tilde{L}\psi = d\alpha$:

```julia
φ = result.phi    # scalar potential: ∇φ ≈ α_exact
ψ = result.psi    # stream potential: rot(ψ) ≈ α_coexact

# The potential φ is defined up to a constant
φ_normalized = φ .- sum(geom.vertex_dual_areas .* φ) / measure(mesh, geom)
```

---

## See also

- [Math: Hodge decomposition](../math/12_hodge_decomposition.md)
- [Math: Discrete exterior calculus](../math/04_dec.md)
- [Math: Tangential vector calculus](../math/11_vector_calculus.md)
- [API: PDE solvers](../api/pdes.md)
