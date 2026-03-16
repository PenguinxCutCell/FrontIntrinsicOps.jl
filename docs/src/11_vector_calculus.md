# Tangential Vector Calculus

## Overview

This module provides a complete toolkit for **intrinsic vector calculus** on
triangulated surfaces.  All operations work with the tangent bundle of the
surface and are implemented using the DEC framework.

---

## Tangential projection

For a vector $v \in \mathbb{R}^3$ and a unit surface normal $\hat{n}$:

$$v^\tau = v - (v \cdot \hat{n})\, \hat{n}$$

This projects $v$ onto the tangent plane.

```julia
# Per vector
v_tang = tangential_project(v, n)   # SVector{3,T}

# Per vertex field
v_tang_field = tangential_project_field(mesh, geom, vfield; location=:vertex)
```

The `location` keyword selects vertex-based ($\hat{n}_i$) or face-based
($\hat{n}_f$) normals for the projection.

---

## Surface gradient

For a scalar vertex field $u : V \to \mathbb{R}$, the discrete **surface gradient**
at each face $f = (a, b, c)$ is computed from the Whitney 1-form formula:

$$\nabla_\Gamma u\big|_f = \frac{1}{2A_f} \sum_{v \in f} u_v \left(\hat{n}_f \times (p_{\text{opp}(v)} - p_v)\right)$$

where $p_{\text{opp}(v)}$ is the vertex of $f$ opposite to $v$, i.e. the edge not
containing $v$.  The result is a tangent vector in $\mathbb{R}^3$ perpendicular
to $\hat{n}_f$.

**Alternative formula** (equivalent, summing over edges):

For edge $e_k = (v_i, v_j)$ of face $f$, let $p_k = p_i - p_j$.  Then:

$$\nabla_\Gamma u\big|_f = \frac{1}{2A_f} \sum_k u_k \left(\hat{n}_f \times p_k^{\perp}\right)$$

This formula follows directly from differentiating the piecewise-linear
interpolant over the face.

```julia
grad_u = gradient_0_to_tangent_vectors(mesh, geom, u; location=:face)
# Returns Vector{SVector{3,T}}, one per face
```

---

## Surface divergence

For a tangential vector field $w_f \in T_f\Gamma$ (one vector per face), the
discrete **surface divergence** is computed as:

$$(\nabla_\Gamma \cdot w)_f \approx \frac{1}{A_f} \sum_{e \in \partial f} \sigma_{fe} \ell_e (w_f \cdot \hat{t}_e)$$

At vertices, the divergence is assembled by distributing face contributions
using barycentric weights.

```julia
div_w = divergence_tangent_vectors(mesh, geom, vfield; location=:face)
# Returns Vector{T}, one per face
```

---

## 1-form ↔ tangent vector conversion

### Tangent vectors → 1-form

A tangential vector field $w$ on faces is converted to a DEC edge 1-form
$\alpha \in \Omega^1$ by projecting $w$ onto each edge tangent:

$$\alpha_e = w_f \cdot \hat{t}_e \cdot \ell_e$$

where $f$ is (one of) the face(s) adjacent to $e$.

```julia
α = tangent_vectors_to_1form(mesh, geom, topo, vfield; location=:face)
# Returns Vector{T}, length N_E
```

### 1-form → tangent vectors

The inverse: reconstruct a tangential vector field from an edge 1-form.  For
face $f$:

$$w_f = \frac{1}{2A_f} \sum_{e \in \partial f} \sigma_{fe} \alpha_e \left(\hat{n}_f \times p_e\right)$$

where $p_e = p_j - p_i$ is the edge vector.

```julia
vfield = oneform_to_tangent_vectors(mesh, geom, topo, α; location=:face)
# Returns Vector{SVector{3,T}}, one per face
```

---

## Surface rot (tangential curl)

For a scalar 0-form $u$, the **surface rotation** (or "surface curl" of a
scalar) is:

$$\mathrm{rot}_\Gamma u = \hat{n} \times \nabla_\Gamma u$$

This produces a tangential vector field perpendicular to $\nabla_\Gamma u$.
On a simply-connected surface $\mathrm{rot}_\Gamma u$ is divergence-free.

```julia
rot_u = surface_rot_0form(mesh, geom, u)   # Vector{SVector{3,T}}
```

**Application:** Given a stream function $\psi$, the velocity field
$v = \mathrm{rot}_\Gamma \psi$ is automatically tangential and
divergence-free — useful for constructing test velocity fields.

---

## Mathematical consistency

The vector calculus operations are consistent with the DEC operators in the
following sense:

1. **Gradient–1-form duality:**
   `tangent_vectors_to_1form(gradient_0_to_tangent_vectors(u))` should
   approximate `d0 * u` (the DEC gradient).

2. **Divergence–codifferential duality:**
   `divergence_1_to_0(d0 * u)` = `L * u` (Laplacian from DEC).

3. **Green's identity:**
   $\int_\Gamma \nabla_\Gamma u \cdot \nabla_\Gamma v \, dA = u^\top L v$
   (exact for piecewise-linear $u$, $v$).

---

## Orthogonality check

After computing a tangent-vector field $w$, verify tangency:

```julia
for i in eachindex(w)
    @assert abs(dot(w[i], geom.vertex_normals[i])) < 1e-10
end
```

The `tangential_project_field` function ensures this by construction.

---

## See also

- [Discrete exterior calculus](04_dec.md) — DEC gradient and divergence
- [Hodge decomposition](12_hodge_decomposition.md) — uses 1-forms and codifferentials
- [High-resolution transport](13_highres_transport.md) — velocity fields as tangent vectors
