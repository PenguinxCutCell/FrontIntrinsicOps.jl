# hodge_decomposition.jl – Hodge decomposition for discrete 1-forms.
#
# Given a discrete 1-form α on a closed triangulated surface, the Hodge
# decomposition reads:
#
#   α = dφ + δβ + h
#
# where:
#   dφ  is exact  (φ is a 0-form scalar potential),
#   δβ  is coexact (β is a 2-form scalar potential),
#   h   is harmonic (d h = 0, δ h = 0).
#
# Discrete implementation
# -----------------------
# The exact component dφ is found by solving the 0-form Poisson problem:
#   Δ₀ φ = δ₁ α       (i.e.  lap0 * φ = δ₁ * α)
# Then dφ = d0 * φ.
#
# The coexact component δβ is found via the dual Poisson problem.
# On a closed orientable surface, the 2-form space (face scalars) is
# related to 0-forms on the dual mesh.  The discrete route here uses:
#   Δ₂ β̃ = d₁ α          where Δ₂ = ⋆₂⁻¹ d₁ ⋆₁⁻¹ d₁ᵀ ⋆₂ is the dual Laplacian
# However, a simpler route is available on an oriented 2-manifold:
#   On a genus-0 surface, we can use the L² orthogonality.
#   The coexact part satisfies:  δ₂(d₂ β) = Δ₂ β  but we work with face 2-forms.
#
# Practical route (following Crane, de Goes, Desbrun, Schröder 2013):
# 1. Compute φ from: Δ₀ φ = δ₁ α  (with zero-mean gauge on closed surfaces).
# 2. Compute the exact part: α_exact = d₀ φ.
# 3. Compute the residual after removing the exact part: α_rem = α - α_exact.
# 4. Compute the coexact part by: solve Δ₁ for the coexact piece, or equivalently
#    solve the dual Poisson: star2 * d1 * star1^{-1} * δ_coexact = d₁ α.
#    On a genus-0 surface this is: (⋆₂⁻¹ d₁ ⋆₁⁻¹ d₁ᵀ ⋆₂) ψ = d₁ α   (ψ face 2-form)
#    Then δβ = ⋆₁⁻¹ d₁ᵀ (⋆₂ ψ).
# 5. Harmonic residual: h = α - α_exact - α_coexact.
#
# For genus-0 (sphere), the harmonic space is trivial (dim 0), so h ≈ 0.
# For higher genus (torus, etc.), h is non-zero but orthogonal to exact and coexact.

# ─────────────────────────────────────────────────────────────────────────────
# Exact component: dφ where Δ₀ φ = δ₁ α
# ─────────────────────────────────────────────────────────────────────────────

"""
    exact_component_1form(mesh, geom, dec, α; reg=1e-10)
        -> (α_exact, φ)

Compute the exact component of the Hodge decomposition of a discrete 1-form α.

Solves the 0-form Poisson problem:
    Δ₀ φ = δ₁ α

where Δ₀ = dec.lap0 and δ₁ is the codifferential.  On a closed surface,
we enforce the compatibility condition and solve with Tikhonov regularisation.

Returns `(α_exact, φ)` where:
- `α_exact = d₀ φ` is the exact component (edge 1-form),
- `φ` is the scalar potential (vertex 0-form).
"""
function exact_component_1form(
        mesh :: SurfaceMesh{T},
        geom :: SurfaceGeometry{T},
        dec  :: SurfaceDEC{T},
        α    :: AbstractVector{T};
        reg  :: Real = 1e-10,
) :: Tuple{Vector{T},Vector{T}} where {T}
    # Compute rhs = δ₁ α = codifferential_1 * α
    δ1  = codifferential_1(mesh, geom, dec)
    rhs = δ1 * α

    # Solve Δ₀ φ = rhs with zero-mean gauge
    φ   = solve_surface_poisson(mesh, geom, dec, rhs; reg=reg)

    # Exact component: dφ = d₀ * φ
    α_exact = dec.d0 * φ
    return α_exact, φ
end

# ─────────────────────────────────────────────────────────────────────────────
# Coexact component: δ₂(⋆₂ ψ) where the dual Poisson is solved for ψ
# ─────────────────────────────────────────────────────────────────────────────

"""
    coexact_component_1form(mesh, geom, dec, α; reg=1e-10)
        -> (α_coexact, ψ)

Compute the coexact component of the Hodge decomposition of a discrete 1-form α.

Uses the dual Poisson system for a 2-form potential ψ (face scalar):
    (⋆₂⁻¹ d₁ ⋆₁⁻¹ d₁ᵀ ⋆₂) ψ = d₁ α

On a closed oriented surface the dual Laplacian

    Δ_dual = ⋆₂⁻¹ d₁ ⋆₁⁻¹ d₁ᵀ ⋆₂

is positive semi-definite on 2-forms (face scalar fields) and has the same
constant nullspace structure as Δ₀.

Returns `(α_coexact, ψ)` where:
- `α_coexact = ⋆₁⁻¹ d₁ᵀ (⋆₂ ψ)` is the coexact component (edge 1-form),
- `ψ` is the face 2-form potential.
"""
function coexact_component_1form(
        mesh :: SurfaceMesh{T},
        geom :: SurfaceGeometry{T},
        dec  :: SurfaceDEC{T},
        α    :: AbstractVector{T};
        reg  :: Real = 1e-10,
) :: Tuple{Vector{T},Vector{T}} where {T}
    nf = length(mesh.faces)

    # star1 inverse diagonal
    s1_diag     = diag(dec.star1)
    inv_s1_diag = [w > eps(T) ? one(T)/w : zero(T) for w in s1_diag]
    inv_s1      = spdiagm(0 => inv_s1_diag)

    # star2 and its inverse
    s2_diag     = diag(dec.star2)
    inv_s2_diag = [w > eps(T) ? one(T)/w : zero(T) for w in s2_diag]
    inv_s2      = spdiagm(0 => inv_s2_diag)
    s2          = dec.star2

    # Dual Laplacian: Δ_dual = inv_s2 * d1 * inv_s1 * d1' * s2
    Δ_dual = inv_s2 * dec.d1 * inv_s1 * dec.d1' * s2

    # RHS: d₁ α
    rhs = dec.d1 * α

    # Enforce compatibility (zero mean of rhs under the dual measure s2)
    s2_diag_vec = diag(s2)
    total_s2    = sum(s2_diag_vec)
    if total_s2 > eps(T)
        rhs_mean = dot(s2_diag_vec, rhs) / total_s2
        rhs     .-= rhs_mean
    end

    # Solve with Tikhonov regularisation
    ε   = T(reg)
    M_f = spdiagm(0 => s2_diag_vec)  # face mass (star2)
    A   = Δ_dual + ε * inv_s2        # regularised system
    ψ   = Array(A) \ rhs

    # Remove mean of ψ under dual measure
    ψ_mean = dot(s2_diag_vec, ψ) / total_s2
    ψ     .-= ψ_mean

    # Coexact component: δ₂(⋆₂ ψ) = ⋆₁⁻¹ d₁ᵀ ⋆₂ ψ
    α_coexact = inv_s1 * dec.d1' * (s2 * ψ)
    return α_coexact, ψ
end

# ─────────────────────────────────────────────────────────────────────────────
# Harmonic component
# ─────────────────────────────────────────────────────────────────────────────

"""
    harmonic_component_1form(mesh, geom, dec, α; reg=1e-10)
        -> Vector{T}

Compute the harmonic component h of the Hodge decomposition:

    h = α - α_exact - α_coexact

This is the residual after removing the exact and coexact components.
On a closed genus-0 surface (sphere), h ≈ 0 (machine precision).
On higher-genus surfaces, h spans the harmonic space (dimension 2g).
"""
function harmonic_component_1form(
        mesh :: SurfaceMesh{T},
        geom :: SurfaceGeometry{T},
        dec  :: SurfaceDEC{T},
        α    :: AbstractVector{T};
        reg  :: Real = 1e-10,
) :: Vector{T} where {T}
    α_exact,   _ = exact_component_1form(mesh, geom, dec, α; reg=reg)
    α_coexact, _ = coexact_component_1form(mesh, geom, dec, α; reg=reg)
    return α .- α_exact .- α_coexact
end

# ─────────────────────────────────────────────────────────────────────────────
# Full Hodge decomposition
# ─────────────────────────────────────────────────────────────────────────────

"""
    hodge_decompose_1form(mesh, geom, dec, α; reg=1e-10)
        -> NamedTuple

Compute the full Hodge decomposition of a discrete 1-form α on a closed
triangulated surface:

    α = α_exact + α_coexact + α_harmonic

Returns a named tuple:
```
(
    exact    = α_exact,      # exact component d₀ φ (edge 1-form)
    coexact  = α_coexact,    # coexact component δ₂(⋆₂ ψ) (edge 1-form)
    harmonic = α_harmonic,   # harmonic residual (edge 1-form)
    phi      = φ,            # 0-form scalar potential for exact part
    psi      = ψ,            # 2-form scalar potential for coexact part
)
```

On a genus-0 surface (sphere), the harmonic component is ≈ 0.

Parameters
----------
- `reg` – Tikhonov regularisation parameter for both Poisson solves.
"""
function hodge_decompose_1form(
        mesh :: SurfaceMesh{T},
        geom :: SurfaceGeometry{T},
        dec  :: SurfaceDEC{T},
        α    :: AbstractVector{T};
        reg  :: Real = 1e-10,
)  where {T}
    α_exact,   φ = exact_component_1form(mesh, geom, dec, α; reg=reg)
    α_coexact, ψ = coexact_component_1form(mesh, geom, dec, α; reg=reg)
    α_harmonic   = α .- α_exact .- α_coexact
    return (
        exact    = α_exact,
        coexact  = α_coexact,
        harmonic = α_harmonic,
        phi      = φ,
        psi      = ψ,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

"""
    hodge_decomposition_residual(mesh, geom, dec, α, decomp) -> T

Compute the L²-residual of the Hodge decomposition:

    residual = ‖α - (α_exact + α_coexact + α_harmonic)‖_{⋆₁} / ‖α‖_{⋆₁}

where ‖β‖_{⋆₁}² = βᵀ ⋆₁ β is the DEC 1-form inner product.

`decomp` should be the named tuple returned by `hodge_decompose_1form`.
"""
function hodge_decomposition_residual(
        mesh   :: SurfaceMesh{T},
        geom   :: SurfaceGeometry{T},
        dec    :: SurfaceDEC{T},
        α      :: AbstractVector{T},
        decomp,
) :: T where {T}
    α_rec = decomp.exact .+ decomp.coexact .+ decomp.harmonic
    diff  = α .- α_rec
    s1    = dec.star1
    num   = dot(diff, s1 * diff)
    den   = dot(α,    s1 * α)
    den < eps(T) && return zero(T)
    return sqrt(num / den)
end

"""
    hodge_inner_products(mesh, geom, dec, decomp) -> NamedTuple

Compute the pairwise ⋆₁ inner products of the Hodge components.  For an
orthogonal decomposition these should all be near zero (up to discretisation).

Returns a named tuple:
```
(
    exact_coexact    = <α_exact, α_coexact>_{⋆₁},
    exact_harmonic   = <α_exact, α_harmonic>_{⋆₁},
    coexact_harmonic = <α_coexact, α_harmonic>_{⋆₁},
)
```
"""
function hodge_inner_products(
        :: SurfaceMesh{T},
        :: SurfaceGeometry{T},
        dec    :: SurfaceDEC{T},
        decomp,
) :: NamedTuple where {T}
    s1  = dec.star1
    ip(a, b) = dot(a, s1 * b)
    return (
        exact_coexact    = ip(decomp.exact, decomp.coexact),
        exact_harmonic   = ip(decomp.exact, decomp.harmonic),
        coexact_harmonic = ip(decomp.coexact, decomp.harmonic),
    )
end
