# kforms.jl – k-form operators: codifferential, Hodge Laplacians, and
#              gradient/divergence-like DEC actions.
#
# Sign convention
# ---------------
# On an oriented 2-manifold embedded in ℝ³, the codifferentials are:
#
#   δ₁ : Ω¹ → Ω⁰     δ₁ = ⋆₀⁻¹ d₀ᵀ ⋆₁   (maps 1-forms to 0-forms)
#   δ₂ : Ω² → Ω¹     δ₂ = ⋆₁⁻¹ d₁ᵀ ⋆₂   (maps 2-forms to 1-forms)
#
# Hodge Laplacians:
#   Δ₀ = δ₁ d₀ = ⋆₀⁻¹ d₀ᵀ ⋆₁ d₀   (matches dec.lap0, the scalar Laplacian)
#   Δ₁ = δ₂ d₁ + d₀ δ₁            (Hodge Laplacian on 1-forms)
#
# All operators are returned as sparse matrices acting on the appropriate
# cochain spaces.

"""
    codifferential_1(mesh::SurfaceMesh, geom::SurfaceGeometry, dec::SurfaceDEC)
        -> SparseMatrixCSC{T,Int}

Assemble the codifferential  δ₁ : Ω¹ → Ω⁰  on a triangulated surface.

    δ₁ = ⋆₀⁻¹ d₀ᵀ ⋆₁

Size: `(nV × nE)`.

On a closed surface, the kernel of δ₁ contains all co-closed 1-forms.
The identity  δ₁ d₀ = dec.lap0  holds (this is the 0-form Hodge Laplacian).
"""
function codifferential_1(
        mesh :: SurfaceMesh{T},
        geom :: SurfaceGeometry{T},
        dec  :: SurfaceDEC{T},
) :: SparseMatrixCSC{T,Int} where {T}
    inv_s0_diag = [da > eps(T) ? one(T)/da : zero(T)
                   for da in geom.vertex_dual_areas]
    inv_s0 = spdiagm(0 => inv_s0_diag)
    # δ₁ = ⋆₀⁻¹ d₀ᵀ ⋆₁
    return inv_s0 * dec.d0' * dec.star1
end

"""
    codifferential_2(mesh::SurfaceMesh, geom::SurfaceGeometry, dec::SurfaceDEC)
        -> SparseMatrixCSC{T,Int}

Assemble the codifferential  δ₂ : Ω² → Ω¹  on a triangulated surface.

    δ₂ = ⋆₁⁻¹ d₁ᵀ ⋆₂

Size: `(nE × nF)`.

On a 2-manifold, Ω² is the space of 2-forms (one per face).  The codifferential
maps a face 2-form to an edge 1-form.  This is the adjoint of the exterior
derivative d₁ : Ω¹ → Ω² with respect to the L² inner products induced by ⋆₁
and ⋆₂.
"""
function codifferential_2(
        mesh :: SurfaceMesh{T},
        geom :: SurfaceGeometry{T},
        dec  :: SurfaceDEC{T},
) :: SparseMatrixCSC{T,Int} where {T}
    # star1 diagonal: need its inverse
    s1_diag   = diag(dec.star1)
    inv_s1_diag = [w > eps(T) ? one(T)/w : zero(T) for w in s1_diag]
    inv_s1 = spdiagm(0 => inv_s1_diag)
    # δ₂ = ⋆₁⁻¹ d₁ᵀ ⋆₂
    return inv_s1 * dec.d1' * dec.star2
end

"""
    hodge_laplacian_0(mesh::SurfaceMesh, geom::SurfaceGeometry, dec::SurfaceDEC;
                      method=:dec) -> SparseMatrixCSC{T,Int}

Return the Hodge Laplacian on 0-forms (scalar fields) on a surface.

    Δ₀ = δ₁ d₀ = ⋆₀⁻¹ d₀ᵀ ⋆₁ d₀

This is identical to the scalar Laplace–Beltrami `dec.lap0`.  Provided for
API consistency with `hodge_laplacian_1`.

The `method` keyword is forwarded to `laplace_matrix`.
"""
function hodge_laplacian_0(
        mesh   :: SurfaceMesh{T},
        geom   :: SurfaceGeometry{T},
        dec    :: SurfaceDEC{T};
        method :: Symbol = :dec,
) :: SparseMatrixCSC{T,Int} where {T}
    return laplace_matrix(mesh, geom, dec; method=method)
end

"""
    hodge_laplacian_1(mesh::SurfaceMesh, geom::SurfaceGeometry, dec::SurfaceDEC)
        -> SparseMatrixCSC{T,Int}

Assemble the Hodge Laplacian on 1-forms on a triangulated surface.

    Δ₁ = δ₂ d₁ + d₀ δ₁

Size: `(nE × nE)`.

This operator is self-adjoint with respect to the ⋆₁ inner product.  Its
kernel consists of the harmonic 1-forms (closed and co-closed).  For a genus-g
surface the kernel has dimension 2g.

Notes
-----
- For a topological sphere (genus 0) the kernel is trivial.
- The Bochner–Weitzenböck formula relates this to curvature; for simplicity
  we assemble purely from the discrete DEC operators without Ricci curvature
  correction.
"""
function hodge_laplacian_1(
        mesh :: SurfaceMesh{T},
        geom :: SurfaceGeometry{T},
        dec  :: SurfaceDEC{T},
) :: SparseMatrixCSC{T,Int} where {T}
    δ1 = codifferential_1(mesh, geom, dec)
    δ2 = codifferential_2(mesh, geom, dec)
    return δ2 * dec.d1 + dec.d0 * δ1
end

# ─────────────────────────────────────────────────────────────────────────────
# Gradient and divergence helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    gradient_0_to_1(mesh::SurfaceMesh, dec::SurfaceDEC, u) -> Vector{T}

Compute the discrete exterior derivative of a 0-form (vertex scalar field) to
a 1-form (edge cochain):

    (d₀ u)[e] = u[j] - u[i]   for edge e = (i → j).

This is the discrete gradient of a scalar function.  The result is a vector of
length `nE` (number of edges).
"""
function gradient_0_to_1(
        :: SurfaceMesh{T},
        dec :: SurfaceDEC{T},
        u   :: AbstractVector{T},
) :: Vector{T} where {T}
    return dec.d0 * u
end

"""
    gradient_0_to_1(mesh::CurveMesh, dec::CurveDEC, u) -> Vector{T}

Discrete gradient on a curve.
"""
function gradient_0_to_1(
        :: CurveMesh{T},
        dec :: CurveDEC{T},
        u   :: AbstractVector{T},
) :: Vector{T} where {T}
    return dec.d0 * u
end

"""
    divergence_1_to_0(mesh::SurfaceMesh, geom::SurfaceGeometry, dec::SurfaceDEC,
                      α) -> Vector{T}

Compute the discrete divergence of an edge 1-form `α` back to vertices via the
codifferential δ₁:

    div α = δ₁ α = ⋆₀⁻¹ d₀ᵀ ⋆₁ α

This is the L²-adjoint of the gradient with respect to the DEC inner products.
The result is a vertex 0-form of length `nV`.
"""
function divergence_1_to_0(
        mesh :: SurfaceMesh{T},
        geom :: SurfaceGeometry{T},
        dec  :: SurfaceDEC{T},
        α    :: AbstractVector{T},
) :: Vector{T} where {T}
    δ1 = codifferential_1(mesh, geom, dec)
    return δ1 * α
end

"""
    divergence_1_to_0(mesh::CurveMesh, geom::CurveGeometry, dec::CurveDEC,
                      α) -> Vector{T}

Discrete divergence on a curve.  Uses the existing `divergence` function.
"""
function divergence_1_to_0(
        mesh :: CurveMesh{T},
        geom :: CurveGeometry{T},
        dec  :: CurveDEC{T},
        α    :: AbstractVector{T},
) :: Vector{T} where {T}
    return divergence(mesh, geom, dec, α)
end

"""
    curl_like_1_to_2(mesh::SurfaceMesh, dec::SurfaceDEC, α) -> Vector{T}

Compute the discrete exterior derivative of an edge 1-form `α` to a face
2-form:

    (d₁ α)[f] = Σ_{e ∈ ∂f} ±α[e]

This is the DEC analogue of the surface curl.  The result is a vector of length
`nF` (number of faces).
"""
function curl_like_1_to_2(
        :: SurfaceMesh{T},
        dec :: SurfaceDEC{T},
        α   :: AbstractVector{T},
) :: Vector{T} where {T}
    return dec.d1 * α
end
