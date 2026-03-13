# operators_curves.jl – DEC operators for CurveMesh.
#
# Implements
# ----------
# * `build_dec(mesh, geom) -> CurveDEC`
# * scalar Laplace–Beltrami: L = ⋆₀⁻¹ d₀ᵀ ⋆₁ d₀
# * `laplace_beltrami(mesh, geom, dec, u)` – apply L to vertex field u
# * `gradient(mesh, dec, u)` – discrete gradient (edge 1-cochain)
# * `divergence(mesh, dec, alpha)` – discrete divergence (vertex 0-cochain)

"""
    build_dec(mesh::CurveMesh{T}, geom::CurveGeometry{T}) -> CurveDEC{T}

Assemble all DEC operators for a curve mesh and return a `CurveDEC` container.

The scalar Laplace–Beltrami is assembled as:
    L = ⋆₀⁻¹ d₀ᵀ ⋆₁ d₀
"""
function build_dec(mesh::CurveMesh{T}, geom::CurveGeometry{T}) :: CurveDEC{T} where {T}
    d0    = incidence_0(mesh)
    s0    = hodge_star_0(mesh, geom)
    s1    = hodge_star_1(mesh, geom)
    # ⋆₀⁻¹ is the diagonal matrix with entries 1/dual_length
    inv_s0_diag = [dl > eps(T) ? one(T)/dl : zero(T)
                   for dl in geom.vertex_dual_lengths]
    inv_s0 = spdiagm(0 => inv_s0_diag)
    lap0   = inv_s0 * d0' * s1 * d0
    return CurveDEC{T}(d0, s0, s1, lap0)
end

"""
    laplace_beltrami(mesh, geom, dec, u) -> Vector{T}

Apply the scalar Laplace–Beltrami operator to vertex field `u`.

Returns `dec.lap0 * u`.
"""
function laplace_beltrami(
        ::CurveMesh,
        ::CurveGeometry,
        dec::CurveDEC{T},
        u::AbstractVector{T},
) :: Vector{T} where {T}
    return dec.lap0 * u
end

"""
    gradient(mesh, dec, u) -> Vector{T}

Compute the discrete gradient of vertex field `u` as an edge 1-cochain.

Returns `dec.d0 * u`.
"""
function gradient(::CurveMesh, dec::CurveDEC{T}, u::AbstractVector{T}) where {T}
    return dec.d0 * u
end

"""
    divergence(mesh, geom, dec, alpha) -> Vector{T}

Compute the discrete divergence of an edge 1-form `alpha` back to vertices.

Using the DEC formula:  div α = ⋆₀⁻¹ d₀ᵀ ⋆₁ α.
"""
function divergence(
        ::CurveMesh,
        geom::CurveGeometry{T},
        dec::CurveDEC{T},
        alpha::AbstractVector{T},
) :: Vector{T} where {T}
    inv_s0_diag = [dl > eps(T) ? one(T)/dl : zero(T)
                   for dl in geom.vertex_dual_lengths]
    inv_s0 = spdiagm(0 => inv_s0_diag)
    return inv_s0 * dec.d0' * dec.star1 * alpha
end
