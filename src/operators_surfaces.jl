# operators_surfaces.jl – DEC operators for SurfaceMesh.
#
# Implements
# ----------
# * `build_dec(mesh, geom) -> SurfaceDEC`
# * scalar Laplace–Beltrami: L = ⋆₀⁻¹ d₀ᵀ ⋆₁ d₀
# * `laplace_beltrami(mesh, geom, dec, u)` – apply L to vertex field u

"""
    build_dec(mesh::SurfaceMesh{T}, geom::SurfaceGeometry{T}) -> SurfaceDEC{T}

Assemble all DEC operators for a triangulated surface and return a
`SurfaceDEC` container.

Laplace–Beltrami
----------------
The scalar Laplace–Beltrami operator is assembled through the DEC factorisation:

    L = ⋆₀⁻¹ d₀ᵀ ⋆₁ d₀

where ⋆₁ uses cotan weights (see `hodge_star_1`).  This operator is
**positive semi-definite** (L = −Δ_Γ in standard Δ = div grad convention).

On a sphere of radius R, the coordinate functions satisfy:
    L x = (2/R²) x   (since Δ_Γ x = −(2/R²) x and L = −Δ_Γ)

This is algebraically equivalent to the classical cotan/DDG formula:

    (L u)ᵢ = (1 / Aᵢ) Σⱼ∈N(i) wᵢⱼ (uᵢ − uⱼ)

where `Aᵢ` is the barycentric dual area at vertex `i` and `wᵢⱼ = (1/2)(cot αᵢⱼ + cot βᵢⱼ)`.

Acceptance criterion
--------------------
For a closed oriented triangulated surface, `d1 * d0 ≈ 0` holds to machine
precision (checked by `check_dec`).
"""
function build_dec(mesh::SurfaceMesh{T}, geom::SurfaceGeometry{T}) :: SurfaceDEC{T} where {T}
    d0 = incidence_0(mesh)
    d1 = incidence_1(mesh)
    s0 = hodge_star_0(mesh, geom)
    s1 = hodge_star_1(mesh, geom)
    s2 = hodge_star_2(mesh, geom)

    # ⋆₀⁻¹ diagonal
    inv_s0_diag = [da > eps(T) ? one(T)/da : zero(T)
                   for da in geom.vertex_dual_areas]
    inv_s0 = spdiagm(0 => inv_s0_diag)

    lap0 = inv_s0 * d0' * s1 * d0

    return SurfaceDEC{T}(d0, d1, s0, s1, s2, lap0)
end

"""
    laplace_beltrami(mesh, geom, dec, u) -> Vector{T}

Apply the scalar Laplace–Beltrami operator to vertex field `u`.

Returns `dec.lap0 * u`.
"""
function laplace_beltrami(
        ::SurfaceMesh,
        ::SurfaceGeometry,
        dec::SurfaceDEC{T},
        u::AbstractVector{T},
) :: Vector{T} where {T}
    return dec.lap0 * u
end

"""
    gradient(mesh, dec, u) -> Vector{T}

Discrete gradient of vertex field `u` to an edge 1-cochain.
"""
function gradient(::SurfaceMesh, dec::SurfaceDEC{T}, u::AbstractVector{T}) where {T}
    return dec.d0 * u
end
