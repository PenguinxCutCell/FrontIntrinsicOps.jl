# surface_pdes_common.jl – Shared low-level PDE assembly and utility functions.
#
# Provides the 0-form mass matrix (from star0), lumped mass helpers, mean
# projection utilities, and compatibility / gauge utilities for closed surfaces.
#
# Sign convention: L = -Δ_Γ is positive semi-definite (inherited from operators).

# ─────────────────────────────────────────────────────────────────────────────
# Mass matrix
# ─────────────────────────────────────────────────────────────────────────────

"""
    mass_matrix(mesh::CurveMesh, geom::CurveGeometry) -> SparseMatrixCSC

Return the lumped 0-form mass matrix for a curve.

The mass matrix is the Hodge star ⋆₀: a diagonal sparse matrix whose entries
are the vertex dual lengths.
"""
function mass_matrix(
        mesh::CurveMesh{T},
        geom::CurveGeometry{T},
) :: SparseMatrixCSC{T,Int} where {T}
    return hodge_star_0(mesh, geom)
end

"""
    mass_matrix(mesh::SurfaceMesh, geom::SurfaceGeometry) -> SparseMatrixCSC

Return the lumped 0-form mass matrix for a surface.

The mass matrix is the Hodge star ⋆₀: a diagonal sparse matrix whose entries
are the vertex dual areas.
"""
function mass_matrix(
        mesh::SurfaceMesh{T},
        geom::SurfaceGeometry{T},
) :: SparseMatrixCSC{T,Int} where {T}
    return hodge_star_0(mesh, geom)
end

# ─────────────────────────────────────────────────────────────────────────────
# Lumped mass vector and apply_mass!
# ─────────────────────────────────────────────────────────────────────────────

"""
    lumped_mass_vector(mesh::CurveMesh, geom::CurveGeometry) -> Vector{T}

Return the diagonal of the 0-form mass matrix as a plain vector.
"""
function lumped_mass_vector(
        mesh::CurveMesh{T},
        geom::CurveGeometry{T},
) :: Vector{T} where {T}
    return copy(geom.vertex_dual_lengths)
end

"""
    lumped_mass_vector(mesh::SurfaceMesh, geom::SurfaceGeometry) -> Vector{T}

Return the diagonal of the 0-form mass matrix as a plain vector.
"""
function lumped_mass_vector(
        mesh::SurfaceMesh{T},
        geom::SurfaceGeometry{T},
) :: Vector{T} where {T}
    return copy(geom.vertex_dual_areas)
end

"""
    apply_mass!(y, mesh::CurveMesh, geom::CurveGeometry, x) -> y

In-place mass-matrix multiplication: y .= M * x  (allocation-conscious path).
"""
function apply_mass!(
        y    :: AbstractVector{T},
        mesh :: CurveMesh{T},
        geom :: CurveGeometry{T},
        x    :: AbstractVector{T},
) :: AbstractVector{T} where {T}
    @inbounds for i in eachindex(x)
        y[i] = geom.vertex_dual_lengths[i] * x[i]
    end
    return y
end

"""
    apply_mass!(y, mesh::SurfaceMesh, geom::SurfaceGeometry, x) -> y

In-place mass-matrix multiplication: y .= M * x  (allocation-conscious path).
"""
function apply_mass!(
        y    :: AbstractVector{T},
        mesh :: SurfaceMesh{T},
        geom :: SurfaceGeometry{T},
        x    :: AbstractVector{T},
) :: AbstractVector{T} where {T}
    @inbounds for i in eachindex(x)
        y[i] = geom.vertex_dual_areas[i] * x[i]
    end
    return y
end

# ─────────────────────────────────────────────────────────────────────────────
# Weighted mean and projection utilities
# ─────────────────────────────────────────────────────────────────────────────

"""
    weighted_mean(mesh::CurveMesh, geom::CurveGeometry, u) -> T

Compute the measure-weighted mean of vertex field `u` on a curve:

    ⟨u⟩ = (Σ_v  dual_length_v * u_v) / total_length
"""
function weighted_mean(
        mesh :: CurveMesh{T},
        geom :: CurveGeometry{T},
        u    :: AbstractVector{T},
) :: T where {T}
    total = sum(geom.vertex_dual_lengths)
    total < eps(T) && return zero(T)
    return dot(geom.vertex_dual_lengths, u) / total
end

"""
    weighted_mean(mesh::SurfaceMesh, geom::SurfaceGeometry, u) -> T

Compute the area-weighted mean of vertex field `u` on a surface:

    ⟨u⟩ = (Σ_v  dual_area_v * u_v) / total_area
"""
function weighted_mean(
        mesh :: SurfaceMesh{T},
        geom :: SurfaceGeometry{T},
        u    :: AbstractVector{T},
) :: T where {T}
    total = sum(geom.vertex_dual_areas)
    total < eps(T) && return zero(T)
    return dot(geom.vertex_dual_areas, u) / total
end

"""
    zero_mean_projection!(u, mesh::CurveMesh, geom::CurveGeometry) -> u

Subtract the weighted mean from `u` in-place so that ⟨u⟩ = 0.
"""
function zero_mean_projection!(
        u    :: AbstractVector{T},
        mesh :: CurveMesh{T},
        geom :: CurveGeometry{T},
) :: AbstractVector{T} where {T}
    μ = weighted_mean(mesh, geom, u)
    u .-= μ
    return u
end

"""
    zero_mean_projection!(u, mesh::SurfaceMesh, geom::SurfaceGeometry) -> u

Subtract the area-weighted mean from `u` in-place so that ⟨u⟩ = 0.
"""
function zero_mean_projection!(
        u    :: AbstractVector{T},
        mesh :: SurfaceMesh{T},
        geom :: SurfaceGeometry{T},
) :: AbstractVector{T} where {T}
    μ = weighted_mean(mesh, geom, u)
    u .-= μ
    return u
end

# Alias for backward compatibility within the PDE layer.
const project_zero_mean! = zero_mean_projection!

# ─────────────────────────────────────────────────────────────────────────────
# Compatibility and gauge for Poisson on closed manifolds
# ─────────────────────────────────────────────────────────────────────────────

"""
    enforce_compatibility!(f, mesh::SurfaceMesh, geom::SurfaceGeometry) -> f

Subtract the weighted mean of `f` in-place to enforce the compatibility
condition  ∫ f dA = 0  required for the Poisson problem  L u = f  on a
closed manifold (where L has a constant nullspace).

This is equivalent to `zero_mean_projection!`.
"""
function enforce_compatibility!(
        f    :: AbstractVector{T},
        mesh :: SurfaceMesh{T},
        geom :: SurfaceGeometry{T},
) :: AbstractVector{T} where {T}
    return zero_mean_projection!(f, mesh, geom)
end
