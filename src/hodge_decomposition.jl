# hodge_decomposition.jl – Backward-compatible wrappers for 1-form Hodge
# decomposition.
#
# The topology-aware implementation now lives in `harmonic_forms.jl`.

"""
    exact_component_1form(mesh, geom, dec, α; gauge=:mean_zero, reg=1e-10, factor_cache=nothing)
        -> (α_exact, φ)

Compute the exact component `α_exact = dφ` of a discrete 1-form `α`.

Gauge convention
----------------
`gauge=:mean_zero` enforces a weighted zero-mean gauge on the scalar potential.
"""
function exact_component_1form(
        mesh :: SurfaceMesh{T},
        geom :: SurfaceGeometry{T},
        dec  :: SurfaceDEC{T},
        α    :: AbstractVector{T};
        gauge :: Symbol = :mean_zero,
        reg   :: Real   = 1e-10,
        factor_cache = nothing,
) :: Tuple{Vector{T},Vector{T}} where {T}
    return _exact_component_impl(mesh, geom, dec, α; gauge=gauge, reg=reg, factor_cache=factor_cache)
end

"""
    coexact_component_1form(mesh, geom, dec, α; gauge=:mean_zero, reg=1e-10, factor_cache=nothing)
        -> (α_coexact, ψ)

Compute the coexact component `α_coexact = δψ` of a discrete 1-form `α`.

Gauge convention
----------------
`gauge=:mean_zero` enforces a weighted zero-mean gauge on the face potential.
"""
function coexact_component_1form(
        mesh :: SurfaceMesh{T},
        geom :: SurfaceGeometry{T},
        dec  :: SurfaceDEC{T},
        α    :: AbstractVector{T};
        gauge :: Symbol = :mean_zero,
        reg   :: Real   = 1e-10,
        factor_cache = nothing,
) :: Tuple{Vector{T},Vector{T}} where {T}
    return _coexact_component_impl(mesh, geom, dec, α; gauge=gauge, reg=reg, factor_cache=factor_cache)
end

"""
    harmonic_component_1form(mesh, geom, dec, α; basis=nothing, reg=1e-10, factor_cache=nothing)
        -> Vector

Compute the harmonic component of `α`.

If `basis` is supplied, projection is done in the supplied harmonic basis.
Otherwise the component is obtained from a full decomposition solve.
"""
function harmonic_component_1form(
        mesh :: SurfaceMesh{T},
        geom :: SurfaceGeometry{T},
        dec  :: SurfaceDEC{T},
        α    :: AbstractVector{T};
        basis = nothing,
        reg   :: Real = 1e-10,
        factor_cache = nothing,
) :: Vector{T} where {T}
    return project_harmonic(α, mesh, geom, dec; basis=basis, reg=reg, factor_cache=factor_cache)
end

"""
    hodge_decompose_1form(mesh, geom, dec, α;
                          basis=nothing,
                          gauge=:mean_zero,
                          reg=1e-10,
                          factor_cache=nothing) -> NamedTuple

Backward-compatible front-end for full 1-form Hodge decomposition.

Returns
-------
A named tuple with legacy keys:
`(exact, coexact, harmonic, phi, psi)`.
"""
function hodge_decompose_1form(
        mesh :: SurfaceMesh{T},
        geom :: SurfaceGeometry{T},
        dec  :: SurfaceDEC{T},
        α    :: AbstractVector{T};
        basis = nothing,
        gauge :: Symbol = :mean_zero,
        reg   :: Real   = 1e-10,
        factor_cache = nothing,
) where {T}
    decomp = hodge_decomposition_full(
        α,
        mesh,
        geom,
        dec;
        basis=basis,
        gauge=gauge,
        reg=reg,
        factor_cache=factor_cache,
    )

    return (
        exact    = decomp.exact,
        coexact  = decomp.coexact,
        harmonic = decomp.harmonic,
        phi      = decomp.potentials.α,
        psi      = decomp.potentials.β,
    )
end

"""
    hodge_decomposition_residual(mesh, geom, dec, α, decomp) -> T

Compute the relative reconstruction residual in the `⋆1` norm.
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
    return sqrt(max(num, zero(T)) / den)
end

"""
    hodge_inner_products(mesh, geom, dec, decomp) -> NamedTuple

Compute pairwise `⋆1` inner products between decomposition components.
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
