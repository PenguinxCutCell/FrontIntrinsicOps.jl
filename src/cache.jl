# cache.jl – Operator and factorization caching for repeated PDE solves.
#
# Provides `SurfacePDECache` and `CurvePDECache` structs that store the
# assembled sparse operators and their factorizations.  Building the cache once
# and reusing it for repeated time stepping eliminates repeated assembly and
# factorization costs.
#
# Usage example:
#
#   cache = build_pde_cache(mesh, geom, dec; μ=0.1, dt=0.01, θ=1.0)
#   for step in 1:nsteps
#       u, _ = step_surface_diffusion_backward_euler(mesh, geom, dec, u, dt, μ;
#                  factorization=cache.diffusion_fac)
#   end
#
# Cache invalidation:
# The cache is NOT automatically invalidated when `μ`, `dt`, or geometry
# changes.  Users must rebuild the cache (or call `update_pde_cache!`) when
# any of these parameters change.

# ─────────────────────────────────────────────────────────────────────────────
# SurfacePDECache
# ─────────────────────────────────────────────────────────────────────────────

"""
    SurfacePDECache{T}

Cache of assembled PDE operators and optional factorizations for a static
triangulated surface.  Build with `build_pde_cache`.

Fields
------
- `laplace`              – Laplace–Beltrami matrix L (nV × nV).
- `mass`                 – Lumped mass matrix M = ⋆₀ (nV × nV, diagonal).
- `mass_vec`             – Diagonal of M as a plain Vector{T}.
- `diffusion_fac`        – Factorisation of `(I + dt θ μ L)` (strong form) if μ, dt, θ are given.
- `helmholtz_fac`        – Factorisation of (L + α M) if α is given.
- `dec`                  – The SurfaceDEC operators stored for reference.
- `params`               – Named tuple of build parameters.
"""
struct SurfacePDECache{T, DecType, FacType}
    laplace       :: SparseMatrixCSC{T,Int}
    mass          :: SparseMatrixCSC{T,Int}
    mass_vec      :: Vector{T}
    dec           :: DecType
    diffusion_fac :: Union{FacType,Nothing}
    helmholtz_fac :: Union{FacType,Nothing}
    params        :: NamedTuple
end

"""
    CurvePDECache{T}

Cache of assembled PDE operators and factorizations for a curve.
"""
struct CurvePDECache{T, DecType, FacType}
    laplace       :: SparseMatrixCSC{T,Int}
    mass          :: SparseMatrixCSC{T,Int}
    mass_vec      :: Vector{T}
    dec           :: DecType
    diffusion_fac :: Union{FacType,Nothing}
    helmholtz_fac :: Union{FacType,Nothing}
    params        :: NamedTuple
end

# ─────────────────────────────────────────────────────────────────────────────
# Cache construction
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_pde_cache(mesh, geom, dec;
                    μ=nothing, dt=nothing, θ=1.0,
                    α_helmholtz=nothing,
                    method=:dec)
        -> SurfacePDECache

Build and return a `SurfacePDECache` for a static triangulated surface.

Pre-assembles:
- Laplace–Beltrami matrix L,
- lumped mass matrix M and its diagonal vector,
- optionally, the factorisation of `(M + dt θ μ L)` (for diffusion/reaction–diffusion)
  if `μ` and `dt` are provided,
- optionally, the factorisation of `(L + α M)` if `α_helmholtz` is provided.

Parameters
----------
- `μ`             – diffusion coefficient for diffusion factorisation.
- `dt`            – time step for diffusion factorisation.
- `θ`             – implicitness parameter (default 1.0 = backward Euler).
- `α_helmholtz`   – Helmholtz shift for Helmholtz factorisation.
- `method`        – Laplace assembly method (`:dec` or `:cotan`).
"""
function build_pde_cache(
        mesh          :: SurfaceMesh{T},
        geom          :: SurfaceGeometry{T},
        dec           :: SurfaceDEC{T};
        μ             = nothing,
        dt            = nothing,
        θ             :: Real = one(T),
        α_helmholtz   = nothing,
        method        :: Symbol = :dec,
) where {T}
    L    = laplace_matrix(mesh, geom, dec; method=method)
    M    = mass_matrix(mesh, geom)
    mvec = copy(geom.vertex_dual_areas)
    nv   = size(L, 1)

    diff_fac = nothing
    if μ !== nothing && dt !== nothing
        # Strong form: (I + dt θ μ L), consistent with step_surface_reaction_diffusion_imex
        A_diff   = sparse(T(1) * LinearAlgebra.I(nv)) + T(dt) * T(θ) * T(μ) * L
        diff_fac = factorize(A_diff)
    end

    helm_fac = nothing
    if α_helmholtz !== nothing
        A_helm   = L + T(α_helmholtz) * M
        helm_fac = factorize(A_helm)
    end

    params = (μ=μ, dt=dt, θ=T(θ), α_helmholtz=α_helmholtz, method=method)

    FacType = diff_fac !== nothing ? typeof(diff_fac) :
              helm_fac !== nothing ? typeof(helm_fac) : Nothing
    DecType = typeof(dec)

    return SurfacePDECache{T,DecType,FacType}(
        L, M, mvec, dec, diff_fac, helm_fac, params)
end

"""
    build_pde_cache(mesh::CurveMesh, geom::CurveGeometry, dec::CurveDEC; ...)
        -> CurvePDECache

Build and return a `CurvePDECache` for a curve.  Same parameters as the
surface version.
"""
function build_pde_cache(
        mesh          :: CurveMesh{T},
        geom          :: CurveGeometry{T},
        dec           :: CurveDEC{T};
        μ             = nothing,
        dt            = nothing,
        θ             :: Real = one(T),
        α_helmholtz   = nothing,
        method        :: Symbol = :dec,
) where {T}
    L    = laplace_matrix(mesh, geom, dec; method=method)
    M    = mass_matrix(mesh, geom)
    mvec = copy(geom.vertex_dual_lengths)
    nv   = size(L, 1)

    diff_fac = nothing
    if μ !== nothing && dt !== nothing
        # Strong form: (I + dt θ μ L), consistent with step_surface_reaction_diffusion_imex
        A_diff   = sparse(T(1) * LinearAlgebra.I(nv)) + T(dt) * T(θ) * T(μ) * L
        diff_fac = factorize(A_diff)
    end

    helm_fac = nothing
    if α_helmholtz !== nothing
        A_helm   = L + T(α_helmholtz) * M
        helm_fac = factorize(A_helm)
    end

    params = (μ=μ, dt=dt, θ=T(θ), α_helmholtz=α_helmholtz, method=method)

    FacType = diff_fac !== nothing ? typeof(diff_fac) :
              helm_fac !== nothing ? typeof(helm_fac) : Nothing
    DecType = typeof(dec)

    return CurvePDECache{T,DecType,FacType}(
        L, M, mvec, dec, diff_fac, helm_fac, params)
end

# ─────────────────────────────────────────────────────────────────────────────
# Cache update helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    update_pde_cache(cache::SurfacePDECache, mesh, geom, dec;
                     μ=nothing, dt=nothing, θ=1.0,
                     α_helmholtz=nothing, method=:dec)
        -> SurfacePDECache

Rebuild the cache with updated parameters (e.g., after a parameter change).
Returns a new cache object; the old one is unchanged.
"""
function update_pde_cache(
        :: SurfacePDECache{T},
        mesh :: SurfaceMesh{T},
        geom :: SurfaceGeometry{T},
        dec  :: SurfaceDEC{T};
        kwargs...
) where {T}
    return build_pde_cache(mesh, geom, dec; kwargs...)
end

"""
    update_pde_cache(cache::CurvePDECache, mesh, geom, dec; ...) -> CurvePDECache
"""
function update_pde_cache(
        :: CurvePDECache{T},
        mesh :: CurveMesh{T},
        geom :: CurveGeometry{T},
        dec  :: CurveDEC{T};
        kwargs...
) where {T}
    return build_pde_cache(mesh, geom, dec; kwargs...)
end

# ─────────────────────────────────────────────────────────────────────────────
# Cached PDE step helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    step_diffusion_cached(cache::SurfacePDECache, uⁿ) -> Vector{T}

Advance the surface diffusion equation by one backward Euler step using the
pre-cached factorisation.  Requires the cache was built with `μ` and `dt`.

    u^{n+1} = fac \\ uⁿ

Uses the strong form  `(I + dt θ μ L) u^{n+1} = u^n`,
consistent with `step_surface_reaction_diffusion_imex` and
`step_surface_diffusion_backward_euler`.
This is allocation-light: only one triangular solve.
"""
function step_diffusion_cached(
        cache :: SurfacePDECache{T},
        uⁿ   :: AbstractVector{T},
) :: Vector{T} where {T}
    cache.diffusion_fac === nothing &&
        error("step_diffusion_cached: cache was built without μ and dt; no factorisation available.")
    return cache.diffusion_fac \ Vector{T}(uⁿ)
end

"""
    step_diffusion_cached(cache::CurvePDECache, uⁿ) -> Vector{T}

Backward Euler diffusion step for a curve using cached factorisation.
"""
function step_diffusion_cached(
        cache :: CurvePDECache{T},
        uⁿ   :: AbstractVector{T},
) :: Vector{T} where {T}
    cache.diffusion_fac === nothing &&
        error("step_diffusion_cached: cache was built without μ and dt; no factorisation available.")
    return cache.diffusion_fac \ Vector{T}(uⁿ)
end

"""
    solve_helmholtz_cached(cache::SurfacePDECache, f) -> Vector{T}

Solve the Helmholtz problem (L + α M) u = f using the cached factorisation.
Requires the cache was built with `α_helmholtz`.
"""
function solve_helmholtz_cached(
        cache :: SurfacePDECache{T},
        f     :: AbstractVector{T},
) :: Vector{T} where {T}
    cache.helmholtz_fac === nothing &&
        error("solve_helmholtz_cached: cache was built without α_helmholtz.")
    return cache.helmholtz_fac \ f
end

"""
    solve_helmholtz_cached(cache::CurvePDECache, f) -> Vector{T}

Helmholtz solve for a curve using the cached factorisation.
"""
function solve_helmholtz_cached(
        cache :: CurvePDECache{T},
        f     :: AbstractVector{T},
) :: Vector{T} where {T}
    cache.helmholtz_fac === nothing &&
        error("solve_helmholtz_cached: cache was built without α_helmholtz.")
    return cache.helmholtz_fac \ f
end
