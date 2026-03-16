# performance.jl – Low-allocation helpers for repeated PDE stepping.
#
# Provides pre-allocated buffer structs and in-place kernels that avoid
# heap allocations inside time-stepping loops.
#
# Design goals:
# - Zero allocations per time step once buffers are allocated.
# - Drop-in replacements for the allocating versions in critical inner loops.
# - Compatible with the rest of the package's type and operator conventions.

# ─────────────────────────────────────────────────────────────────────────────
# Pre-allocated buffer struct
# ─────────────────────────────────────────────────────────────────────────────

"""
    SurfaceDiffusionBuffers{T}

Pre-allocated scratch vectors for a repeated surface diffusion time step.
Allocate once with `alloc_diffusion_buffers` and pass to the in-place step
function to avoid per-step heap allocations.

Fields
------
- `rhs` – right-hand side vector (nV).
- `tmp` – temporary scratch vector (nV).
"""
struct SurfaceDiffusionBuffers{T}
    rhs :: Vector{T}
    tmp :: Vector{T}
end

"""
    alloc_diffusion_buffers(nv::Int, T=Float64) -> SurfaceDiffusionBuffers{T}

Allocate scratch buffers for `nv`-vertex diffusion time stepping.
"""
function alloc_diffusion_buffers(nv::Int, ::Type{T}=Float64) where {T}
    return SurfaceDiffusionBuffers{T}(Vector{T}(undef, nv), Vector{T}(undef, nv))
end

"""
    SurfaceRDBuffers{T}

Pre-allocated scratch vectors for surface reaction–diffusion time stepping.

Fields
------
- `rhs`      – right-hand side vector (nV).
- `reaction` – reaction evaluation buffer (nV).
- `tmp`      – temporary scratch vector (nV).
"""
struct SurfaceRDBuffers{T}
    rhs      :: Vector{T}
    reaction :: Vector{T}
    tmp      :: Vector{T}
end

"""
    alloc_rd_buffers(nv::Int, T=Float64) -> SurfaceRDBuffers{T}

Allocate scratch buffers for `nv`-vertex reaction–diffusion time stepping.
"""
function alloc_rd_buffers(nv::Int, ::Type{T}=Float64) where {T}
    return SurfaceRDBuffers{T}(
        Vector{T}(undef, nv),
        Vector{T}(undef, nv),
        Vector{T}(undef, nv),
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# In-place backward-Euler diffusion step (low-allocation)
# ─────────────────────────────────────────────────────────────────────────────

"""
    step_diffusion_inplace!(u, cache::SurfacePDECache, buf::SurfaceDiffusionBuffers)
        -> u

In-place backward-Euler diffusion step.  Overwrites `u` with u^{n+1}.

Requires `cache` was built with `μ` and `dt`.  Uses `buf` for scratch storage.
Uses the strong form  `(I + dt θ μ L) u^{n+1} = u^n`,
performing exactly one triangular solve.
"""
function step_diffusion_inplace!(
        u     :: AbstractVector{T},
        cache :: SurfacePDECache{T},
        buf   :: SurfaceDiffusionBuffers{T},
) :: AbstractVector{T} where {T}
    cache.diffusion_fac === nothing &&
        error("step_diffusion_inplace!: no diffusion factorisation in cache.")
    # rhs = u^n  (strong form: no mass-matrix multiply needed)
    copy!(buf.tmp, u)
    ldiv!(u, cache.diffusion_fac, buf.tmp)
    return u
end

"""
    step_diffusion_inplace!(u, cache::CurvePDECache, buf) -> u

In-place diffusion step for a curve.
"""
function step_diffusion_inplace!(
        u     :: AbstractVector{T},
        cache :: CurvePDECache{T},
        buf   :: SurfaceDiffusionBuffers{T},
) :: AbstractVector{T} where {T}
    cache.diffusion_fac === nothing &&
        error("step_diffusion_inplace!: no diffusion factorisation in cache.")
    copy!(buf.tmp, u)
    ldiv!(u, cache.diffusion_fac, buf.tmp)
    return u
end

# ─────────────────────────────────────────────────────────────────────────────
# In-place IMEX reaction–diffusion step (low-allocation)
# ─────────────────────────────────────────────────────────────────────────────

"""
    step_rd_inplace!(u, cache::SurfacePDECache, buf::SurfaceRDBuffers,
                     mesh, geom, reaction, t) -> u

In-place IMEX backward-Euler reaction–diffusion step.  Overwrites `u`.

Requires `cache` was built with `μ` and `dt`.  The reaction term is treated
explicitly.  Uses the strong form `(I + dt θ μ L) u^{n+1} = u^n + dt r(u^n)`.

Steps:
1. Evaluate reaction: r = r(u^n, t).
2. Build rhs = u^n + dt * r.
3. Solve: u^{n+1} = fac \\ rhs.
"""
function step_rd_inplace!(
        u        :: AbstractVector{T},
        cache    :: SurfacePDECache{T},
        buf      :: SurfaceRDBuffers{T},
        mesh     :: SurfaceMesh{T},
        geom     :: SurfaceGeometry{T},
        reaction,
        t        :: Real,
) :: AbstractVector{T} where {T}
    cache.diffusion_fac === nothing &&
        error("step_rd_inplace!: no diffusion factorisation in cache.")
    dt_val = cache.params.dt

    # Step 1: evaluate reaction into buf.reaction
    evaluate_reaction!(buf.reaction, reaction, u, mesh, geom, t)

    # Step 2: rhs = u^n + dt * reaction  (strong form)
    @inbounds for i in eachindex(buf.rhs)
        buf.rhs[i] = u[i] + T(dt_val) * buf.reaction[i]
    end

    # Step 3: solve u^{n+1} = fac \ rhs
    copy!(buf.tmp, buf.rhs)
    ldiv!(u, cache.diffusion_fac, buf.tmp)
    return u
end

# ─────────────────────────────────────────────────────────────────────────────
# Additional low-allocation helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    apply_mass_inplace!(y, cache::SurfacePDECache, x) -> y

In-place mass matrix application: y = M * x.
Avoids allocating a new vector; uses `mul!`.
"""
function apply_mass_inplace!(
        y     :: AbstractVector{T},
        cache :: Union{SurfacePDECache{T},CurvePDECache{T}},
        x     :: AbstractVector{T},
) :: AbstractVector{T} where {T}
    mul!(y, cache.mass, x)
    return y
end

"""
    apply_laplace_inplace!(y, cache::SurfacePDECache, x) -> y

In-place Laplace–Beltrami application: y = L * x.
"""
function apply_laplace_inplace!(
        y     :: AbstractVector{T},
        cache :: Union{SurfacePDECache{T},CurvePDECache{T}},
        x     :: AbstractVector{T},
) :: AbstractVector{T} where {T}
    mul!(y, cache.laplace, x)
    return y
end

"""
    l2_norm_cached(cache::SurfacePDECache, u) -> T

Compute the area-weighted L² norm ‖u‖² = uᵀ M u using the cached mass matrix.
"""
function l2_norm_cached(
        cache :: Union{SurfacePDECache{T},CurvePDECache{T}},
        u     :: AbstractVector{T},
) :: T where {T}
    return sqrt(dot(u, cache.mass * u))
end

"""
    energy_norm_cached(cache::SurfacePDECache, u) -> T

Compute the H¹ energy semi-norm ‖∇u‖² = uᵀ L u using the cached Laplacian.
"""
function energy_norm_cached(
        cache :: Union{SurfacePDECache{T},CurvePDECache{T}},
        u     :: AbstractVector{T},
) :: T where {T}
    return sqrt(abs(dot(u, cache.laplace * u)))
end
