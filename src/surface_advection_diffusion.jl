# surface_advection_diffusion.jl – Combined scalar advection–diffusion on
#                                   static surfaces.
#
# PDE (strong form):  du/dt + M⁻¹ A u + μ L u = g
#
# where A is the "raw" edge-flux transport matrix (M du/dt + A u = 0 form),
# L = M⁻¹ K is the scalar Laplace–Beltrami, and g is a pointwise source.
#
# Time-stepping strategies
# ------------------------
# IMEX (implicit-explicit):
#   - Transport (M⁻¹ A u) treated explicitly.
#   - Diffusion (μ L u) treated implicitly.
#   System: (I + dt μ L) uⁿ⁺¹ = uⁿ - dt M⁻¹ A uⁿ + dt g
#
# Backward Euler (fully implicit, both terms):
#   (I + dt M⁻¹ A + dt μ L) uⁿ⁺¹ = uⁿ + dt g

# ─────────────────────────────────────────────────────────────────────────────
# Helper: assemble both operators at once
# ─────────────────────────────────────────────────────────────────────────────

"""
    assemble_advection_diffusion_operators(mesh, geom, dec, vel, μ;
                                           scheme=:upwind, method=:dec)
        -> (A, L, M)

Assemble and return the transport operator `A`, Laplace matrix `L`, and mass
matrix `M` for use in an advection–diffusion solve.

`vel` accepts any format supported by `edge_flux_velocity`.
"""
function assemble_advection_diffusion_operators(
        mesh   :: SurfaceMesh{T},
        geom   :: SurfaceGeometry{T},
        dec    :: SurfaceDEC{T},
        vel;
        scheme :: Symbol = :upwind,
        method :: Symbol = :dec,
) where {T}
    A = assemble_transport_operator(mesh, geom, vel; scheme=scheme)
    L = laplace_matrix(mesh, geom, dec; method=method)
    M = mass_matrix(mesh, geom)
    return A, L, M
end

function assemble_advection_diffusion_operators(
        mesh   :: CurveMesh{T},
        geom   :: CurveGeometry{T},
        dec    :: CurveDEC{T},
        vel;
        scheme :: Symbol = :upwind,
        method :: Symbol = :dec,
) where {T}
    A = assemble_transport_operator(mesh, geom, vel; scheme=scheme)
    L = laplace_matrix(mesh, geom, dec; method=method)
    M = mass_matrix(mesh, geom)
    return A, L, M
end

# ─────────────────────────────────────────────────────────────────────────────
# IMEX step: transport explicit, diffusion implicit
# ─────────────────────────────────────────────────────────────────────────────

"""
    step_surface_advection_diffusion_imex(mesh, geom, dec, uⁿ, vel, dt, μ;
                                          scheme=:upwind,
                                          transport_operator=nothing,
                                          rhs=nothing,
                                          factorization=nothing,
                                          method=:dec)
        -> (uⁿ⁺¹, factorization)

Advance the advection–diffusion equation

    du/dt + M⁻¹ A u + μ L u = g

by one IMEX (implicit-explicit) step:

    (I + dt μ L) uⁿ⁺¹ = uⁿ - dt M⁻¹ A uⁿ + dt g

Transport (M⁻¹ A uⁿ) is treated **explicitly** (evaluated at old time level).
Diffusion (μ L uⁿ⁺¹) is treated **implicitly**.

Parameters
----------
- `vel`                – velocity field (any format accepted by `edge_flux_velocity`).
- `dt`                 – time step size.
- `μ`                  – diffusion coefficient.
- `scheme`             – advection scheme: `:upwind` (default) or `:centered`.
- `transport_operator` – pre-assembled transport matrix `A`; pass to reuse across
                         steps and avoid reassembly (major allocation saving).
- `rhs`                – optional pointwise source term `g` (vertex vector).
- `factorization`      – pre-computed factorization of `(I + dt μ L)`; reuse for
                         efficiency when `dt`, `μ` are constant.
- `method`             – Laplace assembly method (`:dec` or `:cotan`).

Returns
-------
`(uⁿ⁺¹, fac)` – solution at next time level and the factorization for reuse.
"""
function step_surface_advection_diffusion_imex(
        mesh               :: SurfaceMesh{T},
        geom               :: SurfaceGeometry{T},
        dec                :: SurfaceDEC{T},
        uⁿ                 :: AbstractVector{T},
        vel,
        dt                 :: Real,
        μ                  :: Real;
        scheme             :: Symbol = :upwind,
        transport_operator :: Union{AbstractSparseMatrix{T},Nothing} = nothing,
        rhs                :: Union{AbstractVector{T},Nothing} = nothing,
        factorization      :: Any    = nothing,
        method             :: Symbol = :dec,
) where {T}
    A  = transport_operator !== nothing ? transport_operator :
         assemble_transport_operator(mesh, geom, vel; scheme=scheme)
    dt = T(dt); μ = T(μ)
    nv = length(uⁿ)

    # Explicit RHS: uⁿ - dt M⁻¹ A uⁿ + dt g
    # Avoid sparse mass matrix: use vertex_dual_areas directly.
    da   = geom.vertex_dual_areas
    Au   = A * uⁿ
    b    = uⁿ .- dt .* (Au ./ da)
    if rhs !== nothing
        b .+= dt .* rhs
    end

    # Implicit diffusion LHS: I + dt μ L
    # Only assemble L (and factorize) when no factorization is available.
    if factorization === nothing
        L    = laplace_matrix(mesh, geom, dec; method=method)
        Alhs = spdiagm(0 => ones(T, nv)) + dt * μ * L
        fac  = factorize(Alhs)
    else
        fac  = factorization
    end

    uⁿ⁺¹ = fac \ b
    return uⁿ⁺¹, fac
end

"""
    step_surface_advection_diffusion_imex(mesh::CurveMesh, ...) -> (uⁿ⁺¹, fac)

IMEX advection–diffusion step for a curve.
"""
function step_surface_advection_diffusion_imex(
        mesh               :: CurveMesh{T},
        geom               :: CurveGeometry{T},
        dec                :: CurveDEC{T},
        uⁿ                 :: AbstractVector{T},
        vel,
        dt                 :: Real,
        μ                  :: Real;
        scheme             :: Symbol = :upwind,
        transport_operator :: Union{AbstractSparseMatrix{T},Nothing} = nothing,
        rhs                :: Union{AbstractVector{T},Nothing} = nothing,
        factorization      :: Any    = nothing,
        method             :: Symbol = :dec,
) where {T}
    A  = transport_operator !== nothing ? transport_operator :
         assemble_transport_operator(mesh, geom, vel; scheme=scheme)
    dt = T(dt); μ = T(μ)
    nv = length(uⁿ)

    dl   = geom.vertex_dual_lengths
    Au   = A * uⁿ
    b    = uⁿ .- dt .* (Au ./ dl)
    if rhs !== nothing
        b .+= dt .* rhs
    end

    if factorization === nothing
        L    = laplace_matrix(mesh, geom, dec; method=method)
        Alhs = spdiagm(0 => ones(T, nv)) + dt * μ * L
        fac  = factorize(Alhs)
    else
        fac  = factorization
    end

    uⁿ⁺¹ = fac \ b
    return uⁿ⁺¹, fac
end

# ─────────────────────────────────────────────────────────────────────────────
# Fully implicit backward-Euler step
# ─────────────────────────────────────────────────────────────────────────────

"""
    step_surface_advection_diffusion_backward_euler(mesh, geom, dec, uⁿ, vel,
                                                    dt, μ;
                                                    scheme=:upwind,
                                                    rhs=nothing,
                                                    method=:dec)
        -> Vector{T}

Advance the advection–diffusion equation by one fully implicit backward-Euler
step.  Since the transport operator A depends on the velocity field (which is
prescribed and static), we can assemble the full system matrix:

    (I + dt M⁻¹ A + dt μ L) uⁿ⁺¹ = uⁿ + dt g

Note: this builds a new factorization every call because A changes with `vel`.
"""
function step_surface_advection_diffusion_backward_euler(
        mesh   :: SurfaceMesh{T},
        geom   :: SurfaceGeometry{T},
        dec    :: SurfaceDEC{T},
        uⁿ     :: AbstractVector{T},
        vel,
        dt     :: Real,
        μ      :: Real;
        scheme :: Symbol = :upwind,
        rhs    :: Union{AbstractVector{T},Nothing} = nothing,
        method :: Symbol = :dec,
) :: Vector{T} where {T}
    A    = assemble_transport_operator(mesh, geom, vel; scheme=scheme)
    L    = laplace_matrix(mesh, geom, dec; method=method)
    dt   = T(dt); μ = T(μ)
    nv   = length(uⁿ)
    da   = geom.vertex_dual_areas

    b = copy(uⁿ)
    if rhs !== nothing
        b .+= dt .* rhs
    end

    inv_da = [da[i] > eps(T) ? one(T)/da[i] : zero(T) for i in 1:nv]
    Alhs = spdiagm(0 => ones(T, nv)) + dt * spdiagm(0 => inv_da) * A + dt * μ * L
    return Alhs \ b
end

"""
    step_surface_advection_diffusion_backward_euler(mesh::CurveMesh, ...) -> Vector{T}

Fully implicit backward-Euler step for a curve.
"""
function step_surface_advection_diffusion_backward_euler(
        mesh   :: CurveMesh{T},
        geom   :: CurveGeometry{T},
        dec    :: CurveDEC{T},
        uⁿ     :: AbstractVector{T},
        vel,
        dt     :: Real,
        μ      :: Real;
        scheme :: Symbol = :upwind,
        rhs    :: Union{AbstractVector{T},Nothing} = nothing,
        method :: Symbol = :dec,
) :: Vector{T} where {T}
    A    = assemble_transport_operator(mesh, geom, vel; scheme=scheme)
    L    = laplace_matrix(mesh, geom, dec; method=method)
    dt   = T(dt); μ = T(μ)
    nv   = length(uⁿ)
    dl   = geom.vertex_dual_lengths

    b = copy(uⁿ)
    if rhs !== nothing
        b .+= dt .* rhs
    end

    inv_dl = [dl[i] > eps(T) ? one(T)/dl[i] : zero(T) for i in 1:nv]
    Alhs = spdiagm(0 => ones(T, nv)) + dt * spdiagm(0 => inv_dl) * A + dt * μ * L
    return Alhs \ b
end
