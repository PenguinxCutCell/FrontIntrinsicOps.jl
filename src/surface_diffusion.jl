# surface_diffusion.jl – Scalar diffusion and Poisson/Helmholtz solvers on
#                         static surfaces.
#
# Sign convention:  L = -Δ_Γ  (positive semi-definite).
#
# PDE forms
# ---------
# Poisson:            L u      = f
# Helmholtz:         (L + α M) u = f
# Diffusion (strong): du/dt + μ L u = g
# Diffusion (weak):   M du/dt + μ K u = F   (K = M L, F = M g)
#
# Backward-Euler discretisation (I + dt μ L) uⁿ⁺¹ = uⁿ + dt g
# is used for the transient steps.
#
# Gauge on closed manifolds
# -------------------------
# For pure Poisson, L has a 1-D constant nullspace.  We enforce compatibility
# (∫ f dA = 0), solve with a small Tikhonov regularisation (ε‖u‖² added),
# and then project the solution to zero mean.

# ─────────────────────────────────────────────────────────────────────────────
# Laplace matrix helper (thin wrapper for reuse)
# ─────────────────────────────────────────────────────────────────────────────

"""
    laplace_matrix(mesh::CurveMesh, geom::CurveGeometry, dec::CurveDEC;
                   method=:dec) -> SparseMatrixCSC

Return the scalar Laplace–Beltrami matrix L = dec.lap0.
The `method` keyword is accepted for API consistency but ignored (only :dec is
available for curves).
"""
function laplace_matrix(
        :: CurveMesh{T},
        :: CurveGeometry{T},
        dec :: CurveDEC{T};
        method :: Symbol = :dec,
) :: SparseMatrixCSC{T,Int} where {T}
    return dec.lap0
end

"""
    laplace_matrix(mesh::SurfaceMesh, geom::SurfaceGeometry, dec::SurfaceDEC;
                   method=:dec) -> SparseMatrixCSC

Return the scalar Laplace–Beltrami matrix.

- `method=:dec`   – return `dec.lap0` (assembled during `build_dec`).
- `method=:cotan` – re-assemble with the direct cotan formula.
"""
function laplace_matrix(
        mesh   :: SurfaceMesh{T},
        geom   :: SurfaceGeometry{T},
        dec    :: SurfaceDEC{T};
        method :: Symbol = :dec,
) :: SparseMatrixCSC{T,Int} where {T}
    if method === :dec
        return dec.lap0
    elseif method === :cotan
        return build_laplace_beltrami(mesh, geom; method=:cotan)
    else
        error("laplace_matrix: unknown method $(repr(method)). Use :dec or :cotan.")
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Diffusion operator assembly
# ─────────────────────────────────────────────────────────────────────────────

"""
    assemble_diffusion_operator(mesh::CurveMesh, geom::CurveGeometry,
                                dec::CurveDEC; method=:dec) -> SparseMatrixCSC

Return the diffusion operator L for a curve (identical to `dec.lap0`).
"""
function assemble_diffusion_operator(
        mesh   :: CurveMesh{T},
        geom   :: CurveGeometry{T},
        dec    :: CurveDEC{T};
        method :: Symbol = :dec,
) :: SparseMatrixCSC{T,Int} where {T}
    return laplace_matrix(mesh, geom, dec; method=method)
end

"""
    assemble_diffusion_operator(mesh::SurfaceMesh, geom::SurfaceGeometry,
                                dec::SurfaceDEC; method=:dec) -> SparseMatrixCSC

Return the diffusion operator L for a surface (the scalar Laplace–Beltrami).
"""
function assemble_diffusion_operator(
        mesh   :: SurfaceMesh{T},
        geom   :: SurfaceGeometry{T},
        dec    :: SurfaceDEC{T};
        method :: Symbol = :dec,
) :: SparseMatrixCSC{T,Int} where {T}
    return laplace_matrix(mesh, geom, dec; method=method)
end

# ─────────────────────────────────────────────────────────────────────────────
# Poisson solver:  L u = f
# ─────────────────────────────────────────────────────────────────────────────

"""
    solve_surface_poisson(mesh, geom, dec, f;
                          gauge=:zero_mean, method=:dec,
                          reg=1e-10) -> Vector{T}

Solve the surface Poisson problem  L u = f  on a closed manifold.

Parameters
----------
- `f`         – right-hand side vertex field.
- `gauge`     – how to fix the constant nullspace.  Only `:zero_mean` is
                currently supported: the compatibility condition ∫ f dA = 0
                is enforced first, then the system is regularised with a
                small Tikhonov shift, and finally the solution is projected
                to zero mean.
- `method`    – Laplace assembly method (`:dec` or `:cotan`).
- `reg`       – Tikhonov regularisation parameter ε for the shift ε I.

Returns the solution vector `u` with ⟨u⟩ = 0.
"""
function solve_surface_poisson(
        mesh   :: SurfaceMesh{T},
        geom   :: SurfaceGeometry{T},
        dec    :: SurfaceDEC{T},
        f      :: AbstractVector{T};
        gauge  :: Symbol = :zero_mean,
        method :: Symbol = :dec,
        reg    :: Real   = 1e-10,
) :: Vector{T} where {T}
    gauge === :zero_mean ||
        error("solve_surface_poisson: unsupported gauge $(repr(gauge)). Use :zero_mean.")
    nv = length(mesh.points)
    L  = laplace_matrix(mesh, geom, dec; method=method)
    M  = mass_matrix(mesh, geom)

    # Enforce compatibility: project f to zero mean
    f_proj = copy(f)
    enforce_compatibility!(f_proj, mesh, geom)

    # Regularise: solve (L + ε M) u = f_proj
    ε  = T(reg)
    A  = L + ε * M
    u  = Array(A) \ f_proj   # dense solve for small systems; sparse for large

    # Project solution to zero mean
    zero_mean_projection!(u, mesh, geom)
    return u
end

# ─────────────────────────────────────────────────────────────────────────────
# Helmholtz solver:  (L + α M) u = f
# ─────────────────────────────────────────────────────────────────────────────

"""
    solve_surface_helmholtz(mesh, geom, dec, f, α;
                            method=:dec) -> Vector{T}

Solve the surface Helmholtz problem  (L + α M) u = f.

For α > 0 the system is positive definite; no gauge treatment is needed.
Uses a direct sparse factorization.

Parameters
----------
- `f`      – right-hand side vertex field.
- `α`      – shift parameter (must be > 0 for well-posedness).
- `method` – Laplace assembly method (`:dec` or `:cotan`).
"""
function solve_surface_helmholtz(
        mesh   :: SurfaceMesh{T},
        geom   :: SurfaceGeometry{T},
        dec    :: SurfaceDEC{T},
        f      :: AbstractVector{T},
        α      :: Real;
        method :: Symbol = :dec,
) :: Vector{T} where {T}
    α > 0 || @warn "solve_surface_helmholtz: α ≤ 0 may give singular system"
    L = laplace_matrix(mesh, geom, dec; method=method)
    M = mass_matrix(mesh, geom)
    A = L + T(α) * M
    return A \ f
end

# ─────────────────────────────────────────────────────────────────────────────
# Transient diffusion: M du/dt + μ L u = rhs
# ─────────────────────────────────────────────────────────────────────────────

"""
    step_surface_diffusion_backward_euler(mesh, geom, dec, uⁿ, dt, μ;
                                          rhs=nothing,
                                          factorization=nothing,
                                          method=:dec)
        -> (uⁿ⁺¹, factorization)

Advance the surface diffusion equation

    du/dt + μ L u = g

(equivalently, the weak form  M du/dt + μ K u = M g  where K = M L)
by one backward-Euler step of size `dt`:

    (I + dt μ L) uⁿ⁺¹ = uⁿ + dt g

where `I` is the identity and `L` is the scalar Laplace–Beltrami matrix.

Parameters
----------
- `uⁿ`            – solution at time step n.
- `dt`            – time step size.
- `μ`             – diffusion coefficient.
- `rhs`           – optional pointwise source term `g` (vertex vector).
                    If provided, the load vector is added as `dt * g`.
- `factorization` – optional pre-computed factorization of `(I + dt μ L)`.
                    Pass the returned factorization to reuse across steps.
- `method`        – Laplace assembly method (`:dec` or `:cotan`).

Returns
-------
`(uⁿ⁺¹, fac)` where `fac` is the factorization of `(I + dt μ L)` (can be
reused if `dt` and `μ` do not change).
"""
function step_surface_diffusion_backward_euler(
        mesh          :: SurfaceMesh{T},
        geom          :: SurfaceGeometry{T},
        dec           :: SurfaceDEC{T},
        uⁿ            :: AbstractVector{T},
        dt            :: Real,
        μ             :: Real;
        rhs           :: Union{AbstractVector{T},Nothing} = nothing,
        factorization :: Any    = nothing,
        method        :: Symbol = :dec,
) where {T}
    L  = laplace_matrix(mesh, geom, dec; method=method)
    dt = T(dt); μ = T(μ)
    nv = length(uⁿ)

    # Right-hand side:  uⁿ + dt * g
    b = copy(uⁿ)
    if rhs !== nothing
        b .+= dt .* rhs
    end

    # Build or reuse system matrix:  I + dt μ L
    if factorization === nothing
        A   = sparse(T(1) * LinearAlgebra.I(nv)) + dt * μ * L
        fac = factorize(A)
    else
        fac = factorization
    end

    uⁿ⁺¹ = fac \ b
    return uⁿ⁺¹, fac
end

"""
    step_surface_diffusion_crank_nicolson(mesh, geom, dec, uⁿ, dt, μ;
                                          rhs=nothing,
                                          factorization=nothing,
                                          method=:dec)
        -> (uⁿ⁺¹, factorization)

Advance the surface diffusion equation by one Crank–Nicolson step:

    (I + (dt/2) μ L) uⁿ⁺¹ = (I - (dt/2) μ L) uⁿ + dt g

Second-order in time.  The factorization of the left-hand side matrix can be
cached and reused across steps when `dt` and `μ` are constant.
"""
function step_surface_diffusion_crank_nicolson(
        mesh          :: SurfaceMesh{T},
        geom          :: SurfaceGeometry{T},
        dec           :: SurfaceDEC{T},
        uⁿ            :: AbstractVector{T},
        dt            :: Real,
        μ             :: Real;
        rhs           :: Union{AbstractVector{T},Nothing} = nothing,
        factorization :: Any    = nothing,
        method        :: Symbol = :dec,
) where {T}
    L  = laplace_matrix(mesh, geom, dec; method=method)
    dt = T(dt); μ = T(μ)
    half = dt / 2
    nv   = length(uⁿ)
    Iv   = sparse(T(1) * LinearAlgebra.I(nv))

    # Right-hand side: (I - (dt/2) μ L) uⁿ + dt g
    b = (Iv - half * μ * L) * uⁿ
    if rhs !== nothing
        b .+= dt .* rhs
    end

    # Left-hand side: I + (dt/2) μ L
    if factorization === nothing
        A   = Iv + half * μ * L
        fac = factorize(A)
    else
        fac = factorization
    end

    uⁿ⁺¹ = fac \ b
    return uⁿ⁺¹, fac
end

# ─────────────────────────────────────────────────────────────────────────────
# Curve versions (thin wrappers for API symmetry)
# ─────────────────────────────────────────────────────────────────────────────

"""
    solve_surface_poisson(mesh::CurveMesh, geom::CurveGeometry, dec::CurveDEC, f;
                          gauge=:zero_mean, method=:dec, reg=1e-10) -> Vector{T}

Solve  L u = f  on a closed curve (1-D analogue).
"""
function solve_surface_poisson(
        mesh   :: CurveMesh{T},
        geom   :: CurveGeometry{T},
        dec    :: CurveDEC{T},
        f      :: AbstractVector{T};
        gauge  :: Symbol = :zero_mean,
        method :: Symbol = :dec,
        reg    :: Real   = 1e-10,
) :: Vector{T} where {T}
    L = laplace_matrix(mesh, geom, dec; method=method)
    M = mass_matrix(mesh, geom)
    f_proj = copy(f)
    μ_f = weighted_mean(mesh, geom, f_proj)
    f_proj .-= μ_f
    ε = T(reg)
    A = L + ε * M
    u = Array(A) \ f_proj
    zero_mean_projection!(u, mesh, geom)
    return u
end

"""
    solve_surface_helmholtz(mesh::CurveMesh, geom::CurveGeometry, dec::CurveDEC,
                            f, α; method=:dec) -> Vector{T}

Solve  (L + α M) u = f  on a curve.
"""
function solve_surface_helmholtz(
        mesh   :: CurveMesh{T},
        geom   :: CurveGeometry{T},
        dec    :: CurveDEC{T},
        f      :: AbstractVector{T},
        α      :: Real;
        method :: Symbol = :dec,
) :: Vector{T} where {T}
    L = laplace_matrix(mesh, geom, dec; method=method)
    M = mass_matrix(mesh, geom)
    A = L + T(α) * M
    return A \ f
end

"""
    step_surface_diffusion_backward_euler(mesh::CurveMesh, ...) -> (uⁿ⁺¹, fac)

Backward-Euler diffusion step for a curve.  See surface version for details.
"""
function step_surface_diffusion_backward_euler(
        mesh          :: CurveMesh{T},
        geom          :: CurveGeometry{T},
        dec           :: CurveDEC{T},
        uⁿ            :: AbstractVector{T},
        dt            :: Real,
        μ             :: Real;
        rhs           :: Union{AbstractVector{T},Nothing} = nothing,
        factorization :: Any    = nothing,
        method        :: Symbol = :dec,
) where {T}
    L  = laplace_matrix(mesh, geom, dec; method=method)
    dt = T(dt); μ = T(μ)
    nv = length(uⁿ)
    b  = copy(uⁿ)
    if rhs !== nothing
        b .+= dt .* rhs
    end
    if factorization === nothing
        A   = sparse(T(1) * LinearAlgebra.I(nv)) + dt * μ * L
        fac = factorize(A)
    else
        fac = factorization
    end
    uⁿ⁺¹ = fac \ b
    return uⁿ⁺¹, fac
end

"""
    step_surface_diffusion_crank_nicolson(mesh::CurveMesh, ...) -> (uⁿ⁺¹, fac)

Crank–Nicolson diffusion step for a curve.
"""
function step_surface_diffusion_crank_nicolson(
        mesh          :: CurveMesh{T},
        geom          :: CurveGeometry{T},
        dec           :: CurveDEC{T},
        uⁿ            :: AbstractVector{T},
        dt            :: Real,
        μ             :: Real;
        rhs           :: Union{AbstractVector{T},Nothing} = nothing,
        factorization :: Any    = nothing,
        method        :: Symbol = :dec,
) where {T}
    L    = laplace_matrix(mesh, geom, dec; method=method)
    dt   = T(dt); μ = T(μ)
    half = dt / 2
    nv   = length(uⁿ)
    Iv   = sparse(T(1) * LinearAlgebra.I(nv))
    b    = (Iv - half * μ * L) * uⁿ
    if rhs !== nothing
        b .+= dt .* rhs
    end
    if factorization === nothing
        A   = Iv + half * μ * L
        fac = factorize(A)
    else
        fac = factorization
    end
    uⁿ⁺¹ = fac \ b
    return uⁿ⁺¹, fac
end
