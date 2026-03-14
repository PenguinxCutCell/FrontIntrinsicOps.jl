# reaction_diffusion.jl – Surface reaction–diffusion equations on static surfaces.
#
# PDE (strong form):  M du/dt + μ L u = R(u, x, t)
#
# where M is the 0-form mass matrix (star0), L = -ΔΓ is the positive-semi-definite
# scalar Laplace–Beltrami, and R is a nonlinear reaction source.
#
# Time-stepping strategies
# ------------------------
# Explicit Euler (debugging):
#   u^{n+1} = u^n + dt * M^{-1} * (R(u^n, t^n) - μ L u^n)
#
# IMEX θ-scheme (reaction explicit, diffusion implicit):
#   (M + dt θ μ L) u^{n+1} = (M - dt (1-θ) μ L) u^n + dt R(u^n, t^n)
#   θ=1 → backward Euler (default, L-stable)
#   θ=0.5 → Crank–Nicolson
#
# Reaction API
# ------------
# Reactions can be passed as:
#   (a) `reaction!(r, u, mesh, geom, t)` – in-place functional form, writes into r.
#   (b) `(u_i, x_i, t) -> scalar` – per-vertex pointwise function, wrapped internally.
#   (c) A Symbol: :fisher_kpp, :linear_decay, :bistable – built-in reactions.
#
# Built-in example reactions
# --------------------------
# Fisher–KPP:      f(u) = α u (1 - u)
# Linear decay:    f(u) = -α u
# Bistable:        f(u) = α u (1 - u)(u - 0.5)

# ─────────────────────────────────────────────────────────────────────────────
# Reaction evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

"""
    evaluate_reaction!(r, reaction, u, mesh, geom, t)

Evaluate the reaction term and write the result into the preallocated vector `r`.

The `reaction` argument can be:
- A callable `reaction!(r, u, mesh, geom, t)` that writes in-place into `r`.
- A callable `(u_i, x_i, t) -> scalar` evaluated pointwise at each vertex.
- `nothing` – `r` is set to zero.

Returns `r`.
"""
function evaluate_reaction!(
        r        :: AbstractVector{T},
        reaction,
        u        :: AbstractVector{T},
        mesh     :: SurfaceMesh{T},
        geom     :: SurfaceGeometry{T},
        t        :: Real,
) :: AbstractVector{T} where {T}
    if reaction === nothing
        fill!(r, zero(T))
    elseif applicable(reaction, r, u, mesh, geom, t)
        # In-place functional form: reaction(r, u, mesh, geom, t)
        reaction(r, u, mesh, geom, t)
    else
        # Pointwise form: reaction(u_i, x_i, t) -> scalar
        nv = length(u)
        @inbounds for i in 1:nv
            r[i] = T(reaction(u[i], mesh.points[i], T(t)))
        end
    end
    return r
end

"""
    evaluate_reaction!(r, reaction, u, mesh, geom, t)

Evaluate reaction on a `CurveMesh`.  Same dispatch as the surface version.
"""
function evaluate_reaction!(
        r        :: AbstractVector{T},
        reaction,
        u        :: AbstractVector{T},
        mesh     :: CurveMesh{T},
        geom     :: CurveGeometry{T},
        t        :: Real,
) :: AbstractVector{T} where {T}
    if reaction === nothing
        fill!(r, zero(T))
    elseif applicable(reaction, r, u, mesh, geom, t)
        reaction(r, u, mesh, geom, t)
    else
        nv = length(u)
        @inbounds for i in 1:nv
            r[i] = T(reaction(u[i], mesh.points[i], T(t)))
        end
    end
    return r
end

# ─────────────────────────────────────────────────────────────────────────────
# Built-in reaction functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    fisher_kpp_reaction(α) -> Function

Return a Fisher–KPP (logistic) reaction function:
    f(u_i, x_i, t) = α * u_i * (1 - u_i)

This models logistic growth with intrinsic rate `α`.  The steady states are
u = 0 (unstable) and u = 1 (stable).
"""
function fisher_kpp_reaction(α::Real)
    return (u_i, x_i, t) -> α * u_i * (one(u_i) - u_i)
end

"""
    linear_decay_reaction(α) -> Function

Return a linear decay reaction function:
    f(u_i, x_i, t) = -α * u_i

For α > 0 this is a stable linear sink term.  Combined with diffusion, the
solution decays to zero exponentially with rate ≥ α.
"""
function linear_decay_reaction(α::Real)
    return (u_i, x_i, t) -> -α * u_i
end

"""
    bistable_reaction(α) -> Function

Return a bistable (Allen–Cahn prototype) reaction function:
    f(u_i, x_i, t) = α * u_i * (1 - u_i) * (u_i - 0.5)

The stable steady states are u = 0 and u = 1; u = 0.5 is unstable.
"""
function bistable_reaction(α::Real)
    return (u_i, x_i, t) -> α * u_i * (one(u_i) - u_i) * (u_i - oftype(u_i, 0.5))
end

# ─────────────────────────────────────────────────────────────────────────────
# Explicit Euler (debugging / reference)
# ─────────────────────────────────────────────────────────────────────────────

"""
    step_surface_reaction_diffusion_explicit(mesh, geom, dec, uⁿ, dt, μ,
                                             reaction, t;
                                             r_buf=nothing, method=:dec)
        -> uⁿ⁺¹

Advance the surface reaction–diffusion equation

    du/dt + μ L u = r(u, x, t)

by one explicit Euler step (strong form):

    u^{n+1} = u^n + dt * (r(u^n, t) - μ L u^n)

This is first-order in time and subject to the parabolic stability constraint
    dt ≤ 1 / (2 μ λ_max)
where λ_max is the largest eigenvalue of L.  Use only for debugging or
when `dt` is very small.  Prefer the IMEX scheme for practical use.

Parameters
----------
- `r_buf`   – optional preallocated buffer of length nv for reaction evaluation.
- `method`  – Laplace assembly method (`:dec` or `:cotan`).
"""
function step_surface_reaction_diffusion_explicit(
        mesh     :: SurfaceMesh{T},
        geom     :: SurfaceGeometry{T},
        dec      :: SurfaceDEC{T},
        uⁿ       :: AbstractVector{T},
        dt       :: Real,
        μ        :: Real,
        reaction,
        t        :: Real;
        r_buf    :: Union{AbstractVector{T},Nothing} = nothing,
        method   :: Symbol = :dec,
) :: Vector{T} where {T}
    nv = length(uⁿ)
    dt = T(dt); μ = T(μ)
    L  = laplace_matrix(mesh, geom, dec; method=method)

    r = r_buf !== nothing ? r_buf : Vector{T}(undef, nv)
    evaluate_reaction!(r, reaction, uⁿ, mesh, geom, T(t))

    # Strong form: u^{n+1} = u^n + dt * (r(u^n) - μ L u^n)
    Lu = L * uⁿ
    return uⁿ .+ dt .* (r .- μ .* Lu)
end

# ─────────────────────────────────────────────────────────────────────────────
# IMEX θ-scheme (reaction explicit, diffusion implicit)
# ─────────────────────────────────────────────────────────────────────────────

"""
    step_surface_reaction_diffusion_imex(mesh, geom, dec, uⁿ, dt, μ,
                                         reaction, t;
                                         θ=1.0,
                                         r_buf=nothing,
                                         factorization=nothing,
                                         method=:dec)
        -> (uⁿ⁺¹, factorization)

Advance the surface reaction–diffusion equation

    du/dt + μ L u = r(u, x, t)

by one IMEX θ-scheme step (strong form):

    (I + dt θ μ L) u^{n+1} = (I - dt (1-θ) μ L) u^n + dt r(u^n, t^n)

- `θ = 1.0`  → backward Euler in diffusion (L-stable, unconditionally stable).
- `θ = 0.5`  → Crank–Nicolson in diffusion (2nd order in time).

The reaction term `r(u^n, t^n)` is always treated **explicitly** (IMEX).
This avoids a nonlinear solve while keeping the diffusion term implicit.
For μ=0 (pure reaction), reduces to explicit Euler: `u^{n+1} = u^n + dt r(u^n)`.

Parameters
----------
- `θ`             – implicitness parameter for diffusion (0 ≤ θ ≤ 1).
- `r_buf`         – optional preallocated reaction buffer (length nv).
- `factorization` – pre-computed factorization of `(I + dt θ μ L)`.
                    Reuse across steps when `dt`, `μ`, `θ` are constant.
- `method`        – Laplace assembly method (`:dec` or `:cotan`).

Returns
-------
`(uⁿ⁺¹, fac)` where `fac` is the LHS factorization (can be reused).
"""
function step_surface_reaction_diffusion_imex(
        mesh          :: SurfaceMesh{T},
        geom          :: SurfaceGeometry{T},
        dec           :: SurfaceDEC{T},
        uⁿ            :: AbstractVector{T},
        dt            :: Real,
        μ             :: Real,
        reaction,
        t             :: Real;
        θ             :: Real = one(T),
        r_buf         :: Union{AbstractVector{T},Nothing} = nothing,
        factorization :: Any    = nothing,
        method        :: Symbol = :dec,
) where {T}
    nv = length(uⁿ)
    dt = T(dt); μ = T(μ); θ = T(θ)

    # Evaluate reaction at current time level
    r = r_buf !== nothing ? r_buf : Vector{T}(undef, nv)
    evaluate_reaction!(r, reaction, uⁿ, mesh, geom, T(t))

    # Assemble LHS factorization if not cached
    # Strong form: A = I + dt θ μ L
    if factorization === nothing
        L   = laplace_matrix(mesh, geom, dec; method=method)
        A   = sparse(T(1) * LinearAlgebra.I(nv)) + (dt * θ * μ) * L
        fac = factorize(A)
    else
        fac = factorization
        L   = laplace_matrix(mesh, geom, dec; method=method)
    end

    # RHS: (I - dt (1-θ) μ L) u^n + dt r(u^n, t^n)
    if θ ≈ one(T)
        # Avoid extra matrix-vector product: (1-θ) = 0
        b = uⁿ .+ dt .* r
    else
        b = uⁿ .- (dt * (one(T) - θ) * μ) .* (L * uⁿ) .+ dt .* r
    end

    uⁿ⁺¹ = fac \ b
    return uⁿ⁺¹, fac
end

"""
    step_surface_reaction_diffusion_imex(mesh::CurveMesh, ...) -> (uⁿ⁺¹, fac)

IMEX reaction–diffusion step for a curve.
"""
function step_surface_reaction_diffusion_imex(
        mesh          :: CurveMesh{T},
        geom          :: CurveGeometry{T},
        dec           :: CurveDEC{T},
        uⁿ            :: AbstractVector{T},
        dt            :: Real,
        μ             :: Real,
        reaction,
        t             :: Real;
        θ             :: Real = one(T),
        r_buf         :: Union{AbstractVector{T},Nothing} = nothing,
        factorization :: Any    = nothing,
        method        :: Symbol = :dec,
) where {T}
    nv = length(uⁿ)
    dt = T(dt); μ = T(μ); θ = T(θ)

    r = r_buf !== nothing ? r_buf : Vector{T}(undef, nv)
    evaluate_reaction!(r, reaction, uⁿ, mesh, geom, T(t))

    # Strong form: A = I + dt θ μ L
    if factorization === nothing
        L   = laplace_matrix(mesh, geom, dec; method=method)
        A   = sparse(T(1) * LinearAlgebra.I(nv)) + (dt * θ * μ) * L
        fac = factorize(A)
    else
        fac = factorization
        L   = laplace_matrix(mesh, geom, dec; method=method)
    end

    if θ ≈ one(T)
        b = uⁿ .+ dt .* r
    else
        b = uⁿ .- (dt * (one(T) - θ) * μ) .* (L * uⁿ) .+ dt .* r
    end

    uⁿ⁺¹ = fac \ b
    return uⁿ⁺¹, fac
end

# ─────────────────────────────────────────────────────────────────────────────
# Full transient solve
# ─────────────────────────────────────────────────────────────────────────────

"""
    solve_surface_reaction_diffusion(mesh, geom, dec, u0, T_end, dt, μ,
                                     reaction;
                                     θ=1.0,
                                     scheme=:imex,
                                     method=:dec,
                                     callback=nothing,
                                     save_every=0)
        -> (u_final, t_final, [history])

Integrate the surface reaction–diffusion equation

    M du/dt + μ L u = R(u, x, t)

from `t = 0` to `t = T_end` with constant time step `dt`.

Parameters
----------
- `u0`         – initial condition (vertex field).
- `T_end`      – final integration time.
- `dt`         – time step size.
- `μ`          – diffusion coefficient.
- `reaction`   – reaction term (see `evaluate_reaction!` for accepted forms).
- `θ`          – implicitness parameter for IMEX scheme (default 1.0 = BE).
- `scheme`     – time-stepping scheme: `:imex` (default) or `:explicit`.
- `method`     – Laplace assembly method.
- `callback`   – optional `callback(u, t, step)` called after each step.
- `save_every` – if > 0, save solution every `save_every` steps and return
                 a `Vector{Vector{T}}` history as third return value.

Returns
-------
- `u_final` – solution at `T_end`.
- `t_final` – actual final time (may differ from `T_end` by < `dt`).
- `history`  – only returned when `save_every > 0`; a vector of `(t, u)` pairs.
"""
function solve_surface_reaction_diffusion(
        mesh       :: Union{SurfaceMesh{T},CurveMesh{T}},
        geom       :: Union{SurfaceGeometry{T},CurveGeometry{T}},
        dec        :: Union{SurfaceDEC{T},CurveDEC{T}},
        u0         :: AbstractVector{T},
        T_end      :: Real,
        dt         :: Real,
        μ          :: Real,
        reaction;
        θ          :: Real   = one(T),
        scheme     :: Symbol = :imex,
        method     :: Symbol = :dec,
        callback             = nothing,
        save_every :: Int    = 0,
) where {T}
    u   = copy(u0)
    t   = zero(T)
    dt  = T(dt)
    fac = nothing
    nv  = length(u)
    r_buf = Vector{T}(undef, nv)

    history = save_every > 0 ? [(zero(T), copy(u))] : nothing
    step    = 0

    while t < T(T_end) - dt * T(1e-12)
        dt_step = min(dt, T(T_end) - t)

        if scheme === :imex
            u, fac = step_surface_reaction_diffusion_imex(
                mesh, geom, dec, u, dt_step, μ, reaction, t;
                θ=θ, r_buf=r_buf, factorization=fac, method=method)
        elseif scheme === :explicit
            u = step_surface_reaction_diffusion_explicit(
                mesh, geom, dec, u, dt_step, μ, reaction, t;
                r_buf=r_buf, method=method)
            # Factorization not used in explicit scheme
        else
            error("solve_surface_reaction_diffusion: unknown scheme $(repr(scheme))")
        end

        t    += dt_step
        step += 1

        callback !== nothing && callback(u, t, step)

        if save_every > 0 && step % save_every == 0
            push!(history, (t, copy(u)))
        end
    end

    if save_every > 0
        return u, t, history
    else
        return u, t
    end
end
