# perf_utils.jl – Allocation-conscious helpers for PDE kernels.
#
# Provides:
# - mul_diag_left!  / mul_diag_right!  : in-place diagonal scaling
# - apply_laplace!                      : in-place Laplace application
# - weighted_l2_error                   : weighted L² error norm
# - weighted_linf_error                 : weighted L∞ error norm

# ─────────────────────────────────────────────────────────────────────────────
# Diagonal scaling helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    mul_diag_left!(y, d, x) -> y

In-place computation of `y .= d .* x` where `d` is a vector of diagonal
entries.  Equivalent to `y = Diagonal(d) * x` but without allocating the
diagonal matrix.
"""
function mul_diag_left!(
        y :: AbstractVector,
        d :: AbstractVector,
        x :: AbstractVector,
)
    @inbounds for i in eachindex(y)
        y[i] = d[i] * x[i]
    end
    return y
end

"""
    mul_diag_right!(Y, X, d) -> Y

In-place computation of `Y = X * Diagonal(d)` for matrices.  Each column j of
`Y` is `X[:, j] * d[j]`.
"""
function mul_diag_right!(
        Y :: AbstractMatrix,
        X :: AbstractMatrix,
        d :: AbstractVector,
)
    n, m = size(X)
    @inbounds for j in 1:m
        dj = d[j]
        for i in 1:n
            Y[i, j] = X[i, j] * dj
        end
    end
    return Y
end

# ─────────────────────────────────────────────────────────────────────────────
# In-place Laplace application
# ─────────────────────────────────────────────────────────────────────────────

"""
    apply_laplace!(y, dec::SurfaceDEC, u) -> y

In-place application of the scalar Laplace–Beltrami operator:  y = L u.

Uses `mul!` to avoid allocation.
"""
function apply_laplace!(
        y   :: AbstractVector,
        dec :: SurfaceDEC,
        u   :: AbstractVector,
)
    mul!(y, dec.lap0, u)
    return y
end

"""
    apply_laplace!(y, dec::CurveDEC, u) -> y

In-place application of the scalar Laplace–Beltrami operator for a curve.
"""
function apply_laplace!(
        y   :: AbstractVector,
        dec :: CurveDEC,
        u   :: AbstractVector,
)
    mul!(y, dec.lap0, u)
    return y
end

# ─────────────────────────────────────────────────────────────────────────────
# Error norms
# ─────────────────────────────────────────────────────────────────────────────

"""
    weighted_l2_error(mesh::SurfaceMesh, geom::SurfaceGeometry,
                      u, u_exact) -> T

Compute the area-weighted relative L² error:

    e = sqrt( Σ_v A_v (u_v - u_exact_v)² ) / sqrt( Σ_v A_v u_exact_v² )

Returns 0 if the denominator is zero.
"""
function weighted_l2_error(
        mesh    :: SurfaceMesh{T},
        geom    :: SurfaceGeometry{T},
        u       :: AbstractVector{T},
        u_exact :: AbstractVector{T},
) :: T where {T}
    da   = geom.vertex_dual_areas
    num  = zero(T)
    den  = zero(T)
    @inbounds for i in eachindex(u)
        diff  = u[i] - u_exact[i]
        num  += da[i] * diff^2
        den  += da[i] * u_exact[i]^2
    end
    den < eps(T) && return zero(T)
    return sqrt(num / den)
end

"""
    weighted_l2_error(mesh::CurveMesh, geom::CurveGeometry,
                      u, u_exact) -> T

Length-weighted relative L² error on a curve.
"""
function weighted_l2_error(
        mesh    :: CurveMesh{T},
        geom    :: CurveGeometry{T},
        u       :: AbstractVector{T},
        u_exact :: AbstractVector{T},
) :: T where {T}
    dl   = geom.vertex_dual_lengths
    num  = zero(T)
    den  = zero(T)
    @inbounds for i in eachindex(u)
        diff  = u[i] - u_exact[i]
        num  += dl[i] * diff^2
        den  += dl[i] * u_exact[i]^2
    end
    den < eps(T) && return zero(T)
    return sqrt(num / den)
end

"""
    weighted_linf_error(mesh, geom, u, u_exact) -> T

Compute the relative L∞ error:

    e = max_v |u_v - u_exact_v| / max_v |u_exact_v|
"""
function weighted_linf_error(
        :: Union{CurveMesh,SurfaceMesh},
        :: Union{CurveGeometry,SurfaceGeometry},
        u       :: AbstractVector{T},
        u_exact :: AbstractVector{T},
) :: T where {T}
    num = maximum(abs, u .- u_exact)
    den = maximum(abs, u_exact)
    den < eps(T) && return zero(T)
    return num / den
end
