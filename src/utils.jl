# utils.jl – Low-level helper functions used across the package.

"""
    cross2d(a, b) -> Real

2D cross product (scalar z-component of 3D cross product) of two 2-vectors.
"""
@inline function cross2d(a::SVector{2,T}, b::SVector{2,T}) where {T}
    return a[1]*b[2] - a[2]*b[1]
end

"""
    normalize_safe(v) -> SVector

Normalize a vector; return a zero vector of the same type if the input has
near-zero length (to avoid NaN propagation on degenerate elements).
"""
@inline function normalize_safe(v::SVector{N,T}) where {N,T}
    n = norm(v)
    n < eps(T) && return zero(SVector{N,T})
    return v / n
end

"""
    cotangent(a, b) -> Real

Cotangent of the angle between vectors `a` and `b`, used in the cotan
Laplace–Beltrami formula.
"""
@inline function cotangent(a::SVector{N,T}, b::SVector{N,T}) where {N,T}
    cosine = dot(a, b)
    sine   = norm(cross(a, b))     # for 3-vectors; scalar magnitude
    return cosine / (sine + eps(T))
end

# Specialised 2-vector version (cross product returns scalar).
@inline function cotangent(a::SVector{2,T}, b::SVector{2,T}) where {T}
    cosine = dot(a, b)
    sine   = abs(cross2d(a, b))
    return cosine / (sine + eps(T))
end

"""
    cross(a, b) -> SVector{3}

3D cross product as a StaticVector (thin wrapper around LinearAlgebra.cross).
"""
@inline function cross3(a::SVector{3,T}, b::SVector{3,T}) where {T}
    return SVector{3,T}(
        a[2]*b[3] - a[3]*b[2],
        a[3]*b[1] - a[1]*b[3],
        a[1]*b[2] - a[2]*b[1],
    )
end
