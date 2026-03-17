# signed_distance/primitives2d.jl – Exact point/segment closest-point queries.

"""
    closest_point_segment(q, a, b)

Exact closest-point query from `q` to segment `[a,b]`.

Returns `(sqdist, c, t, feature)` where `t ∈ [0,1]` and `feature ∈ (:vertex0, :vertex1, :edge)`.
"""
function closest_point_segment(q::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T}) where {T<:AbstractFloat}
    ab = b - a
    lab2 = dot(ab, ab)
    lab2 > eps(T) || throw(ArgumentError("Degenerate curve segment detected (zero length)."))

    t = clamp(dot(q - a, ab) / lab2, zero(T), one(T))
    c = a + t * ab
    d = q - c
    sqdist = dot(d, d)

    lt = sqrt(lab2)
    ttol = _dist_feature_tol(T, lt) / (lt + eps(T))
    feature = if t <= ttol
        :vertex0
    elseif t >= one(T) - ttol
        :vertex1
    else
        :edge
    end

    return sqdist, c, t, feature
end

function _closest_curve_primitive(
    cache::SignedDistanceCache{2,T,<:CurveMesh{T}},
    q::SVector{2,T},
    primitive::Int,
) where {T<:AbstractFloat}
    e = cache.mesh.edges[primitive]
    a = cache.mesh.points[e[1]]
    b = cache.mesh.points[e[2]]
    sqdist, c, _, feature = closest_point_segment(q, a, b)

    if feature === :edge
        return sqdist, c, :edge, primitive
    elseif feature === :vertex0
        return sqdist, c, :vertex, e[1]
    else
        return sqdist, c, :vertex, e[2]
    end
end
