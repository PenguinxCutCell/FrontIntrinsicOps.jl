# signed_distance/primitives3d.jl – Exact point/triangle closest-point queries.

"""
    closest_point_triangle(q, a, b, c)

Exact closest-point query from point `q` to triangle `(a,b,c)` in 3D using
region tests (Ericson, Real-Time Collision Detection).

Returns `(sqdist, cp, bary, feature)` where:
- `bary = (λ1, λ2, λ3)` are barycentric coordinates,
- `feature ∈ (:face, :edge12, :edge23, :edge31, :vertex1, :vertex2, :vertex3)`.
"""
function closest_point_triangle(
    q::SVector{3,T},
    a::SVector{3,T},
    b::SVector{3,T},
    c::SVector{3,T},
) where {T<:AbstractFloat}
    ab = b - a
    ac = c - a
    ap = q - a

    d1 = dot(ab, ap)
    d2 = dot(ac, ap)
    if d1 <= zero(T) && d2 <= zero(T)
        cp = a
        bary = SVector{3,T}(one(T), zero(T), zero(T))
        return dot(q - cp, q - cp), cp, bary, :vertex1
    end

    bp = q - b
    d3 = dot(ab, bp)
    d4 = dot(ac, bp)
    if d3 >= zero(T) && d4 <= d3
        cp = b
        bary = SVector{3,T}(zero(T), one(T), zero(T))
        return dot(q - cp, q - cp), cp, bary, :vertex2
    end

    vc = d1 * d4 - d3 * d2
    if vc <= zero(T) && d1 >= zero(T) && d3 <= zero(T)
        v = d1 / (d1 - d3)
        cp = a + v * ab
        bary = SVector{3,T}(one(T) - v, v, zero(T))
        return dot(q - cp, q - cp), cp, bary, :edge12
    end

    cpv = q - c
    d5 = dot(ab, cpv)
    d6 = dot(ac, cpv)
    if d6 >= zero(T) && d5 <= d6
        cp0 = c
        bary = SVector{3,T}(zero(T), zero(T), one(T))
        return dot(q - cp0, q - cp0), cp0, bary, :vertex3
    end

    vb = d5 * d2 - d1 * d6
    if vb <= zero(T) && d2 >= zero(T) && d6 <= zero(T)
        w = d2 / (d2 - d6)
        cp0 = a + w * ac
        bary = SVector{3,T}(one(T) - w, zero(T), w)
        return dot(q - cp0, q - cp0), cp0, bary, :edge31
    end

    va = d3 * d6 - d5 * d4
    if va <= zero(T) && (d4 - d3) >= zero(T) && (d5 - d6) >= zero(T)
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        cp0 = b + w * (c - b)
        bary = SVector{3,T}(zero(T), one(T) - w, w)
        return dot(q - cp0, q - cp0), cp0, bary, :edge23
    end

    denom = one(T) / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    u = one(T) - v - w
    cp0 = u * a + v * b + w * c
    bary = SVector{3,T}(u, v, w)

    lscale = max(norm(ab), norm(c - b), norm(a - c), one(T))
    btol = _dist_feature_tol(T, lscale) / lscale
    λ1, λ2, λ3 = bary
    z1 = abs(λ1) <= btol
    z2 = abs(λ2) <= btol
    z3 = abs(λ3) <= btol

    feature = if !z1 && !z2 && !z3
        :face
    elseif z1 && !z2 && !z3
        :edge23
    elseif z2 && !z1 && !z3
        :edge31
    elseif z3 && !z1 && !z2
        :edge12
    elseif z1 && z2
        :vertex3
    elseif z2 && z3
        :vertex1
    else
        :vertex2
    end

    return dot(q - cp0, q - cp0), cp0, bary, feature
end

function _closest_surface_primitive(
    cache::SignedDistanceCache{3,T,<:SurfaceMesh{T}},
    q::SVector{3,T},
    primitive::Int,
) where {T<:AbstractFloat}
    tri = cache.mesh.faces[primitive]
    a, b, c = cache.mesh.points[tri[1]], cache.mesh.points[tri[2]], cache.mesh.points[tri[3]]
    sqdist, cp, _, feature = closest_point_triangle(q, a, b, c)

    if feature === :face
        return sqdist, cp, :face, primitive
    elseif feature === :edge12
        eid = cache.primitive_to_feature_data.face_edges[primitive][1]
        return sqdist, cp, :edge, eid
    elseif feature === :edge23
        eid = cache.primitive_to_feature_data.face_edges[primitive][2]
        return sqdist, cp, :edge, eid
    elseif feature === :edge31
        eid = cache.primitive_to_feature_data.face_edges[primitive][3]
        return sqdist, cp, :edge, eid
    elseif feature === :vertex1
        return sqdist, cp, :vertex, tri[1]
    elseif feature === :vertex2
        return sqdist, cp, :vertex, tri[2]
    else
        return sqdist, cp, :vertex, tri[3]
    end
end
