# signed_distance/winding3d.jl – 3D solid-angle winding number for closed meshes.

function _triangle_solid_angle(
    q::SVector{3,T},
    a::SVector{3,T},
    b::SVector{3,T},
    c::SVector{3,T},
) where {T<:AbstractFloat}
    ra = a - q
    rb = b - q
    rc = c - q

    la = norm(ra)
    lb = norm(rb)
    lc = norm(rc)
    (la > eps(T) && lb > eps(T) && lc > eps(T)) || return zero(T)

    triple = dot(ra, cross3(rb, rc))
    denom = la * lb * lc + dot(ra, rb) * lc + dot(rb, rc) * la + dot(rc, ra) * lb
    return T(2) * atan(triple, denom)
end

function _winding_number_surface(point::SVector{3,T}, mesh::SurfaceMesh{T}) where {T<:AbstractFloat}
    is_closed_surface(mesh) || throw(ArgumentError("winding_number on SurfaceMesh requires a closed oriented surface."))
    ω = zero(T)
    for f in mesh.faces
        a, b, c = mesh.points[f[1]], mesh.points[f[2]], mesh.points[f[3]]
        ω += _triangle_solid_angle(point, a, b, c)
    end
    return ω / (T(4) * T(pi))
end
