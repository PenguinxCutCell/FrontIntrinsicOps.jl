# signed_distance/winding2d.jl – 2D winding number for closed oriented curves.

function _is_left(a::SVector{2,T}, b::SVector{2,T}, q::SVector{2,T}) where {T<:AbstractFloat}
    return cross2d(b - a, q - a)
end

function _winding_number_curve(point::SVector{2,T}, mesh::CurveMesh{T}) where {T<:AbstractFloat}
    is_closed_curve(mesh) || throw(ArgumentError("winding_number on CurveMesh requires a closed curve."))
    wn = 0
    qy = point[2]
    for e in mesh.edges
        a = mesh.points[e[1]]
        b = mesh.points[e[2]]
        if a[2] <= qy
            if b[2] > qy && _is_left(a, b, point) > zero(T)
                wn += 1
            end
        else
            if b[2] <= qy && _is_left(a, b, point) < zero(T)
                wn -= 1
            end
        end
    end
    return wn
end
