# signed_distance/pseudonormals2d.jl – Curve validity, closure, and pseudonormals.

"""
    is_closed_curve(mesh::CurveMesh) -> Bool

Return `true` when all vertices have degree 2 and the curve is a closed loop.
"""
function is_closed_curve(mesh::CurveMesh)
    nv = length(mesh.points)
    ne = length(mesh.edges)
    nv >= 2 || return false
    ne >= 2 || return false

    ve = vertex_to_edges(mesh)
    all(length(v) == 2 for v in ve) || return false

    adj = [Int[] for _ in 1:nv]
    for e in mesh.edges
        i, j = e
        push!(adj[i], j)
        push!(adj[j], i)
    end
    visited = falses(nv)
    stack = Int[1]
    visited[1] = true
    while !isempty(stack)
        v = pop!(stack)
        for w in adj[v]
            if !visited[w]
                visited[w] = true
                push!(stack, w)
            end
        end
    end
    return all(visited)
end

function _validate_curve_mesh(mesh::CurveMesh{T}) where {T<:AbstractFloat}
    nv = length(mesh.points)
    ne = length(mesh.edges)
    nv >= 2 || throw(ArgumentError("CurveMesh must contain at least 2 vertices."))
    ne >= 1 || throw(ArgumentError("CurveMesh must contain at least 1 segment."))

    for (ei, e) in enumerate(mesh.edges)
        i, j = e
        (1 <= i <= nv && 1 <= j <= nv) || throw(ArgumentError("Curve edge $ei references invalid vertex index."))
        i != j || throw(ArgumentError("Curve edge $ei is degenerate (same endpoint)."))
        norm(mesh.points[j] - mesh.points[i]) > eps(T) || throw(ArgumentError("Curve edge $ei has zero geometric length."))
    end

    ve = vertex_to_edges(mesh)
    deg = [length(v) for v in ve]
    if is_closed_curve(mesh)
        return true
    end

    all(d -> (d == 1 || d == 2), deg) || throw(ArgumentError("Open CurveMesh must have only degree-1 endpoints and degree-2 interior vertices."))
    count(==(1), deg) == 2 || throw(ArgumentError("Open CurveMesh must have exactly two endpoints."))
    return false
end

function _prepare_curve_sign_data(mesh::CurveMesh{T}) where {T<:AbstractFloat}
    closed = _validate_curve_mesh(mesh)
    nv = length(mesh.points)
    ne = length(mesh.edges)

    edge_normals = Vector{SVector{2,T}}(undef, ne)
    for (ei, e) in enumerate(mesh.edges)
        a, b = mesh.points[e[1]], mesh.points[e[2]]
        t = normalize_safe(b - a)
        edge_normals[ei] = _rot90cw(t)
    end

    v2e = vertex_to_edges(mesh)
    vertex_normals = Vector{SVector{2,T}}(undef, nv)
    z = zero(SVector{2,T})
    for v in 1:nv
        inc = v2e[v]
        isempty(inc) && throw(ArgumentError("Curve vertex $v is isolated."))
        acc = z
        for ei in inc
            acc += edge_normals[ei]
        end
        vertex_normals[v] = _normalize_with_fallback(acc, edge_normals[inc[1]])
    end

    return closed, edge_normals, vertex_normals
end
