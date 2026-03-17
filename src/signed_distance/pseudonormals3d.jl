# signed_distance/pseudonormals3d.jl – Surface validity, closure, and pseudonormals.

"""
    is_closed_surface(mesh::SurfaceMesh) -> Bool

Return `true` when each undirected edge has exactly two incident faces and face
orientations are consistent.
"""
function is_closed_surface(mesh::SurfaceMesh)
    topo = build_topology(mesh)
    return all(length(ef) == 2 for ef in topo.edge_faces) && has_consistent_orientation(mesh)
end

function _validate_surface_mesh(mesh::SurfaceMesh{T}) where {T<:AbstractFloat}
    nv = length(mesh.points)
    nf = length(mesh.faces)
    nv >= 3 || throw(ArgumentError("SurfaceMesh must contain at least 3 vertices."))
    nf >= 1 || throw(ArgumentError("SurfaceMesh must contain at least 1 face."))

    for (fi, f) in enumerate(mesh.faces)
        i, j, k = f
        (1 <= i <= nv && 1 <= j <= nv && 1 <= k <= nv) || throw(ArgumentError("Surface face $fi references invalid vertex index."))
        length(unique((i, j, k))) == 3 || throw(ArgumentError("Surface face $fi is degenerate (repeated vertex index)."))
        a, b, c = mesh.points[i], mesh.points[j], mesh.points[k]
        area2 = norm(cross3(b - a, c - a))
        area2 > eps(T) || throw(ArgumentError("Surface face $fi is geometrically degenerate."))
    end

    topo = build_topology(mesh)
    for (ei, faces) in enumerate(topo.edge_faces)
        length(faces) <= 2 || throw(ArgumentError("Nonmanifold edge detected at edge $ei (more than 2 incident faces)."))
    end
    return topo
end

function _prepare_surface_sign_data(mesh::SurfaceMesh{T}) where {T<:AbstractFloat}
    topo = _validate_surface_mesh(mesh)
    nf = length(mesh.faces)
    ne = length(topo.edges)
    nv = length(mesh.points)

    face_normals = Vector{SVector{3,T}}(undef, nf)
    face_areas = Vector{T}(undef, nf)
    for (fi, f) in enumerate(mesh.faces)
        a, b, c = mesh.points[f[1]], mesh.points[f[2]], mesh.points[f[3]]
        n = cross3(b - a, c - a)
        area2 = norm(n)
        face_areas[fi] = T(0.5) * area2
        face_normals[fi] = n / area2
    end

    edge_normals = Vector{SVector{3,T}}(undef, ne)
    for ei in 1:ne
        inc = topo.edge_faces[ei]
        if length(inc) == 2
            edge_normals[ei] = normalize_safe(face_normals[inc[1]] + face_normals[inc[2]])
        elseif length(inc) == 1
            edge_normals[ei] = face_normals[inc[1]]
        else
            throw(ArgumentError("Edge $ei has zero incident faces."))
        end
    end

    vertex_faces = topo.vertex_faces
    vertex_normals = Vector{SVector{3,T}}(undef, nv)
    z = zero(SVector{3,T})
    for vi in 1:nv
        incf = vertex_faces[vi]
        isempty(incf) && throw(ArgumentError("Surface vertex $vi is isolated."))
        acc_angle = z
        acc_area = z
        for fi in incf
            tri = mesh.faces[fi]
            i1, i2, i3 = tri
            if vi == i1
                u = mesh.points[i2] - mesh.points[i1]
                v = mesh.points[i3] - mesh.points[i1]
            elseif vi == i2
                u = mesh.points[i3] - mesh.points[i2]
                v = mesh.points[i1] - mesh.points[i2]
            else
                u = mesh.points[i1] - mesh.points[i3]
                v = mesh.points[i2] - mesh.points[i3]
            end
            nu = norm(u)
            nv2 = norm(v)
            ang = if nu > eps(T) && nv2 > eps(T)
                acos(clamp(dot(u, v) / (nu * nv2), -one(T), one(T)))
            else
                zero(T)
            end
            fn = face_normals[fi]
            acc_angle += ang * fn
            acc_area += face_areas[fi] * fn
        end
        vertex_normals[vi] = _normalize_with_fallback(acc_angle, acc_area)
        if norm(vertex_normals[vi]) <= eps(T)
            vertex_normals[vi] = face_normals[incf[1]]
        end
    end

    feature_data = (
        edges = topo.edges,
        face_edges = topo.face_edges,
        edge_faces = topo.edge_faces,
        vertex_faces = topo.vertex_faces,
    )

    return is_closed_surface(mesh), face_normals, edge_normals, vertex_normals, feature_data
end
