# parallel_transport.jl – Discrete tangent-frame transport on triangle meshes.

# -----------------------------------------------------------------------------
# Internal frame helpers
# -----------------------------------------------------------------------------

function _orthonormal_frame_from_normal(
    n::SVector{3,T},
    ref::SVector{3,T},
) where {T}
    n̂ = normalize_safe(n)
    t1 = ref - dot(ref, n̂) * n̂
    if norm(t1) <= eps(T)
        alt = abs(n̂[1]) < T(0.9) ? SVector{3,T}(1,0,0) : SVector{3,T}(0,1,0)
        t1 = alt - dot(alt, n̂) * n̂
    end
    t1 = normalize_safe(t1)
    t2 = normalize_safe(cross(n̂, t1))
    return SMatrix{3,2,T,6}(hcat(t1, t2))
end

function _face_frames(mesh::SurfaceMesh{T}, geom::SurfaceGeometry{T}) where {T}
    nf = length(mesh.faces)
    frames = Vector{SMatrix{3,2,T,6}}(undef, nf)
    for fi in 1:nf
        face = mesh.faces[fi]
        a, b = face[1], face[2]
        ref = mesh.points[b] - mesh.points[a]
        frames[fi] = _orthonormal_frame_from_normal(geom.face_normals[fi], ref)
    end
    return frames
end

function _shared_edge_index(topo::MeshTopology, f1::Int, f2::Int)
    e1 = topo.face_edges[f1]
    e2 = topo.face_edges[f2]
    for k in 1:3
        ei = e1[k]
        for j in 1:3
            if ei == e2[j]
                return ei
            end
        end
    end
    throw(ArgumentError("Faces $f1 and $f2 are not adjacent."))
end

function _face_path_dual(topo::MeshTopology, fstart::Int, fgoal::Int)
    fstart == fgoal && return Int[fstart]
    nf = length(topo.face_edges)
    adj = [Int[] for _ in 1:nf]
    for ef in topo.edge_faces
        if length(ef) == 2
            f1, f2 = ef[1], ef[2]
            push!(adj[f1], f2)
            push!(adj[f2], f1)
        end
    end
    for a in adj
        sort!(a)
    end

    prev = fill(0, nf)
    seen = falses(nf)
    q = Int[fstart]
    seen[fstart] = true
    head = 1
    while head <= length(q)
        f = q[head]
        head += 1
        f == fgoal && break
        for g in adj[f]
            if !seen[g]
                seen[g] = true
                prev[g] = f
                push!(q, g)
            end
        end
    end

    seen[fgoal] || throw(ArgumentError("No dual-face path between faces $fstart and $fgoal."))

    path = Int[fgoal]
    cur = fgoal
    while cur != fstart
        cur = prev[cur]
        push!(path, cur)
    end
    reverse!(path)
    return path
end

function _edge_primary_face(topo::MeshTopology, edgeid::Int)
    ef = topo.edge_faces[edgeid]
    isempty(ef) && throw(ArgumentError("Edge $edgeid has no incident faces."))
    return minimum(ef)
end

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

"""
    face_tangent_frames(mesh, geom) -> Vector

Return per-face orthonormal tangent frames as `SMatrix{3,2}`.
Each frame columns are `(t1, t2)` with `t2 = n × t1`.
"""
function face_tangent_frames(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
) where {T}
    return _face_frames(mesh, geom)
end

"""
    vertex_tangent_frames(mesh, geom; method=:angle_weighted) -> Vector

Return per-vertex orthonormal tangent frames as `SMatrix{3,2}`.

Current implementation uses a deterministic local edge-direction reference;
`method` is reserved for future weighting variants.
"""
function vertex_tangent_frames(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T};
    method::Symbol=:angle_weighted,
) where {T}
    method in (:angle_weighted, :local_edge) ||
        throw(ArgumentError("Unsupported method=$(repr(method))."))

    topo = build_topology(mesh)
    nv = length(mesh.points)
    frames = Vector{SMatrix{3,2,T,6}}(undef, nv)

    for v in 1:nv
        vedges = topo.vertex_edges[v]
        isempty(vedges) && throw(ArgumentError("Vertex $v has no incident edges."))
        ei = vedges[1]
        e = topo.edges[ei]
        other = e[1] == v ? e[2] : e[1]
        ref = mesh.points[other] - mesh.points[v]
        frames[v] = _orthonormal_frame_from_normal(geom.vertex_normals[v], ref)
    end

    return frames
end

"""
    connection_angle_across_edge(mesh, geom, faceL, faceR, edgeid) -> Real

Return the signed connection angle (in radians) mapping `faceL` frame to
`faceR` frame across the shared edge.
"""
function connection_angle_across_edge(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    faceL::Int,
    faceR::Int,
    edgeid::Int,
) where {T}
    frames = _face_frames(mesh, geom)
    FL = frames[faceL]
    FR = frames[faceR]

    topo = build_topology(mesh)
    e = topo.edges[edgeid]
    t = normalize_safe(mesh.points[e[2]] - mesh.points[e[1]])

    uL = SVector{2,T}(dot(t, FL[:, 1]), dot(t, FL[:, 2]))
    uR = SVector{2,T}(dot(t, FR[:, 1]), dot(t, FR[:, 2]))

    θL = atan(uL[2], uL[1])
    θR = atan(uR[2], uR[1])
    θ = θR - θL
    return θ
end

"""
    transport_matrix_across_edge(mesh, geom, faceL, faceR, edgeid) -> SMatrix{2,2}

Return the local 2×2 rotation matrix transporting face-local coordinates from
`faceL` to `faceR`.
"""
function transport_matrix_across_edge(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    faceL::Int,
    faceR::Int,
    edgeid::Int,
) where {T}
    θ = connection_angle_across_edge(mesh, geom, faceL, faceR, edgeid)
    c, s = cos(θ), sin(θ)
    return SMatrix{2,2,T,4}(c, -s, s, c)
end

"""
    parallel_transport_face_vector(v, mesh, geom, faceL, faceR, edgeid)

Transport a 2D face-local tangent vector coordinate `v` from `faceL` to
`faceR` across one shared edge.
"""
function parallel_transport_face_vector(
    v::SVector{2,T},
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    faceL::Int,
    faceR::Int,
    edgeid::Int,
) where {T}
    R = transport_matrix_across_edge(mesh, geom, faceL, faceR, edgeid)
    return R * v
end

"""
    parallel_transport_along_face_path(v, mesh, geom, faces_path) -> SVector{2}

Transport a local 2D face-vector `v` along a sequence of adjacent faces.
The input `v` is interpreted in coordinates of `faces_path[1]`.
"""
function parallel_transport_along_face_path(
    v::SVector{2,T},
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    faces_path::AbstractVector{Int},
) where {T}
    length(faces_path) >= 1 || throw(ArgumentError("faces_path must contain at least one face."))
    length(faces_path) == 1 && return v

    topo = build_topology(mesh)
    out = v
    for k in 1:(length(faces_path)-1)
        fL = faces_path[k]
        fR = faces_path[k+1]
        ei = _shared_edge_index(topo, fL, fR)
        out = parallel_transport_face_vector(out, mesh, geom, fL, fR, ei)
    end
    return out
end

"""
    rotate_in_tangent_frame(v, θ, frame)

Rotate a tangent vector by angle `θ` in a local tangent frame.

- If `v` is `SVector{2}`, returns rotated 2D coordinates.
- If `v` is `SVector{3}`, `frame` must be an `SMatrix{3,2}` and the rotated
  ambient tangent vector is returned.
"""
function rotate_in_tangent_frame(
    v::SVector{2,T},
    θ::Real,
    frame,
) where {T}
    c, s = cos(T(θ)), sin(T(θ))
    R = SMatrix{2,2,T,4}(c, -s, s, c)
    return R * v
end

function rotate_in_tangent_frame(
    v::SVector{3,T},
    θ::Real,
    frame::SMatrix{3,2,T,6},
) where {T}
    coords = SVector{2,T}(dot(v, frame[:, 1]), dot(v, frame[:, 2]))
    c2 = rotate_in_tangent_frame(coords, θ, nothing)
    return frame[:, 1] * c2[1] + frame[:, 2] * c2[2]
end

"""
    parallel_transport_vertex_vector(v, mesh, geom, vid_from, vid_to;
                                     path=:geodesic,
                                     geodesic_cache=nothing)

Transport an ambient tangent vector `v` from vertex `vid_from` to `vid_to`.

Current implementation transports through a dual-face path between one incident
face at `vid_from` and one incident face at `vid_to`.
"""
function parallel_transport_vertex_vector(
    v::SVector{3,T},
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    vid_from::Int,
    vid_to::Int;
    path=:geodesic,
    geodesic_cache=nothing,
) where {T}
    vid_from == vid_to && return tangential_project(v, geom.vertex_normals[vid_from])

    topo = build_topology(mesh)
    vf = vertex_to_faces(mesh)
    isempty(vf[vid_from]) && throw(ArgumentError("vid_from has no incident faces."))
    isempty(vf[vid_to]) && throw(ArgumentError("vid_to has no incident faces."))

    fstart = vf[vid_from][1]
    fgoal = vf[vid_to][1]

    faces_path = if path isa AbstractVector{Int}
        collect(Int, path)
    elseif path === :geodesic || path === :graph
        _face_path_dual(topo, fstart, fgoal)
    else
        throw(ArgumentError("Unsupported path selector $(repr(path))."))
    end

    Ffaces = _face_frames(mesh, geom)
    c0 = SVector{2,T}(dot(v, Ffaces[faces_path[1]][:, 1]), dot(v, Ffaces[faces_path[1]][:, 2]))
    c1 = parallel_transport_along_face_path(c0, mesh, geom, faces_path)

    v_face = Ffaces[faces_path[end]][:, 1] * c1[1] + Ffaces[faces_path[end]][:, 2] * c1[2]
    return tangential_project(v_face, geom.vertex_normals[vid_to])
end

"""
    transport_edge_1form(ω1, mesh, geom, src_edge, dst_edge; path=:graph) -> Real

Transport edge-1form information from `src_edge` to `dst_edge`.

Conventions
-----------
- `ω1` is an edge 1-cochain (line-integral convention).
- The helper reconstructs a face tangent vector from `ω1`, parallel-transports
  that vector through a face path, then re-integrates against the destination
  edge tangent direction.
- For `src_edge == dst_edge`, returns `ω1[src_edge]`.
"""
function transport_edge_1form(
    ω1::AbstractVector{T},
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    src_edge::Int,
    dst_edge::Int;
    path=:graph,
) where {T}
    topo = build_topology(mesh)
    ne = length(topo.edges)
    length(ω1) == ne || throw(DimensionMismatch("ω1 length $(length(ω1)) != nE=$ne"))
    (1 <= src_edge <= ne && 1 <= dst_edge <= ne) || throw(BoundsError("Edge index out of range."))
    src_edge == dst_edge && return ω1[src_edge]

    fstart = _edge_primary_face(topo, src_edge)
    fgoal = _edge_primary_face(topo, dst_edge)

    faces_path = if path isa AbstractVector{Int}
        collect(Int, path)
    elseif path === :graph || path === :geodesic
        _face_path_dual(topo, fstart, fgoal)
    else
        throw(ArgumentError("Unsupported path selector $(repr(path))."))
    end
    isempty(faces_path) && throw(ArgumentError("Empty face path."))

    frames = _face_frames(mesh, geom)
    Vω = oneform_to_tangent_vectors(mesh, geom, topo, ω1; location=:face)
    vstart = Vω[faces_path[1]]
    c0 = SVector{2,T}(dot(vstart, frames[faces_path[1]][:, 1]), dot(vstart, frames[faces_path[1]][:, 2]))
    c1 = parallel_transport_along_face_path(c0, mesh, geom, faces_path)

    fend = faces_path[end]
    vend = frames[fend][:, 1] * c1[1] + frames[fend][:, 2] * c1[2]

    e = topo.edges[dst_edge]
    t = normalize_safe(mesh.points[e[2]] - mesh.points[e[1]])
    return dot(vend, t) * geom.edge_lengths[dst_edge]
end

"""
    holonomy_along_cycle(mesh, geom, cycle_faces) -> Real

Compute the signed holonomy angle accumulated around a closed face cycle.
"""
function holonomy_along_cycle(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    cycle_faces::AbstractVector{Int},
) where {T}
    length(cycle_faces) >= 2 || return zero(T)

    topo = build_topology(mesh)
    Rtot = SMatrix{2,2,T,4}(1, 0, 0, 1)

    for k in 1:length(cycle_faces)
        fL = cycle_faces[k]
        fR = cycle_faces[mod1(k + 1, length(cycle_faces))]
        ei = _shared_edge_index(topo, fL, fR)
        R = transport_matrix_across_edge(mesh, geom, fL, fR, ei)
        Rtot = R * Rtot
    end

    return atan(Rtot[2, 1], Rtot[1, 1])
end
