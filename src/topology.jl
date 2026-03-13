# topology.jl – Extraction of topological structure from mesh types.
#
# Conventions
# -----------
# * All indices are 1-based.
# * An *edge* is represented as `SVector{2,Int}(i, j)` with `i < j`
#   (unoriented canonical form) in the edge list, but signed ±1 in the
#   face-edge incidence matrix.
# * The *oriented* edge for a face `(a,b,c)` is the set `{(a,b),(b,c),(c,a)}`
#   each read as directed; the canonical (unoriented) version has the smaller
#   index first.

# ─────────────────────────────────────────────────────────────────────────────
# Shared types returned by topology functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    MeshTopology

Result type holding all topological adjacency lists and the edge list for a
`SurfaceMesh`.  This is computed once and reused by geometry and DEC
routines.

Fields
------
- `edges         :: Vector{SVector{2,Int}}` – unique unoriented edges `(i<j)`.
- `face_edges    :: Vector{SVector{3,Int}}` – for each face, the indices (into
  `edges`) of its three edges; the sign is absorbed into `face_edge_signs`.
- `face_edge_signs :: Vector{SVector{3,Int}}` – `+1` if the face traverses
  the canonical edge in forward order, `-1` otherwise.
- `vertex_faces  :: Vector{Vector{Int}}` – adjacency: vertex → face indices.
- `vertex_edges  :: Vector{Vector{Int}}` – adjacency: vertex → edge indices.
- `edge_faces    :: Vector{Vector{Int}}` – adjacency: edge → face indices.
"""
struct MeshTopology
    edges            :: Vector{SVector{2,Int}}
    face_edges       :: Vector{SVector{3,Int}}
    face_edge_signs  :: Vector{SVector{3,Int}}
    vertex_faces     :: Vector{Vector{Int}}
    vertex_edges     :: Vector{Vector{Int}}
    edge_faces       :: Vector{Vector{Int}}
end

# ─────────────────────────────────────────────────────────────────────────────
# Surface topology
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_topology(mesh::SurfaceMesh) -> MeshTopology

Extract all topological incidence lists and the unique edge list from
`mesh`.  This is the central topology constructor for surfaces; all
other topology queries delegate to this function.

Complexity: O(F) where F is the number of faces.
"""
function build_topology(mesh::SurfaceMesh{T}) where {T}
    nv = length(mesh.points)
    nf = length(mesh.faces)

    # ----- Step 1: enumerate oriented half-edges per face and canonical edges ----
    # edge_dict maps canonical (i<j) → index in edge list
    edge_dict  = Dict{SVector{2,Int},Int}()
    edges      = SVector{2,Int}[]
    face_edges  = Vector{SVector{3,Int}}(undef, nf)
    face_edge_signs = Vector{SVector{3,Int}}(undef, nf)

    for (fi, face) in enumerate(mesh.faces)
        a, b, c = face[1], face[2], face[3]
        half_edges = (SVector{2,Int}(a,b), SVector{2,Int}(b,c), SVector{2,Int}(c,a))
        local_ei = MVector{3,Int}(0,0,0)
        local_si = MVector{3,Int}(0,0,0)
        for (k, he) in enumerate(half_edges)
            i, j = he[1], he[2]
            canon = i < j ? SVector{2,Int}(i,j) : SVector{2,Int}(j,i)
            sign  = i < j ? +1 : -1
            if !haskey(edge_dict, canon)
                push!(edges, canon)
                edge_dict[canon] = length(edges)
            end
            local_ei[k] = edge_dict[canon]
            local_si[k] = sign
        end
        face_edges[fi]      = SVector{3,Int}(local_ei)
        face_edge_signs[fi] = SVector{3,Int}(local_si)
    end

    ne = length(edges)

    # ----- Step 2: vertex → faces -----------------------------------------------
    vertex_faces = [Int[] for _ in 1:nv]
    for (fi, face) in enumerate(mesh.faces)
        for vi in face
            push!(vertex_faces[vi], fi)
        end
    end

    # ----- Step 3: vertex → edges -----------------------------------------------
    vertex_edges = [Int[] for _ in 1:nv]
    for (ei, e) in enumerate(edges)
        push!(vertex_edges[e[1]], ei)
        push!(vertex_edges[e[2]], ei)
    end

    # ----- Step 4: edge → faces -------------------------------------------------
    edge_faces = [Int[] for _ in 1:ne]
    for (fi, fe) in enumerate(face_edges)
        for ei in fe
            push!(edge_faces[ei], fi)
        end
    end

    return MeshTopology(edges, face_edges, face_edge_signs,
                        vertex_faces, vertex_edges, edge_faces)
end

"""
    vertex_to_faces(mesh::SurfaceMesh) -> Vector{Vector{Int}}

Return a list of face index lists: `result[v]` contains the indices of all
faces incident on vertex `v`.
"""
function vertex_to_faces(mesh::SurfaceMesh)
    return build_topology(mesh).vertex_faces
end

"""
    vertex_to_edges(mesh::SurfaceMesh) -> Vector{Vector{Int}}

Return a list of edge index lists: `result[v]` contains the indices of all
unique edges incident on vertex `v`.
"""
function vertex_to_edges(mesh::SurfaceMesh)
    return build_topology(mesh).vertex_edges
end

"""
    edge_to_faces(mesh::SurfaceMesh) -> Vector{Vector{Int}}

Return a list of face index lists: `result[e]` contains the indices of all
faces incident on edge `e` (length 1 for boundary edges, 2 for interior).
"""
function edge_to_faces(mesh::SurfaceMesh)
    return build_topology(mesh).edge_faces
end

"""
    is_closed(mesh::SurfaceMesh) -> Bool

Return `true` if every edge of `mesh` is shared by exactly two faces
(no boundary edges exist).
"""
function is_closed(mesh::SurfaceMesh)
    topo = build_topology(mesh)
    return all(length(ef) == 2 for ef in topo.edge_faces)
end

"""
    is_manifold(mesh::SurfaceMesh) -> Bool

Return `true` if every edge is shared by at most 2 faces (manifold criterion)
and every vertex has a single connected fan of faces (no pinch points).

This is a conservative check; it returns `false` for non-manifold edges only
in v0.1.
"""
function is_manifold(mesh::SurfaceMesh)
    topo = build_topology(mesh)
    # Check edge-manifold: each edge has at most 2 incident faces.
    return all(length(ef) <= 2 for ef in topo.edge_faces)
end

"""
    has_consistent_orientation(mesh::SurfaceMesh) -> Bool

Return `true` if, for every interior edge, the two adjacent faces traverse it
in opposite orientations (consistent outward-normal convention).
"""
function has_consistent_orientation(mesh::SurfaceMesh)
    topo = build_topology(mesh)
    nf = length(mesh.faces)
    # For each edge shared by 2 faces, check that the two faces traverse the
    # canonical edge with opposite signs.
    for (ei, ef) in enumerate(topo.edge_faces)
        length(ef) == 2 || continue   # boundary edge → skip
        fi1, fi2 = ef[1], ef[2]
        # Find position of this edge in each face.
        sign1 = _face_edge_sign(topo, fi1, ei)
        sign2 = _face_edge_sign(topo, fi2, ei)
        sign1 * sign2 == -1 || return false
    end
    return true
end

function _face_edge_sign(topo::MeshTopology, fi::Int, ei::Int)
    fe = topo.face_edges[fi]
    fs = topo.face_edge_signs[fi]
    for k in 1:3
        fe[k] == ei && return fs[k]
    end
    error("Edge $ei not found in face $fi")
end

# ─────────────────────────────────────────────────────────────────────────────
# Curve topology
# ─────────────────────────────────────────────────────────────────────────────

"""
    vertex_to_edges(mesh::CurveMesh) -> Vector{Vector{Int}}

Return a list of edge index lists: `result[v]` contains the indices of all
edges incident on vertex `v`.
"""
function vertex_to_edges(mesh::CurveMesh)
    nv = length(mesh.points)
    result = [Int[] for _ in 1:nv]
    for (ei, e) in enumerate(mesh.edges)
        push!(result[e[1]], ei)
        push!(result[e[2]], ei)
    end
    return result
end

"""
    is_closed(mesh::CurveMesh) -> Bool

Return `true` if every vertex of the curve has exactly two incident edges,
indicating a closed loop.
"""
function is_closed(mesh::CurveMesh)
    ve = vertex_to_edges(mesh)
    return all(length(v) == 2 for v in ve)
end

"""
    curve_vertex_order(mesh::CurveMesh) -> Vector{Int}

Return the vertex indices in traversal order for a closed or open curve.
Throws an error if the curve is not a simple path or loop.
"""
function curve_vertex_order(mesh::CurveMesh)
    nv = length(mesh.points)
    ve = vertex_to_edges(mesh)
    # Build adjacency list: vertex → adjacent vertices via edges
    adj = [Int[] for _ in 1:nv]
    for e in mesh.edges
        push!(adj[e[1]], e[2])
        push!(adj[e[2]], e[1])
    end
    closed = is_closed(mesh)
    # Find a start vertex: for open curves, start at a degree-1 vertex
    start = 1
    if !closed
        start_candidates = [v for v in 1:nv if length(ve[v]) == 1]
        isempty(start_candidates) && error("Curve is neither closed nor has endpoints.")
        start = start_candidates[1]
    end
    order = [start]
    prev  = -1
    cur   = start
    while true
        nbrs = [v for v in adj[cur] if v != prev]
        isempty(nbrs) && break
        nxt = nbrs[1]
        nxt == start && break   # completed the loop
        push!(order, nxt)
        prev = cur
        cur  = nxt
    end
    return order
end
