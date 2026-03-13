# geometry_surfaces.jl – Intrinsic geometric quantities for SurfaceMesh.
#
# Implements
# ----------
# * face normals (outward, from cross product of edges)
# * face areas
# * unique edge lengths
# * vertex normals (area-weighted average of face normals)
# * vertex dual areas (barycentric: 1/3 of the sum of adjacent face areas)
# * mean-curvature normal and scalar mean curvature (via Laplace–Beltrami of
#   embedding coordinates, assembled in curvature.jl)
#
# Public entry point: `compute_geometry(mesh::SurfaceMesh) -> SurfaceGeometry`

"""
    compute_geometry(mesh::SurfaceMesh{T}) -> SurfaceGeometry{T}

Compute all intrinsic geometric quantities for a triangulated surface and
return a `SurfaceGeometry` container.

Face normals
------------
The outward unit normal of face `(a,b,c)` is the normalised cross product
`(p[b]-p[a]) × (p[c]-p[a])`.

Vertex dual areas
-----------------
The barycentric dual area at vertex `v` is `(1/3) × Σ area(f)` over all
faces `f` incident on `v`.  This is the simplest and most robust choice for
v0.1; circumcentric/Voronoi duals will be added in v0.2.

Vertex normals
--------------
The unit normal at vertex `v` is the area-weighted average of the face normals
of all adjacent faces, normalised.

The `mean_curvature_normal`, `mean_curvature`, and `gaussian_curvature` fields
of the returned `SurfaceGeometry` are empty vectors; they are filled by
`compute_curvature!` in `curvature.jl` if desired.
"""
function compute_geometry(mesh::SurfaceMesh{T}) :: SurfaceGeometry{T} where {T}
    pts   = mesh.points
    faces = mesh.faces
    nv    = length(pts)
    nf    = length(faces)

    # ── Topology: needed for edge list and vertex→face adjacency ──────────────
    topo = build_topology(mesh)
    ne   = length(topo.edges)

    # ── Face normals and areas ─────────────────────────────────────────────────
    face_normals = Vector{SVector{3,T}}(undef, nf)
    face_areas   = Vector{T}(undef, nf)
    for (fi, face) in enumerate(faces)
        a, b, c = pts[face[1]], pts[face[2]], pts[face[3]]
        n_raw   = cross3(b - a, c - a)
        area2   = norm(n_raw)
        face_areas[fi]   = area2 / 2
        face_normals[fi] = area2 > eps(T) ? n_raw / area2 : SVector{3,T}(0,0,1)
    end

    # ── Edge lengths ─────────────────────────────────────────────────────────
    edge_lengths = Vector{T}(undef, ne)
    for (ei, e) in enumerate(topo.edges)
        edge_lengths[ei] = norm(pts[e[2]] - pts[e[1]])
    end

    # ── Vertex dual areas (barycentric) ───────────────────────────────────────
    vertex_dual_areas = zeros(T, nv)
    for (fi, face) in enumerate(faces)
        contrib = face_areas[fi] / 3
        for vi in face
            vertex_dual_areas[vi] += contrib
        end
    end

    # ── Vertex normals (area-weighted) ────────────────────────────────────────
    vertex_normals = Vector{SVector{3,T}}(undef, nv)
    n_acc = [SVector{3,T}(0,0,0) for _ in 1:nv]
    for (fi, face) in enumerate(faces)
        wn = face_areas[fi] * face_normals[fi]
        for vi in face
            n_acc[vi] = n_acc[vi] + wn
        end
    end
    for vi in 1:nv
        vertex_normals[vi] = normalize_safe(n_acc[vi])
    end

    return SurfaceGeometry{T}(
        face_normals,
        face_areas,
        edge_lengths,
        vertex_dual_areas,
        vertex_normals,
        SVector{3,T}[],   # mean_curvature_normal – filled by curvature.jl
        T[],               # mean_curvature
        T[],               # gaussian_curvature
    )
end

"""
    edge_midpoints(mesh::SurfaceMesh{T}) -> Vector{SVector{3,T}}

Return the midpoint of each unique edge in the surface mesh (ordered as in the
topology edge list).
"""
function edge_midpoints(mesh::SurfaceMesh{T}) :: Vector{SVector{3,T}} where {T}
    topo = build_topology(mesh)
    return [T(0.5) * (mesh.points[e[1]] + mesh.points[e[2]]) for e in topo.edges]
end
