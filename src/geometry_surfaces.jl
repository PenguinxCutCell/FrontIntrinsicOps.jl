# geometry_surfaces.jl – Intrinsic geometric quantities for SurfaceMesh.
#
# Implements
# ----------
# * face normals (outward, from cross product of edges)
# * face areas
# * unique edge lengths
# * vertex normals (area-weighted average of face normals)
# * vertex dual areas: barycentric (1/3 of adjacent face areas) or
#   mixed/Voronoi (circumcentric for acute, fallback for obtuse)
# * mean-curvature normal and scalar mean curvature (via Laplace–Beltrami of
#   embedding coordinates, assembled in curvature.jl)
#
# Public entry point: `compute_geometry(mesh::SurfaceMesh; dual_area=:barycentric) -> SurfaceGeometry`

"""
    _vertex_dual_areas_barycentric(faces, face_areas, nv) -> Vector{T}

Barycentric dual area: each vertex receives 1/3 of each adjacent face area.
"""
function _vertex_dual_areas_barycentric(
        faces      :: Vector{SVector{3,Int}},
        face_areas :: Vector{T},
        nv         :: Int,
) :: Vector{T} where {T}
    da = zeros(T, nv)
    for (fi, face) in enumerate(faces)
        contrib = face_areas[fi] / 3
        for vi in face
            da[vi] += contrib
        end
    end
    return da
end

"""
    _vertex_dual_areas_mixed(pts, faces, face_areas, nv) -> Vector{T}

Mixed/Voronoi dual area at each vertex (Meyer et al. 2003).

For each triangle:
- If the triangle is **non-obtuse**: the contribution at vertex `v` is the
  circumcentric (Voronoi) area:
      A_v = (1/8) * (|e1|^2 cot(alpha2) + |e2|^2 cot(alpha1))
  where e1, e2 are the two edges adjacent to v and alpha1, alpha2 are the
  angles at the opposite vertices.
- If the triangle is **obtuse at v**: contribution = area(T) / 2.
- If the triangle is **obtuse but not at v**: contribution = area(T) / 4.

The mixed areas sum to the total surface area and are always positive.
"""
function _vertex_dual_areas_mixed(
        pts        :: Vector{SVector{3,T}},
        faces      :: Vector{SVector{3,Int}},
        face_areas :: Vector{T},
        nv         :: Int,
) :: Vector{T} where {T}
    da = zeros(T, nv)
    for (fi, face) in enumerate(faces)
        ia, ib, ic = face[1], face[2], face[3]
        a, b, c = pts[ia], pts[ib], pts[ic]

        ab = b - a;  ac = c - a
        ba = a - b;  bc = c - b
        ca = a - c;  cb = b - c

        len_ab2 = dot(ab, ab)
        len_ac2 = dot(ac, ac)
        len_bc2 = dot(bc, bc)

        # Cosines to detect obtuse triangles
        cos_a = dot(ab, ac) / sqrt(len_ab2 * len_ac2 + eps(T))
        cos_b = dot(ba, bc) / sqrt(len_ab2 * len_bc2 + eps(T))
        cos_c = dot(ca, cb) / sqrt(len_ac2 * len_bc2 + eps(T))

        fa = face_areas[fi]

        if cos_a < 0   # obtuse at a
            da[ia] += fa / 2
            da[ib] += fa / 4
            da[ic] += fa / 4
        elseif cos_b < 0  # obtuse at b
            da[ia] += fa / 4
            da[ib] += fa / 2
            da[ic] += fa / 4
        elseif cos_c < 0  # obtuse at c
            da[ia] += fa / 4
            da[ib] += fa / 4
            da[ic] += fa / 2
        else
            # Non-obtuse: circumcentric (Voronoi) areas
            # A_voronoi(a) = (1/8)(|ab|^2 cot_c + |ac|^2 cot_b)
            # A_voronoi(b) = (1/8)(|ab|^2 cot_c + |bc|^2 cot_a)
            # A_voronoi(c) = (1/8)(|ac|^2 cot_b + |bc|^2 cot_a)
            cot_a = cotangent(ab, ac)
            cot_b = cotangent(bc, ba)
            cot_c = cotangent(ca, cb)
            da[ia] += T(0.125) * (len_ab2 * cot_c + len_ac2 * cot_b)
            da[ib] += T(0.125) * (len_ab2 * cot_c + len_bc2 * cot_a)
            da[ic] += T(0.125) * (len_ac2 * cot_b + len_bc2 * cot_a)
        end
    end
    return da
end

"""
    compute_geometry(mesh::SurfaceMesh{T}; dual_area=:barycentric) -> SurfaceGeometry{T}

Compute all intrinsic geometric quantities for a triangulated surface and
return a `SurfaceGeometry` container.

Keyword arguments
-----------------
- `dual_area :: Symbol` – dual-area formula for vertex measures:
  - `:barycentric` (default): each vertex receives 1/3 of each adjacent face area.
  - `:mixed` (alias `:voronoi`): circumcentric (Voronoi) area for acute triangles,
    mixed fallback for obtuse triangles (Meyer et al. 2003).  Always positive.

Face normals
------------
The outward unit normal of face `(a,b,c)` is the normalised cross product
`(p[b]-p[a]) x (p[c]-p[a])`.

Vertex normals
--------------
The unit normal at vertex `v` is the area-weighted average of the face normals
of all adjacent faces, normalised.

Curvature fields
----------------
The `mean_curvature_normal`, `mean_curvature`, and `gaussian_curvature` fields
of the returned `SurfaceGeometry` are empty vectors; they are filled by
`compute_curvature` in `curvature.jl` if desired.
"""
function compute_geometry(
        mesh      :: SurfaceMesh{T};
        dual_area :: Symbol = :barycentric,
) :: SurfaceGeometry{T} where {T}
    # Resolve alias :voronoi => :mixed
    method = dual_area === :voronoi ? :mixed : dual_area
    method in (:barycentric, :mixed) ||
        error("compute_geometry: unknown dual_area method $(repr(dual_area)). " *
              "Use :barycentric, :mixed, or :voronoi.")

    pts   = mesh.points
    faces = mesh.faces
    nv    = length(pts)
    nf    = length(faces)

    # -- Topology: needed for edge list ----------------------------------------
    topo = build_topology(mesh)
    ne   = length(topo.edges)

    # -- Face normals and areas -------------------------------------------------
    face_normals = Vector{SVector{3,T}}(undef, nf)
    face_areas   = Vector{T}(undef, nf)
    for (fi, face) in enumerate(faces)
        a, b, c = pts[face[1]], pts[face[2]], pts[face[3]]
        n_raw   = cross3(b - a, c - a)
        area2   = norm(n_raw)
        face_areas[fi]   = area2 / 2
        face_normals[fi] = area2 > eps(T) ? n_raw / area2 : SVector{3,T}(0,0,1)
    end

    # -- Edge lengths -----------------------------------------------------------
    edge_lengths = Vector{T}(undef, ne)
    for (ei, e) in enumerate(topo.edges)
        edge_lengths[ei] = norm(pts[e[2]] - pts[e[1]])
    end

    # -- Vertex dual areas ------------------------------------------------------
    vertex_dual_areas = if method === :barycentric
        _vertex_dual_areas_barycentric(faces, face_areas, nv)
    else  # :mixed
        _vertex_dual_areas_mixed(pts, faces, face_areas, nv)
    end

    # -- Vertex normals (area-weighted) ----------------------------------------
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
        method,            # dual_area_method
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
