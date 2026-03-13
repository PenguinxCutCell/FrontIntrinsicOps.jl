# integrals.jl – Integral quantities over curves and surfaces.

# ─────────────────────────────────────────────────────────────────────────────
# Shared dispatch
# ─────────────────────────────────────────────────────────────────────────────

"""
    measure(mesh, geom) -> T

Return the total intrinsic measure of the mesh:
- For a `CurveMesh` : total arc length.
- For a `SurfaceMesh`: total surface area.
"""
function measure end

"""
    enclosed_measure(mesh) -> T

Return the enclosed measure:
- For a closed `CurveMesh`  : enclosed area (shoelace formula).
- For a closed `SurfaceMesh`: enclosed volume (divergence theorem).

Throws an error if the mesh is not closed.
"""
function enclosed_measure end

# ─────────────────────────────────────────────────────────────────────────────
# Curves
# ─────────────────────────────────────────────────────────────────────────────

"""
    measure(mesh::CurveMesh{T}, geom::CurveGeometry{T}) -> T

Total arc length of the curve.
"""
function measure(mesh::CurveMesh{T}, geom::CurveGeometry{T}) :: T where {T}
    return sum(geom.edge_lengths)
end

"""
    enclosed_measure(mesh::CurveMesh{T}) -> T

Enclosed signed area of a closed polygonal curve using the shoelace formula.

A counter-clockwise curve returns a positive value.
"""
function enclosed_measure(mesh::CurveMesh{T}) :: T where {T}
    is_closed(mesh) || error("enclosed_measure: curve is not closed.")
    pts = mesh.points
    A   = zero(T)
    for e in mesh.edges
        p1 = pts[e[1]]
        p2 = pts[e[2]]
        A += p1[1] * p2[2] - p2[1] * p1[2]
    end
    return abs(A) / 2
end

"""
    integrate_vertex_field(mesh::CurveMesh{T}, geom::CurveGeometry{T}, u) -> T

Integrate a vertex scalar field `u` over the curve using vertex dual lengths
as weights:

    ∫ u ds ≈ Σᵢ u[i] Δᵢ

where `Δᵢ` is the dual length at vertex `i`.
"""
function integrate_vertex_field(
        ::CurveMesh{T},
        geom::CurveGeometry{T},
        u::AbstractVector{T},
) :: T where {T}
    return dot(geom.vertex_dual_lengths, u)
end

"""
    integrate_face_field(mesh::CurveMesh{T}, geom::CurveGeometry{T}, u) -> T

Integrate an edge scalar field `u` over the curve using edge lengths:

    ∫ u ds ≈ Σₑ u[e] |eₑ|
"""
function integrate_face_field(
        ::CurveMesh{T},
        geom::CurveGeometry{T},
        u::AbstractVector{T},
) :: T where {T}
    return dot(geom.edge_lengths, u)
end

# ─────────────────────────────────────────────────────────────────────────────
# Surfaces
# ─────────────────────────────────────────────────────────────────────────────

"""
    measure(mesh::SurfaceMesh{T}, geom::SurfaceGeometry{T}) -> T

Total surface area.
"""
function measure(mesh::SurfaceMesh{T}, geom::SurfaceGeometry{T}) :: T where {T}
    return sum(geom.face_areas)
end

"""
    enclosed_measure(mesh::SurfaceMesh{T}) -> T

Enclosed volume of a closed oriented surface, computed via the divergence
theorem:

    V = (1/6) |Σᶠ (a·(b×c))|

where a, b, c are the three vertex positions of face f.
"""
function enclosed_measure(mesh::SurfaceMesh{T}) :: T where {T}
    is_closed(mesh) || error("enclosed_measure: surface mesh is not closed.")
    pts   = mesh.points
    faces = mesh.faces
    vol   = zero(T)
    for face in faces
        a = pts[face[1]]
        b = pts[face[2]]
        c = pts[face[3]]
        vol += dot(a, cross3(b, c))
    end
    return abs(vol) / 6
end

"""
    integrate_vertex_field(mesh::SurfaceMesh{T}, geom::SurfaceGeometry{T}, u) -> T

Integrate a vertex scalar field `u` over the surface using barycentric dual
areas as weights:

    ∫ u dA ≈ Σᵢ u[i] Aᵢ

where `Aᵢ` is the vertex dual area.
"""
function integrate_vertex_field(
        ::SurfaceMesh{T},
        geom::SurfaceGeometry{T},
        u::AbstractVector{T},
) :: T where {T}
    return dot(geom.vertex_dual_areas, u)
end

"""
    integrate_face_field(mesh::SurfaceMesh{T}, geom::SurfaceGeometry{T}, u) -> T

Integrate a face scalar field `u` over the surface using face areas:

    ∫ u dA ≈ Σᶠ u[f] Aᶠ
"""
function integrate_face_field(
        ::SurfaceMesh{T},
        geom::SurfaceGeometry{T},
        u::AbstractVector{T},
) :: T where {T}
    return dot(geom.face_areas, u)
end
