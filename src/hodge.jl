# hodge.jl – Metric-dependent Hodge star diagonal matrices.
#
# Conventions
# -----------
# All Hodge stars are diagonal sparse matrices.
#
# Curve (1-D manifold embedded in ℝ²)
# ─────────────────────────────────────
# ⋆₀ : diagonal of vertex dual lengths (primal 0-simplex → dual 1-simplex).
# ⋆₁ : diagonal of primal edge lengths (primal 1-simplex → dual 0-simplex).
#      For an embedded curve the dual of a 1-simplex is a dual point; the
#      metric pairing is simply the primal edge length.
#
# Surface (2-D manifold embedded in ℝ³)
# ──────────────────────────────────────
# ⋆₀ : diagonal of vertex dual areas.
# ⋆₂ : diagonal of face areas.
# ⋆₁ : cotan-weight diagonal, giving the scalar Laplace–Beltrami
#      L = ⋆₀⁻¹ d₀ᵀ ⋆₁ d₀   (positive-semi-definite form).
#
#      For each canonical edge (i,j), the cotan weight is
#        w_ij = (1/2)(cot α_ij + cot β_ij)
#      where α_ij and β_ij are the angles opposite to edge (i,j) in the two
#      adjacent triangles.  For boundary edges (only one adjacent triangle) we
#      use only the single available cotan.

"""
    hodge_star_0(mesh::CurveMesh{T}, geom::CurveGeometry{T}) -> SparseMatrixCSC{T,Int}

Hodge ⋆₀ for a curve: diagonal matrix of vertex dual lengths.
"""
function hodge_star_0(
        mesh::CurveMesh{T},
        geom::CurveGeometry{T},
) :: SparseMatrixCSC{T,Int} where {T}
    nv = length(mesh.points)
    return spdiagm(0 => geom.vertex_dual_lengths)
end

"""
    hodge_star_1(mesh::CurveMesh{T}, geom::CurveGeometry{T}) -> SparseMatrixCSC{T,Int}

Hodge ⋆₁ for a curve: diagonal matrix of primal edge lengths.
"""
function hodge_star_1(
        mesh::CurveMesh{T},
        geom::CurveGeometry{T},
) :: SparseMatrixCSC{T,Int} where {T}
    return spdiagm(0 => geom.edge_lengths)
end

# ─────────────────────────────────────────────────────────────────────────────
# Surface Hodge stars
# ─────────────────────────────────────────────────────────────────────────────

"""
    hodge_star_0(mesh::SurfaceMesh{T}, geom::SurfaceGeometry{T}) -> SparseMatrixCSC{T,Int}

Hodge ⋆₀ for a surface: diagonal matrix of vertex dual areas.
"""
function hodge_star_0(
        mesh::SurfaceMesh{T},
        geom::SurfaceGeometry{T},
) :: SparseMatrixCSC{T,Int} where {T}
    return spdiagm(0 => geom.vertex_dual_areas)
end

"""
    hodge_star_2(mesh::SurfaceMesh{T}, geom::SurfaceGeometry{T}) -> SparseMatrixCSC{T,Int}

Hodge ⋆₂ for a surface: diagonal matrix of face areas.
"""
function hodge_star_2(
        mesh::SurfaceMesh{T},
        geom::SurfaceGeometry{T},
) :: SparseMatrixCSC{T,Int} where {T}
    return spdiagm(0 => geom.face_areas)
end

"""
    hodge_star_1(mesh::SurfaceMesh{T}, geom::SurfaceGeometry{T}) -> SparseMatrixCSC{T,Int}

Hodge ⋆₁ for a surface: diagonal matrix of cotan weights.

For each unique edge (i,j) the weight is

    w_ij = (1/2) (cot α + cot β)

where α, β are the angles at the vertex opposite to edge (i,j) in each of the
(at most two) adjacent faces.  This gives the standard cotan Laplace–Beltrami
through `L = ⋆₀⁻¹ d₀ᵀ ⋆₁ d₀`.

Note: negative cotan weights (obtuse triangles) are not clamped in v0.1; the
operator may have small negative off-diagonal entries for coarse meshes with
obtuse triangles.
"""
function hodge_star_1(
        mesh::SurfaceMesh{T},
        geom::SurfaceGeometry{T},
) :: SparseMatrixCSC{T,Int} where {T}
    pts   = mesh.points
    faces = mesh.faces
    topo  = build_topology(mesh)
    ne    = length(topo.edges)

    w = zeros(T, ne)

    for (fi, face) in enumerate(faces)
        a, b, c = pts[face[1]], pts[face[2]], pts[face[3]]
        # For each vertex of the triangle, compute the cotan of its angle.
        # The angle at vertex a is between edges (a→b) and (a→c):
        #   cot at a = dot(ab,ac) / |ab × ac|
        # Each cotan is half the weight for the opposite edge.
        ab = b - a; ac = c - a
        bc = c - b; ba = a - b
        ca = a - c; cb = b - c

        cot_a = cotangent(ab, ac)
        cot_b = cotangent(bc, ba)
        cot_c = cotangent(ca, cb)

        # Edge indices for this face (from topology)
        fe = topo.face_edges[fi]
        # Map face edge positions to vertices:
        #   half-edge 0 = (a→b): edge index fe[1], opposite vertex a... wait:
        # half_edges of face (a,b,c) were (a→b), (b→c), (c→a)
        # Opposite vertices: (c→a) has opposite b, (a→b) has opposite c, (b→c) has opposite a
        # => edge fe[1] = (a,b) has opposite vertex c (angle = cot_c)
        #    edge fe[2] = (b,c) has opposite vertex a (angle = cot_a)
        #    edge fe[3] = (c,a) has opposite vertex b (angle = cot_b)
        w[fe[1]] += T(0.5) * cot_c
        w[fe[2]] += T(0.5) * cot_a
        w[fe[3]] += T(0.5) * cot_b
    end

    return spdiagm(0 => w)
end
