# whitney_forms.jl – Lowest-order Whitney basis evaluation and reconstruction.

# -----------------------------------------------------------------------------
# Internal triangle geometry helper
# -----------------------------------------------------------------------------

struct WhitneyTriangleGeometry{T<:AbstractFloat}
    points::NTuple{3,SVector{3,T}}
    normal::SVector{3,T}
    area::T
    gradlambda::NTuple{3,SVector{3,T}}
end

function _triangle_geometry_from_points(
    p1::SVector{3,T},
    p2::SVector{3,T},
    p3::SVector{3,T},
) where {T<:AbstractFloat}
    nraw = cross(p2 - p1, p3 - p1)
    area2 = norm(nraw)
    area = area2 / 2
    area > eps(T) || throw(ArgumentError("Degenerate triangle for Whitney basis evaluation."))
    n̂ = nraw / area2

    # grad λ_i are constant per triangle.
    g1 = cross(n̂, p3 - p2) / (2 * area)
    g2 = cross(n̂, p1 - p3) / (2 * area)
    g3 = cross(n̂, p2 - p1) / (2 * area)

    return WhitneyTriangleGeometry((p1, p2, p3), n̂, area, (g1, g2, g3))
end

function _triangle_geometry(mesh::SurfaceMesh{T}, geom::SurfaceGeometry{T}, faceid::Int) where {T<:AbstractFloat}
    face = mesh.faces[faceid]
    p1 = mesh.points[face[1]]
    p2 = mesh.points[face[2]]
    p3 = mesh.points[face[3]]
    tri = _triangle_geometry_from_points(p1, p2, p3)

    # Geometry area/normal are the canonical package values; enforce numerical
    # consistency with the mesh geometry container.
    return WhitneyTriangleGeometry(tri.points, geom.face_normals[faceid], geom.face_areas[faceid], tri.gradlambda)
end

function _coerce_triangle_geometry(tri_geom)
    if tri_geom isa WhitneyTriangleGeometry
        return tri_geom
    elseif tri_geom isa NTuple{3,SVector{3,<:AbstractFloat}}
        return _triangle_geometry_from_points(tri_geom[1], tri_geom[2], tri_geom[3])
    elseif tri_geom isa AbstractVector && length(tri_geom) == 3 &&
           all(p -> p isa SVector{3,<:AbstractFloat}, tri_geom)
        return _triangle_geometry_from_points(tri_geom[1], tri_geom[2], tri_geom[3])
    else
        throw(ArgumentError("tri_geom must be WhitneyTriangleGeometry or three 3D points."))
    end
end

@inline function _barycentric_coords(ξ)
    if length(ξ) == 2
        λ2 = ξ[1]
        λ3 = ξ[2]
        λ1 = one(λ2) - λ2 - λ3
        return (λ1, λ2, λ3)
    elseif length(ξ) == 3
        return (ξ[1], ξ[2], ξ[3])
    end
    throw(ArgumentError("Local coordinate ξ must have length 2 (reference) or 3 (barycentric)."))
end

@inline function _local_edge_vertices(i::Int)
    i == 1 && return (1, 2)
    i == 2 && return (2, 3)
    i == 3 && return (3, 1)
    throw(ArgumentError("Local Whitney 1-form index must be 1,2,3 (got $i)."))
end

@inline function _triangle_barycentric_gradients(mesh::SurfaceMesh{T}, geom::SurfaceGeometry{T}, faceid::Int) where {T<:AbstractFloat}
    tri = _triangle_geometry(mesh, geom, faceid)
    return tri.gradlambda, tri.area
end

# -----------------------------------------------------------------------------
# Public local Whitney basis API
# -----------------------------------------------------------------------------

"""
    whitney0_basis_local(tri_geom)

Return local lowest-order Whitney 0-form basis functions on one triangle.

Input
-----
- `tri_geom`: either a `WhitneyTriangleGeometry` value or a tuple/vector of
  three 3D triangle points.

Output
------
A tuple of three callables `(ϕ1, ϕ2, ϕ3)` where each callable accepts local
coordinates `ξ` (length-2 reference coordinates or length-3 barycentric
coordinates).
"""
function whitney0_basis_local(tri_geom)
    tri = _coerce_triangle_geometry(tri_geom)
    return (
        ξ -> eval_whitney0_local(1, ξ, tri),
        ξ -> eval_whitney0_local(2, ξ, tri),
        ξ -> eval_whitney0_local(3, ξ, tri),
    )
end

"""
    whitney1_basis_local(tri_geom)

Return local lowest-order Whitney 1-form basis functions on one triangle.

The basis ordering is `(1→2), (2→3), (3→1)` in local face orientation.
"""
function whitney1_basis_local(tri_geom)
    tri = _coerce_triangle_geometry(tri_geom)
    return (
        ξ -> eval_whitney1_local(1, ξ, tri),
        ξ -> eval_whitney1_local(2, ξ, tri),
        ξ -> eval_whitney1_local(3, ξ, tri),
    )
end

"""
    whitney2_basis_local(tri_geom)

Return local lowest-order Whitney 2-form basis function on one triangle.

The returned single callable corresponds to the face-based DOF convention
`∫_face w2 = 1`.
"""
function whitney2_basis_local(tri_geom)
    tri = _coerce_triangle_geometry(tri_geom)
    return (ξ -> eval_whitney2_local(1, ξ, tri),)
end

"""
    eval_whitney0_local(i, ξ, tri_geom)

Evaluate local Whitney 0 basis function `i ∈ {1,2,3}` at local coordinate `ξ`.
"""
function eval_whitney0_local(i::Int, ξ, tri_geom)
    i in (1, 2, 3) || throw(ArgumentError("Whitney 0 basis index must be 1,2,3."))
    λ = _barycentric_coords(ξ)
    return λ[i]
end

"""
    eval_whitney1_local(i, ξ, tri_geom)

Evaluate local Whitney 1 basis function `i ∈ {1,2,3}` at local coordinate `ξ`.

Output is returned in the tangent vector representation using the surface
metric identification.
"""
function eval_whitney1_local(i::Int, ξ, tri_geom)
    tri = _coerce_triangle_geometry(tri_geom)
    λ = _barycentric_coords(ξ)
    a, b = _local_edge_vertices(i)
    return λ[a] * tri.gradlambda[b] - λ[b] * tri.gradlambda[a]
end

"""
    eval_whitney2_local(i, ξ, tri_geom)

Evaluate the local Whitney 2 basis function on one triangle.

Only `i == 1` is valid for lowest-order face-based 2-forms.
"""
function eval_whitney2_local(i::Int, ξ, tri_geom)
    _ = ξ
    tri = _coerce_triangle_geometry(tri_geom)
    i == 1 || throw(ArgumentError("Lowest-order Whitney 2 basis has a single local basis function (index 1)."))
    return inv(tri.area)
end

# -----------------------------------------------------------------------------
# Reconstruction from global cochains
# -----------------------------------------------------------------------------

"""
    reconstruct_0form_face(c0, faceid, mesh, geom)

Reconstruct a piecewise-linear scalar field on one face from vertex 0-cochain
`c0`.

Returns a named tuple with:
- `coefficients`: vertex values `(c_a, c_b, c_c)` on the face,
- `gradient`: constant in-face gradient,
- `eval(ξ)`: local evaluator.
"""
function reconstruct_0form_face(
    c0::AbstractVector{T},
    faceid::Int,
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
) where {T<:AbstractFloat}
    nv = length(mesh.points)
    length(c0) == nv || throw(DimensionMismatch("c0 length $(length(c0)) != nv=$nv"))

    tri = _triangle_geometry(mesh, geom, faceid)
    face = mesh.faces[faceid]
    coeffs = SVector{3,T}(c0[face[1]], c0[face[2]], c0[face[3]])

    grad = coeffs[1] * tri.gradlambda[1] +
           coeffs[2] * tri.gradlambda[2] +
           coeffs[3] * tri.gradlambda[3]

    eval_fun = let coeffs=coeffs
        ξ -> begin
            λ = _barycentric_coords(ξ)
            return coeffs[1] * λ[1] + coeffs[2] * λ[2] + coeffs[3] * λ[3]
        end
    end

    return (coefficients=coeffs, gradient=grad, eval=eval_fun)
end

"""
    reconstruct_1form_face(c1, faceid, mesh, geom)

Reconstruct a lowest-order Whitney 1-form on one face from global edge
cochain `c1`.

Returns a named tuple with:
- `coefficients`: face-oriented local edge DOFs `(α12, α23, α31)`,
- `tangent_at_centroid`: reconstructed tangent vector at barycenter,
- `eval(ξ)`: local evaluator in tangent-vector representation.
"""
function reconstruct_1form_face(
    c1::AbstractVector{T},
    faceid::Int,
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
) where {T<:AbstractFloat}
    topo = build_topology(mesh)
    ne = length(topo.edges)
    length(c1) == ne || throw(DimensionMismatch("c1 length $(length(c1)) != ne=$ne"))

    tri = _triangle_geometry(mesh, geom, faceid)
    fe = topo.face_edges[faceid]
    fs = topo.face_edge_signs[faceid]

    # Local edge basis order follows face orientation (1→2, 2→3, 3→1).
    coeffs = SVector{3,T}(fs[1] * c1[fe[1]], fs[2] * c1[fe[2]], fs[3] * c1[fe[3]])

    eval_fun = let coeffs=coeffs, tri=tri
        ξ -> begin
            return coeffs[1] * eval_whitney1_local(1, ξ, tri) +
                   coeffs[2] * eval_whitney1_local(2, ξ, tri) +
                   coeffs[3] * eval_whitney1_local(3, ξ, tri)
        end
    end

    ξc = SVector{3,T}(T(1//3), T(1//3), T(1//3))
    vcent = eval_fun(ξc)

    return (coefficients=coeffs, tangent_at_centroid=vcent, eval=eval_fun)
end

"""
    reconstruct_2form_face(c2, faceid, mesh, geom)

Reconstruct a facewise constant 2-form density on one face from face cochain
`c2`.

Convention
----------
If `c2[f]` stores the oriented face integral, reconstructed density is
`ρ_f = c2[f] / area_f`.
"""
function reconstruct_2form_face(
    c2::AbstractVector{T},
    faceid::Int,
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
) where {T<:AbstractFloat}
    nf = length(mesh.faces)
    length(c2) == nf || throw(DimensionMismatch("c2 length $(length(c2)) != nf=$nf"))

    q = c2[faceid]
    A = geom.face_areas[faceid]
    A > eps(T) || throw(ArgumentError("Face $faceid has near-zero area."))
    ρ = q / A
    eval_fun = ξ -> ρ

    return (coefficient=q, density=ρ, eval=eval_fun)
end

"""
    reconstruct_0form(c0, mesh, geom; representation=:facewise)

Reconstruct a Whitney 0-form field from vertex cochain `c0`.

Representations
---------------
- `:facewise` (default): vector of per-face reconstruction packs from
  `reconstruct_0form_face`.
- `:facewise_callable`: vector of evaluators.
"""
function reconstruct_0form(
    c0::AbstractVector{T},
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T};
    representation::Symbol=:facewise,
) where {T<:AbstractFloat}
    nf = length(mesh.faces)
    recs = [reconstruct_0form_face(c0, fi, mesh, geom) for fi in 1:nf]

    if representation === :facewise
        return recs
    elseif representation === :facewise_callable
        return [r.eval for r in recs]
    end

    throw(ArgumentError("Unknown representation=$(repr(representation)) for reconstruct_0form."))
end

"""
    reconstruct_1form(c1, mesh, geom; representation=:facewise_tangent)

Reconstruct a Whitney 1-form field from edge cochain `c1`.

Representations
---------------
- `:facewise_tangent` (default): per-face tangent vectors at barycenters.
- `:facewise`: vector of per-face reconstruction packs.
- `:facewise_callable`: vector of evaluators.
"""
function reconstruct_1form(
    c1::AbstractVector{T},
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T};
    representation::Symbol=:facewise_tangent,
) where {T<:AbstractFloat}
    nf = length(mesh.faces)
    recs = [reconstruct_1form_face(c1, fi, mesh, geom) for fi in 1:nf]

    if representation === :facewise_tangent
        return [r.tangent_at_centroid for r in recs]
    elseif representation === :facewise
        return recs
    elseif representation === :facewise_callable
        return [r.eval for r in recs]
    end

    throw(ArgumentError("Unknown representation=$(repr(representation)) for reconstruct_1form."))
end

"""
    reconstruct_2form(c2, mesh, geom; representation=:facewise_density)

Reconstruct a Whitney 2-form field from face cochain `c2`.

Representations
---------------
- `:facewise_density` (default): per-face constant densities.
- `:facewise`: vector of per-face reconstruction packs.
"""
function reconstruct_2form(
    c2::AbstractVector{T},
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T};
    representation::Symbol=:facewise_density,
) where {T<:AbstractFloat}
    nf = length(mesh.faces)
    recs = [reconstruct_2form_face(c2, fi, mesh, geom) for fi in 1:nf]

    if representation === :facewise_density
        return [r.density for r in recs]
    elseif representation === :facewise
        return recs
    elseif representation === :facewise_callable
        return [r.eval for r in recs]
    end

    throw(ArgumentError("Unknown representation=$(repr(representation)) for reconstruct_2form."))
end
