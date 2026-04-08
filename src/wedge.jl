# wedge.jl – Low-order discrete wedge product on primal cochains.

# -----------------------------------------------------------------------------
# Degree inference helpers
# -----------------------------------------------------------------------------

function _cochain_degree(mesh::CurveMesh, α::AbstractVector)
    n0 = length(mesh.points)
    n1 = length(mesh.edges)
    n = length(α)
    n == n0 && return 0
    n == n1 && return 1
    throw(ArgumentError("Could not infer cochain degree for curve: length=$n (expected $n0 or $n1)."))
end

function _cochain_degree(mesh::SurfaceMesh, α::AbstractVector)
    topo = build_topology(mesh)
    n0 = length(mesh.points)
    n1 = length(topo.edges)
    n2 = length(mesh.faces)
    n = length(α)
    n == n0 && return 0
    n == n1 && return 1
    n == n2 && return 2
    throw(ArgumentError("Could not infer cochain degree for surface: length=$n (expected $n0, $n1, or $n2)."))
end

# -----------------------------------------------------------------------------
# 0 ∧ k
# -----------------------------------------------------------------------------

"""
    wedge0k(f0, αk, mesh, geom, dec)

Compute `f ∧ α` for a 0-form `f` and a k-form `α`.

Conventions
-----------
- `k=0`: pointwise multiplication at vertices.
- `k=1`: edge scaling with arithmetic edge-midpoint averaging of `f`.
- `k=2` (surface): face scaling with arithmetic face-centroid averaging of `f`.
"""
function wedge0k(
    f0::AbstractVector{T},
    αk::AbstractVector{T},
    mesh::CurveMesh{T},
    geom::CurveGeometry{T},
    dec::CurveDEC{T},
) where {T}
    k = _cochain_degree(mesh, αk)
    k == 0 || k == 1 || throw(ArgumentError("Curve wedge0k supports k=0 or k=1 only."))

    if k == 0
        length(f0) == length(αk) || throw(DimensionMismatch("0-form lengths must match."))
        return f0 .* αk
    end

    length(f0) == length(mesh.points) || throw(DimensionMismatch("f0 must be a vertex 0-form."))
    length(αk) == length(mesh.edges) || throw(DimensionMismatch("αk must be an edge 1-form."))

    out = similar(αk)
    for (ei, e) in enumerate(mesh.edges)
        i, j = e[1], e[2]
        favg = (f0[i] + f0[j]) / 2
        out[ei] = favg * αk[ei]
    end
    return out
end

function wedge0k(
    f0::AbstractVector{T},
    αk::AbstractVector{T},
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    dec::SurfaceDEC{T},
) where {T}
    k = _cochain_degree(mesh, αk)
    k in (0, 1, 2) || throw(ArgumentError("Surface wedge0k supports k=0,1,2 only."))

    if k == 0
        length(f0) == length(αk) || throw(DimensionMismatch("0-form lengths must match."))
        return f0 .* αk
    elseif k == 1
        topo = build_topology(mesh)
        length(f0) == length(mesh.points) || throw(DimensionMismatch("f0 must be a vertex 0-form."))
        length(αk) == length(topo.edges) || throw(DimensionMismatch("αk must be an edge 1-form."))

        out = similar(αk)
        for (ei, e) in enumerate(topo.edges)
            i, j = e[1], e[2]
            favg = (f0[i] + f0[j]) / 2
            out[ei] = favg * αk[ei]
        end
        return out
    else
        length(f0) == length(mesh.points) || throw(DimensionMismatch("f0 must be a vertex 0-form."))
        length(αk) == length(mesh.faces) || throw(DimensionMismatch("αk must be a face 2-form."))

        out = similar(αk)
        for (fi, f) in enumerate(mesh.faces)
            favg = (f0[f[1]] + f0[f[2]] + f0[f[3]]) / 3
            out[fi] = favg * αk[fi]
        end
        return out
    end
end

# -----------------------------------------------------------------------------
# 1 ∧ 1 -> 2 (surface)
# -----------------------------------------------------------------------------

"""
    wedge11(α1, β1, mesh, geom, dec) -> Vector

Compute a low-order discrete wedge product `α ∧ β` for two surface 1-forms,
returning a face 2-cochain.

Convention
----------
On each face, we reconstruct tangent vectors from edge 1-cochains and use

`(α ∧ β)_f = area_f * n_f · (V_α × V_β)`.

This construction is antisymmetric by design.
"""
function wedge11(
    α1::AbstractVector{T},
    β1::AbstractVector{T},
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    dec::SurfaceDEC{T},
) where {T}
    topo = build_topology(mesh)
    ne = length(topo.edges)
    length(α1) == ne || throw(DimensionMismatch("α1 must be an edge 1-form of length $ne."))
    length(β1) == ne || throw(DimensionMismatch("β1 must be an edge 1-form of length $ne."))

    Vα = oneform_to_tangent_vectors(mesh, geom, topo, α1; location=:face)
    Vβ = oneform_to_tangent_vectors(mesh, geom, topo, β1; location=:face)

    out = zeros(T, length(mesh.faces))
    for fi in eachindex(mesh.faces)
        out[fi] = geom.face_areas[fi] * dot(geom.face_normals[fi], cross(Vα[fi], Vβ[fi]))
    end
    return out
end

# -----------------------------------------------------------------------------
# Generic wedge dispatcher
# -----------------------------------------------------------------------------

"""
    wedge(a, b, mesh, geom, dec)

Generic low-order wedge dispatcher for supported cochain degree combinations.

Supported cases (surface)
-------------------------
- `0∧0 -> 0`
- `0∧1, 1∧0 -> 1`
- `0∧2, 2∧0 -> 2`
- `1∧1 -> 2`

Supported cases (curve)
-----------------------
- `0∧0 -> 0`
- `0∧1, 1∧0 -> 1`
"""
function wedge(
    a::AbstractVector{T},
    b::AbstractVector{T},
    mesh::CurveMesh{T},
    geom::CurveGeometry{T},
    dec::CurveDEC{T},
) where {T}
    da = _cochain_degree(mesh, a)
    db = _cochain_degree(mesh, b)

    if da == 0
        return wedge0k(a, b, mesh, geom, dec)
    elseif db == 0
        return wedge0k(b, a, mesh, geom, dec)
    end

    throw(ArgumentError("Unsupported curve wedge degrees ($da, $db)."))
end

function wedge(
    a::AbstractVector{T},
    b::AbstractVector{T},
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    dec::SurfaceDEC{T},
) where {T}
    da = _cochain_degree(mesh, a)
    db = _cochain_degree(mesh, b)

    if da == 0
        return wedge0k(a, b, mesh, geom, dec)
    elseif db == 0
        return wedge0k(b, a, mesh, geom, dec)
    elseif da == 1 && db == 1
        return wedge11(a, b, mesh, geom, dec)
    end

    throw(ArgumentError("Unsupported surface wedge degrees ($da, $db)."))
end

