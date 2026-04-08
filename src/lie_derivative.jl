# lie_derivative.jl – Interior product and Lie derivative (Cartan formula).

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

function _face_to_vertex_average(
    face_vals::AbstractVector{T},
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
) where {T}
    nv = length(mesh.points)
    out = zeros(T, nv)
    w = zeros(T, nv)

    for (fi, f) in enumerate(mesh.faces)
        A = geom.face_areas[fi]
        v = face_vals[fi]
        for vi in f
            out[vi] += A * v
            w[vi] += A
        end
    end

    for i in 1:nv
        w[i] > eps(T) && (out[i] /= w[i])
    end
    return out
end

function _require_face_vector_field(X, mesh::SurfaceMesh)
    nf = length(mesh.faces)
    length(X) == nf || throw(DimensionMismatch("Face vector field length $(length(X)) != nf=$nf"))
    return nothing
end

function _curve_edge_speed(
    X,
    mesh::CurveMesh{T},
) where {T}
    nv = length(mesh.points)
    ne = length(mesh.edges)

    if length(X) == nv
        Xv = Vector{T}(X)
        Xe = zeros(T, ne)
        for (ei, e) in enumerate(mesh.edges)
            i, j = e[1], e[2]
            Xe[ei] = (Xv[i] + Xv[j]) / 2
        end
        return Xe
    elseif length(X) == ne
        return Vector{T}(X)
    end

    throw(DimensionMismatch("Curve tangent speed field must have length nv=$nv or ne=$ne."))
end

function _curve_edge_to_vertex_average(
    edge_vals::AbstractVector{T},
    mesh::CurveMesh{T},
    geom::CurveGeometry{T},
) where {T}
    nv = length(mesh.points)
    out = zeros(T, nv)
    w = zeros(T, nv)

    for (ei, e) in enumerate(mesh.edges)
        i, j = e[1], e[2]
        ℓ = geom.edge_lengths[ei]
        out[i] += ℓ * edge_vals[ei]
        out[j] += ℓ * edge_vals[ei]
        w[i] += ℓ
        w[j] += ℓ
    end

    for i in 1:nv
        w[i] > eps(T) && (out[i] /= w[i])
    end
    return out
end

function _curve_degree(
    α::AbstractVector,
    mesh::CurveMesh;
    degree::Union{Nothing,Int}=nothing,
)
    if degree !== nothing
        degree in (0, 1) || throw(ArgumentError("Curve degree must be 0 or 1, got $degree."))
        return degree
    end

    n0 = length(mesh.points)
    n1 = length(mesh.edges)
    n = length(α)

    if n0 == n1 && (n == n0 || n == n1)
        throw(ArgumentError("Ambiguous curve cochain degree on closed curves (nV == nE == $n). Pass degree=0 or degree=1."))
    elseif n == n0
        return 0
    elseif n == n1
        return 1
    end

    throw(DimensionMismatch("Curve cochain length $n does not match nV=$n0 or nE=$n1."))
end

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

"""
    interior_product(X, α, mesh, geom, dec; representation=:face_vector)

Discrete contraction `i_X α` on surfaces using a face-based tangent vector
field representation.

Supported surface cases
-----------------------
- `α` 1-form (edge cochain)  -> 0-form (vertex cochain)
- `α` 2-form (face cochain)  -> 1-form (edge cochain)

For `α` a 0-form, this implementation returns an empty vector (the
continuous `i_X` on 0-forms is identically zero).

Curve support
-------------
For `CurveMesh`, use `representation=:tangent_speed` where `X` is a tangent
speed field on vertices (`length=nV`) or edges (`length=nE`), and pass
`degree=0/1` on closed curves where `nV == nE` to disambiguate cochain degree.
"""
function interior_product(
    X,
    α::AbstractVector{T},
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    dec::SurfaceDEC{T};
    representation::Symbol=:face_vector,
) where {T}
    representation === :face_vector || throw(ArgumentError("Only representation=:face_vector is implemented."))
    _require_face_vector_field(X, mesh)

    k = _cochain_degree(mesh, α)
    topo = build_topology(mesh)

    if k == 0
        return zeros(T, 0)
    elseif k == 1
        # i_X α ≈ <X, α^#> as a face scalar, then averaged to vertices.
        Vα = oneform_to_tangent_vectors(mesh, geom, topo, α; location=:face)
        face_vals = zeros(T, length(mesh.faces))
        for fi in eachindex(face_vals)
            face_vals[fi] = dot(X[fi], Vα[fi])
        end
        return _face_to_vertex_average(face_vals, mesh, geom)
    elseif k == 2
        ne = length(topo.edges)
        out = zeros(T, ne)

        for (ei, e) in enumerate(topo.edges)
            i, j = e[1], e[2]
            t = normalize_safe(mesh.points[j] - mesh.points[i])
            acc = zero(T)
            cnt = 0
            for fi in topo.edge_faces[ei]
                A = geom.face_areas[fi]
                q = A > eps(T) ? α[fi] / A : zero(T)
                acc += q * dot(geom.face_normals[fi], cross(X[fi], t))
                cnt += 1
            end
            cnt > 0 && (out[ei] = (acc / cnt) * geom.edge_lengths[ei])
        end
        return out
    end

    throw(ArgumentError("Unsupported contraction degree k=$k on surfaces."))
end

function interior_product(
    X,
    α::AbstractVector{T},
    mesh::CurveMesh{T},
    geom::CurveGeometry{T},
    dec::CurveDEC{T};
    representation::Symbol=:face_vector,
    degree::Union{Nothing,Int}=nothing,
) where {T}
    representation === :tangent_speed ||
        throw(ArgumentError("Curve contraction expects representation=:tangent_speed."))

    k = _curve_degree(α, mesh; degree=degree)
    if k == 0
        return zeros(T, 0)
    elseif k == 1
        Xe = _curve_edge_speed(X, mesh)
        αt = zeros(T, length(mesh.edges))
        for ei in eachindex(mesh.edges)
            ℓ = geom.edge_lengths[ei]
            αt[ei] = ℓ > eps(T) ? α[ei] / ℓ : zero(T)
        end
        edge_vals = Xe .* αt
        return _curve_edge_to_vertex_average(edge_vals, mesh, geom)
    end

    throw(ArgumentError("Unsupported contraction degree k=$k on curves."))
end

"""
    lie_derivative(X, α, mesh, geom, dec; method=:cartan, representation=:face_vector)

Compute a practical DEC Lie derivative using Cartan's formula

`L_X α = i_X dα + d(i_X α)`

for supported degrees.

Surface
-------
Supported degrees: `α ∈ Ω^0, Ω^1, Ω^2` with `representation=:face_vector`.

Curve
-----
Supported degrees: `α ∈ Ω^0, Ω^1` with `representation=:tangent_speed`.
On closed curves (`nV == nE`), pass `degree=0` or `degree=1` explicitly.
"""
function lie_derivative(
    X,
    α::AbstractVector{T},
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    dec::SurfaceDEC{T};
    method::Symbol=:cartan,
    representation::Symbol=:face_vector,
) where {T}
    method === :cartan || throw(ArgumentError("Only method=:cartan is implemented."))
    representation === :face_vector || throw(ArgumentError("Only representation=:face_vector is implemented."))
    _require_face_vector_field(X, mesh)

    k = _cochain_degree(mesh, α)

    if k == 0
        # Directional derivative of scalar field.
        grads = gradient_0_to_tangent_vectors(mesh, geom, α; location=:face)
        face_vals = zeros(T, length(mesh.faces))
        for fi in eachindex(face_vals)
            face_vals[fi] = dot(X[fi], grads[fi])
        end
        return _face_to_vertex_average(face_vals, mesh, geom)
    elseif k == 1
        iXdα = interior_product(X, dec.d1 * α, mesh, geom, dec; representation=representation)
        iXα = interior_product(X, α, mesh, geom, dec; representation=representation)
        return iXdα .+ dec.d0 * iXα
    elseif k == 2
        iXα = interior_product(X, α, mesh, geom, dec; representation=representation)
        return dec.d1 * iXα
    end

    throw(ArgumentError("Unsupported Lie derivative degree k=$k on surfaces."))
end

function lie_derivative(
    X,
    α::AbstractVector{T},
    mesh::CurveMesh{T},
    geom::CurveGeometry{T},
    dec::CurveDEC{T};
    method::Symbol=:cartan,
    representation::Symbol=:face_vector,
    degree::Union{Nothing,Int}=nothing,
) where {T}
    method === :cartan || throw(ArgumentError("Only method=:cartan is implemented."))
    representation === :tangent_speed ||
        throw(ArgumentError("Curve Lie derivative expects representation=:tangent_speed."))

    k = _curve_degree(α, mesh; degree=degree)
    if k == 0
        # In 1D, L_X f = i_X(df).
        return interior_product(X, dec.d0 * α, mesh, geom, dec; representation=representation, degree=1)
    elseif k == 1
        # In 1D, dα = 0 (top-degree), so Cartan reduces to d(i_X α).
        iXα = interior_product(X, α, mesh, geom, dec; representation=representation, degree=1)
        return dec.d0 * iXα
    end

    throw(ArgumentError("Unsupported Lie derivative degree k=$k on curves."))
end

"""
    cartan_lie_derivative(X, α, mesh, geom, dec; kwargs...)

Alias for `lie_derivative(...; method=:cartan)`.
"""
function cartan_lie_derivative(
    X,
    α,
    mesh,
    geom,
    dec;
    kwargs...,
)
    return lie_derivative(X, α, mesh, geom, dec; method=:cartan, kwargs...)
end
