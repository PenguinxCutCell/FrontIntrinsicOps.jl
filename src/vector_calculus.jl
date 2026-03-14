# vector_calculus.jl – Intrinsic surface vector calculus toolbox.
#
# Provides high-level utilities for working with tangential vector fields,
# scalar gradients as surface vectors, surface divergence, and conversions
# between 1-forms (edge cochains) and tangential vector fields (face/vertex).
#
# Sign / orientation conventions follow the rest of the package:
# - Face normals are outward-pointing (right-hand rule from CCW face orientation).
# - 1-forms are edge cochains α[e] for edge e = (i→j) canonical (i < j).
# - d0[e] = u[j] - u[i] for the discrete exterior derivative.
# - The surface gradient of a scalar field u at a face is computed using the
#   standard Whitney-like formula from the DEC literature.

# ─────────────────────────────────────────────────────────────────────────────
# A. Tangential projection
# ─────────────────────────────────────────────────────────────────────────────

"""
    tangential_project(v, n) -> SVector{3,T}

Project a single 3-D vector `v` onto the tangent plane with normal `n`:

    vτ = v - (v ⋅ n̂) n̂

`n` need not be unit-length; it is normalised internally.
"""
function tangential_project(
        v :: SVector{3,T},
        n :: SVector{3,T},
) :: SVector{3,T} where {T}
    nn = n / norm(n)
    return v - dot(v, nn) * nn
end

"""
    tangential_project!(v, n) -> SVector{3,T}

In-place version: project `v` onto the tangent plane with unit normal `n`.
`n` is assumed normalised.  Returns the projected vector.
"""
function tangential_project!(
        v :: SVector{3,T},
        n :: SVector{3,T},
) :: SVector{3,T} where {T}
    return v - dot(v, n) * n
end

"""
    tangential_project_field(mesh, geom, vfield; location=:vertex)
        -> Vector{SVector{3,T}}

Project an ambient 3-D vector field onto the local tangent plane of the surface.

Parameters
----------
- `vfield`   – per-vertex (`:vertex`) or per-face (`:face`) vector field.
- `location` – `:vertex` uses vertex normals; `:face` uses face normals.

Returns a vector of the same length as the input with each vector projected
to be tangential to the surface.
"""
function tangential_project_field(
        mesh     :: SurfaceMesh{T},
        geom     :: SurfaceGeometry{T},
        vfield   :: Vector{SVector{3,T}};
        location :: Symbol = :vertex,
) :: Vector{SVector{3,T}} where {T}
    if location === :vertex
        nv = length(mesh.points)
        length(vfield) == nv ||
            error("tangential_project_field: vfield length $(length(vfield)) ≠ nv=$nv")
        out = Vector{SVector{3,T}}(undef, nv)
        @inbounds for i in 1:nv
            n = geom.vertex_normals[i]
            out[i] = vfield[i] - dot(vfield[i], n) * n
        end
        return out
    elseif location === :face
        nf = length(mesh.faces)
        length(vfield) == nf ||
            error("tangential_project_field: vfield length $(length(vfield)) ≠ nf=$nf")
        out = Vector{SVector{3,T}}(undef, nf)
        @inbounds for fi in 1:nf
            n = geom.face_normals[fi]
            out[fi] = vfield[fi] - dot(vfield[fi], n) * n
        end
        return out
    else
        error("tangential_project_field: unknown location $(repr(location)). Use :vertex or :face.")
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# B. Surface gradient as tangent vectors
# ─────────────────────────────────────────────────────────────────────────────

"""
    gradient_0_to_tangent_vectors(mesh, geom, u; location=:face)
        -> Vector{SVector{3,T}}

Compute the surface gradient of a vertex scalar field `u` as a tangential
vector field.

For each face f = (a, b, c) with unit normal n̂_f and area A_f:

    ∇_Γ u|_f = (1 / (2 A_f)) * (u_a (n̂_f × (p_c - p_b))
                                + u_b (n̂_f × (p_a - p_c))
                                + u_c (n̂_f × (p_b - p_a)))

This is exact for piecewise-linear fields and matches the DEC gradient `d0 u`
in the sense that projecting the result onto edge tangents recovers `d0`.

Parameters
----------
- `location` – `:face` (default) returns one vector per face.  `:vertex` returns
               an area-weighted average of adjacent face gradients at each vertex.

Returns
-------
A `Vector{SVector{3,T}}` of length `nF` (`:face`) or `nV` (`:vertex`).
"""
function gradient_0_to_tangent_vectors(
        mesh     :: SurfaceMesh{T},
        geom     :: SurfaceGeometry{T},
        u        :: AbstractVector{T};
        location :: Symbol = :face,
) :: Vector{SVector{3,T}} where {T}
    nf = length(mesh.faces)
    nv = length(mesh.points)
    length(u) == nv ||
        error("gradient_0_to_tangent_vectors: u length $(length(u)) ≠ nv=$nv")

    # Compute face gradients
    grads_f = Vector{SVector{3,T}}(undef, nf)
    @inbounds for fi in 1:nf
        a, b, c   = mesh.faces[fi][1], mesh.faces[fi][2], mesh.faces[fi][3]
        pa, pb, pc = mesh.points[a], mesh.points[b], mesh.points[c]
        n_f        = geom.face_normals[fi]
        A_f        = geom.face_areas[fi]
        ua, ub, uc = u[a], u[b], u[c]
        # ∇λ_a = n̂ × (p_c - p_b) / (2A_f), etc.
        grads_f[fi] = (ua * cross(n_f, pc - pb) +
                       ub * cross(n_f, pa - pc) +
                       uc * cross(n_f, pb - pa)) / (2 * A_f)
    end

    if location === :face
        return grads_f
    elseif location === :vertex
        # Area-weighted average of adjacent face gradients
        topo     = build_topology(mesh)
        grads_v  = zeros(SVector{3,T}, nv)
        weights  = zeros(T, nv)
        @inbounds for fi in 1:nf
            A_f = geom.face_areas[fi]
            for vi in mesh.faces[fi]
                grads_v[vi]  = grads_v[vi]  .+ A_f .* grads_f[fi]
                weights[vi]  += A_f
            end
        end
        @inbounds for i in 1:nv
            w = weights[i]
            if w > eps(T)
                grads_v[i] = grads_v[i] ./ w
            end
        end
        return grads_v
    else
        error("gradient_0_to_tangent_vectors: unknown location $(repr(location)). Use :face or :vertex.")
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# C. Surface divergence
# ─────────────────────────────────────────────────────────────────────────────

"""
    divergence_tangent_vectors(mesh, geom, vfield; location=:face)
        -> Vector{T}

Compute the scalar surface divergence of a tangential vector field, returning
a 0-form (vertex field).

For a per-face vector field `vfield`, the divergence at each vertex is computed
using the discrete DEC route:
1. Convert face vectors to a 1-form via `tangent_vectors_to_1form`.
2. Apply the codifferential δ₁ to get the vertex divergence.

Parameters
----------
- `vfield`   – per-face (`:face`) tangential vector field.
- `location` – only `:face` is currently implemented.

Returns
-------
Vertex 0-form of length `nV`.
"""
function divergence_tangent_vectors(
        mesh     :: SurfaceMesh{T},
        geom     :: SurfaceGeometry{T},
        vfield   :: Vector{SVector{3,T}};
        location :: Symbol = :face,
) :: Vector{T} where {T}
    location === :face ||
        error("divergence_tangent_vectors: only location=:face is implemented")
    topo = build_topology(mesh)
    dec  = build_dec(mesh, geom)
    α    = tangent_vectors_to_1form(mesh, geom, topo, vfield; location=:face)
    δ1   = codifferential_1(mesh, geom, dec)
    return δ1 * α
end

# ─────────────────────────────────────────────────────────────────────────────
# D. 1-form ↔ tangent vector conversions
# ─────────────────────────────────────────────────────────────────────────────

"""
    tangent_vectors_to_1form(mesh, geom, topo, vfield; location=:face)
        -> Vector{T}

Convert a tangential vector field to a discrete 1-form (edge cochain).

For each edge e = (i, j) with canonical orientation i→j:
    α[e] = (average of V_f ⋅ t_e over faces adjacent to e) * |e|
where t_e = (p_j - p_i) / |e| is the unit edge tangent.

If `location=:face` and an edge has two adjacent faces, the average of the
two face vectors is projected onto the edge tangent.

Parameters
----------
- `vfield`   – per-face (`:face`) tangential vector field.
- `location` – only `:face` is currently implemented.

Returns
-------
Edge 1-form of length `nE`.
"""
function tangent_vectors_to_1form(
        mesh     :: SurfaceMesh{T},
        geom     :: SurfaceGeometry{T},
        topo     :: MeshTopology,
        vfield   :: Vector{SVector{3,T}};
        location :: Symbol = :face,
) :: Vector{T} where {T}
    location === :face ||
        error("tangent_vectors_to_1form: only location=:face is implemented")
    nf = length(mesh.faces)
    length(vfield) == nf ||
        error("tangent_vectors_to_1form: vfield length $(length(vfield)) ≠ nf=$nf")

    ne = length(topo.edges)
    α  = zeros(T, ne)
    counts = zeros(Int, ne)

    @inbounds for ei in 1:ne
        i, j  = topo.edges[ei][1], topo.edges[ei][2]
        edge_vec = mesh.points[j] - mesh.points[i]
        t_e   = edge_vec / geom.edge_lengths[ei]
        el    = geom.edge_lengths[ei]
        for fi in topo.edge_faces[ei]
            α[ei] += dot(vfield[fi], t_e) * el
            counts[ei] += 1
        end
    end

    @inbounds for ei in 1:ne
        if counts[ei] > 1
            α[ei] /= counts[ei]
        end
    end

    return α
end

"""
    oneform_to_tangent_vectors(mesh, geom, topo, α; location=:face)
        -> Vector{SVector{3,T}}

Convert a discrete 1-form (edge cochain) to a tangential vector field at faces.

For each face f = (a, b, c) with edges e₀, e₁, e₂ and signed face-edge
orientations σ₀, σ₁, σ₂:

    V_f = (1 / (2 A_f)) * n̂_f × (α̃₀ ev₀ + α̃₁ ev₁ + α̃₂ ev₂)

where α̃_k = σ_{f,k} α[e_k] is the face-relative signed value and ev_k is the
signed edge vector in the face-traversal direction.

This is the inverse of `tangent_vectors_to_1form` to leading order.  For
exact 1-forms α = d₀u this recovers the surface gradient.

Parameters
----------
- `location` – only `:face` is currently implemented.

Returns
-------
Per-face tangential vector field of length `nF`.
"""
function oneform_to_tangent_vectors(
        mesh     :: SurfaceMesh{T},
        geom     :: SurfaceGeometry{T},
        topo     :: MeshTopology,
        α        :: AbstractVector{T};
        location :: Symbol = :face,
) :: Vector{SVector{3,T}} where {T}
    location === :face ||
        error("oneform_to_tangent_vectors: only location=:face is implemented")
    ne = length(topo.edges)
    length(α) == ne ||
        error("oneform_to_tangent_vectors: α length $(length(α)) ≠ ne=$ne")

    nf    = length(mesh.faces)
    vecs  = Vector{SVector{3,T}}(undef, nf)

    @inbounds for fi in 1:nf
        fe   = topo.face_edges[fi]
        fs   = topo.face_edge_signs[fi]
        n_f  = geom.face_normals[fi]
        A_f  = geom.face_areas[fi]

        # Accumulate: sum_k σ_k α[e_k] * e_vec_k (signed face direction)
        s = zero(SVector{3,T})
        for k in 1:3
            ek  = fe[k]
            σk  = T(fs[k])
            αk̃  = σk * α[ek]
            # global edge vector (canonical i < j)
            i, j = topo.edges[ek][1], topo.edges[ek][2]
            ev   = mesh.points[j] - mesh.points[i]
            # face-oriented edge vector
            ev_face = σk * ev
            s = s .+ αk̃ .* ev_face
        end
        # V_f = (1/(2A_f)) * n_f × s
        vecs[fi] = cross(n_f, s) / (2 * A_f)
    end

    return vecs
end

# ─────────────────────────────────────────────────────────────────────────────
# E. Curl-like utilities (scalar surface rot)
# ─────────────────────────────────────────────────────────────────────────────

"""
    surface_rot_0form(mesh, geom, topo, u)
        -> Vector{SVector{3,T}}

Compute the rotated surface gradient (surface "curl" of a scalar 0-form):

    rot_Γ u|_f = n̂_f × ∇_Γ u|_f

This is the surface analogue of the 2-D curl curl(∇u) applied to a tangent
plane.  The result is a tangential vector field (per face) that is perpendicular
to the gradient direction in the tangent plane.

On a flat surface this equals the 90°-rotated gradient (in the plane).
"""
function surface_rot_0form(
        mesh :: SurfaceMesh{T},
        geom :: SurfaceGeometry{T},
        u    :: AbstractVector{T},
) :: Vector{SVector{3,T}} where {T}
    grads = gradient_0_to_tangent_vectors(mesh, geom, u; location=:face)
    nf    = length(mesh.faces)
    rots  = Vector{SVector{3,T}}(undef, nf)
    @inbounds for fi in 1:nf
        rots[fi] = cross(geom.face_normals[fi], grads[fi])
    end
    return rots
end
