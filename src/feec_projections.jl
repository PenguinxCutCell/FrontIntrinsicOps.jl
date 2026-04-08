# feec_projections.jl – Canonical FEEC interpolators / projections.

# -----------------------------------------------------------------------------
# Internal callback helpers
# -----------------------------------------------------------------------------

@inline _scalar_from(x::Real, ::Type{T}) where {T<:AbstractFloat} = T(x)

function _vector3_from(v, ::Type{T}) where {T<:AbstractFloat}
    if v isa SVector{3,T}
        return v
    elseif v isa SVector{3,<:Real}
        return SVector{3,T}(T(v[1]), T(v[2]), T(v[3]))
    elseif v isa AbstractVector && length(v) == 3
        return SVector{3,T}(T(v[1]), T(v[2]), T(v[3]))
    elseif v isa NTuple{3,<:Real}
        return SVector{3,T}(T(v[1]), T(v[2]), T(v[3]))
    end
    throw(ArgumentError("Expected 3D vector value from callback, got $(typeof(v))."))
end

function _vector2_from(v, ::Type{T}) where {T<:AbstractFloat}
    if v isa SVector{2,T}
        return v
    elseif v isa SVector{2,<:Real}
        return SVector{2,T}(T(v[1]), T(v[2]))
    elseif v isa AbstractVector && length(v) == 2
        return SVector{2,T}(T(v[1]), T(v[2]))
    elseif v isa NTuple{2,<:Real}
        return SVector{2,T}(T(v[1]), T(v[2]))
    end
    throw(ArgumentError("Expected 2D vector value from callback, got $(typeof(v))."))
end

function _eval_ambient_vector_callback(α, x, t, eid::Int, mesh, geom, ::Type{T}) where {T<:AbstractFloat}
    if applicable(α, x, t, eid, mesh, geom)
        return _vector3_from(α(x, t, eid, mesh, geom), T)
    elseif applicable(α, x, t, eid)
        return _vector3_from(α(x, t, eid), T)
    elseif applicable(α, x, t)
        return _vector3_from(α(x, t), T)
    elseif applicable(α, x)
        return _vector3_from(α(x), T)
    end
    throw(ArgumentError("Ambient vector callback is not callable with supported signatures."))
end

function _eval_ambient_vector_callback_curve(α, x, t, eid::Int, mesh, geom, ::Type{T}) where {T<:AbstractFloat}
    if applicable(α, x, t, eid, mesh, geom)
        return _vector2_from(α(x, t, eid, mesh, geom), T)
    elseif applicable(α, x, t, eid)
        return _vector2_from(α(x, t, eid), T)
    elseif applicable(α, x, t)
        return _vector2_from(α(x, t), T)
    elseif applicable(α, x)
        return _vector2_from(α(x), T)
    end
    throw(ArgumentError("Ambient vector callback is not callable with supported signatures."))
end

function _eval_line_density_callback(α, x, t, eid::Int, mesh, geom, ::Type{T}) where {T<:AbstractFloat}
    if applicable(α, x, t, eid, mesh, geom)
        return T(α(x, t, eid, mesh, geom))
    elseif applicable(α, x, t, eid)
        return T(α(x, t, eid))
    elseif applicable(α, x, t)
        return T(α(x, t))
    elseif applicable(α, x)
        return T(α(x))
    end
    throw(ArgumentError("Line density callback is not callable with supported signatures."))
end

function _eval_face_density_callback(β, x, n, fi::Int, mesh, geom, ::Type{T}) where {T<:AbstractFloat}
    if applicable(β, x, n, fi, mesh, geom)
        return T(β(x, n, fi, mesh, geom))
    elseif applicable(β, x, n, fi)
        return T(β(x, n, fi))
    elseif applicable(β, x, n)
        return T(β(x, n))
    elseif applicable(β, x)
        return T(β(x))
    end
    throw(ArgumentError("Face density callback is not callable with supported signatures."))
end

function _eval_face_integral_callback(β, x, n, A, fi::Int, mesh, geom, ::Type{T}) where {T<:AbstractFloat}
    if applicable(β, fi, mesh, geom)
        return T(β(fi, mesh, geom))
    elseif applicable(β, fi)
        return T(β(fi))
    elseif applicable(β, x, n, A, fi, mesh, geom)
        return T(β(x, n, A, fi, mesh, geom))
    elseif applicable(β, x, n, A, fi)
        return T(β(x, n, A, fi))
    end
    throw(ArgumentError("Face-integral callback is not callable with supported signatures."))
end

# -----------------------------------------------------------------------------
# Canonical FEEC interpolation operators Πk
# -----------------------------------------------------------------------------

"""
    interpolate_0form(f, mesh, geom; target=:whitney) -> c0

Canonical lowest-order interpolation of a scalar 0-form into vertex DOFs.

DOF convention
--------------
`c0[v] = f(x_v)` at mesh vertices.
"""
function interpolate_0form(
    f,
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T};
    target::Symbol=:whitney,
) where {T<:AbstractFloat}
    _ = geom
    target === :whitney || throw(ArgumentError("Unsupported target=$(repr(target))."))
    nv = length(mesh.points)

    if f isa AbstractVector
        length(f) == nv || throw(DimensionMismatch("0-form vector length $(length(f)) != nv=$nv"))
        return T.(f)
    end

    out = Vector{T}(undef, nv)
    @inbounds for i in 1:nv
        out[i] = T(f(mesh.points[i]))
    end
    return out
end

function interpolate_0form(
    f,
    mesh::CurveMesh{T},
    geom::CurveGeometry{T};
    target::Symbol=:whitney,
) where {T<:AbstractFloat}
    _ = geom
    target === :whitney || throw(ArgumentError("Unsupported target=$(repr(target))."))
    nv = length(mesh.points)

    if f isa AbstractVector
        length(f) == nv || throw(DimensionMismatch("0-form vector length $(length(f)) != nv=$nv"))
        return T.(f)
    end

    out = Vector{T}(undef, nv)
    @inbounds for i in 1:nv
        out[i] = T(f(mesh.points[i]))
    end
    return out
end

"""
    interpolate_1form(α, mesh, geom; target=:whitney, representation=:ambient_vector) -> c1

Canonical lowest-order interpolation of a 1-form into oriented edge DOFs.

DOF convention
--------------
`c1[e] = ∫_e α` on each oriented primal edge.

Supported representations (surface)
-----------------------------------
- `:ambient_vector` (default): callback returns an ambient 3D vector field,
  integrated against oriented edge tangents.
- `:line_density`: callback returns a scalar line density along edge tangent.
"""
function interpolate_1form(
    α,
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T};
    target::Symbol=:whitney,
    representation::Symbol=:ambient_vector,
) where {T<:AbstractFloat}
    target === :whitney || throw(ArgumentError("Unsupported target=$(repr(target))."))
    topo = build_topology(mesh)
    ne = length(topo.edges)

    if α isa AbstractVector
        length(α) == ne || throw(DimensionMismatch("1-form vector length $(length(α)) != ne=$ne"))
        return T.(α)
    end

    out = zeros(T, ne)
    @inbounds for ei in 1:ne
        i, j = topo.edges[ei][1], topo.edges[ei][2]
        pi = mesh.points[i]
        pj = mesh.points[j]
        evec = pj - pi
        ℓ = geom.edge_lengths[ei]
        t = ℓ > eps(T) ? evec / ℓ : zero(SVector{3,T})
        xmid = (pi + pj) / 2

        if representation === :ambient_vector
            v = _eval_ambient_vector_callback(α, xmid, t, ei, mesh, geom, T)
            out[ei] = dot(v, evec)
        elseif representation === :line_density
            ρ = _eval_line_density_callback(α, xmid, t, ei, mesh, geom, T)
            out[ei] = ρ * ℓ
        else
            throw(ArgumentError("Unsupported 1-form representation=$(repr(representation))."))
        end
    end

    return out
end

function interpolate_1form(
    α,
    mesh::CurveMesh{T},
    geom::CurveGeometry{T};
    target::Symbol=:whitney,
    representation::Symbol=:tangent_speed,
) where {T<:AbstractFloat}
    target === :whitney || throw(ArgumentError("Unsupported target=$(repr(target))."))
    ne = length(mesh.edges)

    if α isa AbstractVector
        length(α) == ne || throw(DimensionMismatch("1-form vector length $(length(α)) != ne=$ne"))
        return T.(α)
    end

    out = zeros(T, ne)
    @inbounds for ei in 1:ne
        i, j = mesh.edges[ei][1], mesh.edges[ei][2]
        pi = mesh.points[i]
        pj = mesh.points[j]
        evec = pj - pi
        ℓ = geom.edge_lengths[ei]
        t = ℓ > eps(T) ? evec / ℓ : zero(SVector{2,T})
        xmid = (pi + pj) / 2

        if representation === :tangent_speed || representation === :line_density
            ρ = _eval_line_density_callback(α, xmid, t, ei, mesh, geom, T)
            out[ei] = ρ * ℓ
        elseif representation === :ambient_vector
            v = _eval_ambient_vector_callback_curve(α, xmid, t, ei, mesh, geom, T)
            out[ei] = dot(v, evec)
        else
            throw(ArgumentError("Unsupported curve 1-form representation=$(repr(representation))."))
        end
    end

    return out
end

"""
    interpolate_2form(β, mesh, geom; target=:whitney, representation=:density) -> c2

Canonical lowest-order interpolation of a surface 2-form into face DOFs.

DOF convention
--------------
`c2[f] = ∫_f β` on each oriented primal face.

Supported representations
-------------------------
- `:density` (default): callback returns scalar surface density.
- `:face_integral`: callback returns the face integral directly.
"""
function interpolate_2form(
    β,
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T};
    target::Symbol=:whitney,
    representation::Symbol=:density,
) where {T<:AbstractFloat}
    target === :whitney || throw(ArgumentError("Unsupported target=$(repr(target))."))
    nf = length(mesh.faces)

    if β isa AbstractVector
        length(β) == nf || throw(DimensionMismatch("2-form vector length $(length(β)) != nf=$nf"))
        return T.(β)
    end

    out = zeros(T, nf)
    @inbounds for fi in 1:nf
        f = mesh.faces[fi]
        x = (mesh.points[f[1]] + mesh.points[f[2]] + mesh.points[f[3]]) / 3
        n = geom.face_normals[fi]
        A = geom.face_areas[fi]

        if representation === :density
            ρ = _eval_face_density_callback(β, x, n, fi, mesh, geom, T)
            out[fi] = ρ * A
        elseif representation === :face_integral
            out[fi] = _eval_face_integral_callback(β, x, n, A, fi, mesh, geom, T)
        else
            throw(ArgumentError("Unsupported 2-form representation=$(repr(representation))."))
        end
    end

    return out
end

function interpolate_2form(
    β,
    mesh::CurveMesh,
    geom::CurveGeometry;
    target::Symbol=:whitney,
    representation::Symbol=:density,
)
    _ = β
    _ = mesh
    _ = geom
    _ = target
    _ = representation
    throw(ArgumentError("interpolate_2form is not defined for curve meshes (no 2-cells)."))
end

"""
    Π0(f, mesh, geom)

Alias for `interpolate_0form(f, mesh, geom)`.
"""
Π0(f, mesh, geom; kwargs...) = interpolate_0form(f, mesh, geom; kwargs...)

"""
    Π1(α, mesh, geom)

Alias for `interpolate_1form(α, mesh, geom)`.
"""
Π1(α, mesh, geom; kwargs...) = interpolate_1form(α, mesh, geom; kwargs...)

"""
    Π2(β, mesh, geom)

Alias for `interpolate_2form(β, mesh, geom)`.
"""
Π2(β, mesh, geom; kwargs...) = interpolate_2form(β, mesh, geom; kwargs...)

# -----------------------------------------------------------------------------
# Commuting helpers
# -----------------------------------------------------------------------------

"""
    interpolate_exact_gradient(f, mesh, geom) -> c1

Interpolate an exact differential `df` into edge DOFs using endpoint values:

`c1[e=(i→j)] = f(x_j) - f(x_i)`.

This realizes the canonical identity `Π1(df) = d0 Π0(f)` in the discrete
cochain representation.
"""
function interpolate_exact_gradient(
    f,
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
) where {T<:AbstractFloat}
    _ = geom
    c0 = interpolate_0form(f, mesh, geom)
    d0 = incidence_0(mesh)
    return d0 * c0
end

function interpolate_exact_gradient(
    f,
    mesh::CurveMesh{T},
    geom::CurveGeometry{T},
) where {T<:AbstractFloat}
    _ = geom
    c0 = interpolate_0form(f, mesh, geom)
    d0 = incidence_0(mesh)
    return d0 * c0
end

"""
    interpolate_exact_flux_density(α, mesh, geom; kwargs...) -> c2

Interpolate `dα` into face DOFs using Stokes' theorem:

`Π2(dα) = d1 Π1(α)`.
"""
function interpolate_exact_flux_density(
    α,
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T};
    kwargs...,
) where {T<:AbstractFloat}
    c1 = interpolate_1form(α, mesh, geom; kwargs...)
    d1 = incidence_1(mesh)
    return d1 * c1
end

"""
    projection_commutator_01(f, mesh, geom, dec) -> residual

Return the residual vector for the commuting relation:

`Π1(df) - d0 Π0(f)`.
"""
function projection_commutator_01(
    f,
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    dec::SurfaceDEC{T},
) where {T<:AbstractFloat}
    lhs = interpolate_exact_gradient(f, mesh, geom)
    rhs = dec.d0 * interpolate_0form(f, mesh, geom)
    return lhs - rhs
end

function projection_commutator_01(
    f,
    mesh::CurveMesh{T},
    geom::CurveGeometry{T},
    dec::CurveDEC{T},
) where {T<:AbstractFloat}
    lhs = interpolate_exact_gradient(f, mesh, geom)
    rhs = dec.d0 * interpolate_0form(f, mesh, geom)
    return lhs - rhs
end

"""
    projection_commutator_12(α, mesh, geom, dec) -> residual

Return the residual vector for the commuting relation:

`Π2(dα) - d1 Π1(α)`.

The implementation uses the canonical Stokes interpretation of `Π2(dα)`.
"""
function projection_commutator_12(
    α,
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    dec::SurfaceDEC{T};
    kwargs...,
) where {T<:AbstractFloat}
    lhs = interpolate_exact_flux_density(α, mesh, geom; kwargs...)
    rhs = dec.d1 * interpolate_1form(α, mesh, geom; kwargs...)
    return lhs - rhs
end
