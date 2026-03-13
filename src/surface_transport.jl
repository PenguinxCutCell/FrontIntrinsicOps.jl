# surface_transport.jl – Prescribed tangential scalar transport on static fronts.
#
# Implements a conservative edge-flux advection operator for 0-form (vertex)
# scalar fields.  This is a pragmatic first implementation, not a full
# high-order surface finite-volume method.
#
# PDE: M du/dt + A(u; wτ) u = 0
#
# where wτ is a prescribed tangential velocity field and A is the assembled
# advection operator.
#
# Velocity input formats
# ----------------------
# The transport functions accept velocity as:
#   (a) Matrix (nv × 3) or Vector{SVector{3,T}} – per-vertex ambient velocity.
#   (b) AbstractVector{T} of length ne – pre-projected edge-flux velocities.
#   (c) Callable f(pt) -> SVector{3,T} – evaluated on each vertex.
#
# The internal representation used for assembly is the edge-flux velocity:
#   vₑ = (vᵢ + vⱼ)/2 · tₑ   for edge e = (i,j)
# where tₑ is the unit edge tangent.

# ─────────────────────────────────────────────────────────────────────────────
# Tangential projection (surface)
# ─────────────────────────────────────────────────────────────────────────────

"""
    tangential_projection(mesh::SurfaceMesh, geom::SurfaceGeometry, v)
        -> Vector{SVector{3,T}}

Project per-vertex ambient velocity `v` onto the tangent plane of the surface.

The tangential part is:  vτ = v - (v · n̂) n̂

`v` can be:
- `Vector{SVector{3,T}}` – one vector per vertex.
- `AbstractMatrix` of size `(nv, 3)` – each row is a 3-D velocity.
"""
function tangential_projection(
        mesh :: SurfaceMesh{T},
        geom :: SurfaceGeometry{T},
        v    :: Vector{SVector{3,T}},
) :: Vector{SVector{3,T}} where {T}
    nv = length(mesh.points)
    length(v) == nv || error("tangential_projection: velocity length mismatch")
    vτ = Vector{SVector{3,T}}(undef, nv)
    @inbounds for i in 1:nv
        n = geom.vertex_normals[i]
        vτ[i] = v[i] - dot(v[i], n) * n
    end
    return vτ
end

function tangential_projection(
        mesh :: SurfaceMesh{T},
        geom :: SurfaceGeometry{T},
        v    :: AbstractMatrix,
) :: Vector{SVector{3,T}} where {T}
    nv = length(mesh.points)
    size(v, 1) == nv && size(v, 2) == 3 ||
        error("tangential_projection: velocity matrix must be (nv × 3)")
    vsvec = [SVector{3,T}(v[i,1], v[i,2], v[i,3]) for i in 1:nv]
    return tangential_projection(mesh, geom, vsvec)
end

# ─────────────────────────────────────────────────────────────────────────────
# Edge flux velocity
# ─────────────────────────────────────────────────────────────────────────────

"""
    edge_flux_velocity(mesh::CurveMesh, geom::CurveGeometry, vel)
        -> Vector{T}

Compute the scalar edge-flux velocity for a curve.

For edge e = (i → j) the flux velocity is:
    vₑ = (v_i + v_j) / 2 · tₑ
where tₑ is the unit edge tangent.

`vel` can be:
- `Vector{T}` of length nv – scalar tangential speeds at each vertex.
- `Vector{SVector{2,T}}` – 2-D velocity vectors per vertex (projected onto tₑ).
- Callable f(pt::SVector{2,T}) -> SVector{2,T} – evaluated on each vertex.
- `Vector{T}` of length ne – already edge-flux values (returned as-is).
"""
function edge_flux_velocity(
        mesh :: CurveMesh{T},
        geom :: CurveGeometry{T},
        vel  :: Vector{SVector{2,T}},
) :: Vector{T} where {T}
    ne  = length(mesh.edges)
    nv  = length(mesh.points)
    length(vel) == nv || error("edge_flux_velocity: velocity length mismatch")
    vflux = Vector{T}(undef, ne)
    @inbounds for (ei, e) in enumerate(mesh.edges)
        i, j  = e[1], e[2]
        vmid  = (vel[i] + vel[j]) / 2
        t     = geom.edge_tangents[ei]
        vflux[ei] = dot(vmid, t)
    end
    return vflux
end

function edge_flux_velocity(
        mesh :: CurveMesh{T},
        geom :: CurveGeometry{T},
        vel  :: AbstractVector{T},
) :: Vector{T} where {T}
    ne = length(mesh.edges)
    nv = length(mesh.points)
    if length(vel) == ne
        return convert(Vector{T}, vel)   # already edge fluxes
    elseif length(vel) == nv
        # scalar vertex speed along edge tangent
        vflux = Vector{T}(undef, ne)
        @inbounds for (ei, e) in enumerate(mesh.edges)
            i, j = e[1], e[2]
            vflux[ei] = (vel[i] + vel[j]) / 2
        end
        return vflux
    else
        error("edge_flux_velocity: velocity length $(length(vel)) does not " *
              "match nv=$nv or ne=$ne")
    end
end

function edge_flux_velocity(
        mesh :: CurveMesh{T},
        geom :: CurveGeometry{T},
        vel  :: Function,
) :: Vector{T} where {T}
    nv = length(mesh.points)
    vel_vecs = [vel(mesh.points[i]) for i in 1:nv]
    # Convert to SVector{2,T} if needed
    if eltype(vel_vecs) <: SVector{2}
        return edge_flux_velocity(mesh, geom, Vector{SVector{2,T}}(vel_vecs))
    else
        # Assume scalar
        return edge_flux_velocity(mesh, geom, T.(vel_vecs))
    end
end

"""
    edge_flux_velocity(mesh::SurfaceMesh, geom::SurfaceGeometry, vel)
        -> Vector{T}

Compute the scalar edge-flux velocity for a surface.

For edge e = (i, j) with unit tangent tₑ = (pⱼ - pᵢ)/|pⱼ - pᵢ|:
    vₑ = (vᵢ + vⱼ) / 2 · tₑ

`vel` can be:
- `Vector{SVector{3,T}}` – 3-D velocity vectors per vertex.
- `AbstractMatrix` of size (nv, 3) – rows are velocity vectors.
- Callable f(pt) -> SVector{3,T}.
- `Vector{T}` of length ne – already edge-flux values.
"""
function edge_flux_velocity(
        mesh :: SurfaceMesh{T},
        geom :: SurfaceGeometry{T},
        vel  :: Vector{SVector{3,T}},
) :: Vector{T} where {T}
    topo = build_topology(mesh)
    ne   = length(topo.edges)
    nv   = length(mesh.points)
    length(vel) == nv || error("edge_flux_velocity: velocity length mismatch")
    vflux = Vector{T}(undef, ne)
    @inbounds for (ei, e) in enumerate(topo.edges)
        i, j  = e[1], e[2]
        vmid  = (vel[i] + vel[j]) / 2
        dp    = mesh.points[j] - mesh.points[i]
        len   = norm(dp)
        t     = len > eps(T) ? dp / len : zero(SVector{3,T})
        vflux[ei] = dot(vmid, t)
    end
    return vflux
end

function edge_flux_velocity(
        mesh :: SurfaceMesh{T},
        geom :: SurfaceGeometry{T},
        vel  :: AbstractMatrix,
) :: Vector{T} where {T}
    nv = length(mesh.points)
    size(vel, 1) == nv && size(vel, 2) == 3 ||
        error("edge_flux_velocity: velocity matrix must be (nv × 3)")
    vel_vecs = [SVector{3,T}(vel[i,1], vel[i,2], vel[i,3]) for i in 1:nv]
    return edge_flux_velocity(mesh, geom, vel_vecs)
end

function edge_flux_velocity(
        mesh :: SurfaceMesh{T},
        geom :: SurfaceGeometry{T},
        vel  :: AbstractVector{T},
) :: Vector{T} where {T}
    topo = build_topology(mesh)
    ne   = length(topo.edges)
    nv   = length(mesh.points)
    if length(vel) == ne
        return convert(Vector{T}, vel)
    else
        error("edge_flux_velocity: velocity length $(length(vel)) does not " *
              "match ne=$ne. For per-vertex velocities use Vector{SVector{3,T}}.")
    end
end

function edge_flux_velocity(
        mesh :: SurfaceMesh{T},
        geom :: SurfaceGeometry{T},
        vel  :: Function,
) :: Vector{T} where {T}
    nv = length(mesh.points)
    vel_vecs = Vector{SVector{3,T}}(undef, nv)
    @inbounds for i in 1:nv
        v = vel(mesh.points[i])
        vel_vecs[i] = SVector{3,T}(v[1], v[2], v[3])
    end
    return edge_flux_velocity(mesh, geom, vel_vecs)
end

# ─────────────────────────────────────────────────────────────────────────────
# Conservative transport operator assembly
# ─────────────────────────────────────────────────────────────────────────────

"""
    assemble_transport_operator(mesh::CurveMesh, geom::CurveGeometry, vel;
                                scheme=:centered) -> SparseMatrixCSC{T,Int}

Assemble the conservative advection matrix A for a closed curve.

The semi-discrete equation is:  M du/dt + A u = 0.

The advection matrix uses an edge-flux formulation:
- `:centered` scheme: upwind flux averaged (reduces to face-average interpolation).
- `:upwind`   scheme: donor-cell (first-order) upwind.

The returned matrix A is such that `A * u` approximates `M * (v · ∇ u)` in a
conservative sense.

Note: The `vel` argument is passed to `edge_flux_velocity` internally.
"""
function assemble_transport_operator(
        mesh   :: CurveMesh{T},
        geom   :: CurveGeometry{T},
        vel;
        scheme :: Symbol = :centered,
) :: SparseMatrixCSC{T,Int} where {T}
    scheme in (:centered, :upwind) ||
        error("assemble_transport_operator: unknown scheme $(repr(scheme)). " *
              "Use :centered or :upwind.")

    vflux = edge_flux_velocity(mesh, geom, vel)
    ne    = length(mesh.edges)
    nv    = length(mesh.points)

    # For each oriented edge e = (i → j) with flux velocity vₑ:
    # Centered: flux F_e = vₑ * (uᵢ + uⱼ) / 2
    #   contribution to vertex i: -F_e * 1/dual_length_i  (flux out of i)
    #   contribution to vertex j: +F_e * 1/dual_length_j  (flux into j)
    # Upwind: F_e = vₑ * (vₑ > 0 ? uᵢ : uⱼ)
    #
    # We build:  A[i, i] += ..., A[i, j] += ..., etc.
    # Note: M = diag(dual_lengths), so the full PDE form is M du/dt + A u = 0
    # means A has units [dual_length / time / dual_length] = [1/time] after
    # dividing the dual area out here.

    I_ind = Int[]
    J_ind = Int[]
    V_val = T[]

    for (ei, e) in enumerate(mesh.edges)
        i, j = e[1], e[2]
        ve   = vflux[ei]

        if scheme === :centered
            # F = ve * (ui + uj) / 2
            # row i: contribution from this edge = -F (divergence from vertex i)
            # row j: +F (flux into vertex j)
            push!(I_ind, i); push!(J_ind, i); push!(V_val,  ve / 2)
            push!(I_ind, i); push!(J_ind, j); push!(V_val,  ve / 2)
            push!(I_ind, j); push!(J_ind, i); push!(V_val, -ve / 2)
            push!(I_ind, j); push!(J_ind, j); push!(V_val, -ve / 2)
        else  # :upwind
            if ve >= 0
                # flow from i to j: upwind is uᵢ
                push!(I_ind, i); push!(J_ind, i); push!(V_val,  ve)
                push!(I_ind, j); push!(J_ind, i); push!(V_val, -ve)
            else
                # flow from j to i: upwind is uⱼ
                push!(I_ind, i); push!(J_ind, j); push!(V_val,  ve)
                push!(I_ind, j); push!(J_ind, j); push!(V_val, -ve)
            end
        end
    end

    A_raw = sparse(I_ind, J_ind, V_val, nv, nv)

    # Scale by M^{-1} so the PDE becomes du/dt + M^{-1} A_raw u = 0
    # i.e. A = M^{-1} A_raw  and the user adds M back via M du/dt + A_raw u
    # We return A_raw (the mass-weighted form) so the user writes M du/dt + A u
    return A_raw
end

"""
    assemble_transport_operator(mesh::SurfaceMesh, geom::SurfaceGeometry, vel;
                                scheme=:centered) -> SparseMatrixCSC{T,Int}

Assemble the conservative advection matrix A for a triangulated surface.

The semi-discrete equation is:  M du/dt + A u = 0.

This is a pragmatic edge-flux discretisation for vertex-based 0-form unknowns
on the primal mesh.  For each edge e = (i, j):

- `:centered`: flux = vₑ * (uᵢ + uⱼ)/2  (consistent, but not monotone).
- `:upwind`:   donor-cell (first-order upwind for stability).

The edge flux velocity vₑ is computed from `vel` via `edge_flux_velocity`.
"""
function assemble_transport_operator(
        mesh   :: SurfaceMesh{T},
        geom   :: SurfaceGeometry{T},
        vel;
        scheme :: Symbol = :centered,
) :: SparseMatrixCSC{T,Int} where {T}
    scheme in (:centered, :upwind) ||
        error("assemble_transport_operator: unknown scheme $(repr(scheme)). " *
              "Use :centered or :upwind.")

    vflux = edge_flux_velocity(mesh, geom, vel)
    topo  = build_topology(mesh)
    ne    = length(topo.edges)
    nv    = length(mesh.points)

    I_ind = Int[]
    J_ind = Int[]
    V_val = T[]
    sizehint!(I_ind, 4 * ne)
    sizehint!(J_ind, 4 * ne)
    sizehint!(V_val, 4 * ne)

    for (ei, e) in enumerate(topo.edges)
        i, j = e[1], e[2]
        ve   = vflux[ei]

        if scheme === :centered
            push!(I_ind, i); push!(J_ind, i); push!(V_val,  ve / 2)
            push!(I_ind, i); push!(J_ind, j); push!(V_val,  ve / 2)
            push!(I_ind, j); push!(J_ind, i); push!(V_val, -ve / 2)
            push!(I_ind, j); push!(J_ind, j); push!(V_val, -ve / 2)
        else  # :upwind
            if ve >= 0
                push!(I_ind, i); push!(J_ind, i); push!(V_val,  ve)
                push!(I_ind, j); push!(J_ind, i); push!(V_val, -ve)
            else
                push!(I_ind, i); push!(J_ind, j); push!(V_val,  ve)
                push!(I_ind, j); push!(J_ind, j); push!(V_val, -ve)
            end
        end
    end

    return sparse(I_ind, J_ind, V_val, nv, nv)
end

# ─────────────────────────────────────────────────────────────────────────────
# CFL helper
# ─────────────────────────────────────────────────────────────────────────────

"""
    estimate_transport_dt(mesh::CurveMesh, geom::CurveGeometry, vel;
                          cfl=0.5) -> T

Estimate a stable explicit time step for scalar transport on a curve.

Uses the vertex-based stability criterion:

    dt = cfl * min_v ( dual_length_v / Σ_{adj e} |vₑ| )

which ensures the upwind forward-Euler scheme is stable.
"""
function estimate_transport_dt(
        mesh :: CurveMesh{T},
        geom :: CurveGeometry{T},
        vel;
        cfl  :: Real = 0.5,
) :: T where {T}
    vflux = edge_flux_velocity(mesh, geom, vel)
    nv    = length(mesh.points)

    # Accumulate |ve| per vertex
    flux_sum = zeros(T, nv)
    for (ei, e) in enumerate(mesh.edges)
        absv = abs(vflux[ei])
        flux_sum[e[1]] += absv
        flux_sum[e[2]] += absv
    end

    dt_min = typemax(T)
    for i in 1:nv
        flux_sum[i] < eps(T) && continue
        dt_v = geom.vertex_dual_lengths[i] / flux_sum[i]
        dt_min = min(dt_min, dt_v)
    end
    dt_min == typemax(T) && return T(Inf)
    return T(cfl) * dt_min
end

"""
    estimate_transport_dt(mesh::SurfaceMesh, geom::SurfaceGeometry, vel;
                          cfl=0.5) -> T

Estimate a stable explicit time step for scalar transport on a surface.

Uses the vertex-based stability criterion:

    dt = cfl * min_v ( dual_area_v / Σ_{adj e} |vₑ| )

which matches the actual explicit CFL condition for the assembled upwind
operator on an unstructured mesh.
"""
function estimate_transport_dt(
        mesh :: SurfaceMesh{T},
        geom :: SurfaceGeometry{T},
        vel;
        cfl  :: Real = 0.5,
) :: T where {T}
    vflux = edge_flux_velocity(mesh, geom, vel)
    topo  = build_topology(mesh)
    nv    = length(mesh.points)

    # Accumulate |ve| per vertex
    flux_sum = zeros(T, nv)
    for (ei, e) in enumerate(topo.edges)
        absv = abs(vflux[ei])
        flux_sum[e[1]] += absv
        flux_sum[e[2]] += absv
    end

    dt_min = typemax(T)
    for i in 1:nv
        flux_sum[i] < eps(T) && continue
        dt_v = geom.vertex_dual_areas[i] / flux_sum[i]
        dt_min = min(dt_min, dt_v)
    end
    dt_min == typemax(T) && return T(Inf)
    return T(cfl) * dt_min
end

# ─────────────────────────────────────────────────────────────────────────────
# Explicit time-steppers for transport
# ─────────────────────────────────────────────────────────────────────────────

"""
    step_surface_transport_forward_euler(mesh, geom, A, uⁿ, dt) -> Vector{T}

Advance  M du/dt + A u = 0  by one explicit forward-Euler step:

    M uⁿ⁺¹ = M uⁿ - dt * A uⁿ
    uⁿ⁺¹   = uⁿ - dt * M⁻¹ A uⁿ

where `A` is the transport operator from `assemble_transport_operator`.
"""
function step_surface_transport_forward_euler(
        mesh :: Union{CurveMesh{T}, SurfaceMesh{T}},
        geom :: Union{CurveGeometry{T}, SurfaceGeometry{T}},
        A    :: AbstractSparseMatrix{T},
        uⁿ   :: AbstractVector{T},
        dt   :: Real,
) :: Vector{T} where {T}
    M    = mass_matrix(mesh, geom)
    Minv = _inv_diag(M)
    return uⁿ .- T(dt) .* (Minv * (A * uⁿ))
end

"""
    step_surface_transport_ssprk2(mesh, geom, A, uⁿ, dt) -> Vector{T}

Advance transport by one SSP-RK2 (Heun's method / Shu-Osher) step.

Stage 1:  u* = uⁿ - dt * M⁻¹ A uⁿ
Stage 2:  uⁿ⁺¹ = (1/2)(uⁿ + u* - dt * M⁻¹ A u*)
"""
function step_surface_transport_ssprk2(
        mesh :: Union{CurveMesh{T}, SurfaceMesh{T}},
        geom :: Union{CurveGeometry{T}, SurfaceGeometry{T}},
        A    :: AbstractSparseMatrix{T},
        uⁿ   :: AbstractVector{T},
        dt   :: Real,
) :: Vector{T} where {T}
    M    = mass_matrix(mesh, geom)
    Minv = _inv_diag(M)
    dt   = T(dt)

    Lu0 = Minv * (A * uⁿ)
    u1  = uⁿ .- dt .* Lu0

    Lu1 = Minv * (A * u1)
    return T(0.5) .* (uⁿ .+ u1 .- dt .* Lu1)
end

"""
    step_surface_transport_ssprk3(mesh, geom, A, uⁿ, dt) -> Vector{T}

Advance transport by one SSP-RK3 (Shu-Osher 3-stage) step.

Classic 3rd-order strong-stability-preserving Runge–Kutta.
"""
function step_surface_transport_ssprk3(
        mesh :: Union{CurveMesh{T}, SurfaceMesh{T}},
        geom :: Union{CurveGeometry{T}, SurfaceGeometry{T}},
        A    :: AbstractSparseMatrix{T},
        uⁿ   :: AbstractVector{T},
        dt   :: Real,
) :: Vector{T} where {T}
    M    = mass_matrix(mesh, geom)
    Minv = _inv_diag(M)
    dt   = T(dt)

    L(u) = Minv * (A * u)

    u1 = uⁿ .- dt .* L(uⁿ)
    u2 = T(3)/4 .* uⁿ .+ T(1)/4 .* (u1 .- dt .* L(u1))
    return T(1)/3 .* uⁿ .+ T(2)/3 .* (u2 .- dt .* L(u2))
end

# ─────────────────────────────────────────────────────────────────────────────
# Internal helper: diagonal inverse of a diagonal sparse matrix
# ─────────────────────────────────────────────────────────────────────────────

function _inv_diag(M::SparseMatrixCSC{T,Int}) :: SparseMatrixCSC{T,Int} where {T}
    d = diag(M)
    inv_d = [v > eps(T) ? one(T)/v : zero(T) for v in d]
    return spdiagm(0 => inv_d)
end
