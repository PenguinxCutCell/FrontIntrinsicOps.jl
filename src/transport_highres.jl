# transport_highres.jl – High-resolution surface scalar transport with limiters.
#
# Implements slope-limiter-based high-resolution (2nd-order) edge-flux
# transport on triangulated surfaces.  This extends the 1st-order upwind
# scheme in surface_transport.jl with:
#   - minmod and van Leer limiters,
#   - a limited-flux assembly routine,
#   - an SSP-RK2 integration step (for verification / reference).
#
# PDE: M du/dt + A_hr(u; v) u ≈ 0
#
# The limited flux at each edge is:
#   F_e = v_e * (u_upwind + φ(r_e) * (u_centered - u_upwind) / 2)
#
# where v_e is the edge-flux velocity, u_upwind is the 1st-order upwind value,
# u_centered is the 2nd-order centered value, and φ(r) is the limiter function.
#
# Reference
# ---------
# This follows the standard MUSCL-type approach adapted to unstructured meshes.
# The gradient estimation uses the available vertex values from the topology.

# ─────────────────────────────────────────────────────────────────────────────
# Limiter functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    minmod(a, b) -> T

The minmod function: returns the value with smallest absolute value if `a`
and `b` have the same sign, otherwise zero.

Used to limit slope reconstructions to prevent spurious oscillations.
"""
function minmod(a::T, b::T) :: T where {T<:Real}
    if a * b <= zero(T)
        return zero(T)
    elseif abs(a) <= abs(b)
        return a
    else
        return b
    end
end

minmod(a::Real, b::Real) = minmod(promote(a, b)...)

"""
    minmod3(a, b, c) -> T

Three-argument minmod (returns the value of smallest magnitude if all have
the same sign, else zero).
"""
function minmod3(a::T, b::T, c::T) :: T where {T<:Real}
    if a > zero(T) && b > zero(T) && c > zero(T)
        return min(a, b, c)
    elseif a < zero(T) && b < zero(T) && c < zero(T)
        return max(a, b, c)
    else
        return zero(T)
    end
end

"""
    vanleer_limiter(r) -> T

The van Leer flux limiter:
    φ(r) = (r + |r|) / (1 + |r|)  for r ≥ 0, else 0

Smooth and 2nd-order accurate in smooth regions; TVD.
"""
function vanleer_limiter(r::T) :: T where {T<:Real}
    r < zero(T) && return zero(T)
    return (r + abs(r)) / (one(T) + abs(r))
end

"""
    superbee_limiter(r) -> T

The Superbee flux limiter:
    φ(r) = max(0, min(1, 2r), min(2, r))

More compressive than minmod; can produce sharper fronts.
"""
function superbee_limiter(r::T) :: T where {T<:Real}
    r <= zero(T) && return zero(T)
    return max(min(one(T), 2*r), min(2*one(T), r))
end

# ─────────────────────────────────────────────────────────────────────────────
# Internal: face-centred velocity → edge scalar fluxes
# ─────────────────────────────────────────────────────────────────────────────

# Average face velocities to vertices, then project to edge tangents.
function _face_vel_to_edge_flux(
        mesh      :: SurfaceMesh{T},
        topo      :: MeshTopology,
        vel_faces :: AbstractVector{SVector{3,T}},
) :: Vector{T} where {T}
    nf = length(mesh.faces)
    ne = length(topo.edges)
    nv = length(mesh.points)
    length(vel_faces) == nf ||
        error("_face_vel_to_edge_flux: expected $nf face velocities, got $(length(vel_faces))")

    # Accumulate face velocities at each vertex
    vel_verts = fill(zero(SVector{3,T}), nv)
    counts    = zeros(Int, nv)
    @inbounds for fi in 1:nf
        vf   = vel_faces[fi]
        face = mesh.faces[fi]
        for vi in face
            vel_verts[vi] = vel_verts[vi] + vf
            counts[vi]   += 1
        end
    end
    @inbounds for vi in 1:nv
        if counts[vi] > 0
            vel_verts[vi] = vel_verts[vi] / counts[vi]
        end
    end

    # Project averaged vertex velocities to edge tangents
    vflux = Vector{T}(undef, ne)
    @inbounds for ei in 1:ne
        i, j  = topo.edges[ei][1], topo.edges[ei][2]
        vmid  = (vel_verts[i] + vel_verts[j]) / 2
        dp    = mesh.points[j] - mesh.points[i]
        len   = norm(dp)
        t     = len > eps(T) ? dp / len : zero(SVector{3,T})
        vflux[ei] = dot(vmid, t)
    end
    return vflux
end

# ─────────────────────────────────────────────────────────────────────────────
# Limited transport operator
# ─────────────────────────────────────────────────────────────────────────────

"""
    assemble_transport_operator_limited(mesh, geom, topo, vel, u;
                                        limiter=:minmod)
        -> SparseMatrixCSC{T,Int}

Assemble a high-resolution (limited) upwind edge-flux transport matrix.

For each oriented edge e = (i → j) with edge-flux velocity v_e:
- Identify the upwind vertex (i if v_e > 0, j if v_e < 0).
- Compute a limited slope correction using the limiter function.
- Build the matrix such that: A * u gives M du/dt residuals.

The result is assembled as a sparse matrix that can be applied to a vertex
field `u` to get the transport flux `M du/dt`.

Parameters
----------
- `vel`     – edge-flux velocities (Vector{T} of length nE), or any format
              accepted by `edge_flux_velocity`.
- `u`       – current solution (needed for slope ratio computation in limiters).
- `limiter` – `:minmod` (default), `:vanleer`, `:superbee`, or `:upwind`
              (falls back to 1st-order upwind).

Returns
-------
A sparse matrix `A` of size `(nV × nV)` such that the transport residual is
`A * u` (with sign convention: `M du/dt + A * u = 0`).
"""
function assemble_transport_operator_limited(
        mesh    :: SurfaceMesh{T},
        geom    :: SurfaceGeometry{T},
        topo    :: MeshTopology,
        vel,
        u       :: AbstractVector{T};
        limiter :: Symbol = :minmod,
) :: SparseMatrixCSC{T,Int} where {T}

    # Project velocity to edge fluxes; support face-centred SVector velocity
    nv = length(mesh.points)
    nf = length(mesh.faces)
    v_edge = if vel isa AbstractVector{<:SVector} && length(vel) == nf
        _face_vel_to_edge_flux(mesh, topo, vel)
    else
        edge_flux_velocity(mesh, geom, vel)
    end

    # For :upwind fallback, just use the standard assembly with edge flux velocity
    if limiter === :upwind
        return assemble_transport_operator(mesh, geom, v_edge; scheme=:upwind)
    end

    # Select limiter function
    φ_lim = if limiter === :minmod
        (r::T) -> minmod(r, one(T))
    elseif limiter === :vanleer
        vanleer_limiter
    elseif limiter === :superbee
        superbee_limiter
    else
        error("assemble_transport_operator_limited: unknown limiter $(repr(limiter)).")
    end

    # Build vertex-to-face adjacency for slope estimation
    # For each edge (i,j), we need the "far" vertices in adjacent faces to
    # estimate the gradient ratio.
    # Simple approach: for each edge, use the face-opposite vertices.
    edge_face_list = topo.edge_faces
    ne = length(topo.edges)

    I_idx = Int[]
    J_idx = Int[]
    Vvals = T[]
    sizehint!(I_idx, 6 * ne)
    sizehint!(J_idx, 6 * ne)
    sizehint!(Vvals, 6 * ne)

    @inbounds for ei in 1:ne
        i, j = topo.edges[ei][1], topo.edges[ei][2]
        v_e  = v_edge[ei]

        abs(v_e) < eps(T) * 100 && continue

        # Upwind and downwind vertices
        if v_e > zero(T)
            i_up, i_dn = i, j
        else
            i_up, i_dn = j, i
        end

        # Upwind value and a slope estimate using face information
        # Find faces adjacent to this edge and use opposite vertices
        faces_e  = edge_face_list[ei]
        u_up     = u[i_up]
        u_dn     = u[i_dn]
        Δu_edge  = u_dn - u_up   # centered difference across edge

        # Estimate upwind slope using the "further upwind" information
        # from adjacent face vertices opposite to i_up
        slope_far = zero(T)
        n_far     = 0
        for fi in faces_e
            face = mesh.faces[fi]
            for vi in face
                if vi != i && vi != j
                    # This is the "far" vertex in this face
                    u_far    = u[vi]
                    # Sign: positive if far vertex is "further upwind"
                    slope_far += (u_up - u_far)
                    n_far     += 1
                end
            end
        end

        if n_far > 0
            slope_far /= n_far
        end

        # Limited slope: minmod of the two slope estimates
        φ_val = if abs(Δu_edge) > eps(T) * 100
            r     = slope_far / Δu_edge
            φ_lim(r)
        else
            zero(T)
        end

        # High-resolution flux: F_e = v_e * (u_up + φ * Δu_edge / 2)
        # Low-order: F_e_lo = v_e * u_up
        # High-res correction: v_e * φ * Δu_edge / 2
        flux_lo  = v_e  # factor of u_up applied via matrix
        flux_cor = v_e * φ_val * T(0.5)

        # Low-order upwind contribution:  + flux_lo * u_up  → row i_dn, col i_up
        # Also: conservation: net flux at i_up is -flux_lo * u_up, at i_dn is +flux_lo * u_up
        # For A s.t. M du/dt + A u = 0:
        # d/dt u_i_dn += (1/da_i_dn) * flux_lo * u_up  (inflow)
        # d/dt u_i_up -= (1/da_i_up) * flux_lo * u_up  (outflow)

        # Low-order upwind part
        push!(I_idx, i_dn);  push!(J_idx, i_up);  push!(Vvals, -flux_lo)  # inflow to i_dn
        push!(I_idx, i_up);  push!(J_idx, i_up);  push!(Vvals, +flux_lo)  # outflow from i_up

        # High-resolution correction (from centered difference)
        if abs(flux_cor) > eps(T)
            push!(I_idx, i_dn);  push!(J_idx, i_dn);  push!(Vvals, -flux_cor)
            push!(I_idx, i_dn);  push!(J_idx, i_up);  push!(Vvals, -flux_cor)
            push!(I_idx, i_up);  push!(J_idx, i_dn);  push!(Vvals, +flux_cor)
            push!(I_idx, i_up);  push!(J_idx, i_up);  push!(Vvals, +flux_cor)
        end
    end

    A = sparse(I_idx, J_idx, Vvals, nv, nv)
    return A
end

# ─────────────────────────────────────────────────────────────────────────────
# High-resolution transport step
# ─────────────────────────────────────────────────────────────────────────────

"""
    step_surface_transport_limited(mesh, geom, dec, topo, uⁿ, vel, dt;
                                   limiter=:minmod,
                                   method=:ssprk2)
        -> Vector{T}

Advance the surface transport equation

    M du/dt + A(u; v) u = 0

by one time step using a high-resolution (limited) scheme.

Parameters
----------
- `vel`     – velocity field (any format accepted by `edge_flux_velocity`).
- `dt`      – time step size.
- `limiter` – slope limiter: `:minmod` (default), `:vanleer`, `:superbee`,
              or `:upwind` (1st-order fallback).
- `method`  – time integration: `:ssprk2` (default, 2nd-order SSP Runge–Kutta),
              `:euler` (1st-order explicit Euler).

Returns `uⁿ⁺¹`.
"""
function step_surface_transport_limited(
        mesh    :: SurfaceMesh{T},
        geom    :: SurfaceGeometry{T},
        dec     :: SurfaceDEC{T},
        topo    :: MeshTopology,
        uⁿ      :: AbstractVector{T},
        vel,
        dt      :: Real;
        limiter :: Symbol = :minmod,
        method  :: Symbol = :ssprk2,
) :: Vector{T} where {T}
    dt   = T(dt)
    da   = geom.vertex_dual_areas

    _transport_rhs = function(u)
        A  = assemble_transport_operator_limited(mesh, geom, topo, vel, u; limiter=limiter)
        Au = A * u
        return -(Au ./ da)
    end

    if method === :euler
        return uⁿ .+ dt .* _transport_rhs(uⁿ)
    elseif method === :ssprk2
        # SSP-RK2 (Heun's method):
        k1 = _transport_rhs(uⁿ)
        u1 = uⁿ .+ dt .* k1
        k2 = _transport_rhs(u1)
        return T(0.5) .* (uⁿ .+ u1 .+ dt .* k2)
    elseif method === :ssprk3
        # SSP-RK3 (Shu–Osher):
        k1 = _transport_rhs(uⁿ)
        u1 = uⁿ .+ dt .* k1
        k2 = _transport_rhs(u1)
        u2 = T(3)/T(4) .* uⁿ .+ T(1)/T(4) .* (u1 .+ dt .* k2)
        k3 = _transport_rhs(u2)
        return T(1)/T(3) .* uⁿ .+ T(2)/T(3) .* (u2 .+ dt .* k3)
    else
        error("step_surface_transport_limited: unknown method $(repr(method)). Use :euler, :ssprk2, or :ssprk3.")
    end
end
