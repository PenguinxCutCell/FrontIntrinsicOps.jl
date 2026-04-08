# geodesics.jl – Intrinsic geodesic distances and path extraction.

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

function _canonical_sources(sources, nv::Int)
    src = if sources isa Integer
        [Int(sources)]
    else
        sort(unique(Int.(collect(sources))))
    end
    isempty(src) && throw(ArgumentError("At least one source vertex is required."))
    any(s -> s < 1 || s > nv, src) && throw(BoundsError("Source vertex out of range 1:$nv."))
    return src
end

function _check_connected_surface(mesh::SurfaceMesh, topo::MeshTopology)
    _, ncomp = _vertex_components(mesh, topo)
    ncomp == 1 || throw(ArgumentError("geodesic tools currently require a connected surface mesh (found $ncomp components)."))
    return nothing
end

function _auto_heat_timestep(geom::SurfaceGeometry{T}) where {T}
    isempty(geom.edge_lengths) && return one(T)
    h = sum(geom.edge_lengths) / length(geom.edge_lengths)
    return h * h
end

function _heat_factor(
    dec::SurfaceDEC{T},
    t::T;
    factor_cache=nothing,
) where {T}
    key = (:geodesic_heat_factor, objectid(dec.lap0), t)
    return _cache_get_or_build!(factor_cache, key) do
        n = size(dec.lap0, 1)
        A = sparse(T(1) * LinearAlgebra.I(n)) + t * dec.lap0
        factorize(A)
    end
end

function _poisson_factor(
    dec::SurfaceDEC{T},
    reg::T;
    factor_cache=nothing,
) where {T}
    key = (:geodesic_poisson_factor, objectid(dec.lap0), objectid(dec.star0), reg)
    return _cache_get_or_build!(factor_cache, key) do
        factorize(dec.lap0 + reg * dec.star0)
    end
end

function _solve_geodesic_single_source(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    dec::SurfaceDEC{T},
    source::Int;
    timestep=:auto,
    factor_cache=nothing,
    fix_constant::Bool=true,
) where {T}
    nv = length(mesh.points)
    src = Int(source)

    t = timestep === :auto ? _auto_heat_timestep(geom) : T(timestep)
    t > zero(T) || throw(ArgumentError("timestep must be positive."))

    # 1) Heat solve: (I + t L) u = δ_source
    rhs_heat = zeros(T, nv)
    rhs_heat[src] = one(T)
    heat_fac = _heat_factor(dec, t; factor_cache=factor_cache)
    u = heat_fac \ rhs_heat

    # 2) Normalized field X = -∇u / |∇u| on faces
    grad_u = gradient_0_to_tangent_vectors(mesh, geom, u; location=:face)
    X = Vector{SVector{3,T}}(undef, length(grad_u))
    @inbounds for fi in eachindex(grad_u)
        g = grad_u[fi]
        ng = norm(g)
        if ng > eps(T)
            X[fi] = -g / ng
        else
            X[fi] = SVector{3,T}(0, 0, 0)
        end
    end

    # 3) Poisson solve: L φ = div(X)
    divX = divergence_tangent_vectors(mesh, geom, X; location=:face)
    m0 = diag(dec.star0)
    _enforce_weighted_zero_mean!(divX, m0)

    reg = T(1e-10)
    poi_fac = _poisson_factor(dec, reg; factor_cache=factor_cache)
    φ = poi_fac \ divX
    _enforce_weighted_zero_mean!(φ, m0)

    # 4) Fix additive constant
    if fix_constant
        φ .-= φ[src]
    else
        φ .-= minimum(φ)
    end

    # Geodesic distances are nonnegative by construction (small clipping only).
    @inbounds for i in eachindex(φ)
        if φ[i] < zero(T)
            φ[i] = zero(T)
        end
    end
    return φ
end

function _vertex_adjacency_with_edges(topo::MeshTopology, nv::Int)
    adj = [Vector{Tuple{Int,Int}}() for _ in 1:nv] # (neighbor, edge_id)
    for (ei, e) in enumerate(topo.edges)
        i, j = e[1], e[2]
        push!(adj[i], (j, ei))
        push!(adj[j], (i, ei))
    end
    for a in adj
        sort!(a, by = x -> (x[1], x[2]))
    end
    return adj
end

function _dijkstra_path(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    topo::MeshTopology,
    src::Int,
    dst::Int,
) where {T}
    nv = length(mesh.points)
    adj = _vertex_adjacency_with_edges(topo, nv)

    dist = fill(T(Inf), nv)
    prev = fill(0, nv)
    used = falses(nv)
    dist[src] = zero(T)

    for _ in 1:nv
        u = 0
        best = T(Inf)
        for i in 1:nv
            if !used[i] && dist[i] < best
                best = dist[i]
                u = i
            end
        end
        u == 0 && break
        u == dst && break
        used[u] = true

        for (v, ei) in adj[u]
            alt = dist[u] + geom.edge_lengths[ei]
            if alt < dist[v]
                dist[v] = alt
                prev[v] = u
            end
        end
    end

    prev[dst] == 0 && src != dst && return Int[src, dst]

    path = Int[dst]
    cur = dst
    while cur != src && cur != 0
        cur = prev[cur]
        cur == 0 && break
        push!(path, cur)
    end
    reverse!(path)
    return path
end

function _as_point3(::Type{T}, p) where {T}
    if p isa SVector{3,T}
        return p
    elseif p isa SVector{3}
        return SVector{3,T}(p)
    elseif p isa NTuple{3}
        return SVector{3,T}(p[1], p[2], p[3])
    elseif p isa AbstractVector && length(p) == 3
        return SVector{3,T}(T(p[1]), T(p[2]), T(p[3]))
    end
    throw(ArgumentError("Point must be a 3D coordinate (SVector, tuple, or length-3 vector)."))
end

function _nearest_vertex_id(mesh::SurfaceMesh{T}, p) where {T}
    q = _as_point3(T, p)
    best = 1
    bestd = norm(mesh.points[1] - q)
    for i in 2:length(mesh.points)
        d = norm(mesh.points[i] - q)
        if d < bestd
            bestd = d
            best = i
        end
    end
    return best
end

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

"""
    geodesic_distance(mesh, geom, dec, sources;
                      method=:heat,
                      timestep=:auto,
                      factor_cache=nothing,
                      fix_constant=true) -> Vector

Compute intrinsic geodesic distance from one or more source vertices.

Method
------
- `:heat` (default): heat method (Crane et al.) on the surface triangulation.

Notes
-----
For multiple sources, this implementation returns the pointwise minimum of
single-source distances for deterministic behavior.
"""
function geodesic_distance(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    dec::SurfaceDEC{T},
    sources;
    method::Symbol=:heat,
    timestep=:auto,
    factor_cache=nothing,
    fix_constant::Bool=true,
) where {T}
    method === :heat || throw(ArgumentError("Unsupported method=$(repr(method)); only :heat is implemented."))

    topo = build_topology(mesh)
    _check_connected_surface(mesh, topo)

    nv = length(mesh.points)
    src = _canonical_sources(sources, nv)

    if length(src) == 1
        return _solve_geodesic_single_source(mesh, geom, dec, src[1]; timestep=timestep, factor_cache=factor_cache, fix_constant=fix_constant)
    end

    dmin = fill(T(Inf), nv)
    for s in src
        ds = _solve_geodesic_single_source(mesh, geom, dec, s; timestep=timestep, factor_cache=factor_cache, fix_constant=fix_constant)
        dmin = min.(dmin, ds)
    end
    return dmin
end

"""
    geodesic_distance_to_vertex(mesh, geom, dec, vid; kwargs...) -> Vector

Convenience wrapper for one source vertex.
"""
function geodesic_distance_to_vertex(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    dec::SurfaceDEC{T},
    vid::Int;
    kwargs...,
) where {T}
    return geodesic_distance(mesh, geom, dec, [vid]; kwargs...)
end

"""
    geodesic_distance_to_vertices(mesh, geom, dec, vids; kwargs...) -> Vector

Convenience wrapper for multiple source vertices.
"""
function geodesic_distance_to_vertices(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    dec::SurfaceDEC{T},
    vids;
    kwargs...,
) where {T}
    return geodesic_distance(mesh, geom, dec, vids; kwargs...)
end

"""
    geodesic_gradient(distance, mesh, geom, dec) -> Vector{SVector{3,T}}

Return per-face tangent vectors approximating `∇ distance`.
"""
function geodesic_gradient(
    distance::AbstractVector{T},
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    dec::SurfaceDEC{T},
) where {T}
    return gradient_0_to_tangent_vectors(mesh, geom, distance; location=:face)
end

"""
    shortest_path_vertices(mesh, geom, dec, src, dst; distance=nothing, method=:steepest_descent)
        -> Vector{Int}

Extract an approximate shortest vertex path between `src` and `dst`.

Method
------
- `:steepest_descent` (default): descend geodesic distance on the vertex graph
  from `dst` toward `src`, with deterministic Dijkstra fallback when descent
  stalls.
"""
function shortest_path_vertices(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    dec::SurfaceDEC{T},
    src::Int,
    dst::Int;
    distance=nothing,
    method::Symbol=:steepest_descent,
) where {T}
    method === :steepest_descent || throw(ArgumentError("Unsupported method=$(repr(method))."))

    src == dst && return Int[src]

    topo = build_topology(mesh)
    _check_connected_surface(mesh, topo)
    nv = length(mesh.points)
    (1 <= src <= nv && 1 <= dst <= nv) || throw(BoundsError("Vertex index out of range."))

    dsrc = distance === nothing ? geodesic_distance_to_vertex(mesh, geom, dec, src) : Vector{T}(distance)
    length(dsrc) == nv || throw(DimensionMismatch("distance length $(length(dsrc)) != nv=$nv"))

    adj = _vertex_adjacency_with_edges(topo, nv)

    rev_path = Int[dst]
    current = dst
    seen = Set{Int}([dst])

    for _ in 1:nv
        current == src && break
        cand = adj[current]
        isempty(cand) && break

        best = 0
        best_d = dsrc[current]

        # Strict descending step first.
        for (nbr, _) in cand
            dn = dsrc[nbr]
            if dn + T(1e-12) < best_d
                if best == 0 || dn < best_d || (dn == best_d && nbr < best)
                    best = nbr
                    best_d = dn
                end
            end
        end

        # If no strict descent exists, stop and fallback.
        if best == 0 || best in seen
            return _dijkstra_path(mesh, geom, topo, src, dst)
        end

        push!(rev_path, best)
        push!(seen, best)
        current = best
    end

    current == src || return _dijkstra_path(mesh, geom, topo, src, dst)

    reverse!(rev_path)
    return rev_path
end

"""
    shortest_path_points(mesh, geom, dec, src_point, dst_point;
                         distance=nothing, method=:steepest_descent) -> Vector

Extract an approximate intrinsic shortest path as 3D points.

Current behavior
----------------
`src_point` and `dst_point` are snapped to nearest mesh vertices, then
`shortest_path_vertices` is used on the snapped pair. The returned vector
contains mesh vertex coordinates along that path.
"""
function shortest_path_points(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    dec::SurfaceDEC{T},
    src_point,
    dst_point;
    distance=nothing,
    method::Symbol=:steepest_descent,
) where {T}
    src = _nearest_vertex_id(mesh, src_point)
    dst = _nearest_vertex_id(mesh, dst_point)
    vpath = shortest_path_vertices(
        mesh,
        geom,
        dec,
        src,
        dst;
        distance=distance,
        method=method,
    )
    return [mesh.points[i] for i in vpath]
end

"""
    intrinsic_ball(mesh, geom, dec, source, radius) -> Vector{Int}

Return vertex IDs whose geodesic distance from `source` is at most `radius`.
"""
function intrinsic_ball(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    dec::SurfaceDEC{T},
    source::Int,
    radius::Real,
) where {T}
    radius >= 0 || throw(ArgumentError("radius must be nonnegative."))
    d = geodesic_distance_to_vertex(mesh, geom, dec, source)
    r = T(radius)
    return [i for i in eachindex(d) if d[i] <= r]
end

"""
    farthest_point_sampling_geodesic(mesh, geom, dec, k; seed=1) -> Vector{Int}

Geodesic farthest-point sampling on vertices.
"""
function farthest_point_sampling_geodesic(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    dec::SurfaceDEC{T},
    k::Int;
    seed::Int=1,
) where {T}
    nv = length(mesh.points)
    k >= 1 || throw(ArgumentError("k must be at least 1."))
    k <= nv || throw(ArgumentError("k=$k exceeds number of vertices nv=$nv."))
    1 <= seed <= nv || throw(BoundsError("seed must lie in 1:$nv"))

    samples = Int[seed]
    dmin = geodesic_distance_to_vertex(mesh, geom, dec, seed)

    while length(samples) < k
        nxt = argmax(dmin)
        push!(samples, nxt)
        dnxt = geodesic_distance_to_vertex(mesh, geom, dec, nxt)
        dmin = min.(dmin, dnxt)
    end

    return samples
end
