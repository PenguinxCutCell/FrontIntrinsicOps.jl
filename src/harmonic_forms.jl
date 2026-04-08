# harmonic_forms.jl – Topology-aware cohomology and harmonic 1-form tools.
#
# Closed-surface target (v1)
# -------------------------
# The routines in this file focus on closed orientable triangulated surfaces.
# Open-surface support can be added later.

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

function _cache_get_or_build!(builder, cache, key)
    if cache === nothing
        return builder()
    end
    if haskey(cache, key)
        return cache[key]
    end
    val = builder()
    cache[key] = val
    return val
end

function _inv_star0(dec::SurfaceDEC{T}; cache=nothing) where {T}
    key = (:inv_star0, objectid(dec.star0))
    return _cache_get_or_build!(cache, key) do
        d = diag(dec.star0)
        invd = [x > eps(T) ? one(T) / x : zero(T) for x in d]
        spdiagm(0 => invd)
    end
end

function _inv_star1(dec::SurfaceDEC{T}; cache=nothing) where {T}
    key = (:inv_star1, objectid(dec.star1))
    return _cache_get_or_build!(cache, key) do
        # Cotan stars can be negative and may contain exact zeros.
        # Use a pseudo-inverse: keep signed reciprocals above a scale-aware
        # threshold and set near-null entries to zero.
        w = diag(dec.star1)
        dmax = isempty(w) ? one(T) : maximum(abs.(w))
        thresh = max(T(1e-14) * max(one(T), dmax), eps(T))
        invd = Vector{T}(undef, length(w))
        @inbounds for i in eachindex(w)
            wi = w[i]
            if abs(wi) > thresh
                invd[i] = one(T) / wi
            else
                invd[i] = zero(T)
            end
        end
        spdiagm(0 => invd)
    end
end

function _inv_star2(dec::SurfaceDEC{T}; cache=nothing) where {T}
    key = (:inv_star2, objectid(dec.star2))
    return _cache_get_or_build!(cache, key) do
        d = diag(dec.star2)
        invd = [x > eps(T) ? one(T) / x : zero(T) for x in d]
        spdiagm(0 => invd)
    end
end

function _weighted_mean(vals::AbstractVector{T}, weights::AbstractVector{T}) where {T}
    denom = sum(weights)
    denom <= eps(T) && return zero(T)
    return dot(vals, weights) / denom
end

function _enforce_weighted_zero_mean!(vals::AbstractVector{T}, weights::AbstractVector{T}) where {T}
    vals .-= _weighted_mean(vals, weights)
    return vals
end

function _codifferential_1_from_dec(dec::SurfaceDEC{T}; cache=nothing) where {T}
    key = (:codiff1_from_dec, objectid(dec.d0), objectid(dec.star0), objectid(dec.star1))
    return _cache_get_or_build!(cache, key) do
        _inv_star0(dec; cache=cache) * dec.d0' * dec.star1
    end
end

function _codifferential_2_from_dec(dec::SurfaceDEC{T}; cache=nothing) where {T}
    key = (:codiff2_from_dec, objectid(dec.d1), objectid(dec.star1), objectid(dec.star2))
    return _cache_get_or_build!(cache, key) do
        _inv_star1(dec; cache=cache) * dec.d1' * dec.star2
    end
end

function _dual_laplacian_2(dec::SurfaceDEC{T}; cache=nothing) where {T}
    key = (:dual_lap2, objectid(dec.d1), objectid(dec.star1), objectid(dec.star2))
    return _cache_get_or_build!(cache, key) do
        # 2-form Hodge Laplacian on surfaces: Δ₂ = d δ₂ = d ⋆₁⁻¹ dᵀ ⋆₂.
        dec.d1 * _inv_star1(dec; cache=cache) * dec.d1' * dec.star2
    end
end

function _require_closed_oriented_surface(mesh::SurfaceMesh)
    is_closed(mesh) || throw(ArgumentError("This routine currently supports closed surfaces only."))
    has_consistent_orientation(mesh) || throw(ArgumentError("Surface faces must have consistent orientation."))
    return nothing
end

function _vertex_components(mesh::SurfaceMesh, topo::MeshTopology)
    nv = length(mesh.points)
    adj = [Int[] for _ in 1:nv]
    for e in topo.edges
        i, j = e[1], e[2]
        push!(adj[i], j)
        push!(adj[j], i)
    end
    for a in adj
        sort!(a)
    end

    comp = fill(0, nv)
    ncomp = 0
    queue = Int[]

    for root in 1:nv
        comp[root] != 0 && continue
        ncomp += 1
        comp[root] = ncomp
        empty!(queue)
        push!(queue, root)
        head = 1
        while head <= length(queue)
            v = queue[head]
            head += 1
            for w in adj[v]
                if comp[w] == 0
                    comp[w] = ncomp
                    push!(queue, w)
                end
            end
        end
    end
    return comp, ncomp
end

function _primal_spanning_forest(mesh::SurfaceMesh, topo::MeshTopology)
    nv = length(mesh.points)
    ne = length(topo.edges)

    adj = [Vector{Tuple{Int,Int}}() for _ in 1:nv]
    for (ei, e) in enumerate(topo.edges)
        i, j = e[1], e[2]
        push!(adj[i], (j, ei))
        push!(adj[j], (i, ei))
    end
    for a in adj
        sort!(a, by = x -> (x[1], x[2]))
    end

    parent = fill(0, nv)
    parent_edge = fill(0, nv)
    depth = fill(0, nv)
    component = fill(0, nv)
    tree_edge = falses(ne)

    queue = Int[]
    cid = 0
    for root in 1:nv
        component[root] != 0 && continue
        cid += 1
        component[root] = cid
        parent[root] = 0
        depth[root] = 0

        empty!(queue)
        push!(queue, root)
        head = 1
        while head <= length(queue)
            v = queue[head]
            head += 1
            for (w, ei) in adj[v]
                if component[w] == 0
                    component[w] = cid
                    parent[w] = v
                    parent_edge[w] = ei
                    depth[w] = depth[v] + 1
                    tree_edge[ei] = true
                    push!(queue, w)
                end
            end
        end
    end

    return (
        parent = parent,
        parent_edge = parent_edge,
        depth = depth,
        component = component,
        tree_edge = tree_edge,
    )
end

function _dual_cotree_edges(topo::MeshTopology, non_tree_edges::Vector{Int})
    nf = length(topo.face_edges)
    dual_adj = [Vector{Tuple{Int,Int}}() for _ in 1:nf] # (nbr_face, primal_edge)

    for ei in non_tree_edges
        ef = topo.edge_faces[ei]
        length(ef) == 2 || continue
        f1, f2 = ef[1], ef[2]
        push!(dual_adj[f1], (f2, ei))
        push!(dual_adj[f2], (f1, ei))
    end
    for a in dual_adj
        sort!(a, by = x -> (x[1], x[2]))
    end

    dual_comp = fill(0, nf)
    cotree_edge = falses(length(topo.edges))
    queue = Int[]
    cid = 0

    for root in 1:nf
        dual_comp[root] != 0 && continue
        cid += 1
        dual_comp[root] = cid
        empty!(queue)
        push!(queue, root)
        head = 1
        while head <= length(queue)
            f = queue[head]
            head += 1
            for (g, ei) in dual_adj[f]
                if dual_comp[g] == 0
                    dual_comp[g] = cid
                    cotree_edge[ei] = true
                    push!(queue, g)
                end
            end
        end
    end

    return cotree_edge
end

function _vertex_path(parent::Vector{Int}, from::Int, to::Int)
    from_path = Int[]
    cur = from
    while cur != 0
        push!(from_path, cur)
        cur = parent[cur]
    end

    to_path = Int[]
    cur = to
    while cur != 0
        push!(to_path, cur)
        cur = parent[cur]
    end

    from_index = Dict{Int,Int}()
    for (k, v) in enumerate(from_path)
        from_index[v] = k
    end

    lca = 0
    lca_to_idx = 0
    for (k, v) in enumerate(to_path)
        if haskey(from_index, v)
            lca = v
            lca_to_idx = k
            break
        end
    end
    lca != 0 || throw(ArgumentError("Vertices are not connected in the primal spanning forest."))

    path = Int[]
    append!(path, from_path[1:from_index[lca]])
    if lca_to_idx > 1
        append!(path, reverse(to_path[1:lca_to_idx-1]))
    end
    return path
end

function _signed_edge_along(u::Int, v::Int, edge_lookup::Dict{Tuple{Int,Int},Int})
    a, b = min(u, v), max(u, v)
    ei = edge_lookup[(a, b)]
    return u < v ? ei : -ei
end

function _cycle_to_cochain(cycle::Vector{Int}, ne::Int, ::Type{T}) where {T}
    c = zeros(T, ne)
    for s in cycle
        if s > 0
            c[s] += one(T)
        else
            c[-s] -= one(T)
        end
    end
    return c
end

function _fix_vector_sign!(v::AbstractVector{T}; atol::Real=1e-14) where {T}
    idx = 0
    vmax = zero(T)
    for i in eachindex(v)
        a = abs(v[i])
        if a > vmax
            vmax = a
            idx = i
        end
    end
    if idx > 0 && vmax > T(atol) && v[idx] < 0
        v .*= -one(T)
    end
    return v
end

function _hodge_gram(B::AbstractMatrix{T}, dec::SurfaceDEC{T}) where {T}
    return B' * (dec.star1 * B)
end

function _orthonormalize_hodge(B::AbstractMatrix{T}, dec::SurfaceDEC{T}; atol::Real=1e-10, rtol::Real=1e-8) where {T}
    ne, m = size(B)
    if m == 0
        return zeros(T, ne, 0)
    end

    Qcols = Vector{Vector{T}}()
    max_seen = zero(T)

    for j in 1:m
        v = copy(view(B, :, j))

        for q in Qcols
            coeff = dot(q, dec.star1 * v)
            v .-= coeff .* q
        end

        n2 = dot(v, dec.star1 * v)
        n2 = n2 < zero(T) ? zero(T) : n2
        nv = sqrt(n2)
        max_seen = max(max_seen, nv)
        thresh = max(T(atol), T(rtol) * max(one(T), max_seen))
        if nv > thresh
            v ./= nv
            _fix_vector_sign!(v)
            push!(Qcols, v)
        end
    end

    qn = length(Qcols)
    Q = zeros(T, ne, qn)
    for j in 1:qn
        Q[:, j] = Qcols[j]
    end
    return Q
end

function _exact_component_impl(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    dec::SurfaceDEC{T},
    ω1::AbstractVector{T};
    gauge::Symbol=:mean_zero,
    reg::Real=1e-10,
    factor_cache=nothing,
) where {T}
    gauge === :mean_zero || throw(ArgumentError("Unsupported gauge=$(repr(gauge)); only :mean_zero is implemented."))

    δ1 = _codifferential_1_from_dec(dec; cache=factor_cache)
    rhs = Vector{T}(δ1 * ω1)

    m0 = diag(dec.star0)
    _enforce_weighted_zero_mean!(rhs, m0)

    ε = T(reg)
    key = (:exact_factor, objectid(dec.lap0), objectid(dec.star0), ε)
    fac = _cache_get_or_build!(factor_cache, key) do
        factorize(dec.lap0 + ε * dec.star0)
    end

    α = fac \ rhs
    _enforce_weighted_zero_mean!(α, m0)

    de = Vector{T}(dec.d0 * α)
    return de, α
end

function _coexact_component_dual_potential_impl(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    dec::SurfaceDEC{T},
    ω1::AbstractVector{T};
    gauge::Symbol=:mean_zero,
    reg::Real=1e-10,
    factor_cache=nothing,
) where {T}
    gauge === :mean_zero || throw(ArgumentError("Unsupported gauge=$(repr(gauge)); only :mean_zero is implemented."))

    rhs = Vector{T}(dec.d1 * ω1)
    m2 = diag(dec.star2)
    _enforce_weighted_zero_mean!(rhs, m2)

    Δ2 = _dual_laplacian_2(dec; cache=factor_cache)
    ε = T(reg)

    key = (:coexact_factor, objectid(dec.d1), objectid(dec.star1), objectid(dec.star2), ε)
    fac = _cache_get_or_build!(factor_cache, key) do
        factorize(Δ2 + ε * dec.star2)
    end

    β = fac \ rhs
    # One defect-correction sweep tightens the dual solve on ill-conditioned
    # cotan stars while reusing the same factorization.
    defect = rhs .- (Δ2 * β)
    if norm(defect) > sqrt(eps(T)) * (norm(rhs) + one(T))
        β .+= fac \ defect
    end
    _enforce_weighted_zero_mean!(β, m2)

    δ2 = _codifferential_2_from_dec(dec; cache=factor_cache)
    δβ = Vector{T}(δ2 * β)
    return δβ, β
end

function _coexact_component_impl(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    dec::SurfaceDEC{T},
    ω1::AbstractVector{T};
    gauge::Symbol=:mean_zero,
    reg::Real=1e-10,
    factor_cache=nothing,
) where {T}
    return _coexact_component_dual_potential_impl(
        mesh,
        geom,
        dec,
        ω1;
        gauge=gauge,
        reg=reg,
        factor_cache=factor_cache,
    )
end

# -----------------------------------------------------------------------------
# Topology layer
# -----------------------------------------------------------------------------

"""
    betti_numbers(mesh[, topo]) -> NamedTuple

Return `(β0, β1, β2)` for a triangulated surface mesh.

Conventions
-----------
- `β0` is the number of connected components.
- On closed orientable surfaces, `β2 = β0` and
  `β1 = β0 + β2 - χ = 2β0 - χ`, with `χ = V - E + F`.
- On orientable surfaces with boundary, this implementation uses `β2 = 0` and
  `β1 = β0 - χ`.
"""
function betti_numbers(mesh::SurfaceMesh, topo::MeshTopology=build_topology(mesh))
    _, β0 = _vertex_components(mesh, topo)
    χ = length(mesh.points) - length(topo.edges) + length(mesh.faces)

    if is_closed(mesh)
        β2 = β0
        β1 = β0 + β2 - χ
    else
        β2 = 0
        β1 = β0 - χ
    end

    return (β0=β0, β1=β1, β2=β2)
end

"""
    first_betti_number(mesh[, topo]) -> Int

Return the first Betti number `β1` of the surface mesh.
"""
function first_betti_number(mesh::SurfaceMesh, topo::MeshTopology=build_topology(mesh))
    return betti_numbers(mesh, topo).β1
end

"""
    cycle_basis(mesh[, topo]) -> Vector{Vector{Int}}

Construct a deterministic combinatorial generator set for `H1` using a
primal spanning tree / dual spanning cotree decomposition.

Each returned cycle is encoded as a vector of signed edge indices:
- `+e` means traversing edge `e` in canonical orientation `(i<j)`.
- `-e` means traversing edge `e` opposite to canonical orientation.
"""
function cycle_basis(mesh::SurfaceMesh, topo::MeshTopology=build_topology(mesh))
    _require_closed_oriented_surface(mesh)

    ne = length(topo.edges)

    primal = _primal_spanning_forest(mesh, topo)
    non_tree_edges = [ei for ei in 1:ne if !primal.tree_edge[ei]]
    cotree_edge = _dual_cotree_edges(topo, non_tree_edges)

    generators = [ei for ei in non_tree_edges if !cotree_edge[ei]]

    edge_lookup = Dict{Tuple{Int,Int},Int}()
    for (ei, e) in enumerate(topo.edges)
        edge_lookup[(e[1], e[2])] = ei
    end

    cycles = Vector{Vector{Int}}()
    for ei in generators
        i, j = topo.edges[ei][1], topo.edges[ei][2]

        # Generator edge in canonical orientation i -> j.
        cyc = Int[ei]

        # Tree path closes the loop: j -> i.
        path_vertices = _vertex_path(primal.parent, j, i)
        for k in 1:(length(path_vertices)-1)
            u = path_vertices[k]
            v = path_vertices[k+1]
            push!(cyc, _signed_edge_along(u, v, edge_lookup))
        end

        push!(cycles, cyc)
    end

    return cycles
end

"""
    cohomology_basis_1(mesh, geom, dec; method=:cycle_poisson, reg=1e-10, factor_cache=nothing)
        -> Matrix

Build closed 1-form representatives (edge cochains) of independent cohomology
classes prior to harmonic projection.

Method
------
- `:cycle_poisson` (default): start from combinatorial cycle generators and
  remove coexact pollution by solving a dual Poisson projection.
"""
function cohomology_basis_1(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    dec::SurfaceDEC{T};
    method::Symbol=:cycle_poisson,
    reg::Real=1e-10,
    factor_cache=nothing,
) where {T}
    _require_closed_oriented_surface(mesh)
    method === :cycle_poisson || throw(ArgumentError("Unsupported method=$(repr(method))."))

    topo = build_topology(mesh)
    cycles = cycle_basis(mesh, topo)
    ne = length(topo.edges)

    if isempty(cycles)
        return zeros(T, ne, 0)
    end

    nf = length(mesh.faces)
    ε = T(reg)
    Af = dec.d1 * dec.d1' + ε * sparse(T(1) * LinearAlgebra.I(nf))
    key = (:cycle_closed_projection_factor, objectid(dec.d1), ε)
    fac = _cache_get_or_build!(factor_cache, key) do
        factorize(Af)
    end

    B = zeros(T, ne, length(cycles))
    for (j, cyc) in enumerate(cycles)
        c = _cycle_to_cochain(cyc, ne, T)
        # L2 projection onto ker(d1): rep = argmin ||x-c|| subject to d1*x = 0.
        λ = fac \ (dec.d1 * c)
        rep = c .- dec.d1' * λ
        _fix_vector_sign!(rep)
        B[:, j] = rep
    end

    return B
end

# -----------------------------------------------------------------------------
# Harmonic / projection API
# -----------------------------------------------------------------------------

"""
    is_closed_form(ω1, dec; atol=1e-8) -> Bool

Return `true` if a 1-form `ω1` is closed to tolerance, i.e. `‖dω1‖₂ <= atol`.
"""
function is_closed_form(ω1::AbstractVector{T}, dec::SurfaceDEC{T}; atol::Real=1e-8) where {T}
    return norm(dec.d1 * ω1) <= T(atol)
end

"""
    is_coclosed_form(ω1, dec; atol=1e-8) -> Bool

Return `true` if a 1-form `ω1` is coclosed to tolerance, i.e.
`‖δω1‖₂ <= atol` with `δ = ⋆₀⁻¹ d₀ᵀ ⋆₁`.
"""
function is_coclosed_form(ω1::AbstractVector{T}, dec::SurfaceDEC{T}; atol::Real=1e-8) where {T}
    δ1 = _codifferential_1_from_dec(dec)
    return norm(δ1 * ω1) <= T(atol)
end

"""
    harmonic_residuals(ω1, mesh, geom, dec) -> NamedTuple

Return diagnostic norms `(d_norm, δ_norm, d_inf, δ_inf)` for a 1-form.
"""
function harmonic_residuals(
    ω1::AbstractVector{T},
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    dec::SurfaceDEC{T},
) where {T}
    dω = dec.d1 * ω1
    δω = _codifferential_1_from_dec(dec) * ω1
    return (
        d_norm = norm(dω),
        δ_norm = norm(δω),
        d_inf = maximum(abs, dω),
        δ_inf = maximum(abs, δω),
    )
end

"""
    project_exact(ω1, mesh, geom, dec; gauge=:mean_zero, reg=1e-10, factor_cache=nothing) -> Vector

Project an edge 1-cochain onto the exact subspace `im(d0)`.
"""
function project_exact(
    ω1::AbstractVector{T},
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    dec::SurfaceDEC{T};
    gauge::Symbol=:mean_zero,
    reg::Real=1e-10,
    factor_cache=nothing,
) where {T}
    _require_closed_oriented_surface(mesh)
    de, _ = _exact_component_impl(mesh, geom, dec, ω1; gauge=gauge, reg=reg, factor_cache=factor_cache)
    return de
end

"""
    project_coexact(ω1, mesh, geom, dec; gauge=:mean_zero, reg=1e-10, factor_cache=nothing) -> Vector

Project an edge 1-cochain onto the coexact subspace `im(δ2)`.
"""
function project_coexact(
    ω1::AbstractVector{T},
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    dec::SurfaceDEC{T};
    gauge::Symbol=:mean_zero,
    reg::Real=1e-10,
    factor_cache=nothing,
) where {T}
    _require_closed_oriented_surface(mesh)
    δβ, _ = _coexact_component_impl(mesh, geom, dec, ω1; gauge=gauge, reg=reg, factor_cache=factor_cache)
    return δβ
end

"""
    harmonic_basis(mesh, geom, dec; normalize=:hodge, atol=1e-10, rtol=1e-8,
                   reg=1e-10, factor_cache=nothing) -> Matrix

Compute a deterministic basis of harmonic 1-forms (edge cochains) on a closed
oriented triangulated surface.

Construction
------------
1. Build cycle generators with `cycle_basis`.
2. Build closed cohomology representatives with `cohomology_basis_1`.
3. Harmonicize each representative via Hodge projection.
4. Orthonormalize with the Hodge inner product `⟨a,b⟩ = a'⋆1b`.
"""
function harmonic_basis(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    dec::SurfaceDEC{T};
    normalize::Symbol=:hodge,
    atol::Real=1e-10,
    rtol::Real=1e-8,
    reg::Real=1e-10,
    factor_cache=nothing,
) where {T}
    _require_closed_oriented_surface(mesh)

    topo = build_topology(mesh)
    ne = length(topo.edges)
    b1 = first_betti_number(mesh, topo)
    b1 < 0 && throw(ArgumentError("Computed negative β1=$b1; mesh/topology may be invalid."))
    b1 == 0 && return zeros(T, ne, 0)

    cycles = cycle_basis(mesh, topo)
    length(cycles) == b1 || @warn "cycle_basis produced $(length(cycles)) generators; expected β1=$b1."
    C = zeros(T, ne, length(cycles))
    for (j, cyc) in enumerate(cycles)
        C[:, j] = _cycle_to_cochain(cyc, ne, T)
    end

    δ1 = _codifferential_1_from_dec(dec; cache=factor_cache)
    K = dec.d1' * dec.d1 + δ1' * δ1
    ε = T(reg)
    Kreg = K + ε * sparse(T(1) * LinearAlgebra.I(ne))

    # Constrain harmonic representatives by cycle pairings in the Hodge metric.
    G = transpose(C) * dec.star1
    Gs = sparse(G)
    S = [Kreg transpose(Gs); Gs spzeros(T, size(Gs, 1), size(Gs, 1))]

    rhs = vcat(zeros(T, ne, size(Gs, 1)), Matrix{T}(I, size(Gs, 1), size(Gs, 1)))
    X = S \ rhs
    Hcand = X[1:ne, :]

    H = if normalize === :hodge
        _orthonormalize_hodge(Hcand, dec; atol=atol, rtol=rtol)
    elseif normalize === :none
        Hcand
    else
        throw(ArgumentError("Unsupported normalize=$(repr(normalize)). Use :hodge or :none."))
    end

    # Deterministic column sort by dominant edge index, then by sign at that edge.
    if size(H, 2) > 1
        keys = Vector{Tuple{Int,T}}(undef, size(H, 2))
        for j in 1:size(H, 2)
            col = view(H, :, j)
            idx = argmax(abs.(col))
            keys[j] = (idx, col[idx])
        end
        perm = sortperm(1:size(H, 2), by = j -> keys[j])
        H = H[:, perm]
    end

    # Keep only the expected topological dimension when numerical rank is larger.
    if size(H, 2) > b1
        H = H[:, 1:b1]
    end

    return H
end

"""
    project_harmonic(ω1, mesh, geom, dec; basis=nothing, reg=1e-10, factor_cache=nothing) -> Vector

Project a 1-form onto the harmonic subspace.

If `basis` is provided, projection is done in that span with the Hodge metric.
Otherwise, the harmonic component is obtained from a full decomposition solve.
"""
function project_harmonic(
    ω1::AbstractVector{T},
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    dec::SurfaceDEC{T};
    basis=nothing,
    reg::Real=1e-10,
    factor_cache=nothing,
) where {T}
    _require_closed_oriented_surface(mesh)

    if basis === nothing
        B = harmonic_basis(mesh, geom, dec; normalize=:hodge, reg=reg, factor_cache=factor_cache)
        return project_harmonic(ω1, mesh, geom, dec; basis=B, reg=reg, factor_cache=factor_cache)
    end

    B = Matrix{T}(basis)
    size(B, 1) == length(ω1) || throw(DimensionMismatch("basis has $(size(B,1)) rows, expected $(length(ω1))."))
    size(B, 2) == 0 && return zeros(T, length(ω1))

    G = _hodge_gram(B, dec)
    rhs = B' * (dec.star1 * ω1)
    coeffs = G \ rhs
    return B * coeffs
end

"""
    hodge_decomposition_full(ω1, mesh, geom, dec;
                             basis=nothing,
                             gauge=:mean_zero,
                             reg=1e-10,
                             factor_cache=nothing) -> NamedTuple

Compute the full Hodge decomposition

`ω = dα + δβ + h`

on a closed oriented triangulated surface, returning

`(exact, coexact, harmonic, residual, potentials=(α=..., β=...))`.

Gauge convention
----------------
`gauge=:mean_zero` enforces weighted zero-mean gauges for both potentials.
"""
function hodge_decomposition_full(
    ω1::AbstractVector{T},
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    dec::SurfaceDEC{T};
    basis=nothing,
    gauge::Symbol=:mean_zero,
    reg::Real=1e-10,
    factor_cache=nothing,
) where {T}
    _require_closed_oriented_surface(mesh)

    B = basis === nothing ? harmonic_basis(mesh, geom, dec; normalize=:hodge, reg=reg, factor_cache=factor_cache) :
                            Matrix{T}(basis)
    h = project_harmonic(ω1, mesh, geom, dec; basis=B, reg=reg, factor_cache=factor_cache)

    ω_work = ω1 .- h

    de, α = _exact_component_impl(mesh, geom, dec, ω_work; gauge=gauge, reg=reg, factor_cache=factor_cache)
    δβ, β = _coexact_component_impl(mesh, geom, dec, ω_work .- de; gauge=gauge, reg=reg, factor_cache=factor_cache)

    resid = ω1 .- de .- δβ .- h
    return (
        exact = de,
        coexact = δβ,
        harmonic = h,
        residual = resid,
        potentials = (α = α, β = β),
    )
end
