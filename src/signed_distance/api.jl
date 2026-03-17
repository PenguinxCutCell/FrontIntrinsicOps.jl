# signed_distance/api.jl – Public ambient signed-distance API.

"""
    build_signed_distance_cache(mesh; leafsize=8)

Build and return a `SignedDistanceCache` for exact nearest-primitive and
signed-distance queries on a `CurveMesh` or `SurfaceMesh`.
"""
function build_signed_distance_cache(mesh::CurveMesh{T}; leafsize::Int=8) where {T<:AbstractFloat}
    closed, edge_normals, vertex_normals = _prepare_curve_sign_data(mesh)
    ne = length(mesh.edges)

    bbox_min = Vector{SVector{2,T}}(undef, ne)
    bbox_max = Vector{SVector{2,T}}(undef, ne)
    centroids = Vector{SVector{2,T}}(undef, ne)
    for ei in 1:ne
        e = mesh.edges[ei]
        a = mesh.points[e[1]]
        b = mesh.points[e[2]]
        bbox_min[ei] = min.(a, b)
        bbox_max[ei] = max.(a, b)
        centroids[ei] = (a + b) / T(2)
    end

    nodes, order = _build_aabb_tree(bbox_min, bbox_max, centroids; leafsize=leafsize)

    return SignedDistanceCache{2,T,typeof(mesh)}(
        mesh,
        leafsize,
        order,
        nodes,
        bbox_min,
        bbox_max,
        centroids,
        edge_normals,
        SVector{2,T}[],
        vertex_normals,
        nothing,
        nothing,
        closed,
    )
end

function build_signed_distance_cache(mesh::SurfaceMesh{T}; leafsize::Int=8) where {T<:AbstractFloat}
    closed, face_normals, edge_normals, vertex_normals, feature_data = _prepare_surface_sign_data(mesh)
    nf = length(mesh.faces)

    bbox_min = Vector{SVector{3,T}}(undef, nf)
    bbox_max = Vector{SVector{3,T}}(undef, nf)
    centroids = Vector{SVector{3,T}}(undef, nf)
    for fi in 1:nf
        f = mesh.faces[fi]
        a = mesh.points[f[1]]
        b = mesh.points[f[2]]
        c = mesh.points[f[3]]
        bbox_min[fi] = min.(min.(a, b), c)
        bbox_max[fi] = max.(max.(a, b), c)
        centroids[fi] = (a + b + c) / T(3)
    end

    nodes, order = _build_aabb_tree(bbox_min, bbox_max, centroids; leafsize=leafsize)

    return SignedDistanceCache{3,T,typeof(mesh)}(
        mesh,
        leafsize,
        order,
        nodes,
        bbox_min,
        bbox_max,
        centroids,
        face_normals,
        edge_normals,
        vertex_normals,
        feature_data.edges,
        feature_data,
        closed,
    )
end

"""
    signed_distance(points, mesh_or_cache; sign_mode=:auto, lower_bound=0.0, upper_bound=Inf, return_normals=true)

Batched ambient exact signed-distance query.

Inputs:
- `points`: `Vector{SVector{N,T}}` or matrix of size `(N,np)` or `(np,N)`.
- `mesh_or_cache`: a `CurveMesh`/`SurfaceMesh` or prebuilt `SignedDistanceCache`.

Returns `(S, I, C, N)` where:
- `S`: signed distances,
- `I`: closest primitive ids,
- `C`: closest points,
- `N`: signing normals used by pseudonormal mode (zero in other modes).

Sign semantics:
- closed mesh + `:winding`: inside/outside sign,
- open mesh + `:pseudonormal`: oriented side-of-curve/sheet sign,
- open meshes do not define global inside/outside.

Notes:
- points on the front return `0` distance in `:pseudonormal` mode,
- near numerically ambiguous sign configurations, `:pseudonormal` may return
    zero sign when the signing dot product is within tolerance.
"""
function signed_distance(
    points::AbstractVector{SVector{N,T}},
    mesh_or_cache;
    sign_mode::Symbol=:auto,
    lower_bound::Real=0.0,
    upper_bound::Real=Inf,
    return_normals::Bool=true,
) where {N,T<:AbstractFloat}
    cache = mesh_or_cache isa SignedDistanceCache ? mesh_or_cache : build_signed_distance_cache(mesh_or_cache)
    cacheT = eltype(cache.mesh.points[1])
    pts = Vector{SVector{N,cacheT}}(undef, length(points))
    for i in eachindex(points)
        pts[i] = SVector{N,cacheT}(points[i])
    end
    lb = cacheT(lower_bound)
    ub = cacheT(upper_bound)
    return _signed_distance_batch(pts, cache; sign_mode=sign_mode, lower_bound=lb, upper_bound=ub, return_normals=return_normals)
end

function signed_distance(
    points::AbstractMatrix{T},
    mesh_or_cache;
    sign_mode::Symbol=:auto,
    lower_bound::Real=0.0,
    upper_bound::Real=Inf,
    return_normals::Bool=true,
) where {T<:AbstractFloat}
    cache = mesh_or_cache isa SignedDistanceCache ? mesh_or_cache : build_signed_distance_cache(mesh_or_cache)
    N = length(cache.mesh.points[1])
    cacheT = eltype(cache.mesh.points[1])
    pts_raw = _points_to_svectors(points, Val(N))
    pts = Vector{SVector{N,cacheT}}(undef, length(pts_raw))
    for i in eachindex(pts_raw)
        pts[i] = SVector{N,cacheT}(pts_raw[i])
    end
    lb = cacheT(lower_bound)
    ub = cacheT(upper_bound)
    return _signed_distance_batch(pts, cache; sign_mode=sign_mode, lower_bound=lb, upper_bound=ub, return_normals=return_normals)
end

"""
    signed_distance(point::SVector{N,T}, mesh_or_cache; sign_mode=:auto)

Scalar ambient signed-distance query.

Returns named tuple `(distance, primitive, closest, normal)`.
"""
function signed_distance(
    point::SVector{N,T},
    mesh_or_cache;
    sign_mode::Symbol=:auto,
    lower_bound::Real=0.0,
    upper_bound::Real=Inf,
    return_normals::Bool=true,
) where {N,T<:AbstractFloat}
    cache = mesh_or_cache isa SignedDistanceCache ? mesh_or_cache : build_signed_distance_cache(mesh_or_cache)
    Tp = eltype(cache.mesh.points[1])
    q = SVector{N,Tp}(point)
    result, s = _query_one(q, cache; sign_mode=sign_mode, lower_bound=Tp(lower_bound), upper_bound=Tp(upper_bound), return_normals=return_normals)
    return (distance=s, primitive=result.primitive, closest=result.closest, normal=result.signing_normal)
end

"""
    unsigned_distance(points, mesh_or_cache)

Convenience wrapper equivalent to `signed_distance(...; sign_mode=:unsigned)`.
"""
function unsigned_distance(points, mesh_or_cache)
    return signed_distance(points, mesh_or_cache; sign_mode=:unsigned)
end

"""
    winding_number(point, mesh_or_cache)

Compute winding number for closed meshes:
- 2D closed curves: integer winding number,
- 3D closed surfaces: normalized solid-angle winding number.
"""
function winding_number(point::SVector{2,T}, mesh_or_cache) where {T<:AbstractFloat}
    mesh = mesh_or_cache isa SignedDistanceCache ? mesh_or_cache.mesh : mesh_or_cache
    return _winding_number_curve(point, mesh)
end

function winding_number(point::SVector{3,T}, mesh_or_cache) where {T<:AbstractFloat}
    mesh = mesh_or_cache isa SignedDistanceCache ? mesh_or_cache.mesh : mesh_or_cache
    return _winding_number_surface(point, mesh)
end
