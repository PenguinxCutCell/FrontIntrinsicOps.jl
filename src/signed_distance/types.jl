# signed_distance/types.jl – Internal types and tolerances for ambient SDF queries.

struct AABBNode{N,T<:AbstractFloat}
    bmin  :: SVector{N,T}
    bmax  :: SVector{N,T}
    left  :: Int
    right :: Int
    first :: Int
    count :: Int
end

struct SignedDistanceCache{N,T<:AbstractFloat,M}
    mesh                       :: M
    leafsize                   :: Int
    primitive_order            :: Vector{Int}
    nodes                      :: Vector{AABBNode{N,T}}
    bbox_min                   :: Vector{SVector{N,T}}
    bbox_max                   :: Vector{SVector{N,T}}
    primitive_centroids        :: Vector{SVector{N,T}}
    face_normals               :: Vector{SVector{N,T}}
    edge_normals               :: Vector{SVector{N,T}}
    vertex_normals             :: Vector{SVector{N,T}}
    edge_map                   :: Any
    primitive_to_feature_data  :: Any
    closed                     :: Bool
end

struct ClosestPrimitiveResult{N,T<:AbstractFloat}
    sqdist          :: T
    primitive       :: Int
    closest         :: SVector{N,T}
    feature_kind    :: Symbol
    feature_id      :: Int
    signing_normal  :: SVector{N,T}
end

@inline _dist_feature_tol(::Type{T}, scale::T) where {T<:AbstractFloat} = T(64) * eps(T) * max(scale, one(T))
@inline _surface_tol(::Type{T}, scale::T) where {T<:AbstractFloat} = T(256) * eps(T) * max(scale, one(T))
@inline _sign_tol(::Type{T}, scale::T) where {T<:AbstractFloat} = T(256) * eps(T) * max(scale, one(T))

@inline _rot90cw(v::SVector{2,T}) where {T} = SVector{2,T}(v[2], -v[1])

function _normalize_with_fallback(v::SVector{N,T}, fallback::SVector{N,T}) where {N,T<:AbstractFloat}
    nv = norm(v)
    if nv > eps(T)
        return v / nv
    end
    nf = norm(fallback)
    return nf > eps(T) ? fallback / nf : zero(SVector{N,T})
end
