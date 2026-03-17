# signed_distance/query.jl – Core query routines (scalar + batched).

@inline function _feature_normal(cache::SignedDistanceCache{2,T}, kind::Symbol, id::Int) where {T<:AbstractFloat}
    if kind === :edge
        return cache.face_normals[id]
    elseif kind === :vertex
        return cache.vertex_normals[id]
    end
    return zero(SVector{2,T})
end

@inline function _feature_normal(cache::SignedDistanceCache{3,T}, kind::Symbol, id::Int) where {T<:AbstractFloat}
    if kind === :face
        return cache.face_normals[id]
    elseif kind === :edge
        return cache.edge_normals[id]
    elseif kind === :vertex
        return cache.vertex_normals[id]
    end
    return zero(SVector{3,T})
end

function _normalize_sign_mode(cache::SignedDistanceCache, sign_mode::Symbol)
    sign_mode in (:auto, :pseudonormal, :winding, :unsigned) || throw(ArgumentError("Invalid sign_mode=$sign_mode. Use :auto, :pseudonormal, :winding, or :unsigned."))
    if sign_mode === :auto
        return cache.closed ? :winding : :pseudonormal
    end
    if sign_mode === :winding && !cache.closed
        throw(ArgumentError("sign_mode=:winding requires a closed oriented mesh. For open meshes inside/outside is not globally defined; use :pseudonormal or :unsigned."))
    end
    return sign_mode
end

function _sign_from_pseudonormal(
    q::SVector{N,T},
    cp::SVector{N,T},
    nsign::SVector{N,T},
) where {N,T<:AbstractFloat}
    dvec = q - cp
    d = norm(dvec)
    d <= _surface_tol(T, d + one(T)) && return zero(T)

    nuse = normalize_safe(nsign)
    σ = dot(dvec, nuse)
    stol = _sign_tol(T, d + one(T))
    if σ > stol
        return one(T)
    elseif σ < -stol
        return -one(T)
    end
    return zero(T)
end

function _query_one(
    q::SVector{N,T},
    cache::SignedDistanceCache{N,T};
    sign_mode::Symbol=:auto,
    lower_bound::T=zero(T),
    upper_bound::T=T(Inf),
    return_normals::Bool=true,
) where {N,T<:AbstractFloat}
    mode = _normalize_sign_mode(cache, sign_mode)
    sqd, pid, cp, fkind, fid = _closest_primitive(cache, q; lower_bound=lower_bound, upper_bound=upper_bound)
    d = sqrt(sqd)
    s = d
    nsign = zero(SVector{N,T})

    if mode === :unsigned
        s = d
    elseif mode === :pseudonormal
        nsign = _feature_normal(cache, fkind, fid)
        sgn = _sign_from_pseudonormal(q, cp, nsign)
        s = sgn == zero(T) ? zero(T) : sgn * d
    else
        tol = _surface_tol(T, d + one(T))
        if d <= tol
            s = zero(T)
        elseif N == 2
            wn = _winding_number_curve(q, cache.mesh)
            s = wn == 0 ? d : -d
        else
            wn = _winding_number_surface(q, cache.mesh)
            s = wn > T(0.5) ? -d : d
        end
    end

    return ClosestPrimitiveResult{N,T}(sqd, pid, cp, fkind, fid, return_normals ? nsign : zero(SVector{N,T})), s
end

function _points_to_svectors(points::AbstractVector{SVector{N,T}}) where {N,T<:AbstractFloat}
    return points
end

function _points_to_svectors(points::AbstractMatrix{T}, ::Val{N}) where {N,T<:AbstractFloat}
    r, c = size(points)
    if r == N
        return [SVector{N,T}(points[:, j]) for j in 1:c]
    elseif c == N
        return [SVector{N,T}(points[j, :]) for j in 1:r]
    else
        throw(ArgumentError("Matrix query points must have size (N,np) or (np,N), got $(size(points))."))
    end
end

function _signed_distance_batch(
    points::AbstractVector{SVector{N,T}},
    cache::SignedDistanceCache{N,T};
    sign_mode::Symbol=:auto,
    lower_bound::T=zero(T),
    upper_bound::T=T(Inf),
    return_normals::Bool=true,
) where {N,T<:AbstractFloat}
    np = length(points)
    S = Vector{T}(undef, np)
    I = Vector{Int}(undef, np)
    C = Vector{SVector{N,T}}(undef, np)
    Nout = Vector{SVector{N,T}}(undef, np)

    for i in 1:np
        q = points[i]
        result, sd = _query_one(q, cache; sign_mode=sign_mode, lower_bound=lower_bound, upper_bound=upper_bound, return_normals=return_normals)
        S[i] = sd
        I[i] = result.primitive
        C[i] = result.closest
        Nout[i] = result.signing_normal
    end

    return S, I, C, Nout
end
