# signed_distance/aabb.jl – AABB tree build and nearest-neighbour traversal.

@inline function _sqdist_point_aabb(q::SVector{N,T}, bmin::SVector{N,T}, bmax::SVector{N,T}) where {N,T<:AbstractFloat}
    d2 = zero(T)
    @inbounds for k in 1:N
        if q[k] < bmin[k]
            d = bmin[k] - q[k]
            d2 += d*d
        elseif q[k] > bmax[k]
            d = q[k] - bmax[k]
            d2 += d*d
        end
    end
    return d2
end

function _compute_node_bbox(
    order::Vector{Int},
    first::Int,
    count::Int,
    bbox_min::Vector{SVector{N,T}},
    bbox_max::Vector{SVector{N,T}},
) where {N,T<:AbstractFloat}
    i0 = order[first]
    bmin = bbox_min[i0]
    bmax = bbox_max[i0]
    for i in (first + 1):(first + count - 1)
        pid = order[i]
        bmin = min.(bmin, bbox_min[pid])
        bmax = max.(bmax, bbox_max[pid])
    end
    return bmin, bmax
end

function _build_aabb_tree(
    bbox_min::Vector{SVector{N,T}},
    bbox_max::Vector{SVector{N,T}},
    centroids::Vector{SVector{N,T}};
    leafsize::Int,
) where {N,T<:AbstractFloat}
    np = length(bbox_min)
    np == length(bbox_max) == length(centroids) || throw(ArgumentError("Inconsistent primitive arrays while building AABB tree."))
    np > 0 || throw(ArgumentError("Cannot build signed-distance cache on empty primitive set."))
    leafsize >= 1 || throw(ArgumentError("leafsize must be >= 1."))

    order = collect(1:np)
    nodes = AABBNode{N,T}[]

    function build_node(first::Int, count::Int)
        bmin, bmax = _compute_node_bbox(order, first, count, bbox_min, bbox_max)
        node_index = length(nodes) + 1
        push!(nodes, AABBNode{N,T}(bmin, bmax, 0, 0, first, count))

        if count <= leafsize
            return node_index
        end

        ext = bmax - bmin
        axis = argmax(ext)
        sub = view(order, first:(first + count - 1))
        sort!(sub, by=i -> (centroids[i][axis], i))
        left_count = count ÷ 2
        right_count = count - left_count

        li = build_node(first, left_count)
        ri = build_node(first + left_count, right_count)
        nodes[node_index] = AABBNode{N,T}(bmin, bmax, li, ri, first, count)
        return node_index
    end

    build_node(1, np)
    return nodes, order
end

function _closest_primitive(
    cache::SignedDistanceCache{N,T,M},
    q::SVector{N,T};
    lower_bound::T=zero(T),
    upper_bound::T=T(Inf),
) where {N,T<:AbstractFloat,M}
    best_sqd = isfinite(upper_bound) ? upper_bound^2 : T(Inf)
    lower2 = lower_bound^2
    best_id = typemax(Int)
    best_c = zero(SVector{N,T})
    best_kind = :none
    best_fid = 0

    stack = Int[1]
    while !isempty(stack)
        ni = pop!(stack)
        node = cache.nodes[ni]
        dbox = _sqdist_point_aabb(q, node.bmin, node.bmax)
        dbox > best_sqd && continue

        if node.left == 0
            first = node.first
            last = first + node.count - 1
            for ii in first:last
                pid = cache.primitive_order[ii]
                sqd, cp, kind, fid = if N == 2
                    _closest_curve_primitive(cache, q, pid)
                else
                    _closest_surface_primitive(cache, q, pid)
                end
                if (sqd < best_sqd) || (sqd == best_sqd && pid < best_id)
                    best_sqd = sqd
                    best_id = pid
                    best_c = cp
                    best_kind = kind
                    best_fid = fid
                end
            end

            if best_sqd < lower2
                break
            end
        else
            left = cache.nodes[node.left]
            right = cache.nodes[node.right]
            dl = _sqdist_point_aabb(q, left.bmin, left.bmax)
            dr = _sqdist_point_aabb(q, right.bmin, right.bmax)
            if dl <= dr
                dr <= best_sqd && push!(stack, node.right)
                dl <= best_sqd && push!(stack, node.left)
            else
                dl <= best_sqd && push!(stack, node.left)
                dr <= best_sqd && push!(stack, node.right)
            end
        end
    end

    best_id == typemax(Int) && throw(ArgumentError("No primitive found within provided upper_bound."))
    return best_sqd, best_id, best_c, best_kind, best_fid
end
