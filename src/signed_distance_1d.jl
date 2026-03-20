# signed_distance_1d.jl – Minimal signed-distance helpers for 1-D point fronts.

"""
    interface_normals(front::PointFront1D) -> Vector

Return outward normal orientation(s) of the inside region for a 1-D front.

Conventions:
- one marker:
  - inside right (`interval_is_inside=true`)  -> `[+1]`
  - inside left  (`interval_is_inside=false`) -> `[-1]`
- two markers `(xL, xR)`:
  - interval inside (`interval_is_inside=true`)  -> `[-1, +1]`
  - interval outside (`interval_is_inside=false`) -> `[+1, -1]`
"""
function interface_normals(front::PointFront1D{T}) where {T<:Real}
    U = float(T)
    p = one(U)
    m = -one(U)
    if length(front.x) == 1
        return front.interval_is_inside ? U[p] : U[m]
    end
    return front.interval_is_inside ? U[m, p] : U[p, m]
end

"""
    signed_distance(front::PointFront1D, x::Real)

Evaluate signed distance `φ(x)` to a minimal 1-D point front.

Sign convention:
- `φ < 0` inside
- `φ > 0` outside
- `φ = 0` on marker(s)
"""
function signed_distance(front::PointFront1D{T}, x::Real) where {T<:Real}
    U = promote_type(float(T), float(typeof(x)))
    xi = U(x)
    x1 = U(front.x[1])

    if length(front.x) == 1
        # one marker: inside right -> minus sign, inside left -> plus sign
        return front.interval_is_inside ? -(xi - x1) : (xi - x1)
    end

    xL = x1
    xR = U(front.x[2])
    if front.interval_is_inside
        if xi < xL
            return xL - xi
        elseif xi > xR
            return xi - xR
        end
        return -min(xi - xL, xR - xi)
    end

    if xi < xL
        return -(xL - xi)
    elseif xi > xR
        return -(xi - xR)
    end
    return min(xi - xL, xR - xi)
end

"""
    signed_distance(front::PointFront1D, xs::AbstractVector)

Vectorized signed-distance evaluation for 1-D point fronts.
"""
function signed_distance(front::PointFront1D, xs::AbstractVector{<:Real})
    n = length(xs)
    if n == 0
        return Vector{float(eltype(front.x))}(undef, 0)
    end

    i0 = firstindex(xs)
    U = typeof(signed_distance(front, xs[i0]))
    ϕ = Vector{U}(undef, n)
    k = 1
    @inbounds for i in eachindex(xs)
        ϕ[k] = signed_distance(front, xs[i])
        k += 1
    end
    return ϕ
end

"""
    rebuild_signed_distance(front::PointFront1D, xnodes::AbstractVector)

Rebuild signed-distance values on 1-D grid nodes `xnodes`.
"""
function rebuild_signed_distance(front::PointFront1D, xnodes::AbstractVector{<:Real})
    return signed_distance(front, xnodes)
end
