# convergence/metrics.jl – Error norm helpers for convergence studies.

"""
    weighted_L1(u, u_exact, weights) -> scalar

Weighted L1 norm: sum(|u - u_exact| .* weights).
"""
function weighted_L1(u, u_exact, weights)
    return sum(abs.(u .- u_exact) .* weights)
end

"""
    weighted_L2(u, u_exact, weights) -> scalar

Weighted L2 norm: sqrt(sum(|u - u_exact|^2 .* weights)).
"""
function weighted_L2(u, u_exact, weights)
    return sqrt(sum((u .- u_exact).^2 .* weights))
end

"""
    Linf(u, u_exact) -> scalar

Max-norm (L-infinity): maximum absolute pointwise error.
"""
Linf(u, u_exact) = maximum(abs.(u .- u_exact))

"""
    curvature_error(K_discrete, K_exact, dual_areas; norm=:L2) -> scalar

Compute a curvature error norm with dual-area weighting.
`K_exact` can be a scalar (uniform) or a vector matching `K_discrete`.
"""
function curvature_error(K_discrete, K_exact, dual_areas; norm=:L2)
    K_ex = K_exact isa AbstractVector ? K_exact : fill(K_exact, length(K_discrete))
    if norm === :L2
        return weighted_L2(K_discrete, K_ex, dual_areas)
    elseif norm === :L1
        return weighted_L1(K_discrete, K_ex, dual_areas)
    elseif norm === :Linf
        return Linf(K_discrete, K_ex)
    else
        error("Unknown norm $(repr(norm)); use :L1, :L2, or :Linf")
    end
end
