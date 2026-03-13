# convergence/fit.jl – Convergence order estimation utilities.

"""
    pairwise_orders(hs, errs) -> Vector{Float64}

Compute pairwise observed orders of convergence (OOC) from arrays of mesh
sizes `hs` and corresponding errors `errs`.

For consecutive pairs (h_i, h_{i+1}):
    p_i = log(e_i / e_{i+1}) / log(h_i / h_{i+1})

Returns a vector of length length(hs) - 1 (NaN if an error is zero or negative).
"""
function pairwise_orders(hs, errs)
    n  = length(hs)
    n == length(errs) || error("hs and errs must have the same length")
    ps = fill(NaN, n - 1)
    for i in 1:(n-1)
        e1, e2 = errs[i], errs[i+1]
        h1, h2 = hs[i],   hs[i+1]
        if e1 > 0 && e2 > 0 && h1 > 0 && h2 > 0 && e1 != e2
            ps[i] = log(e1 / e2) / log(h1 / h2)
        end
    end
    return ps
end

"""
    fitted_order(hs, errs) -> Float64

Global fitted convergence order by least-squares regression of
    log(e) = p * log(h) + const
on all positive error values.  Returns NaN if fewer than 2 valid points.
"""
function fitted_order(hs, errs)
    valid = findall(e -> e > 0, errs)
    length(valid) >= 2 || return NaN
    log_h = log.(hs[valid])
    log_e = log.(errs[valid])
    # Simple linear regression: p = cov(log_h, log_e) / var(log_h)
    mh = mean(log_h)
    me = mean(log_e)
    num = sum((log_h .- mh) .* (log_e .- me))
    den = sum((log_h .- mh).^2)
    return den > 0 ? num / den : NaN
end

"""
    format_order(p) -> String

Format an order of convergence for terminal printing.
"""
format_order(p::Float64) = isnan(p) ? "  ---" : @sprintf("%+5.2f", p)
