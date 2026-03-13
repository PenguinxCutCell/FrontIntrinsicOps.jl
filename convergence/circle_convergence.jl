# convergence/circle_convergence.jl
#
# Convergence study for a closed polygonal circle.
#
# Study:
#   R = 1.5, N = 16, 32, 64, 128, 256, 512, 1024
#
# Quantities measured:
#   - length error vs 2*pi*R
#   - enclosed area error vs pi*R^2
#   - mean curvature error vs 1/R  (avg of |kappa - 1/R|)
#   - constant nullspace residual: max|L*ones|
#
# Mesh size: h = 2*pi*R / N  (arc length per segment)
#
# Usage:
#   julia --project=. convergence/circle_convergence.jl

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "fit.jl"))
include(joinpath(@__DIR__, "metrics.jl"))

print_header("Circle Convergence Study")

R   = 1.5
Ns  = [16, 32, 64, 128, 256, 512, 1024]
exact_len  = 2π * R
exact_area = π * R^2
exact_kap  = 1.0 / R

@printf("\n  Parameters: R = %.2f\n", R)
@printf("  Exact length  = %.6f\n", exact_len)
@printf("  Exact area    = %.6f\n", exact_area)
@printf("  Exact kappa   = %.6f\n\n", exact_kap)

# Header
@printf("  %-6s  %-10s  %-12s  %-12s  %-12s  %-12s\n",
    "N", "h", "err_len", "err_area", "err_kappa", "L*ones")
print_sep()

hs         = Float64[]
err_lens   = Float64[]
err_areas  = Float64[]
err_kappas = Float64[]
err_null   = Float64[]

for N in Ns
    mesh = sample_circle(R, N)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    h        = 2π * R / N
    err_len  = abs(measure(mesh, geom) - exact_len)
    err_area = abs(enclosed_measure(mesh) - exact_area)

    kappa    = geom.signed_curvature
    err_kap  = mean(abs.(kappa .- exact_kap))

    nv      = length(mesh.points)
    null_r  = maximum(abs, dec.lap0 * ones(nv))

    push!(hs,         h)
    push!(err_lens,   err_len)
    push!(err_areas,  err_area)
    push!(err_kappas, err_kap)
    push!(err_null,   null_r)

    @printf("  %-6d  %10.4e  %12.4e  %12.4e  %12.4e  %12.4e\n",
        N, h, err_len, err_area, err_kap, null_r)
end

# Pairwise orders
print_sep("Pairwise observed orders of convergence")
@printf("  %-6s  %-8s  %-10s  %-10s  %-10s\n",
    "pair", "h_ratio", "OOC_len", "OOC_area", "OOC_kappa")
orders_len   = pairwise_orders(hs, err_lens)
orders_area  = pairwise_orders(hs, err_areas)
orders_kappa = pairwise_orders(hs, err_kappas)
for i in 1:length(orders_len)
    ratio = hs[i] / hs[i+1]
    @printf("  %d->%d   %8.2f  %10s  %10s  %10s\n",
        Ns[i], Ns[i+1], ratio,
        format_order(orders_len[i]),
        format_order(orders_area[i]),
        format_order(orders_kappa[i]))
end

# Global fitted orders
print_sep("Globally fitted orders (least-squares on log-log)")
@printf("  length:    %s\n", format_order(fitted_order(hs, err_lens)))
@printf("  area:      %s\n", format_order(fitted_order(hs, err_areas)))
@printf("  curvature: %s\n", format_order(fitted_order(hs, err_kappas)))

print_sep()
println("  Done: circle_convergence.jl")
