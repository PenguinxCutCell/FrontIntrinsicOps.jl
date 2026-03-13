# convergence/torus_convergence.jl
#
# Convergence study on a standard torus.
#
# Exact formulas (standard torus with major radius R, minor radius r):
#   Area:   A = 4 * pi^2 * R * r
#   Volume: V = 2 * pi^2 * R * r^2
#   K(theta) = cos(theta) / (r * (R + r * cos(theta)))
#   chi = 0  (torus)
#   int K dA = 2 * pi * chi = 0
#
# Mesh size: h = sqrt(A / NF)
#
# Usage:
#   julia --project=. convergence/torus_convergence.jl

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "fit.jl"))
include(joinpath(@__DIR__, "metrics.jl"))

print_header("Torus Convergence Study")

R = 3.0
r = 1.0
exact_area = 4 * π^2 * R * r
exact_vol  = 2 * π^2 * R * r^2

@printf("\n  Parameters: R = %.2f, r = %.2f\n", R, r)
@printf("  Exact area    = %.6f\n", exact_area)
@printf("  Exact volume  = %.6f\n\n", exact_vol)

resolutions = [(8,16), (12,24), (16,32), (24,48), (32,64)]

function run_torus_study(da_method::Symbol)
    print_sep("Torus — dual_area = $da_method")
    @printf("  %-12s  %-8s  %-10s  %-10s  %-10s  %-4s  %-10s  %-10s\n",
        "res", "h", "err_A", "err_V", "int_K", "chi", "GB_res", "dL_inf")
    println("  " * "-" ^ 90)

    hs      = Float64[]
    err_As  = Float64[]
    err_Vs  = Float64[]
    labels  = String[]

    for (nt, np) in resolutions
        mesh  = generate_torus(R, r, nt, np)
        geom  = compute_geometry(mesh; dual_area=da_method)
        dec   = build_dec(mesh, geom)

        h    = mesh_size_surface(mesh, geom)
        A    = measure(mesh, geom)
        V    = enclosed_measure(mesh)
        chi  = euler_characteristic(mesh)
        intK = integrated_gaussian_curvature(mesh, geom)
        gb   = abs(intK - 2π * chi)

        err_A = abs(A - exact_area)
        err_V = abs(V - exact_vol)

        rpt   = compare_laplace_methods(mesh, geom)
        dL    = rpt.norm_inf

        lbl   = "($(nt),$(np))"
        push!(hs,     h)
        push!(err_As, err_A)
        push!(err_Vs, err_V)
        push!(labels, lbl)

        @printf("  %-12s  %8.4e  %10.4e  %10.4e  %10.4e  %4d  %10.4e  %10.4e\n",
            lbl, h, err_A, err_V, intK, chi, gb, dL)
    end

    println()
    @printf("  %-14s  %-8s  %-8s\n", "pair", "OOC_A", "OOC_V")
    oa = pairwise_orders(hs, err_As)
    ov = pairwise_orders(hs, err_Vs)
    for i in 1:length(oa)
        @printf("  %-14s  %-8s  %-8s\n",
            "$(labels[i])->$(labels[i+1])",
            format_order(oa[i]), format_order(ov[i]))
    end

    println()
    @printf("  Fitted order — area: %s,  volume: %s\n",
        format_order(fitted_order(hs, err_As)),
        format_order(fitted_order(hs, err_Vs)))
    println()
end

run_torus_study(:barycentric)
run_torus_study(:mixed)

print_sep()
println("  Done: torus_convergence.jl")
