# convergence/sphere_convergence.jl
#
# Convergence study on sphere meshes (UV-sphere and icosphere).
#
# Mesh size: h = sqrt(total_area / NF)
#
# Quantities measured:
#   - area error vs 4*pi*R^2
#   - volume error vs (4/3)*pi*R^3
#   - mean curvature error vs H_exact = 1/R (sphere has H = 1/R for R=1 unit sphere)
#   - Gaussian curvature error vs K_exact = 1/R^2
#   - chi (Euler characteristic)
#   - |int K dA - 2*pi*chi|  (Gauss-Bonnet residual)
#   - ||L_dec - L_cotan||_inf
#   - max|L*ones| for both methods
#
# For each family, separate tables are printed for barycentric and mixed duals.
#
# Usage:
#   julia --project=. convergence/sphere_convergence.jl

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "fit.jl"))
include(joinpath(@__DIR__, "metrics.jl"))
include(joinpath(@__DIR__, "helpers_generators.jl"))

print_header("Sphere Convergence Study")

R           = 1.0
exact_area  = 4π * R^2
exact_vol   = (4/3) * π * R^3
exact_H     = 1.0 / R      # mean curvature H = 1/R (positive, outward)
exact_K     = 1.0 / R^2    # Gaussian curvature

@printf("\n  R = %.2f,  area = %.6f,  volume = %.6f,  H = %.4f,  K = %.4f\n\n",
    R, exact_area, exact_vol, exact_H, exact_K)

function run_sphere_study(meshes_with_info, family_name::String, da_method::Symbol)
    print_sep("$family_name — dual_area = $da_method")

    # Column headers
    @printf("  %-20s  %-8s  %-10s  %-10s  %-10s  %-10s  %-4s  %-10s  %-10s\n",
        "label", "h", "err_A", "err_V", "err_H", "err_K", "chi", "GB_res", "dL_inf")
    println("  " * "-" ^ 100)

    hs      = Float64[]
    err_As  = Float64[]
    err_Vs  = Float64[]
    err_Hs  = Float64[]
    err_Ks  = Float64[]
    labels  = String[]

    for (label, mesh) in meshes_with_info
        geom = compute_geometry(mesh; dual_area=da_method)
        dec  = build_dec(mesh, geom)

        h    = mesh_size_surface(mesh, geom)
        A    = measure(mesh, geom)
        V    = enclosed_measure(mesh)
        chi  = euler_characteristic(mesh)
        intK = integrated_gaussian_curvature(mesh, geom)
        gb   = abs(intK - 2π * chi)

        err_A = abs(A - exact_area)
        err_V = abs(V - exact_vol)

        # Mean curvature: compare mean of abs(H) to exact
        geom_c = compute_curvature(mesh, geom, dec)
        H_disc = geom_c.mean_curvature
        err_H  = mean(abs.(abs.(H_disc) .- exact_H))

        K_disc  = gaussian_curvature(mesh, geom_c)
        da      = geom.vertex_dual_areas
        err_K   = curvature_error(K_disc, exact_K, da; norm=:L2)

        rpt     = compare_laplace_methods(mesh, geom)
        dL      = rpt.norm_inf

        push!(hs,     h)
        push!(err_As, err_A)
        push!(err_Vs, err_V)
        push!(err_Hs, err_H)
        push!(err_Ks, err_K)
        push!(labels, label)

        @printf("  %-20s  %8.4e  %10.4e  %10.4e  %10.4e  %10.4e  %4d  %10.4e  %10.4e\n",
            label, h, err_A, err_V, err_H, err_K, chi, gb, dL)
    end

    # Pairwise orders
    println()
    @printf("  %-20s  %-8s  %-8s  %-8s  %-8s\n",
        "pair", "OOC_A", "OOC_V", "OOC_H", "OOC_K")
    oa = pairwise_orders(hs, err_As)
    ov = pairwise_orders(hs, err_Vs)
    oh = pairwise_orders(hs, err_Hs)
    ok = pairwise_orders(hs, err_Ks)
    for i in 1:length(oa)
        @printf("  %-20s  %-8s  %-8s  %-8s  %-8s\n",
            "$(labels[i]) -> $(labels[i+1])",
            format_order(oa[i]), format_order(ov[i]),
            format_order(oh[i]), format_order(ok[i]))
    end

    # Fitted orders
    println()
    @printf("  Fitted order — area: %s,  vol: %s,  H: %s,  K: %s\n",
        format_order(fitted_order(hs, err_As)),
        format_order(fitted_order(hs, err_Vs)),
        format_order(fitted_order(hs, err_Hs)),
        format_order(fitted_order(hs, err_Ks)))
    println()
end

# ── UV-sphere ────────────────────────────────────────────────────────────────
print_sep("UV Sphere Family")
uv_meshes = [
    ("uv (8,16)",   generate_uvsphere(R, 8,  16)),
    ("uv (12,24)",  generate_uvsphere(R, 12, 24)),
    ("uv (16,32)",  generate_uvsphere(R, 16, 32)),
    ("uv (24,48)",  generate_uvsphere(R, 24, 48)),
    ("uv (32,64)",  generate_uvsphere(R, 32, 64)),
]

run_sphere_study(uv_meshes, "UV Sphere", :barycentric)
run_sphere_study(uv_meshes, "UV Sphere", :mixed)

# ── Icosphere ────────────────────────────────────────────────────────────────
print_sep("Icosphere Family")
ico_meshes = [
    ("ico level 0", generate_icosphere(R, 0)),
    ("ico level 1", generate_icosphere(R, 1)),
    ("ico level 2", generate_icosphere(R, 2)),
    ("ico level 3", generate_icosphere(R, 3)),
    ("ico level 4", generate_icosphere(R, 4)),
]

run_sphere_study(ico_meshes, "Icosphere", :barycentric)
run_sphere_study(ico_meshes, "Icosphere", :mixed)

print_sep()
println("  Done: sphere_convergence.jl")
