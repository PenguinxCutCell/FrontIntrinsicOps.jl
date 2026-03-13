# convergence/poisson_sphere_convergence.jl
#
# Convergence study for the scalar Poisson equation on a sphere.
#
# Problem:  L u = f  on the sphere,  L = -Delta_Gamma (positive operator)
#
# Manufactured solution:
#   u_exact = x   (x-coordinate function on the sphere)
#   L u_exact = (2/R^2) * x  (since -Delta_Gamma x = (2/R^2) x for sphere)
#   f = (2/R^2) * x
#
# The system L u = f has a one-dimensional nullspace (constants).
# We pin u[1] = u_exact[1] to make the system uniquely solvable.
#
# Errors:
#   - relative solution error: ||u_h - u_exact||_2 / ||u_exact||_2
#   - max |L u_h - f|  (linear residual)
#   - compatibility residual: |ones' * f| (should be near 0 since int f dA = 0)
#
# Usage:
#   julia --project=. convergence/poisson_sphere_convergence.jl

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "fit.jl"))

using SparseArrays

print_header("Poisson on Sphere: Convergence Study")

R       = 1.0
lambda  = 2.0 / R^2  # eigenvalue: L x = lambda * x

@printf("\n  R = %.2f,  lambda = %.4f (eigenvalue for coord functions)\n\n", R, lambda)

function solve_poisson_sphere(mesh, geom, L)
    nv     = length(mesh.points)
    x_pts  = [p[1] for p in mesh.points]
    u_exact = x_pts
    f = lambda .* x_pts

    # Compatibility check: int f dA = 0 for coord functions on sphere
    da    = geom.vertex_dual_areas
    compat = abs(dot(da, f))

    # Pin vertex 1: replace first equation by u[1] = u_exact[1]
    # Build modified system
    Lmod = copy(L)
    Lmod[1, :] .= 0
    Lmod[1, 1]  = 1
    rhs = copy(f)
    rhs[1] = u_exact[1]

    # Solve
    u_h = Matrix(Lmod) \ rhs

    # Subtract mean to align gauges (in case of drift)
    u_h = u_h .- mean(u_h) .+ mean(u_exact)

    rel_err = norm(u_h .- u_exact) / (norm(u_exact) + eps())
    lin_res = maximum(abs, L * u_h .- f)
    return compat, rel_err, lin_res
end

function run_poisson_study(meshes_with_labels, family_name::String, lap_method::Symbol)
    print_sep("$family_name — laplace = $lap_method")
    @printf("  %-20s  %-8s  %-12s  %-12s  %-12s\n",
        "label", "h", "compat", "rel_err", "lin_res")
    println("  " * "-" ^ 80)

    hs       = Float64[]
    rel_errs = Float64[]
    labels   = String[]

    for (label, mesh) in meshes_with_labels
        geom = compute_geometry(mesh)
        L    = build_laplace_beltrami(mesh, geom; method=lap_method)
        h    = mesh_size_surface(mesh, geom)
        compat, rel_err, lin_res = solve_poisson_sphere(mesh, geom, L)
        push!(hs,       h)
        push!(rel_errs, rel_err)
        push!(labels,   label)
        @printf("  %-20s  %8.4e  %12.4e  %12.4e  %12.4e\n",
            label, h, compat, rel_err, lin_res)
    end

    println()
    @printf("  %-20s  %-8s\n", "pair", "OOC_rel")
    orc = pairwise_orders(hs, rel_errs)
    for i in 1:length(orc)
        @printf("  %-20s  %-8s\n",
            "$(labels[i]) -> $(labels[i+1])", format_order(orc[i]))
    end
    println()
    @printf("  Fitted order (rel_err): %s\n\n", format_order(fitted_order(hs, rel_errs)))
end

# Icosphere meshes for convergence
ico_meshes = [
    ("ico level 1", generate_icosphere(R, 1)),
    ("ico level 2", generate_icosphere(R, 2)),
    ("ico level 3", generate_icosphere(R, 3)),
    ("ico level 4", generate_icosphere(R, 4)),
]

run_poisson_study(ico_meshes, "Icosphere", :dec)
run_poisson_study(ico_meshes, "Icosphere", :cotan)

# UV-sphere meshes
uv_meshes = [
    ("uv (8,16)",   generate_uvsphere(R, 8,  16)),
    ("uv (12,24)",  generate_uvsphere(R, 12, 24)),
    ("uv (16,32)",  generate_uvsphere(R, 16, 32)),
    ("uv (24,48)",  generate_uvsphere(R, 24, 48)),
]

run_poisson_study(uv_meshes, "UV Sphere", :dec)
run_poisson_study(uv_meshes, "UV Sphere", :cotan)

print_sep()
println("  Done: poisson_sphere_convergence.jl")
