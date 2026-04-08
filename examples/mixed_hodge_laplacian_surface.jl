using FrontIntrinsicOps
using LinearAlgebra

mesh = generate_icosphere(1.0, 2)
geom = compute_geometry(mesh)

u_exact = [p[3] for p in mesh.points]
rhs = 2.0 .* u_exact

sol = solve_mixed_hodge_laplacian0(mesh, geom, rhs; gauge=:mean_zero)
u = sol.u

u_ref = copy(u_exact)
zero_mean_projection!(u_ref, mesh, geom)
zero_mean_projection!(u, mesh, geom)

relerr = norm(u - u_ref) / max(norm(u_ref), 1e-14)
println("relative error (z-eigenfunction on sphere): ", relerr)

cmp = compare_dec_vs_whitney_laplacian(mesh, geom)
println("DEC vs Whitney Laplacian rel diff: ", cmp.rel_diff)
println("Whitney null residual on constants: ", cmp.null_residual_whitney)
