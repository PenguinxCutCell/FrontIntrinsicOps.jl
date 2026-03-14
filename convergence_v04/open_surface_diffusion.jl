# convergence_v04/open_surface_diffusion.jl
#
# Convergence study: diffusion / Poisson on an open surface (flat patch).
#
# PDE:   −ΔΓ u = 0   in Ω = [0,1]²  (flat patch)
#             u = g   on ∂Ω
#
# We use the linear Dirichlet data g(x,y) = x, for which the exact solution
# is u(x,y) = x (harmonic, satisfies Laplace exactly).
#
# Second study: u = sin(πx) sin(πy) with right-hand side f = 2π² u.
#
# We measure the L² error under mesh refinement and observe O(h²) convergence.
#
# Run:  julia --project=.. open_surface_diffusion.jl

include(joinpath(@__DIR__, "common.jl"))

# ── Helper: build flat patch ───────────────────────────────────────────────────

function make_flat_patch_conv(N, L=1.0)
    pts   = SVector{3,Float64}[]
    faces = SVector{3,Int}[]
    h     = L / N
    for j in 0:N, i in 0:N
        push!(pts, SVector{3,Float64}(i*h, j*h, 0.0))
    end
    idx_v(i, j) = j*(N+1) + i + 1
    for j in 0:(N-1), i in 0:(N-1)
        v00 = idx_v(i,   j  )
        v10 = idx_v(i+1, j  )
        v01 = idx_v(i,   j+1)
        v11 = idx_v(i+1, j+1)
        push!(faces, SVector{3,Int}(v00, v10, v11))
        push!(faces, SVector{3,Int}(v00, v11, v01))
    end
    return SurfaceMesh{Float64}(pts, faces)
end

# ─────────────────────────────────────────────────────────────────────────────
# A. Laplace equation with linear Dirichlet data (u_exact = x)
# ─────────────────────────────────────────────────────────────────────────────

print_header("Open Surface Diffusion: Laplace with u_exact = x (linear BC)")
@printf "  −ΔΓ u = 0,  u = x on ∂Ω,  exact: u = x\n\n"

Ns     = [4, 8, 16, 32]
hs_a   = Float64[]
errs_a = Float64[]

for N in Ns
    mesh = make_flat_patch_conv(N)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    topo = build_topology(mesh)
    nv   = length(mesh.points)
    h    = 1.0 / N

    bv   = detect_boundary_vertices(mesh, topo)
    g    = Float64[mesh.points[i][1] for i in bv]
    f    = zeros(Float64, nv)
    u    = solve_open_surface_poisson(mesh, geom, dec, topo, f, bv, g)

    u_exact = Float64[mesh.points[i][1] for i in 1:nv]
    err     = weighted_l2_error(mesh, geom, u, u_exact)
    push!(hs_a, h); push!(errs_a, err)

    @printf "  N=%3d  nv=%5d  h=%.4e  L2_err=%.4e\n" N nv h err
end

print_convergence_table(hs_a, errs_a; header="Laplace u=x BC: spatial convergence",
                         label="L2 error")
println("Expected: O(h²)  (the linear function is represented exactly by P1 FEM)")
println()

# ─────────────────────────────────────────────────────────────────────────────
# B. Poisson equation with u_exact = sin(πx) sin(πy)
# ─────────────────────────────────────────────────────────────────────────────

print_header("Open Surface Diffusion: Poisson with u_exact = sin(πx)sin(πy)")
@printf "  −ΔΓ u = 2π² sin(πx) sin(πy),  u = 0 on ∂Ω\n\n"

hs_b   = Float64[]
errs_b = Float64[]

for N in Ns
    mesh = make_flat_patch_conv(N)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    topo = build_topology(mesh)
    nv   = length(mesh.points)
    h    = 1.0 / N

    bv   = detect_boundary_vertices(mesh, topo)
    g    = zeros(Float64, length(bv))   # u = 0 on boundary
    f    = Float64[2π^2 * sin(π * mesh.points[i][1]) * sin(π * mesh.points[i][2]) for i in 1:nv]
    u    = solve_open_surface_poisson(mesh, geom, dec, topo, f, bv, g)

    u_exact = Float64[sin(π * mesh.points[i][1]) * sin(π * mesh.points[i][2]) for i in 1:nv]
    err     = weighted_l2_error(mesh, geom, u, u_exact)
    push!(hs_b, h); push!(errs_b, err)

    @printf "  N=%3d  nv=%5d  h=%.4e  L2_err=%.4e\n" N nv h err
end

print_convergence_table(hs_b, errs_b; header="Poisson sin(πx)sin(πy): spatial convergence",
                         label="L2 error")
println("Expected: O(h²)  (cotangent-weight Laplacian on flat mesh)")
println()
println("Open surface diffusion convergence study complete.")
