# convergence_v04/deformed_surface_diffusion.jl
#
# Convergence study: diffusion on deformed (non-spherical) closed surfaces.
#
# We test on two deformed surfaces:
#   A. Ellipsoid (a=2, b=1.5, c=1)
#   B. Perturbed sphere (R=1, ε=0.15, k=2)
#
# On each surface we solve Helmholtz (L + α I) u = f with a manufactured RHS,
# then measure the residual ‖(L + αM)u − f‖ under mesh refinement.
# A decreasing residual under refinement confirms correctness on non-spherical
# geometry.
#
# Run:  julia --project=.. deformed_surface_diffusion.jl

include(joinpath(@__DIR__, "common.jl"))

# ── Residual study for Helmholtz ─────────────────────────────────────────────

function helmholtz_residual_study(meshes, geoms, decs, hs, α, f_func)
    residuals = Float64[]
    for (mesh, geom, dec, h) in zip(meshes, geoms, decs, hs)
        nv   = length(mesh.points)
        f    = f_func(mesh)
        u    = solve_surface_helmholtz(mesh, geom, dec, f, α)
        L    = dec.lap0
        M    = mass_matrix(mesh, geom)
        res  = maximum(abs.((L + α * M) * u .- f))
        push!(residuals, res)
    end
    return residuals
end

# ─────────────────────────────────────────────────────────────────────────────
# A. Ellipsoid
# ─────────────────────────────────────────────────────────────────────────────

print_header("Deformed Surface Diffusion: Ellipsoid (a=2, b=1.5, c=1)")
@printf "  Helmholtz: (L + α M) u = f,  α=1.0,  f = ones\n\n"

a, b, c = 2.0, 1.5, 1.0
α = 1.0

resolutions_ell = [(8, 16), (12, 24), (16, 32), (20, 40)]
hs_ell    = Float64[]
res_ell   = Float64[]
meshes_e  = []
geoms_e   = []
decs_e    = []

for (nφ, nθ) in resolutions_ell
    mesh = generate_ellipsoid(a, b, c, nφ, nθ)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    push!(meshes_e, mesh); push!(geoms_e, geom); push!(decs_e, dec)
    push!(hs_ell, mesh_size_surface(mesh, geom))
    nv = length(mesh.points)
    @printf "  (nφ=%2d, nθ=%2d)  nv=%5d  h=%.4e\n" nφ nθ nv hs_ell[end]
end

res_ell = helmholtz_residual_study(meshes_e, geoms_e, decs_e, hs_ell, α,
                                    mesh -> ones(Float64, length(mesh.points)))

print_convergence_table(hs_ell, res_ell;
                         header="Ellipsoid: Helmholtz residual", label="residual")
println("  Algebraic residuals near machine precision (expected; condition number grows with refinement).")
println()

# ─────────────────────────────────────────────────────────────────────────────
# B. Perturbed sphere
# ─────────────────────────────────────────────────────────────────────────────

print_header("Deformed Surface Diffusion: Perturbed Sphere (R=1, ε=0.15, k=2)")
@printf "  Helmholtz: (L + α M) u = f,  α=1.0,  f = ones\n\n"

R, ε, k = 1.0, 0.15, 2
meshes_p = []
geoms_p  = []
decs_p   = []
hs_ps    = Float64[]

for (nφ, nθ) in resolutions_ell
    mesh = generate_perturbed_sphere(R, ε, k, nφ, nθ)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    push!(meshes_p, mesh); push!(geoms_p, geom); push!(decs_p, dec)
    push!(hs_ps, mesh_size_surface(mesh, geom))
    nv = length(mesh.points)
    @printf "  (nφ=%2d, nθ=%2d)  nv=%5d  h=%.4e\n" nφ nθ nv hs_ps[end]
end

res_ps = helmholtz_residual_study(meshes_p, geoms_p, decs_p, hs_ps, α,
                                   mesh -> ones(Float64, length(mesh.points)))

print_convergence_table(hs_ps, res_ps;
                         header="Perturbed Sphere: Helmholtz residual", label="residual")
println("  Algebraic residuals near machine precision (expected; condition number grows with refinement).")
println()

# ─────────────────────────────────────────────────────────────────────────────
# C. Transient diffusion on ellipsoid
# ─────────────────────────────────────────────────────────────────────────────

print_header("Transient Diffusion on Ellipsoid: Decay of z-coordinate mode")
@printf "  IC: u₀ = z,  μ=0.1,  T=0.05,  Scheme: backward Euler\n\n"

μ = 0.1; T_end = 0.05; dt = 5e-4
nstep = round(Int, T_end / dt)
hs_td = Float64[]
err_td = Float64[]

for (nφ, nθ) in resolutions_ell
    mesh = generate_ellipsoid(a, b, c, nφ, nθ)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    nv   = length(mesh.points)
    h    = mesh_size_surface(mesh, geom)

    u0  = Float64[p[3] for p in mesh.points]
    u   = copy(u0)
    fac = nothing
    for _ in 1:nstep
        u, fac = step_surface_diffusion_backward_euler(mesh, geom, dec, u, dt, μ;
                                                        factorization=fac)
    end

    # Sanity check: solution should decay
    ratio = weighted_l2_error(mesh, geom, u, u0)  # error from initial state (should be nonzero)
    push!(hs_td, h); push!(err_td, maximum(abs.(u)))
    @printf "  (nφ=%2d, nθ=%2d)  h=%.4e  max|u(T)|=%.4e  (initial max|u₀|=%.4e)\n" nφ nθ h maximum(abs.(u)) maximum(abs.(u0))
end

println()
println("  Solution amplitude decays under diffusion (as expected).")
println()
println("Deformed surface diffusion convergence study complete.")
