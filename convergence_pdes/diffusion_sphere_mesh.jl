# convergence_pdes/diffusion_sphere_mesh.jl
#
# Mesh refinement convergence study for transient diffusion on the unit sphere.
#
# PDE:    du/dt + μ L u = 0,   u(0) = z
# Exact:  u(t) = exp(-μ λ₁ t) · z,  λ₁ = 2  (for R=1)
#
# We fix dt and T_end, then refine the spatial mesh (icosphere levels 1..4).
# The time step is chosen fine enough so temporal error is negligible.
# Scheme: backward Euler.
#
# Expected spatial convergence rate: O(h²) for L2 error.
#
# Run:  julia --project=.. diffusion_sphere_mesh.jl

include(joinpath(@__DIR__, "common.jl"))

# ── Parameters ────────────────────────────────────────────────────────────────

R     = 1.0
μ     = 0.1
T_end = 0.1
dt    = 1e-3         # fine time step: temporal error ≪ spatial error
nstep = round(Int, T_end / dt)
λ₁    = 2.0 / R^2

print_header("Mesh Refinement Convergence: Diffusion on Unit Sphere")
@printf "  μ=%.3f,  T=%.3f,  dt=%.4e,  nstep=%d\n" μ T_end dt nstep
@printf "  Scheme: backward Euler\n"
@printf "  Exact:  u(t) = exp(-μ λ₁ t) · z,  λ₁=%.4f\n\n" λ₁

# ── Convergence loop ──────────────────────────────────────────────────────────

levels = 1:4
hs     = Float64[]
errors = Float64[]
nverts = Int[]

for level in levels
    mesh = generate_icosphere(R, level)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    h  = mesh_size_surface(mesh, geom)
    nv = length(mesh.points)

    # Initial condition
    z_0   = Float64[p[3] for p in mesh.points]

    # Time integration
    u_end = let u = copy(z_0), fac = nothing
        for _ in 1:nstep
            u, fac = step_surface_diffusion_backward_euler(mesh, geom, dec, u, dt, μ;
                                                            factorization=fac)
        end
        u
    end

    # Error vs exact
    u_exact = sphere_eigenmode_exact(mesh, T_end, μ; λ=λ₁)
    err     = weighted_l2_error(mesh, geom, u_end, u_exact)

    push!(hs,     h)
    push!(errors, err)
    push!(nverts, nv)

    @printf "  level=%d,  nv=%5d,  h=%.4e,  L2_err=%.4e\n" level nv h err
end

# ── Convergence table ─────────────────────────────────────────────────────────

println()
print_sep("Convergence Table")

@printf "  %-6s  %-8s  %-12s  %-12s  %-8s\n" "level" "nv" "h" "L2 error" "order"
println("  " * "─"^54)
for i in eachindex(levels)
    order_str = i == 1 ? "  —   " :
        @sprintf("%.3f", log(errors[i-1]/errors[i]) / log(hs[i-1]/hs[i]))
    @printf "  %-6d  %-8d  %-12.4e  %-12.4e  %-8s\n" levels[i] nverts[i] hs[i] errors[i] order_str
end

println()
println("Expected convergence rate: ~2.0 (second-order in space).")
println()
println("Diffusion mesh convergence study complete.")
