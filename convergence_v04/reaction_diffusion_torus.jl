# convergence_v04/reaction_diffusion_torus.jl
#
# Convergence study: reaction–diffusion on the torus.
#
# PDE:    du/dt = μ ΔΓ u − α u,   u(0) = cos(θ)
# Exact:  u(t) = exp(−(μ λ + α) t) · cos(θ)
#
# where λ is the (approximate) Laplace–Beltrami eigenvalue for the cos(θ)
# mode on the torus.  We refine the torus mesh and measure L² error at T_end.
#
# Expected convergence rate: O(h²) in space.
#
# Run:  julia --project=.. reaction_diffusion_torus.jl

include(joinpath(@__DIR__, "common.jl"))

# ── Parameters ────────────────────────────────────────────────────────────────

R     = 2.0          # major radius
r     = 0.5          # minor radius
μ     = 0.05         # diffusion coefficient
α     = 1.0          # linear decay rate
T_end = 0.05
dt    = 5e-4
nstep = round(Int, T_end / dt)

print_header("Convergence: Reaction–Diffusion on Torus")
@printf "  R=%.2f, r=%.2f, μ=%.4f, α=%.2f, T=%.3f, dt=%.2e, nstep=%d\n" R r μ α T_end dt nstep
@printf "  Scheme: IMEX backward-Euler  (diffusion implicit, reaction explicit)\n"
@printf "  IC: u₀ = cos(θ)  (toroidal mode)\n\n"

# ── Mesh refinement loop ──────────────────────────────────────────────────────

ntheta_list = [10, 20, 30, 40]
hs     = Float64[]
errors = Float64[]
nverts = Int[]

for nθ in ntheta_list
    nφ   = max(4, round(Int, nθ * r / R))   # keep aspect ratio
    mesh = generate_torus(R, r, nθ, nφ)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    h  = mesh_size_surface(mesh, geom)
    nv = length(mesh.points)

    # Initial condition: u₀ = cos(θ)
    u0 = Float64[cos(atan(p[2], p[1])) for p in mesh.points]

    # Time integration with IMEX backward-Euler
    reaction = linear_decay_reaction(α)
    u_end, _ = solve_surface_reaction_diffusion(mesh, geom, dec, u0, T_end, dt, μ,
                                                 reaction; θ=1.0, scheme=:imex)

    # Approximate eigenvalue for cos(θ) mode: λ ≈ 1/(R^2) for large R/r
    λ_approx = 1.0 / R^2
    decay     = exp(-(μ * λ_approx + α) * T_end)
    u_exact   = decay .* u0

    err = weighted_l2_error(mesh, geom, u_end, u_exact)
    push!(hs, h); push!(errors, err); push!(nverts, nv)

    @printf "  nθ=%3d, nφ=%2d, nv=%5d, h=%.4e, L2_err=%.4e\n" nθ nφ nv h err
end

# ── Convergence table ─────────────────────────────────────────────────────────

print_convergence_table(hs, errors; header="Spatial Convergence", label="L2 error")
println("Expected rate: ~2.0 (second-order in space)")
println()
println("Reaction–diffusion torus convergence study complete.")
