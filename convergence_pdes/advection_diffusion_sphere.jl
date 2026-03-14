# convergence_pdes/advection_diffusion_sphere.jl
#
# IMEX benchmark: rigid rotation + small diffusion on the unit sphere.
#
# PDE:   du/dt + M⁻¹ A u + μ L u = 0
#
# Physical setup:
#   - Velocity:   v(x,y,z) = (-y, x, 0),  ω = 1 rad/s
#   - Diffusion:  μ = 0.01  (Pe = O(1/μ), advection-dominated)
#   - Initial:    u(0) = z  (rotation-invariant dipole)
#
# Since z is invariant under rotation about z, the dynamics is pure diffusion:
#   u(t) ≈ exp(-μ λ₁ t) · z  (with small corrections from upwind dissipation)
#
# We integrate with the IMEX scheme (explicit transport, implicit diffusion)
# and print diagnostics over time.
#
# Run:  julia --project=.. advection_diffusion_sphere.jl

include(joinpath(@__DIR__, "common.jl"))

# ── Parameters ────────────────────────────────────────────────────────────────

R   = 1.0
μ   = 0.01      # small diffusion (advection-dominated regime)
T   = 2.0       # long enough to see diffusive decay
λ₁  = 2.0 / R^2

print_header("IMEX Benchmark: Advection–Diffusion on Unit Sphere")
@printf "  μ=%.4f,  ω=1.0,  T=%.2f,  λ₁=%.4f\n" μ T λ₁
@printf "  Mesh: icosphere level 3\n"
@printf "  Scheme: IMEX (explicit transport / implicit diffusion, upwind)\n\n"

# ── Build mesh ────────────────────────────────────────────────────────────────

mesh = generate_icosphere(R, 3)
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)

nv = length(mesh.points)
@printf "  %d vertices, %d faces, h=%.4e\n\n" nv length(mesh.faces) mesh_size_surface(mesh, geom)

# ── Initial condition and velocity field ──────────────────────────────────────

z      = Float64[p[3] for p in mesh.points]
vel    = SVector{3,Float64}[SVector{3,Float64}(-p[2], p[1], 0.0) for p in mesh.points]
M0     = dot(geom.vertex_dual_areas, z)        # ≈ 0 for z on sphere
M0_abs = dot(geom.vertex_dual_areas, abs.(z))  # use as mass scale

# CFL time step
dt    = estimate_transport_dt(mesh, geom, vel; cfl=0.3)
nstep = max(1, round(Int, T / dt))
dt    = T / nstep

@printf "  dt=%.6f,  nstep=%d\n\n" dt nstep

# ── Diagnostic output ─────────────────────────────────────────────────────────

println("─── Time integration diagnostics ────────────────────────────────────")
@printf "  %-8s  %-12s  %-14s  %-14s  %-14s\n" "step" "t" "|u|∞" "mass_err_rel" "L2_vs_diffexact"

print_every = max(1, nstep ÷ 20)

u, _ = let u = copy(z), fac = nothing
    for k in 1:nstep
        u, fac = step_surface_advection_diffusion_imex(mesh, geom, dec, u, vel, dt, μ;
                                                        scheme=:upwind,
                                                        factorization=fac)

        if k % print_every == 0 || k == nstep
            t_k      = k * dt
            umax     = maximum(abs, u)
            mass_err = abs(dot(geom.vertex_dual_areas, u) - M0) / M0_abs
            u_ref    = exp(-μ * λ₁ * t_k) .* z
            l2_err   = weighted_l2_error(mesh, geom, u, u_ref)
            @printf "  %-8d  %-12.4f  %-14.6f  %-14.4e  %-14.4e\n" k t_k umax mass_err l2_err
        end
    end
    u, fac
end

# ── Final diagnostics ─────────────────────────────────────────────────────────

println()
println("─── Final summary ───────────────────────────────────────────────────")
u_exact_final = exp(-μ * λ₁ * T) .* z
l2_final      = weighted_l2_error(mesh, geom, u, u_exact_final)
linf_final    = weighted_linf_error(mesh, geom, u, u_exact_final)
mass_final    = dot(geom.vertex_dual_areas, u)

@printf "Diffusion-exact decay factor:       %.6f\n" exp(-μ * λ₁ * T)
@printf "Numerical amplitude ratio |u|∞/|z|∞: %.6f\n" maximum(abs, u) / maximum(abs, z)
@printf "L2  error vs diffusion-only exact:  %.4e\n" l2_final
@printf "Linf error vs diffusion-only exact: %.4e\n" linf_final
@printf "Relative mass change |M(T)-M(0)|/∫|u|dA: %.4e\n" abs(mass_final - M0)/M0_abs

println()
println("Notes:")
println("  - IMEX factorization is reused across all steps for efficiency.")
println("  - Upwind dissipation adds O(h) numerical diffusion on top of μ.")
println("  - The z-field is rotation-invariant; dominant error is from mesh discretisation.")
println()
println("IMEX advection–diffusion benchmark complete.")
