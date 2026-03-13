# examples/surface_advection_diffusion_sphere.jl
#
# Combined advection–diffusion on the unit sphere using the IMEX scheme.
#
# PDE:  du/dt + M⁻¹ A u + μ L u = 0
#
# Physical setup:
#   - Velocity:   v(x,y,z) = (-y, x, 0)  (rigid rotation around z-axis, ω=1)
#   - Diffusion:  μ = 0.01  (small, so rotation dominates)
#   - Initial:    u(0) = z  (dipole)
#
# The IMEX scheme treats transport explicitly and diffusion implicitly,
# making it suitable when diffusion is stiff but advection is not.
#
# The field z is invariant under z-rotation, so the exact solution is
# dominated by pure diffusive decay:  u(t) ≈ exp(-μ λ₁ t) · z.
#
# Run:  julia --project examples/surface_advection_diffusion_sphere.jl

using FrontIntrinsicOps
using StaticArrays
using LinearAlgebra
using Printf

# ── Mesh & geometry ──────────────────────────────────────────────────────────

R    = 1.0
mesh = generate_icosphere(R, 3)
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)

nv = length(mesh.points)
@printf "Icosphere level 3: %d vertices, %d faces\n" nv length(mesh.faces)
@printf "Surface area:      %.6f  (exact 4π ≈ %.6f)\n" measure(mesh, geom) 4π*R^2

# ── Parameters ───────────────────────────────────────────────────────────────

μ    = 0.01          # small diffusion coefficient
T    = 1.0           # final time
λ₁   = 2.0 / R^2    # first eigenvalue of L on sphere

# Rigid rotation velocity (angular frequency ω=1)
vel  = SVector{3,Float64}[SVector{3,Float64}(-p[2], p[1], 0.0) for p in mesh.points]

# CFL time step (limited by advection)
dt_cfl = estimate_transport_dt(mesh, geom, vel; cfl=0.3)
dt     = dt_cfl
nstep  = max(1, round(Int, T / dt))
dt     = T / nstep

@printf "\nParameters: μ=%.4f, λ₁=%.4f, T=%.2f\n" μ λ₁ T
@printf "CFL dt:     %.6f,  nstep=%d\n" dt nstep
@printf "Diffusion decay factor: exp(-μ λ₁ T) = %.6f\n\n" exp(-μ * λ₁ * T)

# ── Initial condition ─────────────────────────────────────────────────────────

z      = Float64[p[3] for p in mesh.points]
M0     = dot(geom.vertex_dual_areas, z)        # ≈ 0 for z on sphere
M0_abs = dot(geom.vertex_dual_areas, abs.(z))  # use as mass scale

# Approximate exact solution: z is invariant under rotation, decays by diffusion
u_exact_final = exp(-μ * λ₁ * T) .* z

# ── IMEX time integration ─────────────────────────────────────────────────────

println("─── IMEX (explicit transport, implicit diffusion) ───────────────")
@printf "  %-8s  %-12s  %-14s  %-14s  %-14s\n" "step" "t" "mass_err_rel" "|u|∞" "L2_err_vs_exact"

u, fac_final = let u = copy(z), fac = nothing
    for k in 1:nstep
        u, fac = step_surface_advection_diffusion_imex(mesh, geom, dec, u, vel, dt, μ;
                                                        scheme=:upwind,
                                                        factorization=fac)
        if k == 1 || k % max(1, nstep ÷ 10) == 0 || k == nstep
            t_k      = k * dt
            mass_err = abs(dot(geom.vertex_dual_areas, u) - M0) / M0_abs
            umax     = maximum(abs, u)
            u_ref    = exp(-μ * λ₁ * t_k) .* z
            l2_err   = weighted_l2_error(mesh, geom, u, u_ref)
            @printf "  %-8d  %-12.4f  %-14.4e  %-14.6f  %-14.4e\n" k t_k mass_err umax l2_err
        end
    end
    u, fac
end

# ── Final diagnostics ─────────────────────────────────────────────────────────

println()
println("─── Final diagnostics ───────────────────────────────────────────")
l2_final   = weighted_l2_error(mesh, geom, u, u_exact_final)
linf_final = weighted_linf_error(mesh, geom, u, u_exact_final)
mass_final = dot(geom.vertex_dual_areas, u)

@printf "L2 error vs diffusion-only exact:    %.4e\n" l2_final
@printf "Linf error vs diffusion-only exact:  %.4e\n" linf_final
@printf "Mass change: |M(T) - M(0)| / ∫|u| dA = %.4e\n" abs(mass_final - M0)/M0_abs
@printf "Amplitude ratio |u|∞ / |z|∞ = %.6f  (expected ≈ %.6f)\n" (maximum(abs, u) / maximum(abs, z)) exp(-μ * λ₁ * T)

println()
println("Note: small discrepancy from pure diffusion-exact is expected because")
println("      the upwind advection scheme introduces numerical dissipation.")
println()
println("Surface advection–diffusion (IMEX) example complete.")
