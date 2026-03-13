# examples/surface_diffusion_sphere.jl
#
# Transient scalar diffusion on the unit sphere using the Laplace–Beltrami
# operator from FrontIntrinsicOps.
#
# PDE:  du/dt + μ L u = 0,   u(0) = z
#
# Exact solution:
#   The z-coordinate is an eigenfunction of L with eigenvalue λ₁ = 2/R².
#   Therefore  u(t) = exp(-μ λ₁ t) · z.
#
# We compare backward Euler (1st-order) and Crank–Nicolson (2nd-order), and
# demonstrate factorization reuse for efficient multi-step integration.
#
# Run:  julia --project examples/surface_diffusion_sphere.jl

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
z  = Float64[p[3] for p in mesh.points]

@printf "Icosphere level 3: %d vertices, %d faces\n" nv length(mesh.faces)
@printf "Surface area:      %.6f  (exact 4π ≈ %.6f)\n" measure(mesh, geom) 4π*R^2

# ── Parameters ───────────────────────────────────────────────────────────────

μ     = 0.1          # diffusion coefficient
T_end = 0.5          # final time
λ₁    = 2.0 / R^2   # first non-trivial eigenvalue of L on sphere of radius R
dt    = 1e-2
nstep = round(Int, T_end / dt)

println()
@printf "Parameters: μ=%.3f, λ₁=%.4f, T=%.2f, dt=%.4f, nstep=%d\n" μ λ₁ T_end dt nstep
@printf "Amplitude decay factor: exp(-μ λ₁ T) = %.6f\n" exp(-μ * λ₁ * T_end)

# ── Backward Euler ───────────────────────────────────────────────────────────

println()
println("─── Backward Euler (1st-order) ─────────────────────────────────")
@printf "  %-8s  %-14s  %-14s  %-14s\n" "step" "t" "|u|∞/|z|∞" "L2 error"

u_exact_final = exp(-μ * λ₁ * T_end) .* z

l2_be, linf_be = let u = copy(z), fac = nothing
    for k in 1:nstep
        u, fac = step_surface_diffusion_backward_euler(mesh, geom, dec, u, dt, μ;
                                                        factorization=fac)
        if k == 1 || k % (nstep ÷ 5) == 0 || k == nstep
            t_k     = k * dt
            decay   = maximum(abs, u) / maximum(abs, z)
            u_ref   = exp(-μ * λ₁ * t_k) .* z
            err     = weighted_l2_error(mesh, geom, u, u_ref)
            @printf "  %-8d  %-14.6f  %-14.6f  %-14.4e\n" k t_k decay err
        end
    end
    weighted_l2_error(mesh, geom, u, u_exact_final),
    weighted_linf_error(mesh, geom, u, u_exact_final)
end
@printf "\nBackward Euler final errors: L2=%.4e  Linf=%.4e\n" l2_be linf_be

# ── Crank–Nicolson ───────────────────────────────────────────────────────────

println()
println("─── Crank–Nicolson (2nd-order) ─────────────────────────────────")
@printf "  %-8s  %-14s  %-14s  %-14s\n" "step" "t" "|u|∞/|z|∞" "L2 error"

l2_cn, linf_cn = let u = copy(z), fac = nothing
    for k in 1:nstep
        u, fac = step_surface_diffusion_crank_nicolson(mesh, geom, dec, u, dt, μ;
                                                        factorization=fac)
        if k == 1 || k % (nstep ÷ 5) == 0 || k == nstep
            t_k     = k * dt
            decay   = maximum(abs, u) / maximum(abs, z)
            u_ref   = exp(-μ * λ₁ * t_k) .* z
            err     = weighted_l2_error(mesh, geom, u, u_ref)
            @printf "  %-8d  %-14.6f  %-14.6f  %-14.4e\n" k t_k decay err
        end
    end
    weighted_l2_error(mesh, geom, u, u_exact_final),
    weighted_linf_error(mesh, geom, u, u_exact_final)
end
@printf "\nCrank–Nicolson final errors: L2=%.4e  Linf=%.4e\n" l2_cn linf_cn

# ── Factorization reuse timing ───────────────────────────────────────────────

println()
println("─── Factorization reuse demonstration ──────────────────────────")

# First step builds the factorization
t0 = time()
u_tmp, fac0 = step_surface_diffusion_backward_euler(mesh, geom, dec, z, dt, μ)
t_build = time() - t0
@printf "Step with factorization build:  %.4f s\n" t_build

# Subsequent steps reuse it
t1 = time()
let u = copy(u_tmp), fac = fac0
    for _ in 1:20
        u, fac = step_surface_diffusion_backward_euler(mesh, geom, dec, u, dt, μ;
                                                        factorization=fac)
    end
end
t_reuse = (time() - t1) / 20
@printf "Average step with reuse:        %.4f s  (%.1fx speedup)\n" t_reuse (t_build/t_reuse)

println()
println("Surface diffusion example complete.")
