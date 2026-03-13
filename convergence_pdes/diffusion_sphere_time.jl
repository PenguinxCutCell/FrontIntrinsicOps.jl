# convergence_pdes/diffusion_sphere_time.jl
#
# Time refinement convergence study for transient diffusion on the unit sphere.
#
# PDE:    du/dt + μ L u = 0,   u(0) = z
# Exact:  u(t) = exp(-μ λ₁ t) · z,  λ₁ = 2  (for R=1)
#
# We fix a fine spatial mesh (icosphere level 4) and refine dt.
# Two schemes are compared:
#   - Backward Euler:  1st-order in time  → O(dt)
#   - Crank–Nicolson:  2nd-order in time  → O(dt²)
#
# Run:  julia --project=.. diffusion_sphere_time.jl

include(joinpath(@__DIR__, "common.jl"))

# ── Parameters ────────────────────────────────────────────────────────────────

R     = 1.0
μ     = 0.5    # stronger diffusion → larger temporal errors for same dt
T_end = 0.5
λ₁    = 2.0 / R^2

print_header("Time Refinement Convergence: Diffusion on Unit Sphere")
@printf "  μ=%.3f,  T=%.3f,  spatial mesh: icosphere level 2\n" μ T_end
@printf "  Exact: u(t) = exp(-μ λ₁ t) · z,  λ₁=%.4f\n" λ₁
@printf "  Final decay factor: exp(-μ λ₁ T) = %.4f\n\n" exp(-μ * λ₁ * T_end)

# ── Build spatial mesh ────────────────────────────────────────────────────────
# Level 3: h≈0.099, spatial floor ~1.6e-04.
# CN temporal error at dt=0.25 should dominate spatial floor.

mesh = generate_icosphere(R, 7)
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)

nv = length(mesh.points)
@printf "  Mesh: %d vertices, %d faces, h=%.4e\n\n" nv length(mesh.faces) mesh_size_surface(mesh, geom)

# ── Time step sequence ────────────────────────────────────────────────────────

dts = [0.25, 0.125, 0.05, 0.025, 0.0125]

errors_be = Float64[]
errors_cn = Float64[]

u_exact_final = sphere_eigenmode_exact(mesh, T_end, μ; λ=λ₁)
z             = Float64[p[3] for p in mesh.points]

for dt in dts
    nstep = round(Int, T_end / dt)
    dt_actual = T_end / nstep

    # ── Backward Euler ──────────────────────────────────────────────────
    err_be = let u = copy(z), fac = nothing
        for _ in 1:nstep
            u, fac = step_surface_diffusion_backward_euler(mesh, geom, dec, u, dt_actual, μ;
                                                            factorization=fac)
        end
        weighted_l2_error(mesh, geom, u, u_exact_final)
    end

    # ── Crank–Nicolson ──────────────────────────────────────────────────
    err_cn = let u = copy(z), fac = nothing
        for _ in 1:nstep
            u, fac = step_surface_diffusion_crank_nicolson(mesh, geom, dec, u, dt_actual, μ;
                                                            factorization=fac)
        end
        weighted_l2_error(mesh, geom, u, u_exact_final)
    end

    push!(errors_be, err_be)
    push!(errors_cn, err_cn)

    @printf "  dt=%.4f  nstep=%4d  err_BE=%.4e  err_CN=%.4e\n" dt_actual nstep err_be err_cn
end

# ── Convergence table ─────────────────────────────────────────────────────────

println()
print_sep("Convergence Table")
print_time_convergence_table(dts, errors_be, errors_cn)

println("Expected orders:")
println("  Backward Euler:  ~1.0 (first-order in time)")
println("  Crank–Nicolson:  ~2.0 (second-order in time)")
println()
println("Time refinement convergence study complete.")
