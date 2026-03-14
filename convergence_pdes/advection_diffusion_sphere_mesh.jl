# convergence_pdes/advection_diffusion_sphere_mesh.jl
#
# Mesh refinement convergence study for advection–diffusion on the unit sphere.
#
# PDE:    du/dt + M⁻¹ A u + μ L u = 0
#
# Physical setup:
#   - Velocity:   v(x,y,z) = (-y, x, 0)  (rigid rotation, ω = 1 rad/s)
#   - Diffusion:  μ (varied to test different Péclet regimes)
#   - Initial:    u(0) = z  (rotation-invariant dipole)
#
# Since z is invariant under rotation about z, the PDE reduces to pure diffusion:
#
#   u(t) = exp(-μ λ₁ t) · z,   λ₁ = 2 / R²
#
# Spatial convergence is measured at a fixed final time T_end with a time step
# dt small enough that temporal error is negligible.
# Scheme: IMEX (explicit transport / implicit diffusion).
#
# Expected spatial convergence rates:
#   Upwind transport:   O(h)  — first-order, numerical diffusion ε_num ~ h/2
#   Centered transport: O(h²) — second-order, consistent with Laplace–Beltrami
#
# Run:  julia --project=.. advection_diffusion_sphere_mesh.jl

include(joinpath(@__DIR__, "common.jl"))

# ── Parameters ────────────────────────────────────────────────────────────────

R      = 1.0
μ      = 0.1       # diffusion coefficient
T_end  = 0.1       # final time (short enough for clean convergence)
dt     = 1e-3      # fixed fine time step: temporal error ≪ spatial error
nstep  = round(Int, T_end / dt)
λ₁     = 2.0 / R^2

print_header("Mesh Refinement Convergence: Advection–Diffusion on Unit Sphere")
@printf "  μ=%.3f,  ω=1.0,  T=%.3f,  dt=%.4e,  nstep=%d\n" μ T_end dt nstep
@printf "  Scheme: IMEX (explicit transport, implicit diffusion)\n"
@printf "  Exact:  u(t) = exp(-μ λ₁ t) · z  (rotation-invariant, λ₁=%.4f)\n\n" λ₁

# ── Convergence loop ──────────────────────────────────────────────────────────

levels   = 1:4
hs       = Float64[]
err_up   = Float64[]
err_cen  = Float64[]
nverts   = Int[]

for level in levels
    mesh = generate_icosphere(R, level)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    h  = mesh_size_surface(mesh, geom)
    nv = length(mesh.points)

    z_0 = Float64[p[3] for p in mesh.points]
    vel = SVector{3,Float64}[SVector{3,Float64}(-p[2], p[1], 0.0) for p in mesh.points]

    # Pre-assemble the (constant) transport operators.
    A_upwind   = assemble_transport_operator(mesh, geom, vel; scheme=:upwind)
    A_centered = assemble_transport_operator(mesh, geom, vel; scheme=:centered)

    # Upwind IMEX run
    u_up = let u = copy(z_0), fac = nothing
        for _ in 1:nstep
            u, fac = step_surface_advection_diffusion_imex(
                mesh, geom, dec, u, vel, dt, μ;
                scheme             = :upwind,
                transport_operator = A_upwind,
                factorization      = fac,
            )
        end
        u
    end

    # Centered IMEX run
    u_cen = let u = copy(z_0), fac = nothing
        for _ in 1:nstep
            u, fac = step_surface_advection_diffusion_imex(
                mesh, geom, dec, u, vel, dt, μ;
                scheme             = :centered,
                transport_operator = A_centered,
                factorization      = fac,
            )
        end
        u
    end

    # Error vs diffusion-only exact (valid since z is rotation-invariant)
    u_exact = sphere_eigenmode_exact(mesh, T_end, μ; λ=λ₁)
    eu = weighted_l2_error(mesh, geom, u_up,  u_exact)
    ec = weighted_l2_error(mesh, geom, u_cen, u_exact)

    push!(hs,     h)
    push!(err_up, eu)
    push!(err_cen, ec)
    push!(nverts, nv)

    @printf "  level=%d,  nv=%5d,  h=%.4e,  err_upwind=%.4e,  err_centered=%.4e\n" level nv h eu ec
end

# ── Convergence table ─────────────────────────────────────────────────────────

println()
print_sep("Convergence Table")

@printf "  %-6s  %-8s  %-12s  %-12s  %-8s  %-12s  %-8s\n" "level" "nv" "h" "err_upwind" "ord_up" "err_centered" "ord_cen"
println("  " * "─"^72)

for i in eachindex(levels)
    ord_up  = i == 1 ? "  —   " :
        @sprintf("%.3f", log(err_up[i-1]/err_up[i])   / log(hs[i-1]/hs[i]))
    ord_cen = i == 1 ? "  —   " :
        @sprintf("%.3f", log(err_cen[i-1]/err_cen[i]) / log(hs[i-1]/hs[i]))
    @printf "  %-6d  %-8d  %-12.4e  %-12.4e  %-8s  %-12.4e  %-8s\n" levels[i] nverts[i] hs[i] err_up[i] ord_up err_cen[i] ord_cen
end

println()
println("Expected convergence rates:")
println("  Upwind   (IMEX): ~1.0  (first-order; upwind diffusion ε_num ~ |v|h/2 dominates)")
println("  Centered (IMEX): ~2.0  (second-order; matches Laplace–Beltrami accuracy)")
println()
println("Advection–diffusion mesh convergence study complete.")
