# convergence_pdes/transport_sphere.jl
#
# Spatial convergence study for rigid rotation on the unit sphere.
#
# Setup:
#   - Velocity:  v(x,y,z) = (-y, x, 0)  (rigid rotation, ω=1 rad/s)
#   - Initial:   u(0) = z  (dipole, invariant under z-rotation)
#   - Exact:     u(t) = z  for all t  (z is invariant under z-rotation)
#
# We fix CFL=0.3, rotate by π/4, and refine the spatial mesh (levels 1..3).
# Centered vs upwind scheme are compared.
# Mass conservation error is also reported.
#
# Run:  julia --project=.. transport_sphere.jl

include(joinpath(@__DIR__, "common.jl"))

# ── Parameters ────────────────────────────────────────────────────────────────

R       = 1.0
T_rot   = π / 4       # rotate by 45°
cfl     = 0.3

print_header("Spatial Convergence: Rigid Rotation on Unit Sphere")
@printf "  Rotation angle: π/4 (%.4f rad)\n" T_rot
@printf "  CFL = %.2f, Scheme: SSP-RK3\n\n" cfl

# ── Convergence loop ──────────────────────────────────────────────────────────

levels         = 1:3
hs             = Float64[]
errors_centered = Float64[]
errors_upwind  = Float64[]
mass_errs      = Float64[]
nverts         = Int[]
nsteps_vec     = Int[]

for level in levels
    mesh = generate_icosphere(R, level)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    h  = mesh_size_surface(mesh, geom)
    nv = length(mesh.points)

    z   = Float64[p[3] for p in mesh.points]
    vel = SVector{3,Float64}[SVector{3,Float64}(-p[2], p[1], 0.0) for p in mesh.points]

    dt    = estimate_transport_dt(mesh, geom, vel; cfl=cfl)
    nstep = max(1, round(Int, T_rot / dt))
    dt    = T_rot / nstep

    A_centered = assemble_transport_operator(mesh, geom, vel; scheme=:centered)
    A_upwind   = assemble_transport_operator(mesh, geom, vel; scheme=:upwind)

    M0     = dot(geom.vertex_dual_areas, z)        # ≈ 0 for z on sphere
    M0_abs = dot(geom.vertex_dual_areas, abs.(z))  # use as mass scale

    u_c = let u = copy(z)
        for _ in 1:nstep
            u = step_surface_transport_ssprk3(mesh, geom, A_centered, u, dt)
        end
        u
    end

    u_u = let u = copy(z)
        for _ in 1:nstep
            u = step_surface_transport_ssprk3(mesh, geom, A_upwind, u, dt)
        end
        u
    end

    # Exact solution is z (invariant field)
    u_exact = z

    err_c = weighted_l2_error(mesh, geom, u_c, u_exact)
    err_u = weighted_l2_error(mesh, geom, u_u, u_exact)
    merr  = abs(dot(geom.vertex_dual_areas, u_c) - M0) / M0_abs

    push!(hs,              h)
    push!(errors_centered, err_c)
    push!(errors_upwind,   err_u)
    push!(mass_errs,       merr)
    push!(nverts,          nv)
    push!(nsteps_vec,      nstep)

    @printf "  level=%d  nv=%5d  h=%.4e  nstep=%4d  err_c=%.4e  err_u=%.4e  mass_err=%.4e\n" level nv h nstep err_c err_u merr
end

# ── Convergence table ─────────────────────────────────────────────────────────

println()
print_sep("Convergence Table")

@printf "  %-6s  %-8s  %-12s  %-12s  %-8s  %-12s  %-8s  %-12s\n" "level" "nv" "h" "err_centered" "ord_c" "err_upwind" "ord_u" "mass_err"
println("  " * "─"^90)

n = length(levels)
for i in 1:n
    ord_c = i == 1 ? "  —   " :
        @sprintf("%.3f", log(errors_centered[i-1]/errors_centered[i]) / log(hs[i-1]/hs[i]))
    ord_u = i == 1 ? "  —   " :
        @sprintf("%.3f", log(errors_upwind[i-1]/errors_upwind[i]) / log(hs[i-1]/hs[i]))
    @printf "  %-6d  %-8d  %-12.4e  %-12.4e  %-8s  %-12.4e  %-8s  %-12.4e\n" levels[i] nverts[i] hs[i] errors_centered[i] ord_c errors_upwind[i] ord_u mass_errs[i]
end

println()
println("Expected convergence rates:")
println("  Centered (SSP-RK3): ~2.0 (second-order for smooth fields)")
println("  Upwind   (SSP-RK3): ~1.0 (first-order, numerically dissipative)")
println("Note: z-field is rotation-invariant so error reflects only spatial discretisation.")
println()
println("Transport convergence study complete.")
