# examples/surface_transport_sphere.jl
#
# Rigid rotation of a scalar field on the unit sphere.
#
# Setup:
#   - Initial field:  u(0) = z  (north–south dipole)
#   - Velocity field: v(x,y,z) = (-y, x, 0)  (rotation around z-axis, ω=1 rad/s)
#   - Exact solution at time t: u(x,y,z,t) = z  (z is invariant under z-rotation)
#
# We compare the centered and upwind finite-volume schemes, and track:
#   - mass conservation  ∫ u dA
#   - L2 error versus the exact (invariant) solution
#
# Run:  julia --project examples/surface_transport_sphere.jl

using FrontIntrinsicOps
using StaticArrays
using LinearAlgebra
using Printf

# ── Mesh & geometry ──────────────────────────────────────────────────────────

R    = 1.0
mesh = generate_icosphere(R, 4)
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)

nv = length(mesh.points)
@printf "Icosphere level 4: %d vertices, %d faces\n" nv length(mesh.faces)
@printf "Surface area:      %.6f  (exact 4π ≈ %.6f)\n" measure(mesh, geom) 4π*R^2

# ── Initial condition and velocity field ─────────────────────────────────────

z   = Float64[p[3] for p in mesh.points]   # initial field
vel = SVector{3,Float64}[SVector{3,Float64}(-p[2], p[1], 0.0) for p in mesh.points]

# Because u=z is invariant under rotation about z, the exact solution is z itself.
u_exact = z

# ── CFL-stable time step ─────────────────────────────────────────────────────

dt   = estimate_transport_dt(mesh, geom, vel; cfl=0.3)
T_rot = π / 4          # rotate by 45°
nstep = max(1, round(Int, T_rot / dt))
dt    = T_rot / nstep

@printf "\nRotating by %.4f rad (%.1f°)\n" T_rot rad2deg(T_rot)
@printf "dt = %.6f,  nstep = %d\n" dt nstep

# Use L1 mass (∫|u| dA) for relative error to avoid near-zero denominator
M0_abs = dot(geom.vertex_dual_areas, abs.(z))   # reference scale
M0     = dot(geom.vertex_dual_areas, z)         # should be ≈ 0
@printf "Initial ∫u dA = %.4e (≈0),  ∫|u| dA = %.6f\n\n" M0 M0_abs

# ── Assemble transport operators ─────────────────────────────────────────────

A_centered = assemble_transport_operator(mesh, geom, vel; scheme=:centered)
A_upwind   = assemble_transport_operator(mesh, geom, vel; scheme=:upwind)

# ── Time-step loop ───────────────────────────────────────────────────────────

println("─── Centered scheme (SSP-RK3) ──────────────────────────────────")
@printf "  %-8s  %-14s  %-14s  %-14s\n" "step" "t" "mass_err_rel" "L2_error"

u_c = let u = copy(z)
    for k in 1:nstep
        u = step_surface_transport_ssprk3(mesh, geom, A_centered, u, dt)
        if k == 1 || k % max(1, nstep ÷ 5) == 0 || k == nstep
            t_k      = k * dt
            mass_err = abs(dot(geom.vertex_dual_areas, u) - M0) / M0_abs
            l2_err   = weighted_l2_error(mesh, geom, u, u_exact)
            @printf "  %-8d  %-14.6f  %-14.4e  %-14.4e\n" k t_k mass_err l2_err
        end
    end
    u
end

println()
println("─── Upwind scheme (SSP-RK3) ─────────────────────────────────────")
@printf "  %-8s  %-14s  %-14s  %-14s\n" "step" "t" "mass_err_rel" "L2_error"

u_u = let u = copy(z)
    for k in 1:nstep
        u = step_surface_transport_ssprk3(mesh, geom, A_upwind, u, dt)
        if k == 1 || k % max(1, nstep ÷ 5) == 0 || k == nstep
            t_k      = k * dt
            mass_err = abs(dot(geom.vertex_dual_areas, u) - M0) / M0_abs
            l2_err   = weighted_l2_error(mesh, geom, u, u_exact)
            @printf "  %-8d  %-14.6f  %-14.4e  %-14.4e\n" k t_k mass_err l2_err
        end
    end
    u
end

# ── Summary ───────────────────────────────────────────────────────────────────

println()
println("─── Final comparison ────────────────────────────────────────────")
@printf "%-20s  %-14s  %-14s  %-14s\n" "Scheme" "L2 error" "Linf error" "mass err rel"

for (label, u_fin) in [("Centered", u_c), ("Upwind", u_u)]
    l2   = weighted_l2_error(mesh, geom, u_fin, u_exact)
    linf = weighted_linf_error(mesh, geom, u_fin, u_exact)
    merr = abs(dot(geom.vertex_dual_areas, u_fin) - M0) / M0_abs
    @printf "%-20s  %-14.4e  %-14.4e  %-14.4e\n" label l2 linf merr
end

println()
println("Notes:")
println("  - Centered scheme conserves mass more closely but may be dispersive.")
println("  - Upwind scheme is dissipative but more stable; errors reflect numerical diffusion.")
println("  - Both schemes integrate the invariant field z with errors from mesh discretisation.")
println()
println("Surface transport example complete.")
