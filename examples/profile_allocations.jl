# examples/profile_allocations.jl
#
# Benchmark and allocation profiling for key FrontIntrinsicOps operations.
#
# Operations timed (with @time):
#   1. compute_geometry
#   2. build_dec
#   3. mean_curvature
#   4. gaussian_curvature
#   5. solve_surface_poisson (one solve)
#   6. One implicit diffusion step (backward Euler, fresh factorization)
#   7. One implicit diffusion step (backward Euler, reused factorization)
#   8. One explicit transport step (forward Euler)
#   9. One IMEX advection–diffusion step
#
# Run:  julia --project examples/profile_allocations.jl

using FrontIntrinsicOps
using StaticArrays
using LinearAlgebra
using Printf

println("="^60)
println("  FrontIntrinsicOps allocation / timing benchmark")
println("="^60)
println("Running on icosphere level 4 (production-size mesh)...")
println()

# ── Build mesh once (not timed) ──────────────────────────────────────────────

R    = 1.0
mesh = generate_icosphere(R, 4)
@printf "Icosphere level 4: %d vertices, %d faces\n" length(mesh.points) length(mesh.faces)
println()

# ── Helper: warm up then time ─────────────────────────────────────────────────

function bench(f::Function, label::String; warmup=1)
    for _ in 1:warmup; f(); end
    println("@time $label:")
    @time f()
    println()
end

# ── 1. compute_geometry ───────────────────────────────────────────────────────

bench("compute_geometry") do
    compute_geometry(mesh)
end

geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)

# ── 2. build_dec ─────────────────────────────────────────────────────────────

bench("build_dec") do
    build_dec(mesh, geom)
end

# ── 3. mean_curvature ────────────────────────────────────────────────────────

bench("mean_curvature") do
    mean_curvature(mesh, geom, dec)
end

# ── 4. gaussian_curvature ────────────────────────────────────────────────────

bench("gaussian_curvature") do
    gaussian_curvature(mesh, geom)
end

# ── 5. solve_surface_poisson ─────────────────────────────────────────────────

z = Float64[p[3] for p in mesh.points]

bench("solve_surface_poisson(z)") do
    solve_surface_poisson(mesh, geom, dec, z)
end

# ── 6. Implicit diffusion: first step (builds factorization) ─────────────────

μ  = 0.1
dt = 1e-3

bench("step_surface_diffusion_backward_euler (build factorization)") do
    step_surface_diffusion_backward_euler(mesh, geom, dec, z, dt, μ)
end

# ── 7. Implicit diffusion: reuse factorization ───────────────────────────────

_, fac = step_surface_diffusion_backward_euler(mesh, geom, dec, z, dt, μ)

bench("step_surface_diffusion_backward_euler (reuse factorization)") do
    step_surface_diffusion_backward_euler(mesh, geom, dec, z, dt, μ; factorization=fac)
end

# ── 8. Crank–Nicolson: first step ────────────────────────────────────────────

bench("step_surface_diffusion_crank_nicolson (build factorization)") do
    step_surface_diffusion_crank_nicolson(mesh, geom, dec, z, dt, μ)
end

# ── 9. Forward Euler transport ───────────────────────────────────────────────

vel   = SVector{3,Float64}[SVector{3,Float64}(-p[2], p[1], 0.0) for p in mesh.points]
A_upw = assemble_transport_operator(mesh, geom, vel; scheme=:upwind)
dt_t  = estimate_transport_dt(mesh, geom, vel; cfl=0.3)

bench("step_surface_transport_forward_euler (upwind)") do
    step_surface_transport_forward_euler(mesh, geom, A_upw, z, dt_t)
end

# ── 10. SSP-RK3 transport ────────────────────────────────────────────────────

bench("step_surface_transport_ssprk3 (upwind)") do
    step_surface_transport_ssprk3(mesh, geom, A_upw, z, dt_t)
end

# ── 11. IMEX advection–diffusion ─────────────────────────────────────────────

bench("step_surface_advection_diffusion_imex (build fac)") do
    step_surface_advection_diffusion_imex(mesh, geom, dec, z, vel, dt_t, μ;
                                          scheme=:upwind)
end

_, fac_imex = step_surface_advection_diffusion_imex(mesh, geom, dec, z, vel, dt_t, μ;
                                                     scheme=:upwind)

bench("step_surface_advection_diffusion_imex (reuse fac)") do
    step_surface_advection_diffusion_imex(mesh, geom, dec, z, vel, dt_t, μ;
                                          scheme=:upwind, factorization=fac_imex)
end

println("Profile benchmark complete.")
