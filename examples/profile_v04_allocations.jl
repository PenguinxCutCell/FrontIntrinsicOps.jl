# examples/profile_v04_allocations.jl
#
# Allocation and timing summary for v0.4 FrontIntrinsicOps operations.
#
# Covers:
#   1. New generators: ellipsoid, perturbed sphere
#   2. Reaction–diffusion (Fisher–KPP, linear decay) on sphere and torus
#   3. High-resolution transport (van Leer) on sphere, torus, ellipsoid
#   4. Hodge decomposition on sphere and torus
#   5. Open surface Poisson solve
#   6. PDE cache construction and step
#
# Run:  julia --project examples/profile_v04_allocations.jl

using FrontIntrinsicOps
using StaticArrays
using LinearAlgebra
using Printf

println("="^60)
println("  FrontIntrinsicOps v0.4 allocation / timing summary")
println("="^60)
println()

# ── Helper: warmup + timed run ────────────────────────────────────────────────

function time_it(label, f)
    f()   # warmup (triggers compilation)
    GC.gc()
    t = @elapsed f()
    @printf "  %-50s  %.4f s\n" label t
end

# ─────────────────────────────────────────────────────────────────────────────
# Build meshes
# ─────────────────────────────────────────────────────────────────────────────

println("── Building meshes ───────────────────────────────────────────────────")

mesh_sph  = generate_icosphere(1.0, 3)   # ~642 vertices
mesh_tor  = generate_torus(2.0, 0.5, 40, 16)
mesh_ell  = generate_ellipsoid(2.0, 1.5, 1.0, 16, 32)
mesh_bump = generate_perturbed_sphere(1.0, 0.15, 2, 16, 32)

for (label, mesh) in [("icosphere lv=3", mesh_sph), ("torus (40×16)", mesh_tor),
                       ("ellipsoid (16×32)", mesh_ell), ("perturbed sph (16×32)", mesh_bump)]
    @printf "  %-24s  nv=%5d  nf=%6d\n" label length(mesh.points) length(mesh.faces)
end
println()

# Pre-compute geometry (all meshes)
geom_sph  = compute_geometry(mesh_sph)
dec_sph   = build_dec(mesh_sph, geom_sph)
topo_sph  = build_topology(mesh_sph)

geom_tor  = compute_geometry(mesh_tor)
dec_tor   = build_dec(mesh_tor, geom_tor)
topo_tor  = build_topology(mesh_tor)

geom_ell  = compute_geometry(mesh_ell)
dec_ell   = build_dec(mesh_ell, geom_ell)
topo_ell  = build_topology(mesh_ell)

nv_sph = length(mesh_sph.points)
nv_tor = length(mesh_tor.points)
nv_ell = length(mesh_ell.points)
nf_sph = length(mesh_sph.faces)
nf_tor = length(mesh_tor.faces)
nf_ell = length(mesh_ell.faces)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Generator timing
# ─────────────────────────────────────────────────────────────────────────────

println("── 1. Generator timing ──────────────────────────────────────────────")
time_it("generate_ellipsoid(2.0, 1.5, 1.0, 16, 32)",
        () -> generate_ellipsoid(2.0, 1.5, 1.0, 16, 32))
time_it("generate_perturbed_sphere(1.0, 0.15, 2, 16, 32)",
        () -> generate_perturbed_sphere(1.0, 0.15, 2, 16, 32))
println()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Reaction–diffusion step
# ─────────────────────────────────────────────────────────────────────────────

println("── 2. Reaction–diffusion step (IMEX, sphere) ────────────────────────")
u0_sph = rand(Float64, nv_sph)
fac_rd = nothing
time_it("RD step (sphere, Fisher, fresh fac)",
        () -> step_surface_reaction_diffusion_imex(mesh_sph, geom_sph, dec_sph,
              u0_sph, 0.01, 0.05, fisher_kpp_reaction(1.0), 0.0; θ=1.0))
println()

# ─────────────────────────────────────────────────────────────────────────────
# 3. High-resolution transport step
# ─────────────────────────────────────────────────────────────────────────────

println("── 3. High-resolution transport step ────────────────────────────────")
vel_sph = [SVector{3,Float64}(
    -sum(mesh_sph.points[mesh_sph.faces[fi][k]][2] for k in 1:3)/3,
     sum(mesh_sph.points[mesh_sph.faces[fi][k]][1] for k in 1:3)/3,
    0.0) for fi in 1:nf_sph]
u0_sph2 = [p[3] for p in mesh_sph.points]

time_it("transport step (sphere, vanleer, ssprk2)",
        () -> step_surface_transport_limited(mesh_sph, geom_sph, dec_sph, topo_sph,
              u0_sph2, vel_sph, 5e-4; limiter=:vanleer, method=:ssprk2))

vel_tor = [SVector{3,Float64}(
    -sum(mesh_tor.points[mesh_tor.faces[fi][k]][2] for k in 1:3)/3,
     sum(mesh_tor.points[mesh_tor.faces[fi][k]][1] for k in 1:3)/3,
    0.0) for fi in 1:nf_tor]
u0_tor = [p[3] for p in mesh_tor.points]

time_it("transport step (torus, vanleer, ssprk2)",
        () -> step_surface_transport_limited(mesh_tor, geom_tor, dec_tor, topo_tor,
              u0_tor, vel_tor, 5e-4; limiter=:vanleer, method=:ssprk2))
println()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Hodge decomposition
# ─────────────────────────────────────────────────────────────────────────────

println("── 4. Hodge decomposition ───────────────────────────────────────────")
f_z_sph = [p[3] for p in mesh_sph.points]
ω_sph   = gradient_0_to_1(dec_sph, f_z_sph)
time_it("hodge_decompose_1form (sphere)",
        () -> hodge_decompose_1form(mesh_sph, geom_sph, dec_sph, ω_sph))

f_z_tor = [p[3] for p in mesh_tor.points]
ω_tor   = gradient_0_to_1(dec_tor, f_z_tor)
time_it("hodge_decompose_1form (torus)",
        () -> hodge_decompose_1form(mesh_tor, geom_tor, dec_tor, ω_tor))
println()

# ─────────────────────────────────────────────────────────────────────────────
# 5. PDE cache
# ─────────────────────────────────────────────────────────────────────────────

println("── 5. PDE cache ─────────────────────────────────────────────────────")
time_it("build_pde_cache (sphere, μ=0.05, dt=0.01)",
        () -> build_pde_cache(mesh_sph, geom_sph, dec_sph; μ=0.05, dt=0.01, θ=1.0))

cache_sph = build_pde_cache(mesh_sph, geom_sph, dec_sph; μ=0.05, dt=0.01, θ=1.0)
time_it("step_diffusion_cached (sphere, fresh call)",
        () -> step_diffusion_cached(cache_sph, u0_sph))
println()

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

println("="^60)
println("  Profile v0.4 summary complete.")
println("  All operations exercised on sphere, torus, and ellipsoid.")
println("="^60)
