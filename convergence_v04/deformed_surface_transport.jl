# convergence_v04/deformed_surface_transport.jl
#
# Convergence study: scalar transport on deformed closed surfaces.
#
# We verify that the high-resolution transport scheme remains stable and
# bounded on deformed (non-spherical) surfaces under mesh refinement.
#
# Test surfaces:
#   A. Ellipsoid (a=2, b=1.5, c=1) with a prescribed tangential rotation flow
#   B. Perturbed sphere (R=1, ε=0.15, k=2)
#
# For each surface we run N steps of SSP-RK2 transport and check:
#   - Solution remains bounded (no blow-up)
#   - Conservation of mass under divergence-free flow
#
# Run:  julia --project=.. deformed_surface_transport.jl

include(joinpath(@__DIR__, "common.jl"))

function run_transport_study(mesh, geom, dec, topo, label)
    nv = length(mesh.points)
    nf = length(mesh.faces)

    # Initial condition: smooth function u₀ = z
    u0 = Float64[p[3] for p in mesh.points]

    # Face-centered tangential velocity: rotation about z-axis (projected)
    vel = Vector{SVector{3,Float64}}(undef, nf)
    for fi in 1:nf
        cx = sum(mesh.points[mesh.faces[fi][k]][1] for k in 1:3) / 3
        cy = sum(mesh.points[mesh.faces[fi][k]][2] for k in 1:3) / 3
        vel[fi] = SVector{3,Float64}(-cy, cx, 0.0)
    end

    dt    = 2e-3
    nstep = 20
    M     = mass_matrix(mesh, geom)
    mass0 = sum(M * u0)

    u = copy(u0)
    for _ in 1:nstep
        u = step_surface_transport_limited(mesh, geom, dec, topo, u, vel, dt;
                                            limiter=:vanleer, method=:ssprk2)
    end

    mass_final = sum(M * u)
    mass_err   = abs(mass_final - mass0) / (abs(mass0) + 1e-14)
    max_val    = maximum(abs.(u))
    max_init   = maximum(abs.(u0))

    @printf "  %-30s  nv=%5d  max|u|=%.4e  mass_err=%.2e  finite=%s\n" \
        label nv max_val mass_err string(all(isfinite.(u)))

    return all(isfinite.(u)), mass_err, max_val, max_init
end

# ─────────────────────────────────────────────────────────────────────────────
# A. Ellipsoid
# ─────────────────────────────────────────────────────────────────────────────

print_header("Deformed Surface Transport: Ellipsoid")

a, b, c = 2.0, 1.5, 1.0
resolutions = [(8, 16), (12, 24), (16, 32), (20, 40)]

println("  Rotating flow, SSP-RK2, van Leer limiter")
println()

for (nφ, nθ) in resolutions
    mesh = generate_ellipsoid(a, b, c, nφ, nθ)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    topo = build_topology(mesh)
    label = @sprintf("ellipsoid nφ=%2d nθ=%2d", nφ, nθ)
    run_transport_study(mesh, geom, dec, topo, label)
end

println()

# ─────────────────────────────────────────────────────────────────────────────
# B. Perturbed sphere
# ─────────────────────────────────────────────────────────────────────────────

print_header("Deformed Surface Transport: Perturbed Sphere (ε=0.15, k=2)")

R, ε, k = 1.0, 0.15, 2

println("  Rotating flow, SSP-RK2, van Leer limiter")
println()

for (nφ, nθ) in resolutions
    mesh = generate_perturbed_sphere(R, ε, k, nφ, nθ)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    topo = build_topology(mesh)
    label = @sprintf("perturbed nφ=%2d nθ=%2d", nφ, nθ)
    run_transport_study(mesh, geom, dec, topo, label)
end

println()

# ─────────────────────────────────────────────────────────────────────────────
# C. Torus
# ─────────────────────────────────────────────────────────────────────────────

print_header("Deformed Surface Transport: Torus (R=2, r=0.5)")

R, r = 2.0, 0.5

println("  Toroidal rotation, SSP-RK2, van Leer limiter")
println()

torus_resolutions = [(10, 5), (20, 8), (30, 12), (40, 16)]

for (nθ, nφ) in torus_resolutions
    mesh = generate_torus(R, r, nθ, nφ)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    topo = build_topology(mesh)
    label = @sprintf("torus nθ=%2d nφ=%2d", nθ, nφ)
    run_transport_study(mesh, geom, dec, topo, label)
end

println()
println("Transport on deformed surfaces study complete.")
println("All cases should be finite and mass-conserving to within numerical precision.")
