# convergence_v04/transport_torus.jl
#
# Convergence study: scalar transport on the torus via toroidal rotation.
#
# We transport the initial field u₀ = cos(θ) (toroidal angle θ) by a
# prescribed rigid toroidal rotation velocity field v = ω (-y, x, 0).
# The exact solution is: u(t) = cos(θ - ω t).
#
# We measure the L² error at T_end for increasing spatial resolutions.
#
# Run:  julia --project=.. transport_torus.jl

include(joinpath(@__DIR__, "common.jl"))

# ── Parameters ────────────────────────────────────────────────────────────────

R     = 2.0          # major radius
r     = 0.5          # minor radius
ω     = 0.5          # angular velocity of rotation
T_end = 0.04
dt    = 2e-4
nstep = round(Int, T_end / dt)

print_header("Convergence: Scalar Transport on Torus (Toroidal Rotation)")
@printf "  R=%.2f, r=%.2f, ω=%.2f, T=%.3f, dt=%.2e, nstep=%d\n" R r ω T_end dt nstep
@printf "  Scheme: SSP-RK2, van Leer limiter\n"
@printf "  IC: u₀ = cos(θ); exact u(t) = cos(θ − ω t)\n\n"

# ── Mesh refinement loop ──────────────────────────────────────────────────────

ntheta_list = [10, 20, 30, 40]
hs     = Float64[]
errors = Float64[]
nverts = Int[]

for nθ in ntheta_list
    nφ   = max(4, round(Int, nθ * r / R))
    mesh = generate_torus(R, r, nθ, nφ)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    topo = build_topology(mesh)

    h  = mesh_size_surface(mesh, geom)
    nv = length(mesh.points)
    nf = length(mesh.faces)

    # Initial condition
    u0 = Float64[cos(atan(p[2], p[1])) for p in mesh.points]

    # Face-centered velocity: toroidal rotation v = ω (-y, x, 0)
    vel = Vector{SVector{3,Float64}}(undef, nf)
    for fi in 1:nf
        cx = sum(mesh.points[mesh.faces[fi][k]][1] for k in 1:3) / 3
        cy = sum(mesh.points[mesh.faces[fi][k]][2] for k in 1:3) / 3
        vel[fi] = SVector{3,Float64}(ω * (-cy), ω * cx, 0.0)
    end

    # Time integration
    u = copy(u0)
    for _ in 1:nstep
        u = step_surface_transport_limited(mesh, geom, dec, topo, u, vel, dt;
                                            limiter=:vanleer, method=:ssprk2)
    end

    # Exact solution at T_end
    u_exact = Float64[cos(atan(p[2], p[1]) - ω * T_end) for p in mesh.points]
    err = weighted_l2_error(mesh, geom, u, u_exact)

    push!(hs, h); push!(errors, err); push!(nverts, nv)
    @printf "  nθ=%3d, nφ=%2d, nv=%5d, h=%.4e, L2_err=%.4e\n" nθ nφ nv h err
end

# ── Convergence table ─────────────────────────────────────────────────────────

print_convergence_table(hs, errors; header="Spatial Convergence", label="L2 error")
println("Expected rate: ~1-2 (limited by advection scheme and toroidal geometry)")
println()
println("Transport torus convergence study complete.")
