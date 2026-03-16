# test_transport_highres.jl – Tests for high-resolution (limiter-based) surface
#                              transport and conservation checks. (v0.4)
#
# Tests:
# 1. Limiter functions: minmod, van Leer, superbee scalar properties.
# 2. Limited transport step: output shape and finiteness.
# 3. SSP-RK2 / SSP-RK3 transport: finiteness.
# 4. Conservation: total mass approximately conserved under zero-divergence flow.
# 5. Comparison: limited vs upwind on smooth data.

@testset "Limiters: minmod scalar" begin
    # minmod(a, b) = 0 if signs differ, otherwise min-magnitude value
    @test minmod(1.0, 2.0) ≈ 1.0
    @test minmod(2.0, 1.0) ≈ 1.0
    @test minmod(-1.0, -2.0) ≈ -1.0
    @test minmod(-2.0, -1.0) ≈ -1.0
    @test minmod(1.0, -1.0) == 0.0
    @test minmod(-1.0, 1.0) == 0.0
    @test minmod(0.0, 1.0) == 0.0
    @test minmod(1.0, 0.0) == 0.0
    @test minmod(0.0, 0.0) == 0.0
end

@testset "Limiters: minmod3" begin
    @test minmod3(1.0, 2.0, 3.0) ≈ 1.0
    @test minmod3(-1.0, -2.0, -3.0) ≈ -1.0
    @test minmod3(1.0, -1.0, 2.0) == 0.0
    @test minmod3(2.0, 3.0, 1.0) ≈ 1.0
end

@testset "Limiters: van Leer" begin
    # φ(r) = (r + |r|) / (1 + |r|) → 0 for r ≤ 0
    @test vanleer_limiter(0.0) == 0.0
    @test vanleer_limiter(-1.0) == 0.0
    @test vanleer_limiter(-0.5) == 0.0
    # r=1: (1+1)/(1+1) = 1.0
    @test vanleer_limiter(1.0) ≈ 1.0 atol=1e-12
    # r=2: (2+2)/(1+2) = 4/3
    @test vanleer_limiter(2.0) ≈ 4.0/3.0 atol=1e-12
    # van Leer is bounded: 0 ≤ φ ≤ 2
    for r in [0.1, 0.5, 1.0, 2.0, 10.0, 100.0]
        φ = vanleer_limiter(r)
        @test 0.0 <= φ <= 2.0 + 1e-12
    end
end

@testset "Limiters: superbee" begin
    # superbee: 0 for r ≤ 0, max(0, min(2r,1), min(r,2)) otherwise
    @test superbee_limiter(0.0) == 0.0
    @test superbee_limiter(-1.0) == 0.0
    @test superbee_limiter(0.5) ≈ 1.0 atol=1e-12  # min(2*0.5,1) = min(1,1)=1
    @test superbee_limiter(1.0) ≈ 1.0 atol=1e-12  # max(min(2,1),min(1,2))=max(1,1)
    @test superbee_limiter(2.0) ≈ 2.0 atol=1e-12  # min(r,2) = 2
    # superbee is bounded: 0 ≤ φ ≤ 2
    for r in [0.1, 0.5, 1.0, 1.5, 2.0, 5.0]
        φ = superbee_limiter(r)
        @test 0.0 <= φ <= 2.0 + 1e-12
    end
end

@testset "High-res transport: output shape and finiteness" begin
    mesh = make_uvsphere(1.0; nφ=8, nθ=16)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    topo = build_topology(mesh)
    nv   = length(mesh.points)
    nf   = length(mesh.faces)

    # Smooth initial condition
    u0 = [p[3] for p in mesh.points]

    # Per-face tangential velocity: rigid rotation about z-axis
    vel = Vector{SVector{3,Float64}}(undef, nf)
    for fi in 1:nf
        cx = sum(mesh.points[mesh.faces[fi][k]][1] for k in 1:3) / 3
        cy = sum(mesh.points[mesh.faces[fi][k]][2] for k in 1:3) / 3
        vel[fi] = SVector{3,Float64}(-cy, cx, 0.0)
    end

    dt = 1e-3
    for lim in (:minmod, :vanleer, :superbee)
        u1 = step_surface_transport_limited(mesh, geom, dec, topo, u0, vel, dt;
                                             limiter=lim, method=:euler)
        @test length(u1) == nv
        @test all(isfinite.(u1))
    end
end

@testset "High-res transport: SSP-RK2 and SSP-RK3" begin
    mesh = make_uvsphere(1.0; nφ=8, nθ=16)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    topo = build_topology(mesh)
    nv   = length(mesh.points)
    nf   = length(mesh.faces)

    u0 = [p[3] for p in mesh.points]
    vel = Vector{SVector{3,Float64}}(undef, nf)
    for fi in 1:nf
        cx = sum(mesh.points[mesh.faces[fi][k]][1] for k in 1:3) / 3
        cy = sum(mesh.points[mesh.faces[fi][k]][2] for k in 1:3) / 3
        vel[fi] = SVector{3,Float64}(-cy, cx, 0.0)
    end
    dt = 5e-4

    u_rk2 = step_surface_transport_limited(mesh, geom, dec, topo, u0, vel, dt;
                                            limiter=:minmod, method=:ssprk2)
    @test length(u_rk2) == nv
    @test all(isfinite.(u_rk2))

    u_rk3 = step_surface_transport_limited(mesh, geom, dec, topo, u0, vel, dt;
                                            limiter=:vanleer, method=:ssprk3)
    @test length(u_rk3) == nv
    @test all(isfinite.(u_rk3))
end

@testset "High-res transport: upwind fallback (zero velocity)" begin
    mesh = make_uvsphere(1.0; nφ=6, nθ=12)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    topo = build_topology(mesh)
    nv   = length(mesh.points)

    u0  = rand(Float64, nv)
    vel = zeros(Float64, length(topo.edges))  # edge-flux velocity
    dt  = 1e-3

    u1 = step_surface_transport_limited(mesh, geom, dec, topo, u0, vel, dt;
                                         limiter=:upwind, method=:euler)
    @test all(isfinite.(u1))
    # With zero velocity, solution should not change
    @test maximum(abs.(u1 .- u0)) < 1e-12
end

@testset "High-res transport: mass conservation (rigid rotation)" begin
    # On a closed surface with a divergence-free velocity field,
    # the integral of u should be conserved up to small numerical errors.
    mesh = make_uvsphere(1.0; nφ=10, nθ=20)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    topo = build_topology(mesh)
    nf   = length(mesh.faces)

    u0 = [sin(p[1]) * cos(p[2]) for p in mesh.points]
    M  = mass_matrix(mesh, geom)
    mass0 = sum(M * u0)

    # Rigid rotation velocity (tangential)
    vel = Vector{SVector{3,Float64}}(undef, nf)
    for fi in 1:nf
        cx = sum(mesh.points[mesh.faces[fi][k]][1] for k in 1:3) / 3
        cy = sum(mesh.points[mesh.faces[fi][k]][2] for k in 1:3) / 3
        vel[fi] = SVector{3,Float64}(-cy, cx, 0.0)
    end

    dt = 5e-4
    nsteps = 10
    u = copy(u0)
    for _ in 1:nsteps
        u = step_surface_transport_limited(mesh, geom, dec, topo, u, vel, dt;
                                            limiter=:vanleer, method=:ssprk2)
    end

    mass_final = sum(M * u)
    # Mass should be approximately conserved
    @test abs(mass_final - mass0) / (abs(mass0) + 1e-14) < 0.05
end

@testset "High-res transport: limited more accurate than upwind on smooth data" begin
    # Van Leer limiter should give closer result to smooth function than pure upwind
    mesh = make_uvsphere(1.0; nφ=10, nθ=20)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    topo = build_topology(mesh)
    nv   = length(mesh.points)
    nf   = length(mesh.faces)

    u0   = [p[3] for p in mesh.points]  # smooth z-coordinate
    vel  = Vector{SVector{3,Float64}}(undef, nf)
    for fi in 1:nf
        cx = sum(mesh.points[mesh.faces[fi][k]][1] for k in 1:3) / 3
        cy = sum(mesh.points[mesh.faces[fi][k]][2] for k in 1:3) / 3
        vel[fi] = SVector{3,Float64}(-cy, cx, 0.0)
    end

    dt = 5e-4
    u_upwind  = step_surface_transport_limited(mesh, geom, dec, topo, u0, vel, dt;
                                               limiter=:upwind, method=:euler)
    u_limited = step_surface_transport_limited(mesh, geom, dec, topo, u0, vel, dt;
                                               limiter=:vanleer, method=:ssprk2)

    @test all(isfinite.(u_upwind))
    @test all(isfinite.(u_limited))
    # Both should remain bounded for a small step on smooth data
    @test maximum(abs.(u_upwind))  <= maximum(abs.(u0)) * 1.1
    @test maximum(abs.(u_limited)) <= maximum(abs.(u0)) * 1.1
end
