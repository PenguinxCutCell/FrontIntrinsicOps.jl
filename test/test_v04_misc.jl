# test_v04_misc.jl – Miscellaneous v0.4 tests:
# high-res transport, open surfaces, caching, performance helpers.

# ─────────────────────────────────────────────────────────────────────────────
# High-resolution transport limiter tests
# ─────────────────────────────────────────────────────────────────────────────

@testset "Limiters: minmod" begin
    @test minmod(1.0, 2.0) ≈ 1.0
    @test minmod(2.0, 1.0) ≈ 1.0
    @test minmod(-1.0, -2.0) ≈ -1.0
    @test minmod(1.0, -1.0) == 0.0
    @test minmod(0.0, 1.0) == 0.0
end

@testset "Limiters: van Leer and Superbee" begin
    @test vanleer_limiter(0.0) == 0.0
    @test vanleer_limiter(-1.0) == 0.0
    @test vanleer_limiter(1.0) ≈ 1.0 atol=1e-10
    @test 0.0 <= vanleer_limiter(2.0) <= 2.0

    @test superbee_limiter(0.0) == 0.0
    @test superbee_limiter(-1.0) == 0.0
    @test superbee_limiter(0.5) ≈ 1.0
    @test superbee_limiter(2.0) ≈ 2.0
end

@testset "High-res transport: basic step" begin
    mesh = make_uvsphere(1.0; nφ=8, nθ=16)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    topo = build_topology(mesh)
    nv   = length(mesh.points)

    u0   = [p[3] for p in mesh.points]  # z-coord: smooth initial data
    # Velocity: rigid rotation about z-axis (tangential)
    nf   = length(mesh.faces)
    vel  = Vector{SVector{3,Float64}}(undef, nf)
    for fi in 1:nf
        cx = sum(mesh.points[mesh.faces[fi][k]][1] for k in 1:3) / 3
        cy = sum(mesh.points[mesh.faces[fi][k]][2] for k in 1:3) / 3
        vel[fi] = SVector{3,Float64}(-cy, cx, 0.0)
    end
    dt = 1e-3

    u1 = step_surface_transport_limited(mesh, geom, dec, topo, u0, vel, dt;
                                         limiter=:minmod, method=:euler)
    @test length(u1) == nv
    @test all(isfinite.(u1))

    # SSP-RK2
    u2 = step_surface_transport_limited(mesh, geom, dec, topo, u0, vel, dt;
                                         limiter=:vanleer, method=:ssprk2)
    @test all(isfinite.(u2))

    # SSP-RK3
    u3 = step_surface_transport_limited(mesh, geom, dec, topo, u0, vel, dt;
                                         limiter=:superbee, method=:ssprk3)
    @test all(isfinite.(u3))
end

@testset "High-res transport: upwind fallback" begin
    mesh = make_uvsphere(1.0; nφ=6, nθ=12)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    topo = build_topology(mesh)
    nv   = length(mesh.points)
    u0   = rand(Float64, nv)
    vel  = zeros(Float64, length(topo.edges))  # zero velocity
    dt   = 1e-3

    u1 = step_surface_transport_limited(mesh, geom, dec, topo, u0, vel, dt;
                                         limiter=:upwind, method=:euler)
    @test all(isfinite.(u1))
end

# ─────────────────────────────────────────────────────────────────────────────
# Open surface / boundary condition tests
# ─────────────────────────────────────────────────────────────────────────────

@testset "Open surface: flat patch boundary detection" begin
    mesh = make_flat_patch(; N=5, L=1.0)
    topo = build_topology(mesh)
    geom = compute_geometry(mesh)

    @test is_open_surface(topo)

    bv = detect_boundary_vertices(mesh, topo)
    be = detect_boundary_edges(topo)
    @test !isempty(bv)
    @test !isempty(be)
    # All boundary edge endpoints should be boundary vertices
    bv_set = Set(bv)
    for ei in be
        i, j = topo.edges[ei][1], topo.edges[ei][2]
        @test i in bv_set
        @test j in bv_set
    end
end

@testset "Open surface: apply_dirichlet!" begin
    nv = 10
    u  = zeros(Float64, nv)
    bv = [1, 3, 5]
    apply_dirichlet!(u, bv, 1.0)
    @test u[1] ≈ 1.0
    @test u[3] ≈ 1.0
    @test u[5] ≈ 1.0
    @test u[2] ≈ 0.0

    # With vector values
    apply_dirichlet!(u, bv, [2.0, 3.0, 4.0])
    @test u[1] ≈ 2.0
    @test u[3] ≈ 3.0
    @test u[5] ≈ 4.0
end

@testset "Open surface: Poisson with Dirichlet BCs" begin
    mesh = make_flat_patch(; N=8, L=1.0)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    topo = build_topology(mesh)
    nv   = length(mesh.points)

    bv  = detect_boundary_vertices(mesh, topo)
    # Dirichlet: u = x on boundary
    g   = [mesh.points[i][1] for i in bv]

    # Solve -ΔΓ u = 0 (Laplace equation)
    f = zeros(Float64, nv)
    u = solve_open_surface_poisson(mesh, geom, dec, topo, f, bv, g)
    @test length(u) == nv
    @test all(isfinite.(u))

    # Check boundary values are enforced
    for (k, vi) in enumerate(bv)
        @test abs(u[vi] - g[k]) < 1e-10
    end
end

@testset "Closed sphere: is_open_surface returns false" begin
    mesh = make_uvsphere(1.0; nφ=8, nθ=16)
    topo = build_topology(mesh)
    @test !is_open_surface(topo)
end

@testset "boundary_mass_matrix: flat patch" begin
    mesh = make_flat_patch(; N=4, L=1.0)
    geom = compute_geometry(mesh)
    topo = build_topology(mesh)

    Mb = boundary_mass_matrix(mesh, geom, topo)
    @test size(Mb, 1) == length(mesh.points)
    @test size(Mb, 2) == length(mesh.points)
    # Mb should be non-negative (all entries >= 0 for P1 on boundary)
    vals = nonzeros(Mb)
    @test all(vals .>= 0.0)
    # Total boundary length ≈ 4*L = 4
    total = sum(vals)
    # Sum of Mb diagonal ~ length of boundary ~ 4.0 (for L=1, N=4)
    @test abs(total - 4.0) < 0.5
end

# ─────────────────────────────────────────────────────────────────────────────
# Cache tests
# ─────────────────────────────────────────────────────────────────────────────

@testset "SurfacePDECache: basic construction" begin
    mesh = make_uvsphere(1.0; nφ=8, nθ=16)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    nv   = length(mesh.points)

    cache = build_pde_cache(mesh, geom, dec; μ=0.1, dt=0.01, θ=1.0)

    @test size(cache.laplace, 1) == nv
    @test size(cache.mass, 1)    == nv
    @test length(cache.mass_vec) == nv
    @test cache.diffusion_fac !== nothing
    @test cache.helmholtz_fac === nothing
end

@testset "SurfacePDECache: Helmholtz factorisation" begin
    mesh = make_uvsphere(1.0; nφ=8, nθ=16)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    cache = build_pde_cache(mesh, geom, dec; α_helmholtz=1.0)
    @test cache.helmholtz_fac !== nothing
    @test cache.diffusion_fac === nothing

    # Solve Helmholtz: (L + M) u = f
    nv = length(mesh.points)
    f  = ones(Float64, nv)
    u  = solve_helmholtz_cached(cache, f)
    @test length(u) == nv
    @test all(isfinite.(u))
end

@testset "step_diffusion_cached: consistency with step_surface_reaction_diffusion_imex" begin
    mesh = make_uvsphere(1.0; nφ=8, nθ=16)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    nv   = length(mesh.points)
    u0   = rand(Float64, nv)
    dt   = 0.01; μ = 0.1

    # IMEX step with no reaction (θ=1 backward Euler, strong-form (I + dt θ μ L) system)
    u1_std, _ = step_surface_reaction_diffusion_imex(mesh, geom, dec, u0, dt, μ, nothing, 0.0;
                                                      θ=1.0)

    # Cached step (uses same strong-form (I + dt θ μ L) system)
    cache = build_pde_cache(mesh, geom, dec; μ=μ, dt=dt, θ=1.0)
    u1_cached = step_diffusion_cached(cache, u0)

    @test maximum(abs.(u1_std .- u1_cached)) < 1e-12
end

@testset "CurvePDECache: construction and step" begin
    crv  = sample_circle(1.0, 64)
    geom = compute_geometry(crv)
    dec  = build_dec(crv, geom)
    nv   = length(crv.points)

    cache = build_pde_cache(crv, geom, dec; μ=0.01, dt=0.001, θ=1.0)
    @test size(cache.laplace, 1) == nv
    @test cache.diffusion_fac !== nothing

    u0 = rand(Float64, nv)
    u1 = step_diffusion_cached(cache, u0)
    @test length(u1) == nv
    @test all(isfinite.(u1))
end

# ─────────────────────────────────────────────────────────────────────────────
# Performance helper tests
# ─────────────────────────────────────────────────────────────────────────────

@testset "alloc_diffusion_buffers" begin
    buf = alloc_diffusion_buffers(100)
    @test length(buf.rhs) == 100
    @test length(buf.tmp) == 100
end

@testset "alloc_rd_buffers" begin
    buf = alloc_rd_buffers(50, Float64)
    @test length(buf.rhs)      == 50
    @test length(buf.reaction) == 50
    @test length(buf.tmp)      == 50
end

@testset "step_diffusion_inplace!" begin
    mesh = make_uvsphere(1.0; nφ=8, nθ=16)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    nv   = length(mesh.points)
    dt   = 0.01; μ = 0.1

    cache = build_pde_cache(mesh, geom, dec; μ=μ, dt=dt, θ=1.0)
    buf   = alloc_diffusion_buffers(nv)

    u0 = rand(Float64, nv)
    u  = copy(u0)

    # In-place step
    step_diffusion_inplace!(u, cache, buf)
    @test all(isfinite.(u))
    @test !all(u .≈ u0)  # solution changed
end

@testset "l2_norm_cached and energy_norm_cached" begin
    mesh = make_uvsphere(1.0; nφ=8, nθ=16)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    nv   = length(mesh.points)

    cache = build_pde_cache(mesh, geom, dec)
    u = ones(Float64, nv)

    l2 = l2_norm_cached(cache, u)
    @test l2 > 0.0
    @test isfinite(l2)
    # L² norm of constant 1 on unit sphere ≈ sqrt(4π)
    @test abs(l2 - sqrt(4π)) < 0.1

    en = energy_norm_cached(cache, u)
    @test en >= 0.0  # constant function has zero Laplacian -> energy norm ~ 0
    @test en < 1e-4  # numerical precision on coarse sphere mesh
end

@testset "apply_mass_inplace! and apply_laplace_inplace!" begin
    mesh = make_uvsphere(1.0; nφ=8, nθ=16)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    nv   = length(mesh.points)

    cache = build_pde_cache(mesh, geom, dec)
    u = rand(Float64, nv)
    y = zeros(Float64, nv)

    apply_mass_inplace!(y, cache, u)
    @test !all(y .== 0.0)

    apply_laplace_inplace!(y, cache, u)
    @test all(isfinite.(y))
end
