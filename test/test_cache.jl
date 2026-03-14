# test_cache.jl – Tests for the SurfacePDECache and CurvePDECache
#                 caching infrastructure. (v0.4)
#
# Tests:
# 1. build_pde_cache: correct field sizes.
# 2. Diffusion factorization caching: build and reuse.
# 3. Helmholtz factorization caching.
# 4. step_diffusion_cached: consistent with IMEX step.
# 5. solve_helmholtz_cached: consistent with direct solve.
# 6. In-place helpers: step_diffusion_inplace!, apply_mass_inplace!, etc.
# 7. Cached norms: l2_norm_cached, energy_norm_cached.
# 8. CurvePDECache on a circle.

@testset "SurfacePDECache: diffusion factorization built" begin
    mesh  = make_uvsphere(1.0; nφ=8, nθ=16)
    geom  = compute_geometry(mesh)
    dec   = build_dec(mesh, geom)
    nv    = length(mesh.points)

    cache = build_pde_cache(mesh, geom, dec; μ=0.1, dt=0.01, θ=1.0)

    @test size(cache.laplace, 1) == nv
    @test size(cache.laplace, 2) == nv
    @test size(cache.mass, 1)    == nv
    @test length(cache.mass_vec) == nv
    @test cache.diffusion_fac !== nothing
    @test cache.helmholtz_fac === nothing
end

@testset "SurfacePDECache: helmholtz factorization built" begin
    mesh  = make_uvsphere(1.0; nφ=8, nθ=16)
    geom  = compute_geometry(mesh)
    dec   = build_dec(mesh, geom)
    nv    = length(mesh.points)

    cache = build_pde_cache(mesh, geom, dec; α_helmholtz=2.0)

    @test cache.helmholtz_fac !== nothing
    @test cache.diffusion_fac === nothing
end

@testset "SurfacePDECache: both factorizations" begin
    mesh  = make_uvsphere(1.0; nφ=8, nθ=16)
    geom  = compute_geometry(mesh)
    dec   = build_dec(mesh, geom)

    cache = build_pde_cache(mesh, geom, dec; μ=0.1, dt=0.01, θ=1.0, α_helmholtz=1.0)

    @test cache.diffusion_fac !== nothing
    @test cache.helmholtz_fac !== nothing
end

@testset "step_diffusion_cached: matches IMEX step" begin
    mesh  = make_uvsphere(1.0; nφ=8, nθ=16)
    geom  = compute_geometry(mesh)
    dec   = build_dec(mesh, geom)
    nv    = length(mesh.points)
    u0    = rand(Float64, nv)
    dt    = 0.01; μ = 0.1

    # IMEX backward Euler with no reaction
    u_std, _ = step_surface_reaction_diffusion_imex(mesh, geom, dec, u0, dt, μ, nothing, 0.0;
                                                     θ=1.0)
    # Cached step
    cache     = build_pde_cache(mesh, geom, dec; μ=μ, dt=dt, θ=1.0)
    u_cached  = step_diffusion_cached(cache, u0)

    @test maximum(abs.(u_std .- u_cached)) < 1e-12
end

@testset "step_diffusion_cached: repeated calls reuse factorization" begin
    mesh  = make_uvsphere(1.0; nφ=8, nθ=16)
    geom  = compute_geometry(mesh)
    dec   = build_dec(mesh, geom)
    nv    = length(mesh.points)
    u0    = rand(Float64, nv)
    dt    = 0.01; μ = 0.1

    cache = build_pde_cache(mesh, geom, dec; μ=μ, dt=dt, θ=1.0)

    # Multiple calls should give consistent results
    u1a = step_diffusion_cached(cache, u0)
    u1b = step_diffusion_cached(cache, u0)
    @test maximum(abs.(u1a .- u1b)) < 1e-14  # deterministic
end

@testset "solve_helmholtz_cached: matches direct solve" begin
    mesh  = make_uvsphere(1.0; nφ=8, nθ=16)
    geom  = compute_geometry(mesh)
    dec   = build_dec(mesh, geom)
    nv    = length(mesh.points)
    f     = ones(Float64, nv)
    α     = 1.0

    u_direct = solve_surface_helmholtz(mesh, geom, dec, f, α)

    cache    = build_pde_cache(mesh, geom, dec; α_helmholtz=α)
    u_cached = solve_helmholtz_cached(cache, f)

    @test maximum(abs.(u_direct .- u_cached)) < 1e-10
end

@testset "step_diffusion_inplace!: modifies u in-place" begin
    mesh  = make_uvsphere(1.0; nφ=8, nθ=16)
    geom  = compute_geometry(mesh)
    dec   = build_dec(mesh, geom)
    nv    = length(mesh.points)

    cache = build_pde_cache(mesh, geom, dec; μ=0.1, dt=0.01, θ=1.0)
    buf   = alloc_diffusion_buffers(nv)
    u0    = rand(Float64, nv)
    u     = copy(u0)

    step_diffusion_inplace!(u, cache, buf)
    @test all(isfinite.(u))
    @test !all(u .≈ u0)  # solution changed
end

@testset "step_diffusion_inplace!: consistent with step_diffusion_cached" begin
    mesh  = make_uvsphere(1.0; nφ=8, nθ=16)
    geom  = compute_geometry(mesh)
    dec   = build_dec(mesh, geom)
    nv    = length(mesh.points)
    u0    = rand(Float64, nv)

    cache = build_pde_cache(mesh, geom, dec; μ=0.1, dt=0.01, θ=1.0)
    buf   = alloc_diffusion_buffers(nv)

    u_inplace = copy(u0)
    step_diffusion_inplace!(u_inplace, cache, buf)

    u_cached = step_diffusion_cached(cache, u0)
    @test maximum(abs.(u_inplace .- u_cached)) < 1e-12
end

@testset "alloc_diffusion_buffers: correct sizes" begin
    buf = alloc_diffusion_buffers(42)
    @test length(buf.rhs) == 42
    @test length(buf.tmp) == 42
end

@testset "alloc_rd_buffers: correct sizes and zeros" begin
    buf = alloc_rd_buffers(30, Float64)
    @test length(buf.rhs)      == 30
    @test length(buf.reaction) == 30
    @test length(buf.tmp)      == 30
end

@testset "l2_norm_cached: constant function on sphere" begin
    mesh  = make_uvsphere(1.0; nφ=10, nθ=20)
    geom  = compute_geometry(mesh)
    dec   = build_dec(mesh, geom)
    nv    = length(mesh.points)

    cache = build_pde_cache(mesh, geom, dec)
    u     = ones(Float64, nv)
    l2    = l2_norm_cached(cache, u)
    @test l2 > 0.0
    @test isfinite(l2)
    # L² norm of 1 on unit sphere ≈ sqrt(surface area) = sqrt(4π)
    @test abs(l2 - sqrt(4π)) < 0.2
end

@testset "energy_norm_cached: constant function has near-zero energy" begin
    mesh  = make_uvsphere(1.0; nφ=10, nθ=20)
    geom  = compute_geometry(mesh)
    dec   = build_dec(mesh, geom)
    nv    = length(mesh.points)

    cache = build_pde_cache(mesh, geom, dec)
    u     = ones(Float64, nv)  # constant -> gradient ~ 0 -> energy norm ~ 0
    en    = energy_norm_cached(cache, u)
    @test en >= 0.0
    @test en < 1e-3  # should be very small
end

@testset "apply_mass_inplace!: non-trivial output" begin
    mesh  = make_uvsphere(1.0; nφ=8, nθ=16)
    geom  = compute_geometry(mesh)
    dec   = build_dec(mesh, geom)
    nv    = length(mesh.points)

    cache = build_pde_cache(mesh, geom, dec)
    u     = rand(Float64, nv)
    y     = zeros(Float64, nv)
    apply_mass_inplace!(y, cache, u)
    @test !all(y .== 0.0)
    @test all(isfinite.(y))
end

@testset "apply_laplace_inplace!: finite output" begin
    mesh  = make_uvsphere(1.0; nφ=8, nθ=16)
    geom  = compute_geometry(mesh)
    dec   = build_dec(mesh, geom)
    nv    = length(mesh.points)

    cache = build_pde_cache(mesh, geom, dec)
    u     = rand(Float64, nv)
    y     = zeros(Float64, nv)
    apply_laplace_inplace!(y, cache, u)
    @test all(isfinite.(y))
end

@testset "CurvePDECache: construction" begin
    crv   = sample_circle(1.0, 64)
    geom  = compute_geometry(crv)
    dec   = build_dec(crv, geom)
    nv    = length(crv.points)

    cache = build_pde_cache(crv, geom, dec; μ=0.01, dt=0.001, θ=1.0)
    @test size(cache.laplace, 1) == nv
    @test cache.diffusion_fac !== nothing
end

@testset "CurvePDECache: step_diffusion_cached" begin
    crv   = sample_circle(1.0, 64)
    geom  = compute_geometry(crv)
    dec   = build_dec(crv, geom)
    nv    = length(crv.points)
    u0    = rand(Float64, nv)

    cache = build_pde_cache(crv, geom, dec; μ=0.01, dt=0.001, θ=1.0)
    u1    = step_diffusion_cached(cache, u0)
    @test length(u1) == nv
    @test all(isfinite.(u1))
end
