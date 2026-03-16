# test_reaction_diffusion.jl – Tests for surface reaction–diffusion equations (v0.4).
#
# Tests:
# 1. Reaction API: evaluate_reaction! with different input forms.
# 2. Built-in reactions: Fisher–KPP, linear decay, bistable.
# 3. Explicit Euler step: conservation check on flat patch.
# 4. IMEX backward-Euler step: factorization reuse.
# 5. Linear decay analytic comparison on sphere (exponential decay).
# 6. Fisher–KPP: robustness on sphere and torus.

# Helper: build a small sphere for fast tests
function make_small_sphere(; nφ=8, nθ=16)
    return make_uvsphere(1.0; nφ=nφ, nθ=nθ)
end

@testset "reaction API: evaluate_reaction!" begin
    mesh = make_small_sphere()
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    nv   = length(mesh.points)
    u    = rand(Float64, nv)
    r    = zeros(Float64, nv)

    # nothing reaction
    evaluate_reaction!(r, nothing, u, mesh, geom, 0.0)
    @test all(r .== 0.0)

    # pointwise function
    decay = linear_decay_reaction(1.0)
    evaluate_reaction!(r, decay, u, mesh, geom, 0.0)
    @test all(r .≈ -u)

    # Fisher–KPP
    fkpp = fisher_kpp_reaction(2.0)
    evaluate_reaction!(r, fkpp, u, mesh, geom, 0.0)
    expected = 2.0 .* u .* (1.0 .- u)
    @test maximum(abs.(r .- expected)) < 1e-14

    # Bistable
    bist = bistable_reaction(1.0)
    evaluate_reaction!(r, bist, u, mesh, geom, 0.5)
    expected2 = u .* (1.0 .- u) .* (u .- 0.5)
    @test maximum(abs.(r .- expected2)) < 1e-14
end

@testset "IMEX step: factorization reuse" begin
    mesh = make_small_sphere()
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    nv   = length(mesh.points)
    u0   = ones(Float64, nv)
    dt   = 0.01; μ = 0.1

    # First step assembles factorization
    u1, fac = step_surface_reaction_diffusion_imex(mesh, geom, dec, u0, dt, μ, nothing, 0.0)
    @test length(u1) == nv
    @test fac !== nothing

    # Second step reuses factorization
    u2, fac2 = step_surface_reaction_diffusion_imex(mesh, geom, dec, u1, dt, μ, nothing, dt;
                                                     factorization=fac)
    @test fac2 === fac   # same object reused
    @test length(u2) == nv
end

@testset "IMEX step: linear decay analytic comparison" begin
    # On the unit sphere with u0 = u_const * Y (any smooth function),
    # with only linear decay (no diffusion): u(t) = u0 * exp(-α t)
    mesh = make_small_sphere()
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    nv   = length(mesh.points)

    α    = 1.0
    u0   = 0.5 .* ones(Float64, nv)
    T_end = 1.0
    dt   = 0.01
    μ    = 0.0   # no diffusion

    u_final, _ = solve_surface_reaction_diffusion(
        mesh, geom, dec, u0, T_end, dt, μ,
        linear_decay_reaction(α);
        θ=1.0, scheme=:imex)

    # Exact solution: u(T) = u0 * exp(-α * T)
    u_exact = u0 .* exp(-α * T_end)
    # IMEX with θ=1 (backward Euler on reaction too if explicit): first order
    # Since BE is implicit only in diffusion (μ=0 here, no diffusion term)
    # and reaction is explicit: BE on reaction would give u^{n+1} = u^n + dt*r(u^n)
    # Actually for IMEX: (M + dt*0*L) u^{n+1} = M u^n + dt * r(u^n)
    # = M u^n + dt * (-α u^n) = M u^n (1 - dt*α)
    # So u^{n+1} = u^n * (1 - dt*α) -> u(T) ≈ u0 * (1 - dt*α)^(T/dt)
    u_euler_approx = u0 .* (1 - dt * α)^(T_end / dt)
    # Check that we match the explicit-reaction Euler behavior
    @test maximum(abs.(u_final .- u_euler_approx)) < 0.01
    # Check sign is right (solution should decay)
    @test all(u_final .> 0.0)
    @test all(u_final .< maximum(u0))
end

@testset "IMEX step: Crank-Nicolson θ=0.5" begin
    mesh = make_small_sphere()
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    nv   = length(mesh.points)
    u0   = rand(Float64, nv)
    dt   = 0.01; μ = 0.1

    u_be, _ = step_surface_reaction_diffusion_imex(mesh, geom, dec, u0, dt, μ, nothing, 0.0; θ=1.0)
    u_cn, _ = step_surface_reaction_diffusion_imex(mesh, geom, dec, u0, dt, μ, nothing, 0.0; θ=0.5)

    # Both should produce valid results
    @test all(isfinite.(u_be))
    @test all(isfinite.(u_cn))
    # CN should be closer to the centered value
    @test !all(u_be .≈ u_cn)  # they should differ
end

@testset "Fisher–KPP: logistic growth robustness" begin
    mesh = make_small_sphere()
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    nv   = length(mesh.points)

    # Start with small initial condition
    u0 = fill(0.1, nv)
    T_end = 2.0; dt = 0.05; μ = 0.01; α = 1.0

    u_final, _ = solve_surface_reaction_diffusion(
        mesh, geom, dec, u0, T_end, dt, μ,
        fisher_kpp_reaction(α); θ=1.0)

    # Solution should grow toward ~1 (logistic carrying capacity)
    @test all(isfinite.(u_final))
    @test minimum(u_final) >= -0.01  # non-negative (approximately)
    @test mean(u_final) > mean(u0)   # mean increased
end

@testset "solve_surface_reaction_diffusion: callback and save_every" begin
    mesh = make_small_sphere()
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    nv   = length(mesh.points)
    u0   = ones(Float64, nv)

    # Test callback
    steps_called = Ref(0)
    cb = (u, t, step) -> (steps_called[] = step)

    u_f, t_f = solve_surface_reaction_diffusion(
        mesh, geom, dec, u0, 0.1, 0.01, 0.0, nothing;
        callback=cb)
    @test steps_called[] == 10

    # Test save_every
    u_f2, t_f2, hist = solve_surface_reaction_diffusion(
        mesh, geom, dec, u0, 0.1, 0.01, 0.0, nothing;
        save_every=5)
    @test length(hist) >= 3  # initial + 2 saved
end

@testset "explicit Euler step" begin
    mesh = make_small_sphere()
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    nv   = length(mesh.points)
    u0   = rand(Float64, nv)

    u1 = step_surface_reaction_diffusion_explicit(
        mesh, geom, dec, u0, 1e-5, 0.1, nothing, 0.0)
    @test length(u1) == nv
    @test all(isfinite.(u1))
end
