# test_surface_diffusion.jl – Tests for surface Poisson, Helmholtz, and
#                              transient diffusion.

using Test
using FrontIntrinsicOps
using LinearAlgebra

@testset "Poisson solve on sphere (new API)" begin
    R    = 1.0
    mesh = make_uvsphere(R; nφ=16, nθ=32)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    # f = z has zero mean on sphere
    z = [p[3] for p in mesh.points]
    u = solve_surface_poisson(mesh, geom, dec, z)

    # L u should ≈ f  (mean-adjusted)
    Lu    = dec.lap0 * u
    f_adj = copy(z); enforce_compatibility!(f_adj, mesh, geom)
    res   = maximum(abs, Lu .- f_adj)
    @test res < 1e-6   # small residual after projection

    # Solution should have zero mean
    @test abs(weighted_mean(mesh, geom, u)) < 1e-10
end

@testset "Helmholtz solve on sphere" begin
    R    = 1.0
    mesh = make_uvsphere(R; nφ=12, nθ=24)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    nv = length(mesh.points)
    f  = ones(Float64, nv)
    α  = 1.0

    u = solve_surface_helmholtz(mesh, geom, dec, f, α)

    # Verify: (L + α M) u ≈ f
    L   = dec.lap0
    M   = mass_matrix(mesh, geom)
    res = maximum(abs, (L + α * M) * u .- f)
    @test res < 1e-10
end

@testset "Transient diffusion: eigenmode decay (backward Euler)" begin
    # On a sphere of radius R, the first non-trivial eigenvalue of L is 2/R².
    # If u(0) = z, then u(t) = exp(-μ * (2/R²) * t) * z.
    R  = 1.0
    μ  = 0.1
    T_end = 0.5
    λ  = 2.0 / R^2   # eigenvalue for spherical harmonics of degree 1

    mesh = make_uvsphere(R; nφ=16, nθ=32)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    z  = Float64[p[3] for p in mesh.points]
    u  = copy(z)
    dt = 1e-2
    nsteps = round(Int, T_end / dt)
    fac = nothing
    for _ in 1:nsteps
        u, fac = step_surface_diffusion_backward_euler(mesh, geom, dec, u, dt, μ;
                                                        factorization=fac)
    end

    u_exact = exp(-μ * λ * T_end) .* z
    err = weighted_l2_error(mesh, geom, u, u_exact)
    @test err < 0.05   # <5% error with dt=1e-2 and backward Euler
end

@testset "Transient diffusion: eigenmode decay (Crank–Nicolson)" begin
    R  = 1.0
    μ  = 0.1
    T_end = 0.5
    λ  = 2.0 / R^2

    mesh = make_uvsphere(R; nφ=16, nθ=32)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    z  = Float64[p[3] for p in mesh.points]
    u  = copy(z)
    dt = 1e-2
    nsteps = round(Int, T_end / dt)
    fac = nothing
    for _ in 1:nsteps
        u, fac = step_surface_diffusion_crank_nicolson(mesh, geom, dec, u, dt, μ;
                                                        factorization=fac)
    end

    u_exact = exp(-μ * λ * T_end) .* z
    err = weighted_l2_error(mesh, geom, u, u_exact)
    @test err < 0.02   # CN is 2nd order – should be more accurate
end

@testset "Diffusion: factorization reuse" begin
    mesh = make_uvsphere(1.0; nφ=12, nθ=24)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    z  = Float64[p[3] for p in mesh.points]
    u0 = copy(z)
    dt = 0.01; μ = 0.1

    # First step: no factorization supplied
    u1, fac1 = step_surface_diffusion_backward_euler(mesh, geom, dec, u0, dt, μ)
    @test fac1 !== nothing

    # Second step: reuse factorization
    u2a, _    = step_surface_diffusion_backward_euler(mesh, geom, dec, u1, dt, μ)
    u2b, fac2 = step_surface_diffusion_backward_euler(mesh, geom, dec, u1, dt, μ;
                                                       factorization=fac1)
    @test maximum(abs, u2a .- u2b) < 1e-12
end

@testset "laplace_matrix returns dec.lap0 by default" begin
    mesh = make_uvsphere(1.0; nφ=12, nθ=24)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    L    = laplace_matrix(mesh, geom, dec)
    @test L ≈ dec.lap0
end
