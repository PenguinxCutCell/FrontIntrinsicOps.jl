# test_allocations.jl – Lightweight allocation sanity checks.
#
# We use @allocated to check that repeatedly calling core functions does not
# allocate pathologically.  We do NOT set byte-exact thresholds; instead we
# use broad sanity caps.
#
# Note: the first call to a function allocates due to JIT compilation.
# We always warm up with one call before measuring.

using Test
using FrontIntrinsicOps
using LinearAlgebra
using SparseArrays
using StaticArrays

@testset "Allocation sanity: laplace_matrix reuse" begin
    mesh = generate_icosphere(1.0, 2)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    # Warm-up
    _ = laplace_matrix(mesh, geom, dec)

    # Returning the already-assembled lap0 should be near-zero allocation
    bytes = @allocated laplace_matrix(mesh, geom, dec)
    # Should be very small (just returning a reference)
    @test bytes < 200
end

@testset "Allocation sanity: apply_laplace! is in-place" begin
    mesh = generate_icosphere(1.0, 2)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    nv   = length(mesh.points)

    u = randn(Float64, nv)
    y = similar(u)

    # Warm-up
    apply_laplace!(y, dec, u)

    bytes = @allocated apply_laplace!(y, dec, u)
    # mul! on a sparse matrix allocates minimally
    @test bytes < 10_000
end

@testset "Allocation sanity: diffusion step with reused factorization" begin
    mesh = generate_icosphere(1.0, 2)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    z   = Float64[p[3] for p in mesh.points]
    dt  = 0.01; μ = 0.1

    # Build factorization once
    u1, fac = step_surface_diffusion_backward_euler(mesh, geom, dec, z, dt, μ)

    # Warm-up second call
    u2, _ = step_surface_diffusion_backward_euler(mesh, geom, dec, u1, dt, μ;
                                                   factorization=fac)

    # Measure reused step – should not rebuild matrix
    bytes = @allocated step_surface_diffusion_backward_euler(mesh, geom, dec, u2, dt, μ;
                                                              factorization=fac)
    # Broad cap: should not allocate more than a few MB
    @test bytes < 5_000_000
end

@testset "Allocation sanity: apply_mass! is in-place" begin
    mesh = generate_icosphere(1.0, 2)
    geom = compute_geometry(mesh)
    nv   = length(mesh.points)

    x = randn(Float64, nv)
    y = similar(x)

    # Warm-up
    apply_mass!(y, mesh, geom, x)

    bytes = @allocated apply_mass!(y, mesh, geom, x)
    @test bytes < 1000
end

@testset "Allocation sanity: IMEX step with reused factorization and operator" begin
    mesh = generate_icosphere(1.0, 2)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    z    = Float64[p[3] for p in mesh.points]
    vel  = SVector{3,Float64}[SVector{3,Float64}(-p[2], p[1], 0.0) for p in mesh.points]
    dt   = 0.01; μ = 0.1

    A_up = assemble_transport_operator(mesh, geom, vel; scheme=:upwind)

    # Build factorization on first call
    _, fac = step_surface_advection_diffusion_imex(mesh, geom, dec, z, vel, dt, μ;
                                                    scheme             = :upwind,
                                                    transport_operator = A_up)
    # Warm-up
    _, fac = step_surface_advection_diffusion_imex(mesh, geom, dec, z, vel, dt, μ;
                                                    scheme             = :upwind,
                                                    transport_operator = A_up,
                                                    factorization      = fac)
    # Measure: reusing both A and fac should be cheap (only sparse matvec + backsolve)
    bytes = @allocated step_surface_advection_diffusion_imex(mesh, geom, dec, z, vel, dt, μ;
                                                              scheme             = :upwind,
                                                              transport_operator = A_up,
                                                              factorization      = fac)
    # Should not reassemble L or rebuild factorization: expect < 1 MB
    @test bytes < 1_000_000
end

@testset "Allocation sanity: weighted_mean is non-allocating" begin
    mesh = generate_icosphere(1.0, 2)
    geom = compute_geometry(mesh)
    nv   = length(mesh.points)
    u    = randn(Float64, nv)

    # Warm-up
    _ = weighted_mean(mesh, geom, u)

    bytes = @allocated weighted_mean(mesh, geom, u)
    @test bytes < 1000
end
