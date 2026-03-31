# test_generators.jl – Tests for deterministic mesh generators.

using Test
using FrontIntrinsicOps
using LinearAlgebra
using StaticArrays

@testset "sample_circle" begin
    for N in [8, 16, 64]
        R    = 1.5
        mesh = sample_circle(R, N)
        @test length(mesh.points) == N
        @test length(mesh.edges)  == N
        # Closed
        @test is_closed(mesh)
        # All points on circle of radius R
        for p in mesh.points
            @test abs(norm(p) - R) < 1e-12
        end
    end
end

@testset "sample_perturbed_circle" begin
    R = 1.5
    N = 64
    ϵ = 0.12
    mode = 4
    θ0 = 0.2
    mesh = sample_perturbed_circle(R, N; ϵ=ϵ, mode=mode, θ0=θ0)
    @test length(mesh.points) == N
    @test length(mesh.edges) == N
    @test is_closed(mesh)
    for (k, p) in enumerate(mesh.points)
        θ = 2π * (k - 1) / N
        r_expected = R * (1 + ϵ * cos(mode * (θ - θ0)))
        @test abs(norm(p) - r_expected) < 1e-12
    end

    circle = sample_perturbed_circle(R, N; ϵ=0.0, mode=mode, θ0=θ0)
    for p in circle.points
        @test abs(norm(p) - R) < 1e-12
    end

    @test_throws ArgumentError sample_perturbed_circle(R, N; ϵ=1.0)
    @test_throws ArgumentError sample_perturbed_circle(R, N; ϵ=-1.0)
    @test_throws ArgumentError sample_perturbed_circle(R, N; mode=-1)
end

@testset "generate_uvsphere" begin
    R    = 2.0
    mesh = generate_uvsphere(R, 8, 16)
    report = check_mesh(mesh)
    @test report.closed
    @test report.manifold
    @test report.euler_characteristic == 2
    # All points on sphere
    for p in mesh.points
        @test abs(norm(p) - R) < 1e-12
    end
end

@testset "generate_icosphere" begin
    R = 1.0
    for level in [0, 1, 2]
        mesh = generate_icosphere(R, level)
        report = check_mesh(mesh)
        @test report.closed
        @test report.manifold
        @test report.euler_characteristic == 2
        # All points on sphere
        for p in mesh.points
            @test abs(norm(p) - R) < 1e-12
        end
    end
    # Size checks
    mesh0 = generate_icosphere(R, 0)
    @test length(mesh0.points) == 12
    @test length(mesh0.faces)  == 20
    mesh1 = generate_icosphere(R, 1)
    @test length(mesh1.points) == 42
    @test length(mesh1.faces)  == 80
end

@testset "generate_torus" begin
    R    = 3.0
    r    = 1.0
    mesh = generate_torus(R, r, 16, 32)
    report = check_mesh(mesh)
    @test report.closed
    @test report.manifold
    @test report.euler_characteristic == 0  # torus: chi = 0
    # Check rough area
    geom = compute_geometry(mesh)
    exact_area = 4 * pi^2 * R * r
    @test abs(measure(mesh, geom) - exact_area) / exact_area < 0.05
end
