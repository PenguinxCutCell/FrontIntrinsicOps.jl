# test_curve_circle.jl – Convergence tests on sampled circles.
#
# Tests
# -----
# 1. Total length converges to 2πR.
# 2. Enclosed area converges to πR².
# 3. Mean curvature converges to 1/R.
# 4. Laplace–Beltrami nullspace on constants.

using Test
using FrontIntrinsicOps
using Statistics: mean

@testset "Circle convergence" begin
    R = 2.0
    exact_length = 2π * R
    exact_area   = π * R^2
    exact_kappa  = 1.0 / R

    for N in [16, 64, 256]
        mesh = make_circle(N, R)
        geom = compute_geometry(mesh)

        L   = measure(mesh, geom)
        A   = enclosed_measure(mesh)
        κ   = curvature(mesh, geom)
        κ_mean = mean(κ)

        # Length error should decrease with N
        @test abs(L - exact_length) / exact_length < 10.0 / N^2

        # Area error should decrease with N
        @test abs(A - exact_area) / exact_area < 10.0 / N^2

        # Mean curvature error
        @test abs(κ_mean - exact_kappa) / exact_kappa < 5.0 / N^2
    end
end

@testset "Laplacian constant nullspace on circle" begin
    R = 1.0
    N = 32
    mesh = make_circle(N, R)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    u_const = ones(Float64, N)
    Lu = laplace_beltrami(mesh, geom, dec, u_const)
    @test maximum(abs, Lu) < 1e-10
end

@testset "DEC diagnostics on circle" begin
    R = 1.5
    N = 32
    mesh = make_circle(N, R)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    report = check_dec(mesh, geom, dec)
    @test report.lap_constant_nullspace
    @test report.star0_positive
    @test report.star1_positive
end

@testset "Integration of constant on circle" begin
    R = 1.0
    N = 64
    mesh = make_circle(N, R)
    geom = compute_geometry(mesh)
    L    = measure(mesh, geom)
    u    = ones(Float64, N)
    @test abs(integrate_vertex_field(mesh, geom, u) - L) / L < 1e-14
end
