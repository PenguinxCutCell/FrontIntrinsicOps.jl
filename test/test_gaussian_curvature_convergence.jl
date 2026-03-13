# test_gaussian_curvature_convergence.jl
# Lightweight convergence tests for Gaussian curvature.

using Test
using FrontIntrinsicOps
using LinearAlgebra
using Statistics: mean

@testset "Sphere Gaussian curvature converges under refinement" begin
    R = 1.0
    K_exact = 1.0 / R^2  # = 1.0

    errors = Float64[]
    for level in [1, 2, 3]
        mesh = generate_icosphere(R, level)
        geom = compute_geometry(mesh)
        K    = gaussian_curvature(mesh, geom)
        # L2 error weighted by dual area
        da   = geom.vertex_dual_areas
        err  = sqrt(sum(da .* (K .- K_exact).^2))
        push!(errors, err)
    end

    # Each refinement should reduce the error
    @test errors[2] < errors[1]
    @test errors[3] < errors[2]
end

@testset "Torus integrated Gauss-Bonnet is near zero" begin
    R = 3.0; r = 1.0
    for (nt, np) in [(16, 32), (24, 48)]
        mesh = generate_torus(R, r, nt, np)
        geom = compute_geometry(mesh)
        intK = integrated_gaussian_curvature(mesh, geom)
        @test abs(intK) < 1e-10
    end
end

@testset "Sphere integrated Gauss-Bonnet ≈ 4pi" begin
    R = 1.0
    for level in [1, 2, 3]
        mesh = generate_icosphere(R, level)
        geom = compute_geometry(mesh)
        intK = integrated_gaussian_curvature(mesh, geom)
        @test abs(intK - 4*pi) / (4*pi) < 0.01
    end
end
