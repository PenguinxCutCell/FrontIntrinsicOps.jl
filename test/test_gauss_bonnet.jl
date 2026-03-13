# test_gauss_bonnet.jl – Tests for Euler characteristic and Gauss-Bonnet.

using Test
using FrontIntrinsicOps
using LinearAlgebra

@testset "Euler characteristic sphere" begin
    for mesh in [
        make_uvsphere(1.0; nφ=8, nθ=16),
        generate_uvsphere(1.0, 8, 16),
        generate_icosphere(1.0, 1),
        generate_icosphere(1.0, 2),
    ]
        @test euler_characteristic(mesh) == 2
    end
end

@testset "Euler characteristic torus" begin
    for mesh in [
        generate_torus(3.0, 1.0, 8, 16),
        generate_torus(3.0, 1.0, 16, 32),
    ]
        @test euler_characteristic(mesh) == 0
    end
end

@testset "Gauss-Bonnet on sphere (int K dA = 4pi)" begin
    # Gauss-Bonnet: int K dA = 2 pi chi = 2 pi * 2 = 4 pi for sphere
    target = 4 * pi
    for mesh in [
        generate_uvsphere(1.0, 16, 32),
        generate_icosphere(1.0, 2),
    ]
        geom = compute_geometry(mesh)
        intK = integrated_gaussian_curvature(mesh, geom)
        @test abs(intK - target) / target < 0.01
        # gauss_bonnet_residual should be near zero
        @test gauss_bonnet_residual(mesh, geom) < 1e-10
    end
end

@testset "Gauss-Bonnet on torus (int K dA = 0)" begin
    for mesh in [
        generate_torus(3.0, 1.0, 16, 32),
    ]
        geom = compute_geometry(mesh)
        intK = integrated_gaussian_curvature(mesh, geom)
        @test abs(intK) < 1e-10
        @test gauss_bonnet_residual(mesh, geom) < 1e-10
    end
end
