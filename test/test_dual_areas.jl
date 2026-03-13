# test_dual_areas.jl – Tests for barycentric and mixed/Voronoi dual areas.

using Test
using FrontIntrinsicOps
using LinearAlgebra

@testset "Barycentric dual areas sum to total area" begin
    for mesh in [
        make_uvsphere(1.0; nφ=8, nθ=16),
        generate_icosphere(1.0, 1),
        generate_torus(3.0, 1.0, 8, 16),
    ]
        geom  = compute_geometry(mesh; dual_area=:barycentric)
        total = measure(mesh, geom)
        sum_da = sum(geom.vertex_dual_areas)
        @test abs(sum_da - total) / total < 1e-12
        @test geom.dual_area_method === :barycentric
    end
end

@testset "Mixed dual areas sum to total area" begin
    for mesh in [
        make_uvsphere(1.0; nφ=8, nθ=16),
        generate_icosphere(1.0, 1),
        generate_torus(3.0, 1.0, 8, 16),
    ]
        geom_bary = compute_geometry(mesh; dual_area=:barycentric)
        geom_mix  = compute_geometry(mesh; dual_area=:mixed)
        total     = measure(mesh, geom_bary)
        sum_da    = sum(geom_mix.vertex_dual_areas)
        @test abs(sum_da - total) / total < 1e-12
        @test geom_mix.dual_area_method === :mixed
    end
end

@testset "Mixed dual areas are positive on standard meshes" begin
    for mesh in [
        generate_icosphere(1.0, 1),
        generate_icosphere(1.0, 2),
        generate_torus(3.0, 1.0, 16, 32),
    ]
        geom = compute_geometry(mesh; dual_area=:mixed)
        @test all(geom.vertex_dual_areas .> 0)
    end
end

@testset "Voronoi alias works" begin
    mesh  = generate_icosphere(1.0, 1)
    g1    = compute_geometry(mesh; dual_area=:mixed)
    g2    = compute_geometry(mesh; dual_area=:voronoi)
    @test g1.vertex_dual_areas == g2.vertex_dual_areas
    @test g2.dual_area_method === :mixed
end
