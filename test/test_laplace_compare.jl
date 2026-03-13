# test_laplace_compare.jl – Tests for DEC vs cotan Laplace comparison.

using Test
using FrontIntrinsicOps
using LinearAlgebra

@testset "L_dec * ones ≈ 0" begin
    for mesh in [
        make_uvsphere(1.0; nφ=8, nθ=16),
        generate_icosphere(1.0, 2),
    ]
        geom = compute_geometry(mesh)
        L    = build_laplace_beltrami(mesh, geom; method=:dec)
        nv   = length(mesh.points)
        res  = maximum(abs, L * ones(nv))
        @test res < 1e-10
    end
end

@testset "L_cotan * ones ≈ 0" begin
    for mesh in [
        make_uvsphere(1.0; nφ=8, nθ=16),
        generate_icosphere(1.0, 2),
    ]
        geom = compute_geometry(mesh)
        L    = build_laplace_beltrami(mesh, geom; method=:cotan)
        nv   = length(mesh.points)
        res  = maximum(abs, L * ones(nv))
        @test res < 1e-10
    end
end

@testset "DEC and cotan Laplacians agree on good meshes" begin
    # On an icosphere (all-acute triangles) both paths give near-identical matrices
    mesh = generate_icosphere(1.0, 2)
    geom = compute_geometry(mesh)
    rpt  = compare_laplace_methods(mesh, geom)
    # Both should preserve constants
    @test rpt.dec_nullspace
    @test rpt.cotan_nullspace
    # Difference should be near machine precision on a good mesh
    @test rpt.norm_inf < 1e-12
end

@testset "compare_laplace_methods report fields" begin
    mesh = generate_uvsphere(1.0, 8, 16)
    geom = compute_geometry(mesh)
    rpt  = compare_laplace_methods(mesh, geom)
    @test haskey(rpt, :norm_inf)
    @test haskey(rpt, :norm_frob)
    @test haskey(rpt, :dec_nullspace)
    @test haskey(rpt, :cotan_nullspace)
    @test rpt.dec_nullspace
    @test rpt.cotan_nullspace
end
