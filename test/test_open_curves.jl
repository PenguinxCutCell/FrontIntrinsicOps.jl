using Test
using FrontIntrinsicOps
using StaticArrays
using LinearAlgebra

@testset "Open curve endpoint normals are one-sided" begin
    pts = [
        SVector{2,Float64}(0.0, 0.0),
        SVector{2,Float64}(1.0, 0.0),
        SVector{2,Float64}(1.0, 1.0),
    ]
    mesh = load_curve_points(pts; closed=false)
    geom = compute_geometry(mesh)

    n_start = geom.vertex_normals[1]
    n_end = geom.vertex_normals[3]

    @test isapprox(n_start, SVector{2,Float64}(0.0, 1.0); atol=1e-12)
    @test isapprox(n_end, SVector{2,Float64}(-1.0, 0.0); atol=1e-12)

    @test abs(dot(n_start, geom.edge_tangents[1])) <= 1e-12
    @test abs(dot(n_end, geom.edge_tangents[2])) <= 1e-12

    @test isapprox(geom.signed_curvature[1], 0.0; atol=1e-12)
    @test isapprox(geom.signed_curvature[3], 0.0; atol=1e-12)
end
