# test_curvature.jl – Curvature tests.

using Test
using FrontIntrinsicOps
using LinearAlgebra
using Statistics: mean

@testset "Sphere constant mean curvature" begin
    R    = 2.0
    mesh = make_uvsphere(R; nφ=20, nθ=40)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    H    = mean_curvature(mesh, geom, dec)

    interior = 2:(length(mesh.points)-1)
    H_interior = abs.(H[interior])
    H_mean = mean(H_interior)
    @test abs(H_mean - 1.0/R) / (1.0/R) < 0.15
end

@testset "Planar patch near-zero mean curvature (interior)" begin
    mesh = make_flat_patch(N=10, L=1.0)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    H    = mean_curvature(mesh, geom, dec)

    N = 10
    interior_ids = Int[]
    idx_v(i,j) = j*(N+1) + i + 1
    for j in 2:(N-1), i in 2:(N-1)
        push!(interior_ids, idx_v(i,j))
    end
    H_max = maximum(abs, H[interior_ids])
    @test H_max < 0.1
end

@testset "Gauss–Bonnet on sphere" begin
    R    = 1.0
    mesh = make_uvsphere(R; nφ=16, nθ=32)
    geom = compute_geometry(mesh)
    K    = gaussian_curvature(mesh, geom)
    nv   = length(mesh.points)

    total_K = sum(K[i] * geom.vertex_dual_areas[i] for i in 1:nv)
    @test abs(total_K - 4π) / (4π) < 0.15
end

@testset "Gaussian curvature sign on sphere" begin
    R    = 1.0
    mesh = make_uvsphere(R; nφ=16, nθ=32)
    geom = compute_geometry(mesh)
    K    = gaussian_curvature(mesh, geom)
    @test all(K .> -0.1)
end
