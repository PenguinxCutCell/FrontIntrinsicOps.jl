# test_kforms.jl – Tests for k-form operators: codifferential, Hodge Laplacians,
#                  gradient/divergence.

using Test
using FrontIntrinsicOps
using LinearAlgebra
using SparseArrays

@testset "gradient_0_to_1 equals d0*u" begin
    mesh = make_uvsphere(1.0; nφ=12, nθ=24)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    nv = length(mesh.points)
    u  = randn(Float64, nv)

    g1 = gradient_0_to_1(mesh, dec, u)
    g2 = dec.d0 * u
    @test maximum(abs, g1 .- g2) < 1e-14
end

@testset "d1 * d0 == 0 (discrete d^2 = 0)" begin
    mesh = make_uvsphere(1.0; nφ=12, nθ=24)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    @test maximum(abs, dec.d1 * dec.d0) < 1e-14
end

@testset "hodge_laplacian_0 matches dec.lap0" begin
    mesh = make_uvsphere(1.0; nφ=12, nθ=24)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    Δ0 = hodge_laplacian_0(mesh, geom, dec)
    @test Δ0 ≈ dec.lap0
end

@testset "hodge_laplacian_1 has correct size" begin
    mesh = make_uvsphere(1.0; nφ=12, nθ=24)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    topo = build_topology(mesh)
    ne   = length(topo.edges)

    Δ1 = hodge_laplacian_1(mesh, geom, dec)
    @test size(Δ1, 1) == ne
    @test size(Δ1, 2) == ne
end

@testset "codifferential_1 has correct size" begin
    mesh = make_uvsphere(1.0; nφ=12, nθ=24)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    topo = build_topology(mesh)
    ne   = length(topo.edges)
    nv   = length(mesh.points)

    δ1 = codifferential_1(mesh, geom, dec)
    @test size(δ1, 1) == nv
    @test size(δ1, 2) == ne
end

@testset "codifferential_2 has correct size" begin
    mesh = make_uvsphere(1.0; nφ=12, nθ=24)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    topo = build_topology(mesh)
    ne   = length(topo.edges)
    nf   = length(mesh.faces)

    δ2 = codifferential_2(mesh, geom, dec)
    @test size(δ2, 1) == ne
    @test size(δ2, 2) == nf
end

@testset "divergence_1_to_0 = codifferential action" begin
    mesh = make_uvsphere(1.0; nφ=12, nθ=24)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    topo = build_topology(mesh)
    ne   = length(topo.edges)

    α  = randn(Float64, ne)
    d1 = divergence_1_to_0(mesh, geom, dec, α)
    δ1 = codifferential_1(mesh, geom, dec)
    @test maximum(abs, d1 .- δ1 * α) < 1e-14
end

@testset "curl_like_1_to_2 = d1 * alpha" begin
    mesh = make_uvsphere(1.0; nφ=12, nθ=24)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    topo = build_topology(mesh)
    ne   = length(topo.edges)

    α = randn(Float64, ne)
    c = curl_like_1_to_2(mesh, dec, α)
    @test maximum(abs, c .- dec.d1 * α) < 1e-14
end

@testset "codifferential_1 * d0 = lap0 (δ₁ d₀ = L)" begin
    # The composition δ₁ d₀ should equal the Laplace–Beltrami lap0
    mesh = make_uvsphere(1.0; nφ=12, nθ=24)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    δ1    = codifferential_1(mesh, geom, dec)
    Δ0_via_comp = δ1 * dec.d0
    diff = maximum(abs, Δ0_via_comp - dec.lap0)
    @test diff < 1e-10
end

@testset "hodge_laplacian_1 is finite on icosphere" begin
    mesh = generate_icosphere(1.0, 2)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    Δ1   = hodge_laplacian_1(mesh, geom, dec)
    @test all(isfinite, nonzeros(Δ1))
end
