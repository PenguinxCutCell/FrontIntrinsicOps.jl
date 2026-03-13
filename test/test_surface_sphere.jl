# test_surface_sphere.jl – Convergence tests on UV-sphere approximations.

using Test
using FrontIntrinsicOps
using LinearAlgebra
using Statistics: mean

@testset "UV-sphere area and volume" begin
    R = 2.0
    exact_area   = 4π * R^2
    exact_volume = (4/3) * π * R^3

    for (nφ, nθ) in [(8,16), (16,32)]
        mesh = make_uvsphere(R; nφ=nφ, nθ=nθ)
        geom = compute_geometry(mesh)
        A    = measure(mesh, geom)
        V    = enclosed_measure(mesh)
        @test abs(A - exact_area) / exact_area < 0.1
        @test abs(V - exact_volume) / exact_volume < 0.1
    end
end

@testset "UV-sphere vertex normals align radially" begin
    R = 1.0
    mesh = make_uvsphere(R; nφ=16, nθ=32)
    geom = compute_geometry(mesh)

    errors = Float64[]
    for (i, p) in enumerate(mesh.points)
        n̂_analytical = p / norm(p)
        n̂_discrete   = geom.vertex_normals[i]
        push!(errors, abs(dot(n̂_analytical, n̂_discrete) - 1.0))
    end
    @test mean(errors) < 0.01
end

@testset "UV-sphere mesh checks" begin
    mesh = make_uvsphere(1.0; nφ=8, nθ=16)
    report = check_mesh(mesh)
    @test report.closed
    @test report.manifold
    @test report.euler_characteristic == 2
end

@testset "Integration of constant on sphere" begin
    R    = 1.5
    mesh = make_uvsphere(R; nφ=16, nθ=32)
    geom = compute_geometry(mesh)
    A    = measure(mesh, geom)
    nv   = length(mesh.points)
    u    = ones(Float64, nv)
    @test abs(integrate_vertex_field(mesh, geom, u) - A) / A < 1e-14
end
