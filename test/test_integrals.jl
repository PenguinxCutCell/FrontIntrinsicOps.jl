# test_integrals.jl – Tests for integral quantities.

using Test
using FrontIntrinsicOps
using LinearAlgebra

@testset "Curve: constant field integration = length" begin
    R  = 1.5
    N  = 128
    mesh = make_circle(N, R)
    geom = compute_geometry(mesh)
    L    = measure(mesh, geom)
    u    = ones(Float64, N)
    @test abs(integrate_vertex_field(mesh, geom, u) - L) / L < 1e-14
end

@testset "Curve: edge constant field integration = length" begin
    R  = 1.5
    N  = 128
    mesh = make_circle(N, R)
    geom = compute_geometry(mesh)
    L    = measure(mesh, geom)
    u    = ones(Float64, N)
    @test abs(integrate_face_field(mesh, geom, u) - L) / L < 1e-14
end

@testset "Surface: constant field integration = area" begin
    R    = 2.0
    mesh = make_uvsphere(R; nφ=16, nθ=32)
    geom = compute_geometry(mesh)
    A    = measure(mesh, geom)
    nv   = length(mesh.points)
    u    = ones(Float64, nv)
    @test abs(integrate_vertex_field(mesh, geom, u) - A) / A < 1e-14
end

@testset "Surface: face constant field integration = area" begin
    R    = 2.0
    mesh = make_uvsphere(R; nφ=16, nθ=32)
    geom = compute_geometry(mesh)
    A    = measure(mesh, geom)
    nf   = length(mesh.faces)
    u    = ones(Float64, nf)
    @test abs(integrate_face_field(mesh, geom, u) - A) / A < 1e-14
end

@testset "Circle: enclosed area" begin
    R     = 3.0
    N     = 256
    mesh  = make_circle(N, R)
    A     = enclosed_measure(mesh)
    exact = π * R^2
    @test abs(A - exact) / exact < 1e-3
end

@testset "Sphere: enclosed volume" begin
    R     = 2.0
    mesh  = make_uvsphere(R; nφ=20, nθ=40)
    V     = enclosed_measure(mesh)
    exact = (4/3) * π * R^3
    @test abs(V - exact) / exact < 0.05
end
