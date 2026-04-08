# test_lie_derivative.jl – Interior product and Cartan Lie derivative checks.

using Test
using LinearAlgebra
using StaticArrays
using FrontIntrinsicOps

function _face_field_constant(mesh::SurfaceMesh{T}, geom::SurfaceGeometry{T}, v::SVector{3,T}) where {T}
    vt = tangential_project(v, SVector{3,T}(0, 0, 1))
    return [vt for _ in mesh.faces]
end

@testset "Lie derivative scalar sanity" begin
    mesh = make_flat_patch(N=12, L=1.0)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    topo = build_topology(mesh)

    X = _face_field_constant(mesh, geom, SVector(1.0, 0.0, 0.0))
    Z = [SVector(0.0, 0.0, 0.0) for _ in mesh.faces]

    c = fill(2.0, length(mesh.points))
    Lc = lie_derivative(X, c, mesh, geom, dec)
    @test norm(Lc) < 1e-12

    u = [p[1] for p in mesh.points]  # linear x field
    Lu = lie_derivative(X, u, mesh, geom, dec)

    bnd = Set(detect_boundary_vertices(mesh, topo))
    interior = [i for i in 1:length(mesh.points) if !(i in bnd)]
    mean_err = mean(abs.(Lu[interior] .- 1.0))
    @test mean_err < 0.15

    Lz = lie_derivative(Z, u, mesh, geom, dec)
    @test norm(Lz) < 1e-12
end

@testset "Cartan identity implementation for 1-forms" begin
    mesh = generate_icosphere(1.0, 1)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    X = [tangential_project(SVector(0.2, -0.1, 0.3), geom.face_normals[fi]) for fi in 1:length(mesh.faces)]

    α = dec.d0 * [sin(p[1]) + 0.1 * p[2] for p in mesh.points]
    L1 = lie_derivative(X, α, mesh, geom, dec)
    L2 = cartan_lie_derivative(X, α, mesh, geom, dec)

    @test length(L1) == length(build_topology(mesh).edges)
    @test norm(L1 - L2) < 1e-12

    Z = [SVector(0.0, 0.0, 0.0) for _ in mesh.faces]
    @test norm(lie_derivative(Z, α, mesh, geom, dec)) < 1e-12
end

@testset "Interior product degree behavior" begin
    mesh = generate_icosphere(1.0, 1)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    X = [tangential_project(SVector(0.4, 0.2, -0.1), geom.face_normals[fi]) for fi in 1:length(mesh.faces)]

    α1 = dec.d0 * [p[2] for p in mesh.points]
    iXα1 = interior_product(X, α1, mesh, geom, dec)
    @test length(iXα1) == length(mesh.points)

    α2 = [geom.face_areas[fi] for fi in 1:length(mesh.faces)]
    iXα2 = interior_product(X, α2, mesh, geom, dec)
    @test length(iXα2) == length(build_topology(mesh).edges)

    iX0 = interior_product(X, fill(1.0, length(mesh.points)), mesh, geom, dec)
    @test length(iX0) == 0
end

@testset "Curve interior/Lie support" begin
    mesh = sample_circle(1.0, 64)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    # Tangential speed represented at vertices.
    X = [1.0 + 0.2 * p[1] for p in mesh.points]
    Z = zeros(length(mesh.points))

    c = fill(2.0, length(mesh.points))
    @test norm(lie_derivative(X, c, mesh, geom, dec; representation=:tangent_speed, degree=0)) < 1e-12
    @test norm(lie_derivative(Z, c, mesh, geom, dec; representation=:tangent_speed, degree=0)) < 1e-12

    α1 = dec.d0 * [p[1] for p in mesh.points]
    iXα1 = interior_product(X, α1, mesh, geom, dec; representation=:tangent_speed, degree=1)
    @test length(iXα1) == length(mesh.points)
    @test all(isfinite, iXα1)

    L1 = lie_derivative(X, α1, mesh, geom, dec; representation=:tangent_speed, degree=1)
    @test length(L1) == length(mesh.edges)
    @test all(isfinite, L1)

    @test norm(lie_derivative(Z, α1, mesh, geom, dec; representation=:tangent_speed, degree=1)) < 1e-12
end
