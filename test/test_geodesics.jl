# test_geodesics.jl – Heat-method geodesics and path extraction.

using Test
using LinearAlgebra
using FrontIntrinsicOps

function _path_length(mesh::SurfaceMesh{T}, geom::SurfaceGeometry{T}, path::Vector{Int}) where {T}
    topo = build_topology(mesh)
    edge_id = Dict{Tuple{Int,Int},Int}()
    for (ei, e) in enumerate(topo.edges)
        edge_id[(e[1], e[2])] = ei
    end
    L = zero(T)
    for k in 1:(length(path)-1)
        i, j = path[k], path[k+1]
        a, b = min(i, j), max(i, j)
        ei = edge_id[(a, b)]
        L += geom.edge_lengths[ei]
    end
    return L
end

function _sphere_error(level::Int)
    mesh = generate_icosphere(1.0, level)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    src = argmax([p[3] for p in mesh.points])
    d = geodesic_distance_to_vertex(mesh, geom, dec, src)

    ps = mesh.points[src]
    exact = [acos(clamp(dot(ps, p), -1.0, 1.0)) for p in mesh.points]

    rel = norm(d .- exact) / (norm(exact) + 1e-14)
    return rel
end

@testset "Geodesic distance on sphere" begin
    err1 = _sphere_error(1)
    err2 = _sphere_error(2)

    @test err1 < 0.25
    @test err2 < 0.2
    @test err2 < err1
end

@testset "Geodesic distance on near-planar patch" begin
    mesh = make_flat_patch(N=14, L=1.0)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    # Center source.
    src = argmin([norm(p - SVector(0.5, 0.5, 0.0)) for p in mesh.points])
    d = geodesic_distance_to_vertex(mesh, geom, dec, src)

    p0 = mesh.points[src]
    exact = [norm(p - p0) for p in mesh.points]

    interior = [i for i in eachindex(exact) if exact[i] < 0.35]
    rel = norm(d[interior] .- exact[interior]) / (norm(exact[interior]) + 1e-14)
    @test rel < 0.2
end

@testset "Geodesic distance sanity on torus" begin
    mesh = generate_torus(2.0, 0.8, 18, 24)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    src = 1
    d = geodesic_distance_to_vertex(mesh, geom, dec, src)

    @test minimum(d) >= 0.0
    @test d[src] < 1e-10
    @test maximum(d) > 0.5

    g = geodesic_gradient(d, mesh, geom, dec)
    @test length(g) == length(mesh.faces)
end

@testset "Multi-source geodesic distance" begin
    mesh = generate_icosphere(1.0, 1)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    s1 = argmax([p[3] for p in mesh.points])
    s2 = argmin([p[3] for p in mesh.points])

    d1 = geodesic_distance_to_vertex(mesh, geom, dec, s1)
    d2 = geodesic_distance_to_vertex(mesh, geom, dec, s2)
    d12 = geodesic_distance_to_vertices(mesh, geom, dec, [s1, s2])

    @test maximum(abs.(d12 .- min.(d1, d2))) < 1e-12
end

@testset "Shortest path extraction" begin
    mesh = generate_icosphere(1.0, 1)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    src = argmax([p[3] for p in mesh.points])
    dst = argmin([p[3] for p in mesh.points])

    d = geodesic_distance_to_vertex(mesh, geom, dec, src)
    path = shortest_path_vertices(mesh, geom, dec, src, dst; distance=d)

    @test first(path) == src
    @test last(path) == dst
    @test length(path) >= 2

    Lp = _path_length(mesh, geom, path)
    @test Lp >= d[dst] - 1e-10
    @test Lp <= 1.7 * d[dst] + 1e-8
end

@testset "Point-to-point shortest path extraction" begin
    mesh = generate_icosphere(1.0, 1)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    src = argmax([p[3] for p in mesh.points])
    dst = argmin([p[3] for p in mesh.points])

    psrc = mesh.points[src] .+ SVector(0.01, 0.0, 0.0)
    pdst = mesh.points[dst] .+ SVector(-0.01, 0.0, 0.0)
    ppath = shortest_path_points(mesh, geom, dec, psrc, pdst)

    @test length(ppath) >= 2
    @test ppath[1] == mesh.points[src]
    @test ppath[end] == mesh.points[dst]
end

@testset "Intrinsic ball and geodesic FPS" begin
    mesh = generate_icosphere(1.0, 1)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    src = 1
    ball = intrinsic_ball(mesh, geom, dec, src, 0.6)
    d = geodesic_distance_to_vertex(mesh, geom, dec, src)

    @test src in ball
    @test all(i -> d[i] <= 0.6 + 1e-12, ball)

    fps = farthest_point_sampling_geodesic(mesh, geom, dec, 5; seed=src)
    @test length(fps) == 5
    @test length(unique(fps)) == 5
    @test fps[1] == src
end
