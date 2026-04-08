# test_parallel_transport.jl – Tangent-frame parallel transport and holonomy.

using Test
using LinearAlgebra
using StaticArrays
using FrontIntrinsicOps

function _adjacent_face_pair(mesh::SurfaceMesh)
    topo = build_topology(mesh)
    for (ei, ef) in enumerate(topo.edge_faces)
        if length(ef) == 2
            return ef[1], ef[2], ei
        end
    end
    error("No adjacent face pair found.")
end

function _ordered_faces_around_vertex(mesh::SurfaceMesh{T}, geom::SurfaceGeometry{T}, vid::Int) where {T}
    vf = vertex_to_faces(mesh)[vid]
    frames = vertex_tangent_frames(mesh, geom)
    Fv = frames[vid]
    p0 = mesh.points[vid]

    vals = Vector{Tuple{Float64,Int}}()
    for fi in vf
        f = mesh.faces[fi]
        c = (mesh.points[f[1]] + mesh.points[f[2]] + mesh.points[f[3]]) / 3
        v = c - p0
        x = dot(v, Fv[:, 1])
        y = dot(v, Fv[:, 2])
        push!(vals, (atan(y, x), fi))
    end
    sort!(vals, by = x -> x[1])
    return [x[2] for x in vals]
end

@testset "Parallel transport on flat patch" begin
    mesh = make_flat_patch(N=12, L=1.0)
    geom = compute_geometry(mesh)

    f1, f2, ei = _adjacent_face_pair(mesh)
    R12 = transport_matrix_across_edge(mesh, geom, f1, f2, ei)
    R21 = transport_matrix_across_edge(mesh, geom, f2, f1, ei)

    v = SVector(0.7, -0.3)
    vf = parallel_transport_face_vector(v, mesh, geom, f1, f2, ei)
    vb = parallel_transport_face_vector(vf, mesh, geom, f2, f1, ei)

    @test norm(vf) ≈ norm(v) atol=1e-12
    @test norm(vb - v) < 1e-10
    @test norm(R21 * R12 - I) < 1e-10

    bnd = Set(detect_boundary_vertices(mesh, build_topology(mesh)))
    center = findfirst(v -> !(v in bnd), 1:length(mesh.points))
    cycle = _ordered_faces_around_vertex(mesh, geom, center)
    hol = holonomy_along_cycle(mesh, geom, cycle)
    @test abs(hol) < 1e-6
end

@testset "Parallel transport on sphere" begin
    mesh = generate_icosphere(1.0, 2)
    geom = compute_geometry(mesh)

    # Holonomy around a small face cycle around one vertex should be nonzero.
    vid = 1
    cycle = _ordered_faces_around_vertex(mesh, geom, vid)
    hol = holonomy_along_cycle(mesh, geom, cycle)
    hol_rev = holonomy_along_cycle(mesh, geom, reverse(cycle))

    @test abs(hol) > 1e-3
    @test abs(hol + hol_rev) < 1e-8

    # Norm preservation along a face path.
    path = cycle[1:min(4, length(cycle))]
    v = SVector(0.25, 0.6)
    vt = parallel_transport_along_face_path(v, mesh, geom, path)
    @test abs(norm(vt) - norm(v)) < 1e-10

    # Same-face path is identity.
    @test parallel_transport_along_face_path(v, mesh, geom, [path[1]]) == v
end

@testset "Vertex vector transport" begin
    mesh = generate_icosphere(1.0, 1)
    geom = compute_geometry(mesh)

    src = 1
    dst = argmax([p[3] for p in mesh.points])

    nsrc = geom.vertex_normals[src]
    v0 = tangential_project(SVector(0.7, -0.2, 0.4), nsrc)

    v_same = parallel_transport_vertex_vector(v0, mesh, geom, src, src)
    @test norm(v_same - v0) < 1e-12

    v1 = parallel_transport_vertex_vector(v0, mesh, geom, src, dst)
    @test abs(norm(v1) - norm(v0)) / (norm(v0) + 1e-14) < 5e-2

    # Reversal with explicit face paths.
    fL, fR, _ = _adjacent_face_pair(mesh)
    fpath = [fL, fR]
    v2 = parallel_transport_vertex_vector(v0, mesh, geom, src, dst; path=fpath)
    v3 = parallel_transport_vertex_vector(v2, mesh, geom, dst, src; path=reverse(fpath))
    @test norm(v3 - v0) / (norm(v0) + 1e-14) < 1e-1
end

@testset "Edge 1-form transport helper" begin
    mesh = make_flat_patch(N=12, L=1.0)
    geom = compute_geometry(mesh)
    topo = build_topology(mesh)

    # Exact 1-form from linear x scalar field (constant covector on flat patch).
    dec = build_dec(mesh, geom)
    ω = dec.d0 * [p[1] for p in mesh.points]

    src_edge = 1
    dst_edge = argmax(geom.edge_lengths)
    val = transport_edge_1form(ω, mesh, geom, src_edge, dst_edge)

    @test transport_edge_1form(ω, mesh, geom, src_edge, src_edge) == ω[src_edge]
    @test isfinite(val)
    @test abs(val - ω[dst_edge]) / (abs(ω[dst_edge]) + 1e-14) < 1e-6

    # Sphere sanity check (finite transport value).
    ms = generate_icosphere(1.0, 1)
    gs = compute_geometry(ms)
    ds = build_dec(ms, gs)
    ts = build_topology(ms)
    ωs = ds.d0 * [p[3] for p in ms.points]
    v2 = transport_edge_1form(ωs, ms, gs, 1, length(ts.edges))
    @test isfinite(v2)
end
