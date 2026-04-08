using Test
using LinearAlgebra
using StaticArrays
using FrontIntrinsicOps

# -----------------------------------------------------------------------------
# Single triangle checks
# -----------------------------------------------------------------------------

let
    pts = SVector{3,Float64}[
        SVector(0.0, 0.0, 0.0),
        SVector(1.0, 0.0, 0.0),
        SVector(0.0, 1.0, 0.0),
    ]
    faces = SVector{3,Int}[SVector(1, 2, 3)]
    mesh = SurfaceMesh{Float64}(pts, faces)
    geom = compute_geometry(mesh)
    topo = build_topology(mesh)

    tri = (pts[1], pts[2], pts[3])

    @testset "Whitney0 local basis partition of unity" begin
        b0 = whitney0_basis_local(tri)
        ξs = (SVector(0.2, 0.3), SVector(0.1, 0.1), SVector(0.3, 0.2))
        for ξ in ξs
            @test abs((b0[1](ξ) + b0[2](ξ) + b0[3](ξ)) - 1.0) < 1e-14
        end
    end

    @testset "Whitney0 affine reproduction" begin
        f = x -> 2.0 * x[1] - 3.0 * x[2] + 1.0
        c0 = [f(p) for p in pts]
        rec = reconstruct_0form_face(c0, 1, mesh, geom)

        ξs = (SVector(0.2, 0.3), SVector(0.1, 0.2), SVector(0.45, 0.1))
        for ξ in ξs
            λ = SVector(1 - ξ[1] - ξ[2], ξ[1], ξ[2])
            x = λ[1] * pts[1] + λ[2] * pts[2] + λ[3] * pts[3]
            @test abs(rec.eval(ξ) - f(x)) < 1e-13
        end
    end

    @testset "Whitney1 orientation Kronecker-like DOFs" begin
        fe = topo.face_edges[1]
        fs = topo.face_edge_signs[1]
        ne = length(topo.edges)

        for k in 1:3
            c1 = zeros(Float64, ne)
            c1[fe[k]] = 1.0
            rec = reconstruct_1form_face(c1, 1, mesh, geom)
            @test abs(rec.coefficients[k] - fs[k]) < 1e-14
            for j in 1:3
                if j != k
                    @test abs(rec.coefficients[j]) < 1e-14
                end
            end
        end
    end

    @testset "Whitney1 reconstructed vector is tangential" begin
        c1 = [0.3, -0.7, 1.2]
        v = reconstruct_1form(c1, mesh, geom; representation=:facewise_tangent)[1]
        @test abs(dot(v, geom.face_normals[1])) < 1e-12
    end

    @testset "Whitney2 face integral consistency" begin
        c2 = [2.5]
        rec2 = reconstruct_2form_face(c2, 1, mesh, geom)
        @test abs(rec2.density * geom.face_areas[1] - c2[1]) < 1e-14
    end
end

# -----------------------------------------------------------------------------
# Two-triangle patch: shared-edge sign/continuity check
# -----------------------------------------------------------------------------

let
    mesh = make_flat_patch(N=1, L=1.0)
    geom = compute_geometry(mesh)
    topo = build_topology(mesh)

    ne = length(topo.edges)
    c1 = [sin(0.3 * i) for i in 1:ne]

    # Pick the unique interior shared edge.
    shared = findfirst(ei -> length(topo.edge_faces[ei]) == 2, 1:ne)
    @test shared !== nothing

    faces_sh = topo.edge_faces[shared]
    fiA, fiB = faces_sh[1], faces_sh[2]

    recA = reconstruct_1form_face(c1, fiA, mesh, geom)
    recB = reconstruct_1form_face(c1, fiB, mesh, geom)

    function midpoint_bary_for_local_edge(k::Int)
        if k == 1
            return SVector(0.5, 0.5, 0.0)
        elseif k == 2
            return SVector(0.0, 0.5, 0.5)
        elseif k == 3
            return SVector(0.5, 0.0, 0.5)
        end
        error("invalid local edge")
    end

    kA = findfirst(==(shared), topo.face_edges[fiA])
    kB = findfirst(==(shared), topo.face_edges[fiB])
    ξA = midpoint_bary_for_local_edge(kA)
    ξB = midpoint_bary_for_local_edge(kB)

    e = topo.edges[shared]
    t = normalize(mesh.points[e[2]] - mesh.points[e[1]])

    ta = dot(recA.eval(ξA), t)
    tb = dot(recB.eval(ξB), t)

    @test abs(ta - tb) < 1e-10
end
