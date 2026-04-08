using Test
using LinearAlgebra
using SparseArrays
using FrontIntrinsicOps

@testset "Surface Whitney de Rham sequence" begin
    mesh = generate_torus(1.0, 0.35, 14, 12)
    geom = compute_geometry(mesh)

    complex = build_de_rham_sequence(mesh, geom; family=:whitney, mass=:consistent)
    topo = build_topology(mesh)

    @test size(complex.d0) == (length(topo.edges), length(mesh.points))
    @test size(complex.d1) == (length(mesh.faces), length(topo.edges))

    @test verify_subcomplex(complex; atol=1e-12)

    rpt = de_rham_report(complex)
    @test rpt.ndofs0 == length(mesh.points)
    @test rpt.ndofs1 == length(topo.edges)
    @test rpt.ndofs2 == length(mesh.faces)

    @test rpt.nnz_d0 == nnz(complex.d0)
    @test rpt.nnz_d1 == nnz(complex.d1)

    @test rpt.mass_symmetry.M0 < 1e-11
    @test rpt.mass_symmetry.M1 < 1e-11
    @test rpt.mass_symmetry.M2 < 1e-13

    @test rpt.positive_definiteness.M0.min_diag > 0
    @test rpt.positive_definiteness.M2.min_diag > 0

    comm = verify_commuting_projection(mesh, geom; tests=(:k01, :k12), atol=1e-10)
    @test comm.pass
end

@testset "Curve Whitney sequence" begin
    mesh = sample_circle(1.0, 48)
    geom = compute_geometry(mesh)

    complex = build_de_rham_sequence(mesh, geom)
    @test size(complex.d1, 1) == 0
    @test size(complex.M2) == (0, 0)

    comm = verify_commuting_projection(mesh, geom; tests=(:k01,), atol=1e-11)
    @test comm.pass
end
