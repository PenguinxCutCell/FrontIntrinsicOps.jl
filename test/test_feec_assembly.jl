using Test
using LinearAlgebra
using StaticArrays
using FrontIntrinsicOps

@testset "Whitney mass/stiffness assembly" begin
    mesh = make_flat_patch(N=6, L=1.0)
    geom = compute_geometry(mesh)

    M0 = assemble_whitney_mass0(mesh, geom)
    M1 = assemble_whitney_mass1(mesh, geom)
    M2 = assemble_whitney_mass2(mesh, geom)
    K0 = assemble_whitney_stiffness0(mesh, geom)

    @test norm(M0 - M0') < 1e-11
    @test norm(M1 - M1') < 1e-10
    @test norm(M2 - M2') < 1e-13
    @test norm(K0 - K0') < 1e-11

    @test minimum(diag(M0)) > 0
    @test minimum(diag(M2)) > 0
end

@testset "Whitney Laplacian nullspace and solve trend" begin
    errs = Float64[]

    for lev in (1, 2)
        mesh = generate_icosphere(1.0, lev)
        geom = compute_geometry(mesh)

        L0 = assemble_whitney_hodge_laplacian0(mesh, geom)
        onesv = ones(Float64, length(mesh.points))
        @test norm(L0 * onesv) < 1e-9

        u_exact = [p[3] for p in mesh.points]
        rhs = 2.0 .* u_exact

        sol = solve_mixed_hodge_laplacian0(mesh, geom, rhs; gauge=:mean_zero)
        u_num = copy(sol.u)
        u_ref = copy(u_exact)
        zero_mean_projection!(u_num, mesh, geom)
        zero_mean_projection!(u_ref, mesh, geom)

        push!(errs, norm(u_num - u_ref) / max(norm(u_ref), 1e-14))
    end

    @test errs[2] < errs[1]
end

@testset "DEC vs Whitney diagnostics" begin
    mesh = generate_icosphere(1.0, 2)
    geom = compute_geometry(mesh)

    cmpM = compare_dec_vs_whitney_mass(mesh, geom)
    @test cmpM.size_dec == cmpM.size_whitney
    @test cmpM.nnz_whitney > 0
    @test isfinite(cmpM.rel_diff)

    cmpL = compare_dec_vs_whitney_laplacian(mesh, geom)
    @test cmpL.size_dec == cmpL.size_whitney
    @test cmpL.nnz_whitney > 0
    @test cmpL.null_residual_whitney < 1e-8
end

@testset "Whitney 1-form mixed solve sanity" begin
    mesh = generate_torus(1.0, 0.35, 16, 12)
    geom = compute_geometry(mesh)
    topo = build_topology(mesh)

    f0 = interpolate_0form(p -> p[1], mesh, geom)
    rhs = incidence_0(mesh) * f0

    sol = solve_mixed_hodge_laplacian1(mesh, geom, rhs; gauge=:harmonic_orthogonal)
    @test length(sol.u) == length(topo.edges)
    @test all(isfinite, sol.u)

    H = sol.harmonic_basis
    if size(H, 2) > 0
        M1 = assemble_whitney_mass1(mesh, geom)
        @test norm(H' * (M1 * sol.u)) < 1e-6
    end
end
