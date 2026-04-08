using Test
using LinearAlgebra
using StaticArrays
using FrontIntrinsicOps

@testset "FEEC projections on planar patch" begin
    mesh = make_flat_patch(N=6, L=1.0)
    geom = compute_geometry(mesh)
    dec = build_dec(mesh, geom)

    f = x -> x[1]^2 + 0.5 * x[2] - 0.3
    r01 = projection_commutator_01(f, mesh, geom, dec)
    @test norm(r01) < 1e-12

    αline = (x, t, ei) -> x[1] - 2.0 * x[2] + 0.1 * ei
    r12 = projection_commutator_12(αline, mesh, geom, dec; representation=:line_density)
    @test norm(r12) < 1e-12

    c0a = interpolate_0form(f, mesh, geom)
    c0b = Π0(f, mesh, geom)
    @test c0a == c0b

    c1a = interpolate_1form(αline, mesh, geom; representation=:line_density)
    c1b = Π1(αline, mesh, geom; representation=:line_density)
    @test c1a == c1b

    β = (x, n) -> 2.0 + 0.1 * dot(x, n)
    c2a = interpolate_2form(β, mesh, geom)
    c2b = Π2(β, mesh, geom)
    @test c2a == c2b
end

@testset "Sphere commuting sanity under refinement" begin
    errs = Float64[]

    for lev in (1, 2)
        mesh = generate_icosphere(1.0, lev)
        geom = compute_geometry(mesh)

        f = x -> x[1] * x[2] + 0.2 * x[3]
        gradf = x -> SVector{3,Float64}(x[2], x[1], 0.2)

        lhs = interpolate_1form(gradf, mesh, geom; representation=:ambient_vector)
        rhs = incidence_0(mesh) * interpolate_0form(f, mesh, geom)

        push!(errs, norm(lhs - rhs) / sqrt(length(lhs)))
    end

    @test errs[2] < errs[1]
end

@testset "Curve FEEC interpolation" begin
    mesh = sample_circle(1.0, 32)
    geom = compute_geometry(mesh)
    dec = build_dec(mesh, geom)

    f = x -> cos(x[1]) + sin(x[2])
    r01 = projection_commutator_01(f, mesh, geom, dec)
    @test norm(r01) < 1e-12

    speed = (x, t, ei) -> 1.0 + 0.1 * ei + 0.2 * dot(x, t)
    c1 = interpolate_1form(speed, mesh, geom; representation=:tangent_speed)
    @test length(c1) == length(mesh.edges)
    @test all(isfinite, c1)
end
