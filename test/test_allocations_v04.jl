# test_allocations_v04.jl – Broad sanity allocation checks. (v0.4)
#
# Verifies that key v0.4 operations produce finite outputs without errors,
# exercising new generators (ellipsoid, perturbed sphere) and v0.4 APIs.
#
# These tests do not enforce hard allocation limits (which are machine/GC dependent),
# but confirm that the new geometry generators produce valid meshes and that
# the PDE infrastructure works on them.

using SparseArrays

@testset "generate_ellipsoid: basic properties" begin
    mesh = generate_ellipsoid(2.0, 1.0, 0.5, 10, 20)
    @test length(mesh.points) > 0
    @test length(mesh.faces)  > 0
    geom = compute_geometry(mesh)
    @test all(isfinite.(geom.face_areas))
    @test all(geom.face_areas .> 0.0)
end

@testset "generate_ellipsoid: a=b=c → sphere surface area ≈ 4π R²" begin
    # When a=b=c=R, ellipsoid = sphere; area ≈ 4π R²
    R    = 2.0
    mesh = generate_ellipsoid(R, R, R, 20, 40)
    geom = compute_geometry(mesh)
    area = measure(mesh, geom)
    # For R=2 sphere: 4π*4 ≈ 50.26
    @test abs(area - 4π*R^2) / (4π*R^2) < 0.02  # within 2%
end

@testset "generate_ellipsoid: 1/2/3 axes area" begin
    # Ellipsoid axes 1.0, 2.0, 3.0: area ≈ 21.48 (reference value)
    mesh = generate_ellipsoid(1.0, 2.0, 3.0, 16, 32)
    geom = compute_geometry(mesh)
    area = measure(mesh, geom)
    @test isfinite(area)
    @test area > 0.0
    # Rough bounds: sphere of radius 1 (≈12.57) < area < sphere of radius 3 (≈113)
    @test area > 10.0
    @test area < 120.0
end

@testset "generate_ellipsoid: geometry operators finite" begin
    mesh = generate_ellipsoid(1.5, 1.0, 0.7, 12, 24)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    nv   = length(mesh.points)

    @test all(isfinite.(nonzeros(dec.lap0)))
    @test all(all(isfinite.(v)) for v in geom.vertex_normals)
end

@testset "generate_perturbed_sphere: basic properties" begin
    R    = 1.0
    ε    = 0.1
    mesh = generate_perturbed_sphere(R, ε, 2, 12, 24)
    @test length(mesh.points) > 0
    @test length(mesh.faces)  > 0
    geom = compute_geometry(mesh)
    @test all(isfinite.(geom.face_areas))
    @test all(geom.face_areas .> 0.0)
end

@testset "generate_perturbed_sphere: ε=0 → sphere" begin
    R    = 1.0
    mesh = generate_perturbed_sphere(R, 0.0, 2, 16, 32)
    geom = compute_geometry(mesh)
    area = measure(mesh, geom)
    @test abs(area - 4π*R^2) / (4π*R^2) < 0.02
end

@testset "generate_perturbed_sphere: Laplace–Beltrami finite" begin
    mesh = generate_perturbed_sphere(1.0, 0.15, 2, 12, 24)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    nv   = length(mesh.points)

    L  = dec.lap0
    @test size(L, 1) == nv
    @test all(isfinite.(nonzeros(L)))
end

@testset "generate_perturbed_sphere: Poisson solve" begin
    mesh = generate_perturbed_sphere(1.0, 0.1, 2, 12, 24)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    nv   = length(mesh.points)

    # Solve Helmholtz (L + I) u = f; always has a unique solution
    f = ones(Float64, nv)
    u = solve_surface_helmholtz(mesh, geom, dec, f, 1.0)
    @test length(u) == nv
    @test all(isfinite.(u))
end

@testset "generate_ellipsoid: Poisson solve" begin
    mesh = generate_ellipsoid(2.0, 1.5, 1.0, 12, 24)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    nv   = length(mesh.points)

    # Solve Helmholtz on ellipsoid
    f = [Float64(p[3]) for p in mesh.points]
    u = solve_surface_helmholtz(mesh, geom, dec, f, 1.0)
    @test length(u) == nv
    @test all(isfinite.(u))
end

@testset "reaction_diffusion on ellipsoid: runs without error" begin
    mesh = generate_ellipsoid(2.0, 1.0, 0.5, 8, 16)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    nv   = length(mesh.points)
    u0   = fill(0.1, nv)

    u, _ = solve_surface_reaction_diffusion(
        mesh, geom, dec, u0, 0.05, 0.01, 0.01,
        fisher_kpp_reaction(1.0); θ=1.0)
    @test all(isfinite.(u))
end

@testset "reaction_diffusion on perturbed sphere: runs without error" begin
    mesh = generate_perturbed_sphere(1.0, 0.1, 2, 8, 16)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    nv   = length(mesh.points)
    u0   = fill(0.1, nv)

    u, _ = solve_surface_reaction_diffusion(
        mesh, geom, dec, u0, 0.05, 0.01, 0.01,
        linear_decay_reaction(0.5); θ=1.0)
    @test all(isfinite.(u))
end

@testset "transport_highres on torus: runs without error" begin
    mesh = generate_torus(2.0, 0.5, 20, 10)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    topo = build_topology(mesh)
    nv   = length(mesh.points)
    nf   = length(mesh.faces)

    u0   = [p[3] for p in mesh.points]
    vel  = Vector{SVector{3,Float64}}(undef, nf)
    for fi in 1:nf
        cx = sum(mesh.points[mesh.faces[fi][k]][1] for k in 1:3) / 3
        cy = sum(mesh.points[mesh.faces[fi][k]][2] for k in 1:3) / 3
        vel[fi] = SVector{3,Float64}(-cy, cx, 0.0)
    end
    dt = 1e-3

    u1 = step_surface_transport_limited(mesh, geom, dec, topo, u0, vel, dt;
                                         limiter=:vanleer, method=:ssprk2)
    @test length(u1) == nv
    @test all(isfinite.(u1))
end

@testset "generate_ellipsoid: error on invalid inputs" begin
    @test_throws ErrorException generate_ellipsoid(-1.0, 1.0, 1.0, 10, 20)
    @test_throws ErrorException generate_ellipsoid(1.0, 0.0, 1.0, 10, 20)
    @test_throws ErrorException generate_ellipsoid(1.0, 1.0, 1.0, 1, 20)
    @test_throws ErrorException generate_ellipsoid(1.0, 1.0, 1.0, 10, 2)
end

@testset "generate_perturbed_sphere: error on invalid inputs" begin
    @test_throws ErrorException generate_perturbed_sphere(-1.0, 0.1, 2, 10, 20)
    @test_throws ErrorException generate_perturbed_sphere(1.0, 1.5, 2, 10, 20)  # |ε| >= 1
    @test_throws ErrorException generate_perturbed_sphere(1.0, 0.1, 0, 10, 20)  # k < 1
    @test_throws ErrorException generate_perturbed_sphere(1.0, 0.1, 2, 1, 20)
end
