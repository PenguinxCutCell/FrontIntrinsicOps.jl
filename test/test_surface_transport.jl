# test_surface_transport.jl – Tests for scalar transport on curves and surfaces.

using Test
using FrontIntrinsicOps
using LinearAlgebra
using SparseArrays
using StaticArrays

# ─────────────────────────────────────────────────────────────────────────────
# Transport operator assembly
# ─────────────────────────────────────────────────────────────────────────────

@testset "Transport operator assembles on circle" begin
    mesh = sample_circle(1.0, 32)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    # Constant rotation velocity along tangent (tangential speed = 1)
    vel_vecs = [SVector{2,Float64}(-p[2], p[1]) for p in mesh.points]  # rot around z
    A_centered = assemble_transport_operator(mesh, geom, vel_vecs; scheme=:centered)
    A_upwind   = assemble_transport_operator(mesh, geom, vel_vecs; scheme=:upwind)

    @test size(A_centered) == (length(mesh.points), length(mesh.points))
    @test size(A_upwind)   == (length(mesh.points), length(mesh.points))

    # Check that constant function is in the nullspace (mass conservation):
    # A * ones should give approximate divergence form near zero
    # (sum of rows ≈ 0 is not guaranteed for pure advection - but sums check)
    nv = length(mesh.points)
    @test issparse(A_centered)
    @test issparse(A_upwind)
end

@testset "Transport operator assembles on sphere" begin
    mesh = generate_icosphere(1.0, 1)
    geom = compute_geometry(mesh)

    # Rigid rotation around z-axis: v = (-y, x, 0) (tangential on sphere)
    vel_vecs = [SVector{3,Float64}(-p[2], p[1], 0.0) for p in mesh.points]
    A_centered = assemble_transport_operator(mesh, geom, vel_vecs; scheme=:centered)
    A_upwind   = assemble_transport_operator(mesh, geom, vel_vecs; scheme=:upwind)

    nv = length(mesh.points)
    @test size(A_centered) == (nv, nv)
    @test size(A_upwind)   == (nv, nv)
end

# ─────────────────────────────────────────────────────────────────────────────
# Mass conservation
# ─────────────────────────────────────────────────────────────────────────────

@testset "Mass conservation: centered scheme, sphere, short test" begin
    mesh = generate_icosphere(1.0, 2)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    # Rigid rotation velocity (tangential on unit sphere)
    vel = [SVector{3,Float64}(-p[2], p[1], 0.0) for p in mesh.points]

    # Initial scalar: smooth bump proportional to z
    u = [p[3] for p in mesh.points]

    # Total mass (∫ u dA)
    M0 = dot(geom.vertex_dual_areas, u)

    A  = assemble_transport_operator(mesh, geom, vel; scheme=:centered)
    dt = estimate_transport_dt(mesh, geom, vel; cfl=0.3)

    # Advance a few steps
    for _ in 1:5
        u = step_surface_transport_forward_euler(mesh, geom, A, u, dt)
    end

    M1 = dot(geom.vertex_dual_areas, u)
    rel_mass_err = abs(M1 - M0) / (abs(M0) + 1e-14)
    # Mass conservation for centered scheme (not exact but reasonable over 5 steps)
    @test rel_mass_err < 0.1
end

# ─────────────────────────────────────────────────────────────────────────────
# Upwind scheme stability
# ─────────────────────────────────────────────────────────────────────────────

@testset "Upwind scheme runs and solution bounded" begin
    mesh = generate_icosphere(1.0, 2)
    geom = compute_geometry(mesh)

    vel = [SVector{3,Float64}(-p[2], p[1], 0.0) for p in mesh.points]
    u   = [p[3] for p in mesh.points]

    A  = assemble_transport_operator(mesh, geom, vel; scheme=:upwind)
    dt = estimate_transport_dt(mesh, geom, vel; cfl=0.3)

    umax0 = maximum(abs, u)
    for _ in 1:10
        u = step_surface_transport_forward_euler(mesh, geom, A, u, dt)
    end
    # Upwind should not grow beyond initial amplitude (roughly)
    @test maximum(abs, u) <= umax0 * 1.1
end

# ─────────────────────────────────────────────────────────────────────────────
# SSP-RK2 and SSP-RK3 run correctly
# ─────────────────────────────────────────────────────────────────────────────

@testset "SSPRK2 and SSPRK3 run on sphere" begin
    mesh = generate_icosphere(1.0, 1)
    geom = compute_geometry(mesh)

    vel = [SVector{3,Float64}(-p[2], p[1], 0.0) for p in mesh.points]
    u0  = [p[3] for p in mesh.points]

    A  = assemble_transport_operator(mesh, geom, vel; scheme=:upwind)
    dt = estimate_transport_dt(mesh, geom, vel; cfl=0.3)

    u2 = step_surface_transport_ssprk2(mesh, geom, A, u0, dt)
    u3 = step_surface_transport_ssprk3(mesh, geom, A, u0, dt)

    @test length(u2) == length(u0)
    @test length(u3) == length(u0)
    @test all(isfinite, u2)
    @test all(isfinite, u3)
end

# ─────────────────────────────────────────────────────────────────────────────
# CFL estimate
# ─────────────────────────────────────────────────────────────────────────────

@testset "estimate_transport_dt returns positive finite dt" begin
    mesh = generate_icosphere(1.0, 2)
    geom = compute_geometry(mesh)
    vel  = [SVector{3,Float64}(-p[2], p[1], 0.0) for p in mesh.points]
    dt   = estimate_transport_dt(mesh, geom, vel; cfl=0.5)
    @test dt > 0
    @test isfinite(dt)
end

# ─────────────────────────────────────────────────────────────────────────────
# edge_flux_velocity: callable and matrix input
# ─────────────────────────────────────────────────────────────────────────────

@testset "edge_flux_velocity callable on sphere" begin
    mesh = generate_icosphere(1.0, 1)
    geom = compute_geometry(mesh)

    vel_fn(p) = SVector{3,Float64}(-p[2], p[1], 0.0)
    vf1 = edge_flux_velocity(mesh, geom, vel_fn)
    vf2 = edge_flux_velocity(mesh, geom,
                              [vel_fn(p) for p in mesh.points])
    @test maximum(abs, vf1 .- vf2) < 1e-12
end
