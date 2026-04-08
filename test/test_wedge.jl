# test_wedge.jl – Discrete wedge product checks.

using Test
using LinearAlgebra
using FrontIntrinsicOps

@testset "Surface wedge antisymmetry and self-wedge" begin
    mesh = generate_icosphere(1.0, 1)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    topo = build_topology(mesh)

    u = [p[1] + 0.3 * p[3] for p in mesh.points]
    v = [p[2] - 0.2 * p[3] for p in mesh.points]
    α = dec.d0 * u
    β = dec.d0 * v

    wαβ = wedge11(α, β, mesh, geom, dec)
    wβα = wedge11(β, α, mesh, geom, dec)

    denom = max(norm(wαβ), 1e-14)
    @test norm(wαβ .+ wβα) / denom < 1e-8

    wαα = wedge11(α, α, mesh, geom, dec)
    @test norm(wαα) / (norm(wαβ) + 1e-14) < 1e-8
end

@testset "Surface wedge 0-form scaling conventions" begin
    mesh = generate_torus(2.0, 0.7, 16, 20)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    topo = build_topology(mesh)

    f0 = [1.0 + 0.1 * p[1] - 0.2 * p[2] for p in mesh.points]
    g0 = [0.5 + 0.3 * p[3] for p in mesh.points]
    α = dec.d0 * [sin(p[1]) for p in mesh.points]
    τ = [cos(0.7 * c[3]) for c in [ (mesh.points[f[1]] + mesh.points[f[2]] + mesh.points[f[3]]) / 3 for f in mesh.faces ]]

    @test wedge(f0, g0, mesh, geom, dec) ≈ f0 .* g0 atol=1e-14

    w01 = wedge(f0, α, mesh, geom, dec)
    w10 = wedge(α, f0, mesh, geom, dec)
    @test w01 ≈ w10 atol=1e-14

    manual01 = similar(α)
    for (ei, e) in enumerate(topo.edges)
        i, j = e[1], e[2]
        manual01[ei] = 0.5 * (f0[i] + f0[j]) * α[ei]
    end
    @test w01 ≈ manual01 atol=1e-14

    w02 = wedge(f0, τ, mesh, geom, dec)
    w20 = wedge(τ, f0, mesh, geom, dec)
    @test w02 ≈ w20 atol=1e-14
end

@testset "Curve wedge 0∧1 support" begin
    mesh = sample_circle(1.0, 32)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    f0 = [1.0 + 0.2 * p[1] for p in mesh.points]
    α = dec.d0 * [sin(2π * k / length(mesh.points)) for k in 1:length(mesh.points)]

    w = wedge(f0, α, mesh, geom, dec)
    w2 = wedge(α, f0, mesh, geom, dec)
    @test w ≈ w2 atol=1e-14
    @test length(w) == length(mesh.edges)
end
