# test_harmonic_basis.jl – Topology-aware harmonic basis and full Hodge split.

using Test
using LinearAlgebra
using StaticArrays
using FrontIntrinsicOps

_hodge_norm(v, dec) = sqrt(max(dot(v, dec.star1 * v), 0.0))

function _generate_genus2_surface(; R=2.0, r=0.7, n_major=12, n_minor=16, offset=3.0)
    m1 = generate_torus(R, r, n_major, n_minor)
    m2 = generate_torus(R, r, n_major, n_minor)
    m1 = SurfaceMesh{Float64}([p + SVector(-offset, 0.0, 0.0) for p in m1.points], m1.faces)
    m2 = SurfaceMesh{Float64}([p + SVector( offset, 0.0, 0.0) for p in m2.points], m2.faces)

    rm1 = 1
    rm2 = 1
    f1 = m1.faces[rm1]
    f2 = m2.faces[rm2]
    off = length(m1.points)

    points = vcat(m1.points, m2.points)
    faces = SVector{3,Int}[]

    for (i, f) in enumerate(m1.faces)
        i == rm1 && continue
        push!(faces, f)
    end
    for (i, f) in enumerate(m2.faces)
        i == rm2 && continue
        push!(faces, SVector{3,Int}(f[1] + off, f[2] + off, f[3] + off))
    end

    a1, a2, a3 = f1
    b1, b2, b3 = (f2[1] + off, f2[3] + off, f2[2] + off)
    append!(faces, [
        SVector(a1, a2, b2),
        SVector(a1, b2, b1),
        SVector(a2, a3, b3),
        SVector(a2, b3, b2),
        SVector(a3, a1, b1),
        SVector(a3, b1, b3),
    ])

    return SurfaceMesh{Float64}(points, faces)
end

@testset "Betti numbers and harmonic basis on sphere" begin
    mesh = generate_icosphere(1.0, 1)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    b = betti_numbers(mesh)
    @test b.β0 == 1
    @test b.β1 == 0
    @test b.β2 == 1
    @test first_betti_number(mesh) == 0

    H = harmonic_basis(mesh, geom, dec)
    @test size(H, 1) == length(build_topology(mesh).edges)
    @test size(H, 2) == 0

    α = dec.d0 * [p[3] for p in mesh.points]
    decomp = hodge_decomposition_full(α, mesh, geom, dec)
    @test _hodge_norm(decomp.harmonic, dec) / (_hodge_norm(α, dec) + 1e-14) < 5e-2
end

@testset "Genus-2 regression: Betti and harmonic rank" begin
    mesh = _generate_genus2_surface()
    @test is_closed(mesh)
    @test is_manifold(mesh)
    @test has_consistent_orientation(mesh)

    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    b = betti_numbers(mesh)
    @test b.β0 == 1
    @test b.β1 == 4
    @test b.β2 == 1

    cyc = cycle_basis(mesh)
    @test length(cyc) == 4

    H = harmonic_basis(mesh, geom, dec)
    @test size(H, 2) == 4

    for j in 1:size(H, 2)
        @test is_closed_form(view(H, :, j), dec; atol=2e-6)
        @test is_coclosed_form(view(H, :, j), dec; atol=2e-6)
    end
end

@testset "Betti numbers and harmonic basis on torus" begin
    mesh = generate_torus(2.0, 0.8, 18, 24)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    b = betti_numbers(mesh)
    @test b.β0 == 1
    @test b.β1 == 2
    @test b.β2 == 1

    cyc = cycle_basis(mesh)
    @test length(cyc) == 2

    C = cohomology_basis_1(mesh, geom, dec)
    @test size(C, 2) == 2

    H = harmonic_basis(mesh, geom, dec)
    @test size(H, 2) == 2

    # Closed and coclosed to tolerance.
    for j in 1:size(H, 2)
        @test is_closed_form(view(H, :, j), dec; atol=5e-6)
        @test is_coclosed_form(view(H, :, j), dec; atol=5e-6)
    end

    # Hodge-orthonormality.
    G = H' * (dec.star1 * H)
    @test norm(G - I, Inf) < 1e-5
end

@testset "Manufactured full decomposition recovery" begin
    mesh = generate_torus(2.2, 0.7, 20, 24)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    H = harmonic_basis(mesh, geom, dec)
    @test size(H, 2) == 2

    ne = length(build_topology(mesh).edges)

    α0 = [sin(2π * p[1] / 3) + 0.3 * p[3] for p in mesh.points]
    exact_true = dec.d0 * α0

    coexact_true = zeros(Float64, ne)

    harmonic_true = 0.7 .* H[:, 1] .- 0.4 .* H[:, 2]

    ω = exact_true .+ coexact_true .+ harmonic_true
    decomp = hodge_decomposition_full(ω, mesh, geom, dec; basis=H)

    e_err = _hodge_norm(decomp.exact .- exact_true, dec) / (_hodge_norm(exact_true, dec) + 1e-14)
    h_err = _hodge_norm(decomp.harmonic .- harmonic_true, dec) / (_hodge_norm(harmonic_true, dec) + 1e-14)
    r_rel = _hodge_norm(decomp.residual, dec) / (_hodge_norm(ω, dec) + 1e-14)

    @test e_err < 1.5e-1
    @test h_err < 1e-2
    @test r_rel < 1e-8
    @test _hodge_norm(decomp.coexact, dec) / (_hodge_norm(ω, dec) + 1e-14) < 5e-3

    # Pairwise orthogonality in the Hodge inner product.
    ips = hodge_inner_products(mesh, geom, dec, (;
        exact=decomp.exact,
        coexact=decomp.coexact,
        harmonic=decomp.harmonic,
    ))
    denom = _hodge_norm(ω, dec)^2 + 1e-14
    @test abs(ips.exact_harmonic) / denom < 1e-2
    @test abs(ips.exact_coexact) / denom < 1e-2
end
