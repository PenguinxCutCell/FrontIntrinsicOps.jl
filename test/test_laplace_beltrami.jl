# test_laplace_beltrami.jl – Tests for the scalar Laplace–Beltrami operator.
#
# Convention note
# ---------------
# Our operator L = ⋆₀⁻¹ d₀ᵀ ⋆₁ d₀ is the *positive* Laplacian (−Δ_Γ).
# On a sphere of radius R, coordinate functions satisfy:
#   L x = (2/R²) x,  L y = (2/R²) y,  L z = (2/R²) z
# since Δ_Γ x = −(2/R²) x and L = −Δ_Γ.

using Test
using FrontIntrinsicOps
using LinearAlgebra
using Statistics: mean

@testset "Constant nullspace on sphere" begin
    R  = 1.0
    mesh = make_uvsphere(R; nφ=16, nθ=32)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    nv   = length(mesh.points)

    ones_v = ones(Float64, nv)
    Lu     = laplace_beltrami(mesh, geom, dec, ones_v)
    @test maximum(abs, Lu) < 1e-10
end

@testset "Coordinate eigenvalue on sphere" begin
    # L = −Δ_Γ (positive operator), so L x = (2/R²) x.
    R    = 2.0
    mesh = make_uvsphere(R; nφ=20, nθ=40)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    x = [p[1] for p in mesh.points]
    y = [p[2] for p in mesh.points]
    z = [p[3] for p in mesh.points]

    λ_expected = +2.0 / R^2   # positive because L = −Δ_Γ

    Lx = laplace_beltrami(mesh, geom, dec, x)
    Ly = laplace_beltrami(mesh, geom, dec, y)
    Lz = laplace_beltrami(mesh, geom, dec, z)

    mask = abs.(x) .> 0.1 * R
    if any(mask)
        ratios_x = Lx[mask] ./ x[mask]
        @test abs(mean(ratios_x) - λ_expected) / abs(λ_expected) < 0.15
    end

    mask = abs.(y) .> 0.1 * R
    if any(mask)
        ratios_y = Ly[mask] ./ y[mask]
        @test abs(mean(ratios_y) - λ_expected) / abs(λ_expected) < 0.15
    end

    mask = abs.(z) .> 0.1 * R
    if any(mask)
        ratios_z = Lz[mask] ./ z[mask]
        @test abs(mean(ratios_z) - λ_expected) / abs(λ_expected) < 0.15
    end
end

@testset "d1*d0 = 0 on sphere" begin
    mesh = make_uvsphere(1.0; nφ=12, nθ=24)
    dec  = build_dec(mesh, compute_geometry(mesh))
    residual = maximum(abs, dec.d1 * dec.d0)
    @test residual < 1e-14
end

@testset "DEC check report on sphere" begin
    R    = 1.0
    mesh = make_uvsphere(R; nφ=16, nθ=32)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    report = check_dec(mesh, geom, dec)
    @test report.d1_d0_zero
    @test report.lap_constant_nullspace
    @test report.star0_positive
end
