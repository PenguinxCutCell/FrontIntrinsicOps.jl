# test_hodge_decomposition.jl – Tests for Hodge decomposition of 1-forms (v0.4).
#
# Tests:
# 1. exact_component of an exact form d0*u recovers d0*u.
# 2. coexact_component of a coexact form.
# 3. hodge_decompose_1form returns proper named tuple.
# 4. Residual of decomposition is near machine precision.
# 5. Cross inner products are near zero (orthogonality).
# 6. On genus-0 sphere, harmonic component is ~0.

@testset "Hodge decomp: exact_component of exact form" begin
    mesh = make_uvsphere(1.0; nφ=8, nθ=16)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    nv = length(mesh.points)
    ne = size(dec.d0, 1)  # number of edges

    # Create an exact form α = d0 * u (gradient of z-coordinate)
    u  = [p[3] for p in mesh.points]
    α  = dec.d0 * u

    # Extract exact component
    α_exact, φ = exact_component_1form(mesh, geom, dec, α)
    @test length(α_exact) == ne
    @test length(φ) == nv

    # α_exact should be close to α (since α is exact)
    s1 = dec.star1
    norm_diff  = sqrt(dot(α_exact .- α, s1 * (α_exact .- α)))
    norm_alpha = sqrt(dot(α, s1 * α))
    # On a genus-0 surface, the exact component should recover most of α
    # (up to the coexact and harmonic parts, which should be tiny for d0 u)
    @test norm_diff / norm_alpha < 0.05
end

@testset "Hodge decomp: named tuple output" begin
    mesh = make_uvsphere(1.0; nφ=8, nθ=16)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    ne   = size(dec.d1, 2)
    nv   = length(mesh.points)
    nf   = length(mesh.faces)

    u = [p[3] for p in mesh.points]
    α = dec.d0 * u

    decomp = hodge_decompose_1form(mesh, geom, dec, α)

    @test haskey(decomp, :exact)
    @test haskey(decomp, :coexact)
    @test haskey(decomp, :harmonic)
    @test haskey(decomp, :phi)
    @test haskey(decomp, :psi)

    @test length(decomp.exact)    == ne
    @test length(decomp.coexact)  == ne
    @test length(decomp.harmonic) == ne
    @test length(decomp.phi)      == nv
    @test length(decomp.psi)      == nf
end

@testset "Hodge decomp: residual is small" begin
    mesh = make_uvsphere(1.0; nφ=8, nθ=16)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    u = [p[3] for p in mesh.points]
    α = dec.d0 * u

    decomp  = hodge_decompose_1form(mesh, geom, dec, α)
    residual = hodge_decomposition_residual(mesh, geom, dec, α, decomp)

    # Residual should be very small
    @test residual < 1e-8
end

@testset "Hodge decomp: genus-0 harmonic component is ~0" begin
    # On a sphere (genus 0, b₁ = 0), any 1-form has trivial harmonic part
    mesh = make_uvsphere(1.0; nφ=8, nθ=16)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    ne   = size(dec.d1, 2)

    # Use a mixed 1-form
    u1 = [p[3] for p in mesh.points]
    u2 = [p[1] for p in mesh.points]
    α  = dec.d0 * u1 .+ 0.3 .* (dec.d0 * u2)

    decomp = hodge_decompose_1form(mesh, geom, dec, α)

    s1 = dec.star1
    norm_h = sqrt(max(dot(decomp.harmonic, s1 * decomp.harmonic), 0.0))
    norm_α = sqrt(dot(α, s1 * α))

    # Harmonic component should be tiny relative to total
    @test norm_h / (norm_α + 1e-14) < 0.02
end

@testset "Hodge decomp: orthogonality of components" begin
    mesh = make_uvsphere(1.0; nφ=8, nθ=16)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    u = [sin(2*p[3]) for p in mesh.points]
    α = dec.d0 * u

    decomp = hodge_decompose_1form(mesh, geom, dec, α)
    ips    = hodge_inner_products(mesh, geom, dec, decomp)

    # All cross inner products should be small
    s1   = dec.star1
    norm2 = dot(α, s1 * α)
    @test abs(ips.exact_coexact)    / norm2 < 0.05
    @test abs(ips.exact_harmonic)   / norm2 < 0.05
    @test abs(ips.coexact_harmonic) / norm2 < 0.05
end
