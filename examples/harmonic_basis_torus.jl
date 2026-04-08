#!/usr/bin/env julia

# Harmonic basis demo on a genus-1 torus.

using FrontIntrinsicOps
using LinearAlgebra
using Printf

function main()
    mesh = generate_torus(2.0, 0.7, 24, 32)
    geom = compute_geometry(mesh)
    dec = build_dec(mesh, geom)

    b = betti_numbers(mesh)
    @printf("β = (β0=%d, β1=%d, β2=%d)\n", b.β0, b.β1, b.β2)

    cycles = cycle_basis(mesh)
    @printf("cycle generators = %d\n", length(cycles))

    C = cohomology_basis_1(mesh, geom, dec)
    @printf("cohomology reps = %d\n", size(C, 2))

    H = harmonic_basis(mesh, geom, dec)
    @printf("harmonic basis columns = %d\n", size(H, 2))

    for j in 1:size(H, 2)
        r = harmonic_residuals(H[:, j], mesh, geom, dec)
        @printf("h[%d]: ||d h||=%.3e, ||δ h||=%.3e\n", j, r.d_norm, r.δ_norm)
    end

    if size(H, 2) > 0
        G = H' * (dec.star1 * H)
        @printf("orthonormality error ||H'⋆1H-I||∞ = %.3e\n", norm(G - I, Inf))
    end
end

main()
