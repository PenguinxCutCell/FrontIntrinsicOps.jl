#!/usr/bin/env julia

# Full Hodge decomposition demo on a torus:
# ω = dα + δβ + h

using FrontIntrinsicOps
using LinearAlgebra
using Printf

function main()
    mesh = generate_torus(2.0, 0.7, 24, 32)
    geom = compute_geometry(mesh)
    dec = build_dec(mesh, geom)
    topo = build_topology(mesh)

    b = betti_numbers(mesh, topo)
    @printf("β = (β0=%d, β1=%d, β2=%d)\n", b.β0, b.β1, b.β2)

    H = harmonic_basis(mesh, geom, dec)
    @printf("harmonic basis columns = %d (expected β1)\n", size(H, 2))

    # Manufactured components.
    α_true = [0.4 * p[1] - 0.2 * p[2] + 0.1 * p[3] for p in mesh.points]
    exact_true = dec.d0 * α_true

    harmonic_true = size(H, 2) >= 2 ? (0.75 .* H[:, 1] .- 0.35 .* H[:, 2]) :
                                      zeros(eltype(exact_true), length(exact_true))

    ω = exact_true .+ harmonic_true

    decomp = hodge_decomposition_full(ω, mesh, geom, dec; basis=H)

    hnorm(v) = sqrt(max(dot(v, dec.star1 * v), 0.0))
    relerr(a, b) = hnorm(a .- b) / (hnorm(a) + 1e-14)

    @printf("relative error exact    : %.3e\n", relerr(exact_true, decomp.exact))
    @printf("relative error harmonic : %.3e\n", relerr(harmonic_true, decomp.harmonic))
    @printf("relative coexact leakage: %.3e\n", hnorm(decomp.coexact) / (hnorm(ω) + 1e-14))
    @printf("relative residual       : %.3e\n", hnorm(decomp.residual) / (hnorm(ω) + 1e-14))

    ips = hodge_inner_products(mesh, geom, dec, (;
        exact = decomp.exact,
        coexact = decomp.coexact,
        harmonic = decomp.harmonic,
    ))
    @printf("⟨exact,coexact⟩    = %.3e\n", ips.exact_coexact)
    @printf("⟨exact,harmonic⟩   = %.3e\n", ips.exact_harmonic)
    @printf("⟨coexact,harmonic⟩ = %.3e\n", ips.coexact_harmonic)
end

main()
