# Topology-Aware DEC

This page documents the cohomology/harmonic layer on closed orientable triangulated surfaces.

## Mathematical object

For edge 1-cochains `ω ∈ Ω¹`, we use the discrete decomposition

`ω = dα + δβ + h`

with:

- `dα` exact (`im(d0)`),
- `δβ` coexact (`im(δ2)`),
- `h` harmonic (`ker(d1) ∩ ker(δ1)`).

The harmonic dimension is the first Betti number `β1` (for closed genus-`g` surfaces, `β1 = 2g`).

## Cochain space / degree

- `β`-functions and cycle/cohomology basis: topology of the primal mesh.
- `harmonic_basis`, `project_*`, and `hodge_decomposition_full`: edge 1-cochains (`length = nE`).
- Potentials:
  - `α` on vertices (`Ω⁰`, length `nV`),
  - `β` on faces (`Ω²`, length `nF`).

## Orientation convention

- Edge orientation is canonical `(i,j)` with `i < j`.
- Signed cycle entries:
  - `+e` means traversal along canonical edge orientation,
  - `-e` means opposite traversal.
- Inner product is the Hodge metric:
  - `⟨a,b⟩ = a' * star1 * b`.

## Storage convention

- All 1-forms are dense vectors in edge ordering from `build_topology(mesh).edges`.
- Harmonic basis is an `nE × β1` matrix; each column is one harmonic generator.

## API

```julia
betti_numbers
first_betti_number
cycle_basis
cohomology_basis_1
harmonic_basis
project_harmonic
project_exact
project_coexact
hodge_decomposition_full
is_closed_form
is_coclosed_form
harmonic_residuals
```

## Minimal example

```julia
using FrontIntrinsicOps
using LinearAlgebra

mesh = generate_torus(2.0, 0.7, 20, 24)
geom = compute_geometry(mesh)
dec = build_dec(mesh, geom)

b = betti_numbers(mesh)
H = harmonic_basis(mesh, geom, dec)

ω = dec.d0 * [p[3] for p in mesh.points] + 0.4 .* H[:, 1]
decomp = hodge_decomposition_full(ω, mesh, geom, dec; basis=H)

(β=b, nbasis=size(H, 2), residual=norm(decomp.residual))
```

## Limitations and non-goals

- Closed orientable surfaces only in v1; open-surface harmonic tools are not implemented here.
- Basis vectors are deterministic by construction, but harmonic sign/basis choices remain topologically equivalent (not unique analytically).
- Implementation is low-order simplicial DEC; no symbolic FEEC framework is introduced.
