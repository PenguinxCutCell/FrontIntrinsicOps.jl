# Commuting projections

This page describes the canonical lowest-order FEEC interpolation operators.

## Mathematical objects and DOFs

- `Π0`: scalar 0-form interpolation by vertex evaluation.
- `Π1`: 1-form interpolation by oriented edge integrals.
- `Π2`: 2-form interpolation by oriented face integrals.

## Commuting identities

For suitable smooth fields, the layer enforces the canonical discrete identities:

- `Π1(df) ≈ d0 Π0(f)`
- `Π2(dα) ≈ d1 Π1(α)`

`projection_commutator_01` and `projection_commutator_12` expose residual vectors for these checks.

## Orientation and storage convention

- Edge DOFs follow canonical edge orientation (`i<j`).
- Face DOFs follow mesh face orientation.
- The returned cochains are dense Julia vectors (`Vector{T}`) in the package's global simplex ordering.

## API

- `interpolate_0form`, `interpolate_1form`, `interpolate_2form`
- aliases: `Π0`, `Π1`, `Π2`
- helpers: `interpolate_exact_gradient`, `interpolate_exact_flux_density`
- diagnostics: `projection_commutator_01`, `projection_commutator_12`, `verify_commuting_projection`

## Minimal example

```julia
using FrontIntrinsicOps, LinearAlgebra

mesh = generate_icosphere(1.0, 1)
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)

f = x -> x[1]^2 - 0.3x[2] + 0.1
r = projection_commutator_01(f, mesh, geom, dec)
println(norm(r))
```

## Limitations and non-goals

- Canonical interpolators only (no smoothed bounded projections yet).
- Lowest-order simplicial setting only.
- `Π2` currently targets surface meshes (no curve 2-cells).
