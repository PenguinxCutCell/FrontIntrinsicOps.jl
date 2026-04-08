# FEEC overview (lowest order)

`FrontIntrinsicOps.jl` now exposes a **lowest-order FEEC-compatible layer** on top of the existing DEC core.

This layer is additive:
- DEC operators (`d0`, `d1`, diagonal Hodge stars, existing PDE solvers) remain unchanged.
- FEEC adds Whitney spaces, canonical interpolation, and consistent variational assembly.

## Discrete de Rham sequence

On triangulated surfaces:

`0 -> Λh0 --d0--> Λh1 --d1--> Λh2 -> 0`

with DOFs:
- `Λh0`: vertex values,
- `Λh1`: oriented edge integrals,
- `Λh2`: oriented face integrals.

## Public entry points

- `build_whitney_complex(mesh, geom)`
- `build_de_rham_sequence(mesh, geom)`
- `de_rham_report(complex)`
- `verify_subcomplex(complex)`

## Storage conventions

- 0-form cochains: `Vector` length `nV`.
- 1-form cochains: `Vector` length `nE` in canonical oriented-edge order (`i<j`).
- 2-form cochains: `Vector` length `nF` in face order.

## Minimal example

```julia
using FrontIntrinsicOps

mesh = generate_icosphere(1.0, 2)
geom = compute_geometry(mesh)

W = build_whitney_complex(mesh, geom)
rpt = de_rham_report(W)

println(rpt.ndofs0, " ", rpt.ndofs1, " ", rpt.ndofs2)
println("d1*d0 residual = ", rpt.exactness_residuals.d1_d0)
```

## Limitations and non-goals

- Lowest-order only (`Whitney0/1/2`).
- Simplicial meshes only (curves/surfaces).
- Canonical FEEC interpolators only (no smoothed bounded projectors yet).
- No higher-order `P_r` / `P_r^-` families in this round.
