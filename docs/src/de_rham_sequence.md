# Discrete de Rham sequence

The FEEC layer makes the sequence explicit in code and diagnostics.

## Sequence

On triangulated surfaces:

`0 -> Λh0 --d0--> Λh1 --d1--> Λh2 -> 0`

where:
- `Λh0`: vertex DOFs,
- `Λh1`: oriented edge DOFs,
- `Λh2`: oriented face DOFs.

## API

- `build_whitney_complex(mesh, geom)`
- `build_de_rham_sequence(mesh, geom)`
- `de_rham_report(complex)`
- `verify_subcomplex(complex)`

## What `de_rham_report` provides

- DOF counts (`ndofs0`, `ndofs1`, `ndofs2`),
- sparsity metrics (`nnz_d0`, `nnz_d1`, `nnz_M*`),
- exactness residual `||d1*d0||`,
- mass symmetry and basic positivity diagnostics.

## Minimal example

```julia
using FrontIntrinsicOps

mesh = generate_torus(1.0, 0.35, 12, 10)
geom = compute_geometry(mesh)

W = build_de_rham_sequence(mesh, geom)
println(verify_subcomplex(W; atol=1e-12))
println(de_rham_report(W).exactness_residuals)
```

## Limitations and non-goals

- This is an explicit *lowest-order* sequence layer.
- No higher-order `P_r` / `P_r^-` families yet.
- No replacement of existing DEC core operators.
