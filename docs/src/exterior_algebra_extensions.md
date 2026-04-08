# Exterior Algebra Extensions

This page documents low-order practical operators added for DEC-form workflows.

## Mathematical object

Implemented additions:

- discrete wedge product (`wedge`, `wedge0k`, `wedge11`),
- interior product / contraction (`interior_product`),
- Lie derivative (`lie_derivative`) via Cartan formula:
  - `L_X α = i_X dα + d(i_X α)`.

## Cochain space / degree

### Surface support

- `0∧0 -> 0`
- `0∧1`, `1∧0 -> 1`
- `0∧2`, `2∧0 -> 2`
- `1∧1 -> 2`
- contraction:
  - `i_X : Ω¹ -> Ω⁰`,
  - `i_X : Ω² -> Ω¹`
- Lie derivative:
  - `Ω⁰ -> Ω⁰`,
  - `Ω¹ -> Ω¹`,
  - `Ω² -> Ω²`

### Curve support

- wedge: `0∧0`, `0∧1`, `1∧0`
- contraction/Lie:
  - `i_X : Ω¹ -> Ω⁰` with `X` as tangent-speed field,
  - `L_X` on `Ω⁰` and `Ω¹` via Cartan formula in 1D.

## Orientation convention

- Wedge products are orientation-dependent through face normals and edge orientation.
- `wedge11(α,β)` is antisymmetric (`α∧β = -β∧α`) under this orientation convention.

## Storage convention

- Forms are stored as vectors in vertex/edge/face ordering.
- Vector field `X` for contraction/Lie derivative is currently represented as per-face tangent vectors (`representation=:face_vector`).
- On closed curves (`nV == nE`), pass `degree=0` or `degree=1` for
  `interior_product` / `lie_derivative` to disambiguate cochain degree.

## API

```julia
wedge
wedge0k
wedge11
interior_product
lie_derivative
cartan_lie_derivative
```

## Minimal example

```julia
using FrontIntrinsicOps
using LinearAlgebra

mesh = generate_icosphere(1.0, 1)
geom = compute_geometry(mesh)
dec = build_dec(mesh, geom)

u = [p[1] for p in mesh.points]
v = [p[2] for p in mesh.points]
α = dec.d0 * u
β = dec.d0 * v

w = wedge11(α, β, mesh, geom, dec)
X = [0.2 .* geom.face_normals[fi] for fi in 1:length(mesh.faces)]
Lα = lie_derivative(X, α, mesh, geom, dec)

(antisymmetry=norm(w + wedge11(β, α, mesh, geom, dec)), lie_norm=norm(Lα))
```

## Limitations and non-goals

- This is intentionally low-order and practical; no symbolic/all-degree abstraction layer.
- Interior/Lie currently use practical fixed representations:
  - surface: `representation=:face_vector`,
  - curve: `representation=:tangent_speed`.
- The implementation is designed for PDE operators and diagnostics, not for exact algebraic identities on arbitrary high-order elements.
