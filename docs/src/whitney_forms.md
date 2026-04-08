# Whitney forms

This page documents the lowest-order Whitney basis/reconstruction layer.

## Mathematical objects

- Whitney 0-forms: piecewise-linear scalar basis on vertices.
- Whitney 1-forms: edge-based basis on triangles.
- Whitney 2-forms: face-based constant area-density basis.

## Degrees and cochain spaces

- degree-0: `c0::Vector` on vertices,
- degree-1: `c1::Vector` on oriented edges,
- degree-2: `c2::Vector` on oriented faces.

## Orientation convention

- Global edge orientation follows canonical edge ordering (`i < j`) from `build_topology`.
- Face-local Whitney-1 ordering is `(1→2), (2→3), (3→1)` in face orientation.
- Local/global sign transfer uses `topo.face_edge_signs`.

## Storage convention

- Reconstructed 0-form: facewise affine scalar.
- Reconstructed 1-form: tangent-vector representation (`α^#`) evaluated facewise.
- Reconstructed 2-form: facewise constant density (`c2[f] / area[f]`).

## API

- `whitney0_basis_local`, `whitney1_basis_local`, `whitney2_basis_local`
- `eval_whitney0_local`, `eval_whitney1_local`, `eval_whitney2_local`
- `reconstruct_0form_face`, `reconstruct_1form_face`, `reconstruct_2form_face`
- `reconstruct_0form`, `reconstruct_1form`, `reconstruct_2form`

## Minimal example

```julia
using FrontIntrinsicOps, StaticArrays

mesh = generate_icosphere(1.0, 1)
geom = compute_geometry(mesh)

c1 = interpolate_1form(x -> SVector{3,Float64}(-x[2], x[1], 0.0), mesh, geom)
vface = reconstruct_1form(c1, mesh, geom; representation=:facewise_tangent)
println(norm(vface[1]))
```

## Limitations and non-goals

- Lowest-order only.
- Facewise reconstruction API (not full global point-location/evaluation).
- No higher-order Whitney-like families in this round.
