# Ambient Signed Distance

This page documents the ambient signed-distance layer in `FrontIntrinsicOps.jl`.

## Minimal 1-D point fronts

In addition to ambient curve/surface queries, the package provides a minimal
1-D front primitive `PointFront1D` with exactly two supported configurations:

- one marker `xΓ`,
- two markers `(xL, xR)` with `xL < xR`.

Sign convention matches the ambient API:
- inside: negative distance,
- outside: positive distance,
- interface marker(s): zero.

The helper API is:

```julia
signed_distance(front::PointFront1D, x::Real)
signed_distance(front::PointFront1D, xs::AbstractVector)
rebuild_signed_distance(front::PointFront1D, xnodes::AbstractVector)
interface_normals(front::PointFront1D)
```

`interface_normals` returns outward normal orientation(s) of the inside region:
- one marker: `[+1]` (inside right) or `[-1]` (inside left),
- two markers: `[-1, +1]` (interval inside) or `[+1, -1]` (interval outside).

This 1-D support is intentionally lightweight and does not include 1-D DEC,
1-D PDE operators, or general 1-D graph topology machinery.

## Problem definition

Given a point $q \in \mathbb{R}^N$ and a piecewise-linear front mesh $\Gamma_h$:

- 2D: polygonal curve (`CurveMesh`) made of oriented segments,
- 3D: triangulated surface (`SurfaceMesh`) made of oriented triangles,

the distance magnitude is

$$
d(q,\Gamma_h) = \min_{x\in\Gamma_h} \lVert q-x\rVert_2.
$$

Implementation uses exact primitive kernels (segment/triangle closest-point)
and an AABB tree for acceleration, but the returned nearest primitive and
distance match brute force up to floating-point roundoff.

## Sign modes

`signed_distance(...; sign_mode=...)` supports:

- `:unsigned`: always returns $+d$.
- `:pseudonormal`: oriented sign from the closest feature normal.
- `:winding`: global inside/outside sign for closed meshes only.
- `:auto`: `:winding` on closed meshes, `:pseudonormal` on open meshes.

If `sign_mode=:winding` is requested on an open mesh, an `ArgumentError` is
thrown because inside/outside is not globally defined.

Support summary:

| Mesh type | `:pseudonormal` | `:winding` | `:unsigned` | `:auto` |
|-----------|------------------|------------|-------------|---------|
| Closed `CurveMesh` | ✓ | ✓ | ✓ | winding |
| Open `CurveMesh` | ✓ | error | ✓ | pseudonormal |
| Closed `SurfaceMesh` | ✓ | ✓ | ✓ | winding |
| Open `SurfaceMesh` | ✓ | error | ✓ | pseudonormal |

## Pseudonormal sign

Let $c$ be the closest point on the closest feature and $n_\ast$ the feature
normal (face/edge/vertex pseudonormal). Then

$$
\sigma = \operatorname{sign}\big((q-c)\cdot n_\ast\big),\qquad
s = \sigma\, d.
$$

This mode is valid on both open and closed meshes.

- On closed meshes, it is an oriented signed distance near the surface.
- On open meshes, it is a **side-of-sheet / side-of-curve** sign, not a global
  occupancy classification.
- Points exactly on the front return zero distance.
- Near numerically ambiguous configurations, if the signing dot-product is
  within tolerance the sign can evaluate to zero.

## Winding sign

For closed meshes only:

- 2D curves: integer winding number test.
- 3D surfaces: solid-angle winding number
  $$w(q) = \frac{1}{4\pi}\sum_{f}\Omega_f(q).$$

Sign convention:

- outside $\Rightarrow +d$,
- inside $\Rightarrow -d$.

With package orientation conventions:

- closed CCW curve in 2D: outside positive, inside negative,
- outward-oriented closed surface in 3D: outside positive, inside negative.

## Public API

```julia
cache = build_signed_distance_cache(mesh; leafsize=8)

S, I, C, N = signed_distance(points, cache; sign_mode=:auto)
r = signed_distance(point, cache; sign_mode=:auto)

U = unsigned_distance(points, cache)
w = winding_number(point, cache)
```

See `examples/sdf_curve_closed_circle.jl`,
`examples/sdf_curve_open_polyline.jl`,
`examples/sdf_surface_closed_sphere.jl`, and
`examples/sdf_surface_open_patch.jl`.
