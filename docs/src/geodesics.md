# Geodesics

This page documents intrinsic geodesic-distance and path tools on triangulated surfaces.

## Mathematical object

Distance is computed with a heat-method pipeline on vertex 0-forms:

1. Short-time heat solve for `u`,
2. normalized face vector field `X = -∇u / |∇u|`,
3. Poisson solve for distance potential `φ`,
4. constant shift to pin source value.

## Cochain space / degree

- Distance field: vertex 0-form (`length = nV`).
- Geodesic gradient: per-face tangent vectors (`length = nF`).
- Shortest path output: vertex index sequence.

## Orientation convention

- Distances are nonnegative scalars and orientation-independent.
- Path extraction uses vertex graph adjacency from oriented mesh topology, but returned path is geometric (ordered vertices from `src` to `dst`).

## Storage convention

- Distances are vectors indexed by mesh vertex order.
- Multi-source distances use pointwise minimum over sourcewise distance vectors.

## API

```julia
geodesic_distance
geodesic_distance_to_vertex
geodesic_distance_to_vertices
geodesic_gradient
shortest_path_vertices
shortest_path_points
intrinsic_ball
farthest_point_sampling_geodesic
```

## Minimal example

```julia
using FrontIntrinsicOps
using LinearAlgebra

mesh = generate_icosphere(1.0, 1)
geom = compute_geometry(mesh)
dec = build_dec(mesh, geom)

src = argmax([p[3] for p in mesh.points])
d = geodesic_distance_to_vertex(mesh, geom, dec, src)
path = shortest_path_vertices(mesh, geom, dec, src, argmin([p[3] for p in mesh.points]); distance=d)

(mind=minimum(d), maxd=maximum(d), pathlen=length(path))
```

## Limitations and non-goals

- Surface-only geodesic tools in this page (curve geodesic distance is not included here).
- Shortest paths are approximate vertex paths (steepest-descent + fallback), not exact polyline geodesics.
- Connected-surface assumption is enforced with explicit error handling.
