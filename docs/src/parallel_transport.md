# Parallel Transport

This page documents practical triangle-frame transport and discrete connection tools.

## Mathematical object

Transport is built from local orthonormal tangent frames:

- per-face frame `(t1, t2)`,
- across-face transport as a 2×2 rotation aligning shared-edge tangent direction,
- along-path transport by matrix composition.

Connection angle and loop holonomy are derived from those rotations.

## Cochain space / degree

- Tangent vectors are represented in local 2D frame coordinates (`SVector{2}`), or ambient tangential vectors (`SVector{3}`) for vertex transport interfaces.
- Connection/holonomy are scalar angles (radians).

## Orientation convention

- Face orientation follows mesh orientation from `SurfaceMesh.faces`.
- Shared-edge direction uses canonical primal edge orientation `(i<j)` for reproducibility.
- Holonomy sign depends on loop traversal order.

## Storage convention

- Frames are stored as `SMatrix{3,2}` with columns `(t1, t2)`.
- `transport_matrix_across_edge` maps coordinates from `faceL` frame to `faceR` frame.

## API

```julia
face_tangent_frames
vertex_tangent_frames
connection_angle_across_edge
transport_matrix_across_edge
parallel_transport_face_vector
parallel_transport_along_face_path
parallel_transport_vertex_vector
transport_edge_1form
rotate_in_tangent_frame
holonomy_along_cycle
```

## Minimal example

```julia
using FrontIntrinsicOps
using LinearAlgebra

mesh = generate_icosphere(1.0, 1)
geom = compute_geometry(mesh)
topo = build_topology(mesh)

edgeid = findfirst(ef -> length(ef) == 2, topo.edge_faces)
fL, fR = topo.edge_faces[edgeid]
R = transport_matrix_across_edge(mesh, geom, fL, fR, edgeid)

v = R[:, 1]
v2 = parallel_transport_face_vector(v, mesh, geom, fL, fR, edgeid)

(orthogonality=norm(R' * R - I), norm_change=abs(norm(v2) - norm(v)))
```

## Limitations and non-goals

- Low-order frame-based transport only; no high-order connection discretization.
- Vertex-to-vertex transport currently uses a simple dual-face path selector (graph/geodesic keyword maps to deterministic face-path traversal).
- The current layer targets intrinsic research operators, not exact smooth Levi-Civita reconstruction.
