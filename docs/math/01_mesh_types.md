# Mesh Types and Data Structures

## Overview

FrontIntrinsicOps.jl represents meshes as **plain Julia structs** containing
arrays of vertices, edges, and faces.  The package is parameterised on the
floating-point precision `T` (default `Float64`).

---

## `CurveMesh{T}`

A closed or open 2-D polygonal curve in $\mathbb{R}^2$ (or optionally $\mathbb{R}^3$).

```
CurveMesh{T}
  points  :: Vector{SVector{2,T}}   # vertex coordinates
  edges   :: Vector{Tuple{Int,Int}} # ordered edge list (i → j)
  closed  :: Bool                   # true if last edge reconnects to vertex 1
```

### Topology

An $N$-vertex closed curve has exactly $N$ edges.  Edge $k$ connects vertex $k$
to vertex $k+1$ (mod $N$).  An open curve with $N$ vertices has $N-1$ edges.

### Orientation convention

Edges are stored in traversal order.  The tangent field points in the positive
traversal direction.  Vertex normals point 90° left of the tangent (i.e. inward
for a counterclockwise curve).

---

## `SurfaceMesh{T}`

A triangulated closed or open surface in $\mathbb{R}^3$.

```
SurfaceMesh{T}
  points :: Vector{SVector{3,T}}       # N_V vertex positions
  faces  :: Vector{Tuple{Int,Int,Int}} # N_F triangles  (a, b, c) 1-based
```

Edges are **not** stored explicitly; they are derived on demand by
`build_topology` and cached in `MeshTopology`.

### Orientation convention

Every face triple $(a, b, c)$ is assumed to be consistently oriented so that
the outward normal is

$$\hat{n}_f = \frac{(p_b - p_a) \times (p_c - p_a)}{\|(p_b - p_a) \times (p_c - p_a)\|}$$

On a **closed, orientable** manifold two adjacent faces must traverse their
shared edge in opposite directions.  The function `has_consistent_orientation`
verifies this property.

---

## `CurveGeometry{T}` and `SurfaceGeometry{T}`

These structs hold pre-computed **intrinsic geometric quantities**.  They are
populated by `compute_geometry(mesh)` and should be treated as read-only after
construction.

### `CurveGeometry{T}`

| Field | Type | Description |
|-------|------|-------------|
| `edge_lengths` | `Vector{T}` | $\ell_e = \|p_j - p_i\|$ |
| `edge_tangents` | `Vector{SVector{2,T}}` | Unit tangent $(p_j - p_i)/\ell_e$ |
| `vertex_dual_lengths` | `Vector{T}` | $\ell_i^* = (\ell_{e_{i-1}} + \ell_{e_i})/2$ |
| `vertex_normals` | `Vector{SVector{2,T}}` | Left-normal of averaged tangent |
| `signed_curvature` | `Vector{T}` | $\kappa_i$ (see [Curvature](06_curvature.md)) |

### `SurfaceGeometry{T}`

| Field | Type | Description |
|-------|------|-------------|
| `face_normals` | `Vector{SVector{3,T}}` | Outward unit normals $\hat{n}_f$ |
| `face_areas` | `Vector{T}` | $A_f = \tfrac12 \|(p_b-p_a)\times(p_c-p_a)\|$ |
| `edge_lengths` | `Vector{T}` | Primal edge lengths $\ell_e$ |
| `vertex_dual_areas` | `Vector{T}` | $A_i^*$ (barycentric or mixed) |
| `vertex_normals` | `Vector{SVector{3,T}}` | Area-weighted vertex normals |
| `mean_curvature` | `Vector{T}` | $H_i$ (filled by `compute_curvature!`) |
| `gaussian_curvature` | `Vector{T}` | $K_i$ (filled by `compute_curvature!`) |
| `dual_area_method` | `Symbol` | `:barycentric` or `:mixed` |

---

## `CurveDEC{T}` and `SurfaceDEC{T}`

These structs bundle the **discrete exterior calculus operators** assembled from
the mesh and its geometry.  They are populated by `build_dec(mesh, geom)`.

### `SurfaceDEC{T}`

| Field | Type | Mathematical object |
|-------|------|---------------------|
| `d0` | `SparseMatrixCSC` | $d_0 : \Omega^0 \to \Omega^1$ (vertex-to-edge coboundary) |
| `d1` | `SparseMatrixCSC` | $d_1 : \Omega^1 \to \Omega^2$ (edge-to-face coboundary) |
| `star0` | `SparseMatrixCSC` | $\star_0 : \Omega^0 \to \Omega^0$ (diagonal, dual areas) |
| `star1` | `SparseMatrixCSC` | $\star_1 : \Omega^1 \to \Omega^1$ (diagonal, cotan weights) |
| `star2` | `SparseMatrixCSC` | $\star_2 : \Omega^2 \to \Omega^2$ (diagonal, face areas) |
| `laplacian` | `SparseMatrixCSC` | $L = \star_0^{-1} d_0^\top \star_1 d_0$ |

See [Discrete Exterior Calculus](04_dec.md) and
[Laplace–Beltrami](05_laplace_beltrami.md) for the mathematics.

---

## Primal–dual picture

The DEC framework rests on a **primal–dual mesh pair**:

```
Primal mesh                Dual mesh
──────────────────────     ─────────────────────────────
0-cells : vertices         2-cells : dual faces (Voronoi)
1-cells : edges            1-cells : dual edges
2-cells : faces            0-cells : circumcentres
```

In practice FrontIntrinsicOps.jl uses a **lumped / diagonal** approximation of
the dual:

- Dual areas $A_i^*$ approximate the Voronoi cell area at vertex $i$.
- The off-diagonal geometry of the dual is absorbed into the cotan weights
  $w_e = \tfrac12(\cot\alpha_e + \cot\beta_e)$ of the primal edges.

This choice keeps all global operators **sparse and diagonal-dominant**, which
simplifies both assembly and linear solves.

---

## Memory layout and indexing

- Vertices are indexed $1, \ldots, N_V$ (Julia 1-based).
- Edges are canonically stored with the smaller index first: $(i, j)$, $i < j$.
  This guarantees uniqueness and makes edge lookup via a `Dict` or sorted list
  straightforward.
- Faces store the original orientation given at construction.

---

## See also

- [Topology and incidence matrices](02_topology.md)
- [Discrete geometry and dual areas](03_geometry.md)
- [Discrete exterior calculus](04_dec.md)
