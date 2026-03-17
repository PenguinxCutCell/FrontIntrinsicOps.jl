# API: Types

## `CurveMesh{T}`

Closed or open 2-D polygonal curve.

```julia
CurveMesh{T <: AbstractFloat}
  points :: Vector{SVector{2,T}}    # vertex coordinates (2-D)
  edges  :: Vector{Tuple{Int,Int}}  # ordered edge list
  closed :: Bool                    # whether the last edge closes to vertex 1
```

**Constructor:** Load from file or build analytically via `sample_circle`.

---

## `SurfaceMesh{T}`

Triangulated surface in $\mathbb{R}^3$.

```julia
SurfaceMesh{T <: AbstractFloat}
  points :: Vector{SVector{3,T}}          # vertex positions
  faces  :: Vector{Tuple{Int,Int,Int}}    # triangle indices (1-based)
```

---

## `CurveGeometry{T}`

Pre-computed intrinsic geometry of a `CurveMesh`.

```julia
CurveGeometry{T}
  edge_lengths         :: Vector{T}
  edge_tangents        :: Vector{SVector{2,T}}
  vertex_dual_lengths  :: Vector{T}
  vertex_normals       :: Vector{SVector{2,T}}
  signed_curvature     :: Vector{T}
```

Build with `compute_geometry(mesh::CurveMesh)`.

---

## `SurfaceGeometry{T}`

Pre-computed intrinsic geometry of a `SurfaceMesh`.

```julia
SurfaceGeometry{T}
  face_normals          :: Vector{SVector{3,T}}
  face_areas            :: Vector{T}
  edge_lengths          :: Vector{T}
  vertex_dual_areas     :: Vector{T}
  vertex_normals        :: Vector{SVector{3,T}}
  mean_curvature        :: Vector{T}      # filled by compute_curvature!
  gaussian_curvature    :: Vector{T}      # filled by compute_curvature!
  dual_area_method      :: Symbol         # :barycentric or :mixed
```

Build with `compute_geometry(mesh::SurfaceMesh; dual_area=:barycentric)`.

---

## `CurveDEC{T}`

Assembled DEC operators for a `CurveMesh`.

```julia
CurveDEC{T}
  d0         :: SparseMatrixCSC   # N_E × N_V  coboundary
  star0      :: SparseMatrixCSC   # N_V × N_V  diagonal (dual lengths)
  star1      :: SparseMatrixCSC   # N_E × N_E  diagonal (edge/dual-length ratio)
  laplacian  :: SparseMatrixCSC   # N_V × N_V  Laplace–Beltrami (= -Δ)
```

Build with `build_dec(mesh::CurveMesh, geom::CurveGeometry)`.

---

## `SurfaceDEC{T}`

Assembled DEC operators for a `SurfaceMesh`.

```julia
SurfaceDEC{T}
  d0         :: SparseMatrixCSC   # N_E × N_V
  d1         :: SparseMatrixCSC   # N_F × N_E
  star0      :: SparseMatrixCSC   # N_V × N_V  diagonal (dual areas)
  star1      :: SparseMatrixCSC   # N_E × N_E  diagonal (cotan weights)
  star2      :: SparseMatrixCSC   # N_F × N_F  diagonal (inverse face areas)
  laplacian  :: SparseMatrixCSC   # N_V × N_V  Laplace–Beltrami
```

Build with `build_dec(mesh::SurfaceMesh, geom::SurfaceGeometry; laplace=:dec)`.

---

## `MeshTopology`

Topological adjacency lists derived from a `SurfaceMesh`.

```julia
MeshTopology
  edges            :: Vector{Tuple{Int,Int}}    # canonical (i<j) edge list
  edge_index       :: Dict{Tuple{Int,Int},Int}  # fast lookup
  face_edges       :: Vector{Vector{Int}}       # face → 3 edge indices
  face_edge_signs  :: Vector{Vector{Int}}       # ±1 orientation signs
  vertex_faces     :: Vector{Vector{Int}}       # vertex → adjacent faces
  vertex_edges     :: Vector{Vector{Int}}       # vertex → adjacent edges
  edge_faces       :: Vector{Vector{Int}}       # edge → 1 or 2 faces
```

Build with `build_topology(mesh::SurfaceMesh)`.

---

## `SurfacePDECache{T}`

Pre-assembled PDE operators and factorizations for fast repeated solves.

```julia
SurfacePDECache{T}
  dec          :: SurfaceDEC
  mass         :: SparseMatrixCSC   # M = star0
  laplacian    :: SparseMatrixCSC   # L
  system       :: SparseMatrixCSC   # (M + dt θ μ L)
  factorization                     # sparse LU or Cholesky
  μ            :: T
  dt           :: T
  θ            :: T
```

Build with `build_pde_cache(mesh, geom, dec; μ, dt, θ)`.

---

## `SurfaceDiffusionBuffers{T}` and `SurfaceRDBuffers{T}`

Pre-allocated scratch buffers for zero-allocation time steps.

```julia
SurfaceDiffusionBuffers{T}
  rhs :: Vector{T}
  tmp :: Vector{T}

SurfaceRDBuffers{T}
  rhs      :: Vector{T}
  reaction :: Vector{T}
  tmp      :: Vector{T}
```

Build with `alloc_diffusion_buffers(nv)` or `alloc_rd_buffers(nv)`.

---

## See also

- [Mesh types (math)](01_mesh_types.md)
- [Discrete exterior calculus (math)](04_dec.md)
- [Caching and performance (math)](15_caching.md)

---

## Ambient signed-distance API (v0.5)

### `build_signed_distance_cache`

```julia
build_signed_distance_cache(mesh; leafsize=8) -> SignedDistanceCache
```

Builds an exact nearest-primitive acceleration cache (AABB tree + sign data)
for `CurveMesh` and `SurfaceMesh`.

### `signed_distance` (batched)

```julia
signed_distance(points, mesh_or_cache;
                sign_mode=:auto,
                lower_bound=0.0,
                upper_bound=Inf,
                return_normals=true)
```

Accepted `points` formats:
- `Vector{SVector{N,T}}`
- `Matrix{T}` with shape `(N,np)` or `(np,N)`

Returns `(S, I, C, N)`:
- `S`: signed distances,
- `I`: closest primitive ids,
- `C`: closest points,
- `N`: signing normals (pseudonormal mode).

### `signed_distance` (scalar)

```julia
signed_distance(point::SVector{N,T}, mesh_or_cache; sign_mode=:auto)
```

Returns named tuple `(distance, primitive, closest, normal)`.

### `unsigned_distance`

```julia
unsigned_distance(points, mesh_or_cache)
```

Convenience wrapper for `signed_distance(...; sign_mode=:unsigned)`.

### `winding_number`

```julia
winding_number(point, mesh_or_cache)
```

- Closed 2D curve: integer winding number.
- Closed 3D oriented surface: normalized solid-angle winding number.

### `is_closed_curve` and `is_closed_surface`

```julia
is_closed_curve(mesh::CurveMesh) -> Bool
is_closed_surface(mesh::SurfaceMesh) -> Bool
```

Closure predicates used by `sign_mode=:auto`.
