# types.jl – Core mesh containers, geometry containers, and DEC containers.

# ─────────────────────────────────────────────────────────────────────────────
# Mesh types
# ─────────────────────────────────────────────────────────────────────────────

"""
    CurveMesh{T}

A closed or open polygonal curve embedded in ℝ².

Fields
------
- `points :: Vector{SVector{2,T}}` – ordered vertex positions.
- `edges  :: Vector{SVector{2,Int}}` – each edge as `[i, j]` (1-based indices
  into `points`, with the convention that the curve traversal goes from `i` to
  `j`).

Indexing convention
-------------------
Vertices are 1-indexed.  For a closed curve with *N* vertices the standard
construction is `edges[k] = [k, mod1(k+1, N)]` for `k = 1, …, N`.
"""
struct CurveMesh{T<:AbstractFloat}
    points :: Vector{SVector{2,T}}
    edges  :: Vector{SVector{2,Int}}
end

"""
    SurfaceMesh{T}

A triangulated surface embedded in ℝ³.

Fields
------
- `points :: Vector{SVector{3,T}}` – vertex positions.
- `faces  :: Vector{SVector{3,Int}}` – each triangle as `[i, j, k]` (1-based
  indices into `points`), oriented counter-clockwise when viewed from outside.

Indexing convention
-------------------
Vertices are 1-indexed.  The outward normal of face `f` is
`(p[j]-p[i]) × (p[k]-p[i])` (un-normalised) following the right-hand rule.
"""
struct SurfaceMesh{T<:AbstractFloat}
    points :: Vector{SVector{3,T}}
    faces  :: Vector{SVector{3,Int}}
end

# ─────────────────────────────────────────────────────────────────────────────
# Geometry containers
# ─────────────────────────────────────────────────────────────────────────────

"""
    CurveGeometry{T}

Geometric quantities derived from a `CurveMesh{T}`.

Fields
------
- `edge_lengths    :: Vector{T}` – primal 1-simplex (edge) lengths.
- `edge_tangents   :: Vector{SVector{2,T}}` – unit tangent vectors per edge,
  pointing in the direction of the edge orientation.
- `vertex_dual_lengths :: Vector{T}` – dual 0-cell measure at each vertex
  (average of the two adjacent half-edge lengths, or half of the summed
  adjacent edge lengths for the closed-curve case).
- `vertex_normals  :: Vector{SVector{2,T}}` – unit inward-pointing normals at
  each vertex (90° rotation of the averaged tangents).
- `signed_curvature :: Vector{T}` – discrete signed curvature κᵥ at each
  vertex = turning angle / dual arc-length.
"""
struct CurveGeometry{T<:AbstractFloat}
    edge_lengths        :: Vector{T}
    edge_tangents       :: Vector{SVector{2,T}}
    vertex_dual_lengths :: Vector{T}
    vertex_normals      :: Vector{SVector{2,T}}
    signed_curvature    :: Vector{T}
end

"""
    SurfaceGeometry{T}

Geometric quantities derived from a `SurfaceMesh{T}`.

Fields
------
- `face_normals   :: Vector{SVector{3,T}}` – outward unit normals per face.
- `face_areas     :: Vector{T}` – area of each triangle.
- `edge_lengths   :: Vector{T}` – length of each unique edge (ordered as in
  the edge list produced by `topology`).
- `vertex_dual_areas :: Vector{T}` – barycentric dual area at each vertex
  (one-third of the sum of areas of adjacent faces).
- `vertex_normals :: Vector{SVector{3,T}}` – area-weighted vertex normals
  (unit).
- `mean_curvature_normal :: Vector{SVector{3,T}}` – discrete mean-curvature
  normal vector `Hₙ` at each vertex (populated after calling `curvature`).
- `mean_curvature :: Vector{T}` – scalar mean curvature `H` at each vertex.
- `gaussian_curvature :: Vector{T}` – angle-defect Gaussian curvature at each
  vertex (optional; empty vector if not computed).
"""
struct SurfaceGeometry{T<:AbstractFloat}
    face_normals          :: Vector{SVector{3,T}}
    face_areas            :: Vector{T}
    edge_lengths          :: Vector{T}
    vertex_dual_areas     :: Vector{T}
    vertex_normals        :: Vector{SVector{3,T}}
    mean_curvature_normal :: Vector{SVector{3,T}}
    mean_curvature        :: Vector{T}
    gaussian_curvature    :: Vector{T}
end

# ─────────────────────────────────────────────────────────────────────────────
# DEC (Discrete Exterior Calculus) containers
# ─────────────────────────────────────────────────────────────────────────────

"""
    CurveDEC{T}

Assembled sparse DEC operators for a `CurveMesh{T}`.

Convention
----------
All operators follow the *primal* cochain convention:
- `d0 :: SparseMatrixCSC{T,Int}` – boundary/coboundary operator mapping
  0-cochains (vertex fields) to 1-cochains (edge fields):
  `(d0 u)[e] = u[j] - u[i]` for edge `e = (i→j)`.
- `star0 :: SparseMatrixCSC{T,Int}` – Hodge star ⋆₀: diagonal matrix of
  vertex dual lengths.
- `star1 :: SparseMatrixCSC{T,Int}` – Hodge star ⋆₁: diagonal matrix of
  dual edge lengths (≈ primal edge lengths for a uniform 1-manifold).
- `lap0  :: SparseMatrixCSC{T,Int}` – scalar Laplace–Beltrami on vertices:
  `L = ⋆₀⁻¹ d₀ᵀ ⋆₁ d₀`.

Notes
-----
`lap0` as assembled here is a positive-semi-definite operator (positive
eigenvalues for non-constant modes) following the sign convention
`Δf = −div grad f` with the sign chosen so that the cotan operator matches
the usual convention in the DDG literature.
"""
struct CurveDEC{T<:AbstractFloat}
    d0    :: SparseMatrixCSC{T,Int}
    star0 :: SparseMatrixCSC{T,Int}
    star1 :: SparseMatrixCSC{T,Int}
    lap0  :: SparseMatrixCSC{T,Int}
end

"""
    SurfaceDEC{T}

Assembled sparse DEC operators for a `SurfaceMesh{T}`.

Convention
----------
Primal/dual pairing follows the standard DEC convention on oriented
2-manifolds:

- `d0 :: SparseMatrixCSC{T,Int}` – coboundary map from 0-cochains to
  1-cochains: `(d0 u)[e] = u[j] - u[i]` for edge `e = (i→j)`.
- `d1 :: SparseMatrixCSC{T,Int}` – coboundary map from 1-cochains to
  2-cochains: `(d1 α)[f] = ±α[e₀] ± α[e₁] ± α[e₂]` with signs from the
  face-edge orientation.
- `star0 :: SparseMatrixCSC{T,Int}` – Hodge ⋆₀: diagonal of vertex dual
  areas.
- `star1 :: SparseMatrixCSC{T,Int}` – Hodge ⋆₁: diagonal edge-based metric
  compatible with scalar Laplace–Beltrami (cotan weights).
- `star2 :: SparseMatrixCSC{T,Int}` – Hodge ⋆₂: diagonal of face areas.
- `lap0  :: SparseMatrixCSC{T,Int}` – scalar Laplace–Beltrami on vertices:
  `L = ⋆₀⁻¹ d₀ᵀ ⋆₁ d₀`.

Identity
--------
For a closed oriented triangulated surface, `d1 * d0 == 0` holds to machine
precision (this is the discrete analogue of `d² = 0`).
"""
struct SurfaceDEC{T<:AbstractFloat}
    d0    :: SparseMatrixCSC{T,Int}
    d1    :: SparseMatrixCSC{T,Int}
    star0 :: SparseMatrixCSC{T,Int}
    star1 :: SparseMatrixCSC{T,Int}
    star2 :: SparseMatrixCSC{T,Int}
    lap0  :: SparseMatrixCSC{T,Int}
end
