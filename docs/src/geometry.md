# API: Geometry and DEC

## Geometry

### `compute_geometry`

```julia
compute_geometry(mesh::CurveMesh) → CurveGeometry{T}
compute_geometry(mesh::SurfaceMesh; dual_area=:barycentric) → SurfaceGeometry{T}
```

Compute all intrinsic geometric quantities.

| Keyword | Values | Description |
|---------|--------|-------------|
| `dual_area` | `:barycentric` (default) | One-third of adjacent face areas |
| | `:mixed` or `:voronoi` | Meyer et al. 2003 mixed Voronoi areas |

---

## DEC assembly

### `build_dec`

```julia
build_dec(mesh::CurveMesh, geom::CurveGeometry) → CurveDEC{T}
build_dec(mesh::SurfaceMesh, geom::SurfaceGeometry; laplace=:dec) → SurfaceDEC{T}
```

Assemble the full discrete exterior calculus structure.

| Keyword | Values | Description |
|---------|--------|-------------|
| `laplace` | `:dec` (default) | $L = \star_0^{-1} d_0^\top \star_1 d_0$ |
| | `:cotan` | Direct cotan assembly (numerically equivalent) |

### `build_laplace_beltrami`

```julia
build_laplace_beltrami(mesh, geom; method=:dec) → SparseMatrixCSC
```

Return only the Laplacian matrix (no other DEC operators).

### `laplace_beltrami`

```julia
laplace_beltrami(mesh, geom, dec, u::Vector) → Vector
```

Apply the Laplacian to a scalar field: returns $L u$.

---

## Hodge stars (individual access)

```julia
hodge_star_0(mesh, geom) → SparseMatrixCSC   # diagonal: dual areas / lengths
hodge_star_1(mesh, geom) → SparseMatrixCSC   # diagonal: cotan weights (surface)
hodge_star_2(mesh, geom) → SparseMatrixCSC   # diagonal: 1/face_area
```

---

## Exterior derivatives (individual access)

```julia
incidence_0(mesh) → SparseMatrixCSC   # d0: N_E × N_V
incidence_1(mesh) → SparseMatrixCSC   # d1: N_F × N_E
```

---

## Codifferentials

```julia
codifferential_1(mesh, geom, dec) → SparseMatrixCSC   # δ₁ = ⋆₀⁻¹ d₀ᵀ ⋆₁
codifferential_2(mesh, geom, dec) → SparseMatrixCSC   # δ₂ = ⋆₁⁻¹ d₁ᵀ ⋆₂
```

---

## DEC gradient and divergence

```julia
gradient_0_to_1(mesh, dec, u)        → Vector  # d0 * u
divergence_1_to_0(mesh, geom, dec, α) → Vector  # δ₁ α
curl_like_1_to_2(mesh, dec, α)       → Vector  # d1 * α
```

---

## Topology

```julia
build_topology(mesh::SurfaceMesh) → MeshTopology

is_closed(mesh)                 → Bool
is_manifold(mesh)               → Bool
has_consistent_orientation(mesh) → Bool
euler_characteristic(mesh)      → Int    # V - E + F
```

---

## Curvature

```julia
# Curves
curvature(mesh::CurveMesh, geom) → Vector{T}         # signed κ

# Surfaces
mean_curvature(mesh, geom, dec) → Vector{T}           # scalar H
mean_curvature_normal(mesh, geom, dec) → Vector{SVector{3,T}}
gaussian_curvature(mesh, geom) → Vector{T}            # angle-defect K
compute_curvature(mesh, geom, dec) → SurfaceGeometry  # updates curvature fields
```

---

## Integrals

```julia
measure(mesh, geom) → T                          # total arc length or area
enclosed_measure(mesh) → T                       # enclosed area or volume
integrate_vertex_field(mesh, geom, u) → T        # Σ u_i A_i*
integrate_face_field(mesh, geom, u) → T          # Σ u_f A_f
integrated_gaussian_curvature(mesh, geom) → T    # Σ K_i A_i*
```

---

## Diagnostics

```julia
check_mesh(mesh) → NamedTuple
# Fields: n_vertices, n_edges, [n_faces], closed, manifold,
#         consistent_orientation, euler_characteristic, warnings

check_dec(mesh, geom, dec; tol=1e-10) → NamedTuple
# Fields: d1_d0_zero, d1_d0_max_residual, lap_constant_nullspace,
#         star0_positive, star1_positive, warnings

gauss_bonnet_residual(mesh, geom) → T        # |∫K dA - 2πχ|
star1_sign_report(dec) → NamedTuple          # n_nonpositive, frac, min_entry
compare_laplace_methods(mesh, geom) → NamedTuple  # ‖L_dec − L_cotan‖
```

---

## See also

- [Geometry (math)](03_geometry.md)
- [DEC (math)](04_dec.md)
- [Laplace–Beltrami (math)](05_laplace_beltrami.md)
- [Curvature (math)](06_curvature.md)
