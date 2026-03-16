# API: Diagnostics

## Mesh checks

### `check_mesh`

```julia
check_mesh(mesh::SurfaceMesh) → NamedTuple
```

Returns a named tuple with:

| Field | Type | Description |
|-------|------|-------------|
| `n_vertices` | `Int` | Number of vertices |
| `n_edges` | `Int` | Number of (canonical) edges |
| `n_faces` | `Int` | Number of triangles |
| `closed` | `Bool` | All edges have 2 adjacent faces |
| `manifold` | `Bool` | All edges have ≤ 2 adjacent faces |
| `consistent_orientation` | `Bool` | Interior edges traversed oppositely |
| `euler_characteristic` | `Int` | $\chi = V - E + F$ |
| `warnings` | `Vector{String}` | List of detected problems |

### `euler_characteristic`

```julia
euler_characteristic(mesh) → Int    # V - E + F
```

### `is_closed` / `is_manifold` / `has_consistent_orientation`

```julia
is_closed(mesh) → Bool
is_manifold(mesh) → Bool
has_consistent_orientation(mesh) → Bool
```

---

## DEC checks

### `check_dec`

```julia
check_dec(mesh, geom, dec; tol=1e-10) → NamedTuple
```

Returns:

| Field | Description |
|-------|-------------|
| `d1_d0_zero` | `Bool`: $\|d_1 d_0\|_\infty < \text{tol}$ |
| `d1_d0_max_residual` | `T`: $\max|[d_1 d_0]_{ij}|$ |
| `lap_constant_nullspace` | `Bool`: $\|L \mathbf{1}\|_\infty < \text{tol}$ |
| `star0_positive` | `Bool`: all dual areas $> 0$ |
| `star1_positive` | `Bool`: all cotan weights $> 0$ (may be false for obtuse meshes) |
| `warnings` | `Vector{String}` |

---

## Gauss–Bonnet diagnostic

### `gauss_bonnet_residual`

```julia
gauss_bonnet_residual(mesh, geom) → T
```

Returns $\left|\int_\Gamma K \, dA - 2\pi\chi\right|$.  Should be near
machine precision for any mesh produced by the built-in generators.

### `integrated_gaussian_curvature`

```julia
integrated_gaussian_curvature(mesh, geom) → T    # Σ K_i A_i*
```

---

## Hodge star quality

### `star1_sign_report`

```julia
star1_sign_report(dec) → NamedTuple
```

| Field | Description |
|-------|-------------|
| `n_entries` | Total number of ⋆₁ diagonal entries |
| `n_nonpositive` | Count of entries $\leq 0$ |
| `frac_nonpositive` | Fraction of non-positive entries |
| `min_entry` | Minimum value (negative if obtuse triangles present) |
| `all_positive` | `Bool` |

---

## Laplacian comparison

### `compare_laplace_methods`

```julia
compare_laplace_methods(mesh, geom) → NamedTuple
```

Assembles both $L_{\text{dec}}$ and $L_{\text{cotan}}$ and returns:

| Field | Description |
|-------|-------------|
| `norm_inf` | $\|L_{\text{dec}} - L_{\text{cotan}}\|_\infty$ |
| `norm_frob` | $\|L_{\text{dec}} - L_{\text{cotan}}\|_F$ |
| `dec_nullspace` | $\|L_{\text{dec}} \mathbf{1}\|_\infty$ |
| `cotan_nullspace` | $\|L_{\text{cotan}} \mathbf{1}\|_\infty$ |
| `max_dec_res` | Maximum residual of DEC eigenvalue test |
| `max_cotan_res` | Maximum residual of cotan eigenvalue test |

On well-shaped meshes `norm_inf < 1e-12`.

---

## IO

### Loading meshes

```julia
load_surface_stl(path::String) → SurfaceMesh{Float32}
# Returns Float32 (native STL precision)

load_curve_csv(path::String) → CurveMesh{Float64}
# CSV format: one "x,y" or "x,y,z" coordinate per line

load_curve_points(pts; closed=true) → CurveMesh{T}
# pts: Vector{SVector{2,T}} or Vector{Tuple{T,T}}
```

---

## See also

- [Topology (math)](../math/02_topology.md)
- [Geometry (math)](../math/03_geometry.md)
- [DEC (math)](../math/04_dec.md)
- [Curvature (math)](../math/06_curvature.md)
