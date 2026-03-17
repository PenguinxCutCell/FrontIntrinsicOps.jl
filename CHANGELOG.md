# Changelog

## v0.5.0 — Ambient signed distance

### Added
- Exact ambient point-to-front distance queries for 2D `CurveMesh` and 3D `SurfaceMesh`.
- Public API: `build_signed_distance_cache`, `signed_distance`, `unsigned_distance`, `winding_number`, `is_closed_curve`, `is_closed_surface`.
- Sign backends: `:unsigned`, `:pseudonormal`, `:winding`, and `:auto` dispatch.
- AABB-accelerated nearest-primitive queries with exact primitive kernels (segment / triangle).

### Semantics
- Closed meshes + `:winding`: inside/outside signed distance.
- Open meshes + `:pseudonormal`: oriented side-of-curve / side-of-sheet signed distance.
- Open meshes do not define a global inside/outside region.

### Notes
- This release does not add fast winding, narrow-band methods, reinitialization, or bulk-grid coupling.
