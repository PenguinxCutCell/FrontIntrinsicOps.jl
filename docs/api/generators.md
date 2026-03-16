# API: Mesh Generators

All generators return a `SurfaceMesh{Float64}` (or `CurveMesh{Float64}`)
with outward-oriented faces.

---

## Curves

### `sample_circle`

```julia
sample_circle(R::T, N::Int) → CurveMesh{T}
```

Generate a regular $N$-gon inscribed in a circle of radius $R$ centered at the
origin in the $xy$-plane.

| Argument | Description |
|----------|-------------|
| `R` | Circle radius |
| `N` | Number of vertices (= number of edges for a closed curve) |

---

## Surfaces — sphere variants

### `generate_uvsphere`

```julia
generate_uvsphere(R, nphi, ntheta) → SurfaceMesh{T}
```

UV (latitude-longitude) sphere of radius `R`.
`nphi` meridional slices, `ntheta` latitudinal bands.
Poor triangle quality at poles; prefer `generate_icosphere` for PDE studies.

### `generate_icosphere`

```julia
generate_icosphere(R, level::Int) → SurfaceMesh{T}
```

Icosahedron recursively subdivided `level` times, then projected to radius `R`.
Uniform triangle quality.  Recommended for all convergence studies.

| Level | $N_V$ | $N_F$ | Mesh size $h$ (approx.) |
|-------|-------|-------|------------------------|
| 2 | 162 | 320 | 0.27 |
| 3 | 642 | 1280 | 0.14 |
| 4 | 2562 | 5120 | 0.07 |
| 5 | 10242 | 20480 | 0.035 |

---

## Surfaces — torus

### `generate_torus`

```julia
generate_torus(R, r, ntheta::Int, nphi::Int) → SurfaceMesh{T}
```

Standard torus with major radius `R` and minor radius `r`.
`ntheta` toroidal samples, `nphi` poloidal samples.
Topology: genus 1, $\chi = 0$.

---

## Surfaces — deformed geometries (v0.4)

### `generate_ellipsoid`

```julia
generate_ellipsoid(a, b, c, nphi::Int, ntheta::Int) → SurfaceMesh{T}
```

Axis-aligned ellipsoid with semi-axes $a$ ($x$), $b$ ($y$), $c$ ($z$).
UV parameterization with `nphi` × `ntheta` resolution.
When `a == b == c` reduces to a UV sphere.

### `generate_perturbed_sphere`

```julia
generate_perturbed_sphere(R, ε, k, nphi::Int, ntheta::Int) → SurfaceMesh{T}
```

"Bumpy sphere" with radial perturbation $r(\phi,\theta) = R(1 + \varepsilon \cos(k\phi)\cos(k\theta))$.

| Argument | Description |
|----------|-------------|
| `R` | Base radius |
| `ε` | Perturbation amplitude (0.05–0.3 recommended) |
| `k` | Perturbation mode (2–4 recommended) |
| `nphi` | Number of azimuthal samples |
| `ntheta` | Number of polar samples |

---

## See also

- [Mesh generators (math)](../math/16_generators.md)
- [Types](types.md)
