# Mesh Generators

## Overview

FrontIntrinsicOps.jl provides analytical mesh generators for standard
geometries.  All generators produce consistently-oriented, outward-normal
meshes suitable for convergence studies.

---

## `sample_circle` — regular polygon

```julia
mesh = sample_circle(R, N)
```

Generates a regular $N$-gon inscribed in a circle of radius $R$ in the $xy$-plane.

**Vertices:**

$$p_k = \left(R \cos\!\frac{2\pi k}{N},\; R \sin\!\frac{2\pi k}{N}\right), \quad k = 0, \ldots, N-1$$

**Convergence:**
- Length $\to 2\pi R$ as $N \to \infty$ with rate $O(1/N^2)$.
- Enclosed area $\to \pi R^2$ with rate $O(1/N^2)$.
- Curvature $\to 1/R$ with rate $O(1/N^2)$.

---

## `generate_uvsphere` — UV sphere

```julia
mesh = generate_uvsphere(R, nphi, ntheta)
```

Generates a latitude-longitude sphere of radius $R$ with `nphi` meridians and
`ntheta` latitude bands.

**Parameterization:**

$$p(\phi, \theta) = \left(R\cos\phi\sin\theta,\; R\sin\phi\sin\theta,\; R\cos\theta\right)$$

- $\phi \in [0, 2\pi)$ — longitude (azimuth), `nphi` values
- $\theta \in (0, \pi)$ — colatitude (polar angle), `ntheta` values

**Notes:**
- Pole vertices are singular (high-valence, poor triangle quality).
- Not recommended for convergence studies — use `generate_icosphere` instead.
- Useful when longitude/latitude alignment is required.

---

## `generate_icosphere` — icosphere (recommended)

```julia
mesh = generate_icosphere(R, level)
```

Generates a sphere of radius $R$ by recursively subdividing a regular
icosahedron.

**Algorithm:**
1. Start with a regular icosahedron (12 vertices, 20 faces).
2. At each subdivision level, replace each triangle by 4 sub-triangles
   (midpoint subdivision).
3. Project all vertices to radius $R$.

**Vertex counts by level:**

| Level | $N_V$ | $N_F$ |
|-------|-------|-------|
| 0 | 12 | 20 |
| 1 | 42 | 80 |
| 2 | 162 | 320 |
| 3 | 642 | 1280 |
| 4 | 2562 | 5120 |
| 5 | 10242 | 20480 |

**Properties:**
- All triangles have approximately equal area: ideal for convergence studies.
- No singular vertices (uniform valence ≈ 6 for interior vertices).
- Gaussian curvature convergence: $O(h^2)$.

---

## `generate_torus` — standard torus

```julia
mesh = generate_torus(R, r, ntheta, nphi)
```

Generates a torus with major radius $R$ (distance from centre to tube centre)
and minor radius $r$ (tube radius).

**Parameterization:**

$$p(\theta, \phi) = \left((R + r\cos\phi)\cos\theta,\;
                          (R + r\cos\phi)\sin\theta,\;
                          r\sin\phi\right)$$

- $\theta \in [0, 2\pi)$ — toroidal angle (`ntheta` values)
- $\phi \in [0, 2\pi)$ — poloidal angle (`nphi` values)

**Topology:**
- Genus 1: $\chi = 0$, $\int_\Gamma K \, dA = 0$.
- Two independent cycles: toroidal ($\theta$) and poloidal ($\phi$).

**Convergence reference:** The torus is the canonical non-simply-connected
closed surface for testing Hodge decomposition and transport convergence.

---

## `generate_ellipsoid` — axis-aligned ellipsoid

```julia
mesh = generate_ellipsoid(a, b, c, nphi, ntheta)
```

Generates an ellipsoid with semi-axes $a$ (along $x$), $b$ (along $y$),
$c$ (along $z$).

**Parameterization:**

$$p(\phi, \theta) = \left(a\cos\phi\sin\theta,\; b\sin\phi\sin\theta,\; c\cos\theta\right)$$

**When $a = b = c = R$:** reduces to a UV sphere of radius $R$.

**Curvature formulas:**

Mean curvature at the poles $(0, 0, \pm c)$: $H = (a^2 + b^2 - c^2) / (a^2 b c)$
(approximate; exact only for a sphere of revolution with $a = b$).

**Use in convergence studies:** The ellipsoid geometry breaks spherical symmetry
while remaining smooth and simply-connected.

---

## `generate_perturbed_sphere` — "bumpy sphere"

```julia
mesh = generate_perturbed_sphere(R, ε, k, nphi, ntheta)
```

Generates a radially perturbed sphere with equation:

$$r(\phi, \theta) = R\bigl(1 + \varepsilon \cos(k\phi)\cos(k\theta)\bigr)$$

in spherical coordinates, giving vertices:

$$p(\phi, \theta) = r(\phi, \theta) \cdot \left(\cos\phi\sin\theta,\; \sin\phi\sin\theta,\; \cos\theta\right)$$

**Parameters:**
- $R$ — base radius
- $\varepsilon$ — perturbation amplitude (typically $0.05$ to $0.3$)
- $k$ — mode number (typically $2$–$4$; odd $k$ breaks anti-podal symmetry)

**When $\varepsilon = 0$:** reduces to a UV sphere.

**Use in convergence studies:** The bumpy sphere retains spherical topology
but is distinctly non-spherical, making it ideal for verifying that PDE solvers
do not exploit sphere symmetry.

---

## Mesh quality diagnostics

```julia
mesh = generate_icosphere(1.0, 3)
info = check_mesh(mesh)
println(info)
# (n_vertices=642, n_edges=1920, n_faces=1280, closed=true,
#  manifold=true, consistent_orientation=true, euler_characteristic=2)
```

All built-in generators produce closed, consistently-oriented manifold meshes
with the correct Euler characteristic.

---

## Summary table

| Generator | Surface | $\chi$ | Triangle quality | Notes |
|-----------|---------|--------|-----------------|-------|
| `sample_circle` | Circle (1-D) | — | Uniform | 2-D curve |
| `generate_uvsphere` | Sphere | 2 | Poor at poles | Avoid for convergence |
| `generate_icosphere` | Sphere | 2 | Excellent | Recommended |
| `generate_torus` | Torus | 0 | Good | Genus-1 reference |
| `generate_ellipsoid` | Ellipsoid | 2 | Moderate | Non-spherical geometry |
| `generate_perturbed_sphere` | Bumpy sphere | 2 | Moderate | Non-symmetric test |

---

## See also

- [Mesh types and data structures](01_mesh_types.md)
- [Topology and incidence matrices](02_topology.md)
