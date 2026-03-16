# Curvature

## Overview

FrontIntrinsicOps.jl computes three types of discrete curvature:

1. **Signed curvature** $\kappa$ — for polygonal curves in $\mathbb{R}^2$.
2. **Mean curvature** $H$ — for triangulated surfaces.
3. **Gaussian curvature** $K$ — via the angle-defect formula.

---

## Signed curvature of a polygonal curve

For a vertex $v_i$ on a closed polygonal curve with adjacent edges
$e_{i-1} = (v_{i-1}, v_i)$ and $e_i = (v_i, v_{i+1})$:

**Turning angle:**

$$\theta_i = \text{signed angle from } \hat{t}_{i-1} \text{ to } \hat{t}_i$$

where $\hat{t}_e$ is the unit tangent of edge $e$.

**Discrete signed curvature:**

$$\kappa_i = \frac{\theta_i}{\ell_i^*}$$

where $\ell_i^* = (\ell_{i-1} + \ell_i)/2$ is the dual length at $v_i$.

**Verification (circle of radius $R$, $N$ vertices):**

- Total length $\to 2\pi R$ as $N \to \infty$.
- $\kappa_i \to 1/R$ for all vertices.

```julia
mesh = sample_circle(1.0, 128)
geom = compute_geometry(mesh)
κ    = curvature(mesh, geom)    # ≈ 1.0 for unit circle
```

---

## Mean curvature of a triangulated surface

### Mean-curvature vector

The **mean-curvature normal** at vertex $v_i$ is defined via the
Laplace–Beltrami applied to the position vector:

$$\vec{H}_i = \frac{1}{2} \Delta_\Gamma \vec{p}\big|_i = \frac{1}{2} (L\vec{p})_i$$

where $\vec{p} = (p_x, p_y, p_z)^\top$ is the vector of coordinate values and
$L$ is applied column-wise.

**Discrete formula:**

$$\vec{H}_i = \frac{1}{2A_i^*} \sum_{j \in N(i)} w_{ij} (p_i - p_j)$$

This is the **cotangent-weight mean-curvature vector** of Pinkall and Polthier
(1993).

### Scalar mean curvature

$$H_i = \|\vec{H}_i\| \cdot \mathrm{sign}(\vec{H}_i \cdot \hat{n}_i)$$

The sign correction aligns the scalar mean curvature with the outward normal
convention: $H > 0$ for a convex surface (sphere), $H < 0$ for a saddle.

**Verification (sphere of radius $R$):**

On a sphere $H = 1/R$ at every point (principal curvatures $\kappa_1 = \kappa_2 = 1/R$,
$H = (\kappa_1 + \kappa_2)/2 = 1/R$).

```julia
R    = 2.0
mesh = generate_icosphere(R, 4)
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)
H    = mean_curvature(mesh, geom, dec)
println(maximum(abs, H .- 1/R))   # → 0 as mesh is refined
```

### Mean-curvature normal vector field

```julia
Hn = mean_curvature_normal(mesh, geom, dec)  # Vector{SVector{3,T}}
```

---

## Gaussian curvature — angle-defect formula

The discrete Gaussian curvature at an interior vertex $v_i$ is the
**angle defect** normalised by the dual area:

$$K_i = \frac{2\pi - \displaystyle\sum_{f \ni i} \theta_{f,i}}{A_i^*}$$

where $\theta_{f,i}$ is the interior angle of face $f$ at vertex $i$.

**Derivation:** On a smooth surface the Gauss–Bonnet theorem for a small disk
of radius $\varepsilon$ around $v_i$ gives

$$\int_{\text{disk}} K \, dA \approx K_i A_i^* = 2\pi - \text{(sum of angles)}$$

The formula uses this relation as a definition.

**Verification (sphere of radius $R$):**

$K = 1/R^2$ at every point; $\int_\Gamma K \, dA = 4\pi$ for any radius.

```julia
K     = gaussian_curvature(mesh, geom)
intK  = integrated_gaussian_curvature(mesh, geom)   # Σ K_i A_i^*
println(intK)   # ≈ 4π for sphere
```

---

## Gauss–Bonnet theorem

For a closed orientable surface:

$$\int_\Gamma K \, dA = 2\pi \chi$$

where $\chi = V - E + F$ is the Euler characteristic.

| Surface | $\chi$ | $\int K \, dA$ |
|---------|--------|-----------------|
| Sphere | $2$ | $4\pi \approx 12.566$ |
| Torus | $0$ | $0$ |
| Genus-$g$ surface | $2 - 2g$ | $2\pi(2-2g)$ |

**Discrete identity:** The angle-defect formula satisfies Gauss–Bonnet
**exactly** (before floating-point rounding):

$$\sum_{i=1}^{N_V} K_i A_i^* = 2\pi\chi$$

**Residual diagnostic:**

```julia
gb_res = gauss_bonnet_residual(mesh, geom)   # |∫K dA - 2π χ|
# gb_res < 1e-12 on well-constructed meshes
```

---

## Accessing curvature fields

```julia
geom_with_curv = compute_curvature(mesh, geom, dec)
# geom_with_curv.mean_curvature    Vector{T}
# geom_with_curv.gaussian_curvature Vector{T}
```

Or directly:

```julia
H  = mean_curvature(mesh, geom, dec)         # scalar mean curvature
Hn = mean_curvature_normal(mesh, geom, dec)  # mean-curvature vector
K  = gaussian_curvature(mesh, geom)          # angle-defect Gaussian curvature
```

---

## References

- Pinkall, U. & Polthier, K. (1993). *Computing discrete minimal surfaces and
  their conjugates.* Experimental Mathematics 2(1):15–36.
- Meyer, M., Desbrun, M., Schröder, P., & Barr, A.H. (2003). *Discrete
  differential-geometry operators for triangulated 2-manifolds.* In Visualization
  and Mathematics III, pp. 35–57.

---

## See also

- [Discrete geometry and dual areas](03_geometry.md)
- [Laplace–Beltrami operator](05_laplace_beltrami.md)
