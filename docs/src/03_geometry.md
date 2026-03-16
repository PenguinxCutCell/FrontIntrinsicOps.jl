# Discrete Geometry and Dual Areas

## Overview

The geometry module computes **intrinsic geometric quantities** on the primal
mesh.  The key outputs are face normals and areas, edge lengths, and — most
importantly for the PDE layer — **vertex dual areas** $A_i^*$ that represent
the area element associated with each vertex.

---

## Face geometry

For a triangle with vertices $p_a, p_b, p_c \in \mathbb{R}^3$:

**Outward face normal (unit):**

$$\hat{n}_f = \frac{(p_b - p_a) \times (p_c - p_a)}{\|(p_b - p_a) \times (p_c - p_a)\|}$$

**Face area:**

$$A_f = \frac{1}{2} \|(p_b - p_a) \times (p_c - p_a)\|$$

**Interior angles:**

$$\theta_a = \arccos\!\left(\frac{(p_b - p_a) \cdot (p_c - p_a)}{|p_b - p_a|\, |p_c - p_a|}\right)$$

and cyclically for $\theta_b$, $\theta_c$.

---

## Edge lengths

For a canonical edge $(i, j)$:

$$\ell_{ij} = \|p_j - p_i\|$$

---

## Vertex normals

The vertex normal at $v$ is an area-weighted average of the adjacent face normals:

$$\hat{n}_v = \frac{\displaystyle\sum_{f \ni v} A_f \, \hat{n}_f}{\left\|\displaystyle\sum_{f \ni v} A_f \, \hat{n}_f\right\|}$$

This is the most common definition for smooth-surface rendering and curvature
estimation.

---

## Dual areas

The **dual area** $A_i^*$ at vertex $i$ serves as the integration weight for
vertex-based fields.  It approximates the area of the Voronoi cell dual to
vertex $i$.

Two formulations are supported via `compute_geometry(mesh; dual_area=:xxx)`.

### Barycentric dual (`:barycentric`, default)

The simplest choice: each face contributes one third of its area to each of its
three vertices.

$$A_i^{\text{bary}} = \frac{1}{3} \sum_{f \ni i} A_f$$

**Properties:**
- Always positive.
- Sums to the total surface area: $\sum_i A_i^{\text{bary}} = A_\Gamma$.
- Simple and robust even for low-quality meshes.
- Not geometrically meaningful for the dual Voronoi cell on irregular meshes.

### Mixed / Voronoi dual (`:mixed` or `:voronoi`)

Based on Meyer, Desbrun, Schröder, Barr (2003), this formula uses the actual
Voronoi circumcentric areas for acute triangles and falls back to barycentric
areas for obtuse triangles.

For each triangle $T = (a, b, c)$ with opposite edge lengths $\ell_a, \ell_b, \ell_c$
and angles $\theta_a, \theta_b, \theta_c$:

**Case 1: Non-obtuse triangle** (all angles $\leq \pi/2$):

The Voronoi contribution to vertex $a$ is:

$$A_a^{\text{Vor}} = \frac{1}{8}\Bigl(\ell_b^2 \cot\theta_c + \ell_c^2 \cot\theta_b\Bigr)$$

where $\ell_b = |ac|$ and $\ell_c = |ab|$.

**Case 2: Obtuse at $a$** ($\theta_a > \pi/2$):

$$A_a = \frac{A_T}{2}, \quad A_b = A_c = \frac{A_T}{4}$$

**Case 3: Obtuse at $b$ (or $c$)**: analogous with the obtuse vertex receiving
$A_T/4$ and the opposite vertex receiving $A_T/2$.

**Properties:**
- Always positive (no negative areas).
- Equals the Voronoi cell area for Delaunay triangulations.
- More accurate for curvature estimation than barycentric.
- Sum: $\sum_i A_i^{\text{mixed}} = A_\Gamma$.

**Selection:**

```julia
geom = compute_geometry(mesh; dual_area=:barycentric)  # default
geom = compute_geometry(mesh; dual_area=:mixed)        # Meyer et al. 2003
geom = compute_geometry(mesh; dual_area=:voronoi)      # alias for :mixed
```

---

## Curve geometry

For a `CurveMesh` the analogous quantities are:

**Edge length:** $\ell_e = \|p_{j} - p_i\|$

**Edge tangent:** $\hat{t}_e = (p_j - p_i)/\ell_e$

**Vertex dual length:**

$$\ell_i^* = \frac{\ell_{e_{i-1}} + \ell_{e_i}}{2}$$

the half-sum of the two adjacent edge lengths.

**Vertex normal:** The left-perpendicular of the averaged tangent at $i$.

---

## Integration formulas

Given vertex-based field $u$ and face-based field $w$:

$$\int_\Gamma u \, dA \approx \sum_{i=1}^{N_V} u_i \, A_i^*, \qquad
\int_\Gamma w \, dA \approx \sum_{f=1}^{N_F} w_f \, A_f$$

Total surface area and enclosed volume:

```julia
area   = measure(mesh, geom)          # Σ A_f
volume = enclosed_measure(mesh)       # divergence theorem: (1/6)|Σ a·(b×c)|
```

---

## Non-positive cotan weights (star1)

The cotan weight for edge $(i, j)$ shared by two triangles is

$$w_{ij} = \frac{1}{2}(\cot\alpha_{ij} + \cot\beta_{ij})$$

where $\alpha_{ij}$ and $\beta_{ij}$ are the angles **opposite** edge $(i,j)$
in the two adjacent triangles.

For an **obtuse** triangle, one angle exceeds $\pi/2$, making the cotangent
**negative**.  If both opposite angles are obtuse (impossible for a valid
triangle but possible after degenerate mesh operations), $w_{ij}$ itself can
be negative.  This does not break correctness of the Laplacian (which remains
consistent) but can cause instabilities in transport schemes that rely on
$w_{ij} > 0$.

**Diagnostic:**

```julia
report = star1_sign_report(dec)
# report.n_nonpositive   number of negative entries
# report.frac_nonpositive  fraction
# report.min_entry       most negative entry
```

---

## See also

- [Mesh types](01_mesh_types.md)
- [Discrete exterior calculus](04_dec.md) — how dual areas enter $\star_0$
- [Curvature](06_curvature.md) — uses dual areas as denominators
