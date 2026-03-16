# Discrete Exterior Calculus (DEC)

## Overview

**Discrete Exterior Calculus** is a coordinate-free calculus on triangulated
manifolds that mirrors smooth exterior calculus.  The key objects are:

- **$k$-cochains** — discrete $k$-forms: scalar values assigned to
  $k$-simplices (vertices, edges, faces).
- **Exterior derivatives** $d_0$, $d_1$ — encode topological (combinatorial)
  differentiation.
- **Hodge stars** $\star_0$, $\star_1$, $\star_2$ — encode the metric
  (geometry) by mapping primal cochains to dual cochains.

All global operators are assembled as sparse matrices.

---

## The cochain complex

$$\Omega^0 \xrightarrow{d_0} \Omega^1 \xrightarrow{d_1} \Omega^2$$

| Space | Discrete object | Size |
|-------|-----------------|------|
| $\Omega^0$ | scalar vertex fields $u : V \to \mathbb{R}$ | $N_V$ |
| $\Omega^1$ | edge 1-forms $\alpha : E \to \mathbb{R}$ | $N_E$ |
| $\Omega^2$ | face 2-forms $\beta : F \to \mathbb{R}$ | $N_F$ |

The exterior derivatives $d_0$ and $d_1$ are the sparse incidence matrices
described in [Topology](02_topology.md).

---

## Hodge stars

The Hodge star converts a primal $k$-cochain to a dual $(n-k)$-cochain
(where $n=2$ for surfaces, $n=1$ for curves).  In the lumped / diagonal
approximation used here, all three Hodge stars are **diagonal matrices**.

### $\star_0$ — mass matrix

$$\star_0 = \mathrm{diag}(A_1^*, A_2^*, \ldots, A_{N_V}^*)$$

where $A_i^*$ is the vertex dual area (barycentric or mixed — see
[Geometry](03_geometry.md)).

For curves: $[\star_0]_{ii} = \ell_i^*$ (vertex dual length).

**Physical meaning:** $\star_0$ is the discrete mass matrix.  The integral of
a vertex field is $\mathbf{1}^\top \star_0 \, u = \sum_i A_i^* u_i$.

### $\star_1$ — edge Hodge star (cotan weights)

$$[\star_1]_{ee} = w_e = \frac{1}{2}(\cot\alpha_e + \cot\beta_e)$$

where $\alpha_e$ and $\beta_e$ are the angles opposite edge $e$ in the two
triangles sharing it.

For a boundary edge (one adjacent triangle only): $w_e = \frac{1}{2}\cot\alpha_e$.

For curves: $[\star_1]_{ee} = \ell_e / \ell_e^*$.

**Physical meaning:** $\star_1$ encodes the **dual edge length** divided by
the **primal edge length**.  On a Delaunay mesh this is the actual ratio of
dual-to-primal edge lengths.  The formula $w_e = \frac{1}{2}(\cot\alpha+\cot\beta)$
is a classical result of Pinkall and Polthier (1993).

### $\star_2$ — face Hodge star

$$[\star_2]_{ff} = \frac{1}{A_f}$$

(inverse face area — maps primal 2-forms to dual 0-forms).

---

## Codifferential

The **codifferential** is the $L^2$-adjoint of the exterior derivative.
In DEC it is:

$$\delta_1 = \star_0^{-1} d_0^\top \star_1 : \Omega^1 \to \Omega^0$$

$$\delta_2 = \star_1^{-1} d_1^\top \star_2 : \Omega^2 \to \Omega^1$$

These are assembled in `kforms.jl`.

---

## Hodge Laplacians

The **scalar Hodge Laplacian** (Laplace–Beltrami on 0-forms):

$$\Delta_0 = \delta_1 d_0 = \star_0^{-1} d_0^\top \star_1 d_0 : \Omega^0 \to \Omega^0$$

The sign convention in this package is $L = -\Delta_\Gamma$, so $L$ is
**positive semi-definite** (see [Laplace–Beltrami](05_laplace_beltrami.md)).

The **1-form Hodge Laplacian**:

$$\Delta_1 = \delta_2 d_1 + d_0 \delta_1 : \Omega^1 \to \Omega^1$$

This is the full Hodge Laplacian on 1-forms, used in
[Hodge Decomposition](12_hodge_decomposition.md).

---

## DEC gradient and divergence

**Gradient** of a 0-form (vertex scalar):

$$\nabla_h u = d_0 u \in \Omega^1$$

This is a 1-form whose values are the differences $u_j - u_i$ along each edge.

**Divergence** of a 1-form:

$$\nabla_h \cdot \alpha = \delta_1 \alpha = \star_0^{-1} d_0^\top \star_1 \alpha \in \Omega^0$$

These functions are accessible as `gradient_0_to_1(mesh, dec, u)` and
`divergence_1_to_0(mesh, geom, dec, α)`.

---

## Whitney interpolation and the geometric meaning of $\star_1$

The cotan weight $w_e$ arises naturally from the **Whitney 1-form** associated
with edge $e$.  Given a scalar function $u$ on a triangle with vertices
$\{a, b, c\}$, the Whitney 1-form for edge $(a,b)$ is

$$\phi^{ab} = u_a \lambda_b d\lambda_a - u_b \lambda_a d\lambda_b$$

where $\lambda_a, \lambda_b$ are barycentric coordinates.  Integrating the
inner product $\langle \phi^{ab}, \phi^{ab} \rangle$ over the triangle yields
the cotan weight.  This connection is made explicit in:

> Desbrun M., Hirani A.N., Leok M., Marsden J.E. (2005).
> *Discrete exterior calculus.*

---

## Key identities

| Identity | Meaning |
|----------|---------|
| $d_1 d_0 = 0$ | Boundary of boundary is empty |
| $\star_0 \succ 0$ | Positive dual areas |
| $\star_2 \succ 0$ | Positive face areas |
| $\ker(d_0) = \{0\}$ on trees | Trivial for connected mesh |
| $\ker(d_1) \supseteq \mathrm{im}(d_0)$ | Closed forms contain exact forms |
| $\dim \ker \Delta_0 = $ 1 on closed surface | Constant function |
| $\dim \ker \Delta_1 = 2g$ on genus-$g$ surface | Harmonic 1-forms |

---

## Assemblying the DEC

```julia
mesh = generate_icosphere(1.0, 3)
geom = compute_geometry(mesh)            # dual areas, edge lengths, etc.
dec  = build_dec(mesh, geom)             # d0, d1, star0, star1, star2, L

# Access individual operators
d0     = dec.d0      # N_E × N_V  sparse
d1     = dec.d1      # N_F × N_E  sparse
star0  = dec.star0   # N_V × N_V  diagonal sparse
star1  = dec.star1   # N_E × N_E  diagonal sparse
L      = dec.laplacian  # N_V × N_V  symmetric sparse positive semi-definite
```

---

## See also

- [Topology and incidence matrices](02_topology.md)
- [Laplace–Beltrami operator](05_laplace_beltrami.md)
- [Hodge Decomposition](12_hodge_decomposition.md)
