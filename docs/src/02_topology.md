# Topology and Incidence Matrices

## Overview

The topology module extracts all combinatorial structure from a `SurfaceMesh`
(or `CurveMesh`) and assembles the **incidence matrices** that form the
backbone of discrete exterior calculus.

---

## `MeshTopology`

`build_topology(mesh::SurfaceMesh) → MeshTopology`

| Field | Type | Description |
|-------|------|-------------|
| `edges` | `Vector{Tuple{Int,Int}}` | All $N_E$ canonical edges $(i<j)$ |
| `edge_index` | `Dict{Tuple{Int,Int},Int}` | Fast lookup: edge → index |
| `face_edges` | `Vector{Vector{Int}}` | For each face, the 3 edge indices |
| `face_edge_signs` | `Vector{Vector{Int}}` | $\pm1$ — does face traverse edge in canonical direction? |
| `vertex_faces` | `Vector{Vector{Int}}` | For each vertex, adjacent face indices |
| `vertex_edges` | `Vector{Vector{Int}}` | For each vertex, adjacent edge indices |
| `edge_faces` | `Vector{Vector{Int}}` | For each edge, 1 or 2 adjacent face indices |

### Edge canonicalization

Given a triangle with vertices $(a, b, c)$, the three edges are represented as:

$$\{(min(a,b),\, max(a,b)),\quad (min(b,c),\, max(b,c)),\quad (min(a,c),\, max(a,c))\}$$

This canonical form ensures each physical edge appears exactly once in the
global edge list regardless of the faces it belongs to.

### Boundary detection

An edge is a **boundary edge** if `length(edge_faces[e]) == 1`.

A mesh is **closed** if all edges are interior: `is_closed(mesh)`.

A mesh is **manifold** if all edges have at most 2 adjacent faces: `is_manifold(mesh)`.

---

## Incidence matrix $d_0$ — vertex-to-edge coboundary

$d_0$ is the **exterior derivative on 0-cochains** (scalar vertex fields).
It maps a scalar $u : V \to \mathbb{R}$ to a 1-cochain $\alpha : E \to \mathbb{R}$:

$$[d_0 u]_e = u_{j} - u_{i}, \quad e = (i \to j)$$

**Matrix entry rule:**

$$[d_0]_{e,v} = \begin{cases}
+1 & \text{if } v = j \text{ (head of edge } e) \\
-1 & \text{if } v = i \text{ (tail of edge } e) \\
0  & \text{otherwise}
\end{cases}$$

**Size:** $N_E \times N_V$.  Each row has exactly two non-zeros: $+1$ and $-1$.

**Physical meaning:** $(d_0 u)_e$ is the **difference** of $u$ across edge $e$,
i.e. the edge-based discrete 1-form representing the gradient of $u$.

---

## Incidence matrix $d_1$ — edge-to-face coboundary

$d_1$ maps a 1-cochain $\alpha : E \to \mathbb{R}$ to a 2-cochain
$\beta : F \to \mathbb{R}$:

$$[d_1 \alpha]_f = \sum_{e \in \partial f} \sigma_{fe} \alpha_e$$

where $\sigma_{fe} \in \{+1, -1\}$ is the **relative orientation** of edge $e$
in face $f$ with respect to the canonical edge orientation.

**Matrix entry rule:**

$$[d_1]_{f,e} = \begin{cases}
+1 & \text{if edge } e \text{ is traversed positively in face } f \\
-1 & \text{if edge } e \text{ is traversed negatively in face } f \\
0  & \text{otherwise}
\end{cases}$$

**Size:** $N_F \times N_E$.  Each row has exactly three non-zeros: $+1$, $+1$, $-1$
or $-1$, $-1$, $+1$ (depending on orientation).

**Physical meaning:** $(d_1 \alpha)_f$ is the **circulation** of $\alpha$
around the boundary of face $f$ — the discrete analogue of the 2-form $d\alpha$.

---

## The chain complex identity: $d_1 \circ d_0 = 0$

This is the discrete analogue of $d^2 = 0$ (boundary of a boundary is empty).

**Proof sketch:** For any 0-cochain $u$,

$$[d_1 d_0 u]_f = \sum_{e \in \partial f} \sigma_{fe} [d_0 u]_e
= \sum_{e \in \partial f} \sigma_{fe}(u_j - u_i)$$

Each vertex $v$ in face $f$ appears in exactly two edges of $\partial f$,
once as head and once as tail.  The orientation signs ensure these contributions
cancel exactly: $[d_1 d_0 u]_f = 0$ for all $f$.

**Numerical check:**

```julia
dec = build_dec(mesh, geom)
residual = maximum(abs, dec.d1 * dec.d0)  # should be < 1e-14
```

`check_dec(mesh, geom, dec)` reports this as `d1_d0_max_residual`.

---

## Orientation consistency

A mesh has **consistent orientation** if every interior edge $(i, j)$ is
traversed as $(i \to j)$ by one adjacent face and as $(j \to i)$ by the other.
In matrix terms this means the two non-zero entries in column $e$ of $d_1$
have **opposite signs**.

`has_consistent_orientation(mesh)` checks this property.  STL files loaded
by `load_surface_stl` do not guarantee consistent orientation; use a mesh
repair tool if needed.

---

## Euler characteristic

For a triangulated 2-manifold (with or without boundary):

$$\chi = V - E + F$$

| Surface | $\chi$ |
|---------|--------|
| Sphere | $2$ |
| Disk (open) | $1$ |
| Torus | $0$ |
| Genus-$g$ closed surface | $2 - 2g$ |

**Computation:**

```julia
χ = euler_characteristic(mesh)   # returns V - E + F as Int
```

The Euler characteristic is topological: it is invariant under mesh refinement
of the same underlying surface.

---

## Curve topology

For a `CurveMesh`, the topology is simply an ordered sequence of vertices.
`curve_vertex_order(mesh)` returns the traversal order as a `Vector{Int}`.

A closed curve has $N$ edges and $N$ vertices; an open curve has $N-1$ edges.

---

## See also

- [Discrete exterior calculus](04_dec.md) — Hodge stars built on top of $d_0$, $d_1$
- [Open surfaces](14_open_surfaces.md) — Boundary edge/vertex detection
