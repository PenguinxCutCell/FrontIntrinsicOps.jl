# Open Surfaces and Boundary Conditions

## Overview

An **open surface** is a triangulated manifold-with-boundary.  The boundary
$\partial\Gamma$ is a collection of edges each adjacent to only one face.
This module provides:

- Automatic detection of boundary edges and vertices.
- Strong Dirichlet boundary condition enforcement.
- Symmetric Dirichlet enforcement (preserves sparsity pattern).
- Neumann flux boundary condition assembly.
- A complete open-surface Poisson solver.

---

## Boundary detection

A **boundary edge** is an edge with exactly one adjacent face:

```julia
bnd_edges = detect_boundary_edges(topo)    # Vector{Int} — edge indices
```

A **boundary vertex** is a vertex incident to at least one boundary edge:

```julia
bnd_verts = detect_boundary_vertices(mesh, topo)   # Vector{Int} — vertex indices
```

An open surface has at least one boundary edge:

```julia
is_open = is_open_surface(topo)   # Bool
```

---

## Dirichlet boundary conditions

### Strong enforcement (asymmetric)

For each boundary vertex $v$ with prescribed value $g_v$:

1. Zero out row $v$ in the system matrix $A$.
2. Set $A[v, v] = 1$.
3. Set $b[v] = g_v$.

This enforces $u_v = g_v$ exactly.  The resulting matrix is **no longer
symmetric**.

```julia
apply_dirichlet_to_system!(A, b, bnd_verts, values)   # in-place modification
```

### Symmetric enforcement

To preserve matrix symmetry:

1. For each Dirichlet DOF $v$ with value $g_v$:
   - Zero out row $v$ and column $v$.
   - Set $A[v,v] = 1$, $b[v] = g_v$.
   - For each interior neighbor $j$ of $v$: subtract $A[j,v] g_v$ from $b[j]$.
2. The resulting system has the same solution as the asymmetric form but
   preserves symmetry for efficiency.

```julia
apply_dirichlet_symmetric!(A, b, bnd_verts, values)
```

---

## Neumann boundary conditions

The Neumann condition prescribes the **normal flux** on the boundary:

$$\nabla_\Gamma u \cdot \hat{\nu} = g \quad \text{on } \partial\Gamma$$

where $\hat{\nu}$ is the outward boundary normal (in the tangent plane of $\Gamma$,
pointing away from the interior).

The weak form adds a boundary integral to the right-hand side:

$$\int_{\partial\Gamma} g \, \phi_i \, ds \quad \text{for each interior vertex } i$$

**Discrete assembly:**

```julia
add_neumann_rhs!(b, mesh, geom, topo, g)   # adds ∫_{∂Γ} g φ_i ds to b
```

The boundary mass matrix (1-D integral on $\partial\Gamma$):

$$[M_b]_{ij} = \int_{\partial\Gamma} \phi_i \phi_j \, ds$$

```julia
Mb = boundary_mass_matrix(mesh, geom, topo)
```

---

## Poisson problem on an open surface

$$-\Delta_\Gamma u = f \quad \text{in } \Gamma \setminus \partial\Gamma$$
$$u = g \quad \text{on } \partial\Gamma$$

**Solvability:** With Dirichlet BCs on all of $\partial\Gamma$, the system is
uniquely solvable (no compatibility condition needed — the constant nullspace
of $L$ is broken by the boundary conditions).

```julia
u = solve_open_surface_poisson(mesh, geom, dec, topo, f, bnd_verts, bnd_vals)
```

**Algorithm:**
1. Assemble $L$ (full Laplacian).
2. Apply `apply_dirichlet_symmetric!(L, f, bnd_verts, bnd_vals)`.
3. Solve the modified system.

---

## Convergence on open surfaces

**Test case (Laplace on a flat patch):** Let $\Omega = [0,1]^2$ meshed as
a flat triangulated square.  The exact solution $u(x,y) = x$ satisfies
$\Delta u = 0$.  For P1 finite elements the solution is **exact** (no
discretization error).

**Test case (Poisson on a flat patch):** The exact solution
$u(x,y) = \sin(\pi x)\sin(\pi y)$ with $f = 2\pi^2 u$ satisfies $\Delta u = -f$.
Expected convergence rate: $O(h^2)$ in the $L^\infty$ norm.

---

## Detecting open vs. closed meshes

```julia
mesh_closed = generate_icosphere(1.0, 3)
mesh_open   = generate_spherical_cap(1.0, 0.7, 30, 30)   # custom generator

topo_closed = build_topology(mesh_closed)
topo_open   = build_topology(mesh_open)

is_open_surface(topo_closed)   # false
is_open_surface(topo_open)     # true
```

---

## Boundary normal flux evaluation

After solving, verify the flux condition:

```julia
flux = boundary_normal_flux(mesh, geom, topo, u)   # Vector{T} at boundary edges
```

The normal flux $\partial_\nu u$ is approximated by a one-sided finite difference
along each boundary edge.

---

## Relationship to Dirichlet Laplacian

On an open surface with Dirichlet BCs, the system matrix after BC enforcement
has all real positive eigenvalues (it is positive definite).  The smallest
eigenvalue corresponds to the first Dirichlet eigenfunction.

For the flat unit square the first Dirichlet eigenvalue is:

$$\lambda_1 = \pi^2 + \pi^2 = 2\pi^2 \approx 19.74$$

---

## See also

- [Laplace–Beltrami operator](05_laplace_beltrami.md)
- [Surface diffusion, Poisson, Helmholtz](07_surface_diffusion.md)
- [Topology and incidence matrices](02_topology.md)
