# Continuum-to-Discrete Mathematical Development for `FrontIntrinsicOps.jl`

This chapter gives a complete mathematical map from continuum geometry/PDEs to the discrete operators implemented in `FrontIntrinsicOps.jl`.

It covers:

1. geometric and analytic continuum definitions,
2. simplicial/cochain discretization and DEC operators,
3. FEEC-compatible Whitney spaces and projections,
4. topology/harmonic/cohomology machinery,
5. transport, diffusion, reaction-diffusion, advection-diffusion, and boundary-value problems,
6. geodesic/parallel-transport/exterior-calculus utilities.

The goal is implementation-level precision: each defined object matches package conventions.

---

## 1. Domain, Unknowns, and Conventions

## 1.1 Geometric domain

The package works on static front meshes approximating a smooth manifold `\Gamma`:

- curve case: `\Gamma` is a 1D manifold in `\mathbb{R}^2` (polygonal chain/loop),
- surface case: `\Gamma` is a 2D manifold in `\mathbb{R}^3` (triangulated surface).

Primary mesh types:

- `CurveMesh{T}` with vertices `x_i \in \mathbb{R}^2` and oriented edges `e=(i\to j)`,
- `SurfaceMesh{T}` with vertices `x_i \in \mathbb{R}^3` and oriented faces `f=(i,j,k)`.

## 1.2 Sign conventions

The package uses the positive-semidefinite Laplace convention

```math
L = -\Delta_\Gamma,
```

with `\Delta_\Gamma = \operatorname{div}_\Gamma \nabla_\Gamma` the continuum Laplace-Beltrami operator.

Hence constants lie in the nullspace of `L` (closed connected manifold), and non-constant smooth modes have positive eigenvalues.

## 1.3 Edge orientation convention

For surfaces, global unique primal edges are stored in canonical unoriented order `(i,j)` with `i<j`. Orientation enters through signs.

- canonical oriented edge for cochains: `i\to j` with `i<j`,
- face local boundary orientation induces signs `\sigma_{f,e}\in\{\pm1\}`.

For curves, edges are already stored as directed pairs in `mesh.edges`.

---

## 2. Continuum Differential Geometry and PDE Operators

Let `\Gamma` be smooth, with tangent projector `P(x):\mathbb{R}^d\to T_x\Gamma`.

For scalar `u` and tangential vector field `v`:

```math
\nabla_\Gamma u = P\,\nabla \tilde u,
\qquad
\operatorname{div}_\Gamma v = \operatorname{tr}(P\nabla v),
\qquad
\Delta_\Gamma u = \operatorname{div}_\Gamma(\nabla_\Gamma u).
```

On surfaces (`m=2`), differential forms satisfy the de Rham complex:

```math
\Omega^0(\Gamma) \xrightarrow{d} \Omega^1(\Gamma) \xrightarrow{d} \Omega^2(\Gamma),
\qquad d^2=0.
```

Hodge star and codifferential:

```math
\delta_k = (-1)^{m(k+1)+1}\,*\,d\,*.
```

Hodge Laplacian on `k`-forms:

```math
\Delta_k = d\delta + \delta d.
```

Hodge decomposition on closed orientable surfaces:

```math
\omega = d\alpha + \delta\beta + h,
\qquad dh=0,\ \delta h=0.
```

---

## 3. Simplicial Discretization and Cochains

Let `K` be the simplicial mesh of `\Gamma`.

- primal 0-cells: vertices,
- primal 1-cells: edges,
- primal 2-cells: faces (surface only).

A discrete `k`-form is a primal `k`-cochain (vector of DOFs on oriented `k`-simplices).

## 3.1 Topology container (`MeshTopology`)

For surfaces, topology extraction defines:

- unique edge list `E={e_1,\dots,e_{n_E}}`,
- per-face edge indices `face_edges[f]`,
- per-face orientation signs `face_edge_signs[f]`,
- adjacency lists `vertex_faces`, `vertex_edges`, `edge_faces`.

This separates combinatorics from metric quantities.

## 3.2 Incidence (discrete exterior derivative)

For curves and surfaces, `d_0` is vertex-to-edge incidence:

```math
(d_0 u)[e=(i\to j)] = u_j - u_i.
```

For surfaces, `d_1` is edge-to-face incidence:

```math
(d_1\alpha)[f] = \sum_{e\subset \partial f} \sigma_{f,e}\,\alpha[e].
```

Exactness:

```math
d_1 d_0 = 0
```

(up to floating-point roundoff).

---

## 4. Discrete Geometry Definitions

## 4.1 Curve geometry (`CurveGeometry`)

Given edge `e=(i\to j)`:

```math
\ell_e = \|x_j-x_i\|,
\qquad
\hat t_e = (x_j-x_i)/\ell_e.
```

Vertex dual length (barycentric 1D dual):

```math
|\star v_i| = \frac12\sum_{e\ni i} \ell_e.
```

Signed turning-angle curvature at vertex `i`:

```math
\kappa_i = \theta_i / |\star v_i|,
```

where `\theta_i` is signed angle from incoming to outgoing tangent.

## 4.2 Surface geometry (`SurfaceGeometry`)

For face `f=(a,b,c)`:

```math
n_f^{raw}=(x_b-x_a)\times(x_c-x_a),
\quad
A_f=\frac12\|n_f^{raw}\|,
\quad
\hat n_f=n_f^{raw}/\|n_f^{raw}\|.
```

Unique edge length: `\ell_e=\|x_j-x_i\|`.

### 4.2.1 Vertex dual areas

Two methods are implemented.

1. Barycentric dual:

```math
A_i^\star = \sum_{f\ni i} \frac{A_f}{3}.
```

2. Mixed/Voronoi (Meyer et al.) per triangle:

- non-obtuse triangle: circumcentric contribution,
- obtuse at vertex `i`: `A_f/2` to that vertex,
- obtuse elsewhere: `A_f/4` to each non-obtuse vertex.

For non-obtuse triangle and vertex `a`:

```math
A_{a,f}^{vor} = \frac18\Big(\|x_b-x_a\|^2\cot\angle c + \|x_c-x_a\|^2\cot\angle b\Big),
```

with analogous formulas for other vertices.

### 4.2.2 Vertex normals

Area-weighted average, normalized:

```math
\hat n_i = \frac{\sum_{f\ni i} A_f\hat n_f}{\left\|\sum_{f\ni i} A_f\hat n_f\right\|}.
```

---

## 5. Discrete Hodge Stars and DEC Inner Products

All stars are diagonal sparse matrices.

## 5.1 Curve stars

```math
\star_0 = \operatorname{diag}(|\star v_i|),
\qquad
\star_1 = \operatorname{diag}(1/\ell_e).
```

(`\star_1=1/\ell_e` is required for consistency of `L=\star_0^{-1}d_0^T\star_1d_0`.)

## 5.2 Surface stars

```math
\star_0 = \operatorname{diag}(A_i^\star),
\qquad
\star_2 = \operatorname{diag}(A_f),
```

```math
\star_1 = \operatorname{diag}(w_e),
\qquad
w_e = \frac12(\cot\alpha_e+\cot\beta_e),
```

where `\alpha_e,\beta_e` are angles opposite edge `e` in adjacent triangles (single cotan on boundary edges).

Discrete inner products are induced by stars, e.g.

```math
\langle a,b\rangle_{\star_1}=a^T\star_1 b.
```

---

## 6. DEC Differential Operators

## 6.1 Scalar Laplace-Beltrami

Implemented as

```math
L = \star_0^{-1}d_0^T\star_1 d_0.
```

Equivalent direct cotan form at vertex `i`:

```math
(Lu)_i = \frac{1}{A_i^\star}\sum_{j\in N(i)} w_{ij}(u_i-u_j).
```

Both `:dec` and `:cotan` assembly paths are available.

## 6.2 Gradient/divergence/curl-like maps

- gradient (`0\to1`): `d_0 u`,
- divergence (`1\to0`) on surfaces:

```math
\delta_1\alpha = \star_0^{-1}d_0^T\star_1\alpha,
```

- curl-like (`1\to2`): `d_1\alpha`.

## 6.3 Codifferentials and Hodge Laplacians

On surfaces:

```math
\delta_1 = \star_0^{-1}d_0^T\star_1,
\qquad
\delta_2 = \star_1^{-1}d_1^T\star_2,
```

```math
\Delta_0 = \delta_1 d_0,
\qquad
\Delta_1 = \delta_2 d_1 + d_0\delta_1.
```

---

## 7. Curvature and Integral Geometry

## 7.1 Mean curvature on surfaces

Using embedding coordinates `x=(x^1,x^2,x^3)` and discrete `L`:

```math
H_n = \frac12 Lx
```

(vertex-wise vector).

Scalar mean curvature is magnitude with orientation-dependent sign using vertex normals.

## 7.2 Gaussian curvature (angle defect)

At vertex `i`:

```math
K_i = \frac{2\pi - \sum_{f\ni i}\theta_{i,f}}{A_i^\star}.
```

Integrated Gaussian curvature (dual-area independent in angle-defect form):

```math
\int_\Gamma K\,dA \approx \sum_i\Big(2\pi-\sum_{f\ni i}\theta_{i,f}\Big).
```

## 7.3 Measures and enclosed measures

- curve measure: `\sum_e \ell_e`,
- surface measure: `\sum_f A_f`,
- enclosed polygon area (shoelace),
- enclosed surface volume:

```math
V = \frac16\left|\sum_f x_a\cdot(x_b\times x_c)\right|.
```

---

## 8. Topological Diagnostics and Invariants

Euler characteristic (surface):

```math
\chi = V-E+F.
```

Gauss-Bonnet residual:

```math
\left|\int K\,dA - 2\pi\chi\right|.
```

DEC checks include:

- `\|d_1d_0\|` near zero,
- `\|L\mathbf{1}\|` near zero,
- positivity/sign report for star diagonals.

---

## 9. FEEC Layer: Lowest-Order Whitney Complex

The FEEC-compatible layer adds explicit spaces and consistent mass/stiffness assembly.

## 9.1 Spaces and sequence

Surface:

```math
0 \to \Lambda_h^0 \xrightarrow{d_0} \Lambda_h^1 \xrightarrow{d_1} \Lambda_h^2 \to 0.
```

DOF placement:

- `\Lambda_h^0`: vertex values,
- `\Lambda_h^1`: oriented edge integrals,
- `\Lambda_h^2`: oriented face integrals.

Curve:

```math
0 \to \Lambda_h^0 \xrightarrow{d_0} \Lambda_h^1 \to 0.
```

## 9.2 Local Whitney basis on one triangle

For barycentric coordinates `\lambda_1,\lambda_2,\lambda_3`:

```math
w_i^{(0)} = \lambda_i,
```

```math
w_{ij}^{(1)} = \lambda_i\nabla\lambda_j - \lambda_j\nabla\lambda_i,
```

with local ordering `(1\to2),(2\to3),(3\to1)`.

The package computes constant `\nabla\lambda_i` via

```math
\nabla\lambda_1 = \frac{\hat n\times(x_3-x_2)}{2A},
\quad
\nabla\lambda_2 = \frac{\hat n\times(x_1-x_3)}{2A},
\quad
\nabla\lambda_3 = \frac{\hat n\times(x_2-x_1)}{2A}.
```

Lowest-order 2-form basis (one per face):

```math
w^{(2)} = 1/A,
```

so that `\int_f w^{(2)}\,dA=1`.

## 9.3 Reconstruction from cochains

Given vertex cochain `c_0`, face-restricted scalar field:

```math
u_h|_f = \sum_{i\in f} c_{0,i}\lambda_i,
\qquad
\nabla u_h|_f = \sum_{i\in f} c_{0,i}\nabla\lambda_i.
```

Given edge cochain `c_1`, face-restricted 1-form (vector representation):

```math
\alpha_h|_f = \sum_{k=1}^3 \tilde c_{1,k}\,w_k^{(1)},
```

where `\tilde c_{1,k}` includes face-edge orientation sign transfer.

Given face cochain `c_2[f]=\int_f\beta`, reconstructed density:

```math
\rho_f = c_2[f]/A_f.
```

---

## 10. FEEC Interpolators and Commuting Structure

Canonical interpolation operators:

```math
\Pi_0: \Omega^0\to\Lambda_h^0,
\quad
(\Pi_0 f)_i = f(x_i),
```

```math
\Pi_1: \Omega^1\to\Lambda_h^1,
\quad
(\Pi_1\alpha)_e \approx \int_e \alpha,
```

```math
\Pi_2: \Omega^2\to\Lambda_h^2,
\quad
(\Pi_2\beta)_f \approx \int_f \beta.
```

Implemented input representations:

- `\Pi_1`: ambient vector or line density,
- `\Pi_2`: density or direct face integral callback.

Commuting identities are verified by dedicated diagnostics/tests:

```math
\Pi_1(df) \approx d_0\Pi_0(f),
\qquad
\Pi_2(d\alpha) \approx d_1\Pi_1(\alpha).
```

---

## 11. Consistent Whitney Mass/Stiffness Assembly

## 11.1 `M0` (0-form mass)

Surface local triangle matrix:

```math
M_0^{loc} = \frac{A_f}{12}
\begin{bmatrix}
2&1&1\\
1&2&1\\
1&1&2
\end{bmatrix}.
```

Curve local segment matrix:

```math
M_0^{loc} = \frac{\ell_e}{6}
\begin{bmatrix}
2&1\\
1&2
\end{bmatrix}.
```

## 11.2 `M1` (1-form mass)

Surface: assembled per face via degree-2 exact barycentric quadrature on Whitney-1 basis.

Curve: diagonal with entries `1/\ell_e` (segment-wise 1-form basis normalization).

## 11.3 `M2` (2-form mass)

Surface diagonal:

```math
M_2[f,f] = 1/A_f.
```

## 11.4 `K0` (0-form stiffness)

Surface:

```math
K_{ij} = \sum_f A_f\,\nabla\lambda_i\cdot\nabla\lambda_j.
```

Curve segment local matrix:

```math
K^{loc}=\frac1{\ell_e}
\begin{bmatrix}
1&-1\\
-1&1
\end{bmatrix}.
```

## 11.5 Strong-form Whitney Laplacians

Implemented approximations:

```math
L_0^{wh} = M_{0,lumped}^{-1}K_0,
```

```math
L_1^{wh} \approx \delta_2 d_1 + d_0\delta_1
```

with lumped inverses in codifferentials.

---

## 12. Mixed FEEC Solves and Gauges

## 12.1 0-form mixed solve

For closed manifolds, mean-zero gauge is enforced by augmenting with a Lagrange multiplier `\lambda`:

```math
\begin{bmatrix}
L_0 & w\\
w^T & 0
\end{bmatrix}
\begin{bmatrix}
u\\\lambda
\end{bmatrix}
=
\begin{bmatrix}
b\\0
\end{bmatrix},
```

where `w` is lumped mass row-sum vector.

## 12.2 1-form mixed solve

RHS is projected orthogonally to harmonic subspace in `M1` inner product. A mild diagonal regularization `\tau I` is added when solving singular systems; solution is reprojected to harmonic-orthogonal gauge.

---

## 13. Cohomology, Betti Numbers, Harmonic Basis, Hodge Decomposition

## 13.1 Betti numbers

From connectivity and Euler characteristic:

- closed orientable: `\beta_2=\beta_0`, `\beta_1=\beta_0+\beta_2-\chi`,
- open orientable (current formula): `\beta_2=0`, `\beta_1=\beta_0-\chi`.

## 13.2 Cycle basis construction

Implemented via primal spanning forest + dual cotree. Non-tree/non-cotree edges generate signed edge cycles (combinatorial `H_1` generators).

## 13.3 Cohomology representatives and harmonicization

Cycle cochains are projected to `\ker(d_1)` using least-squares constraint solve; then harmonic basis is extracted by constrained solve and Hodge-orthonormalization (`\star_1` inner product).

## 13.4 Full discrete decomposition

For edge cochain `\omega`:

```math
\omega = d\alpha + \delta\beta + h + r,
```

where residual `r` is numerical (regularization/tolerance controlled). API returns exact, coexact, harmonic components and potentials.

---

## 14. Surface Vector Calculus Utilities

## 14.1 Tangential projection

```math
v_\tau = v-(v\cdot\hat n)\hat n.
```

## 14.2 Face gradient reconstruction from vertex scalar field

For `f=(a,b,c)`:

```math
\nabla_\Gamma u|_f = \frac{1}{2A_f}
\left(
 u_a\,\hat n_f\times(x_c-x_b)
+u_b\,\hat n_f\times(x_a-x_c)
+u_c\,\hat n_f\times(x_b-x_a)
\right).
```

## 14.3 1-form <-> tangent vector conversion

- `\alpha` (edge cochain) to face tangent vector via face-edge signed accumulation and cross with face normal,
- face tangent vector to edge 1-form by edge-tangent projection and face averaging.

Divergence of tangent field is computed through this conversion and `\delta_1`.

---

## 15. Discrete Exterior Algebra: Wedge, Interior Product, Lie Derivative

## 15.1 Wedge

Implemented low-order cases:

- `0\wedge k`: scalar averaging on target simplex (`k=0,1,2` as applicable),
- surface `1\wedge1\to2`:

```math
(\alpha\wedge\beta)_f = A_f\,\hat n_f\cdot(V_\alpha\times V_\beta).
```

## 15.2 Interior product `i_X`

Implemented practical discrete contractions for supported degrees:

- surface: `1\to0`, `2\to1`,
- curve: `1\to0` with tangent-speed representation.

## 15.3 Lie derivative (Cartan)

```math
\mathcal{L}_X\alpha = i_X d\alpha + d(i_X\alpha)
```

implemented for supported degrees on curves and surfaces.

---

## 16. Geodesics and Parallel Transport

## 16.1 Heat-method geodesic distance

For source set `S`:

1. solve heat:

```math
(I+tL)u = \delta_S,
```

2. face vector field:

```math
X = -\nabla u / \|\nabla u\|,
```

3. solve Poisson:

```math
L\phi = \operatorname{div}X,
```

with mean-zero compatibility and small regularization,

4. constant fixing (`\phi(s)=0` at source) and clipping to nonnegative distance.

## 16.2 Parallel transport

Each face has an orthonormal tangent frame `(t_1,t_2)`. Across shared edge, transport is a planar rotation determined by edge-direction coordinates in both frames. Path transport composes these `2\times2` rotations. Holonomy around face cycles is accumulated rotation angle.

---

## 17. PDE Discretizations in the Package

Let `M=\star_0` and `L=-\Delta_\Gamma` (discrete). Unknowns are vertex 0-cochains.

## 17.1 Poisson and Helmholtz

Poisson (closed manifold):

```math
L u = f,
```

with compatibility `\langle f,1\rangle_M=0`, regularized solve `(L+\varepsilon M)u=f`, then zero-mean projection.

Helmholtz:

```math
(L+\alpha M)u=f.
```

## 17.2 Diffusion

Strong form used in time stepping:

```math
\partial_t u + \mu L u = g.
```

Backward Euler:

```math
(I+\Delta t\,\mu L)u^{n+1}=u^n+\Delta t\,g.
```

Crank-Nicolson:

```math
(I+\tfrac12\Delta t\,\mu L)u^{n+1}
=
(I-\tfrac12\Delta t\,\mu L)u^n+\Delta t\,g.
```

## 17.3 Conservative transport

Semidiscrete equation:

```math
M\,\dot u + A(u;v)=0.
```

Edge flux velocity on edge `e=(i,j)`:

```math
v_e = \frac{v_i+v_j}{2}\cdot \hat t_e.
```

Surface transport uses DEC-consistent geometric factor

```math
q_e = w_e\,\ell_e\,v_e
```

(`w_e` cotan weight).

Centered edge flux:

```math
F_e = q_e\frac{u_i+u_j}{2},
```

with matrix contributions `(Au)_i += F_e`, `(Au)_j -= F_e`.

Upwind edge flux:

```math
F_e = q_e\,u_{upwind}
```

(donor-cell by sign of `q_e`).

Explicit stepping uses `M^{-1}A` in FE/SSPRK2/SSPRK3 forms.

CFL estimator (surface):

```math
\Delta t \lesssim cfl\cdot \min_i
\frac{A_i^\star}{\sum_{e\ni i} w_e |v_e|\ell_e}.
```

## 17.4 Advection-diffusion

IMEX scheme:

```math
(I+\Delta t\,\mu L)u^{n+1}
=
 u^n - \Delta t\,M^{-1}A u^n + \Delta t\,g.
```

Fully implicit backward Euler:

```math
(I+\Delta t\,M^{-1}A+\Delta t\,\mu L)u^{n+1}=u^n+\Delta t\,g.
```

## 17.5 Reaction-diffusion

Strong form:

```math
\partial_t u + \mu L u = R(u,x,t).
```

Explicit Euler:

```math
u^{n+1}=u^n+\Delta t\,(R(u^n)-\mu Lu^n).
```

IMEX `\theta`-scheme (reaction explicit, diffusion implicit):

```math
(I+\Delta t\,\theta\mu L)u^{n+1}
=
(I-\Delta t(1-\theta)\mu L)u^n + \Delta t\,R(u^n,t^n).
```

Built-in reactions:

- Fisher-KPP: `\alpha u(1-u)`,
- linear decay: `-\alpha u`,
- bistable: `\alpha u(1-u)(u-1/2)`.

## 17.6 High-resolution limited transport

Limited edge flux (MUSCL-like):

```math
F_e = |q_e|\left(u_{upwind} + \frac12\phi(r_e)(u_{downwind}-u_{upwind})\right),
```

with limiter `\phi` (`minmod`, `van Leer`, `superbee`). Integrated with explicit SSP schemes (`:ssprk2`, `:ssprk3`, or Euler).

---

## 18. Open-Surface Boundary Operators

Boundary edge: edge with exactly one incident face. Boundary vertices are endpoints of boundary edges.

## 18.1 Dirichlet data

Two strong-enforcement methods:

1. row elimination (set row to identity, set RHS value),
2. symmetric elimination (adjust RHS by column contribution, zero row+column, set diagonal 1).

## 18.2 Neumann RHS contribution

For boundary edge `e=(i,j)` and piecewise linear boundary flux values `g_i,g_j`:

```math
b_i \mathrel{+}= g_i\,\ell_e/2,
\qquad
b_j \mathrel{+}= g_j\,\ell_e/2.
```

## 18.3 Boundary mass matrix

Per boundary edge `e=(i,j)`:

```math
M_b^{(e)} = \frac{\ell_e}{6}
\begin{bmatrix}
2&1\\
1&2
\end{bmatrix}
```

assembled into global `(n_V\times n_V)` matrix supported on boundary vertices.

---

## 19. Caching and Low-Allocation Formulation

`SurfacePDECache` / `CurvePDECache` store:

- `L`, `M`, mass diagonal vector,
- optional factorization of `(I+\Delta t\,\theta\mu L)`,
- optional factorization of `(L+\alpha M)`.

This makes repeated solves equivalent mathematically, but computationally cheaper.

Low-allocation in-place kernels implement the same equations while reusing preallocated buffers.

---

## 20. Symbol-to-Code Dictionary

- manifold mesh: `CurveMesh`, `SurfaceMesh`,
- topology: `MeshTopology`,
- geometry: `CurveGeometry`, `SurfaceGeometry`,
- DEC operators: `CurveDEC`, `SurfaceDEC`,
- incidence: `incidence_0`, `incidence_1`,
- stars: `hodge_star_0`, `hodge_star_1`, `hodge_star_2`,
- scalar Laplacian: `build_laplace_beltrami`, `dec.lap0`,
- codifferentials/Hodge Laplacians: `codifferential_1`, `codifferential_2`, `hodge_laplacian_0`, `hodge_laplacian_1`,
- FEEC complex: `Whitney0Space`, `Whitney1Space`, `Whitney2Space`, `WhitneyComplex`, `build_whitney_complex`,
- Whitney basis/reconstruction: `eval_whitney*`, `reconstruct_*`,
- FEEC projections: `interpolate_*`, `Π0`, `Π1`, `Π2`,
- consistent FEEC assembly: `assemble_whitney_mass*`, `assemble_whitney_stiffness0`,
- harmonic/cohomology/Hodge decomposition: `betti_numbers`, `cycle_basis`, `harmonic_basis`, `hodge_decomposition_full`,
- PDE assembly/stepping: `mass_matrix`, `solve_surface_poisson`, `solve_surface_helmholtz`, `step_surface_diffusion_*`, `assemble_transport_operator`, `step_surface_advection_diffusion_*`, `step_surface_reaction_diffusion_*`,
- open-surface BC tools: `detect_boundary_edges`, `apply_dirichlet_*`, `add_neumann_rhs!`, `boundary_mass_matrix`.

---

## 21. Scope and Current Limits

This theory matches the current implementation and therefore intentionally reflects current scope:

- lowest-order FEEC/Whitney only,
- simplicial meshes only,
- static-manifold operators (no remeshing/topology change evolution here),
- boundary-value tooling for open surfaces, but many advanced operators remain closed-surface-focused.

