# FrontIntrinsicOps.jl v0.2 — Convergence Scripts

This folder contains reproducible convergence studies for `FrontIntrinsicOps.jl v0.2`.

All scripts print results to the terminal and write nothing to disk.

## Running

From the repository root:

```bash
# Run a single study
julia --project=. convergence/circle_convergence.jl
julia --project=. convergence/sphere_convergence.jl
julia --project=. convergence/torus_convergence.jl
julia --project=. convergence/poisson_sphere_convergence.jl

# Run all studies
julia --project=. convergence/run_all.jl
```

## Files

| File | Description |
|------|-------------|
| `common.jl` | Package activation, header printing, formatting utilities |
| `metrics.jl` | Error norms: `weighted_L1`, `weighted_L2`, `Linf`, `curvature_error` |
| `fit.jl` | `pairwise_orders`, `fitted_order`, `format_order` |
| `helpers_generators.jl` | Mesh ladder helpers wrapping package generators |
| `circle_convergence.jl` | Convergence on a polygonal circle |
| `sphere_convergence.jl` | Convergence on UV-sphere and icosphere families |
| `torus_convergence.jl` | Convergence on a torus |
| `poisson_sphere_convergence.jl` | Poisson equation on sphere, DEC vs cotan |
| `run_all.jl` | Run all studies sequentially |

## Scripts

### `circle_convergence.jl`

Study a closed polygonal circle of radius `R = 1.5`.

Refinement ladder: `N = 16, 32, 64, 128, 256, 512, 1024`.

Mesh size: `h = 2πR/N` (arc length per segment).

Quantities:
- Length error vs `2πR` — expected order 2
- Enclosed area error vs `πR²` — expected order 2
- Mean curvature error vs `1/R` — expected order 2
- `max|L·ones|` — nullspace residual (should be near machine precision)

### `sphere_convergence.jl`

Study two sphere mesh families: UV-sphere and icosphere.

Mesh size: `h = sqrt(area/NF)`.

For each family, two dual-area methods are compared (`:barycentric`, `:mixed`).

Quantities:
- Area error vs `4πR²`
- Volume error vs `(4/3)πR³`
- Mean curvature error vs `H = 1/R`
- Gaussian curvature L2 error vs `K = 1/R²`
- Euler characteristic `χ` (must equal 2)
- Gauss–Bonnet residual `|∫K dA − 2πχ|` (near machine precision)
- `‖L_dec − L_cotan‖_∞` (method comparison)

Exact solution: `K = 1/R²`, `H = 1/R` for the standard unit sphere.

### `torus_convergence.jl`

Study a torus with major radius `R = 3`, minor radius `r = 1`.

Exact formulas:
- Area: `A = 4π²Rr`
- Volume: `V = 2π²Rr²`
- Gaussian curvature: `K(θ) = cos(θ) / (r(R + r cos θ))`
- `χ = 0`, so `∫K dA = 0` (Gauss–Bonnet)

Quantities:
- Area and volume errors
- `∫K dA` (should stay near 0)
- Euler characteristic (must equal 0)
- Gauss–Bonnet residual
- `‖L_dec − L_cotan‖_∞`

### `poisson_sphere_convergence.jl`

Solve `L u = f` on the sphere using a manufactured solution.

Exact solution: `u = x` (x-coordinate function).
Since `L x = (2/R²) x` on a sphere, the right-hand side is `f = (2/R²) x`.

The linear system is made uniquely solvable by pinning `u[1] = u_exact[1]`.

Both `method=:dec` and `method=:cotan` are compared on icosphere and UV-sphere ladders.

Quantities:
- Compatibility residual: `|∫f dA|` (should be near machine precision)
- Relative solution error: `‖u_h − u_exact‖₂ / ‖u_exact‖₂`
- Linear residual: `max|L u_h − f|`
- Pairwise and fitted orders

## Observed orders of convergence

Pairwise OOC between consecutive refinement levels:
```
p_i = log(e_i / e_{i+1}) / log(h_i / h_{i+1})
```

Global fitted order by least-squares regression of `log(e)` on `log(h)`:
```
p = cov(log h, log e) / var(log h)
```

## Gauss–Bonnet convention

This package uses the standard Gauss–Bonnet theorem:
```
∫_Γ K dA = 2π χ
```
- Sphere (χ = 2): `∫K dA = 4π`
- Torus  (χ = 0): `∫K dA = 0`
