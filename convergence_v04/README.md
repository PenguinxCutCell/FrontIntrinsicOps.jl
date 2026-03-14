# convergence_v04/README.md
#
# v0.4 Surface PDE Convergence Studies
# =====================================
#
# This folder contains convergence studies for surface PDEs beyond the sphere.
# All scripts print results to the terminal only (no CSV/JSON/plot files).
#
# ## Files
#
# | File | Description |
# |------|-------------|
# | `common.jl`                      | Shared utilities (mesh helpers, table printing) |
# | `reaction_diffusion_torus.jl`    | Reaction–diffusion on the torus |
# | `transport_torus.jl`             | Scalar transport on the torus via toroidal rotation |
# | `deformed_surface_diffusion.jl`  | Helmholtz/diffusion on ellipsoid and perturbed sphere |
# | `deformed_surface_transport.jl`  | Scalar transport on ellipsoid, perturbed sphere, torus |
# | `open_surface_diffusion.jl`      | Poisson/diffusion on flat patch with Dirichlet BCs |
#
# ## Running
#
# From the repository root:
#
# ```bash
# cd convergence_v04
# julia --project=.. reaction_diffusion_torus.jl
# julia --project=.. transport_torus.jl
# julia --project=.. deformed_surface_diffusion.jl
# julia --project=.. deformed_surface_transport.jl
# julia --project=.. open_surface_diffusion.jl
# ```
#
# ## A. Torus studies
#
# ### `reaction_diffusion_torus.jl`
# Solves du/dt = μ ΔΓ u − α u on the torus with initial condition
# u₀ = cos(θ) (toroidal mode). The exact solution decays exponentially.
# Spatial convergence under mesh refinement is reported.
#
# ### `transport_torus.jl`
# Transports the scalar u₀ = cos(θ) by a prescribed toroidal rotation
# velocity field v = ω (−y, x, 0). The exact solution is cos(θ − ω t).
# L² errors are reported under mesh refinement.
#
# ## B. Deformed smooth surfaces
#
# ### `deformed_surface_diffusion.jl`
# Tests the Helmholtz solver (L + αM)u = f and transient backward-Euler
# diffusion on:
# - Ellipsoid (a=2, b=1.5, c=1): generated via `generate_ellipsoid`
# - Perturbed (bumpy) sphere (R=1, ε=0.15, k=2): generated via `generate_perturbed_sphere`
#
# Residuals decrease under refinement, confirming correctness on non-spherical
# geometry.
#
# ### `deformed_surface_transport.jl`
# Runs SSP-RK2 transport with van Leer limiter on ellipsoid, perturbed sphere,
# and torus. Reports finiteness and mass conservation for each refinement level.
#
# ## C. Open surface Poisson
#
# ### `open_surface_diffusion.jl`
# Two Poisson tests on a flat square patch with Dirichlet BCs:
# 1. Laplace with u = x (linear — exact for P1 FEM)
# 2. Poisson with u = sin(πx)sin(πy) (expected O(h²) convergence)
#
# ## Acceptance criteria
#
# - PDE errors decrease under refinement on the torus and at least one
#   deformed surface ✓
# - Vector calculus and transport remain stable on non-spherical geometry ✓
# - All output is terminal-only (no CSV/JSON/plot files generated) ✓
