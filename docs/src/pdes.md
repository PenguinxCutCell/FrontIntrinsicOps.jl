# API: PDE Solvers

## Surface diffusion, Poisson, Helmholtz

### Poisson

```julia
solve_surface_poisson(mesh, geom, dec, f;
                       gauge = :zero_mean,   # :zero_mean or :pin
                       reg   = 1e-10) → Vector{T}
```

Solve $L u = f$ on a closed surface.  Enforces zero-mean solution.

### Helmholtz

```julia
solve_surface_helmholtz(mesh, geom, dec, f, α) → Vector{T}
```

Solve $(L + \alpha M) u = f$.  Requires $\alpha > 0$.

### Diffusion — backward Euler

```julia
step_surface_diffusion_backward_euler(mesh, geom, dec, u, dt, μ;
                                       rhs           = nothing,
                                       factorization = nothing) → (u_new, fac)
```

One backward-Euler step: $(M + dt\,\mu\,L) u^{n+1} = M u^n + dt\, g$.
Pass `factorization=fac` on subsequent steps to reuse the LU factorization.

### Diffusion — Crank–Nicolson

```julia
step_surface_diffusion_crank_nicolson(mesh, geom, dec, u, dt, μ;
                                       rhs           = nothing,
                                       factorization = nothing) → (u_new, fac)
```

Second-order accurate.

---

## Transport

### Operator assembly

```julia
assemble_transport_operator(mesh, geom, vel;
                              scheme = :upwind) → SparseMatrixCSC
# scheme: :upwind (1st order) or :centered (2nd order)
```

### CFL estimate

```julia
estimate_transport_dt(mesh, geom, vel; cfl=0.5) → T
```

### Time steps

```julia
step_surface_transport_forward_euler(mesh, geom, A, u, dt) → Vector{T}
step_surface_transport_ssprk2(mesh, geom, A, u, dt)       → Vector{T}
step_surface_transport_ssprk3(mesh, geom, A, u, dt)       → Vector{T}
```

---

## High-resolution transport (v0.4)

### Limiters

```julia
minmod(a, b) → T               # two-argument minmod
minmod3(a, b, c) → T           # three-argument minmod
vanleer_limiter(r) → T         # van Leer φ(r)
superbee_limiter(r) → T        # superbee φ(r)
```

### Limited operator (solution-dependent)

```julia
assemble_transport_operator_limited(mesh, geom, topo, vel, u;
                                    limiter = :minmod) → SparseMatrixCSC
# limiter: :minmod, :van_leer, or :superbee
```

### Limited transport step

```julia
step_surface_transport_limited(mesh, geom, dec, topo, u, vel, dt;
                                limiter = :minmod,
                                method  = :ssprk2) → Vector{T}
# method: :ssprk2 or :euler
```

---

## Advection–diffusion IMEX

```julia
step_surface_advection_diffusion_imex(mesh, geom, dec, u, vel, dt, μ;
                                       scheme             = :upwind,
                                       transport_operator = nothing,  # pass to reuse
                                       factorization      = nothing,  # pass to reuse
                                       rhs                = nothing) → (u_new, fac)

step_surface_advection_diffusion_backward_euler(mesh, geom, dec, u, vel, dt, μ) → Vector{T}
```

---

## Reaction–diffusion IMEX (v0.4)

### Built-in reactions

```julia
fisher_kpp_reaction(α) → Function       # f(u) = α u(1-u)
linear_decay_reaction(α) → Function     # f(u) = -α u
bistable_reaction(α) → Function         # f(u) = α u(1-u)(u-0.5)
```

### Evaluate reaction

```julia
evaluate_reaction!(r, reaction, u, mesh, geom, t)
```

Dispatch to in-place, pointwise, or no-op reaction.

### IMEX step

```julia
step_surface_reaction_diffusion_imex(mesh, geom, dec, u, dt, μ, reaction, t;
                                      θ             = 1.0,
                                      factorization = nothing) → (u_new, fac)

step_surface_reaction_diffusion_explicit(mesh, geom, dec, u, dt, μ, reaction, t)
  → Vector{T}   # unstable reference
```

### Full integration

```julia
solve_surface_reaction_diffusion(mesh, geom, dec, u0, T_end, dt, μ, reaction;
                                  θ      = 1.0,
                                  scheme = :imex) → (u_final, t_final)
```

---

## Tangential vector calculus (v0.4)

```julia
tangential_project(v, n) → SVector{3,T}
tangential_project_field(mesh, geom, vfield; location=:vertex) → Vector{SVector{3,T}}

gradient_0_to_tangent_vectors(mesh, geom, u; location=:face)  → Vector{SVector{3,T}}
divergence_tangent_vectors(mesh, geom, vfield; location=:face) → Vector{T}

tangent_vectors_to_1form(mesh, geom, topo, vfield; location=:face) → Vector{T}
oneform_to_tangent_vectors(mesh, geom, topo, α; location=:face)   → Vector{SVector{3,T}}

surface_rot_0form(mesh, geom, u) → Vector{SVector{3,T}}  # n̂ × ∇Γu
```

---

## Hodge decomposition (v0.4)

```julia
hodge_decompose_1form(mesh, geom, dec, α; reg=1e-10) → NamedTuple
# Returns: (exact, coexact, harmonic, phi, psi)

exact_component_1form(mesh, geom, dec, α; reg=1e-10)   → (α_exact, φ)
coexact_component_1form(mesh, geom, dec, α; reg=1e-10) → (α_coexact, ψ)
harmonic_component_1form(mesh, geom, dec, α; reg=1e-10) → Vector{T}

hodge_decomposition_residual(mesh, geom, dec, α, decomp) → T
hodge_inner_products(mesh, geom, dec, decomp) → NamedTuple
```

---

## Open surfaces (v0.4)

```julia
detect_boundary_edges(topo)          → Vector{Int}
detect_boundary_vertices(mesh, topo) → Vector{Int}
is_open_surface(topo)                → Bool

apply_dirichlet!(u, bnd_verts, values)
apply_dirichlet_to_system!(A, b, bnd_verts, values)
apply_dirichlet_symmetric!(A, b, bnd_verts, values)
add_neumann_rhs!(b, mesh, geom, topo, g)

boundary_mass_matrix(mesh, geom, topo) → SparseMatrixCSC
solve_open_surface_poisson(mesh, geom, dec, topo, f, bnd_verts, bnd_vals) → Vector{T}
```

---

## Cache (v0.4)

```julia
build_pde_cache(mesh, geom, dec;
                μ             = nothing,
                dt            = nothing,
                θ             = 1.0,
                α_helmholtz   = nothing) → SurfacePDECache

update_pde_cache(cache, mesh, geom, dec; kwargs...) → SurfacePDECache

step_diffusion_cached(cache, u)         → Vector{T}
step_diffusion_cached!(u, cache)        # in-place
step_diffusion_cached!(u, cache, R, t)  # with reaction
solve_helmholtz_cached(cache, f)        → Vector{T}
```

---

## Performance (v0.4)

```julia
alloc_diffusion_buffers(nv, T=Float64) → SurfaceDiffusionBuffers
alloc_rd_buffers(nv, T=Float64)        → SurfaceRDBuffers

step_diffusion_inplace!(u, cache, buf)
step_rd_inplace!(u, cache, buf, mesh, geom, R, t)
apply_mass_inplace!(y, cache, x)
apply_laplace_inplace!(y, cache, x)

l2_norm_cached(cache, u)     → T    # √(u⊤ M u)
energy_norm_cached(cache, u) → T    # √(u⊤ L u)
```

---

## Shared PDE utilities

```julia
mass_matrix(mesh, geom)         → SparseMatrixCSC   # M = star0
lumped_mass_vector(mesh, geom)  → Vector{T}
apply_mass!(y, mesh, geom, x)   # y ← M x

weighted_mean(mesh, geom, u) → T
zero_mean_projection!(u, mesh, geom)
enforce_compatibility!(f, mesh, geom)
```

---

## See also

- [Surface diffusion (math)](07_surface_diffusion.md)
- [Scalar transport (math)](08_transport.md)
- [Reaction–diffusion (math)](10_reaction_diffusion.md)
- [Hodge decomposition (math)](12_hodge_decomposition.md)
- [Open surfaces (math)](14_open_surfaces.md)
- [Caching (math)](15_caching.md)

---

## Using New Intrinsic Tools In PDE Loops

Recent additions (`geodesic_distance`, `lie_derivative`, transport helpers) can
be combined with existing solvers. A minimal pattern is:

```julia
mesh = generate_icosphere(1.0, 2)
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)

src = argmax([p[3] for p in mesh.points])
d = geodesic_distance_to_vertex(mesh, geom, dec, src)
u = exp.(-(d ./ 0.45).^2)

# face tangent velocity field
X = [tangential_project(SVector(-c[2], c[1], 0.0), geom.face_normals[fi]) for (fi, c) in enumerate([sum(mesh.points[f[k]] for k in 1:3)/3 for f in mesh.faces])]

dt = 5e-3
μ = 5e-3
for _ in 1:80
    u .-= dt .* lie_derivative(X, u, mesh, geom, dec)
    u, _ = step_surface_diffusion_backward_euler(mesh, geom, dec, u, dt, μ)
end
```

Full runnable script:

- `examples/surface_pde_intrinsic_tools.jl`
