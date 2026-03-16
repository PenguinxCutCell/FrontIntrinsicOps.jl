# examples/open_surface_poisson.jl
#
# Poisson/diffusion on an open surface (disk / flat patch) with Dirichlet BCs.
#
# Problems demonstrated:
#   1. Laplace equation on a flat square with linear Dirichlet data (exact: u=x)
#   2. Poisson on a flat square: −ΔΓ u = f,  f = 2π² sin(πx)sin(πy)
#   3. Transient diffusion on an open surface with fixed boundary
#
# Run:  julia --project examples/open_surface_poisson.jl

using FrontIntrinsicOps
using StaticArrays
using LinearAlgebra
using Printf

println("="^60)
println("  Poisson and Diffusion on an Open Surface")
println("="^60)
println()

# ── Build flat square patch ────────────────────────────────────────────────────

function make_flat_patch_demo(N::Int, L::Float64=1.0)
    pts   = SVector{3,Float64}[]
    faces = SVector{3,Int}[]
    h     = L / N
    for j in 0:N, i in 0:N
        push!(pts, SVector{3,Float64}(i*h, j*h, 0.0))
    end
    idx_v(i, j) = j*(N+1) + i + 1
    for j in 0:(N-1), i in 0:(N-1)
        v00 = idx_v(i,   j  )
        v10 = idx_v(i+1, j  )
        v01 = idx_v(i,   j+1)
        v11 = idx_v(i+1, j+1)
        push!(faces, SVector{3,Int}(v00, v10, v11))
        push!(faces, SVector{3,Int}(v00, v11, v01))
    end
    return SurfaceMesh{Float64}(pts, faces)
end

# ─────────────────────────────────────────────────────────────────────────────
# 1. Laplace equation: u = x on boundary → u = x everywhere
# ─────────────────────────────────────────────────────────────────────────────

println("── 1. Laplace equation with u=x Dirichlet BC ────────────────────────")
N    = 16
mesh = make_flat_patch_demo(N)
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)
topo = build_topology(mesh)
nv   = length(mesh.points)

@printf "  Patch: [0,1]², N=%d,  nv=%d vertices\n" N nv

bv   = detect_boundary_vertices(mesh, topo)
@printf "  Boundary: %d vertices detected\n" length(bv)

g    = Float64[mesh.points[i][1] for i in bv]    # u = x on boundary
f    = zeros(Float64, nv)
u    = solve_open_surface_poisson(mesh, geom, dec, topo, f, bv, g)

u_exact = Float64[mesh.points[i][1] for i in 1:nv]
err     = maximum(abs.(u .- u_exact))
@printf "  Max error (vs. exact u=x): %.4e  (exact for P1 FEM)\n\n" err

# ─────────────────────────────────────────────────────────────────────────────
# 2. Poisson equation: −ΔΓ u = 2π² sin(πx)sin(πy), u=0 on ∂Ω
# ─────────────────────────────────────────────────────────────────────────────

println("── 2. Poisson: −ΔΓu = 2π² sin(πx)sin(πy), u=0 on ∂Ω ──────────────")
g2  = zeros(Float64, length(bv))    # u = 0 on boundary
f2  = Float64[2π^2 * sin(π * mesh.points[i][1]) * sin(π * mesh.points[i][2])
              for i in 1:nv]
u2  = solve_open_surface_poisson(mesh, geom, dec, topo, f2, bv, g2)

u2_exact = Float64[sin(π * mesh.points[i][1]) * sin(π * mesh.points[i][2])
                   for i in 1:nv]
err2     = weighted_l2_error(mesh, geom, u2, u2_exact)
@printf "  L² error (vs. exact sin(πx)sin(πy)): %.4e  (N=%d, expect O(h²))\n\n" err2 N

# ─────────────────────────────────────────────────────────────────────────────
# 3. Neumann boundary condition
# ─────────────────────────────────────────────────────────────────────────────

println("── 3. Poisson with Neumann BC ────────────────────────────────────────")
# Solve −ΔΓ u + u = f with homogeneous Neumann (natural BC)
# No Dirichlet boundary, no special treatment needed for Neumann in weak form
L3 = dec.lap0
M3 = mass_matrix(mesh, geom)
A3 = L3 + M3   # Helmholtz operator (unique solution even without Dirichlet BC)
f3 = Float64[sin(π * mesh.points[i][1]) * sin(π * mesh.points[i][2]) for i in 1:nv]
u3 = Array(A3) \ f3
@printf "  Helmholtz (L+M)u = f:  max|u| = %.4e, all finite: %s\n\n" maximum(abs.(u3)) string(all(isfinite.(u3)))

# ─────────────────────────────────────────────────────────────────────────────
# 4. Transient diffusion with fixed boundary
# ─────────────────────────────────────────────────────────────────────────────

println("── 4. Transient diffusion with Dirichlet BC ─────────────────────────")
# IC: random interior field, boundary fixed to zero.  Should diffuse to zero.
μ     = 0.1
T     = 0.5
dt    = 0.01
nstep = round(Int, T / dt)

u0  = zeros(Float64, nv)
# Set non-zero interior values
for i in 1:nv
    x, y = mesh.points[i][1], mesh.points[i][2]
    u0[i] = sin(π*x) * sin(π*y)  # zero on boundary, non-zero inside
end

@printf "  IC: max|u₀| = %.4f,  boundary at zero\n" maximum(abs.(u0))
@printf "  Diffusing: μ=%.2f, T=%.1f, dt=%.3f, nstep=%d\n" μ T dt nstep

u   = copy(u0)
fac = nothing
for _ in 1:nstep
    # One backward Euler step
    u, fac = step_surface_diffusion_backward_euler(mesh, geom, dec, u, dt, μ;
                                                    factorization=fac)
    # Re-enforce Dirichlet BC
    apply_dirichlet!(u, bv, 0.0)
end

@printf "  Final: max|u(T)| = %.4e  (decays toward zero as expected)\n\n" maximum(abs.(u))

println("Open surface Poisson example complete.")
