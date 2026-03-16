# examples/reaction_diffusion_sphere.jl
#
# Fisher/logistic reaction–diffusion on the unit sphere.
#
# PDE:  du/dt = μ ΔΓ u + α u (1 − u),   u(0) = u₀  (small initial seed)
#
# This is the Fisher–KPP equation on the sphere.  Starting from a small
# localized seed, the solution evolves toward a travelling wave that
# ultimately fills the sphere.
#
# We also demonstrate linear decay (u → 0) as a simpler reference case.
#
# Run:  julia --project examples/reaction_diffusion_sphere.jl

using FrontIntrinsicOps
using StaticArrays
using LinearAlgebra
using Printf

# ── Build sphere ──────────────────────────────────────────────────────────────

function make_uvsphere(R=1.0; nφ=16, nθ=32)
    pts = SVector{3,Float64}[]
    push!(pts, SVector{3,Float64}(0.0, 0.0, -R))
    for i in 1:(nφ-1)
        φ = -π/2 + i * π / nφ
        for j in 0:(nθ-1)
            θ = j * 2π / nθ
            push!(pts, SVector{3,Float64}(R*cos(φ)*cos(θ), R*cos(φ)*sin(θ), R*sin(φ)))
        end
    end
    push!(pts, SVector{3,Float64}(0.0, 0.0, R))
    faces = SVector{3,Int}[]
    south = 1; north = length(pts)
    for j in 0:(nθ-1)
        push!(faces, SVector{3,Int}(south, 2+mod(j+1,nθ), 2+j))
    end
    for i in 1:(nφ-2), j in 0:(nθ-1)
        v00 = 2+(i-1)*nθ+j; v01 = 2+(i-1)*nθ+mod(j+1,nθ)
        v10 = 2+i*nθ+j;     v11 = 2+i*nθ+mod(j+1,nθ)
        push!(faces, SVector{3,Int}(v00,v01,v11))
        push!(faces, SVector{3,Int}(v00,v11,v10))
    end
    base = 2+(nφ-2)*nθ
    for j in 0:(nθ-1)
        push!(faces, SVector{3,Int}(north, base+j, base+mod(j+1,nθ)))
    end
    return SurfaceMesh{Float64}(pts, faces)
end

println("="^60)
println("  Fisher–KPP Reaction–Diffusion on the Unit Sphere")
println("="^60)
println()

mesh = make_uvsphere(1.0; nφ=20, nθ=40)
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)
nv   = length(mesh.points)

@printf "  Mesh: %d vertices, %d faces\n\n" nv length(mesh.faces)

# ─────────────────────────────────────────────────────────────────────────────
# Case 1: Linear decay  du/dt = −α u  (exact: u(t) = u₀ exp(−αt))
# ─────────────────────────────────────────────────────────────────────────────

println("── Case 1: Linear decay (du/dt = −α u) ──────────────────────────")
α   = 1.0
T   = 1.0
dt  = 0.02
u0  = ones(Float64, nv)

u_end, _ = solve_surface_reaction_diffusion(mesh, geom, dec, u0, T, dt, 0.0,
                                             linear_decay_reaction(α); θ=1.0)
u_exact   = u0 .* exp(-α * T)
err       = maximum(abs.(u_end .- u_exact))
@printf "  α=%.2f,  T=%.1f:  max error vs exp(−αT) = %.4e\n" α T err
@printf "  Mean u(T) = %.6f  (exact: %.6f)\n\n" mean_surface(mesh, geom, u_end) exp(-α*T)

# ─────────────────────────────────────────────────────────────────────────────
# Case 2: Fisher–KPP  du/dt = μ ΔΓ u + α u(1−u)
# ─────────────────────────────────────────────────────────────────────────────

println("── Case 2: Fisher–KPP (logistic growth + diffusion) ────────────────")
α  = 2.0
μ  = 0.05
T  = 3.0
dt = 0.05
u0 = fill(0.05, nv)   # small initial seed

u_end, _ = solve_surface_reaction_diffusion(mesh, geom, dec, u0, T, dt, μ,
                                             fisher_kpp_reaction(α); θ=1.0)

@printf "  α=%.2f, μ=%.3f, T=%.1f, dt=%.3f\n" α μ T dt
@printf "  Initial mean: %.4f  Final mean: %.4f\n" mean_surface(mesh, geom, u0) mean_surface(mesh, geom, u_end)
@printf "  min/max u(T): %.4f / %.4f\n\n" minimum(u_end) maximum(u_end)

if mean_surface(mesh, geom, u_end) > mean_surface(mesh, geom, u0)
    println("  ✓ Solution grew from initial seed toward carrying capacity.")
else
    println("  ! Solution did not grow as expected.")
end
println()

# ─────────────────────────────────────────────────────────────────────────────
# Case 3: Bistable reaction  du/dt = μ ΔΓ u + u(1−u)(u−a)
# ─────────────────────────────────────────────────────────────────────────────

println("── Case 3: Bistable reaction du/dt = μ ΔΓ u + u(1−u)(u−a) ────────")
a_bist = 0.3
μ      = 0.05
T      = 2.0
dt     = 0.05
u0     = fill(0.5, nv)   # start near unstable equilibrium

u_end, _ = solve_surface_reaction_diffusion(mesh, geom, dec, u0, T, dt, μ,
                                             bistable_reaction(a_bist); θ=1.0)
@printf "  a=%.2f, μ=%.3f, T=%.1f\n" a_bist μ T
@printf "  min/max u(T): %.4f / %.4f\n" minimum(u_end) maximum(u_end)
println()

# ─────────────────────────────────────────────────────────────────────────────
# Helper: mean over sphere
# ─────────────────────────────────────────────────────────────────────────────

function mean_surface(mesh, geom, u)
    M    = mass_matrix(mesh, geom)
    area = measure(mesh, geom)
    return sum(M * u) / area
end

println("Reaction–diffusion sphere example complete.")
