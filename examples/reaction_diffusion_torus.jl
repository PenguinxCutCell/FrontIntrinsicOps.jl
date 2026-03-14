# examples/reaction_diffusion_torus.jl
#
# Fisher/logistic reaction–diffusion on the torus.
#
# PDE:  du/dt = μ ΔΓ u + α u (1 − u),   u(0) = u₀
#
# We demonstrate the same PDE as reaction_diffusion_sphere.jl but on a
# non-spherical closed surface to confirm the solver works beyond spheres.
#
# Run:  julia --project examples/reaction_diffusion_torus.jl

using FrontIntrinsicOps
using StaticArrays
using LinearAlgebra
using Printf

println("="^60)
println("  Reaction–Diffusion on the Torus (R=2, r=0.5)")
println("="^60)
println()

R  = 2.0
r  = 0.5
nθ = 40
nφ = 16
mesh = generate_torus(R, r, nθ, nφ)
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)
nv   = length(mesh.points)

@printf "  Mesh: %d vertices, %d faces\n" nv length(mesh.faces)
@printf "  Surface area: %.4f  (torus: 4π²Rr = %.4f)\n\n" measure(mesh, geom) 4π^2*R*r

# ─────────────────────────────────────────────────────────────────────────────
# Case 1: Linear decay
# ─────────────────────────────────────────────────────────────────────────────

println("── Case 1: Linear decay (α=1.0, T=0.5) ─────────────────────────────")
α   = 1.0
T   = 0.5
dt  = 0.01
u0  = [cos(atan(p[2], p[1])) .^ 2 for p in mesh.points]  # smooth initial data

u_end, _ = solve_surface_reaction_diffusion(mesh, geom, dec, u0, T, dt, 0.0,
                                             linear_decay_reaction(α); θ=1.0)

M    = mass_matrix(mesh, geom)
area = measure(mesh, geom)
mean0 = sum(M * u0) / area
meanT = sum(M * u_end) / area

@printf "  Initial mean: %.4f  Final mean: %.4f  (decay factor %.4f, exact %.4f)\n\n" \
    mean0 meanT meanT/mean0 exp(-α*T)

# ─────────────────────────────────────────────────────────────────────────────
# Case 2: Fisher–KPP on torus
# ─────────────────────────────────────────────────────────────────────────────

println("── Case 2: Fisher–KPP (α=2.0, μ=0.05, T=2.0) ──────────────────────")
α  = 2.0
μ  = 0.05
T  = 2.0
dt = 0.05
u0 = fill(0.05, nv)

u_end, _ = solve_surface_reaction_diffusion(mesh, geom, dec, u0, T, dt, μ,
                                             fisher_kpp_reaction(α); θ=1.0)

mean0 = sum(M * u0) / area
meanT = sum(M * u_end) / area

@printf "  Initial mean: %.4f  Final mean: %.4f\n" mean0 meanT
@printf "  min/max u(T): %.4f / %.4f\n" minimum(u_end) maximum(u_end)

if meanT > mean0
    println("  ✓ Logistic growth toward carrying capacity observed on torus.")
else
    println("  ! Solution did not grow as expected.")
end
println()

# ─────────────────────────────────────────────────────────────────────────────
# Case 3: Same on ellipsoid (deformed surface)
# ─────────────────────────────────────────────────────────────────────────────

println("── Case 3: Fisher–KPP on ellipsoid (a=2, b=1.5, c=1) ───────────────")
mesh_e = generate_ellipsoid(2.0, 1.5, 1.0, 16, 32)
geom_e = compute_geometry(mesh_e)
dec_e  = build_dec(mesh_e, geom_e)
nv_e   = length(mesh_e.points)
u0_e   = fill(0.05, nv_e)

u_end_e, _ = solve_surface_reaction_diffusion(mesh_e, geom_e, dec_e, u0_e, 2.0, 0.05, 0.05,
                                               fisher_kpp_reaction(2.0); θ=1.0)
M_e    = mass_matrix(mesh_e, geom_e)
area_e = measure(mesh_e, geom_e)
mean0_e = sum(M_e * u0_e) / area_e
meanT_e = sum(M_e * u_end_e) / area_e

@printf "  Ellipsoid area: %.4f  Initial mean: %.4f  Final mean: %.4f\n" area_e mean0_e meanT_e
@printf "  min/max u(T): %.4f / %.4f\n" minimum(u_end_e) maximum(u_end_e)
println()

println("Reaction–diffusion torus example complete.")
