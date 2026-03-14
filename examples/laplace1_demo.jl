# examples/laplace1_demo.jl
#
# Demonstration of 1-form (edge-based) DEC operators on the unit sphere.
#
# Operators demonstrated:
#   - d0   : exterior derivative 0→1  (gradient on vertices → edges)
#   - d1   : exterior derivative 1→2  (curl-like: edges → faces)
#   - δ1   : codifferential 1→0       (divergence on edges → vertices)
#   - δ2   : codifferential 2→1       (adjoint of d1)
#   - Δ₀   : Hodge Laplacian on 0-forms  (== dec.lap0)
#   - Δ₁   : Hodge Laplacian on 1-forms  (d0 δ1 + δ2 d1)
#
# Key identity checked: d1 * d0 == 0  (discrete d² = 0)
#
# Run:  julia --project examples/laplace1_demo.jl

using FrontIntrinsicOps
using StaticArrays
using LinearAlgebra
using SparseArrays
using Printf

# ── Mesh & operators ──────────────────────────────────────────────────────────

R    = 1.0
mesh = generate_icosphere(R, 3)
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)
topo = build_topology(mesh)

nv = length(mesh.points)
ne = length(topo.edges)
nf = length(mesh.faces)

println("="^60)
println("  DEC 1-form operators on icosphere level 3")
println("="^60)
@printf "Vertices (0-forms): %d\n" nv
@printf "Edges    (1-forms): %d\n" ne
@printf "Faces    (2-forms): %d\n" nf
println()

# ── Retrieve operators ────────────────────────────────────────────────────────

d0 = dec.d0            # exterior derivative 0→1,  size (ne × nv)
d1 = dec.d1            # exterior derivative 1→2,  size (nf × ne)

δ1 = codifferential_1(mesh, geom, dec)   # 1→0,  size (nv × ne)
δ2 = codifferential_2(mesh, geom, dec)   # 2→1,  size (ne × nf)

Δ₀ = hodge_laplacian_0(mesh, geom, dec)  # 0-form Laplacian (nv × nv)
Δ₁ = hodge_laplacian_1(mesh, geom, dec)  # 1-form Laplacian (ne × ne)

println("─── Operator sizes ──────────────────────────────────────────────")
@printf "d0   (∇: 0→1):   %d × %d\n" size(d0,1) size(d0,2)
@printf "d1   (d: 1→2):   %d × %d\n" size(d1,1) size(d1,2)
@printf "δ1   (δ: 1→0):   %d × %d\n" size(δ1,1) size(δ1,2)
@printf "δ2   (δ: 2→1):   %d × %d\n" size(δ2,1) size(δ2,2)
@printf "Δ₀   (0-Hodge):  %d × %d\n" size(Δ₀,1) size(Δ₀,2)
@printf "Δ₁   (1-Hodge):  %d × %d\n" size(Δ₁,1) size(Δ₁,2)
println()

# ── Key identity: d² = 0 ─────────────────────────────────────────────────────

d1_d0     = d1 * d0
max_d1_d0 = maximum(abs, d1_d0)
println("─── Exactness check: d1 * d0 == 0 ──────────────────────────────")
@printf "max |d1 * d0| = %.4e  (should be 0 exactly)\n" max_d1_d0
println()

# ── Gradient, divergence, and curl on a test function ────────────────────────

z = Float64[p[3] for p in mesh.points]   # 0-form: z-coordinate

α = gradient_0_to_1(mesh, dec, z)        # 1-form: gradient of z
@printf "─── gradient_0_to_1(z) ─────────────────────────────────────────\n"
@printf "‖∇z‖₂ = %.6f  (1-form edge vector, %d entries)\n" norm(α) ne

div_α = divergence_1_to_0(mesh, geom, dec, α)   # 0-form: δ(∇z) = Δ₀ z
@printf "‖δ(∇z)‖₂ = %.6f  (should ≈ ‖Δ₀ z‖₂ = %.6f)\n" norm(div_α) norm(Δ₀ * z)
@printf "max |δ(∇z) - Δ₀ z| = %.4e\n" maximum(abs, div_α .- Δ₀ * z)
println()

curl_α = curl_like_1_to_2(mesh, dec, α)   # 2-form: d1(d0 z) = 0 (exact form)
@printf "─── curl_like_1_to_2(∇z) ───────────────────────────────────────\n"
@printf "‖d1(∇z)‖₂ = %.4e  (should be 0: ∇z is exact, d(df)=0)\n" norm(curl_α)
println()

# ── Hodge Laplacian on 0-forms vs dec.lap0 ───────────────────────────────────

diff_Δ₀_vs_lap0 = maximum(abs, Δ₀ - dec.lap0)
println("─── Hodge Laplacian consistency ─────────────────────────────────")
@printf "max |Δ₀ - dec.lap0| = %.4e  (should be 0)\n" diff_Δ₀_vs_lap0
println()

# ── Eigenvalue test: L z = (2/R²) z on sphere ────────────────────────────────

λ₁_expected = 2.0 / R^2
Lz = dec.lap0 * z
# Remove constant component (projected out)
Lz_proj = Lz .- dot(Lz, ones(nv)) / nv
z_proj  = z  .- dot(z,  ones(nv)) / nv
ratio   = Lz_proj' * z_proj / (z_proj' * z_proj)

println("─── Eigenvalue test: L z = (2/R²) z ────────────────────────────")
@printf "Expected λ₁ = 2/R² = %.6f\n"              λ₁_expected
@printf "Numerical:   ‖Lz‖/‖z‖ Rayleigh = %.6f\n" ratio
@printf "Relative error in eigenvalue: %.4e\n" abs(ratio - λ₁_expected) / λ₁_expected
println()

# ── 1-form Hodge Laplacian applied to the gradient ───────────────────────────

println("─── Δ₁ applied to ∇z ───────────────────────────────────────────")
Δ₁α = Δ₁ * α
@printf "‖Δ₁ ∇z‖₂ = %.6f\n" norm(Δ₁α)
@printf "‖Δ₁ ∇z - λ₁ ∇z‖₂ = %.4e  (∇z should be 1-eigenform)\n" norm(Δ₁α .- λ₁_expected .* α)
println()

println("1-form DEC operators demo complete.")
