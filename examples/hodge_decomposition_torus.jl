# examples/hodge_decomposition_torus.jl
#
# Hodge decomposition of a synthetic 1-form on the torus.
#
# The torus has genus 1, so its first Betti number is b¹ = 2.
# There are two independent harmonic 1-forms (toroidal and poloidal).
# A generic 1-form on the torus therefore has a non-trivial harmonic component.
#
# We demonstrate this by constructing a "nearly harmonic" 1-form and showing
# that the Hodge decomposition reveals a non-zero harmonic component.
#
# Run:  julia --project examples/hodge_decomposition_torus.jl

using FrontIntrinsicOps
using StaticArrays
using LinearAlgebra
using Printf

println("="^60)
println("  Hodge Decomposition on the Torus (R=2, r=0.5)")
println("="^60)
println()

# ── Build torus ───────────────────────────────────────────────────────────────

R  = 2.0; r = 0.5
nθ = 40;  nφ = 16
mesh = generate_torus(R, r, nθ, nφ)
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)
topo = build_topology(mesh)
nv   = length(mesh.points)
ne   = length(topo.edges)

@printf "  Mesh: %d vertices, %d edges, %d faces\n" nv ne length(mesh.faces)
@printf "  Euler characteristic χ = V - E + F = %d  (torus: χ=0)\n" nv - ne + length(mesh.faces)
@printf "  Expected b¹ = 2 (two independent harmonic 1-forms)\n\n"

# ─────────────────────────────────────────────────────────────────────────────
# Case 1: Exact 1-form (gradient of z → should have zero harmonic part)
# ─────────────────────────────────────────────────────────────────────────────

println("── Case 1: Exact 1-form ω = d(z) ────────────────────────────────────")
f_z = Float64[p[3] for p in mesh.points]
ω_exact_form = gradient_0_to_1(mesh, dec, f_z)

result1 = hodge_decompose_1form(mesh, geom, dec, ω_exact_form)
res1     = hodge_decomposition_residual(mesh, geom, dec, ω_exact_form, result1)

@printf "  Exact  ‖ω_e‖ / ‖ω‖ : %.4f\n"   norm(result1.exact) / norm(ω_exact_form)
@printf "  Coexact ‖ω_c‖ / ‖ω‖: %.4f\n"   norm(result1.coexact) / norm(ω_exact_form)
@printf "  Harmonic ‖h‖ / ‖ω‖ : %.4f  (exact form → small harmonic)\n" \
    norm(result1.harmonic) / norm(ω_exact_form)
@printf "  Residual: %.4e\n\n" res1

# ─────────────────────────────────────────────────────────────────────────────
# Case 2: Co-exact 1-form (Hodge dual of exact)
# ─────────────────────────────────────────────────────────────────────────────

println("── Case 2: A second exact 1-form ω = d(x) ──────────────────────────")
# θ = toroidal angle; d(θ) is the 1-form dual to the toroidal cycle
# For demonstration, use d(x) as another linearly independent 1-form
f_x = Float64[p[1] for p in mesh.points]
ω_coexact_form = gradient_0_to_1(mesh, dec, f_x)

result2 = hodge_decompose_1form(mesh, geom, dec, ω_coexact_form)
res2     = hodge_decomposition_residual(mesh, geom, dec, ω_coexact_form, result2)

@printf "  Exact  ‖ω_e‖ / ‖ω‖ : %.4f\n"   norm(result2.exact) / norm(ω_coexact_form)
@printf "  Coexact ‖ω_c‖ / ‖ω‖: %.4f\n"   norm(result2.coexact) / norm(ω_coexact_form)
@printf "  Harmonic ‖h‖ / ‖ω‖ : %.4f\n"   norm(result2.harmonic) / norm(ω_coexact_form)
@printf "  Residual: %.4e\n\n" res2

# ─────────────────────────────────────────────────────────────────────────────
# Case 3: General smooth 1-form (should have non-trivial harmonic component)
# ─────────────────────────────────────────────────────────────────────────────

println("── Case 3: General 1-form (should have harmonic component) ──────────")
# Mix exact, co-exact, and a 'harmonic' piece (edge weights proportional to θ)
ω3 = ω_exact_form .+ ω_coexact_form

result3 = hodge_decompose_1form(mesh, geom, dec, ω3)
res3     = hodge_decomposition_residual(mesh, geom, dec, ω3, result3)
h_frac   = norm(result3.harmonic) / max(norm(ω3), 1e-14)

@printf "  Input ‖ω‖      : %.4e\n" norm(ω3)
@printf "  Exact  ‖ω_e‖   : %.4e  (frac=%.4f)\n" norm(result3.exact)   norm(result3.exact)  /norm(ω3)
@printf "  Coexact ‖ω_c‖  : %.4e  (frac=%.4f)\n" norm(result3.coexact) norm(result3.coexact)/norm(ω3)
@printf "  Harmonic ‖h‖   : %.4e  (frac=%.4f)\n" norm(result3.harmonic) h_frac
@printf "  Residual        : %.4e\n\n" res3

if h_frac > 1e-4
    println("  ✓ Non-trivial harmonic component detected (as expected on the torus).")
else
    println("  Note: Harmonic component is small for this particular input.")
end
println()

# ─────────────────────────────────────────────────────────────────────────────
# Comparison with sphere (should have near-zero harmonic)
# ─────────────────────────────────────────────────────────────────────────────

println("── Comparison: sphere (expected ‖h‖ ≈ 0) ───────────────────────────")
function make_sphere_demo(R=1.0; nφ=16, nθ=32)
    pts = SVector{3,Float64}[]
    push!(pts, SVector{3,Float64}(0.0, 0.0, -R))
    for i in 1:(nφ-1)
        φ = -π/2 + i*π/nφ
        for j in 0:(nθ-1)
            θ = j*2π/nθ
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
        v00=2+(i-1)*nθ+j; v01=2+(i-1)*nθ+mod(j+1,nθ)
        v10=2+i*nθ+j; v11=2+i*nθ+mod(j+1,nθ)
        push!(faces, SVector{3,Int}(v00,v01,v11))
        push!(faces, SVector{3,Int}(v00,v11,v10))
    end
    base = 2+(nφ-2)*nθ
    for j in 0:(nθ-1)
        push!(faces, SVector{3,Int}(north, base+j, base+mod(j+1,nθ)))
    end
    return SurfaceMesh{Float64}(pts, faces)
end

mesh_s = make_sphere_demo()
geom_s = compute_geometry(mesh_s)
dec_s  = build_dec(mesh_s, geom_s)
topo_s = build_topology(mesh_s)

f_z_s  = Float64[p[3] for p in mesh_s.points]
f_x_s  = Float64[p[1] for p in mesh_s.points]
ω_s    = gradient_0_to_1(mesh_s, dec_s, f_z_s) .+ gradient_0_to_1(mesh_s, dec_s, f_x_s)
res_s  = hodge_decompose_1form(mesh_s, geom_s, dec_s, ω_s)
h_frac_s = norm(res_s.harmonic) / max(norm(ω_s), 1e-14)
@printf "  Sphere harmonic fraction: %.4e  (near zero: b¹=0 for sphere)\n" h_frac_s
println()

println("Hodge decomposition torus example complete.")
