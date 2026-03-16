# examples/hodge_decomposition_sphere.jl
#
# Hodge decomposition of a synthetic 1-form on the unit sphere.
#
# Every smooth 1-form ω on the sphere can be uniquely decomposed as:
#   ω = dα + δβ + h
# where:
#   - dα  is the exact component (gradient of a 0-form α)
#   - δβ  is the co-exact component (co-differential of a 2-form β)
#   - h   is the harmonic component
#
# On the sphere (genus 0), the harmonic 1-forms are trivial (b¹ = 0),
# so h should be near zero for any input.
#
# Run:  julia --project examples/hodge_decomposition_sphere.jl

using FrontIntrinsicOps
using StaticArrays
using LinearAlgebra
using Printf

println("="^60)
println("  Hodge Decomposition on the Unit Sphere")
println("="^60)
println()

# ── Build sphere ──────────────────────────────────────────────────────────────

function make_sphere(R=1.0; nφ=20, nθ=40)
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

mesh = make_sphere(1.0; nφ=16, nθ=32)
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)
topo = build_topology(mesh)
nv   = length(mesh.points)
ne   = length(topo.edges)

@printf "  Mesh: %d vertices, %d edges, %d faces\n\n" nv ne length(mesh.faces)

# ─────────────────────────────────────────────────────────────────────────────
# Case 1: Exact form ω = d(f), f = z
# ─────────────────────────────────────────────────────────────────────────────

println("── Case 1: Exact 1-form ω = d(z) ────────────────────────────────────")
f_z = Float64[p[3] for p in mesh.points]
ω1  = gradient_0_to_1(mesh, dec, f_z)   # ω = d(z) is exact

result1 = hodge_decompose_1form(mesh, geom, dec, ω1)
res1     = hodge_decomposition_residual(mesh, geom, dec, ω1, result1)

@printf "  Input norm  : %.4e\n"   sqrt(sum(abs2, ω1))
@printf "  Exact  norm : %.4e\n"   sqrt(sum(abs2, result1.exact))
@printf "  Coexact norm: %.4e\n"   sqrt(sum(abs2, result1.coexact))
@printf "  Harmonic norm: %.4e  (should be near zero on sphere)\n" sqrt(sum(abs2, result1.harmonic))
@printf "  Reconstruction residual: %.4e\n\n" res1

# ─────────────────────────────────────────────────────────────────────────────
# Case 2: Mixed form ω = d(z) + *d(x)
# ─────────────────────────────────────────────────────────────────────────────

println("── Case 2: Mixed form ω = d(z) + d(x) ──────────────────────────────")
f_x  = Float64[p[1] for p in mesh.points]
ω_ex = gradient_0_to_1(mesh, dec, f_z)   # exact: d(z)
ω_ce = gradient_0_to_1(mesh, dec, f_x)   # also exact: d(x), but orthogonal to d(z)
ω2   = ω_ex .+ ω_ce

result2 = hodge_decompose_1form(mesh, geom, dec, ω2)
res2     = hodge_decomposition_residual(mesh, geom, dec, ω2, result2)

@printf "  Input norm  : %.4e\n"   sqrt(sum(abs2, ω2))
@printf "  Exact  norm : %.4e\n"   sqrt(sum(abs2, result2.exact))
@printf "  Coexact norm: %.4e\n"   sqrt(sum(abs2, result2.coexact))
@printf "  Harmonic norm: %.4e  (near zero: sphere has b¹=0)\n" sqrt(sum(abs2, result2.harmonic))
@printf "  Reconstruction residual: %.4e\n\n" res2

# ─────────────────────────────────────────────────────────────────────────────
# Case 3: Orthogonality check
# ─────────────────────────────────────────────────────────────────────────────

println("── Case 3: Orthogonality of components ──────────────────────────────")
# Exact and co-exact should be L²-orthogonal
dot_ec = dot(result2.exact, result2.coexact)
dot_eh = dot(result2.exact, result2.harmonic)
dot_ch = dot(result2.coexact, result2.harmonic)

@printf "  ⟨exact, coexact⟩  = %.4e  (should be ~0)\n"   dot_ec
@printf "  ⟨exact, harmonic⟩ = %.4e  (should be ~0)\n"   dot_eh
@printf "  ⟨coexact, harmonic⟩ = %.4e  (should be ~0)\n" dot_ch
println()

println("Hodge decomposition sphere example complete.")
