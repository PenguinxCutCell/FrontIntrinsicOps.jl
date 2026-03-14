# examples/vector_calculus_demo.jl
#
# Demonstrate tangential projection, gradient, divergence, and 0-form/1-form
# operations on the sphere.
#
# Run:  julia --project examples/vector_calculus_demo.jl

using FrontIntrinsicOps
using StaticArrays
using LinearAlgebra
using Printf

println("="^60)
println("  Vector Calculus on the Unit Sphere")
println("="^60)
println()

# ── Build sphere ──────────────────────────────────────────────────────────────

function make_uvsphere_demo(R=1.0; nφ=16, nθ=32)
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
    for j in 0:(nφ-1 <= 0 ? 0 : nθ-1)
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

mesh = make_uvsphere_demo(1.0; nφ=16, nθ=32)
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)
topo = build_topology(mesh)
nv   = length(mesh.points)
ne   = length(topo.edges)

@printf "  Mesh: %d vertices, %d edges, %d faces\n\n" nv ne length(mesh.faces)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Tangential projection
# ─────────────────────────────────────────────────────────────────────────────

println("── 1. Tangential projection ──────────────────────────────────────────")
# Take a 3D vector field (e.g., constant (0,0,1)) and project onto tangent plane
f_const = [SVector{3,Float64}(0.0, 0.0, 1.0) for _ in 1:nv]
f_tang  = tangential_projection(mesh, geom, f_const)

# The projected field should be orthogonal to the vertex normal
max_dot = maximum(dot(f_tang[i], geom.vertex_normals[i]) for i in 1:nv)
@printf "  Max |v_tang · n|: %.2e  (should be near zero)\n" max_dot

# The projection should reduce the magnitude of the field
avg_before = sum(norm(v) for v in f_const) / nv
avg_after  = sum(norm(v) for v in f_tang) / nv
@printf "  Avg |v| before: %.4f,  after projection: %.4f\n\n" avg_before avg_after

# ─────────────────────────────────────────────────────────────────────────────
# 2. Gradient: ∇Γ f for a 0-form f
# ─────────────────────────────────────────────────────────────────────────────

println("── 2. Surface gradient (0-form to 1-form) ───────────────────────────")
# f = z (spherical harmonic l=1, m=0)
f0 = Float64[p[3] for p in mesh.points]

# The gradient of a constant should be near zero
f_const0 = ones(Float64, nv)
df_const  = gradient_0_to_1(mesh, dec, f_const0)   # should be ~0 everywhere
@printf "  max |∇Γ(1)|: %.2e  (should be near zero)\n" maximum(abs.(df_const))

# Gradient of z: should give the tangent part of (0,0,1) = spherical grad
df = gradient_0_to_1(mesh, dec, f0)
@printf "  max |∇Γ(z)|: %.4e  (non-zero, z has non-trivial gradient)\n\n" maximum(abs.(df))

# ─────────────────────────────────────────────────────────────────────────────
# 3. Divergence: 1-form to 0-form
# ─────────────────────────────────────────────────────────────────────────────

println("── 3. Divergence (1-form to 0-form) ────────────────────────────────")
# d* applied to an exact form d(f) should give -ΔΓ f (up to sign convention)
# For exact form: d* d f ≈ L f
df_z = gradient_0_to_1(mesh, dec, f0)
div_df_z = divergence_1_to_0(mesh, geom, dec, df_z)

# Compare with L applied to f0
Lf = dec.lap0 * f0
# The residual (L f - d* d f) should be small on a good mesh
max_diff = maximum(abs.(Lf .- div_df_z))
@printf "  max |Lf − d*df|: %.4e  (should be small)\n\n" max_diff

# ─────────────────────────────────────────────────────────────────────────────
# 4. 1-form ↔ tangent vector conversion
# ─────────────────────────────────────────────────────────────────────────────

println("── 4. 1-form ↔ tangent vector conversion ───────────────────────────")
# Convert gradient 1-form to tangent vectors
tvecs   = oneform_to_tangent_vectors(mesh, geom, topo, df_z)
# All tangent vectors should be orthogonal to the face normals
max_n_comp = maximum(abs(dot(tvecs[fi], geom.face_normals[fi]))
                     for fi in 1:length(mesh.faces))
@printf "  max |v · n_face|: %.2e  (tangential vectors should be orthogonal to normals)\n" max_n_comp

# Round-trip: tangent vector → 1-form → tangent vector
df_rt = tangent_vectors_to_1form(mesh, geom, topo, tvecs)
@printf "  1-form round-trip max diff: %.4e\n\n" maximum(abs.(df_z .- df_rt))

# ─────────────────────────────────────────────────────────────────────────────
# 5. Surface rotation of a 0-form (rot = Hodge dual of exterior derivative)
# ─────────────────────────────────────────────────────────────────────────────

println("── 5. Surface curl / rotation (rot₀) ───────────────────────────────")
# surface_rot_0form returns per-face tangent vectors (rotation of gradient)
rot_f0 = surface_rot_0form(mesh, geom, f0)  # Vector{SVector{3,T}}, face-centered
@printf "  max |rot₀(z)|_face: %.4e\n" maximum(norm.(rot_f0))
@printf "  (Non-zero: z has non-trivial tangential rotation on the sphere)\n"
# All rot vectors should be in the tangent plane
max_n = maximum(abs(dot(rot_f0[fi], geom.face_normals[fi])) for fi in 1:length(mesh.faces))
@printf "  max |rot · n_face|: %.2e  (should be near zero)\n\n" max_n

println("Vector calculus demo complete.")
