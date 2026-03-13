# examples/surface_poisson.jl
#
# Solve a scalar Laplace–Beltrami / Poisson problem on a closed surface.
#
# Problem: find u on the sphere satisfying
#   L u = f   with compatibility condition  ∫ f dA = 0
#
# We use f = z (the z-coordinate restricted to the sphere), which has zero
# mean by symmetry on a sphere centred at the origin.  The exact solution
# is  u = −R² z / 2  (since L z = (2/R²) z implies L(−R²z/2) = −z = −f... 
# wait: L z = (2/R²) z, so if Lu = f = z, then u = (R²/2) z.
#
# We solve the singular system by projecting out the nullspace (constants)
# before factorising with a regularised direct solve.

using FrontIntrinsicOps
using StaticArrays
using LinearAlgebra
using SparseArrays
using Printf

# ── Build a UV sphere ─────────────────────────────────────────────────────────

function make_uvsphere(R::Float64=1.0; nφ::Int=24, nθ::Int=48)
    pts = SVector{3,Float64}[]
    push!(pts, SVector{3,Float64}(0.0, 0.0, -R))
    for i in 1:(nφ-1)
        φ = -π/2 + i * π / nφ
        for j in 0:(nθ-1)
            θ = j * 2π / nθ
            push!(pts, SVector{3,Float64}(R*cos(φ)*cos(θ), R*cos(φ)*sin(θ), R*sin(φ)))
        end
    end
    push!(pts, SVector{3,Float64}(0.0, 0.0, +R))
    faces = SVector{3,Int}[]
    south = 1; north = length(pts)
    for j in 0:(nθ-1)
        push!(faces, SVector{3,Int}(south, 2+mod(j+1,nθ), 2+j))
    end
    for i in 1:(nφ-2)
        for j in 0:(nθ-1)
            v00 = 2+(i-1)*nθ+j; v01 = 2+(i-1)*nθ+mod(j+1,nθ)
            v10 = 2+i*nθ+j;     v11 = 2+i*nθ+mod(j+1,nθ)
            push!(faces, SVector{3,Int}(v00,v01,v11))
            push!(faces, SVector{3,Int}(v00,v11,v10))
        end
    end
    base = 2+(nφ-2)*nθ
    for j in 0:(nθ-1)
        push!(faces, SVector{3,Int}(north, base+j, base+mod(j+1,nθ)))
    end
    return SurfaceMesh{Float64}(pts, faces)
end

R    = 1.5
mesh = make_uvsphere(R; nφ=24, nθ=48)
geom = compute_geometry(mesh)
dec  = build_dec(mesh, geom)

nv = length(mesh.points)
z  = [p[3] for p in mesh.points]

# Right-hand side: f = z (mean-zero on sphere by symmetry)
f = z

# Compatibility check: ∫ f dA ≈ 0
int_f = integrate_vertex_field(mesh, geom, f)
@printf "Compatibility check: ∫ f dA = %.4e  (should be ≈ 0)\n"  int_f

# ── Solve L u = f ────────────────────────────────────────────────────────────
# The Laplacian L has a one-dimensional nullspace (constants).
# Strategy: add a small diagonal shift ε I to regularise, then correct.
# Alternatively: fix one degree of freedom.
# Here we pin vertex 1 to 0 and solve the reduced system.

L = dec.lap0
L_mod = copy(L)

# Pin vertex 1: set row and column 1 to identity
rows1 = L_mod[1, :]
L_mod[1, :] .= 0.0
for j in 1:nv
    if j != 1; L_mod[j, 1] = 0.0; end
end
L_mod[1, 1] = 1.0
dropzeros!(L_mod)

f_mod = copy(f)
f_mod[1] = 0.0   # pin u[1] = 0

u = Matrix(L_mod) \ f_mod  # dense solve (small example)

# Exact solution: u_exact = (R²/2) z  (up to a constant, we pin to 0)
u_exact = (R^2 / 2) .* z
# Shift so u_exact[1] matches our pin
u_exact .-= u_exact[1]

# Residual
u_res = L * u  # should equal f up to pin correction
residual_at_interior = maximum(abs, u_res[2:end] - f[2:end])
@printf "Max residual |Lu - f| (interior): %.4e\n"  residual_at_interior

# Error vs analytic
err = maximum(abs, u - u_exact) / (maximum(abs, u_exact) + 1e-14)
@printf "Relative error vs analytic solution: %.4e\n"  err

println("\nPoisson solve on sphere complete.")
println("The operator L = −Δ_Γ is usable for surface PDEs.")
