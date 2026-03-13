# examples/torus_from_stl.jl
#
# Demonstrates torus geometry: non-uniform curvature distribution.
# Generates a torus parametrically (no STL needed) or loads from STL.
#
# Usage:
#   julia --project examples/torus_from_stl.jl [path/to/torus.stl]

using FrontIntrinsicOps
using StaticArrays
using Printf
using LinearAlgebra
using Statistics: mean, std

"""
Build a torus mesh with major radius R and tube radius r.
"""
function make_torus(R::Float64=2.0, r::Float64=0.5; nθ::Int=40, nφ::Int=20)
    pts = SVector{3,Float64}[]
    for j in 0:(nφ-1)
        φ = j * 2π / nφ
        for i in 0:(nθ-1)
            θ = i * 2π / nθ
            x = (R + r*cos(φ)) * cos(θ)
            y = (R + r*cos(φ)) * sin(θ)
            z = r * sin(φ)
            push!(pts, SVector{3,Float64}(x, y, z))
        end
    end

    idx(i,j) = mod(j, nφ)*nθ + mod(i, nθ) + 1
    faces = SVector{3,Int}[]
    for j in 0:(nφ-1)
        for i in 0:(nθ-1)
            v00 = idx(i,   j  )
            v10 = idx(i+1, j  )
            v01 = idx(i,   j+1)
            v11 = idx(i+1, j+1)
            push!(faces, SVector{3,Int}(v00, v10, v11))
            push!(faces, SVector{3,Int}(v00, v11, v01))
        end
    end
    return SurfaceMesh{Float64}(pts, faces)
end

R_major = 2.0
r_tube  = 0.5

if length(ARGS) > 0
    println("Loading torus from STL: ", ARGS[1])
    mesh = load_surface_stl(ARGS[1])
    R_major = NaN; r_tube = NaN
else
    println("Generating torus: R=$(R_major), r=$(r_tube), nθ=40, nφ=20")
    mesh = make_torus(R_major, r_tube; nθ=40, nφ=20)
end

# ── Mesh report ──────────────────────────────────────────────────────────────
rpt = check_mesh(mesh)
println("\n── Mesh topology ──────────────────────────────────────────")
@printf "  vertices : %d\n"  rpt.n_vertices
@printf "  edges    : %d\n"  rpt.n_edges
@printf "  faces    : %d\n"  rpt.n_faces
@printf "  χ (V−E+F): %d  (torus: 0)\n"  rpt.euler_characteristic
@printf "  closed   : %s\n"  string(rpt.closed)

# ── Geometry ─────────────────────────────────────────────────────────────────
geom = compute_geometry(mesh)
A    = measure(mesh, geom)

if isfinite(R_major)
    A_exact = 4π^2 * R_major * r_tube
    @printf "\n── Surface area ────────────────────────────────────────────\n"
    @printf "  computed : %.6f\n"  A
    @printf "  exact    : %.6f\n"  A_exact
    @printf "  rel err  : %.4e\n"  abs(A - A_exact)/A_exact
end

# ── DEC + curvature ──────────────────────────────────────────────────────────
dec = build_dec(mesh, geom)
H   = mean_curvature(mesh, geom, dec)

println("\n── Mean curvature distribution (non-uniform on torus) ──────")
@printf "  mean |H| : %.4f\n"  mean(abs.(H))
@printf "  std  |H| : %.4f  (> 0 shows variation)\n"  std(abs.(H))
@printf "  min  |H| : %.4f\n"  minimum(abs.(H))
@printf "  max  |H| : %.4f\n"  maximum(abs.(H))

K = gaussian_curvature(mesh, geom)
println("\n── Gaussian curvature distribution ─────────────────────────")
@printf "  mean K   : %.4f\n"  mean(K)
@printf "  min  K   : %.4f  (negative on inner ring)\n"  minimum(K)
@printf "  max  K   : %.4f  (positive on outer ring)\n"  maximum(K)

# Gauss–Bonnet check: ∫ K dA = 2π χ = 0 for torus
nv = length(mesh.points)
total_K = sum(K[i] * geom.vertex_dual_areas[i] for i in 1:nv)
@printf "\n  ∫ K dA  : %.4f  (Gauss–Bonnet: should be ≈ 0)\n"  total_K

println("\nDone.")
