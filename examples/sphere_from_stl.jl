# examples/sphere_from_stl.jl
#
# Demonstrates loading a surface from an STL file, computing geometry,
# assembling DEC operators, and reporting area, volume, and mean curvature.
#
# Usage:
#   julia --project examples/sphere_from_stl.jl path/to/sphere.stl
#
# If no STL file is supplied, a UV-sphere is generated programmatically.

using FrontIntrinsicOps
using StaticArrays
using LinearAlgebra
using Printf

# ── Build or load a sphere ───────────────────────────────────────────────────

function make_uvsphere(R::Float64=1.0; nφ::Int=32, nθ::Int=64)
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

R = 1.0  # nominal radius for analytic comparison

if length(ARGS) > 0
    println("Loading STL: ", ARGS[1])
    mesh = load_surface_stl(ARGS[1])
    R = NaN  # no analytic radius known
else
    println("No STL provided – generating a UV-sphere with nφ=32, nθ=64.")
    mesh = make_uvsphere(R; nφ=32, nθ=64)
end

# ── Topology check ──────────────────────────────────────────────────────────
mesh_report = check_mesh(mesh)
println("\n── Mesh topology ──────────────────────────────────────────")
@printf "  vertices : %d\n"   mesh_report.n_vertices
@printf "  edges    : %d\n"   mesh_report.n_edges
@printf "  faces    : %d\n"   mesh_report.n_faces
@printf "  χ (V−E+F): %d\n"   mesh_report.euler_characteristic
@printf "  closed   : %s\n"   string(mesh_report.closed)
@printf "  manifold : %s\n"   string(mesh_report.manifold)
for w in mesh_report.warnings
    println("  WARNING: ", w)
end

# ── Geometry ────────────────────────────────────────────────────────────────
geom = compute_geometry(mesh)
A    = measure(mesh, geom)
V    = enclosed_measure(mesh)

println("\n── Geometry ────────────────────────────────────────────────")
@printf "  total surface area      : %.6f\n"  A
if isfinite(R)
    @printf "  analytic (4πR²)         : %.6f\n"  4π*R^2
    @printf "  relative area error     : %.4e\n"   abs(A - 4π*R^2)/(4π*R^2)
end
@printf "  enclosed volume         : %.6f\n"  V
if isfinite(R)
    @printf "  analytic (4πR³/3)       : %.6f\n"  (4/3)*π*R^3
    @printf "  relative volume error   : %.4e\n"   abs(V - (4/3)*π*R^3)/((4/3)*π*R^3)
end

# ── DEC operators ────────────────────────────────────────────────────────────
dec = build_dec(mesh, geom)
dec_report = check_dec(mesh, geom, dec)

println("\n── DEC diagnostics ─────────────────────────────────────────")
@printf "  d₁*d₀ = 0         : %s  (residual %.2e)\n"  string(dec_report.d1_d0_zero) dec_report.d1_d0_max_residual
@printf "  L*ones ≈ 0        : %s\n"  string(dec_report.lap_constant_nullspace)
@printf "  ⋆₀ positive       : %s\n"  string(dec_report.star0_positive)
@printf "  ⋆₁ positive       : %s\n"  string(dec_report.star1_positive)
for w in dec_report.warnings
    println("  WARNING: ", w)
end

# ── Curvature ────────────────────────────────────────────────────────────────
H = mean_curvature(mesh, geom, dec)
interior = 2:(length(mesh.points)-1)
H_interior = H[interior]
H_mean = sum(abs, H_interior) / length(H_interior)

println("\n── Mean curvature ───────────────────────────────────────────")
@printf "  mean |H| (interior) : %.6f\n"  H_mean
if isfinite(R)
    @printf "  analytic 1/R        : %.6f\n"  1/R
    @printf "  relative error      : %.4e\n"   abs(H_mean - 1/R)/(1/R)
end
@printf "  min |H|             : %.6f\n"  minimum(abs, H)
@printf "  max |H|             : %.6f\n"  maximum(abs, H)

println("\nDone.")
