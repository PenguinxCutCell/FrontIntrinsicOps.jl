# convergence_pdes/common.jl
#
# Shared utilities for PDE convergence studies.
#
# Usage: include(joinpath(@__DIR__, "common.jl"))

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using FrontIntrinsicOps
using StaticArrays
using LinearAlgebra
using Printf
using Statistics: mean

# ── UV-sphere mesh builder ────────────────────────────────────────────────────

"""
    make_uvsphere(R=1.0; nφ=16, nθ=32) -> SurfaceMesh

Build a UV-sphere triangulation of radius R with nφ parallels and nθ meridians.
Identical to the version in test/test_helpers.jl.
"""
function make_uvsphere(R::Float64=1.0; nφ::Int=16, nθ::Int=32)
    pts = SVector{3,Float64}[]
    push!(pts, SVector{3,Float64}(0.0, 0.0, -R))  # south pole
    for i in 1:(nφ-1)
        φ = -π/2 + i * π / nφ
        for j in 0:(nθ-1)
            θ = j * 2π / nθ
            push!(pts, SVector{3,Float64}(R*cos(φ)*cos(θ), R*cos(φ)*sin(θ), R*sin(φ)))
        end
    end
    push!(pts, SVector{3,Float64}(0.0, 0.0, +R))  # north pole

    faces = SVector{3,Int}[]
    south = 1
    north = length(pts)

    for j in 0:(nθ-1)
        v1 = 2 + j
        v2 = 2 + mod(j+1, nθ)
        push!(faces, SVector{3,Int}(south, v2, v1))
    end

    for i in 1:(nφ-2)
        for j in 0:(nθ-1)
            v00 = 2 + (i-1)*nθ + j
            v01 = 2 + (i-1)*nθ + mod(j+1, nθ)
            v10 = 2 + i*nθ + j
            v11 = 2 + i*nθ + mod(j+1, nθ)
            push!(faces, SVector{3,Int}(v00, v01, v11))
            push!(faces, SVector{3,Int}(v00, v11, v10))
        end
    end

    base = 2 + (nφ-2)*nθ
    for j in 0:(nθ-1)
        v1 = base + j
        v2 = base + mod(j+1, nθ)
        push!(faces, SVector{3,Int}(north, v1, v2))
    end

    return SurfaceMesh{Float64}(pts, faces)
end

# ── Characteristic mesh size ──────────────────────────────────────────────────

"""
    mesh_size_surface(mesh, geom) -> h

Characteristic mesh size for a surface mesh: h = sqrt(total_area / N_faces).
"""
function mesh_size_surface(mesh::SurfaceMesh, geom::SurfaceGeometry)
    return sqrt(measure(mesh, geom) / length(mesh.faces))
end

# ── Exact solution for eigenmode diffusion ────────────────────────────────────

"""
    sphere_eigenmode_exact(mesh, t, μ; λ=2.0) -> Vector{Float64}

Exact solution for diffusion of the z-eigenmode on a unit sphere:

    u(x,y,z,t) = exp(-μ λ t) · z

where λ = 2/R² is the first non-trivial eigenvalue (default λ=2 for R=1).
"""
function sphere_eigenmode_exact(mesh::SurfaceMesh, t::Float64, μ::Float64;
                                 λ::Float64=2.0)
    z = Float64[p[3] for p in mesh.points]
    return exp(-μ * λ * t) .* z
end

# ── Convergence table printing ────────────────────────────────────────────────

"""
    print_convergence_table(hs, errors; header="", fmt_h="%.4e", fmt_e="%.4e")

Print a convergence table with computed orders.

Columns: h | error | order
"""
function print_convergence_table(hs::Vector{Float64}, errors::Vector{Float64};
                                  header::String="",
                                  extra_cols::Vector{Pair{String,Vector{Float64}}}=Pair{String,Vector{Float64}}[])
    if !isempty(header)
        println()
        println(header)
        println("─"^72)
    end

    # Build column header
    col_names = ["h", "error", "order"]
    for (name, _) in extra_cols
        push!(col_names, name, "order")
    end

    # Header row
    @printf "  %-12s  %-12s  %-8s" "h" "error" "order"
    for (name, _) in extra_cols
        @printf "  %-12s  %-8s" name "order"
    end
    println()
    @printf "  %s" "─"^12
    @printf "  %s  %s" "─"^12 "─"^8
    for _ in extra_cols
        @printf "  %s  %s" "─"^12 "─"^8
    end
    println()

    n = length(hs)
    for i in 1:n
        order = i == 1 ? "   —  " :
            @sprintf("%.3f", log(errors[i-1]/errors[i]) / log(hs[i-1]/hs[i]))
        @printf "  %-12.4e  %-12.4e  %-8s" hs[i] errors[i] order
        for (_, vals) in extra_cols
            ord_extra = i == 1 ? "   —  " :
                @sprintf("%.3f", log(vals[i-1]/vals[i]) / log(hs[i-1]/hs[i]))
            @printf "  %-12.4e  %-8s" vals[i] ord_extra
        end
        println()
    end
    println()
end

"""
    print_time_convergence_table(dts, errors_be, errors_cn)

Print a time-refinement convergence table for two schemes side by side.
"""
function print_time_convergence_table(dts::Vector{Float64},
                                       errors_be::Vector{Float64},
                                       errors_cn::Vector{Float64})
    @printf "  %-12s  %-12s  %-8s  %-12s  %-8s\n" "dt" "err_BE" "ord_BE" "err_CN" "ord_CN"
    println("  " * "─"^60)
    n = length(dts)
    for i in 1:n
        ord_be = i == 1 ? "  —   " :
            @sprintf("%.3f", log(errors_be[i-1]/errors_be[i]) / log(dts[i-1]/dts[i]))
        ord_cn = i == 1 ? "  —   " :
            @sprintf("%.3f", log(errors_cn[i-1]/errors_cn[i]) / log(dts[i-1]/dts[i]))
        @printf "  %-12.4e  %-12.4e  %-8s  %-12.4e  %-8s\n" dts[i] errors_be[i] ord_be errors_cn[i] ord_cn
    end
    println()
end

# ── Terminal formatting ───────────────────────────────────────────────────────

function print_header(title::String)
    n = 72
    println()
    println("="^n)
    println("  " * title)
    println("="^n)
end

function print_sep(title::String="")
    n = 72
    if isempty(title)
        println("─"^n)
    else
        pad = n - length(title) - 4
        println("── " * title * " " * "─"^max(0, pad))
    end
end
