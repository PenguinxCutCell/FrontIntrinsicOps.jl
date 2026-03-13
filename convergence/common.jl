# convergence/common.jl – Shared utilities for convergence studies.
#
# Load this file at the top of each convergence script:
#   include(joinpath(@__DIR__, "common.jl"))

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using FrontIntrinsicOps
using LinearAlgebra
using Printf
using Statistics: mean

# ─────────────────────────────────────────────────────────────────────────────
# Terminal formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

"""Print a section header to stdout."""
function print_header(title::String)
    n = 72
    println()
    println("=" ^ n)
    println("  " * title)
    println("=" ^ n)
end

"""Print a sub-section separator."""
function print_sep(title::String = "")
    n = 72
    if isempty(title)
        println("-" ^ n)
    else
        pad = n - length(title) - 4
        println("-- " * title * " " * "-" ^ max(0, pad))
    end
end

"""Right-align a float in scientific notation in a field of given width."""
rpad_num(x::AbstractFloat, width::Int=12) = lpad(@sprintf("%.4e", x), width)
rpad_num(x::Int, width::Int=6) = lpad(string(x), width)

"""
    mesh_size_surface(mesh, geom) -> h

Characteristic mesh size for a surface mesh:
    h = sqrt(total_area / NF)
"""
function mesh_size_surface(mesh::SurfaceMesh, geom::SurfaceGeometry)
    return sqrt(measure(mesh, geom) / length(mesh.faces))
end

"""
    mesh_size_curve(mesh, geom) -> h

Characteristic mesh size for a curve mesh (mean edge length):
    h = total_length / NE
"""
function mesh_size_curve(mesh::CurveMesh, geom::CurveGeometry)
    return measure(mesh, geom) / length(mesh.edges)
end
