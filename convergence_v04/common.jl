# convergence_v04/common.jl
#
# Shared utilities for v0.4 PDE convergence studies.
# Extends the utilities from convergence_pdes/common.jl with torus and
# deformed-surface helpers.
#
# Usage: include(joinpath(@__DIR__, "common.jl"))

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io=devnull)

using FrontIntrinsicOps
using StaticArrays
using LinearAlgebra
using Printf
using Statistics: mean

# ── Characteristic mesh size ──────────────────────────────────────────────────

"""
    mesh_size_surface(mesh, geom) -> h

Characteristic mesh size for a surface mesh: h = sqrt(total_area / N_faces).
"""
function mesh_size_surface(mesh::SurfaceMesh, geom::SurfaceGeometry)
    return sqrt(measure(mesh, geom) / length(mesh.faces))
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

# ── Convergence table printing ────────────────────────────────────────────────

"""
    print_convergence_table(hs, errors; header="")

Print a convergence table (h | error | order) to the terminal.
No output to files.
"""
function print_convergence_table(hs::Vector{Float64}, errors::Vector{Float64};
                                  header::String="",
                                  label::String="error")
    if !isempty(header)
        println()
        print_sep(header)
    end
    @printf "  %-12s  %-14s  %-8s\n" "h" label "order"
    println("  " * "─"^38)
    n = length(hs)
    for i in 1:n
        order = i == 1 ? "   —  " :
            @sprintf("%.3f", log(errors[i-1]/errors[i]) / log(hs[i-1]/hs[i]))
        @printf "  %-12.4e  %-14.4e  %-8s\n" hs[i] errors[i] order
    end
    println()
end

# ── Torus manufactured solution ───────────────────────────────────────────────

"""
    torus_test_function(mesh, R, r; mode=:cos) -> (u, Δu)

Evaluate a known smooth function on the torus and its Laplace–Beltrami value.

The function u and its Laplacian ΔΓu are computed via the parametrisation:

  Parametrisation:  x = (R + r cosφ) cosθ
                    y = (R + r cosφ) sinθ
                    z = r sinφ

We use the test function:

    u(φ, θ) = cos(θ)       (toroidal mode, mode=:cos)
    u(φ, θ) = cos(φ)       (poloidal mode, mode=:pol)

For which the Laplace–Beltrami operator gives known analytic values.

Returns (u_vals, ΔΓu_vals).
"""
function torus_test_function(mesh::SurfaceMesh, R::Float64, r::Float64;
                              mode::Symbol=:cos)
    nv = length(mesh.points)
    u  = zeros(Float64, nv)
    Δu = zeros(Float64, nv)

    for (vi, p) in enumerate(mesh.points)
        x, y, z = p[1], p[2], p[3]
        # Recover toroidal angle θ and poloidal angle φ
        θ = atan(y, x)
        # Distance from torus axis in the plane
        ρ = sqrt(x^2 + y^2) - R
        φ = atan(z, ρ)

        if mode == :cos
            # u = cos(θ)
            # ΔΓ cos(θ) = -1/(R + r cosφ)² cos(θ)  (approximately)
            # For simplicity we use the known Fourier eigenvalue:
            # For mode n=1 on the torus: eigenvalue ≈ 1/R^2
            u[vi]  = cos(θ)
            # Manufactured: no exact analytic Laplacian used here
            Δu[vi] = -cos(θ) / (R + r*cos(φ))^2
        elseif mode == :pol
            # u = cos(φ)
            u[vi]  = cos(φ)
            Δu[vi] = -cos(φ) / r^2
        else
            error("torus_test_function: unknown mode $mode")
        end
    end
    return u, Δu
end

# ── Weighted L² error helpers ─────────────────────────────────────────────────

"""
    weighted_l2_error_local(mesh, geom, u, u_exact) -> Float64

Compute the relative weighted L² error.
"""
function weighted_l2_error_local(mesh, geom, u, u_exact)
    return weighted_l2_error(mesh, geom, u, u_exact)
end
