# test_helpers.jl – Shared helper functions for tests.
# This file is included by runtests.jl and individual test files.

using FrontIntrinsicOps
using StaticArrays
using LinearAlgebra
using Statistics: mean

"""Build a regular N-gon sampling of a circle of radius R."""
function make_circle(N::Int, R::Float64=1.0)
    pts = [R * SVector{2,Float64}(cos(2π*k/N), sin(2π*k/N)) for k in 0:N-1]
    return load_curve_points(pts; closed=true)
end

"""
Build a UV-sphere triangulation of radius R with `nφ` parallels and `nθ`
meridians.  Returns a properly oriented (outward-normal) SurfaceMesh.
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

"""Build a flat square patch triangulated grid in the xy-plane."""
function make_flat_patch(; N::Int=10, L::Float64=1.0)
    pts   = SVector{3,Float64}[]
    faces = SVector{3,Int}[]
    h     = L / N
    for j in 0:N, i in 0:N
        push!(pts, SVector{3,Float64}(i*h, j*h, 0.0))
    end
    idx_v(i,j) = j*(N+1) + i + 1
    for j in 0:(N-1), i in 0:(N-1)
        v00 = idx_v(i,   j  )
        v10 = idx_v(i+1, j  )
        v01 = idx_v(i,   j+1)
        v11 = idx_v(i+1, j+1)
        push!(faces, SVector{3,Int}(v00, v10, v11))
        push!(faces, SVector{3,Int}(v00, v11, v01))
    end
    return SurfaceMesh{Float64}(pts, faces)
end
