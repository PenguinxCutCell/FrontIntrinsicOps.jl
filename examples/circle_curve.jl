# examples/circle_curve.jl
#
# Demonstrates constructing a sampled circle, computing curve geometry,
# assembling DEC operators, and verifying convergence.

using FrontIntrinsicOps
using StaticArrays
using Printf
using LinearAlgebra

R = 1.5
println("── Sampled circle, R = $R ──────────────────────────────────")

for N in [16, 64, 256, 1024]
    pts  = [R * SVector{2,Float64}(cos(2π*k/N), sin(2π*k/N)) for k in 0:N-1]
    mesh = load_curve_points(pts; closed=true)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)

    L    = measure(mesh, geom)
    A    = enclosed_measure(mesh)
    κ    = curvature(mesh, geom)
    κ_mean = sum(κ) / length(κ)

    Lu   = laplace_beltrami(mesh, geom, dec, ones(Float64, N))
    Lu_max = maximum(abs, Lu)

    @printf "  N=%4d  len=%.6f  (2πR=%.6f)  err_len=%.2e\n"  N L (2π*R) abs(L-2π*R)/(2π*R)
    @printf "         area=%.6f  (πR²=%.6f)  err_area=%.2e\n"  A (π*R^2) abs(A-π*R^2)/(π*R^2)
    @printf "         κ_mean=%.6f  (1/R=%.6f)  err_κ=%.2e\n"  κ_mean (1/R) abs(κ_mean-1/R)/(1/R)
    @printf "         max|L*1|=%.2e\n"  Lu_max
end

println("\nDone.")
