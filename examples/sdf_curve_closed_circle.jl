using FrontIntrinsicOps
using StaticArrays
using Statistics

mesh = sample_circle(1.0, 128)
cache = build_signed_distance_cache(mesh)

xs = range(-1.8, 1.8; length=120)
ys = range(-1.8, 1.8; length=120)
pts = SVector{2,Float64}[SVector(x, y) for y in ys for x in xs]

S, I, C, N = signed_distance(pts, cache; sign_mode=:auto)
println("Closed circle SDF stats: min=$(minimum(S)), max=$(maximum(S)), mean=$(mean(S))")
println("Closest primitive id range: [$(minimum(I)), $(maximum(I))]")

@assert signed_distance(SVector(0.0, 0.0), cache; sign_mode=:winding).distance < 0
@assert signed_distance(SVector(2.0, 0.0), cache; sign_mode=:winding).distance > 0
@assert abs(signed_distance(SVector(1.0, 0.0), cache; sign_mode=:winding).distance) ≤ 1e-9
