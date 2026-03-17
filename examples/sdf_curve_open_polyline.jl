using FrontIntrinsicOps
using StaticArrays
using Statistics

pts_curve = [
    SVector(-1.0, 0.0),
    SVector(-0.3, 0.4),
    SVector(0.4, -0.1),
    SVector(1.2, 0.3),
]
mesh = load_curve_points(pts_curve; closed=false)
cache = build_signed_distance_cache(mesh)

xs = range(-1.6, 1.6; length=90)
ys = range(-1.2, 1.2; length=90)
queries = SVector{2,Float64}[SVector(x, y) for y in ys for x in xs]

S, _, _, _ = signed_distance(queries, cache; sign_mode=:pseudonormal)
println("Open polyline oriented-distance stats: min=$(minimum(S)), max=$(maximum(S)), mean=$(mean(S))")

s_up = signed_distance(SVector(0.0, 0.8), cache; sign_mode=:pseudonormal).distance
s_dn = signed_distance(SVector(0.0, -0.8), cache; sign_mode=:pseudonormal).distance
println("Reference signs: above=$s_up, below=$s_dn")
@assert s_up * s_dn < 0
