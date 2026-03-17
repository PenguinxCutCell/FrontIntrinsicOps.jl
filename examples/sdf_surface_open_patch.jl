using FrontIntrinsicOps
using StaticArrays
using Statistics

pts = [
    SVector(0.0, 0.0, 0.0),
    SVector(1.0, 0.0, 0.0),
    SVector(1.0, 1.0, 0.0),
    SVector(0.0, 1.0, 0.0),
]
faces = [SVector(1, 2, 3), SVector(1, 3, 4)]
mesh = SurfaceMesh{Float64}(pts, faces)
cache = build_signed_distance_cache(mesh)

vals = range(-0.5, 1.5; length=26)
zs = range(-0.8, 0.8; length=40)
queries = SVector{3,Float64}[SVector(x, y, z) for z in zs for y in vals for x in vals]

S, _, _, _ = signed_distance(queries, cache; sign_mode=:pseudonormal)
println("Open patch oriented-distance stats: min=$(minimum(S)), max=$(maximum(S)), mean=$(mean(S))")

s_top = signed_distance(SVector(0.5, 0.5, 0.5), cache; sign_mode=:pseudonormal).distance
s_bot = signed_distance(SVector(0.5, 0.5, -0.5), cache; sign_mode=:pseudonormal).distance
println("Reference signs: top=$s_top, bottom=$s_bot")
@assert s_top * s_bot < 0
