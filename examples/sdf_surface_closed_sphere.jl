using FrontIntrinsicOps
using StaticArrays
using Statistics

mesh = generate_icosphere(1.0, 2)
cache = build_signed_distance_cache(mesh)

vals = range(-1.6, 1.6; length=34)
queries = SVector{3,Float64}[SVector(x, y, z) for z in vals for y in vals for x in vals]

S, I, _, _ = signed_distance(queries, cache; sign_mode=:auto)
println("Closed sphere SDF stats: min=$(minimum(S)), max=$(maximum(S)), mean=$(mean(S))")
println("Closest face id range: [$(minimum(I)), $(maximum(I))]")

@assert signed_distance(SVector(0.0, 0.0, 0.0), cache; sign_mode=:winding).distance < 0
@assert signed_distance(SVector(2.0, 0.0, 0.0), cache; sign_mode=:winding).distance > 0
@assert abs(signed_distance(SVector(1.0, 0.0, 0.0), cache; sign_mode=:winding).distance) ≤ 2e-2
