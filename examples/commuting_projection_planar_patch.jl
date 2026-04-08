using FrontIntrinsicOps
using LinearAlgebra
using StaticArrays

mesh = generate_icosphere(1.0, 1)
# flatten to a planar patch by dropping z and rebuilding a local patch mesh
# for an exact planar commuting check we use a Cartesian patch directly:
mesh = let
    pts = SVector{3,Float64}[]
    faces = SVector{3,Int}[]
    N = 12
    h = 1.0 / N
    for j in 0:N, i in 0:N
        push!(pts, SVector(i * h, j * h, 0.0))
    end
    vid(i, j) = j * (N + 1) + i + 1
    for j in 0:(N - 1), i in 0:(N - 1)
        v00 = vid(i, j)
        v10 = vid(i + 1, j)
        v01 = vid(i, j + 1)
        v11 = vid(i + 1, j + 1)
        push!(faces, SVector(v00, v10, v11))
        push!(faces, SVector(v00, v11, v01))
    end
    SurfaceMesh{Float64}(pts, faces)
end

geom = compute_geometry(mesh)
dec = build_dec(mesh, geom)

f = x -> x[1]^2 - 0.5x[2] + 0.2
res01 = projection_commutator_01(f, mesh, geom, dec)
println("||Π1(df) - d0Π0(f)|| = ", norm(res01))

α = (x, t, eid) -> x[1] + x[2] + 0.01 * eid
res12 = projection_commutator_12(α, mesh, geom, dec; representation=:line_density)
println("||Π2(dα) - d1Π1(α)|| = ", norm(res12))
