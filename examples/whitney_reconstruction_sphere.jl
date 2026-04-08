using FrontIntrinsicOps
using StaticArrays
using LinearAlgebra

mesh = generate_icosphere(1.0, 2)
geom = compute_geometry(mesh)
topo = build_topology(mesh)

println("sphere mesh: nv=$(length(mesh.points)) ne=$(length(topo.edges)) nf=$(length(mesh.faces))")

# Sample smooth scalar and reconstruct facewise values
c0 = [sin(2p[1]) + 0.3p[2] - 0.1p[3] for p in mesh.points]
uf = reconstruct_0form(c0, mesh, geom; representation=:facewise)
println("first face u(centroid) = ", uf[1].eval(SVector(1/3, 1/3, 1/3)))

# Sample 1-form DOFs from an ambient vector field
v = x -> SVector{3,Float64}(-x[2], x[1], 0.0)
c1 = interpolate_1form(v, mesh, geom; representation=:ambient_vector)
αf = reconstruct_1form(c1, mesh, geom; representation=:facewise_tangent)
println("first face alpha# norm = ", norm(αf[1]))

# Sample 2-form density and reconstruct
β = (x, n) -> 1.0 + 0.2 * dot(x, n)
c2 = interpolate_2form(β, mesh, geom)
ρ = reconstruct_2form(c2, mesh, geom; representation=:facewise_density)
println("first face density = ", ρ[1])
