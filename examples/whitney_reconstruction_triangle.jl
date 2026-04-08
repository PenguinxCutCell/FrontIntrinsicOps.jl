using FrontIntrinsicOps
using StaticArrays

# One triangle in the xy-plane
pts = SVector{3,Float64}[
    SVector(0.0, 0.0, 0.0),
    SVector(1.0, 0.0, 0.0),
    SVector(0.0, 1.0, 0.0),
]
faces = SVector{3,Int}[SVector(1, 2, 3)]
mesh = SurfaceMesh{Float64}(pts, faces)
geom = compute_geometry(mesh)

topo = build_topology(mesh)
println("triangle: nv=$(length(mesh.points)) ne=$(length(topo.edges)) nf=$(length(mesh.faces))")

# 0-form reconstruction (piecewise linear scalar)
c0 = [1.0, 2.0, -0.5]
rec0 = reconstruct_0form_face(c0, 1, mesh, geom)
println("u(centroid) = ", rec0.eval(SVector(1/3, 1/3, 1/3)))
println("grad u = ", rec0.gradient)

# 1-form reconstruction (Whitney 1-form)
c1 = [0.7, -0.2, 1.1]
rec1 = reconstruct_1form_face(c1, 1, mesh, geom)
println("alpha coefficients (face-oriented) = ", rec1.coefficients)
println("alpha#(centroid) = ", rec1.tangent_at_centroid)

# 2-form reconstruction (face density)
c2 = [2.0]
rec2 = reconstruct_2form_face(c2, 1, mesh, geom)
println("beta density = ", rec2.density)
println("check integral = ", rec2.density * geom.face_areas[1])
