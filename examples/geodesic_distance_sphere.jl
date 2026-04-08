#!/usr/bin/env julia

# Heat-method geodesic distance demo on a unit sphere.

using FrontIntrinsicOps
using LinearAlgebra
using Printf

function main()
    mesh = generate_icosphere(1.0, 2)
    geom = compute_geometry(mesh)
    dec = build_dec(mesh, geom)

    src = argmax([p[3] for p in mesh.points]) # north-ish pole
    d = geodesic_distance_to_vertex(mesh, geom, dec, src)

    ps = mesh.points[src]
    exact = [acos(clamp(dot(ps, p), -1.0, 1.0)) for p in mesh.points]
    rel = norm(d .- exact) / (norm(exact) + 1e-14)

    @printf("source vertex = %d\n", src)
    @printf("distance range = [%.3e, %.3e]\n", minimum(d), maximum(d))
    @printf("relative error vs great-circle = %.3e\n", rel)
end

main()
