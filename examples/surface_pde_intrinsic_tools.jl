#!/usr/bin/env julia

# Use recent intrinsic tools (geodesics + Lie derivative) in a surface PDE loop.
# Model: explicit Cartan advection + implicit diffusion on a static sphere.

using FrontIntrinsicOps
using StaticArrays
using LinearAlgebra
using Printf

function main()
    mesh = generate_icosphere(1.0, 2)
    geom = compute_geometry(mesh)
    dec = build_dec(mesh, geom)

    src = argmax([p[3] for p in mesh.points])
    d = geodesic_distance_to_vertex(mesh, geom, dec, src)
    σ = 0.45
    u = exp.(-(d ./ σ).^2)

    # Tangential rigid-rotation field.
    X = [tangential_project(SVector(-p[2], p[1], 0.0), geom.face_normals[fi]) for (fi, p) in enumerate([sum(mesh.points[f[k]] for k in 1:3) / 3 for f in mesh.faces])]

    dt = 5e-3
    μ = 5e-3
    nsteps = 80
    fac = nothing
    M = mass_matrix(mesh, geom)
    mass0 = dot(diag(M), u)

    for _ in 1:nsteps
        Lu = lie_derivative(X, u, mesh, geom, dec)
        u .= u .- dt .* Lu
        u, fac = step_surface_diffusion_backward_euler(mesh, geom, dec, u, dt, μ; factorization=fac)
    end

    mass1 = dot(diag(M), u)
    @printf("L2 norm(u) = %.6e\n", norm(u))
    @printf("mass drift = %.3e\n", abs(mass1 - mass0) / (abs(mass0) + 1e-14))
    @printf("range(u)   = [%.6e, %.6e]\n", minimum(u), maximum(u))
end

main()
