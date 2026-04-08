#!/usr/bin/env julia

# Geodesic farthest-point sampling demo on a torus.

using FrontIntrinsicOps
using Printf

function main()
    mesh = generate_torus(2.0, 0.7, 24, 32)
    geom = compute_geometry(mesh)
    dec = build_dec(mesh, geom)

    samples = farthest_point_sampling_geodesic(mesh, geom, dec, 12; seed=1)
    println("geodesic FPS vertex ids:")
    println(samples)

    d1 = geodesic_distance_to_vertex(mesh, geom, dec, samples[1])
    @printf("distance spread from first sample: min=%.3e max=%.3e\n", minimum(d1), maximum(d1))
end

main()
