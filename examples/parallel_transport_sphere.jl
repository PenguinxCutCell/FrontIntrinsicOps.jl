#!/usr/bin/env julia

# Parallel-transport holonomy demo on a sphere.

using FrontIntrinsicOps
using StaticArrays
using LinearAlgebra
using Printf

function ordered_faces_around_vertex(mesh::SurfaceMesh{T}, geom::SurfaceGeometry{T}, vid::Int) where {T}
    vf = vertex_to_faces(mesh)[vid]
    Fv = vertex_tangent_frames(mesh, geom)[vid]
    p0 = mesh.points[vid]

    vals = Vector{Tuple{T,Int}}()
    for fi in vf
        f = mesh.faces[fi]
        c = (mesh.points[f[1]] + mesh.points[f[2]] + mesh.points[f[3]]) / 3
        q = c - p0
        push!(vals, (atan(dot(q, Fv[:, 2]), dot(q, Fv[:, 1])), fi))
    end
    sort!(vals, by=x -> x[1])
    return [x[2] for x in vals]
end

function main()
    mesh = generate_icosphere(1.0, 2)
    geom = compute_geometry(mesh)

    vid = 1
    cycle = ordered_faces_around_vertex(mesh, geom, vid)
    hol = holonomy_along_cycle(mesh, geom, cycle)

    @printf("cycle length = %d faces\n", length(cycle))
    @printf("holonomy angle (rad) = %.6e\n", hol)

    # Face-path norm preservation check.
    v0 = SVector(0.3, -0.4)
    v1 = parallel_transport_along_face_path(v0, mesh, geom, cycle[1:min(end, 5)])
    @printf("|v0|=%.6e  |v1|=%.6e  rel.diff=%.3e\n",
            norm(v0), norm(v1), abs(norm(v0) - norm(v1)) / max(norm(v0), 1e-14))
end

main()
