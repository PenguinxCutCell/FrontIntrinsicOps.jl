# convergence/helpers_generators.jl
# Thin wrappers and ladder helpers for the package mesh generators.

"""
    sphere_uvsphere_ladder(R; levels=[(8,16),(12,24),(16,32),(24,48),(32,64)])

Return a vector of UV-sphere meshes at increasing resolution.
"""
function sphere_uvsphere_ladder(R::Float64;
        levels=[(8,16),(12,24),(16,32),(24,48),(32,64)])
    [(np, nt, generate_uvsphere(R, np, nt)) for (np, nt) in levels]
end

"""
    sphere_icosphere_ladder(R; levels=0:4)

Return a vector of icosphere meshes at increasing subdivision level.
"""
function sphere_icosphere_ladder(R::Float64; levels=0:4)
    [(lvl, generate_icosphere(R, lvl)) for lvl in levels]
end

"""
    torus_ladder(R, r; resolutions=[(8,16),(12,24),(16,32),(24,48)])

Return a vector of torus meshes at increasing resolution.
`resolutions` is a vector of (ntheta, nphi) pairs.
"""
function torus_ladder(R::Float64, r::Float64;
        resolutions=[(8,16),(12,24),(16,32),(24,48)])
    [(nt, np, generate_torus(R, r, nt, np)) for (nt, np) in resolutions]
end

"""
    circle_ladder(R; Ns=[16,32,64,128,256,512,1024])

Return a vector of circle meshes at increasing resolution.
"""
function circle_ladder(R::Float64; Ns=[16,32,64,128,256,512,1024])
    [(N, sample_circle(R, N)) for N in Ns]
end
