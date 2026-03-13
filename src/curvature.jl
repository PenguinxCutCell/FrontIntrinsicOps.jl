# curvature.jl – Discrete curvature computations.
#
# For curves  : signed curvature (already in CurveGeometry from geometry_curves.jl)
# For surfaces: mean curvature normal, mean curvature scalar, Gaussian curvature
#
# The mean-curvature normal is obtained from the discrete identity:
#   Hₙ(v) = (1/2) (L x)(v)
# where L is the scalar Laplace–Beltrami and x is the coordinate embedding.
# The sign convention here gives an inward-pointing mean-curvature normal
# for a convex closed surface oriented with outward normals.

"""
    curvature(mesh::CurveMesh{T}, geom::CurveGeometry{T}) -> Vector{T}

Return the discrete signed curvature at each vertex of the curve.

This is the field `geom.signed_curvature` (computed in `compute_geometry`);
this function is the public API entry point for curvature.
"""
function curvature(mesh::CurveMesh{T}, geom::CurveGeometry{T}) :: Vector{T} where {T}
    return geom.signed_curvature
end

"""
    mean_curvature_normal(mesh::SurfaceMesh{T}, geom::SurfaceGeometry{T}, dec::SurfaceDEC{T})
        -> Vector{SVector{3,T}}

Compute the discrete mean-curvature normal vector at each vertex.

Convention
----------
The mean-curvature normal is defined as

    Hₙ(v) = (1 / (2 Aᵥ)) Σⱼ wᵢⱼ (pⱼ − pᵢ)

which equals `(1/2) (L p)(v)` coordinate-by-coordinate, where `L` is the
scalar Laplace–Beltrami operator and `p` the coordinate embedding.

For a sphere of radius R oriented outward, `‖Hₙ‖ → 1/R` at interior vertices.

Note
----
This function assembles the mean-curvature normal but does *not* mutate
`geom`.  Use `compute_curvature!` to store results back into a `SurfaceGeometry`.
"""
function mean_curvature_normal(
        mesh::SurfaceMesh{T},
        geom::SurfaceGeometry{T},
        dec::SurfaceDEC{T},
) :: Vector{SVector{3,T}} where {T}
    nv = length(mesh.points)
    Lx = dec.lap0 * [p[1] for p in mesh.points]
    Ly = dec.lap0 * [p[2] for p in mesh.points]
    Lz = dec.lap0 * [p[3] for p in mesh.points]
    # H·n = (1/2) L p  (using the sign convention L = ⋆₀⁻¹ d₀ᵀ ⋆₁ d₀)
    return [SVector{3,T}(Lx[i]/2, Ly[i]/2, Lz[i]/2) for i in 1:nv]
end

"""
    mean_curvature(mesh::SurfaceMesh{T}, geom::SurfaceGeometry{T}, dec::SurfaceDEC{T})
        -> Vector{T}

Compute the discrete scalar mean curvature at each vertex.

The mean curvature scalar is defined as

    H(v) = ‖Hₙ(v)‖ × sign(Hₙ · n̂)

where `Hₙ` is the mean-curvature normal vector and `n̂` is the outward
unit normal at `v`.  The result is positive for convex regions (sphere).
"""
function mean_curvature(
        mesh::SurfaceMesh{T},
        geom::SurfaceGeometry{T},
        dec::SurfaceDEC{T},
) :: Vector{T} where {T}
    Hn = mean_curvature_normal(mesh, geom, dec)
    nv = length(mesh.points)
    H  = Vector{T}(undef, nv)
    for i in 1:nv
        magnitude = norm(Hn[i])
        # Sign: positive if Hn points in the same direction as the inward normal
        # (i.e., opposite to the outward vertex normal)
        sgn = dot(Hn[i], geom.vertex_normals[i]) > 0 ? one(T) : -one(T)
        H[i] = magnitude * sgn
    end
    return H
end

"""
    gaussian_curvature(mesh::SurfaceMesh{T}, geom::SurfaceGeometry{T}) -> Vector{T}

Compute the discrete Gaussian curvature at each vertex using the angle-defect
formula:

    K(v) = (2π − Σᶠ θ_f) / Aᵥ

where `θ_f` is the interior angle of face `f` at vertex `v`, and `Aᵥ` is the
barycentric dual area at `v`.  For a closed orientable surface,
Σᵥ K(v) Aᵥ = 2π χ (Gauss–Bonnet).
"""
function gaussian_curvature(
        mesh::SurfaceMesh{T},
        geom::SurfaceGeometry{T},
) :: Vector{T} where {T}
    pts   = mesh.points
    faces = mesh.faces
    nv    = length(pts)

    angle_sum = zeros(T, nv)
    for face in faces
        a, b, c = pts[face[1]], pts[face[2]], pts[face[3]]
        # Angles at each vertex of the triangle
        ab = b - a; ac = c - a
        ba = a - b; bc = c - b
        ca = a - c; cb = b - c
        θ_a = acos(clamp(dot(normalize_safe(ab), normalize_safe(ac)), -one(T), one(T)))
        θ_b = acos(clamp(dot(normalize_safe(ba), normalize_safe(bc)), -one(T), one(T)))
        θ_c = acos(clamp(dot(normalize_safe(ca), normalize_safe(cb)), -one(T), one(T)))
        angle_sum[face[1]] += θ_a
        angle_sum[face[2]] += θ_b
        angle_sum[face[3]] += θ_c
    end

    K = Vector{T}(undef, nv)
    for vi in 1:nv
        da = geom.vertex_dual_areas[vi]
        K[vi] = da > eps(T) ? (2*T(π) - angle_sum[vi]) / da : zero(T)
    end
    return K
end

"""
    compute_curvature!(geom::SurfaceGeometry, mesh, dec)

Compute mean-curvature normal, mean curvature, and Gaussian curvature and
store them in a *new* `SurfaceGeometry` (structs are immutable; this returns
the updated struct).
"""
function compute_curvature(
        mesh::SurfaceMesh{T},
        geom::SurfaceGeometry{T},
        dec::SurfaceDEC{T},
) :: SurfaceGeometry{T} where {T}
    Hn = mean_curvature_normal(mesh, geom, dec)
    H  = mean_curvature(mesh, geom, dec)
    K  = gaussian_curvature(mesh, geom)
    return SurfaceGeometry{T}(
        geom.face_normals,
        geom.face_areas,
        geom.edge_lengths,
        geom.vertex_dual_areas,
        geom.vertex_normals,
        Hn, H, K,
    )
end
