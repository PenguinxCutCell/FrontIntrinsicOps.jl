# generators.jl – Deterministic mesh generators for convergence studies.
#
# Provides closed polygonal curves and triangulated closed surfaces suitable
# for systematic convergence studies.  All generators are self-contained and
# do not depend on external files.
#
# Curve generators
# ----------------
# * `sample_circle(R, N)` – regular N-gon approximation of a circle.
# * `sample_perturbed_circle(R, N; ϵ, mode, θ0)` – deterministic radial perturbation.
#
# Surface generators
# ------------------
# * `generate_uvsphere(R, nphi, ntheta)`     – UV-sphere (latitude-longitude).
# * `generate_icosphere(R, level)`           – icosahedron refined to level k.
# * `generate_torus(R, r, ntheta, nphi)`     – torus with major radius R, minor r.
# * `generate_ellipsoid(a, b, c, nphi, ntheta)` – axis-aligned ellipsoid.
# * `generate_perturbed_sphere(R, ε, k, nphi, ntheta)` – bumpy sphere.

"""
    single_marker_front(xΓ; inside_right=true) -> PointFront1D

Construct a one-marker 1-D front at `xΓ`.

Convention:
- `inside_right=true`  => inside is `x >= xΓ`
- `inside_right=false` => inside is `x <= xΓ`
"""
function single_marker_front(xΓ::Real; inside_right::Bool=true)
    T = float(typeof(xΓ))
    return PointFront1D(T[T(xΓ)], inside_right)
end

"""
    interval_front(xL, xR; interval_is_inside=true) -> PointFront1D

Construct a two-marker 1-D front with markers `(xL, xR)`.

Requires strict ordering `xL < xR`.
"""
function interval_front(xL::Real, xR::Real; interval_is_inside::Bool=true)
    T = promote_type(float(typeof(xL)), float(typeof(xR)))
    xLt = T(xL)
    xRt = T(xR)
    xLt < xRt || throw(ArgumentError("interval_front requires xL < xR, got xL=$xL and xR=$xR."))
    return PointFront1D(T[xLt, xRt], interval_is_inside)
end

# ─────────────────────────────────────────────────────────────────────────────
# Curve generators
# ─────────────────────────────────────────────────────────────────────────────

"""
    sample_circle(R::T, N::Int) -> CurveMesh{T}

Generate a closed polygonal approximation of a circle of radius `R` with `N`
uniformly spaced vertices.

The vertices are placed at angles `2πk/N` for `k = 0, …, N-1`, and the edges
close the loop.
"""
function sample_circle(R::T, N::Int) :: CurveMesh{T} where {T<:AbstractFloat}
    N >= 3 || error("sample_circle: N must be at least 3, got $N")
    pts   = [R * SVector{2,T}(cos(2T(π)*k/N), sin(2T(π)*k/N)) for k in 0:N-1]
    edges = [SVector{2,Int}(k, mod1(k+1, N)) for k in 1:N]
    return CurveMesh{T}(pts, edges)
end

"""
    sample_perturbed_circle(R::T, N::Int; ϵ::Real=0, mode::Int=4, θ0::Real=0) -> CurveMesh{T}

Generate a closed polygonal approximation of a perturbed circle:

`r(θ) = R * (1 + ϵ * cos(mode * (θ - θ0)))`, for `θ = 2πk/N`.

`ϵ` controls perturbation amplitude, `mode` controls harmonic fold, and `θ0`
rotates the perturbation pattern.
"""
function sample_perturbed_circle(
    R::T,
    N::Int;
    ϵ::Real=0,
    mode::Int=4,
    θ0::Real=0,
) :: CurveMesh{T} where {T<:AbstractFloat}
    N >= 3 || throw(ArgumentError("sample_perturbed_circle: N must be at least 3, got $N"))
    mode >= 0 || throw(ArgumentError("sample_perturbed_circle: mode must be >= 0, got $mode"))
    R > zero(T) || throw(ArgumentError("sample_perturbed_circle: R must be positive, got $R"))

    ϵT = convert(T, ϵ)
    abs(ϵT) < one(T) || throw(ArgumentError("sample_perturbed_circle: require |ϵ| < 1 to keep positive radius, got ϵ=$ϵ"))

    θ0T = convert(T, θ0)
    modeT = convert(T, mode)
    twoπ = 2T(π)
    pts = [begin
        θ = twoπ * k / N
        r = R * (one(T) + ϵT * cos(modeT * (θ - θ0T)))
        r * SVector{2,T}(cos(θ), sin(θ))
    end for k in 0:N-1]
    edges = [SVector{2,Int}(k, mod1(k + 1, N)) for k in 1:N]
    return CurveMesh{T}(pts, edges)
end

# ─────────────────────────────────────────────────────────────────────────────
# Surface generators
# ─────────────────────────────────────────────────────────────────────────────

"""
    generate_uvsphere(R::T, nphi::Int, ntheta::Int) -> SurfaceMesh{T}

Generate a UV-sphere (latitude-longitude) of radius `R`.

Parameters
----------
- `nphi`   – number of latitude bands (parallels; ≥ 2).
- `ntheta` – number of longitude sectors (meridians; ≥ 3).

The mesh has `(nphi-1)*ntheta + 2` vertices and `(nphi-2)*ntheta*2 + 2*ntheta`
faces.  The south pole is at `(0,0,-R)` and the north pole at `(0,0,+R)`.
Faces are oriented with outward normals.
"""
function generate_uvsphere(R::T, nphi::Int, ntheta::Int) :: SurfaceMesh{T} where {T<:AbstractFloat}
    nphi   >= 2 || error("generate_uvsphere: nphi must be >= 2")
    ntheta >= 3 || error("generate_uvsphere: ntheta must be >= 3")

    pts   = SVector{3,T}[]
    faces = SVector{3,Int}[]

    # South pole
    push!(pts, SVector{3,T}(0, 0, -R))

    # Interior latitude rings
    for i in 1:(nphi-1)
        phi = -T(π)/2 + i * T(π) / nphi
        for j in 0:(ntheta-1)
            theta = j * 2T(π) / ntheta
            push!(pts, SVector{3,T}(R*cos(phi)*cos(theta),
                                    R*cos(phi)*sin(theta),
                                    R*sin(phi)))
        end
    end

    # North pole
    push!(pts, SVector{3,T}(0, 0, +R))

    south = 1
    north = length(pts)

    # Bottom cap (south pole fans)
    for j in 0:(ntheta-1)
        v1 = 2 + j
        v2 = 2 + mod(j+1, ntheta)
        push!(faces, SVector{3,Int}(south, v2, v1))
    end

    # Interior quads (split into two triangles each)
    for i in 1:(nphi-2)
        for j in 0:(ntheta-1)
            v00 = 2 + (i-1)*ntheta + j
            v01 = 2 + (i-1)*ntheta + mod(j+1, ntheta)
            v10 = 2 + i*ntheta + j
            v11 = 2 + i*ntheta + mod(j+1, ntheta)
            push!(faces, SVector{3,Int}(v00, v01, v11))
            push!(faces, SVector{3,Int}(v00, v11, v10))
        end
    end

    # Top cap (north pole fans)
    base = 2 + (nphi-2)*ntheta
    for j in 0:(ntheta-1)
        v1 = base + j
        v2 = base + mod(j+1, ntheta)
        push!(faces, SVector{3,Int}(north, v1, v2))
    end

    return SurfaceMesh{T}(pts, faces)
end

# ─────────────────────────────────────────────────────────────────────────────
# Icosphere (icosahedron + subdivision)
# ─────────────────────────────────────────────────────────────────────────────

"""
    generate_icosphere(R::T, level::Int) -> SurfaceMesh{T}

Generate an icosphere of radius `R` at refinement `level`.

Starting from a regular icosahedron (level 0, 12 vertices, 20 faces), each
triangle is recursively subdivided into 4 sub-triangles and all vertices are
projected back onto the sphere.

Level sizes:
- level 0: 12 vertices,  20 faces
- level 1: 42 vertices,  80 faces
- level 2: 162 vertices, 320 faces
- level 3: 642 vertices, 1280 faces
- level 4: 2562 vertices, 5120 faces
"""
function generate_icosphere(R::T, level::Int) :: SurfaceMesh{T} where {T<:AbstractFloat}
    level >= 0 || error("generate_icosphere: level must be >= 0")

    # ---- Icosahedron vertices ------------------------------------------------
    phi = (1 + sqrt(T(5))) / 2  # golden ratio
    raw_verts = SVector{3,T}[
        (-1,  phi, 0), (1,  phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1,  phi), (0,  1,  phi), (0, -1, -phi), (0,  1, -phi),
        ( phi, 0, -1), ( phi, 0,  1), (-phi, 0, -1), (-phi, 0,  1),
    ]
    pts = [normalize_safe(SVector{3,T}(v)) * R for v in raw_verts]

    # ---- Icosahedron faces (1-indexed) ---------------------------------------
    faces = SVector{3,Int}[
        (1,12,6),(1,6,2),(1,2,8),(1,8,11),(1,11,12),
        (2,6,10),(6,12,5),(12,11,3),(11,8,7),(8,2,9),
        (4,10,5),(4,5,3),(4,3,7),(4,7,9),(4,9,10),
        (5,10,6),(3,5,12),(7,3,11),(9,7,8),(10,9,2),
    ]

    # ---- Subdivision ---------------------------------------------------------
    for _ in 1:level
        pts, faces = _icosphere_subdivide(pts, faces, R)
    end

    return SurfaceMesh{T}(pts, faces)
end

function _icosphere_subdivide(
        pts   :: Vector{SVector{3,T}},
        faces :: Vector{SVector{3,Int}},
        R     :: T,
) where {T}
    new_pts   = copy(pts)
    new_faces = SVector{3,Int}[]
    midpoints = Dict{Tuple{Int,Int},Int}()

    function get_mid(i::Int, j::Int)
        key = i < j ? (i, j) : (j, i)
        if haskey(midpoints, key)
            return midpoints[key]
        end
        mid = normalize_safe(new_pts[i] + new_pts[j]) * R
        push!(new_pts, mid)
        idx = length(new_pts)
        midpoints[key] = idx
        return idx
    end

    for face in faces
        a, b, c = face[1], face[2], face[3]
        mab = get_mid(a, b)
        mbc = get_mid(b, c)
        mca = get_mid(c, a)
        push!(new_faces, SVector{3,Int}(a, mab, mca))
        push!(new_faces, SVector{3,Int}(b, mbc, mab))
        push!(new_faces, SVector{3,Int}(c, mca, mbc))
        push!(new_faces, SVector{3,Int}(mab, mbc, mca))
    end

    return new_pts, new_faces
end

# ─────────────────────────────────────────────────────────────────────────────
# Torus generator
# ─────────────────────────────────────────────────────────────────────────────

"""
    generate_torus(R::T, r::T, ntheta::Int, nphi::Int) -> SurfaceMesh{T}

Generate a triangulated torus with major radius `R` (from torus centre to tube
centre) and minor radius `r` (tube radius).

Parameters
----------
- `ntheta` – number of segments around the tube (poloidal direction; ≥ 3).
- `nphi`   – number of segments around the torus hole (toroidal direction; ≥ 3).

Parametrisation:
  x(phi, theta) = (R + r cos theta) cos phi
  y(phi, theta) = (R + r cos theta) sin phi
  z(phi, theta) = r sin theta

Exact geometry
--------------
- Area:   A = 4 pi^2 R r
- Volume: V = 2 pi^2 R r^2
- Gaussian curvature: K(theta) = cos(theta) / (r (R + r cos theta))
- Euler characteristic: chi = 0 (torus)
"""
function generate_torus(
        R      :: T,
        r      :: T,
        ntheta :: Int,
        nphi   :: Int,
) :: SurfaceMesh{T} where {T<:AbstractFloat}
    R > 0 || error("generate_torus: R must be positive")
    r > 0 || error("generate_torus: r must be positive")
    r < R || error("generate_torus: r must be less than R (tube must not self-intersect)")
    ntheta >= 3 || error("generate_torus: ntheta must be >= 3")
    nphi   >= 3 || error("generate_torus: nphi must be >= 3")

    pts   = SVector{3,T}[]
    faces = SVector{3,Int}[]

    # Vertex grid: index (i,j) -> i*ntheta + j + 1  (1-based)
    # i = 0..nphi-1  (phi direction, around torus hole)
    # j = 0..ntheta-1 (theta direction, around tube)
    for i in 0:(nphi-1)
        phi = 2T(π) * i / nphi
        for j in 0:(ntheta-1)
            theta = 2T(π) * j / ntheta
            x = (R + r*cos(theta)) * cos(phi)
            y = (R + r*cos(theta)) * sin(phi)
            z = r * sin(theta)
            push!(pts, SVector{3,T}(x, y, z))
        end
    end

    vidx(i, j) = mod(i, nphi) * ntheta + mod(j, ntheta) + 1

    for i in 0:(nphi-1)
        for j in 0:(ntheta-1)
            v00 = vidx(i,   j  )
            v10 = vidx(i+1, j  )
            v01 = vidx(i,   j+1)
            v11 = vidx(i+1, j+1)
            # Outward normal: for increasing phi,theta the outward direction
            # points away from the tube axis.  The two triangles per quad:
            push!(faces, SVector{3,Int}(v00, v10, v11))
            push!(faces, SVector{3,Int}(v00, v11, v01))
        end
    end

    return SurfaceMesh{T}(pts, faces)
end

# ─────────────────────────────────────────────────────────────────────────────
# Ellipsoid generator
# ─────────────────────────────────────────────────────────────────────────────

"""
    generate_ellipsoid(a::T, b::T, c::T, nphi::Int, ntheta::Int) -> SurfaceMesh{T}

Generate a triangulated ellipsoid with semi-axes `a`, `b`, `c` using a
UV-sphere topology.

Parametrisation:
  x(φ, θ) = a cos(φ) cos(θ)
  y(φ, θ) = b cos(φ) sin(θ)
  z(φ, θ) = c sin(φ)

For `a = b = c = R` this reduces to a sphere of radius R.

Parameters
----------
- `a`, `b`, `c` – semi-axis lengths (all > 0).
- `nphi`        – number of latitude bands (≥ 2).
- `ntheta`      – number of longitude sectors (≥ 3).
"""
function generate_ellipsoid(
        a      :: T,
        b      :: T,
        c      :: T,
        nphi   :: Int,
        ntheta :: Int,
) :: SurfaceMesh{T} where {T<:AbstractFloat}
    a > 0 || error("generate_ellipsoid: a must be positive")
    b > 0 || error("generate_ellipsoid: b must be positive")
    c > 0 || error("generate_ellipsoid: c must be positive")
    nphi   >= 2 || error("generate_ellipsoid: nphi must be >= 2")
    ntheta >= 3 || error("generate_ellipsoid: ntheta must be >= 3")

    pts   = SVector{3,T}[]
    faces = SVector{3,Int}[]

    # South pole
    push!(pts, SVector{3,T}(zero(T), zero(T), -c))

    # Interior latitude rings
    for i in 1:(nphi-1)
        phi = -T(π)/2 + i * T(π) / nphi
        for j in 0:(ntheta-1)
            theta = j * 2T(π) / ntheta
            push!(pts, SVector{3,T}(a * cos(phi) * cos(theta),
                                    b * cos(phi) * sin(theta),
                                    c * sin(phi)))
        end
    end

    # North pole
    push!(pts, SVector{3,T}(zero(T), zero(T), c))

    south = 1
    north = length(pts)

    # Bottom cap
    for j in 0:(ntheta-1)
        v1 = 2 + j
        v2 = 2 + mod(j+1, ntheta)
        push!(faces, SVector{3,Int}(south, v2, v1))
    end

    # Interior quads
    for i in 1:(nphi-2)
        for j in 0:(ntheta-1)
            v00 = 2 + (i-1)*ntheta + j
            v01 = 2 + (i-1)*ntheta + mod(j+1, ntheta)
            v10 = 2 + i*ntheta + j
            v11 = 2 + i*ntheta + mod(j+1, ntheta)
            push!(faces, SVector{3,Int}(v00, v01, v11))
            push!(faces, SVector{3,Int}(v00, v11, v10))
        end
    end

    # Top cap
    base = 2 + (nphi-2)*ntheta
    for j in 0:(ntheta-1)
        v1 = base + j
        v2 = base + mod(j+1, ntheta)
        push!(faces, SVector{3,Int}(north, v1, v2))
    end

    return SurfaceMesh{T}(pts, faces)
end

# ─────────────────────────────────────────────────────────────────────────────
# Perturbed (bumpy) sphere generator
# ─────────────────────────────────────────────────────────────────────────────

"""
    generate_perturbed_sphere(R::T, ε::T, k::Int, nphi::Int, ntheta::Int)
        -> SurfaceMesh{T}

Generate a "bumpy sphere" — a sphere of radius `R` perturbed by a smooth
radial bump of amplitude `ε` and mode number `k`.

The radial map is:
    r(φ, θ) = R * (1 + ε cos(k φ) cos(k θ))

For ε = 0 this reduces exactly to a sphere.  Small |ε| < 1 gives a smooth
closed surface topologically equivalent to a sphere, useful for verifying
that PDE solvers work on non-spherically-symmetric geometries.

Parameters
----------
- `R`      – mean radius (> 0).
- `ε`      – perturbation amplitude; keep |ε| < 1 for a valid embedding.
- `k`      – perturbation mode number (≥ 1).
- `nphi`   – number of latitude bands (≥ 2).
- `ntheta` – number of longitude sectors (≥ 3).
"""
function generate_perturbed_sphere(
        R      :: T,
        ε      :: T,
        k      :: Int,
        nphi   :: Int,
        ntheta :: Int,
) :: SurfaceMesh{T} where {T<:AbstractFloat}
    R > 0 || error("generate_perturbed_sphere: R must be positive")
    abs(ε) < one(T) || error("generate_perturbed_sphere: |ε| must be < 1")
    k >= 1 || error("generate_perturbed_sphere: k must be >= 1")
    nphi   >= 2 || error("generate_perturbed_sphere: nphi must be >= 2")
    ntheta >= 3 || error("generate_perturbed_sphere: ntheta must be >= 3")

    pts   = SVector{3,T}[]
    faces = SVector{3,Int}[]

    # Helper: radial map
    r_map(phi, theta) = R * (one(T) + ε * cos(T(k) * phi) * cos(T(k) * theta))

    # South pole
    phi_s = -T(π)/2
    rs    = r_map(phi_s, zero(T))
    push!(pts, SVector{3,T}(zero(T), zero(T), -abs(rs)))

    # Interior latitude rings
    for i in 1:(nphi-1)
        phi = -T(π)/2 + i * T(π) / nphi
        for j in 0:(ntheta-1)
            theta = j * 2T(π) / ntheta
            r     = r_map(phi, theta)
            push!(pts, SVector{3,T}(r * cos(phi) * cos(theta),
                                    r * cos(phi) * sin(theta),
                                    r * sin(phi)))
        end
    end

    # North pole
    phi_n = T(π)/2
    rn    = r_map(phi_n, zero(T))
    push!(pts, SVector{3,T}(zero(T), zero(T), abs(rn)))

    south = 1
    north = length(pts)

    # Bottom cap
    for j in 0:(ntheta-1)
        v1 = 2 + j
        v2 = 2 + mod(j+1, ntheta)
        push!(faces, SVector{3,Int}(south, v2, v1))
    end

    # Interior quads
    for i in 1:(nphi-2)
        for j in 0:(ntheta-1)
            v00 = 2 + (i-1)*ntheta + j
            v01 = 2 + (i-1)*ntheta + mod(j+1, ntheta)
            v10 = 2 + i*ntheta + j
            v11 = 2 + i*ntheta + mod(j+1, ntheta)
            push!(faces, SVector{3,Int}(v00, v01, v11))
            push!(faces, SVector{3,Int}(v00, v11, v10))
        end
    end

    # Top cap
    base = 2 + (nphi-2)*ntheta
    for j in 0:(ntheta-1)
        v1 = base + j
        v2 = base + mod(j+1, ntheta)
        push!(faces, SVector{3,Int}(north, v1, v2))
    end

    return SurfaceMesh{T}(pts, faces)
end
