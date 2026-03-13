# checks.jl – Mesh and DEC diagnostics.

"""
    check_mesh(mesh::CurveMesh) -> NamedTuple

Run structural diagnostics on a curve mesh and return a named tuple report.

Fields of the returned tuple
-----------------------------
- `n_vertices :: Int`
- `n_edges    :: Int`
- `closed     :: Bool`  – every vertex has exactly 2 incident edges
- `manifold   :: Bool`  – every vertex has ≤ 2 incident edges
- `warnings   :: Vector{String}`
"""
function check_mesh(mesh::CurveMesh) :: NamedTuple
    nv = length(mesh.points)
    ne = length(mesh.edges)
    ve = vertex_to_edges(mesh)
    degrees = [length(v) for v in ve]
    closed  = all(d == 2 for d in degrees)
    manifold = all(d <= 2 for d in degrees)
    warnings = String[]
    if !closed
        n_open = count(d != 2 for d in degrees)
        push!(warnings, "Curve has $n_open non-degree-2 vertices (not closed).")
    end
    if !manifold
        n_bad = count(d > 2 for d in degrees)
        push!(warnings, "Curve has $n_bad vertices with degree > 2 (non-manifold).")
    end
    return (
        n_vertices = nv,
        n_edges    = ne,
        closed     = closed,
        manifold   = manifold,
        warnings   = warnings,
    )
end

"""
    check_mesh(mesh::SurfaceMesh) -> NamedTuple

Run structural diagnostics on a surface mesh and return a named tuple report.

Fields of the returned tuple
-----------------------------
- `n_vertices          :: Int`
- `n_edges             :: Int`
- `n_faces             :: Int`
- `closed              :: Bool`
- `manifold            :: Bool`
- `consistent_orientation :: Bool`
- `euler_characteristic :: Int`   – V − E + F (should be 2 for sphere-like)
- `warnings            :: Vector{String}`
"""
function check_mesh(mesh::SurfaceMesh) :: NamedTuple
    nv   = length(mesh.points)
    nf   = length(mesh.faces)
    topo = build_topology(mesh)
    ne   = length(topo.edges)

    edge_valences = [length(ef) for ef in topo.edge_faces]
    closed   = all(v == 2 for v in edge_valences)
    manifold = all(v <= 2 for v in edge_valences)
    consistent_orientation = has_consistent_orientation(mesh)

    χ = nv - ne + nf  # Euler characteristic

    warnings = String[]
    if !closed
        n_bnd = count(v == 1 for v in edge_valences)
        push!(warnings, "Surface has $n_bnd boundary edges (not closed).")
    end
    if !manifold
        n_bad = count(v > 2 for v in edge_valences)
        push!(warnings, "Surface has $n_bad non-manifold edges (> 2 adjacent faces).")
    end
    if !consistent_orientation
        push!(warnings, "Surface has inconsistently oriented faces.")
    end

    return (
        n_vertices              = nv,
        n_edges                 = ne,
        n_faces                 = nf,
        closed                  = closed,
        manifold                = manifold,
        consistent_orientation  = consistent_orientation,
        euler_characteristic    = χ,
        warnings                = warnings,
    )
end

"""
    check_dec(mesh, geom, dec; tol=1e-10) -> NamedTuple

Run DEC-level diagnostics and return a named tuple report.

For a `SurfaceDEC`, checks include:
- `d1_d0_zero :: Bool` – whether `d₁ * d₀ ≈ 0` (max entry < `tol`).
- `lap_constant_nullspace :: Bool` – whether `L * ones ≈ 0` (max entry < `sqrt(tol)`).
- `star0_positive :: Bool` – all diagonal entries of ⋆₀ positive.
- `star1_positive :: Bool` – all diagonal entries of ⋆₁ positive (fails for
  obtuse triangulations).

For a `CurveDEC`:
- `lap_constant_nullspace :: Bool`
- `star0_positive :: Bool`
- `star1_positive :: Bool`
"""
function check_dec(
        mesh::SurfaceMesh{T},
        geom::SurfaceGeometry{T},
        dec::SurfaceDEC{T};
        tol::Float64 = 1e-10,
) :: NamedTuple where {T}
    nv = length(mesh.points)

    # d₁ d₀ = 0
    residual_d1d0 = maximum(abs, (dec.d1 * dec.d0))
    d1_d0_zero    = residual_d1d0 < tol

    # Constant nullspace
    ones_v = ones(T, nv)
    Lu     = dec.lap0 * ones_v
    lap_nullspace = maximum(abs, Lu) < sqrt(tol)

    # Positivity
    s0_diag = diag(dec.star0)
    s1_diag = diag(dec.star1)
    star0_pos = all(x > zero(T) for x in s0_diag)
    star1_pos = all(x > zero(T) for x in s1_diag)

    warnings = String[]
    d1_d0_zero   || push!(warnings, "d1*d0 != 0 (max residual = $residual_d1d0)")
    lap_nullspace || push!(warnings, "Laplace–Beltrami constant nullspace check failed.")
    star1_pos     || push!(warnings, "⋆₁ has non-positive entries (obtuse triangulation).")

    return (
        d1_d0_zero             = d1_d0_zero,
        d1_d0_max_residual     = residual_d1d0,
        lap_constant_nullspace = lap_nullspace,
        star0_positive         = star0_pos,
        star1_positive         = star1_pos,
        warnings               = warnings,
    )
end

function check_dec(
        mesh::CurveMesh{T},
        geom::CurveGeometry{T},
        dec::CurveDEC{T};
        tol::Float64 = 1e-10,
) :: NamedTuple where {T}
    nv    = length(mesh.points)
    ones_v = ones(T, nv)
    Lu    = dec.lap0 * ones_v
    lap_nullspace = maximum(abs, Lu) < sqrt(tol)
    s0_diag = diag(dec.star0)
    s1_diag = diag(dec.star1)
    star0_pos = all(x > zero(T) for x in s0_diag)
    star1_pos = all(x > zero(T) for x in s1_diag)

    warnings = String[]
    lap_nullspace || push!(warnings, "Laplace–Beltrami constant nullspace check failed.")

    return (
        lap_constant_nullspace = lap_nullspace,
        star0_positive         = star0_pos,
        star1_positive         = star1_pos,
        warnings               = warnings,
    )
end

"""
    euler_characteristic(mesh::SurfaceMesh) -> Int

Compute the Euler characteristic chi = V - E + F for a triangulated surface.

Standard values:
- Sphere (genus 0): chi = 2
- Torus  (genus 1): chi = 0
- Genus-g surface:  chi = 2 - 2g
"""
function euler_characteristic(mesh::SurfaceMesh) :: Int
    nv   = length(mesh.points)
    nf   = length(mesh.faces)
    topo = build_topology(mesh)
    ne   = length(topo.edges)
    return nv - ne + nf
end

"""
    gauss_bonnet_residual(mesh::SurfaceMesh{T}, geom::SurfaceGeometry{T}) -> T

Return the absolute residual of the discrete Gauss-Bonnet theorem:

    |int K dA - 2 pi chi|

where `int K dA` is the integrated discrete Gaussian curvature (angle-defect
sum) and `chi` is the Euler characteristic.  For a well-formed closed surface
this should be near machine precision.
"""
function gauss_bonnet_residual(
        mesh :: SurfaceMesh{T},
        geom :: SurfaceGeometry{T},
) :: T where {T}
    chi    = euler_characteristic(mesh)
    intK   = integrated_gaussian_curvature(mesh, geom)
    target = 2 * T(π) * chi
    return abs(intK - target)
end

"""
    star1_sign_report(dec::SurfaceDEC) -> NamedTuple

Return a report on the sign structure of the Hodge-star star1 diagonal.

Fields of the returned tuple
-----------------------------
- `n_entries     :: Int`   – total number of diagonal entries.
- `n_nonpositive :: Int`   – number of entries <= 0.
- `frac_nonpositive :: Float64` – fraction non-positive.
- `min_entry     :: T`     – minimum diagonal entry.
- `all_positive  :: Bool`  – true if all entries are strictly positive.
"""
function star1_sign_report(dec::SurfaceDEC{T}) :: NamedTuple where {T}
    d = diag(dec.star1)
    n    = length(d)
    n_np = count(x -> x <= zero(T), d)
    return (
        n_entries       = n,
        n_nonpositive   = n_np,
        frac_nonpositive = n_np / n,
        min_entry       = minimum(d),
        all_positive    = n_np == 0,
    )
end

"""
    compare_laplace_methods(mesh::SurfaceMesh{T}, geom::SurfaceGeometry{T}) -> NamedTuple

Assemble the scalar Laplace-Beltrami using both the DEC factored path and the
direct cotan path, then compare.

Fields of the returned tuple
-----------------------------
- `norm_inf    :: T`  – max absolute entry-wise difference: ||L_dec - L_cotan||_inf.
- `norm_frob   :: T`  – Frobenius norm of the difference.
- `dec_nullspace  :: Bool` – L_dec * ones approx 0 (tol = sqrt(eps)).
- `cotan_nullspace :: Bool` – L_cotan * ones approx 0.
- `max_dec_res  :: T` – max |L_dec * ones|.
- `max_cotan_res :: T` – max |L_cotan * ones|.
"""
function compare_laplace_methods(
        mesh :: SurfaceMesh{T},
        geom :: SurfaceGeometry{T},
) :: NamedTuple where {T}
    nv   = length(mesh.points)
    L_dec   = build_laplace_beltrami(mesh, geom; method=:dec)
    L_cotan = build_laplace_beltrami(mesh, geom; method=:cotan)
    diff = L_dec - L_cotan

    norm_inf  = maximum(abs, diff)
    norm_frob = sqrt(sum(x -> x^2, diff.nzval))

    tol = sqrt(eps(T))
    ones_v = ones(T, nv)
    res_dec   = L_dec   * ones_v
    res_cotan = L_cotan * ones_v
    max_dec   = maximum(abs, res_dec)
    max_cotan = maximum(abs, res_cotan)

    return (
        norm_inf         = norm_inf,
        norm_frob        = norm_frob,
        dec_nullspace    = max_dec   < tol,
        cotan_nullspace  = max_cotan < tol,
        max_dec_res      = max_dec,
        max_cotan_res    = max_cotan,
    )
end
