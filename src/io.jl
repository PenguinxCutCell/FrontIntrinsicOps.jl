# io.jl – Input/output for front meshes.
#
# Supported formats for v0.1
# --------------------------
# * STL (ASCII or binary) for triangulated 3-D surfaces via MeshIO + FileIO.
# * Simple CSV (one x,y per line) or a plain Julia vector of points for 2-D
#   curves.
#
# The `GeometryBasics.Mesh` object is used only inside this file; it must not
# leak beyond the IO boundary.

"""
    load_surface_stl(path) -> SurfaceMesh{Float64}

Load a triangulated surface from an STL file at `path` and return a
`SurfaceMesh{Float64}`.

The STL is read through `FileIO.load` / `MeshIO`.  The resulting
`GeometryBasics.Mesh` is converted to internal types immediately; no
`GeometryBasics` objects are returned to the caller.

Deduplication
-------------
STL files store vertex data per-face (each triangle lists its own three
vertices).  This function deduplicates vertices using exact floating-point
comparison; if your STL has slightly non-matching vertices due to rounding,
use the optional keyword `tol` (default `0.0` for exact match) to merge
vertices within distance `tol`.

Notes
-----
Only triangulated STL is supported in v0.1.
"""
function load_surface_stl(path::AbstractString; tol::Float64=0.0) :: SurfaceMesh{Float64}
    isfile(path) || error("File not found: $path")
    raw = FileIO.load(path)
    return _gb_mesh_to_surface(raw, tol)
end

# ─── Internal conversion helpers ──────────────────────────────────────────────

function _gb_mesh_to_surface(raw, tol::Float64) :: SurfaceMesh{Float64}
    # GeometryBasics.Mesh has .position and .faces (or similar accessor).
    # We use the coordinates() / faces() accessors which are stable across
    # GeometryBasics 0.4 and 0.5.
    gb_points = GeometryBasics.coordinates(raw)
    gb_faces  = GeometryBasics.faces(raw)

    # Build raw per-face point list.
    raw_pts = [SVector{3,Float64}(Float64(p[1]), Float64(p[2]), Float64(p[3]))
               for p in gb_points]

    # Build face list (1-indexed).
    face_list = SVector{3,Int}[]
    for f in gb_faces
        # GeometryBasics face indices may be 0-based (OffsetInteger) or 1-based.
        idx = Int.(f) .+ (minimum(Int.(f)) == 0 ? 1 : 0)
        push!(face_list, SVector{3,Int}(idx[1], idx[2], idx[3]))
    end

    if tol == 0.0
        # Exact deduplication via a dictionary.
        return _deduplicate_exact(raw_pts, face_list)
    else
        return _deduplicate_tol(raw_pts, face_list, tol)
    end
end

function _deduplicate_exact(
        raw_pts  :: Vector{SVector{3,Float64}},
        face_list :: Vector{SVector{3,Int}},
) :: SurfaceMesh{Float64}
    # Map from point → new index.
    index_map = Dict{SVector{3,Float64},Int}()
    new_pts   = SVector{3,Float64}[]
    for p in raw_pts
        if !haskey(index_map, p)
            push!(new_pts, p)
            index_map[p] = length(new_pts)
        end
    end
    new_faces = [SVector{3,Int}(index_map[raw_pts[f[1]]],
                                index_map[raw_pts[f[2]]],
                                index_map[raw_pts[f[3]]]) for f in face_list]
    return SurfaceMesh{Float64}(new_pts, new_faces)
end

function _deduplicate_tol(
        raw_pts   :: Vector{SVector{3,Float64}},
        face_list :: Vector{SVector{3,Int}},
        tol       :: Float64,
) :: SurfaceMesh{Float64}
    n = length(raw_pts)
    id = collect(1:n)
    for i in 1:n
        if id[i] == i                    # not yet merged
            for j in (i+1):n
                if norm(raw_pts[i] - raw_pts[j]) <= tol
                    id[j] = i
                end
            end
        end
    end
    # Compact the surviving representatives.
    remap = Dict{Int,Int}()
    new_pts = SVector{3,Float64}[]
    for i in 1:n
        root = id[i]
        if !haskey(remap, root)
            push!(new_pts, raw_pts[root])
            remap[root] = length(new_pts)
        end
        remap[i] = remap[root]
    end
    new_faces = [SVector{3,Int}(remap[f[1]], remap[f[2]], remap[f[3]])
                 for f in face_list]
    return SurfaceMesh{Float64}(new_pts, new_faces)
end

# ─────────────────────────────────────────────────────────────────────────────
# 2-D curve loading
# ─────────────────────────────────────────────────────────────────────────────

"""
    load_curve_csv(path; closed=true) -> CurveMesh{Float64}

Load a 2-D polygonal curve from a CSV file at `path`.

The file must have one vertex per line with two comma-separated numbers
(no header).  Lines beginning with `#` are treated as comments and skipped.

If `closed=true` (default) the last edge connects the last vertex back to
the first, forming a closed curve.

Example CSV format::

    0.0, 1.0
    -0.7071, 0.7071
    -1.0,  0.0
    …
"""
function load_curve_csv(path::AbstractString; closed::Bool=true) :: CurveMesh{Float64}
    isfile(path) || error("File not found: $path")
    pts = SVector{2,Float64}[]
    open(path, "r") do io
        for line in eachline(io)
            line = strip(line)
            isempty(line) && continue
            startswith(line, "#") && continue
            parts = split(line, ",")
            length(parts) >= 2 || error("Expected at least 2 columns, got: $line")
            x = parse(Float64, strip(parts[1]))
            y = parse(Float64, strip(parts[2]))
            push!(pts, SVector{2,Float64}(x, y))
        end
    end
    isempty(pts) && error("No points found in $path")
    return load_curve_points(pts; closed=closed)
end

"""
    load_curve_points(points; closed=true) -> CurveMesh{Float64}

Construct a `CurveMesh` from an ordered list of 2-D points.

`points` may be any `AbstractVector` whose elements are convertible to
`SVector{2,Float64}` (e.g., `Vector{Tuple{Float64,Float64}}` or a
`Vector{SVector{2,Float64}}`).

If `closed=true` (default) an edge from the last vertex to the first vertex
is appended automatically.
"""
function load_curve_points(points::AbstractVector; closed::Bool=true) :: CurveMesh{Float64}
    pts = [SVector{2,Float64}(Float64(p[1]), Float64(p[2])) for p in points]
    n   = length(pts)
    n >= 2 || error("A curve requires at least 2 points.")
    edges = SVector{2,Int}[]
    for i in 1:(n-1)
        push!(edges, SVector{2,Int}(i, i+1))
    end
    if closed
        push!(edges, SVector{2,Int}(n, 1))
    end
    return CurveMesh{Float64}(pts, edges)
end
