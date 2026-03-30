# geometry_curves.jl – Intrinsic geometric quantities for CurveMesh.
#
# Implements
# ----------
# * edge lengths
# * unit edge tangents
# * vertex dual lengths (barycentric: half the sum of the two adjacent edge lengths)
# * vertex normals (left-rotated averaged tangent, pointing inward for CCW curves)
# * discrete signed curvature (turning-angle / dual arc-length)
#
# Public entry point: `compute_geometry(mesh::CurveMesh) -> CurveGeometry`

"""
    compute_geometry(mesh::CurveMesh{T}) -> CurveGeometry{T}

Compute all intrinsic geometric quantities for a curve mesh and return a
`CurveGeometry` container.

For a closed curve sampled from a circle of radius R with N uniform vertices,
the quantities converge to:
- total length → 2πR
- enclosed area → πR²
- vertex curvature → 1/R

The signed curvature is positive for a counter-clockwise oriented circle
(standard mathematical orientation).
"""
function compute_geometry(mesh::CurveMesh{T}) :: CurveGeometry{T} where {T}
    pts   = mesh.points
    edges = mesh.edges
    nv    = length(pts)
    ne    = length(edges)

    # ── Edge lengths and unit tangents ────────────────────────────────────────
    edge_lengths  = Vector{T}(undef, ne)
    edge_tangents = Vector{SVector{2,T}}(undef, ne)
    for (ei, e) in enumerate(edges)
        v = pts[e[2]] - pts[e[1]]
        len = norm(v)
        edge_lengths[ei]  = len
        edge_tangents[ei] = len > eps(T) ? v / len : SVector{2,T}(one(T), zero(T))
    end

    # ── Vertex dual lengths (barycentric) ────────────────────────────────────
    # For each vertex, sum half the length of each adjacent edge.
    vertex_dual_lengths = zeros(T, nv)
    for (ei, e) in enumerate(edges)
        half = edge_lengths[ei] / 2
        vertex_dual_lengths[e[1]] += half
        vertex_dual_lengths[e[2]] += half
    end

    # ── Vertex normals and signed curvature ───────────────────────────────────
    # For each vertex, compute the turning angle between the incoming and
    # outgoing tangents, then divide by the dual length.
    # Normal = left-perpendicular of the averaged tangent.
    vertex_normals   = Vector{SVector{2,T}}(undef, nv)
    signed_curvature = Vector{T}(undef, nv)

    # Build: for each vertex, which edges are incoming/outgoing?
    # An edge (i→j): incoming for j, outgoing for i.
    incoming = [Int[] for _ in 1:nv]
    outgoing = [Int[] for _ in 1:nv]
    for (ei, e) in enumerate(edges)
        push!(outgoing[e[1]], ei)
        push!(incoming[e[2]], ei)
    end

    for vi in 1:nv
        has_in = !isempty(incoming[vi])
        has_out = !isempty(outgoing[vi])
        if !has_in && !has_out
            throw(ArgumentError("Curve vertex $vi is isolated and has no incident edges."))
        elseif !has_in
            # Open-curve start endpoint: use one-sided outgoing tangent.
            t_out = edge_tangents[outgoing[vi][1]]
            vertex_normals[vi] = SVector{2,T}(-t_out[2], t_out[1])
            signed_curvature[vi] = zero(T)
            continue
        elseif !has_out
            # Open-curve end endpoint: use one-sided incoming tangent.
            t_in = edge_tangents[incoming[vi][1]]
            vertex_normals[vi] = SVector{2,T}(-t_in[2], t_in[1])
            signed_curvature[vi] = zero(T)
            continue
        end
        ei_in  = incoming[vi][1]
        ei_out = outgoing[vi][1]
        t_in   = edge_tangents[ei_in]
        t_out  = edge_tangents[ei_out]
        # Turning angle = signed angle from t_in to t_out
        theta  = atan(cross2d(t_in, t_out), dot(t_in, t_out))
        dl     = vertex_dual_lengths[vi]
        kappa  = dl > eps(T) ? theta / dl : zero(T)
        signed_curvature[vi] = kappa
        # Vertex normal: left-perpendicular (rotate t_avg by +90°)
        t_avg = normalize_safe(t_in + t_out)
        vertex_normals[vi] = SVector{2,T}(-t_avg[2], t_avg[1])
    end

    return CurveGeometry{T}(
        edge_lengths,
        edge_tangents,
        vertex_dual_lengths,
        vertex_normals,
        signed_curvature,
    )
end

"""
    edge_midpoints(mesh::CurveMesh{T}) -> Vector{SVector{2,T}}

Return the midpoint of each edge in the curve mesh.
"""
function edge_midpoints(mesh::CurveMesh{T}) :: Vector{SVector{2,T}} where {T}
    return [T(0.5) * (mesh.points[e[1]] + mesh.points[e[2]]) for e in mesh.edges]
end
