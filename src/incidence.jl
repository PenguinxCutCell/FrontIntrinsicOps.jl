# incidence.jl – Topological (metric-free) incidence matrices d₀ and d₁.
#
# Conventions
# -----------
# For a directed edge e = (i → j) with canonical form (min,max):
#   (d₀ u)[e] = u[j] − u[i]
# For a face f with oriented boundary ∂f = e₀ + e₁ + e₂:
#   (d₁ α)[f] = ±α[e₀] ± α[e₁] ± α[e₂]
# with the sign being +1 if the face traverses the canonical edge forward.
#
# Both matrices are sparse.  The identity d₁ d₀ = 0 (mod 2) holds exactly
# for closed oriented triangulated surfaces.

# ─────────────────────────────────────────────────────────────────────────────
# Curve: d₀  (V → E)
# ─────────────────────────────────────────────────────────────────────────────

"""
    incidence_0(mesh::CurveMesh{T}) -> SparseMatrixCSC{T,Int}

Build the vertex-to-edge incidence (coboundary) matrix d₀ for a curve.

Size: `(nE × nV)` where `nE = length(mesh.edges)`, `nV = length(mesh.points)`.

For edge `e = (i → j)`:
    `d₀[e, j] = +1`,  `d₀[e, i] = -1`.

This maps a vertex 0-cochain `u` to the edge 1-cochain `(d₀ u)[e] = u[j] - u[i]`.
"""
function incidence_0(mesh::CurveMesh{T}) :: SparseMatrixCSC{T,Int} where {T}
    nv = length(mesh.points)
    ne = length(mesh.edges)
    I  = Int[]
    J  = Int[]
    V  = T[]
    for (ei, e) in enumerate(mesh.edges)
        i, j = e[1], e[2]
        push!(I, ei); push!(J, i); push!(V, -one(T))
        push!(I, ei); push!(J, j); push!(V, +one(T))
    end
    return sparse(I, J, V, ne, nv)
end

# ─────────────────────────────────────────────────────────────────────────────
# Surface: d₀  (V → E)  and  d₁  (E → F)
# ─────────────────────────────────────────────────────────────────────────────

"""
    incidence_0(mesh::SurfaceMesh{T}) -> SparseMatrixCSC{T,Int}

Build the vertex-to-edge incidence (coboundary) matrix d₀ for a surface mesh.

Size: `(nE × nV)`.

For each unique (unoriented) canonical edge `(i,j)` with `i < j`:
    `d₀[e, j] = +1`,  `d₀[e, i] = -1`.
"""
function incidence_0(mesh::SurfaceMesh{T}) :: SparseMatrixCSC{T,Int} where {T}
    topo = build_topology(mesh)
    nv   = length(mesh.points)
    ne   = length(topo.edges)
    I    = Int[]
    J    = Int[]
    V    = T[]
    for (ei, e) in enumerate(topo.edges)
        i, j = e[1], e[2]   # canonical: i < j
        push!(I, ei); push!(J, i); push!(V, -one(T))
        push!(I, ei); push!(J, j); push!(V, +one(T))
    end
    return sparse(I, J, V, ne, nv)
end

"""
    incidence_1(mesh::SurfaceMesh{T}) -> SparseMatrixCSC{T,Int}

Build the edge-to-face incidence (coboundary) matrix d₁ for a surface mesh.

Size: `(nF × nE)`.

For face `f` and its k-th edge `eₖ` with orientation sign `sₖ ∈ {+1,-1}`:
    `d₁[f, eₖ] = sₖ`.

For a closed oriented mesh the identity `d₁ * d₀ == 0` holds to machine
precision.
"""
function incidence_1(mesh::SurfaceMesh{T}) :: SparseMatrixCSC{T,Int} where {T}
    topo = build_topology(mesh)
    nf   = length(mesh.faces)
    ne   = length(topo.edges)
    I    = Int[]
    J    = Int[]
    V    = T[]
    for (fi, fe) in enumerate(topo.face_edges)
        fs = topo.face_edge_signs[fi]
        for k in 1:3
            push!(I, fi); push!(J, fe[k]); push!(V, T(fs[k]))
        end
    end
    return sparse(I, J, V, nf, ne)
end
