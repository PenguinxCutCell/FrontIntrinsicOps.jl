# open_surfaces.jl – Boundary detection and BCs for open / bounded surfaces.
#
# An "open surface" is a surface with boundary (e.g., a cap, a patch, or a
# triangulated disk).  Boundary edges are those adjacent to exactly one face.
# Boundary vertices are those incident to at least one boundary edge.
#
# This module provides:
# - Detection of boundary edges and vertices.
# - Dirichlet boundary condition application (modify system matrix and rhs).
# - Neumann (natural) boundary condition support (add boundary flux term to rhs).
# - A simple helper `boundary_mass_matrix` for the boundary 1-D integral.
#
# Note: the DEC operators (Laplacian, Hodge stars, mass matrix) in the rest of
# the package already work on open surfaces provided by `compute_geometry` and
# `build_dec`.  The boundary condition tools here let users solve BVPs.

# ─────────────────────────────────────────────────────────────────────────────
# Boundary detection
# ─────────────────────────────────────────────────────────────────────────────

"""
    detect_boundary_edges(topo::MeshTopology) -> Vector{Int}

Return the indices (into `topo.edges`) of boundary edges.

A boundary edge is any edge adjacent to exactly one face.
On a closed surface this returns an empty vector.
"""
function detect_boundary_edges(topo::MeshTopology) :: Vector{Int}
    bndry = Int[]
    for ei in eachindex(topo.edge_faces)
        length(topo.edge_faces[ei]) == 1 && push!(bndry, ei)
    end
    return bndry
end

"""
    detect_boundary_vertices(mesh::SurfaceMesh, topo::MeshTopology) -> Vector{Int}

Return the indices of boundary vertices.

A boundary vertex is one that is an endpoint of at least one boundary edge.
"""
function detect_boundary_vertices(
        :: SurfaceMesh,
        topo :: MeshTopology,
) :: Vector{Int}
    be = detect_boundary_edges(topo)
    bv = Set{Int}()
    for ei in be
        push!(bv, topo.edges[ei][1], topo.edges[ei][2])
    end
    return sort!(collect(bv))
end

"""
    is_open_surface(topo::MeshTopology) -> Bool

Return `true` if the surface has boundary edges (i.e., is not closed).
"""
function is_open_surface(topo::MeshTopology) :: Bool
    return any(ef -> length(ef) == 1, topo.edge_faces)
end

# ─────────────────────────────────────────────────────────────────────────────
# Dirichlet boundary conditions
# ─────────────────────────────────────────────────────────────────────────────

"""
    apply_dirichlet!(u, boundary_vertices, values)

Set the values of `u` at `boundary_vertices` to the corresponding entries of
`values`.  This is an in-place strong enforcement of Dirichlet data.

Parameters
----------
- `u`                 – vertex field to modify in-place.
- `boundary_vertices` – vector of vertex indices where Dirichlet data is imposed.
- `values`            – Dirichlet values at the boundary vertices (same length as
                        `boundary_vertices`, or a scalar broadcast to all).
"""
function apply_dirichlet!(
        u                  :: AbstractVector{T},
        boundary_vertices  :: AbstractVector{Int},
        values             :: Union{AbstractVector{T},T,Real},
) where {T}
    if values isa AbstractVector
        length(values) == length(boundary_vertices) ||
            error("apply_dirichlet!: length(values) ≠ length(boundary_vertices)")
        @inbounds for (k, vi) in enumerate(boundary_vertices)
            u[vi] = T(values[k])
        end
    else
        val = T(values)
        @inbounds for vi in boundary_vertices
            u[vi] = val
        end
    end
    return u
end

"""
    apply_dirichlet_to_system!(A, b, boundary_vertices, values)

Modify the linear system `A x = b` to enforce Dirichlet conditions at
`boundary_vertices` using the "row elimination" approach:
- Set row `i` of `A` to the identity row (zeros except A[i,i] = 1).
- Set `b[i]` to the Dirichlet value.

This is done in-place.  The resulting system is no longer symmetric.
For symmetric factorization use `apply_dirichlet_symmetric!` instead.

Parameters
----------
- `A`                 – sparse system matrix (modified in-place; must be mutable).
- `b`                 – right-hand side vector (modified in-place).
- `boundary_vertices` – indices where Dirichlet data is imposed.
- `values`            – Dirichlet values (same length or scalar).
"""
function apply_dirichlet_to_system!(
        A                  :: SparseMatrixCSC{T,Int},
        b                  :: AbstractVector{T},
        boundary_vertices  :: AbstractVector{Int},
        values             :: Union{AbstractVector{T},T,Real},
) where {T}
    nv = size(A, 1)
    for (k, vi) in enumerate(boundary_vertices)
        val = values isa AbstractVector ? T(values[k]) : T(values)
        # Zero row vi: iterate over all columns and set entry (vi, col) to zero.
        # In CSC format this requires scanning all columns.
        for col in 1:nv
            for ptr in A.colptr[col]:(A.colptr[col+1]-1)
                if A.rowval[ptr] == vi
                    A.nzval[ptr] = zero(T)
                end
            end
        end
        A[vi, vi] = one(T)
        b[vi] = val
    end
    return A, b
end

"""
    apply_dirichlet_symmetric!(A, b, boundary_vertices, values)

Enforce Dirichlet conditions while preserving symmetry of `A`:
1. For each Dirichlet vertex i with value g_i:
   - Move A[j,i] * g_i to rhs for all j ≠ i (update b[j] -= A[j,i] * g_i).
   - Zero row i and column i of A.
   - Set A[i,i] = 1, b[i] = g_i.

This preserves symmetry and is suitable for symmetric solvers.
"""
function apply_dirichlet_symmetric!(
        A                  :: SparseMatrixCSC{T,Int},
        b                  :: AbstractVector{T},
        boundary_vertices  :: AbstractVector{Int},
        values             :: Union{AbstractVector{T},T,Real},
) where {T}
    bv_set = Set(boundary_vertices)
    nv     = size(A, 1)

    # Collect Dirichlet values
    dvals = Dict{Int,T}()
    for (k, vi) in enumerate(boundary_vertices)
        dvals[vi] = values isa AbstractVector ? T(values[k]) : T(values)
    end

    # Step 1: update rhs b[j] -= A[j,i]*g_i for free dofs j
    # We iterate over columns i ∈ bv
    for (vi, gi) in dvals
        for ptr in A.colptr[vi]:(A.colptr[vi+1]-1)
            j    = A.rowval[ptr]
            j in bv_set && continue
            b[j] -= A.nzval[ptr] * gi
        end
    end

    # Step 2: zero rows and columns of Dirichlet dofs, set diagonal to 1
    for col in 1:nv
        for ptr in A.colptr[col]:(A.colptr[col+1]-1)
            row = A.rowval[ptr]
            if row in bv_set || col in bv_set
                A.nzval[ptr] = zero(T)
            end
        end
    end
    for (vi, gi) in dvals
        A[vi, vi] = one(T)
        b[vi]     = gi
    end

    return A, b
end

# ─────────────────────────────────────────────────────────────────────────────
# Neumann boundary conditions
# ─────────────────────────────────────────────────────────────────────────────

"""
    add_neumann_rhs!(b, mesh, geom, topo, g; boundary_edges=nothing)

Add the Neumann (natural) boundary contribution to the right-hand side `b`:

    b[v] += ∫_{∂Γ} g n̂·∇u ds   ≈  Σ_{e ∈ ∂Γ} g_e * |e| * (contribution to v)

For a piecewise-linear Neumann condition g (given at vertices), the weak form
contribution to vertex `v` from boundary edge e = (i, j) is:

    b[v] += g_v * |e| / 2   for v = i or v = j

where `|e|` is the edge length.

Parameters
----------
- `b`               – right-hand side vector to modify in-place.
- `g`               – Neumann flux values at vertices (length nV) or a scalar.
- `boundary_edges`  – indices of boundary edges; auto-detected if not provided.
"""
function add_neumann_rhs!(
        b             :: AbstractVector{T},
        mesh          :: SurfaceMesh{T},
        geom          :: SurfaceGeometry{T},
        topo          :: MeshTopology,
        g             :: Union{AbstractVector{T},T,Real};
        boundary_edges :: Union{AbstractVector{Int},Nothing} = nothing,
) where {T}
    be = boundary_edges !== nothing ? boundary_edges : detect_boundary_edges(topo)
    @inbounds for ei in be
        i, j = topo.edges[ei][1], topo.edges[ei][2]
        el   = geom.edge_lengths[ei]
        gi   = g isa AbstractVector ? T(g[i]) : T(g)
        gj   = g isa AbstractVector ? T(g[j]) : T(g)
        b[i] += gi * el / 2
        b[j] += gj * el / 2
    end
    return b
end

# ─────────────────────────────────────────────────────────────────────────────
# Boundary mass matrix (1-D integral along boundary)
# ─────────────────────────────────────────────────────────────────────────────

"""
    boundary_mass_matrix(mesh, geom, topo) -> SparseMatrixCSC{T,Int}

Assemble the boundary 1-D mass matrix for an open surface:
    M_b[i, j] = ∫_{∂Γ} φ_i φ_j ds

where φ_k are piecewise-linear hat functions on the boundary.

For each boundary edge e = (i, j) with length |e|:
    M_b[i,i] += |e|/3,   M_b[j,j] += |e|/3,   M_b[i,j] += |e|/6.

Returns a sparse matrix of size (nV × nV) that is supported only on
boundary vertices.
"""
function boundary_mass_matrix(
        mesh :: SurfaceMesh{T},
        geom :: SurfaceGeometry{T},
        topo :: MeshTopology,
) :: SparseMatrixCSC{T,Int} where {T}
    nv = length(mesh.points)
    be = detect_boundary_edges(topo)
    II = Int[]
    JJ = Int[]
    VV = T[]
    sizehint!(II, 3 * length(be))
    sizehint!(JJ, 3 * length(be))
    sizehint!(VV, 3 * length(be))
    @inbounds for ei in be
        i, j = topo.edges[ei][1], topo.edges[ei][2]
        el   = geom.edge_lengths[ei]
        push!(II, i); push!(JJ, i); push!(VV, el / 3)
        push!(II, j); push!(JJ, j); push!(VV, el / 3)
        push!(II, i); push!(JJ, j); push!(VV, el / 6)
        push!(II, j); push!(JJ, i); push!(VV, el / 6)
    end
    return sparse(II, JJ, VV, nv, nv)
end

# ─────────────────────────────────────────────────────────────────────────────
# Convenience: solve Poisson on open surface with Dirichlet BCs
# ─────────────────────────────────────────────────────────────────────────────

"""
    solve_open_surface_poisson(mesh, geom, dec, topo, f, boundary_vertices,
                               boundary_values; method=:dec)
        -> Vector{T}

Solve the Poisson problem on an open surface with Dirichlet boundary conditions:

    -ΔΓ u = f    in Γ (interior)
         u = g   on ∂Γ (boundary)

Uses row-elimination to enforce Dirichlet conditions and solves the resulting
sparse linear system.

Parameters
----------
- `f`                – right-hand side vertex field (interior source).
- `boundary_vertices`– vertex indices where Dirichlet data is imposed.
- `boundary_values`  – Dirichlet values at boundary vertices.
- `method`           – Laplace assembly method (`:dec` or `:cotan`).
"""
function solve_open_surface_poisson(
        mesh              :: SurfaceMesh{T},
        geom              :: SurfaceGeometry{T},
        dec               :: SurfaceDEC{T},
        topo              :: MeshTopology,
        f                 :: AbstractVector{T},
        boundary_vertices :: AbstractVector{Int},
        boundary_values   :: Union{AbstractVector{T},T,Real};
        method            :: Symbol = :dec,
) :: Vector{T} where {T}
    L = laplace_matrix(mesh, geom, dec; method=method)
    b = copy(f)
    A = copy(L)

    apply_dirichlet_to_system!(A, b, boundary_vertices, boundary_values)

    return Array(A) \ b
end
