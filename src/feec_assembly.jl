# feec_assembly.jl – Consistent Whitney mass/stiffness assembly and minimal FEEC solves.

# -----------------------------------------------------------------------------
# Sparse assembly helpers
# -----------------------------------------------------------------------------

@inline function _push_local3x3!(I::Vector{Int}, J::Vector{Int}, V::Vector{T}, ids, A) where {T}
    @inbounds for a in 1:3, b in 1:3
        push!(I, ids[a])
        push!(J, ids[b])
        push!(V, A[a, b])
    end
    return nothing
end

@inline function _push_local2x2!(I::Vector{Int}, J::Vector{Int}, V::Vector{T}, ids, A) where {T}
    @inbounds for a in 1:2, b in 1:2
        push!(I, ids[a])
        push!(J, ids[b])
        push!(V, A[a, b])
    end
    return nothing
end

function _lumped_inverse(M::SparseMatrixCSC{T,Int}) where {T<:AbstractFloat}
    m = vec(sum(M, dims=2))
    invm = [abs(mi) > eps(T) ? inv(mi) : zero(T) for mi in m]
    return spdiagm(0 => invm), m
end

function _augment_mean_zero_system(L::SparseMatrixCSC{T,Int}, w::AbstractVector{T}) where {T<:AbstractFloat}
    n = size(L, 1)
    size(L, 2) == n || throw(DimensionMismatch("L must be square."))
    length(w) == n || throw(DimensionMismatch("weights length $(length(w)) != n=$n"))

    Ii, Jj, Vv = findnz(L)
    I = Vector{Int}(undef, length(Ii) + 2n)
    J = Vector{Int}(undef, length(Jj) + 2n)
    V = Vector{T}(undef, length(Vv) + 2n)

    copyto!(I, 1, Ii, 1, length(Ii))
    copyto!(J, 1, Jj, 1, length(Jj))
    copyto!(V, 1, Vv, 1, length(Vv))

    off = length(Ii)
    @inbounds for i in 1:n
        I[off + i] = i
        J[off + i] = n + 1
        V[off + i] = w[i]
    end
    off2 = off + n
    @inbounds for i in 1:n
        I[off2 + i] = n + 1
        J[off2 + i] = i
        V[off2 + i] = w[i]
    end

    return sparse(I, J, V, n + 1, n + 1)
end

# -----------------------------------------------------------------------------
# Whitney mass matrices
# -----------------------------------------------------------------------------

"""
    assemble_whitney_mass0(mesh, geom) -> sparse matrix

Assemble the consistent lowest-order Whitney 0-form mass matrix.

Conventions
-----------
- Surface mesh: vertex-based linear Lagrange mass matrix on triangles.
- Curve mesh: vertex-based linear Lagrange mass matrix on segments.
"""
function assemble_whitney_mass0(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
) where {T<:AbstractFloat}
    nv = length(mesh.points)
    I = Int[]
    J = Int[]
    V = T[]

    @inbounds for (fi, face) in enumerate(mesh.faces)
        A = geom.face_areas[fi]
        loc = (A / 12) * @SMatrix [2 1 1; 1 2 1; 1 1 2]
        ids = (face[1], face[2], face[3])
        _push_local3x3!(I, J, V, ids, loc)
    end

    return sparse(I, J, V, nv, nv)
end

function assemble_whitney_mass0(
    mesh::CurveMesh{T},
    geom::CurveGeometry{T},
) where {T<:AbstractFloat}
    nv = length(mesh.points)
    I = Int[]
    J = Int[]
    V = T[]

    @inbounds for (ei, e) in enumerate(mesh.edges)
        ℓ = geom.edge_lengths[ei]
        loc = (ℓ / 6) * @SMatrix [2 1; 1 2]
        ids = (e[1], e[2])
        _push_local2x2!(I, J, V, ids, loc)
    end

    return sparse(I, J, V, nv, nv)
end

"""
    assemble_whitney_mass1(mesh, geom) -> sparse matrix

Assemble the consistent lowest-order Whitney 1-form mass matrix.

Conventions
-----------
- Surface mesh: edge-based Whitney-1 mass assembled per triangle with exact
  degree-2 barycentric quadrature.
- Curve mesh: edge-based 1-form mass with basis `w_e = 1/|e|` on each segment,
  yielding diagonal entries `1/|e|`.
"""
function assemble_whitney_mass1(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
) where {T<:AbstractFloat}
    topo = build_topology(mesh)
    ne = length(topo.edges)

    I = Int[]
    J = Int[]
    V = T[]

    # Degree-2 exact quadrature in barycentric coordinates.
    qλ = (
        SVector{3,T}(T(2//3), T(1//6), T(1//6)),
        SVector{3,T}(T(1//6), T(2//3), T(1//6)),
        SVector{3,T}(T(1//6), T(1//6), T(2//3)),
    )

    @inbounds for fi in 1:length(mesh.faces)
        tri = _triangle_geometry(mesh, geom, fi)
        fe = topo.face_edges[fi]
        fs = topo.face_edge_signs[fi]

        Mloc = zeros(T, 3, 3)
        for q in qλ
            b1 = eval_whitney1_local(1, q, tri)
            b2 = eval_whitney1_local(2, q, tri)
            b3 = eval_whitney1_local(3, q, tri)
            wq = tri.area / 3

            Mloc[1, 1] += wq * dot(b1, b1)
            Mloc[1, 2] += wq * dot(b1, b2)
            Mloc[1, 3] += wq * dot(b1, b3)
            Mloc[2, 1] += wq * dot(b2, b1)
            Mloc[2, 2] += wq * dot(b2, b2)
            Mloc[2, 3] += wq * dot(b2, b3)
            Mloc[3, 1] += wq * dot(b3, b1)
            Mloc[3, 2] += wq * dot(b3, b2)
            Mloc[3, 3] += wq * dot(b3, b3)
        end

        for a in 1:3, b in 1:3
            push!(I, fe[a])
            push!(J, fe[b])
            push!(V, T(fs[a] * fs[b]) * Mloc[a, b])
        end
    end

    return sparse(I, J, V, ne, ne)
end

function assemble_whitney_mass1(
    mesh::CurveMesh{T},
    geom::CurveGeometry{T},
) where {T<:AbstractFloat}
    ne = length(mesh.edges)
    d = [geom.edge_lengths[e] > eps(T) ? inv(geom.edge_lengths[e]) : zero(T) for e in 1:ne]
    return spdiagm(0 => d)
end

"""
    assemble_whitney_mass2(mesh, geom) -> sparse matrix

Assemble the consistent lowest-order Whitney 2-form mass matrix on surfaces.

Convention
----------
For face DOF `c2[f] = ∫_f β`, reconstructed density is `β_f = c2[f]/A_f`.
Therefore, the L2 mass pairing is diagonal with entries `1/A_f`.
"""
function assemble_whitney_mass2(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
) where {T<:AbstractFloat}
    _ = mesh
    d = [A > eps(T) ? inv(A) : zero(T) for A in geom.face_areas]
    return spdiagm(0 => d)
end

# -----------------------------------------------------------------------------
# Stiffness and Hodge-Laplacian operators
# -----------------------------------------------------------------------------

"""
    assemble_whitney_stiffness0(mesh, geom; form=:h1) -> sparse matrix

Assemble the Whitney 0-form stiffness matrix on simplicial meshes.

Surface
-------
`K_ij = ∫_Γ ∇ϕ_i · ∇ϕ_j dA` (piecewise-linear FEM stiffness).

Curve
-----
Segment-wise linear FEM stiffness `K = Σ_e (1/|e|) [1 -1; -1 1]`.
"""
function assemble_whitney_stiffness0(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T};
    form::Symbol=:h1,
) where {T<:AbstractFloat}
    form === :h1 || throw(ArgumentError("Unsupported form=$(repr(form))."))

    nv = length(mesh.points)
    I = Int[]
    J = Int[]
    V = T[]

    @inbounds for fi in 1:length(mesh.faces)
        face = mesh.faces[fi]
        grads, A = _triangle_barycentric_gradients(mesh, geom, fi)
        Kloc = zeros(T, 3, 3)
        for a in 1:3, b in 1:3
            Kloc[a, b] = A * dot(grads[a], grads[b])
        end
        ids = (face[1], face[2], face[3])
        _push_local3x3!(I, J, V, ids, Kloc)
    end

    return sparse(I, J, V, nv, nv)
end

function assemble_whitney_stiffness0(
    mesh::CurveMesh{T},
    geom::CurveGeometry{T};
    form::Symbol=:h1,
) where {T<:AbstractFloat}
    form === :h1 || throw(ArgumentError("Unsupported form=$(repr(form))."))

    nv = length(mesh.points)
    I = Int[]
    J = Int[]
    V = T[]

    @inbounds for (ei, e) in enumerate(mesh.edges)
        ℓ = geom.edge_lengths[ei]
        c = ℓ > eps(T) ? inv(ℓ) : zero(T)
        Kloc = c * @SMatrix [1 -1; -1 1]
        ids = (e[1], e[2])
        _push_local2x2!(I, J, V, ids, Kloc)
    end

    return sparse(I, J, V, nv, nv)
end

"""
    assemble_whitney_hodge_laplacian0(mesh, geom) -> sparse matrix

Assemble a sparse strong-form Whitney 0-form Hodge-Laplacian approximation:

`L0 = M0_lumped^{-1} * K0`,

where `K0` is the consistent stiffness matrix and `M0_lumped` is the row-sum
lumped Whitney mass.
"""
function assemble_whitney_hodge_laplacian0(
    mesh,
    geom,
)
    M0 = assemble_whitney_mass0(mesh, geom)
    K0 = assemble_whitney_stiffness0(mesh, geom)
    invMl, _ = _lumped_inverse(M0)
    return invMl * K0
end

"""
    assemble_whitney_hodge_laplacian1(mesh, geom) -> sparse matrix

Assemble a sparse strong-form Whitney 1-form Hodge-Laplacian approximation on
surfaces:

`Δ1 ≈ δ2 d1 + d0 δ1`,
with lumped inverses in the codifferentials.
"""
function assemble_whitney_hodge_laplacian1(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
) where {T<:AbstractFloat}
    dec = build_dec(mesh, geom)
    M0 = assemble_whitney_mass0(mesh, geom)
    M1 = assemble_whitney_mass1(mesh, geom)
    M2 = assemble_whitney_mass2(mesh, geom)

    invM0, _ = _lumped_inverse(M0)
    invM1, _ = _lumped_inverse(M1)

    δ1 = invM0 * dec.d0' * M1
    δ2 = invM1 * dec.d1' * M2

    return δ2 * dec.d1 + dec.d0 * δ1
end

# -----------------------------------------------------------------------------
# Minimal mixed/Hodge-Laplacian solves
# -----------------------------------------------------------------------------

"""
    solve_mixed_hodge_laplacian0(mesh, geom, rhs; bc=:none, gauge=:mean_zero)

Solve a minimal FEEC 0-form Hodge-Laplacian problem using the assembled
Whitney operator.

Gauge convention
----------------
For closed manifolds and `gauge=:mean_zero`, an augmented mean-zero constraint
is imposed with lumped mass weights.
"""
function solve_mixed_hodge_laplacian0(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    rhs::AbstractVector{T};
    bc::Symbol=:none,
    gauge::Symbol=:mean_zero,
) where {T<:AbstractFloat}
    bc === :none || throw(ArgumentError("Only bc=:none is currently supported."))

    nv = length(mesh.points)
    length(rhs) == nv || throw(DimensionMismatch("rhs length $(length(rhs)) != nv=$nv"))

    L0 = assemble_whitney_hodge_laplacian0(mesh, geom)
    M0 = assemble_whitney_mass0(mesh, geom)
    b = copy(rhs)

    if gauge === :mean_zero
        w = vec(sum(M0, dims=2))
        denom = sum(w)
        if abs(denom) > eps(T)
            b .-= (dot(w, b) / denom)
        end
        Aaug = _augment_mean_zero_system(L0, w)
        rhs_aug = zeros(T, nv + 1)
        rhs_aug[1:nv] .= b
        sol = Aaug \ rhs_aug
        u = sol[1:nv]
        λ = sol[end]
        return (u=u, lagrange=λ, operator=L0, rhs=b, mass=M0, gauge=gauge)
    else
        throw(ArgumentError("Unsupported gauge=$(repr(gauge))."))
    end
end

function solve_mixed_hodge_laplacian0(
    mesh::CurveMesh{T},
    geom::CurveGeometry{T},
    rhs::AbstractVector{T};
    bc::Symbol=:none,
    gauge::Symbol=:mean_zero,
) where {T<:AbstractFloat}
    bc === :none || throw(ArgumentError("Only bc=:none is currently supported."))

    nv = length(mesh.points)
    length(rhs) == nv || throw(DimensionMismatch("rhs length $(length(rhs)) != nv=$nv"))

    L0 = assemble_whitney_hodge_laplacian0(mesh, geom)
    M0 = assemble_whitney_mass0(mesh, geom)
    b = copy(rhs)

    if gauge === :mean_zero
        w = vec(sum(M0, dims=2))
        denom = sum(w)
        if abs(denom) > eps(T)
            b .-= (dot(w, b) / denom)
        end
        Aaug = _augment_mean_zero_system(L0, w)
        rhs_aug = zeros(T, nv + 1)
        rhs_aug[1:nv] .= b
        sol = Aaug \ rhs_aug
        u = sol[1:nv]
        λ = sol[end]
        return (u=u, lagrange=λ, operator=L0, rhs=b, mass=M0, gauge=gauge)
    else
        throw(ArgumentError("Unsupported gauge=$(repr(gauge))."))
    end
end

"""
    solve_mixed_hodge_laplacian1(mesh, geom, rhs; bc=:none, gauge=:harmonic_orthogonal)

Solve a minimal FEEC 1-form Hodge-Laplacian problem on surfaces.

Gauge convention
----------------
With `gauge=:harmonic_orthogonal`, the RHS (and returned solution) is projected
to be orthogonal to the DEC harmonic basis in the Whitney `M1` inner product.
"""
function solve_mixed_hodge_laplacian1(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
    rhs::AbstractVector{T};
    bc::Symbol=:none,
    gauge::Symbol=:harmonic_orthogonal,
) where {T<:AbstractFloat}
    bc === :none || throw(ArgumentError("Only bc=:none is currently supported."))

    topo = build_topology(mesh)
    ne = length(topo.edges)
    length(rhs) == ne || throw(DimensionMismatch("rhs length $(length(rhs)) != ne=$ne"))

    L1 = assemble_whitney_hodge_laplacian1(mesh, geom)
    M1 = assemble_whitney_mass1(mesh, geom)
    b = copy(rhs)

    H = zeros(T, ne, 0)
    if gauge === :harmonic_orthogonal
        dec = build_dec(mesh, geom)
        H = harmonic_basis(mesh, geom, dec)
        if size(H, 2) > 0
            G = Symmetric(H' * M1 * H)
            c = G \ (H' * (M1 * b))
            b .-= H * c
        end
    else
        throw(ArgumentError("Unsupported gauge=$(repr(gauge))."))
    end

    # Mild regularization for singular nullspaces.
    τ = sqrt(eps(T))
    Areg = L1 + spdiagm(0 => fill(τ, ne))
    u = Areg \ b

    if size(H, 2) > 0
        G = Symmetric(H' * M1 * H)
        c = G \ (H' * (M1 * u))
        u .-= H * c
    end

    return (u=u, operator=L1, rhs=b, mass=M1, gauge=gauge, harmonic_basis=H)
end

# -----------------------------------------------------------------------------
# DEC vs Whitney diagnostics
# -----------------------------------------------------------------------------

"""
    compare_dec_vs_whitney_mass(mesh, geom)

Return basic diagnostics comparing DEC and Whitney 0-form mass matrices.
"""
function compare_dec_vs_whitney_mass(
    mesh,
    geom,
)
    Mdec = mass_matrix(mesh, geom)
    Mwh = assemble_whitney_mass0(mesh, geom)

    nd = norm(Mdec - Mwh)
    nref = max(norm(Mdec), eps(eltype(nd)))

    return (
        size_dec=size(Mdec),
        size_whitney=size(Mwh),
        nnz_dec=nnz(Mdec),
        nnz_whitney=nnz(Mwh),
        norm_diff=nd,
        rel_diff=nd / nref,
    )
end

"""
    compare_dec_vs_whitney_laplacian(mesh, geom)

Return basic diagnostics comparing DEC scalar Laplacian and the Whitney
mass-lumped strong-form 0-Laplacian.
"""
function compare_dec_vs_whitney_laplacian(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T},
) where {T<:AbstractFloat}
    dec = build_dec(mesh, geom)
    Ldec = dec.lap0
    Lwh = assemble_whitney_hodge_laplacian0(mesh, geom)

    n = size(Ldec, 1)
    onesv = ones(T, n)
    nd = norm(Ldec - Lwh)
    nref = max(norm(Ldec), eps(T))

    return (
        size_dec=size(Ldec),
        size_whitney=size(Lwh),
        nnz_dec=nnz(Ldec),
        nnz_whitney=nnz(Lwh),
        norm_diff=nd,
        rel_diff=nd / nref,
        null_residual_dec=norm(Ldec * onesv),
        null_residual_whitney=norm(Lwh * onesv),
    )
end
