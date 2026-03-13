# operators_surfaces.jl – DEC operators for SurfaceMesh.
#
# Implements
# ----------
# * `build_dec(mesh, geom; laplace=:dec) -> SurfaceDEC`
# * `build_laplace_beltrami(mesh, geom; method=:dec/:cotan) -> SparseMatrixCSC`
# * scalar Laplace-Beltrami: L = star0^{-1} d0' star1 d0
# * `laplace_beltrami(mesh, geom, dec, u)` -- apply L to vertex field u

"""
    build_laplace_beltrami(mesh::SurfaceMesh{T}, geom::SurfaceGeometry{T};
                           method=:dec) -> SparseMatrixCSC{T,Int}

Assemble the scalar Laplace-Beltrami operator for a triangulated surface.

Two assembly paths are supported:

- `method = :dec` (default): factored DEC form `L = star0^{-1} d0' star1 d0`
  using the incidence matrix `d0` and the cotan-weight Hodge star `star1`.
  Uses `geom.vertex_dual_areas` for `star0^{-1}`.

- `method = :cotan`: direct assembly from the classic cotan formula
    (L u)_i = (1 / A_i) sum_{j in N(i)} w_{ij} (u_i - u_j)
  where w_{ij} = (1/2)(cot alpha + cot beta) are the cotan weights of edge (i,j)
  and A_i = geom.vertex_dual_areas[i].

Both methods:
- return a positive-semi-definite matrix (L = -Delta_Gamma).
- preserve the constant nullspace: L * ones approx 0.
- use the same vertex ordering and the same dual areas from `geom`.

On well-shaped meshes the two methods agree to near machine precision.
"""
function build_laplace_beltrami(
        mesh   :: SurfaceMesh{T},
        geom   :: SurfaceGeometry{T};
        method :: Symbol = :dec,
) :: SparseMatrixCSC{T,Int} where {T}
    method in (:dec, :cotan) ||
        error("build_laplace_beltrami: unknown method $(repr(method)). " *
              "Use :dec or :cotan.")
    if method === :dec
        return _build_laplace_dec(mesh, geom)
    else
        return _build_laplace_cotan(mesh, geom)
    end
end

# DEC factored assembly: L = star0^{-1} d0' star1 d0
function _build_laplace_dec(mesh::SurfaceMesh{T}, geom::SurfaceGeometry{T}) where {T}
    d0 = incidence_0(mesh)
    s1 = hodge_star_1(mesh, geom)
    inv_s0_diag = [da > eps(T) ? one(T)/da : zero(T)
                   for da in geom.vertex_dual_areas]
    inv_s0 = spdiagm(0 => inv_s0_diag)
    return inv_s0 * d0' * s1 * d0
end

# Direct cotan assembly
function _build_laplace_cotan(mesh::SurfaceMesh{T}, geom::SurfaceGeometry{T}) where {T}
    pts   = mesh.points
    faces = mesh.faces
    nv    = length(pts)

    # We accumulate row, col, val triplets for the sparse matrix.
    # For each face (a,b,c) with cotan weights cot_c (for edge a-b),
    # cot_a (for edge b-c), cot_b (for edge c-a):
    #   off-diagonal: (i,j) and (j,i) get -w_{ij}
    #   diagonal:     (i,i) gets +w_{ij}
    I_ind = Int[]
    J_ind = Int[]
    V_val = T[]

    function add_edge!(vi, vj, w)
        push!(I_ind, vi); push!(J_ind, vj); push!(V_val, -w)
        push!(I_ind, vj); push!(J_ind, vi); push!(V_val, -w)
        push!(I_ind, vi); push!(J_ind, vi); push!(V_val,  w)
        push!(I_ind, vj); push!(J_ind, vj); push!(V_val,  w)
    end

    for face in faces
        ia, ib, ic = face[1], face[2], face[3]
        a, b, c = pts[ia], pts[ib], pts[ic]

        ab = b - a;  ac = c - a
        ba = a - b;  bc = c - b
        ca = a - c;  cb = b - c

        cot_a = cotangent(ab, ac)   # angle at a, opposite edge b-c
        cot_b = cotangent(bc, ba)   # angle at b, opposite edge a-c
        cot_c = cotangent(ca, cb)   # angle at c, opposite edge a-b

        # edge (a,b): weight = 0.5 * cot_c
        add_edge!(ia, ib, T(0.5) * cot_c)
        # edge (b,c): weight = 0.5 * cot_a
        add_edge!(ib, ic, T(0.5) * cot_a)
        # edge (c,a): weight = 0.5 * cot_b
        add_edge!(ic, ia, T(0.5) * cot_b)
    end

    # Assemble and apply dual-area scaling
    W = sparse(I_ind, J_ind, V_val, nv, nv)
    inv_da = [da > eps(T) ? one(T)/da : zero(T) for da in geom.vertex_dual_areas]
    return spdiagm(0 => inv_da) * W
end

"""
    build_dec(mesh::SurfaceMesh{T}, geom::SurfaceGeometry{T};
              laplace=:dec) -> SurfaceDEC{T}

Assemble all DEC operators for a triangulated surface and return a
`SurfaceDEC` container.

Keyword arguments
-----------------
- `laplace :: Symbol` – Laplace-Beltrami assembly path (`:dec` or `:cotan`).
  Defaults to `:dec`.

Laplace-Beltrami
----------------
The scalar Laplace-Beltrami operator is assembled through the DEC factorisation:

    L = star0^{-1} d0' star1 d0

where star1 uses cotan weights (see `hodge_star_1`).  This operator is
**positive semi-definite** (L = -Delta_Gamma in standard Delta = div grad convention).

On a sphere of radius R, the coordinate functions satisfy:
    L x = (2/R^2) x   (since Delta_Gamma x = -(2/R^2) x and L = -Delta_Gamma)

This is algebraically equivalent to the classical cotan/DDG formula:

    (L u)_i = (1 / A_i) sum_{j in N(i)} w_{ij} (u_i - u_j)

where `A_i` is the vertex dual area and `w_{ij} = (1/2)(cot alpha_{ij} + cot beta_{ij})`.

Acceptance criterion
--------------------
For a closed oriented triangulated surface, `d1 * d0 approx 0` holds to machine
precision (checked by `check_dec`).
"""
function build_dec(
        mesh    :: SurfaceMesh{T},
        geom    :: SurfaceGeometry{T};
        laplace :: Symbol = :dec,
) :: SurfaceDEC{T} where {T}
    d0 = incidence_0(mesh)
    d1 = incidence_1(mesh)
    s0 = hodge_star_0(mesh, geom)
    s1 = hodge_star_1(mesh, geom)
    s2 = hodge_star_2(mesh, geom)

    lap0 = build_laplace_beltrami(mesh, geom; method=laplace)

    return SurfaceDEC{T}(d0, d1, s0, s1, s2, lap0)
end

"""
    laplace_beltrami(mesh, geom, dec, u) -> Vector{T}

Apply the scalar Laplace-Beltrami operator to vertex field `u`.

Returns `dec.lap0 * u`.
"""
function laplace_beltrami(
        ::SurfaceMesh,
        ::SurfaceGeometry,
        dec::SurfaceDEC{T},
        u::AbstractVector{T},
) :: Vector{T} where {T}
    return dec.lap0 * u
end

"""
    gradient(mesh, dec, u) -> Vector{T}

Discrete gradient of vertex field `u` to an edge 1-cochain.
"""
function gradient(::SurfaceMesh, dec::SurfaceDEC{T}, u::AbstractVector{T}) where {T}
    return dec.d0 * u
end
