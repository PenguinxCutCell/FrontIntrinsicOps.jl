# feec_spaces.jl – Lightweight FEEC Whitney space descriptors and complex builder.

"""
    AbstractFEECSpace

Abstract supertype for lowest-order FEEC-compatible spaces.
"""
abstract type AbstractFEECSpace end

"""
    Whitney0Space

Lowest-order Whitney 0-form space on a simplicial front mesh.

Fields
------
- `mesh`: owning mesh.
- `geom`: geometry container used by reconstruction/assembly.
- `ndofs`: number of global degrees of freedom.
- `degree`: form degree (`0`).
- `element_type`: `:curve_edge` or `:surface_triangle`.
"""
struct Whitney0Space{M,G} <: AbstractFEECSpace
    mesh::M
    geom::G
    ndofs::Int
    degree::Int
    element_type::Symbol
end

"""
    Whitney1Space

Lowest-order Whitney 1-form space on a simplicial front mesh.

Fields
------
- `mesh`: owning mesh.
- `geom`: geometry container used by reconstruction/assembly.
- `ndofs`: number of global degrees of freedom.
- `degree`: form degree (`1`).
- `element_type`: `:curve_edge` or `:surface_triangle`.
"""
struct Whitney1Space{M,G} <: AbstractFEECSpace
    mesh::M
    geom::G
    ndofs::Int
    degree::Int
    element_type::Symbol
end

"""
    Whitney2Space

Lowest-order Whitney 2-form space on triangulated surfaces.

For curves this space has `ndofs == 0`.

Fields
------
- `mesh`: owning mesh.
- `geom`: geometry container used by reconstruction/assembly.
- `ndofs`: number of global degrees of freedom.
- `degree`: form degree (`2`).
- `element_type`: `:curve_edge` or `:surface_triangle`.
"""
struct Whitney2Space{M,G} <: AbstractFEECSpace
    mesh::M
    geom::G
    ndofs::Int
    degree::Int
    element_type::Symbol
end

"""
    WhitneyComplex

Explicit lowest-order discrete de Rham complex container:

`0 -> Λh0 --d0--> Λh1 --d1--> Λh2 -> 0`

The mass matrices `M0`, `M1`, `M2` are consistent Whitney mass matrices by
default (see `build_whitney_complex`).
"""
struct WhitneyComplex{M,G,T<:AbstractFloat}
    mesh::M
    geom::G
    V0::Whitney0Space{M,G}
    V1::Whitney1Space{M,G}
    V2::Whitney2Space{M,G}
    d0::SparseMatrixCSC{T,Int}
    d1::SparseMatrixCSC{T,Int}
    M0::SparseMatrixCSC{T,Int}
    M1::SparseMatrixCSC{T,Int}
    M2::SparseMatrixCSC{T,Int}
end

@inline _whitney_element_type(::CurveMesh) = :curve_edge
@inline _whitney_element_type(::SurfaceMesh) = :surface_triangle

"""
    build_whitney_complex(mesh, geom; mass=:consistent, cache=true) -> WhitneyComplex

Build the lowest-order Whitney complex compatible with the existing DEC
incidence conventions.

Conventions
-----------
- `Λh0` DOFs live on vertices.
- `Λh1` DOFs live on oriented primal edges.
- `Λh2` DOFs live on oriented primal faces (surface meshes only).
- `d0` and `d1` are the existing incidence matrices from the DEC layer.

Keyword arguments
-----------------
- `mass`: `:consistent` (default) or `:lumped`.
- `cache`: accepted for API compatibility; currently no persistent cache object
  is returned (assembly is deterministic and stateless).
"""
function build_whitney_complex(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T};
    mass::Symbol=:consistent,
    cache::Bool=true,
) where {T<:AbstractFloat}
    _ = cache
    topo = build_topology(mesh)
    nv = length(mesh.points)
    ne = length(topo.edges)
    nf = length(mesh.faces)

    V0 = Whitney0Space(mesh, geom, nv, 0, _whitney_element_type(mesh))
    V1 = Whitney1Space(mesh, geom, ne, 1, _whitney_element_type(mesh))
    V2 = Whitney2Space(mesh, geom, nf, 2, _whitney_element_type(mesh))

    d0 = incidence_0(mesh)
    d1 = incidence_1(mesh)

    mass in (:consistent, :lumped) || throw(ArgumentError("Unknown mass=$(repr(mass)). Use :consistent or :lumped."))

    M0 = if mass === :consistent
        assemble_whitney_mass0(mesh, geom)
    else
        mass_matrix(mesh, geom)
    end

    M1 = if mass === :consistent
        assemble_whitney_mass1(mesh, geom)
    else
        M1c = assemble_whitney_mass1(mesh, geom)
        s1 = vec(sum(M1c, dims=2))
        spdiagm(0 => s1)
    end

    M2 = assemble_whitney_mass2(mesh, geom)

    return WhitneyComplex(mesh, geom, V0, V1, V2, d0, d1, M0, M1, M2)
end

function build_whitney_complex(
    mesh::CurveMesh{T},
    geom::CurveGeometry{T};
    mass::Symbol=:consistent,
    cache::Bool=true,
) where {T<:AbstractFloat}
    _ = cache
    nv = length(mesh.points)
    ne = length(mesh.edges)

    V0 = Whitney0Space(mesh, geom, nv, 0, _whitney_element_type(mesh))
    V1 = Whitney1Space(mesh, geom, ne, 1, _whitney_element_type(mesh))
    V2 = Whitney2Space(mesh, geom, 0, 2, _whitney_element_type(mesh))

    d0 = incidence_0(mesh)
    d1 = spzeros(T, 0, ne)

    mass in (:consistent, :lumped) || throw(ArgumentError("Unknown mass=$(repr(mass)). Use :consistent or :lumped."))

    M0 = if mass === :consistent
        assemble_whitney_mass0(mesh, geom)
    else
        mass_matrix(mesh, geom)
    end
    M1 = if mass === :consistent
        assemble_whitney_mass1(mesh, geom)
    else
        M1c = assemble_whitney_mass1(mesh, geom)
        s1 = vec(sum(M1c, dims=2))
        spdiagm(0 => s1)
    end
    M2 = spzeros(T, 0, 0)

    return WhitneyComplex(mesh, geom, V0, V1, V2, d0, d1, M0, M1, M2)
end
