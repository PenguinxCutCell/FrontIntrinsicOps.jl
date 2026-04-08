# de_rham_sequence.jl – Explicit discrete de Rham sequence API for Whitney FEEC layer.

function _symmetry_residual(A::SparseMatrixCSC)
    return norm(A - A')
end

function _min_eig_if_small(A::SparseMatrixCSC{T,Int}; max_n::Int=250) where {T<:AbstractFloat}
    n = size(A, 1)
    if n == 0
        return zero(T)
    elseif n <= max_n
        As = Symmetric(Matrix((A + A') / 2))
        return eigmin(As)
    end
    # For larger systems avoid dense eigensolves in diagnostics.
    return T(NaN)
end

function _mass_diagnostics(A::SparseMatrixCSC{T,Int}) where {T<:AbstractFloat}
    n = size(A, 1)
    d = n > 0 ? diag(A) : T[]
    return (
        symmetry_residual=_symmetry_residual(A),
        min_diag=(isempty(d) ? zero(T) : minimum(d)),
        min_eigenvalue=_min_eig_if_small(A),
    )
end

"""
    build_de_rham_sequence(mesh, geom; family=:whitney, mass=:consistent) -> WhitneyComplex

Build an explicit lowest-order discrete de Rham complex view.

Current family support
----------------------
- `family=:whitney` only.
"""
function build_de_rham_sequence(
    mesh,
    geom;
    family::Symbol=:whitney,
    mass::Symbol=:consistent,
)
    family === :whitney || throw(ArgumentError("Unsupported de Rham family=$(repr(family))."))
    return build_whitney_complex(mesh, geom; mass=mass)
end

"""
    de_rham_report(complex) -> NamedTuple

Return structural and numerical diagnostics for a `WhitneyComplex`.

Returned fields include
-----------------------
- `ndofs0`, `ndofs1`, `ndofs2`
- `nnz_d0`, `nnz_d1`
- `nnz_M0`, `nnz_M1`, `nnz_M2`
- `exactness_residuals` for `d1*d0`
- `mass_symmetry` diagnostics
- `positive_definiteness` diagnostics (where computable)
"""
function de_rham_report(complex::WhitneyComplex)
    d10 = complex.d1 * complex.d0

    m0d = _mass_diagnostics(complex.M0)
    m1d = _mass_diagnostics(complex.M1)
    m2d = _mass_diagnostics(complex.M2)

    return (
        ndofs0=complex.V0.ndofs,
        ndofs1=complex.V1.ndofs,
        ndofs2=complex.V2.ndofs,
        nnz_d0=nnz(complex.d0),
        nnz_d1=nnz(complex.d1),
        nnz_M0=nnz(complex.M0),
        nnz_M1=nnz(complex.M1),
        nnz_M2=nnz(complex.M2),
        exactness_residuals=(d1_d0=norm(d10),),
        mass_symmetry=(
            M0=m0d.symmetry_residual,
            M1=m1d.symmetry_residual,
            M2=m2d.symmetry_residual,
        ),
        positive_definiteness=(
            M0=m0d,
            M1=m1d,
            M2=m2d,
        ),
    )
end

"""
    verify_subcomplex(complex; atol=1e-12) -> Bool

Verify the discrete subcomplex condition `d1*d0 ≈ 0`.
"""
function verify_subcomplex(
    complex::WhitneyComplex;
    atol::Real=1e-12,
)
    return norm(complex.d1 * complex.d0) <= atol
end

"""
    verify_commuting_projection(mesh, geom; tests=(:k01, :k12), atol=1e-10) -> NamedTuple

Evaluate commuting projection residuals for deterministic manufactured fields.

Tests
-----
- `:k01`: `Π1(df) - d0 Π0(f)`
- `:k12`: `Π2(dα) - d1 Π1(α)` (surface only)
"""
function verify_commuting_projection(
    mesh::SurfaceMesh{T},
    geom::SurfaceGeometry{T};
    tests::Tuple=(:k01, :k12),
    atol::Real=1e-10,
) where {T<:AbstractFloat}
    dec = build_dec(mesh, geom)

    f = x -> sin(T(1.3) * x[1]) + T(0.2) * x[2] - T(0.1) * x[3]
    α = x -> SVector{3,T}(-x[2], x[1], T(0.5) * x[3])

    r01 = zeros(T, 0)
    r12 = zeros(T, 0)

    if :k01 in tests
        r01 = projection_commutator_01(f, mesh, geom, dec)
    end
    if :k12 in tests
        r12 = projection_commutator_12(α, mesh, geom, dec)
    end

    n01 = isempty(r01) ? zero(T) : norm(r01)
    n12 = isempty(r12) ? zero(T) : norm(r12)

    pass01 = :k01 in tests ? n01 <= atol : true
    pass12 = :k12 in tests ? n12 <= atol : true

    return (
        k01_norm=n01,
        k12_norm=n12,
        k01_pass=pass01,
        k12_pass=pass12,
        pass=(pass01 && pass12),
    )
end

function verify_commuting_projection(
    mesh::CurveMesh{T},
    geom::CurveGeometry{T};
    tests::Tuple=(:k01,),
    atol::Real=1e-10,
) where {T<:AbstractFloat}
    dec = build_dec(mesh, geom)

    f = x -> sin(T(1.2) * x[1]) + T(0.3) * x[2]
    r01 = projection_commutator_01(f, mesh, geom, dec)
    n01 = norm(r01)

    return (
        k01_norm=n01,
        k01_pass=n01 <= atol,
        pass=n01 <= atol,
    )
end
