# test_open_surfaces.jl – Tests for open surfaces, boundary detection, and
#                          Dirichlet/Neumann BC support. (v0.4)
#
# Tests:
# 1. Boundary edge and vertex detection on known open meshes.
# 2. is_open_surface / is_closed distinction.
# 3. apply_dirichlet! with scalar and vector values.
# 4. apply_dirichlet_to_system!: matrix rows/columns zeroed.
# 5. solve_open_surface_poisson: boundary values enforced.
# 6. boundary_mass_matrix: non-negative, correct total length.

using SparseArrays

@testset "Open surface: is_open_surface on flat patch" begin
    mesh = make_flat_patch(; N=5, L=1.0)
    topo = build_topology(mesh)
    @test is_open_surface(topo)
end

@testset "Open surface: is_open_surface on closed sphere" begin
    mesh = make_uvsphere(1.0; nφ=8, nθ=16)
    topo = build_topology(mesh)
    @test !is_open_surface(topo)
end

@testset "Open surface: detect_boundary_edges" begin
    mesh = make_flat_patch(; N=4, L=1.0)
    topo = build_topology(mesh)
    be   = detect_boundary_edges(topo)
    @test !isempty(be)
    # All returned indices are valid
    @test all(1 .<= be .<= length(topo.edges))
    # Boundary edges have exactly one adjacent face
    for ei in be
        @test length(topo.edge_faces[ei]) == 1
    end
end

@testset "Open surface: detect_boundary_vertices" begin
    mesh = make_flat_patch(; N=5, L=1.0)
    topo = build_topology(mesh)
    geom = compute_geometry(mesh)
    bv   = detect_boundary_vertices(mesh, topo)
    @test !isempty(bv)
    @test all(1 .<= bv .<= length(mesh.points))
    # Boundary vertices should be on the boundary edges
    be     = detect_boundary_edges(topo)
    bv_set = Set(bv)
    for ei in be
        i, j = topo.edges[ei][1], topo.edges[ei][2]
        @test i in bv_set
        @test j in bv_set
    end
end

@testset "Open surface: boundary vertices on N=3 patch corners" begin
    # For a 3×3 patch (N=3, L=1) all four corners and edge mid-points are boundary
    mesh  = make_flat_patch(; N=3, L=1.0)
    topo  = build_topology(mesh)
    geom  = compute_geometry(mesh)
    bv    = detect_boundary_vertices(mesh, topo)
    # Total vertices = 4^2 = 16; interior = 2^2 = 4; boundary = 12
    @test length(bv) == 12
end

@testset "apply_dirichlet!: scalar value" begin
    nv = 10
    u  = zeros(Float64, nv)
    bv = [1, 4, 7]
    apply_dirichlet!(u, bv, 3.14)
    for vi in bv
        @test u[vi] ≈ 3.14
    end
    # Interior unchanged
    @test u[2] == 0.0
    @test u[5] == 0.0
end

@testset "apply_dirichlet!: vector values" begin
    nv = 8
    u  = zeros(Float64, nv)
    bv = [2, 5, 8]
    g  = [1.0, 2.0, 3.0]
    apply_dirichlet!(u, bv, g)
    @test u[2] ≈ 1.0
    @test u[5] ≈ 2.0
    @test u[8] ≈ 3.0
    # Others unchanged
    @test u[1] == 0.0
end

@testset "apply_dirichlet_to_system!: boundary rows zeroed" begin
    n  = 6
    # Simple tridiagonal system
    A  = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1))
    b  = ones(Float64, n)
    bv = [1, n]
    g  = [0.0, 1.0]
    apply_dirichlet_to_system!(A, b, bv, g)
    # Boundary rows should be identity rows
    @test A[1, 1] ≈ 1.0
    @test b[1] ≈ 0.0
    @test A[n, n] ≈ 1.0
    @test b[n] ≈ 1.0
    # Off-diagonal entries in boundary rows should be zero
    for j in 2:n
        @test A[1, j] ≈ 0.0 atol=1e-15
    end
end

@testset "apply_dirichlet_symmetric!: symmetry preserved" begin
    n  = 5
    # Symmetric tridiagonal
    A  = spdiagm(-1 => -ones(n-1), 0 => 2*ones(n), 1 => -ones(n-1))
    b  = ones(Float64, n)
    bv = [1, n]
    g  = [0.0, 0.0]
    apply_dirichlet_symmetric!(A, b, bv, g)
    # After symmetric enforcement, A should still be symmetric
    @test maximum(abs.(A - A')) < 1e-12
    # Diagonal for BV should be 1
    @test A[1, 1] ≈ 1.0
    @test A[n, n] ≈ 1.0
end

@testset "solve_open_surface_poisson: Laplace with Dirichlet BCs" begin
    mesh = make_flat_patch(; N=6, L=1.0)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    topo = build_topology(mesh)
    nv   = length(mesh.points)

    bv = detect_boundary_vertices(mesh, topo)
    # Dirichlet: u = x on boundary (linear function is harmonic -> exact interior)
    g = [Float64(mesh.points[i][1]) for i in bv]

    f = zeros(Float64, nv)
    u = solve_open_surface_poisson(mesh, geom, dec, topo, f, bv, g)

    @test length(u) == nv
    @test all(isfinite.(u))

    # Check boundary values are enforced
    for (k, vi) in enumerate(bv)
        @test abs(u[vi] - g[k]) < 1e-8
    end

    # For a harmonic function (Laplace eq) on flat patch with u=x BC,
    # the exact solution is u=x everywhere.  Check interior error.
    u_exact = [Float64(mesh.points[i][1]) for i in 1:nv]
    interior = setdiff(1:nv, bv)
    err = maximum(abs.(u[interior] .- u_exact[interior]))
    @test err < 0.1  # coarse mesh, but should be small
end

@testset "solve_open_surface_poisson: constant BCs" begin
    mesh = make_flat_patch(; N=5, L=1.0)
    geom = compute_geometry(mesh)
    dec  = build_dec(mesh, geom)
    topo = build_topology(mesh)
    nv   = length(mesh.points)

    bv = detect_boundary_vertices(mesh, topo)
    # All boundary set to same constant -> interior should also be constant (for flat Laplace)
    g  = fill(2.0, length(bv))
    f  = zeros(Float64, nv)
    u  = solve_open_surface_poisson(mesh, geom, dec, topo, f, bv, g)

    @test all(isfinite.(u))
    # Solution should be nearly constant everywhere
    @test maximum(abs.(u .- 2.0)) < 0.1
end

@testset "boundary_mass_matrix: non-negative entries" begin
    mesh = make_flat_patch(; N=4, L=1.0)
    geom = compute_geometry(mesh)
    topo = build_topology(mesh)

    Mb = boundary_mass_matrix(mesh, geom, topo)
    @test size(Mb, 1) == length(mesh.points)
    @test size(Mb, 2) == length(mesh.points)
    # All non-zero entries should be non-negative
    @test all(nonzeros(Mb) .>= 0.0)
end

@testset "boundary_mass_matrix: total boundary length" begin
    # For a unit square patch (L=1, N=4), boundary perimeter = 4.0
    mesh  = make_flat_patch(; N=4, L=1.0)
    geom  = compute_geometry(mesh)
    topo  = build_topology(mesh)
    Mb    = boundary_mass_matrix(mesh, geom, topo)
    # Row sums should equal the boundary length contributions per vertex.
    # The total sum of all entries should give twice the perimeter (P1 mass matrix),
    # or the 1-vector's contribution: 1ᵀ Mb 1 ≈ perimeter
    ones_v = ones(Float64, length(mesh.points))
    total  = dot(ones_v, Mb * ones_v)
    @test abs(total - 4.0) < 0.5
end

@testset "add_neumann_rhs!: increases rhs at boundary" begin
    mesh = make_flat_patch(; N=4, L=1.0)
    geom = compute_geometry(mesh)
    topo = build_topology(mesh)
    nv   = length(mesh.points)

    b    = zeros(Float64, nv)
    flux = 1.0
    add_neumann_rhs!(b, mesh, geom, topo, flux)

    # Boundary vertices should have non-zero RHS contributions
    bv   = detect_boundary_vertices(mesh, topo)
    @test any(b[bv] .!= 0.0)
    # Interior vertices should remain zero (Neumann only affects boundary)
    interior = setdiff(1:nv, bv)
    @test all(b[interior] .== 0.0)
end
