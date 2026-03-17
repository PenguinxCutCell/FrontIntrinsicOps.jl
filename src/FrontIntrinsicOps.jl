"""
    FrontIntrinsicOps

A static interface-geometry / DDG / DEC package for triangulated front meshes.

`FrontIntrinsicOps` computes intrinsic geometric quantities and intrinsic
discrete operators on a front mesh.  The primary inputs are:

- Triangulated 3-D closed surfaces (from STL files or mesh generators).
- Closed 2-D polygonal curves (from CSV files, point lists, or generators).

This package provides (v0.4):
- Internal mesh types (`CurveMesh`, `SurfaceMesh`) independent of IO.
- Deterministic mesh generators for convergence studies (v0.2).
- Topology extraction (edges, adjacency, orientation, manifold checks).
- Geometry computation (normals, areas, dual measures, curvatures).
- Dual-area methods: barycentric and mixed/Voronoi (v0.2).
- DEC operators (incidence matrices d0, d1; Hodge stars star0, star1, star2;
  scalar Laplace-Beltrami via DEC or direct cotan assembly).
- Curvature (signed curvature on curves, mean curvature on surfaces,
  Gaussian curvature via angle defect).
- Integral quantities (length, area, enclosed volume, integrated Gaussian
  curvature).
- Mesh and DEC diagnostics (Euler characteristic, Gauss-Bonnet, star1 sign
  report, Laplace method comparison).
- 0-form mass matrix from star0 (v0.3).
- Surface Poisson / Helmholtz solvers (v0.3).
- Transient diffusion on static surfaces (v0.3).
- Scalar transport on static surfaces (v0.3).
- Advection–diffusion IMEX time-stepping (v0.3).
- Codifferential and 1-form Hodge Laplacian (v0.3).
- Allocation-conscious PDE helpers (v0.3).
- Surface reaction–diffusion equations with IMEX time stepping (v0.4).
- Surface vector calculus: projection, gradient, divergence, 1-form conversions (v0.4).
- Hodge decomposition for discrete 1-forms (v0.4).
- High-resolution (limiter-based) surface transport (v0.4).
- Open surface support: boundary detection and Dirichlet/Neumann BCs (v0.4).
- Operator and factorization caching for repeated solves (v0.4).
- Low-allocation in-place PDE stepping helpers (v0.4).

Non-goals
---------
This package does **not** implement front advection, remeshing, marker
redistribution, topology changes, or coupling to bulk solvers.  Those
belong in separate packages.

See the README for conventions, examples, and the mathematical background.
"""
module FrontIntrinsicOps

using LinearAlgebra
using SparseArrays
using StaticArrays
using FileIO
import MeshIO
import GeometryBasics

# Source files
include("utils.jl")
include("types.jl")
include("io.jl")
include("topology.jl")
include("incidence.jl")
include("geometry_curves.jl")
include("geometry_surfaces.jl")
include("hodge.jl")
include("operators_curves.jl")
include("operators_surfaces.jl")
include("curvature.jl")
include("integrals.jl")
include("checks.jl")
include("generators.jl")
# v0.3: PDE layer and k-form operators
include("surface_pdes_common.jl")
include("surface_diffusion.jl")
include("surface_transport.jl")
include("surface_advection_diffusion.jl")
include("kforms.jl")
include("perf_utils.jl")
# v0.4: New PDE capabilities
include("reaction_diffusion.jl")
include("vector_calculus.jl")
include("hodge_decomposition.jl")
include("transport_highres.jl")
include("open_surfaces.jl")
include("cache.jl")
include("performance.jl")
include("plotting_stubs.jl")

# Public API

export
    # Types
    CurveMesh,
    SurfaceMesh,
    CurveGeometry,
    SurfaceGeometry,
    CurveDEC,
    SurfaceDEC,

    # IO
    load_surface_stl,
    load_curve_csv,
    load_curve_points,

    # Mesh generators (v0.2)
    sample_circle,
    generate_uvsphere,
    generate_icosphere,
    generate_torus,
    generate_ellipsoid,
    generate_perturbed_sphere,

    # Topology
    MeshTopology,
    build_topology,
    vertex_to_faces,
    vertex_to_edges,
    edge_to_faces,
    is_closed,
    is_manifold,
    has_consistent_orientation,
    curve_vertex_order,

    # Incidence
    incidence_0,
    incidence_1,

    # Geometry
    compute_geometry,
    edge_midpoints,

    # DEC operators
    build_dec,
    build_laplace_beltrami,
    laplace_beltrami,
    gradient,
    divergence,

    # Hodge stars (exported for power users)
    hodge_star_0,
    hodge_star_1,
    hodge_star_2,

    # Curvature
    curvature,
    mean_curvature_normal,
    mean_curvature,
    gaussian_curvature,
    compute_curvature,

    # Integrals
    measure,
    enclosed_measure,
    integrate_vertex_field,
    integrate_face_field,
    integrated_gaussian_curvature,

    # Checks and diagnostics
    check_mesh,
    check_dec,
    euler_characteristic,
    gauss_bonnet_residual,
    star1_sign_report,
    compare_laplace_methods,

    # v0.3: PDE common utilities
    mass_matrix,
    lumped_mass_vector,
    apply_mass!,
    weighted_mean,
    zero_mean_projection!,
    project_zero_mean!,
    enforce_compatibility!,

    # v0.3: Surface diffusion and Poisson/Helmholtz
    laplace_matrix,
    assemble_diffusion_operator,
    solve_surface_poisson,
    solve_surface_helmholtz,
    step_surface_diffusion_backward_euler,
    step_surface_diffusion_crank_nicolson,

    # v0.3: Scalar transport
    tangential_projection,
    edge_flux_velocity,
    assemble_transport_operator,
    step_surface_transport_forward_euler,
    step_surface_transport_ssprk2,
    step_surface_transport_ssprk3,
    estimate_transport_dt,

    # v0.3: Advection–diffusion
    assemble_advection_diffusion_operators,
    step_surface_advection_diffusion_imex,
    step_surface_advection_diffusion_backward_euler,

    # v0.3: k-form operators
    codifferential_1,
    codifferential_2,
    hodge_laplacian_0,
    hodge_laplacian_1,
    gradient_0_to_1,
    divergence_1_to_0,
    curl_like_1_to_2,

    # v0.3: Performance helpers
    mul_diag_left!,
    mul_diag_right!,
    apply_laplace!,
    weighted_l2_error,
    weighted_linf_error,

    # v0.4: Surface reaction–diffusion
    evaluate_reaction!,
    fisher_kpp_reaction,
    linear_decay_reaction,
    bistable_reaction,
    step_surface_reaction_diffusion_explicit,
    step_surface_reaction_diffusion_imex,
    solve_surface_reaction_diffusion,

    # v0.4: Surface vector calculus
    tangential_project,
    tangential_project!,
    tangential_project_field,
    gradient_0_to_tangent_vectors,
    divergence_tangent_vectors,
    tangent_vectors_to_1form,
    oneform_to_tangent_vectors,
    surface_rot_0form,

    # v0.4: Hodge decomposition
    exact_component_1form,
    coexact_component_1form,
    harmonic_component_1form,
    hodge_decompose_1form,
    hodge_decomposition_residual,
    hodge_inner_products,

    # v0.4: High-resolution transport
    minmod,
    minmod3,
    vanleer_limiter,
    superbee_limiter,
    assemble_transport_operator_limited,
    step_surface_transport_limited,

    # v0.4: Open surfaces / boundary conditions
    detect_boundary_edges,
    detect_boundary_vertices,
    is_open_surface,
    apply_dirichlet!,
    apply_dirichlet_to_system!,
    apply_dirichlet_symmetric!,
    add_neumann_rhs!,
    boundary_mass_matrix,
    solve_open_surface_poisson,

    # v0.4: Operator caching
    SurfacePDECache,
    CurvePDECache,
    build_pde_cache,
    update_pde_cache,
    step_diffusion_cached,
    solve_helmholtz_cached,

    # v0.4: Low-allocation performance helpers
    SurfaceDiffusionBuffers,
    SurfaceRDBuffers,
    alloc_diffusion_buffers,
    alloc_rd_buffers,
    step_diffusion_inplace!,
    step_rd_inplace!,
    apply_mass_inplace!,
    apply_laplace_inplace!,
    l2_norm_cached,
    energy_norm_cached,

    # Optional Makie plotting API (implemented in ext/MakieExt.jl)
    makie_theme,
    set_makie_theme!,
    plot_front,
    plot_normals,
    plot_wireframe,
    plot_vertices,
    plot_faces,
    boundingbox_limits

end # module FrontIntrinsicOps
