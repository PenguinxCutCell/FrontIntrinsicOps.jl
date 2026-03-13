"""
    FrontIntrinsicOps

A static interface-geometry / DDG / DEC package for triangulated front meshes.

`FrontIntrinsicOps` computes intrinsic geometric quantities and intrinsic
discrete operators on a front mesh.  The primary inputs are:

- Triangulated 3-D closed surfaces (from STL files or mesh generators).
- Closed 2-D polygonal curves (from CSV files, point lists, or generators).

This package provides (v0.3):
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

Non-goals (v0.3)
----------------
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
    weighted_linf_error

end # module FrontIntrinsicOps
