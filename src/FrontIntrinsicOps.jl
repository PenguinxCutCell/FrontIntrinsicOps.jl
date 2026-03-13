"""
    FrontIntrinsicOps

A static interface-geometry / DDG / DEC package for triangulated front meshes.

`FrontIntrinsicOps` computes intrinsic geometric quantities and intrinsic
discrete operators on a front mesh.  The primary inputs are:

- Triangulated 3-D closed surfaces (from STL files or mesh generators).
- Closed 2-D polygonal curves (from CSV files, point lists, or generators).

This package provides:
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

Non-goals (v0.2)
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
    compare_laplace_methods

end # module FrontIntrinsicOps
