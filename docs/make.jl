using Documenter
using FrontIntrinsicOps

makedocs(
    modules = [FrontIntrinsicOps],
    authors = "PenguinxCutCell contributors",
    sitename = "FrontIntrinsicOps.jl",
    format = Documenter.HTML(
        canonical = "https://PenguinxCutCell.github.io/FrontIntrinsicOps.jl",
        repolink = "https://github.com/PenguinxCutCell/FrontIntrinsicOps.jl",
        collapselevel = 2,
    ),
    pages = [
        "Home" => "index.md",
        "API" => [
            "API Documentation" => "types.md",
            "Diagnostics" => "diagnostics.md",
            "Generators" => "generators.md",
            "Geometry" => "geometry.md",
            "PDEs" => "pdes.md",
            "Plotting with Makie" => "plotting.md",
        ],
        "Math" => [
            "Mesh Types" => "01_mesh_types.md",
            "Topology" => "02_topology.md",
            "Geometry" => "03_geometry.md",
            "Discretization" => "04_dec.md",
            "Laplace-Beltrami" => "05_laplace_beltrami.md",
            "Curvature" => "06_curvature.md",
            "Surface Diffusion" => "07_surface_diffusion.md",
            "Transport" => "08_transport.md",
            "Advection-Diffusion" => "09_advection_diffusion.md",
            "Reaction-Diffusion" => "10_reaction_diffusion.md",
            "Vector Calculus" => "11_vector_calculus.md",
            "Hodge Decomposition" => "12_hodge_decomposition.md",
            "Topology-aware DEC" => "topology_aware_dec.md",
            "Geodesics" => "geodesics.md",
            "Parallel Transport" => "parallel_transport.md",
            "Exterior Algebra Extensions" => "exterior_algebra_extensions.md",
            "FEEC Overview" => "feec_overview.md",
            "Continuum to Discrete Theory" => "theory_continuum_discrete_surface_pde.md",
            "Whitney Forms" => "whitney_forms.md",
            "Commuting Projections" => "commuting_projections.md",
            "de Rham Sequence" => "de_rham_sequence.md",
            "High-Resolution Transport" => "13_highres_transport.md",
            "Open Surfaces" => "14_open_surfaces.md",
            "Caching" => "15_caching.md",
            "Generators" => "16_generators.md",
            "Ambient Signed Distance" => "17_signed_distance.md",
        ],
        "Tutorials" => [
            "Getting Started" => "01_getting_started.md",
            "Surface Diffusion" => "02_surface_diffusion.md",
            "Transport" => "03_transport.md",
            "Reaction-Diffusion" => "04_reaction_diffusion.md",
            "Hodge Decomposition" => "05_hodge_decomposition.md",
            "Open Surfaces" => "06_open_surfaces.md",
    ],
    ],
    pagesonly = true,
    warnonly = true,
    remotes = nothing,
)

if get(ENV, "CI", "") == "true"
    deploydocs(
        repo = "github.com/PenguinxCutCell/FrontIntrinsicOps.jl",
        push_preview = true,
    )
end
