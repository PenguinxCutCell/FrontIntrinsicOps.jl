using Test
using Statistics
using StaticArrays

include("test_helpers.jl")

@testset "FrontIntrinsicOps" begin
    @testset "Curve: circle" begin
        include("test_curve_circle.jl")
    end
    @testset "Surface: sphere" begin
        include("test_surface_sphere.jl")
    end
    @testset "Laplace–Beltrami" begin
        include("test_laplace_beltrami.jl")
    end
    @testset "Curvature" begin
        include("test_curvature.jl")
    end
    @testset "Integrals" begin
        include("test_integrals.jl")
    end
    @testset "Generators" begin
        include("test_generators.jl")
    end
    @testset "Dual areas" begin
        include("test_dual_areas.jl")
    end
    @testset "Gauss–Bonnet" begin
        include("test_gauss_bonnet.jl")
    end
    @testset "Laplace method comparison" begin
        include("test_laplace_compare.jl")
    end
    @testset "Gaussian curvature convergence" begin
        include("test_gaussian_curvature_convergence.jl")
    end
    # v0.3 tests
    @testset "Surface diffusion and Poisson" begin
        include("test_surface_diffusion.jl")
    end
    @testset "Surface transport" begin
        include("test_surface_transport.jl")
    end
    @testset "k-form operators" begin
        include("test_kforms.jl")
    end
    @testset "Allocation sanity" begin
        include("test_allocations.jl")
    end
    # v0.4 tests
    @testset "Surface reaction–diffusion" begin
        include("test_reaction_diffusion.jl")
    end
    @testset "Surface vector calculus" begin
        include("test_vector_calculus.jl")
    end
    @testset "Hodge decomposition" begin
        include("test_hodge_decomposition.jl")
    end
    @testset "Transport limiters, open surfaces, caching, performance" begin
        include("test_v04_misc.jl")
    end
end
