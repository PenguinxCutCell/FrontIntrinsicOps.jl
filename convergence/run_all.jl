# convergence/run_all.jl – Run all convergence studies sequentially.
#
# Usage:
#   julia --project=. convergence/run_all.jl

t_start = time()

include(joinpath(@__DIR__, "circle_convergence.jl"))
println()
include(joinpath(@__DIR__, "sphere_convergence.jl"))
println()
include(joinpath(@__DIR__, "torus_convergence.jl"))
println()
include(joinpath(@__DIR__, "poisson_sphere_convergence.jl"))

println()
println("=" ^ 72)
@printf("  All convergence studies completed in %.1f seconds.\n", time() - t_start)
println("=" ^ 72)
