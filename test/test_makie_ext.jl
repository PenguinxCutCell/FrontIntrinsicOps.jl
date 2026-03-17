using FrontIntrinsicOps
using Makie
using CairoMakie

@testset "FrontIntrinsicOps MakieExt smoke" begin
    FrontIntrinsicOps.set_makie_theme!()
    th = FrontIntrinsicOps.makie_theme()
    @test th isa Makie.Theme

    curve = sample_circle(1.0, 64)
    srf = generate_icosphere(1.0, 1)
    cgeom = compute_geometry(curve)
    sgeom = compute_geometry(srf)

    fig1, _, _ = plot_front(curve; show_vertices=true, title="curve")
    @test fig1 isa Makie.Figure
    fig1n, _, _ = plot_normals(curve, cgeom; scale=0.05, every=4)
    @test fig1n isa Makie.Figure

    fig2, _, _ = plot_front(srf; wireframe=true, title="surface")
    @test fig2 isa Makie.Figure
    fig2n, _, _ = plot_normals(srf, sgeom; scale=0.1, every=8)
    @test fig2n isa Makie.Figure

    png1 = tempname() * ".png"
    png2 = tempname() * ".png"
    save(png1, fig1)
    save(png2, fig2)
    @test isfile(png1)
    @test isfile(png2)

    # Recipe / conversion smoke
    fig3 = Makie.plot(curve)
    fig4 = Makie.plot(srf)
    @test fig3 isa Makie.Figure
    @test fig4 isa Makie.Figure
end
