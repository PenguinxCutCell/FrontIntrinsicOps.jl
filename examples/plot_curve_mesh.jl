using FrontIntrinsicOps
using CairoMakie

FrontIntrinsicOps.set_makie_theme!()

mesh = sample_circle(1.0, 128)
fig, _, _ = plot_front(mesh; show_vertices=true, title="CurveMesh")
save("plot_curve_mesh.png", fig)
println("saved: plot_curve_mesh.png")
