using FrontIntrinsicOps
using CairoMakie

FrontIntrinsicOps.set_makie_theme!()

mesh = generate_icosphere(1.0, 2)
fig, _, _ = plot_front(mesh; wireframe=true, transparency=true, alpha=0.9, title="SurfaceMesh")
save("plot_surface_mesh.png", fig)
println("saved: plot_surface_mesh.png")
