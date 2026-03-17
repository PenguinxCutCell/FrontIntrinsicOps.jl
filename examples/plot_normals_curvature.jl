using FrontIntrinsicOps
using CairoMakie

FrontIntrinsicOps.set_makie_theme!()

curve = sample_circle(1.0, 96)
cgeom = compute_geometry(curve)
fig1, ax1, _ = plot_front(curve, cgeom; curvature=true, title="Curve curvature")
plot_normals(curve, cgeom; figure=fig1, axis=ax1, scale=0.08, every=6)
save("plot_normals_curvature_curve.png", fig1)
println("saved: plot_normals_curvature_curve.png")

surf = generate_icosphere(1.0, 2)
sgeom = compute_geometry(surf)
fig2, ax2, _ = plot_front(surf, sgeom; curvature=false, wireframe=true, title="Surface curvature")
plot_normals(surf, sgeom; figure=fig2, axis=ax2, scale=0.08, every=10)
save("plot_normals_curvature_surface.png", fig2)
println("saved: plot_normals_curvature_surface.png")
