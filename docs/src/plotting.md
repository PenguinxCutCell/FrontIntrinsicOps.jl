# Plotting with Makie

`FrontIntrinsicOps.jl` provides an optional Makie extension for plotting raw
front meshes (`CurveMesh`, `SurfaceMesh`) and geometry overlays.

Plotting is activated when a Makie backend is loaded, for example:

```julia
using CairoMakie
using FrontIntrinsicOps
```

## Theme

```julia
set_makie_theme!()
```

This applies a lightweight publication-oriented style used across examples.

## Curve plotting

```julia
mesh = sample_circle(1.0, 128)
fig, ax, p = plot_front(mesh; show_vertices=true, title="curve")
save("curve.png", fig)
```

## Surface plotting

```julia
mesh = generate_icosphere(1.0, 2)
fig, ax, p = plot_front(mesh; wireframe=true, transparency=true, alpha=0.9)
save("surface.png", fig)
```

## Normals overlay

```julia
geom = compute_geometry(mesh)
fig, ax, p = plot_front(mesh)
plot_normals(mesh, geom; figure=fig, axis=ax, scale=0.05, every=8)
save("normals.png", fig)
```

## Convenience behavior

The extension defines Makie conversions for mesh types, so these also work:

```julia
fig = plot(sample_circle(1.0, 64))
fig = plot(generate_icosphere(1.0, 1))
```

If Makie is not loaded, plotting functions throw a clear error explaining how
to enable a backend.
