# plotting_stubs.jl – Optional plotting API fallbacks.

const _MAKIE_ERR = "Makie plotting requires loading a Makie backend, e.g. `using CairoMakie`."

makie_theme(args...; kwargs...) = error(_MAKIE_ERR)
set_makie_theme!(args...; kwargs...) = error(_MAKIE_ERR)
plot_front(args...; kwargs...) = error(_MAKIE_ERR)
plot_normals(args...; kwargs...) = error(_MAKIE_ERR)
plot_wireframe(args...; kwargs...) = error(_MAKIE_ERR)
plot_vertices(args...; kwargs...) = error(_MAKIE_ERR)
plot_faces(args...; kwargs...) = error(_MAKIE_ERR)
boundingbox_limits(args...; kwargs...) = error(_MAKIE_ERR)
