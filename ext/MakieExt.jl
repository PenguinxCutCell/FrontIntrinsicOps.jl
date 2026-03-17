module MakieExt

using Makie
import FrontIntrinsicOps as FIO

const GB = FIO.GeometryBasics

function __init__()
    @info "FrontIntrinsicOps Makie extension loaded"
end

# -----------------------------------------------------------------------------
# Theme
# -----------------------------------------------------------------------------

function FIO.makie_theme()
    return Makie.Theme(
        fontsize = 14,
        backgroundcolor = :white,
        Axis = (
            xlabel = "x",
            ylabel = "y",
            aspect = Makie.DataAspect(),
            xgridvisible = true,
            ygridvisible = true,
        ),
        Axis3 = (
            xlabel = "x",
            ylabel = "y",
            zlabel = "z",
        ),
    )
end

function FIO.set_makie_theme!()
    Makie.set_theme!(FIO.makie_theme())
    return nothing
end

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

_curve_points_for_lines(mesh::FIO.CurveMesh) = begin
    pts = [Makie.Point2f(p[1], p[2]) for p in mesh.points]
    isempty(pts) || push!(pts, first(pts))
    pts
end

_surface_geometrybasics_mesh(mesh::FIO.SurfaceMesh) = begin
    pts = [GB.Point3f(Float32(p[1]), Float32(p[2]), Float32(p[3])) for p in mesh.points]
    faces = [GB.TriangleFace{Int}(f[1], f[2], f[3]) for f in mesh.faces]
    GB.Mesh(pts, faces)
end

function _clear_axis!(ax)
    for plt in copy(ax.scene.plots)
        delete!(ax.scene, plt)
    end
    return ax
end

function _figure_axis_2d(; figure=nothing, axis=nothing, title=nothing)
    if axis !== nothing
        if title !== nothing
            axis.title = title
        end
        return figure === nothing ? axis.figure : figure, axis
    end
    fig = figure === nothing ? Makie.Figure() : figure
    ax = Makie.Axis(fig[1, 1]; title=title === nothing ? "" : title)
    return fig, ax
end

function _figure_axis_3d(; figure=nothing, axis=nothing, title=nothing)
    if axis !== nothing
        if title !== nothing
            axis.title = title
        end
        return figure === nothing ? axis.figure : figure, axis
    end
    fig = figure === nothing ? Makie.Figure() : figure
    ax = Makie.Axis3(fig[1, 1]; title=title === nothing ? "" : title, xlabel="x", ylabel="y", zlabel="z")
    return fig, ax
end

# -----------------------------------------------------------------------------
# Bounding box utility
# -----------------------------------------------------------------------------

function FIO.boundingbox_limits(mesh::FIO.CurveMesh)
    xs = map(p -> p[1], mesh.points)
    ys = map(p -> p[2], mesh.points)
    return ((minimum(xs), maximum(xs)), (minimum(ys), maximum(ys)))
end

function FIO.boundingbox_limits(mesh::FIO.SurfaceMesh)
    xs = map(p -> p[1], mesh.points)
    ys = map(p -> p[2], mesh.points)
    zs = map(p -> p[3], mesh.points)
    return ((minimum(xs), maximum(xs)), (minimum(ys), maximum(ys)), (minimum(zs), maximum(zs)))
end

# -----------------------------------------------------------------------------
# Public plotting helpers
# -----------------------------------------------------------------------------

function FIO.plot_front(mesh::FIO.CurveMesh;
    figure=nothing,
    axis=nothing,
    clear_axis::Bool=false,
    title=nothing,
    color=:royalblue,
    linewidth::Real=2,
    show_vertices::Bool=false,
    vertex_color=:black,
    markersize::Real=6,
    kwargs...,
)
    fig, ax = _figure_axis_2d(; figure=figure, axis=axis, title=title)
    clear_axis && _clear_axis!(ax)
    pts = _curve_points_for_lines(mesh)
    p = Makie.lines!(ax, pts; color=color, linewidth=linewidth, kwargs...)
    if show_vertices
        Makie.scatter!(ax, [Makie.Point2f(p[1], p[2]) for p in mesh.points]; color=vertex_color, markersize=markersize)
    end
    return fig, ax, p
end

function FIO.plot_front(mesh::FIO.SurfaceMesh;
    figure=nothing,
    axis=nothing,
    clear_axis::Bool=false,
    title=nothing,
    color=:steelblue,
    transparency::Bool=false,
    alpha::Real=1.0,
    wireframe::Bool=false,
    wire_color=:black,
    wire_width::Real=1,
    show_vertices::Bool=false,
    vertex_color=:black,
    markersize::Real=4,
    kwargs...,
)
    fig, ax = _figure_axis_3d(; figure=figure, axis=axis, title=title)
    clear_axis && _clear_axis!(ax)
    gb = _surface_geometrybasics_mesh(mesh)
    p = Makie.mesh!(ax, gb; color=color, transparency=transparency, alpha=alpha, kwargs...)
    if wireframe
        Makie.wireframe!(ax, gb; color=wire_color, linewidth=wire_width)
    end
    if show_vertices
        pts = [Makie.Point3f(p[1], p[2], p[3]) for p in mesh.points]
        Makie.scatter!(ax, pts; color=vertex_color, markersize=markersize)
    end
    return fig, ax, p
end

function FIO.plot_faces(mesh::FIO.SurfaceMesh; kwargs...)
    return FIO.plot_front(mesh; kwargs...)
end

function FIO.plot_wireframe(mesh::FIO.SurfaceMesh;
    figure=nothing,
    axis=nothing,
    clear_axis::Bool=false,
    title=nothing,
    color=:black,
    linewidth::Real=1,
)
    fig, ax = _figure_axis_3d(; figure=figure, axis=axis, title=title)
    clear_axis && _clear_axis!(ax)
    gb = _surface_geometrybasics_mesh(mesh)
    p = Makie.wireframe!(ax, gb; color=color, linewidth=linewidth)
    return fig, ax, p
end

function FIO.plot_vertices(mesh::FIO.CurveMesh;
    figure=nothing,
    axis=nothing,
    clear_axis::Bool=false,
    title=nothing,
    color=:black,
    markersize::Real=8,
)
    fig, ax = _figure_axis_2d(; figure=figure, axis=axis, title=title)
    clear_axis && _clear_axis!(ax)
    pts = [Makie.Point2f(p[1], p[2]) for p in mesh.points]
    p = Makie.scatter!(ax, pts; color=color, markersize=markersize)
    return fig, ax, p
end

function FIO.plot_vertices(mesh::FIO.SurfaceMesh;
    figure=nothing,
    axis=nothing,
    clear_axis::Bool=false,
    title=nothing,
    color=:black,
    markersize::Real=4,
)
    fig, ax = _figure_axis_3d(; figure=figure, axis=axis, title=title)
    clear_axis && _clear_axis!(ax)
    pts = [Makie.Point3f(p[1], p[2], p[3]) for p in mesh.points]
    p = Makie.scatter!(ax, pts; color=color, markersize=markersize)
    return fig, ax, p
end

function FIO.plot_normals(mesh::FIO.CurveMesh, geom::FIO.CurveGeometry;
    figure=nothing,
    axis=nothing,
    scale::Real=0.05,
    every::Int=1,
    color=:crimson,
    linewidth::Real=1.5,
)
    fig, ax = _figure_axis_2d(; figure=figure, axis=axis)
    segs = Makie.Point2f[]
    for i in 1:max(1, every):length(mesh.points)
        p = mesh.points[i]
        n = geom.vertex_normals[i]
        q = p + scale * n
        push!(segs, Makie.Point2f(p[1], p[2]))
        push!(segs, Makie.Point2f(q[1], q[2]))
    end
    p = Makie.linesegments!(ax, segs; color=color, linewidth=linewidth)
    return fig, ax, p
end

function FIO.plot_normals(mesh::FIO.SurfaceMesh, geom::FIO.SurfaceGeometry;
    figure=nothing,
    axis=nothing,
    scale::Real=0.05,
    every::Int=1,
    color=:crimson,
    linewidth::Real=1.5,
)
    fig, ax = _figure_axis_3d(; figure=figure, axis=axis)
    segs = Makie.Point3f[]
    for i in 1:max(1, every):length(mesh.points)
        p = mesh.points[i]
        n = geom.vertex_normals[i]
        q = p + scale * n
        push!(segs, Makie.Point3f(p[1], p[2], p[3]))
        push!(segs, Makie.Point3f(q[1], q[2], q[3]))
    end
    p = Makie.linesegments!(ax, segs; color=color, linewidth=linewidth)
    return fig, ax, p
end

FIO.plot_normals(mesh::FIO.CurveMesh; kwargs...) = FIO.plot_normals(mesh, FIO.compute_geometry(mesh); kwargs...)
FIO.plot_normals(mesh::FIO.SurfaceMesh; kwargs...) = FIO.plot_normals(mesh, FIO.compute_geometry(mesh); kwargs...)

function FIO.plot_front(mesh::FIO.CurveMesh, geom::FIO.CurveGeometry;
    normals::Bool=false,
    curvature::Bool=false,
    normal_scale::Real=0.05,
    normal_every::Int=1,
    kwargs...,
)
    fig, ax, p = FIO.plot_front(mesh; kwargs...)
    if curvature
        vals = geom.signed_curvature
        pts = [Makie.Point2f(q[1], q[2]) for q in mesh.points]
        Makie.scatter!(ax, pts; color=vals, colormap=:viridis, markersize=6)
    end
    if normals
        FIO.plot_normals(mesh, geom; figure=fig, axis=ax, scale=normal_scale, every=normal_every)
    end
    return fig, ax, p
end

function FIO.plot_front(mesh::FIO.SurfaceMesh, geom::FIO.SurfaceGeometry;
    normals::Bool=false,
    curvature::Bool=false,
    normal_scale::Real=0.05,
    normal_every::Int=1,
    kwargs...,
)
    fig, ax, p = if curvature
        color_values = if isempty(geom.mean_curvature)
            dec = FIO.build_dec(mesh, geom)
            cgeom = FIO.compute_curvature(mesh, geom, dec)
            cgeom.mean_curvature
        else
            geom.mean_curvature
        end
        FIO.plot_front(mesh; color=color_values, kwargs...)
    else
        FIO.plot_front(mesh; kwargs...)
    end
    if normals
        FIO.plot_normals(mesh, geom; figure=fig, axis=ax, scale=normal_scale, every=normal_every)
    end
    return fig, ax, p
end

# -----------------------------------------------------------------------------
# Makie conversions/recipes for raw mesh types
# -----------------------------------------------------------------------------

Makie.plottype(::FIO.CurveMesh) = Makie.Lines
Makie.plottype(::FIO.SurfaceMesh) = Makie.Mesh

function Makie.convert_arguments(::Type{<:Makie.Lines}, mesh::FIO.CurveMesh)
    return (_curve_points_for_lines(mesh),)
end

function Makie.convert_arguments(::Type{<:Makie.Mesh}, mesh::FIO.SurfaceMesh)
    return (_surface_geometrybasics_mesh(mesh),)
end

end
