# examples/heat_equation_moebius_strip.jl
#
# Pure advection of a Gaussian spike on an open, non-orientable Moebius strip.
#
# PDE:  du/dt + M^{-1} A(v) u = 0
#
# where A(v) is the conservative transport operator assembled from edge fluxes.
# We use the upwind scheme in space and SSP-RK3 in time.
#
# Run: julia --project=. examples/heat_equation_moebius_strip.jl

using FrontIntrinsicOps
using StaticArrays
using LinearAlgebra
using Printf

println("="^70)
println("  Advection of a Gaussian Spike on a Moebius Strip")
println("="^70)
println()

# -----------------------------------------------------------------------------
# 1) Moebius strip mesh (local to this example)
# -----------------------------------------------------------------------------

@inline function moebius_point(u::Float64, s::Float64, R::Float64) :: SVector{3,Float64}
    cu  = cos(u)
    su  = sin(u)
    cu2 = cos(0.5 * u)
    su2 = sin(0.5 * u)
    x   = (R + s * cu2) * cu
    y   = (R + s * cu2) * su
    z   = s * su2
    return SVector{3,Float64}(x, y, z)
end

@inline function moebius_tangent_u(u::Float64, s::Float64, R::Float64) :: SVector{3,Float64}
    cu  = cos(u)
    su  = sin(u)
    cu2 = cos(0.5 * u)
    su2 = sin(0.5 * u)

    A   = R + s * cu2
    dA  = -0.5 * s * su2

    dxdu = dA * cu - A * su
    dydu = dA * su + A * cu
    dzdu = 0.5 * s * cu2
    return SVector{3,Float64}(dxdu, dydu, dzdu)
end

function make_moebius_strip(
        nu::Int,
        nv::Int;
        R::Float64=1.0,
        w::Float64=0.25,
) :: SurfaceMesh{Float64}
    nu >= 3 || error("make_moebius_strip: nu must be >= 3.")
    nv >= 2 || error("make_moebius_strip: nv must be >= 2.")
    w > 0.0 || error("make_moebius_strip: w must be positive.")

    pts = Vector{SVector{3,Float64}}(undef, nu * (nv + 1))
    idx(i, j) = i * (nv + 1) + j + 1   # i in 0:(nu-1), j in 0:nv

    for i in 0:(nu - 1)
        u = 2pi * i / nu
        for j in 0:nv
            s = -w + (2w * j) / nv
            pts[idx(i, j)] = moebius_point(u, s, R)
        end
    end

    faces = SVector{3,Int}[]

    # Regular longitudinal strips
    for i in 0:(nu - 2), j in 0:(nv - 1)
        v00 = idx(i, j)
        v01 = idx(i, j + 1)
        v10 = idx(i + 1, j)
        v11 = idx(i + 1, j + 1)
        push!(faces, SVector{3,Int}(v00, v10, v11))
        push!(faces, SVector{3,Int}(v00, v11, v01))
    end

    # Seam strip: connect i=nu-1 back to i=0 with Moebius reversal j -> nv-j
    i = nu - 1
    for j in 0:(nv - 1)
        jr0 = nv - j
        jr1 = nv - (j + 1)
        v00 = idx(i, j)
        v01 = idx(i, j + 1)
        v10 = idx(0, jr0)
        v11 = idx(0, jr1)
        push!(faces, SVector{3,Int}(v00, v10, v11))
        push!(faces, SVector{3,Int}(v00, v11, v01))
    end

    return SurfaceMesh{Float64}(pts, faces)
end

function count_boundary_components(topo::MeshTopology) :: Int
    be = detect_boundary_edges(topo)
    isempty(be) && return 0

    adj = Dict{Int,Vector{Int}}()
    for ei in be
        a, b = topo.edges[ei]
        haskey(adj, a) || (adj[a] = Int[])
        haskey(adj, b) || (adj[b] = Int[])
        push!(adj[a], b)
        push!(adj[b], a)
    end

    visited = Set{Int}()
    ncomp = 0
    for v in keys(adj)
        v in visited && continue
        ncomp += 1
        stack = [v]
        push!(visited, v)
        while !isempty(stack)
            cur = pop!(stack)
            for nb in adj[cur]
                if !(nb in visited)
                    push!(visited, nb)
                    push!(stack, nb)
                end
            end
        end
    end
    return ncomp
end

# -----------------------------------------------------------------------------
# 2) Mesh, geometry, topology diagnostics
# -----------------------------------------------------------------------------

nu = 80
nv = 20
R  = 1.0
w  = 0.25

mesh = make_moebius_strip(nu, nv; R=R, w=w)
geom = compute_geometry(mesh; dual_area=:mixed)
topo = build_topology(mesh)

nverts = length(mesh.points)
nfaces = length(mesh.faces)
bv     = detect_boundary_vertices(mesh, topo)
bcount = count_boundary_components(topo)

@printf "Mesh:                %d vertices, %d faces\n" nverts nfaces
@printf "is_closed(mesh):     %s\n" string(is_closed(mesh))
@printf "is_manifold(mesh):   %s\n" string(is_manifold(mesh))
@printf "consistent orient.:  %s  (false is expected for Moebius)\n" string(has_consistent_orientation(mesh))
@printf "boundary vertices:   %d\n" length(bv)
@printf "boundary components: %d\n\n" bcount

@assert !is_closed(mesh)
@assert is_manifold(mesh)
@assert bcount == 1

# -----------------------------------------------------------------------------
# 3) Velocity field and Gaussian initial condition
# -----------------------------------------------------------------------------

idx(i, j) = i * (nv + 1) + j + 1

u_param = Vector{Float64}(undef, nverts)
s_param = Vector{Float64}(undef, nverts)
for i in 0:(nu - 1), j in 0:nv
    vi = idx(i, j)
    u_param[vi] = 2pi * i / nu
    s_param[vi] = -w + (2w * j) / nv
end

# Longitudinal advection velocity (tangent to u-lines of the Moebius strip).
omega = 1.0
vel   = Vector{SVector{3,Float64}}(undef, nverts)
for vi in 1:nverts
    vel[vi] = omega * moebius_tangent_u(u_param[vi], s_param[vi], R)
end

# Gaussian spike in ambient distance, centered off the midline so twist is visible.
u_center = 0.35 * pi
s_center = 0.10
p_center = moebius_point(u_center, s_center, R)
sigma    = 0.12
u0 = Float64[
    exp(-sum(abs2, mesh.points[vi] - p_center) / (2 * sigma^2))
    for vi in 1:nverts
]

@printf "Initial spike: center (u,s)=(%.3fπ, %.3f), sigma=%.3f\n\n" u_center / pi s_center sigma

# -----------------------------------------------------------------------------
# 4) Transport operator and time integration
# -----------------------------------------------------------------------------

A = assemble_transport_operator(mesh, geom, vel; scheme=:upwind)

dt_cfl = estimate_transport_dt(mesh, geom, vel; cfl=0.10)
T_end  = 2.5
nstep  = max(1, round(Int, T_end / dt_cfl))
dt     = T_end / nstep

diag_every = max(1, nstep ÷ 10)
save_every = max(1, nstep ÷ 140)   # subsample snapshots for animation

@printf "Advection setup:\n"
@printf "  omega = %.3f\n" omega
@printf "  dt_cfl = %.6f,  dt = %.6f,  nstep = %d,  T_end = %.3f\n" dt_cfl dt nstep T_end
@printf "  scheme = centered, time integrator = SSP-RK3\n\n"

function l2_norm_surface(mesh, geom, u::AbstractVector{<:Real}) :: Float64
    return sqrt(integrate_vertex_field(mesh, geom, u .^ 2))
end

function total_mass_surface(mesh, geom, u::AbstractVector{<:Real}) :: Float64
    return integrate_vertex_field(mesh, geom, u)
end

function print_diag(step::Int, t::Float64, u::Vector{Float64}, mesh, geom)
    maxu = maximum(u)
    minu = minimum(u)
    l2n  = l2_norm_surface(mesh, geom, u)
    mass = total_mass_surface(mesh, geom, u)
    @printf "  %-8d  %-10.4f  %-12.6e  %-12.6e  %-12.6e  %-12.6e\n" step t maxu minu l2n mass
    return maxu, minu, l2n, mass
end

function run_advection(
        mesh,
        geom,
        A,
        u0::Vector{Float64},
        dt::Float64,
        nstep::Int;
        diag_every::Int=1,
        save_every::Int=1,
)
    u = copy(u0)
    u_hist = Vector{Vector{Float64}}()
    t_hist = Float64[]
    push!(u_hist, copy(u))
    push!(t_hist, 0.0)

    println("Diagnostics:")
    @printf "  %-8s  %-10s  %-12s  %-12s  %-12s  %-12s\n" "step" "t" "max(u)" "min(u)" "L2 norm" "mass"
    print_diag(0, 0.0, u, mesh, geom)

    for k in 1:nstep
        u = step_surface_transport_ssprk3(mesh, geom, A, u, dt)

        if k == 1 || (k % diag_every == 0) || k == nstep
            print_diag(k, k * dt, u, mesh, geom)
        end
        if (k % save_every == 0) || k == nstep
            push!(u_hist, copy(u))
            push!(t_hist, k * dt)
        end
    end

    return u, u_hist, t_hist
end

u, u_hist, t_hist = run_advection(
    mesh, geom, A, u0, dt, nstep;
    diag_every=diag_every,
    save_every=save_every,
)

l2_0 = l2_norm_surface(mesh, geom, u0)
l2_T = l2_norm_surface(mesh, geom, u)
m0   = total_mass_surface(mesh, geom, u0)
mT   = total_mass_surface(mesh, geom, u)
rel_change = norm(u .- u0) / max(norm(u0), eps(Float64))

println()
@printf "Initial L2 norm:  %.6e\n" l2_0
@printf "Final   L2 norm:  %.6e\n" l2_T
@printf "Initial mass:     %.6e\n" m0
@printf "Final   mass:     %.6e\n" mT
@printf "Relative field change ||u(T)-u0||/||u0||: %.6e\n" rel_change

@assert all(isfinite, u)
@assert minimum(abs.(u)) > -1e-8
@assert rel_change > 1e-2

# -----------------------------------------------------------------------------
# 5) Optional plotting + moving-camera animation
# -----------------------------------------------------------------------------

try
    @eval using CairoMakie
    set_makie_theme!()

    cmax = max(maximum(u0), maximum(u))
    clim = (0.0, cmax)
    cmap = :turbo

    # Fixed bounds prevent clipping when the camera rotates.
    xs = Float64[p[1] for p in mesh.points]
    ys = Float64[p[2] for p in mesh.points]
    zs = Float64[p[3] for p in mesh.points]
    xmin, xmax = minimum(xs), maximum(xs)
    ymin, ymax = minimum(ys), maximum(ys)
    zmin, zmax = minimum(zs), maximum(zs)
    xmid, ymid, zmid = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
    xr, yr, zr = (xmax - xmin) / 2, (ymax - ymin) / 2, (zmax - zmin) / 2
    pad = 1.15
    set_limits!(ax) = begin
        CairoMakie.xlims!(ax, xmid - pad * xr, xmid + pad * xr)
        CairoMakie.ylims!(ax, ymid - pad * yr, ymid + pad * yr)
        CairoMakie.zlims!(ax, zmid - pad * zr, zmid + pad * zr)
        nothing
    end

    az0 = 0.72 * pi
    el0 = 0.22 * pi

    # -- Static figure: initial and final fields --
    fig_static = CairoMakie.Figure(size=(1300, 560))
    ax0 = CairoMakie.Axis3(
        fig_static[1, 1];
        title     = "Initial Gaussian spike",
        azimuth   = az0,
        elevation = el0,
        aspect    = :data,
        viewmode  = :fit,
    )
    axT = CairoMakie.Axis3(
        fig_static[1, 2];
        title     = @sprintf("Final field  t = %.2f", T_end),
        azimuth   = az0,
        elevation = el0,
        aspect    = :data,
        viewmode  = :fit,
    )
    set_limits!(ax0)
    set_limits!(axT)

    plot_front(mesh; figure=fig_static, axis=ax0, color=u0, colorrange=clim, colormap=cmap,
               wireframe=true, wire_color=(:black, 0.25), wire_width=0.6)
    plot_front(mesh; figure=fig_static, axis=axT, color=u,  colorrange=clim, colormap=cmap,
               wireframe=true, wire_color=(:black, 0.25), wire_width=0.6)

    out_png = joinpath(@__DIR__, "moebius_advection_initial_final.png")
    CairoMakie.save(out_png, fig_static)

    # -- Animation: moving camera --
    fig_anim = CairoMakie.Figure(size=(900, 700))
    ax_anim  = CairoMakie.Axis3(
        fig_anim[1, 1];
        title     = "Advection of Gaussian spike",
        azimuth   = az0,
        elevation = el0,
        aspect    = :data,
        viewmode  = :fit,
    )
    set_limits!(ax_anim)

    # Keep one mesh plot and only update field/camera each frame.
    u_obs = CairoMakie.Observable(u_hist[1])
    plot_front(mesh; figure=fig_anim, axis=ax_anim, color=u_obs,
               colorrange=clim, colormap=cmap,
               wireframe=true, wire_color=(:black, 0.25), wire_width=0.6)

    function update_frame!(field::Vector{Float64}, t::Float64, xi::Float64)
        # Start from static view and rotate one full turn.
        ax_anim.azimuth   = az0 + 2pi * xi
        ax_anim.elevation = el0
        u_obs[] = field
        ax_anim.title = @sprintf("Advection of Gaussian spike  t = %.2f", t)
        return nothing
    end

    nframe = length(u_hist)
    out_mp4 = joinpath(@__DIR__, "moebius_advection_gaussian.mp4")
    CairoMakie.record(fig_anim, out_mp4, 1:nframe; framerate=24) do k
        xi = (k - 1) / max(1, nframe - 1)
        update_frame!(u_hist[k], t_hist[k], xi)
    end

    @printf "\nSaved optional plot:      %s\n" out_png
    @printf "Saved optional animation: %s\n" out_mp4
catch err
    @printf "\nOptional plotting skipped: %s\n" sprint(showerror, err)
end

println()
println("Moebius strip advection example complete.")
