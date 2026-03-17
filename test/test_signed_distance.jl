using LinearAlgebra
using Random

@testset "Primitive kernels" begin
    q = SVector(0.5, 1.0)
    a = SVector(0.0, 0.0)
    b = SVector(1.0, 0.0)
    d2, c, t, feat = FrontIntrinsicOps.closest_point_segment(q, a, b)
    @test isapprox(d2, 1.0; atol=1e-14)
    @test isapprox(c[1], 0.5; atol=1e-14)
    @test isapprox(c[2], 0.0; atol=1e-14)
    @test feat == :edge
    @test 0.0 <= t <= 1.0

    d2a, _, _, feata = FrontIntrinsicOps.closest_point_segment(SVector(-2.0, 0.0), a, b)
    d2b, _, _, featb = FrontIntrinsicOps.closest_point_segment(SVector(3.0, 0.0), a, b)
    @test feata == :vertex0
    @test featb == :vertex1
    @test isapprox(d2a, 4.0; atol=1e-14)
    @test isapprox(d2b, 4.0; atol=1e-14)

    ta = SVector(0.0, 0.0, 0.0)
    tb = SVector(1.0, 0.0, 0.0)
    tc = SVector(0.0, 1.0, 0.0)

    d2f, cf, _, ff = FrontIntrinsicOps.closest_point_triangle(SVector(0.2, 0.2, 1.0), ta, tb, tc)
    @test ff == :face
    @test isapprox(cf[1], 0.2; atol=1e-12)
    @test isapprox(cf[2], 0.2; atol=1e-12)
    @test isapprox(cf[3], 0.0; atol=1e-12)
    @test isapprox(d2f, 1.0; atol=1e-12)

    @test FrontIntrinsicOps.closest_point_triangle(SVector(-0.2, 0.2, 0.1), ta, tb, tc)[4] == :edge31
    @test FrontIntrinsicOps.closest_point_triangle(SVector(0.8, -0.2, 0.1), ta, tb, tc)[4] == :edge12
    @test FrontIntrinsicOps.closest_point_triangle(SVector(0.6, 0.6, 0.1), ta, tb, tc)[4] == :edge23
    @test FrontIntrinsicOps.closest_point_triangle(SVector(-0.3, -0.2, 0.1), ta, tb, tc)[4] == :vertex1
    @test FrontIntrinsicOps.closest_point_triangle(SVector(2.0, -0.1, 0.2), ta, tb, tc)[4] == :vertex2
    @test FrontIntrinsicOps.closest_point_triangle(SVector(-0.1, 2.0, 0.2), ta, tb, tc)[4] == :vertex3
end

@testset "2D sign and winding" begin
    open_curve = load_curve_points([SVector(-1.0, 0.0), SVector(1.0, 0.0)]; closed=false)
    copen = build_signed_distance_cache(open_curve)
    s_above = signed_distance(SVector(0.0, 1.0), copen; sign_mode=:pseudonormal).distance
    s_below = signed_distance(SVector(0.0, -1.0), copen; sign_mode=:pseudonormal).distance
    @test s_above < 0
    @test s_below > 0

    square = load_curve_points([
        SVector(-1.0, -1.0),
        SVector(1.0, -1.0),
        SVector(1.0, 1.0),
        SVector(-1.0, 1.0),
    ]; closed=true)
    csq = build_signed_distance_cache(square)
    @test is_closed_curve(square)
    @test winding_number(SVector(0.0, 0.0), csq) != 0
    @test signed_distance(SVector(0.0, 0.0), csq; sign_mode=:winding).distance < 0
    @test signed_distance(SVector(3.0, 0.0), csq; sign_mode=:winding).distance > 0

    square_flip = CurveMesh{Float64}(square.points, [SVector{2,Int}(e[2], e[1]) for e in square.edges])
    cflip = build_signed_distance_cache(square_flip)
    s1 = signed_distance(SVector(0.0, 1.8), csq; sign_mode=:pseudonormal).distance
    s2 = signed_distance(SVector(0.0, 1.8), cflip; sign_mode=:pseudonormal).distance
    @test isapprox(s1, -s2; atol=1e-12)
end

@testset "3D sign and winding" begin
    tri = SurfaceMesh{Float64}(
        [SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0)],
        [SVector(1, 2, 3)],
    )
    ctri = build_signed_distance_cache(tri)
    s_top = signed_distance(SVector(0.2, 0.2, 0.5), ctri; sign_mode=:pseudonormal).distance
    s_bot = signed_distance(SVector(0.2, 0.2, -0.5), ctri; sign_mode=:pseudonormal).distance
    @test s_top > 0
    @test s_bot < 0
    @test_throws ArgumentError signed_distance(SVector(0.2, 0.2, 0.5), ctri; sign_mode=:winding)

    tetra_pts = [
        SVector(0.0, 0.0, 0.0),
        SVector(1.0, 0.0, 0.0),
        SVector(0.0, 1.0, 0.0),
        SVector(0.0, 0.0, 1.0),
    ]
    tetra_faces = [
        SVector(1, 3, 2),
        SVector(1, 2, 4),
        SVector(2, 3, 4),
        SVector(3, 1, 4),
    ]
    tetra = SurfaceMesh{Float64}(tetra_pts, tetra_faces)
    ctet = build_signed_distance_cache(tetra)
    @test is_closed_surface(tetra)
    @test winding_number(SVector(0.1, 0.1, 0.1), ctet) > 0.5
    @test signed_distance(SVector(0.1, 0.1, 0.1), ctet; sign_mode=:winding).distance < 0
    @test signed_distance(SVector(2.0, 2.0, 2.0), ctet; sign_mode=:winding).distance > 0

    tetra_flip = SurfaceMesh{Float64}(tetra_pts, [SVector{3,Int}(f[1], f[3], f[2]) for f in tetra_faces])
    cflip = build_signed_distance_cache(tetra_flip)
    sa = signed_distance(SVector(2.0, 2.0, 2.0), ctet; sign_mode=:pseudonormal).distance
    sb = signed_distance(SVector(2.0, 2.0, 2.0), cflip; sign_mode=:pseudonormal).distance
    @test isapprox(sa, -sb; atol=1e-12)
end

@testset "AABB vs brute-force" begin
    rng = MersenneTwister(7)

    curve = sample_circle(1.0, 64)
    ccache = build_signed_distance_cache(curve; leafsize=4)
    for _ in 1:200
        q = SVector(rand(rng)*4 - 2, rand(rng)*4 - 2)
        s = signed_distance(q, ccache; sign_mode=:unsigned)
        best = Inf
        besti = typemax(Int)
        bestc = zero(SVector{2,Float64})
        for (ei, e) in enumerate(curve.edges)
            d2, c, _, _ = FrontIntrinsicOps.closest_point_segment(q, curve.points[e[1]], curve.points[e[2]])
            if d2 < best || (d2 == best && ei < besti)
                best = d2
                besti = ei
                bestc = c
            end
        end
        @test isapprox(abs(s.distance), sqrt(best); atol=1e-12)
        @test s.primitive == besti
        @test norm(s.closest - bestc) ≤ 1e-12
    end

    sphere = generate_icosphere(1.0, 1)
    scache = build_signed_distance_cache(sphere; leafsize=4)
    for _ in 1:120
        q = SVector(rand(rng)*3 - 1.5, rand(rng)*3 - 1.5, rand(rng)*3 - 1.5)
        s = signed_distance(q, scache; sign_mode=:unsigned)
        best = Inf
        besti = typemax(Int)
        bestc = zero(SVector{3,Float64})
        for (fi, f) in enumerate(sphere.faces)
            d2, c, _, _ = FrontIntrinsicOps.closest_point_triangle(q, sphere.points[f[1]], sphere.points[f[2]], sphere.points[f[3]])
            if d2 < best || (d2 == best && fi < besti)
                best = d2
                besti = fi
                bestc = c
            end
        end
        @test isapprox(abs(s.distance), sqrt(best); atol=1e-11)
        @test s.primitive == besti
        @test norm(s.closest - bestc) ≤ 1e-11
    end
end

@testset "On-surface and matrix batch" begin
    curve = sample_circle(1.0, 32)
    ccache = build_signed_distance_cache(curve)
    p0 = curve.points[1]
    em = (curve.points[curve.edges[1][1]] + curve.points[curve.edges[1][2]]) / 2
    S, _, C, _ = signed_distance([p0, em], ccache; sign_mode=:auto)
    @test abs(S[1]) ≤ 1e-12
    @test abs(S[2]) ≤ 1e-12
    @test norm(C[1] - p0) ≤ 1e-12

    surf = generate_icosphere(1.0, 1)
    scache = build_signed_distance_cache(surf)
    v = surf.points[1]
    f = surf.faces[1]
    cen = (surf.points[f[1]] + surf.points[f[2]] + surf.points[f[3]]) / 3
    Q = hcat(v, cen)
    S2, _, _, _ = signed_distance(Q, scache; sign_mode=:auto)
    @test abs(S2[1]) ≤ 1e-10
    @test abs(S2[2]) ≤ 1e-10
end

@testset "Convergence smoke" begin
    pts2d = [SVector(1.7, 0.0), SVector(0.2, 0.0), SVector(0.0, -1.3), SVector(0.6, 0.8)]
    c1 = sample_circle(1.0, 32)
    c2 = sample_circle(1.0, 128)
    s1 = [abs(signed_distance(p, c1; sign_mode=:unsigned).distance - (norm(p) - 1.0)) for p in pts2d]
    s2 = [abs(signed_distance(p, c2; sign_mode=:unsigned).distance - (norm(p) - 1.0)) for p in pts2d]
    @test mean(s2) < mean(s1)

    pts3d = [
        SVector(1.8, 0.0, 0.0),
        SVector(0.0, -1.4, 0.0),
        SVector(0.0, 0.0, 1.6),
        SVector(1.2, 1.2, 0.0),
        SVector(0.45, 0.3, 0.2),
    ]
    m1 = generate_icosphere(1.0, 1)
    m3 = generate_icosphere(1.0, 3)
    e1 = [abs(signed_distance(p, m1; sign_mode=:unsigned).distance - (norm(p) - 1.0)) for p in pts3d]
    e3 = [abs(signed_distance(p, m3; sign_mode=:unsigned).distance - (norm(p) - 1.0)) for p in pts3d]
    @test mean(e3) < mean(e1)
end
