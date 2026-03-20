using Test
using FrontIntrinsicOps

@testset "PointFront1D: constructor and checks" begin
    f1 = PointFront1D([0.3], true)
    @test FrontIntrinsicOps.check(f1)
    @test f1.x == [0.3]

    f2 = PointFront1D([0.2, 0.8], false)
    @test FrontIntrinsicOps.check(f2)
    @test f2.x == [0.2, 0.8]

    @test_throws ArgumentError PointFront1D(Float64[], true)
    @test_throws ArgumentError PointFront1D([0.1, 0.3, 0.9], true)
    @test_throws ArgumentError PointFront1D([0.4, 0.4], true)
    @test_throws ArgumentError PointFront1D([0.8, 0.2], true)
    @test_throws ArgumentError PointFront1D([NaN], true)
    @test_throws ArgumentError PointFront1D([Inf], true)
end

@testset "PointFront1D: single-marker signed distance" begin
    xg = 0.3
    fr = single_marker_front(xg; inside_right=true)
    fl = single_marker_front(xg; inside_right=false)

    @test signed_distance(fr, 0.3) == 0.0
    @test signed_distance(fr, 0.7) < 0
    @test signed_distance(fr, 0.0) > 0

    @test signed_distance(fl, 0.3) == 0.0
    @test signed_distance(fl, 0.7) > 0
    @test signed_distance(fl, 0.0) < 0
end

@testset "PointFront1D: two-marker signed distance (interval inside)" begin
    f = interval_front(0.2, 0.8; interval_is_inside=true)
    @test signed_distance(f, 0.2) == 0.0
    @test signed_distance(f, 0.8) == 0.0
    @test signed_distance(f, 0.5) < 0
    @test signed_distance(f, 0.1) > 0
    @test signed_distance(f, 0.9) > 0
    @test isapprox(abs(signed_distance(f, 0.1)), 0.1; atol=1e-12)
    @test isapprox(abs(signed_distance(f, 0.5)), 0.3; atol=1e-12)
end

@testset "PointFront1D: two-marker signed distance (interval outside)" begin
    f = interval_front(0.2, 0.8; interval_is_inside=false)
    @test signed_distance(f, 0.2) == 0.0
    @test signed_distance(f, 0.8) == 0.0
    @test signed_distance(f, 0.5) > 0
    @test signed_distance(f, 0.1) < 0
    @test signed_distance(f, 0.9) < 0
end

@testset "PointFront1D: interface normals" begin
    @test interface_normals(single_marker_front(0.3; inside_right=true)) == [1.0]
    @test interface_normals(single_marker_front(0.3; inside_right=false)) == [-1.0]
    @test interface_normals(interval_front(0.2, 0.8; interval_is_inside=true)) == [-1.0, 1.0]
    @test interface_normals(interval_front(0.2, 0.8; interval_is_inside=false)) == [1.0, -1.0]
end

@testset "PointFront1D: grid rebuild" begin
    f = interval_front(0.2, 0.8; interval_is_inside=true)
    xs = collect(range(0.0, 1.0; length=17))
    phi = rebuild_signed_distance(f, xs)
    phi_ref = [signed_distance(f, x) for x in xs]
    @test phi == phi_ref
end
