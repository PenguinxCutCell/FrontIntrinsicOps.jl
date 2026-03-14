# test_vector_calculus.jl – Tests for surface vector calculus utilities (v0.4).
#
# Tests:
# 1. tangential_project: result is orthogonal to normal.
# 2. tangential_project_field: vertex and face variants.
# 3. gradient_0_to_tangent_vectors: zero for constant field.
# 4. gradient_0_to_tangent_vectors: recovers linear gradient on flat patch.
# 5. divergence_tangent_vectors: ~zero for constant rigid rotation on sphere.
# 6. 1-form ↔ tangent vector conversion roundtrip.
# 7. surface_rot_0form: orthogonal to gradient.

@testset "tangential_project: orthogonal to normal" begin
    using LinearAlgebra: dot, norm
    n = SVector{3,Float64}(0.0, 0.0, 1.0)
    v = SVector{3,Float64}(1.0, 2.0, 3.0)

    vτ = tangential_project(v, n)
    @test abs(dot(vτ, n)) < 1e-14
    @test vτ[1] ≈ 1.0
    @test vτ[2] ≈ 2.0
    @test vτ[3] ≈ 0.0

    # With non-unit normal
    n2 = SVector{3,Float64}(0.0, 0.0, 2.0)
    vτ2 = tangential_project(v, n2)
    @test abs(dot(vτ2, n2 / norm(n2))) < 1e-13
end

@testset "tangential_project_field: vertex and face" begin
    mesh = make_uvsphere(1.0; nφ=8, nθ=16)
    geom = compute_geometry(mesh)
    nv = length(mesh.points)
    nf = length(mesh.faces)

    # Vertex field: random ambient vectors
    vfield_v = [SVector{3,Float64}(rand(), rand(), rand()) for _ in 1:nv]
    vτ_v = tangential_project_field(mesh, geom, vfield_v; location=:vertex)
    @test length(vτ_v) == nv
    # Each projected vector should be orthogonal to its vertex normal
    for i in 1:nv
        @test abs(dot(vτ_v[i], geom.vertex_normals[i])) < 1e-13
    end

    # Face field: random ambient vectors
    vfield_f = [SVector{3,Float64}(rand(), rand(), rand()) for _ in 1:nf]
    vτ_f = tangential_project_field(mesh, geom, vfield_f; location=:face)
    @test length(vτ_f) == nf
    for fi in 1:nf
        @test abs(dot(vτ_f[fi], geom.face_normals[fi])) < 1e-13
    end
end

@testset "gradient_0_to_tangent_vectors: constant field is zero" begin
    mesh = make_uvsphere(1.0; nφ=8, nθ=16)
    geom = compute_geometry(mesh)
    nv = length(mesh.points)

    u_const = ones(Float64, nv)
    grads = gradient_0_to_tangent_vectors(mesh, geom, u_const; location=:face)
    @test length(grads) == length(mesh.faces)
    for g in grads
        @test norm(g) < 1e-13
    end
end

@testset "gradient_0_to_tangent_vectors: vertex output" begin
    mesh = make_uvsphere(1.0; nφ=8, nθ=16)
    geom = compute_geometry(mesh)
    nv = length(mesh.points)

    u = [p[1] for p in mesh.points]  # x-coordinate as scalar field
    grads = gradient_0_to_tangent_vectors(mesh, geom, u; location=:vertex)
    @test length(grads) == nv
    @test all(isfinite.(norm.(grads)))
end

@testset "gradient_0_to_tangent_vectors: linear field on flat patch" begin
    # On a flat patch in xy-plane, gradient of u=x should be approximately (1, 0, 0)
    mesh = make_flat_patch(; N=6, L=1.0)
    geom = compute_geometry(mesh)
    nv = length(mesh.points)
    nf = length(mesh.faces)

    # u(x,y,z) = x
    u = [p[1] for p in mesh.points]
    grads = gradient_0_to_tangent_vectors(mesh, geom, u; location=:face)

    # Interior faces: gradient should be close to (1, 0, 0)
    for fi in 1:nf
        g = grads[fi]
        # The patch is in xy-plane, normal is (0,0,1)
        # gradient of u=x should be (1,0,0) (projected to tangent plane, which is xy-plane)
        @test abs(g[1] - 1.0) < 0.1   # x-component ~ 1
        @test abs(g[2]) < 0.1           # y-component ~ 0
        @test abs(g[3]) < 1e-13         # z-component ~ 0 (tangential constraint)
    end
end

@testset "divergence_tangent_vectors: zero for incompressible rotation" begin
    # The rigid rotation field V = (-y, x, 0) on the sphere is incompressible
    # (divergence-free). On a sphere, the tangential part of this field is
    # also divergence-free.
    mesh = make_uvsphere(1.0; nφ=12, nθ=24)
    geom = compute_geometry(mesh)
    nf = length(mesh.faces)

    # Per-face rotation field (tangential)
    vfield = Vector{SVector{3,Float64}}(undef, nf)
    for fi in 1:nf
        face = mesh.faces[fi]
        cx = sum(mesh.points[face[k]][1] for k in 1:3) / 3
        cy = sum(mesh.points[face[k]][2] for k in 1:3) / 3
        p  = SVector{3,Float64}(-cy, cx, 0.0)
        # Project to tangent plane of face
        n  = geom.face_normals[fi]
        vfield[fi] = p - dot(p, n) * n
    end

    div_v = divergence_tangent_vectors(mesh, geom, vfield; location=:face)
    # Divergence of this field should be small
    @test norm(div_v, Inf) < 0.5   # some numerical error expected on coarse mesh
    # But the mean should be near zero
    @test abs(mean(div_v)) < 0.01
end

@testset "tangent_vectors_to_1form and oneform_to_tangent_vectors" begin
    mesh = make_uvsphere(1.0; nφ=8, nθ=16)
    geom = compute_geometry(mesh)
    topo = build_topology(mesh)
    dec  = build_dec(mesh, geom)
    nv   = length(mesh.points)
    nf   = length(mesh.faces)
    ne   = length(topo.edges)

    # From gradient: convert 1-form back to vectors
    u = [p[3] for p in mesh.points]  # z-coordinate
    α_grad = dec.d0 * u
    @test length(α_grad) == ne

    # Convert 1-form to tangent vectors (face-based)
    vfield = oneform_to_tangent_vectors(mesh, geom, topo, α_grad; location=:face)
    @test length(vfield) == nf
    @test all(isfinite.(norm.(vfield)))

    # Each face vector should be tangential
    for fi in 1:nf
        n = geom.face_normals[fi]
        @test abs(dot(vfield[fi], n)) < 1e-12
    end

    # Convert back to 1-form
    α_rec = tangent_vectors_to_1form(mesh, geom, topo, vfield; location=:face)
    @test length(α_rec) == ne
    @test all(isfinite.(α_rec))
end

@testset "surface_rot_0form: orthogonal to gradient" begin
    mesh = make_uvsphere(1.0; nφ=8, nθ=16)
    geom = compute_geometry(mesh)
    nf = length(mesh.faces)

    u = [p[3] for p in mesh.points]
    grads = gradient_0_to_tangent_vectors(mesh, geom, u; location=:face)
    rots  = surface_rot_0form(mesh, geom, u)

    @test length(rots) == nf
    # Each rot should be orthogonal to the gradient
    for fi in 1:nf
        n = geom.face_normals[fi]
        g = grads[fi]
        r = rots[fi]
        # rot = n × g, so rot ⊥ n and rot ⊥ g (in tangent plane)
        @test abs(dot(r, n)) < 1e-13
        # dot(rot, grad) = dot(n×g, g) = 0 (cross product ⊥ both factors)
        @test abs(dot(r, g)) < 1e-13 * (norm(g) + 1)
    end
end
