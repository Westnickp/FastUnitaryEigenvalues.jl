using FastUnitaryEigenvalues, LinearAlgebra, Test

function example(n)
    U = sample_factored_form(10)
    U2 = apply_ctchain(U, Matrix(I(10)).+0.0im)
    v1 = qr_iterate(U)
    v2 = eigvals(U2)
    v1 = sort(v1, lt = (x,y) -> real(x)==real(y) ? imag(x)<imag(y) : real(x)<real(y))
	v2 = sort(v2, lt = (x,y) -> real(x)==real(y) ? imag(x)<imag(y) : real(x)<real(y))
    return norm(v1 .- v2)
end

@testset "FastUnitaryEigenvalues.jl" begin
    @test example(4) < 1e-14
    @test example(6) < 1e-14
    @test example(8) < 1e-14
    @test example(10) < 1e-14
    @test example(12) < 1e-14
end