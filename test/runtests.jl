using FastUnitaryEigenvalues
using Test

@testset "FastUnitaryEigenvalues.jl" begin
    @test FastUnitaryEigenvalues.greet_your_package_name() == "Hello FastUnitaryEigenvalues"
    @test FastUnitaryEigenvalues.greet_your_package_name() != "Hello world!"
end