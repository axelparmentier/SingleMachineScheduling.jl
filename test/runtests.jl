using SingleMachineScheduling
using Test

@testset "SingleMachineScheduling.jl" begin
    @testset verbose=true "1_rj_sumCj_exact.jl" begin
        include("1_rj_sumCj_exact.jl")
    end
end
