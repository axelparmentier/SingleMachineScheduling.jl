

"""
    function milp_solve_1_rj_sumCj(
        inst::Instance1_rj_sumCj{T}; 
        srpt_cuts=true, 
        MILP_solver=GLPK.Optimizer
    ) where {T}


returns `objective_value(model), solution` where `solution` is a permutation enoced as a `Vector{Int}`
"""
function milp_solve_1_rj_sumCj(
    inst::Instance1_rj_sumCj{T}; 
    srpt_cuts=true, 
    MILP_solver=GLPK.Optimizer
    ) where {T}
    
    model = Model(MILP_solver)

    @variable(model,x[1:inst.nb_jobs,1:inst.nb_jobs],Bin)
    @variable(model,C[1:inst.nb_jobs] >= 0)

    @objective(model,Min,sum(C[j] for j in 1:inst.nb_jobs))

    @constraint(model, job_in_single_position[i in 1:inst.nb_jobs], sum(x[i,j] for j in 1:inst.nb_jobs) == 1)
    @constraint(model, single_job_in_position[j in 1:inst.nb_jobs], sum(x[i,j] for i in 1:inst.nb_jobs) == 1)
    @constraint(model, first_job_completion_time, C[1] == sum((inst.processing_times[i] + inst.release_times[i]) * x[i,1] for i in 1:inst.nb_jobs))
    @constraint(model, job_completion_time_and_previous_job[j in 2:inst.nb_jobs], C[j] >= C[j-1] + sum(inst.processing_times[i] * x[i,j] for i in 1:inst.nb_jobs))
    @constraint(model, job_completion_time_and_release[j in 2:inst.nb_jobs], C[j] >= sum((inst.processing_times[i] + inst.release_times[i]) * x[i,j] for i in 1:inst.nb_jobs))

    if srpt_cuts
        srpt_solution = srpt_1_rj_sumCj(inst)
        sorted_completion_times = sort(srpt_solution.completion_times)
        println(sorted_completion_times)
        @assert length(sorted_completion_times) == inst.nb_jobs
        @constraint(model, srpt_completion[j in 1:inst.nb_jobs], C[j] >= sorted_completion_times[j])
    end

    optimize!(model)

    solution = Int[]
    for j in 1:inst.nb_jobs
        for i in 1:inst.nb_jobs
            if abs(value(x[i,j]) - 1) < 0.00001
                push!(solution,i)
            end
        end
    end

    return objective_value(model), solution
end