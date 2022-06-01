
struct Instance1_rj_sumCj{T} 
    nb_jobs::Int
    release_times::Vector{T}
    processing_times::Vector{T}
end

function build_instance_1_rj_sumCj(;seed::Int=0,nb_jobs::Int=10,range::Float64=0.8)
    Random.seed!(seed)
    release_times = Vector{Int}(undef,nb_jobs)
    processing_times = Vector{Int}(undef,nb_jobs)
    for j in 1:nb_jobs
        processing_times[j] = abs(rand(Int,1)[1]) % 100 + 1
        release_times[j] = ceil(rand(1)[1] * 50.5 * nb_jobs * range)
    end
    return Instance1_rj_sumCj(nb_jobs, release_times, processing_times)
end

function evaluate_solution_1_rj_sumCj(inst::Instance1_rj_sumCj,sol::Vector{I}) where {I <: Integer}
    current_completion_time = zero(inst.release_times[1])
    objective = zero(inst.release_times[1])
    for j in 1:inst.nb_jobs
        job = sol[j]
        current_completion_time = max(current_completion_time, inst.release_times[job]) + inst.processing_times[job]
        objective += current_completion_time
    end
    return objective
end

function milp_solve_1_rj_sumCj(inst::Instance1_rj_sumCj; MILP_solver=GLPK.Optimizer)
    model = Model(MILP_solver)

    @variable(model,x[1:inst.nb_jobs,1:inst.nb_jobs],Bin)
    @variable(model,C[1:inst.nb_jobs] >= 0)

    @objective(model,Min,sum(C[j] for j in 1:inst.nb_jobs))

    @constraint(model, job_in_single_position[i in 1:inst.nb_jobs], sum(x[i,j] for j in 1:inst.nb_jobs) == 1)
    @constraint(model, single_job_in_position[j in 1:inst.nb_jobs], sum(x[i,j] for i in 1:inst.nb_jobs) == 1)
    @constraint(model, first_job_completion_time, C[1] == sum((inst.processing_times[i] + inst.release_times[i]) * x[i,1] for i in 1:inst.nb_jobs))
    @constraint(model, job_completion_time_and_previous_job[j in 2:inst.nb_jobs], C[j] >= C[j-1] + sum(inst.processing_times[i] * x[i,j] for i in 1:inst.nb_jobs))
    @constraint(model, job_completion_time_and_release[j in 2:inst.nb_jobs], C[j] >= sum((inst.processing_times[i] + inst.release_times[i]) * x[i,j] for i in 1:inst.nb_jobs))

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