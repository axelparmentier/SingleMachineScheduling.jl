"""
    Instance1_rj_sumCj{T} 
    
    nb_jobs::Int
    release_times::Vector{T}
    processing_times::Vector{T}
"""
struct Instance1_rj_sumCj{T} 
    nb_jobs::Int
    release_times::Vector{T}
    processing_times::Vector{T}
end

"""
    function build_instance_1_rj_sumCj(;seed::Int=0,nb_jobs::Int=10,range::Float64=0.8)

generates an `Instance1_rj_sumCj{T}` instance with `nb_jobs` jobs

 - `seed` is an Integer
 - `range` is a float between 0.2 and 3 typically. Most difficult instances are obtained with range around 0.8
"""
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

"""
    function evaluate_solution_1_rj_sumCj(inst::Instance1_rj_sumCj,sol::Vector{I}) where {I <: Integer}

returns the objective value for the solution (permutation of the indices) encoded in sol
"""
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

function evaluate_solution_1_rj_sumCj_completion_times(inst::Instance1_rj_sumCj{T},sol::Vector{I}) where {T, I <: Integer}
    current_completion_time = zero(inst.release_times[1])
    completion_times = deepcopy(inst.processing_times)
    for j in 1:inst.nb_jobs
        job = sol[j]
        current_completion_time = max(current_completion_time, inst.release_times[job]) + inst.processing_times[job]
        completion_times[job] = current_completion_time
    end

    return completion_times
end

struct Preemptive_solution_1_rj_sumCj{T}
    obj_value::T
    job_sequence::Vector{Int}
    processing_time_sequence::Vector{T}
    completion_times::Vector{T}
end

function evaluate_preemptive_solution_1_rj_sumCj(
    inst::Instance1_rj_sumCj{T},
    job_sequence::Vector{Int},
    processing_time_sequence::Vector{T}
    ) where {T}
    @assert length(job_sequence) == length(processing_time_sequence)
    current_time = zero(T)
    completion_times = zeros(T, inst.nb_jobs)
    remaining_times = deepcopy(inst.processing_times)

    objective = zero(T)

    count = 0

    for (index,job) in enumerate(job_sequence)
        remaining_times[job] -= processing_time_sequence[index]
        if current_time < inst.release_times[job]
            current_time = inst.release_times[job]
        end
        current_time += processing_time_sequence[index]
        if findnext(j -> j== job, job_sequence, index+1) == nothing
            objective += current_time
            completion_times[job] = current_time
            count += 1
        end
    end

    @assert count == inst.nb_jobs

    for t in remaining_times
        @assert abs(t) < 0.00001
    end

    return objective, completion_times
end

mutable struct JobWithRemainingProcessingTime{T}
    job::Int
    remaining_time::T
end


"""
function srpt_1_rj_sumCj(inst::Instance1_rj_sumCj{T}) where {T}

computes the preemptive solution of inst, and returns

    struct Preemptive_solution_1_rj_sumCj{T}
        obj_value::T
        job_sequence::Vector{Int}
        processing_time_sequence::Vector{T}
        completion_times::Vector{T}
    end

where obj_value is the preemptive objective value
"""
function srpt_1_rj_sumCj(inst::Instance1_rj_sumCj{T}) where {T}

    job_sequence = Int[]
    processing_time_sequence = T[]

    non_available_jobs_per_release_time = sortperm(inst.release_times)
    println(non_available_jobs_per_release_time)
    available_jobs = JobWithRemainingProcessingTime[]

    current_time = zero(T)

    while (length(non_available_jobs_per_release_time) > 0)        
        # add_jobs_that_have_become_ready!()
        
        while length(non_available_jobs_per_release_time) > 0 && inst.release_times[non_available_jobs_per_release_time[1]] <= current_time 
            j = popfirst!(non_available_jobs_per_release_time)
            push!(available_jobs, JobWithRemainingProcessingTime(j , inst.processing_times[j]))
        end
        sort!(available_jobs,by = jp -> jp.remaining_time)

        if length(available_jobs) == 0
            # No availabe job to schedule : we directly add the first available one
            j = popfirst!(non_available_jobs_per_release_time)
            current_time = inst.release_times[j]
            push!(available_jobs, JobWithRemainingProcessingTime(j , inst.processing_times[j]))
        else
            # Test if the current best job is going to be preempted
            preempted = false
            preemption_time = zero(T)
            for j in non_available_jobs_per_release_time
                if inst.release_times[j] >= current_time + available_jobs[1].remaining_time
                    break
                end
                if inst.release_times[j] + inst.processing_times[j]  < current_time + available_jobs[1].remaining_time 
                    preempted = true
                    preemption_time = inst.release_times[j]
                    break
                end
            end

            # Schedule current best job 
            if preempted
                scheduled_time = preemption_time - current_time
                @assert scheduled_time > 0
                @assert scheduled_time <= inst.processing_times[available_jobs[1].job]
                available_jobs[1].remaining_time -= scheduled_time
                push!(job_sequence, available_jobs[1].job)
                push!(processing_time_sequence, scheduled_time)
                current_time = preemption_time

            else
                push!(job_sequence, available_jobs[1].job)
                push!(processing_time_sequence, available_jobs[1].remaining_time)
                current_time += available_jobs[1].remaining_time
                popfirst!(available_jobs)
            end
        end
    end
    
    println(available_jobs)

    while (length(available_jobs) > 0)
        job_with_remaining_time = popfirst!(available_jobs)
        push!(job_sequence, job_with_remaining_time.job)
        push!(processing_time_sequence, job_with_remaining_time.remaining_time)        
    end

    objective, completion_times = evaluate_preemptive_solution_1_rj_sumCj(inst,job_sequence, processing_time_sequence)

    return Preemptive_solution_1_rj_sumCj(objective,job_sequence, processing_time_sequence,completion_times)
end


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