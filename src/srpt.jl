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
    
    while (length(available_jobs) > 0)
        job_with_remaining_time = popfirst!(available_jobs)
        push!(job_sequence, job_with_remaining_time.job)
        push!(processing_time_sequence, job_with_remaining_time.remaining_time)        
    end

    objective, completion_times = evaluate_preemptive_solution_1_rj_sumCj(inst,job_sequence, processing_time_sequence)

    return Preemptive_solution_1_rj_sumCj(objective,job_sequence, processing_time_sequence,completion_times)
end
