"""
    struct Instance1_rj_sumCj{T} 
        nb_jobs::Int
        release_times::Vector{T}
        processing_times::Vector{T}
    end
"""
struct Instance1_rj_sumCj{T} 
    nb_jobs::Int
    release_times::Vector{T}
    processing_times::Vector{T}
end

function Instance1_rj_sumCj(my_dict::AbstractDict)
    return Instance1_rj_sumCj(
        my_dict["nb_jobs"],
        my_dict["release_times"],
        my_dict["processing_times"]
    )
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
