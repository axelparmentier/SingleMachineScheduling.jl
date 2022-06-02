""" function fast_local_descent_1_rj_sumCj!(inst::Instance1_rj_sumCj,sol::Vector{I}) where {I}

Performs a fast local descent on the solution `sol` of instance `inst`. Solution `sol` is modified and contains the result at the end.
"""
function fast_local_descent_1_rj_sumCj!(inst::Instance1_rj_sumCj{T},sol::AbstractVector) where {T}
	completion_times = Vector{Int}(undef, inst.nb_jobs)
	current_index = 1  # current index considered in loop
	current_completion_time = zero(T) # completion time of job at this index in solution
    current_job = -1
	while (current_index < inst.nb_jobs)
		if inst.release_times[sol[current_index]] > current_completion_time
            current_completion_time = inst.release_times[sol[current_index]]
        end
		if (
            inst.release_times[sol[current_index + 1]] <= current_completion_time
			&& 
                (
                    inst.processing_times[sol[current_index]] > inst.processing_times[sol[current_index + 1]]
			        || 
                    (
                        inst.processing_times[sol[current_index]] == inst.processing_times[sol[current_index + 1]]
                        && inst.release_times[sol[current_index + 1]]<inst.release_times[sol[current_index]]
                    )
                )
            )
		    # We swap
			current_job = sol[current_index];
			sol[current_index] = sol[current_index + 1];
			sol[current_index + 1] = current_job;
			if (current_index > 2) 
				current_completion_time = completion_times[current_index - 2];
				current_index -= 1
            end
			if (current_index == 2)
				current_completion_time = zero(T)
				current_index -= 1
            end
			if (current_index == 1) 
                current_completion_time = zero(T)
            end
		else
			current_completion_time += inst.processing_times[sol[current_index]]
			completion_times[current_index] = current_completion_time
			current_index += 1
        end
	end
end

## Release Data Iteration heuristic (RDI)

function rdi_local_search_subrouting_from_release_time!(
    inst::Instance1_rj_sumCj{T}, 
    releaseTimesUsed::Vector{T}, 
    dispatching_rule::AbstractVector, 
    sol::AbstractVector
) where {T}
	current_job = -1
    completion_times = Vector{T}(undef,inst.nb_jobs)
	current_completion_time = zero(T)
	current_index = 1

	# Sort by increasing release times
    sortperm!(sol,releaseTimesUsed)

	while (current_index< inst.nb_jobs)
		# Swap job in current_index with the next one if both have release time smaller than current end current_completion_time and the next one has a smaller easyProblemProcessingTime
		if (
                releaseTimesUsed[sol[current_index]] <= current_completion_time 
                && releaseTimesUsed[sol[current_index + 1]] <= current_completion_time
				&& dispatching_rule[sol[current_index]]>dispatching_rule[sol[current_index + 1]]
            )
			current_job = sol[current_index];
			sol[current_index] = sol[current_index + 1];
			sol[current_index + 1] = current_job;
			# get back to previous position because the schedule has changed
			if (current_index > 2)
				current_completion_time = completion_times[current_index - 2];
				current_index -= 1;
			else
				if (current_index == 2)
					current_index -= 1;
                end
				current_completion_time = zero(T);
            end	
		else
			if (releaseTimesUsed[sol[current_index]] > current_completion_time) 
                current_completion_time = releaseTimesUsed[sol[current_index]]; # Remark that current_index is not increased in that case
			else 
				current_completion_time += dispatching_rule[sol[current_index]];
				completion_times[current_index] = current_completion_time;
				current_index += 1
            end
		end
	end
end
"""
	function rdi!(
		inst::Instance1_rj_sumCj{T}, 
		dispatching_rule::AbstractVector, 
		sol::AbstractVector
	) where {T}

Release Data iteration heuristic

"""
function rdi!(
    inst::Instance1_rj_sumCj{T}, 
    dispatching_rule::AbstractVector, 
    sol::AbstractVector
) where {T}
	current_release_times_used = deepcopy(inst.release_times)
	swapped_release_times_used = deepcopy(current_release_times_used)
	best_solution = deepcopy(sol)
	current_index = -1
    current_solution_value = zero(T)
	best_solution_value = evaluate_solution_1_rj_sumCj(inst,sol);
	current_index=1;
	while (current_index < inst.nb_jobs)
		if (inst.processing_times[current_index] > inst.processing_times[(current_index + 1)])
		
			splice!(swapped_release_times_used,1:inst.nb_jobs,current_release_times_used)
			swapped_release_times_used[current_index] = swapped_release_times_used[current_index + 1];
			rdi_local_search_subrouting_from_release_time!(inst,swapped_release_times_used,dispatching_rule,sol)
            fast_local_descent_1_rj_sumCj!(inst,sol)
			current_solution_value = evaluate_solution_1_rj_sumCj(inst,sol);
			
			if (current_solution_value < best_solution_value)
				best_solution_value = current_solution_value;
                splice!(current_release_times_used, 1:inst.nb_jobs, swapped_release_times_used)
				splice!(best_solution,1:inst.nb_jobs,sol);
				current_index = 1;
			else 
                current_index += 1
            end
		else 
            current_index += 1
		end
	end
    splice!(sol,1:inst.nb_jobs,best_solution);
end

# Alternative Priority Rule for Total Flowtime (APRTF)

current_release_time(inst::Instance1_rj_sumCj{T}, current_time::T, job_index) where {T} = max(current_time, inst.release_times[job_index])

"""
	prtf(inst::Instance1_rj_sumCj{T}, current_time::T, job_index) where {T}

Priority Rule for Total Flowtime (PRTF)
"""
prtf(inst::Instance1_rj_sumCj{T}, current_time::T, job_index) where {T} = (2 * current_release_time(inst, current_time,job_index) + inst.processing_times[job_index]);

"""
	function remove!(a, item)

removes all occurences of `item` in `a`
"""
function remove!(a, item)
    deleteat!(a, findall(x->x==item, a))
end

"""
	function aprtf(inst::Instance1_rj_sumCj{T}) where {T}

Implements aprtf as in [Chu, 1992, Efficient heuristics to minimize total flowtime with release dates](https://www.sciencedirect.com/science/article/pii/016763779290092H). See this reference for the notations and the details of the algorithm

returns `obj_value, sol`
"""
function aprtf(inst::Instance1_rj_sumCj{T}) where {T}
	remaining_jobs = [i for i in 1:inst.nb_jobs]
	sol = Int[]
	current_time = zero(T)
	obj_value = zero(T)

	while (length(remaining_jobs) > 0)
		# Select one job to schedule
		best_prtf_val = best_current_release_time_val = typemax(T)
		best_job_prtf = -1 # Denoted α in [Chu, 1992]
		for job in remaining_jobs
			if (
				prtf(inst,current_time, job) < best_prtf_val 
				|| 
				(
					prtf(inst,current_time, job) == best_prtf_val 
					&& current_release_time(inst,current_time, job) < best_current_release_time_val
				)
			)
				best_prtf_val = prtf(inst,current_time, job);
				best_current_release_time_val = current_release_time(inst,current_time, job);
				best_job_prtf = job;
			end
		end
		best_current_release_time_val = best_processing_time = typemax(T)
		best_job_spt = -1 # Denoted β in [Chu, 1992]
		for job in remaining_jobs
			if (
				current_release_time(inst, current_time,job) < best_current_release_time_val 
				|| 
				(
					current_release_time(inst, current_time, job) == best_current_release_time_val 
					&& inst.processing_times[job] < best_processing_time
				)
			)
				best_current_release_time_val = current_release_time(inst, current_time, job);
				best_processing_time = inst.processing_times[job]
				best_job_spt = job
			end
		end

		# Compute constant for step 3 of APrTF in [Chu, 1992]
		μ = length(remaining_jobs) # See [Chu, 1992] for definition of μ
		if best_job_spt != best_job_prtf 
			μ -= 2;
		else 
			μ -= 1;
		end

		τ = typemax(T) # See [Chu, 1992] for definition of τ
		for job in remaining_jobs
			if (inst.release_times[job] < τ && job != best_job_prtf && job != best_job_spt) 
				τ = inst.release_times[job];
			end
		end
		
		F_αβ = current_release_time(inst, current_time, best_job_prtf) + inst.processing_times[best_job_prtf] - inst.release_times[best_job_prtf]
		C_βα = current_release_time(inst, F_αβ, best_job_spt) + inst.processing_times[best_job_spt] - inst.release_times[best_job_spt] # C_β(C_α(Δ)) in [Chu, 1992] notations
		F_αβ += current_release_time(inst, F_αβ, best_job_spt) + inst.processing_times[best_job_spt] - inst.release_times[best_job_spt]
		F_βα = current_release_time(inst, current_time, best_job_spt) + inst.processing_times[best_job_spt] - inst.release_times[best_job_spt]
		F_βα += current_release_time(inst, F_βα, best_job_prtf) + inst.processing_times[best_job_prtf] - inst.release_times[best_job_prtf]
		R_α = current_release_time(inst, current_time, best_job_prtf)
		R_β = current_release_time(inst, current_time, best_job_spt)
		
		# step 3 of APrTF in [Chu, 1992]: Choose the next job to be scheduled
		if μ * min(R_α - R_β, C_βα - τ) <= F_βα - F_αβ
			# schedule job best_job_prtf
			current_time = current_release_time(inst, current_time, best_job_prtf) + inst.processing_times[best_job_prtf]
			obj_value += current_time
			push!(sol, best_job_prtf)
			remove!(remaining_jobs,best_job_prtf)
		else
			# Schedule job best_job_spt
			current_time = current_release_time(inst, current_time, best_job_spt) + inst.processing_times[best_job_spt]
			obj_value += current_time;
			push!(sol, best_job_spt)
			remove!(remaining_jobs,best_job_spt)
		end
	end
	return obj_value, sol
end

function jobs_positions_in_sol(sol)
	result = Vector{Int}(undef, length(sol))
	for (pos,job) in enumerate(sol)
		result[job] = pos
	end
	return result
end