
"""
    function encoder_1_rj_sumCj(inst::Instance1_rj_sumCj{T}) where {T}

returns a `27 * inst.nb_jobs` matrix with the value of the features
"""
function encoder_1_rj_sumCj(inst::Instance1_rj_sumCj{T}) where {T}
    nb_features = 27
    X = zeros(nb_features, inst.nb_jobs)

    ## SP RT
    sorted = Vector{Int}(undef,inst.nb_jobs)

    # SPT
    sortperm!(sorted, inst.processing_times)
    println(sorted)
    for j in 1:inst.nb_jobs
        X[1,sorted[j]] = j / inst.nb_jobs
    end

    # SRT
    sortperm!(sorted, inst.release_times)
    for j in 1:inst.nb_jobs
        X[2,sorted[j]] = j / inst.nb_jobs
    end    

    #SP+RT
    sortperm!(sorted, inst.processing_times + inst.release_times)
    for j in 1:inst.nb_jobs
        X[3,sorted[j]] = j / inst.nb_jobs
    end 
    
    ## Ratio prop r_j / prop p_j
    sum_ri = sum(inst.release_times)
    sum_pi = sum(inst.processing_times)
    sum_ripi = sum_pi + sum_ri
    for j in 1:inst.nb_jobs
        X[4,j] = inst.release_times[j] / sum_ri * sum_pi / inst.processing_times[j]
        X[5,j] = 1 / X[4,j]
        X[6,j] = inst.release_times[j] / sum_ri
        X[7,j] = inst.processing_times[j] / sum_ri
        X[8,j] = (inst.release_times[j] + inst.processing_times[j])/sum_ri # This feature is useless
        X[9,j] = inst.release_times[j] / sum_pi
        X[10,j] = inst.processing_times[j] / sum_pi
        X[11,j] = (inst.release_times[j] + inst.processing_times[j])/sum_pi # This feature is useless
        X[12,j] = inst.release_times[j] / sum_ripi
        X[13,j] = inst.processing_times[j] / sum_ripi
        X[14,j] = (inst.release_times[j] + inst.processing_times[j])/sum_ripi # This feature is useless    
    end

    ## SRPT
    srpt_solution = srpt_1_rj_sumCj(inst)
    number_of_preemptions = -ones(Int,inst.nb_jobs)
    for j in srpt_solution.job_sequence
        number_of_preemptions[j] += 1
    end
    total_number_preemptions = length(srpt_solution.job_sequence) - inst.nb_jobs
    
    processing_time_before_first_preemption = Vector{T}(undef,inst.nb_jobs)
    processing_time_of_preempting = Vector{T}(undef,inst.nb_jobs)
    # preempting_job = [j for j in 1:inst.nb_jobs]
    for i in length(srpt_solution.job_sequence):-1:1
        j = srpt_solution.job_sequence[i]
        processing_time_before_first_preemption[j] = srpt_solution.processing_time_sequence[i]
        processing_time_of_preempting[j] = processing_time_before_first_preemption[j] 
        if (
            i < length(srpt_solution.job_sequence) 
            && srpt_solution.processing_time_sequence[i+1] < processing_time_of_preempting[j] 
        )
            processing_time_of_preempting[j] = srpt_solution.processing_time_sequence[i+1]
            # preempting_job[j] = srpt_solution.
        end
    end
    processing_time_minus_preempting = processing_time_before_first_preemption - processing_time_of_preempting
    sum_processing_time_minus_preempting = sum(processing_time_minus_preempting)
    sortperm!(sorted, srpt_solution.completion_times)
    
    for j in 1:inst.nb_jobs
        # processing_time_minus_preempting
        X[15,j] = processing_time_minus_preempting[j] / sum_processing_time_minus_preempting
        X[16,j] = X[15,j] / processing_time_of_preempting[j]
        X[17,j] = X[15,j] / inst.processing_times[j]

        # Deciles 
        X[18, j] = X[2,j] # probably useless
        X[19, j] = inst.release_times[j] /(1+X[2,j]) 
        X[20,j] = X[1,j] # probably useless
        X[21,j] = inst.processing_times[j] / (1+X[1,j])  
        
        # Nb preemptions
        X[22,j] = number_of_preemptions[j] / total_number_preemptions

        # SRPT position
        X[23,j] = sorted[j] / inst.nb_jobs
    end
    
    # SRPT cardinal
    nb_smaller_p_before = zeros(Int,inst.nb_jobs)
    nb_smaller_r_before = zeros(Int,inst.nb_jobs)
    nb_larger_p_before = zeros(Int,inst.nb_jobs)
    nb_larger_r_before = zeros(Int,inst.nb_jobs)
    for j in 1:inst.nb_jobs
        for k in sorted
            if j == k
                break
            else
                if inst.processing_times[k] < inst.processing_times[j]
                    nb_smaller_p_before[j] += 1
                elseif inst.processing_times[k] > inst.processing_times[j]
                    nb_larger_p_before[j] += 1
                end

                if inst.release_times[k] < inst.release_times[j]
                    nb_smaller_r_before[j] += 1
                elseif inst.release_times[k] > inst.release_times[j]
                    nb_larger_r_before[j] += 1
                end
            end
        end
    end

    sum_nb_smaller_p_before = sum(nb_smaller_p_before)
    sum_nb_smaller_r_before = sum(nb_smaller_r_before)
    sum_nb_larger_p_before = sum(nb_larger_p_before)
    sum_nb_larger_r_before = sum(nb_larger_r_before)

    for j in 1:inst.nb_jobs
        X[24,j] = nb_smaller_p_before[j] / sum_nb_smaller_p_before
        X[25,j] = nb_smaller_r_before[j] / sum_nb_smaller_r_before
        X[26,j] = nb_larger_p_before[j] / sum_nb_larger_p_before
        X[27,j] = nb_larger_r_before[j] / sum_nb_larger_r_before
    end

    return X
end