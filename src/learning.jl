"""
    function nb_features_encoder(encoder)

returns the number of features used by the encoder
"""
nb_features_encoder(encoder) = -1

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
    
    sum_ri_div_pi = sum([inst.release_times[i]/inst.processing_times[i] for i in 1:inst.nb_jobs])
    sum_pi_div_ri = sum([inst.processing_times[i]/inst.release_times[i] for i in 1:inst.nb_jobs])

    for j in 1:inst.nb_jobs
        X[4,j] =  inst.release_times[j] / (inst.processing_times[j] * sum_ri_div_pi) # inst.release_times[j] / sum_ri * sum_pi / inst.processing_times[j] # Not equal
        X[5,j] = inst.processing_times[j] / (inst.release_times[j] * sum_pi_div_ri) # 1 / X[4,j] # Not equal
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
        processing_time_of_preempting[j] = inst.processing_times[j] 
        if (
            i < length(srpt_solution.job_sequence) 
            && srpt_solution.processing_time_sequence[i+1] < processing_time_of_preempting[j] 
        )
            processing_time_of_preempting[j] = inst.processing_times[srpt_solution.job_sequence[i+1]]
            # preempting_job[j] = srpt_solution.
        end
    end
    # processing_time_minus_preempting = processing_time_before_first_preemption - processing_time_of_preempting
    # sum_processing_time_minus_preempting = sum(processing_time_minus_preempting)
    processing_time_minus_before_preemption = inst.processing_times - processing_time_before_first_preemption
    sum_processing_time_minus_before_preemption = sum(processing_time_minus_before_preemption)
    sortperm!(sorted, srpt_solution.completion_times)

    ratio__processing_time_before_first_preemption__prempting = [processing_time_minus_before_preemption[j] / processing_time_of_preempting[j] for j in 1:inst.nb_jobs]
    sum_ratio__processing_time_before_first_preemption__prempting = sum(ratio__processing_time_before_first_preemption__prempting)

    ratio__processing_time_before_first_preemption__processing = [processing_time_minus_before_preemption[j] / inst.processing_times[j] for j in 1:inst.nb_jobs]
    sum_ratio__processing_time_before_first_preemption__processing = sum(ratio__processing_time_before_first_preemption__processing)
    
    deciles_release = quantile(inst.release_times,0.0:0.1:1.0)
    deciles_processing = quantile(inst.processing_times,0.0:0.1:1.0)
    function decile_of_x_in_v(x,v)
        for i in 1:10
            d = i-1
            if v[i] <= x && x <= v[i+1]
                return d
            end
        end
        return 9
    end
    
    
    for j in 1:inst.nb_jobs
        # processing_time_minus_preempting
        X[15,j] = processing_time_minus_before_preemption[j] / sum_processing_time_minus_before_preemption #Not equal
        X[16,j] = ratio__processing_time_before_first_preemption__prempting[j] / sum_ratio__processing_time_before_first_preemption__prempting #Not equal
        X[17,j] = ratio__processing_time_before_first_preemption__processing[j] / sum_ratio__processing_time_before_first_preemption__processing #Not equal

    end

    # Deciles 
    decile_rj = Vector{Int}(undef,inst.nb_jobs)
    decile_pj = Vector{Int}(undef,inst.nb_jobs)
    for j in 1:inst.nb_jobs
        decile_rj[j] = decile_of_x_in_v(inst.release_times[j],deciles_release)
        decile_pj[j] = decile_of_x_in_v(inst.processing_times[j], deciles_processing)
    end

    release_div_decile_release = [inst.release_times[j] / (decile_rj[j] + 1) for j in 1:inst.nb_jobs]
    sum_release_div_decile_release = sum(release_div_decile_release)
    processing_div_decile_processing = [inst.processing_times[j] / (decile_pj[j] + 1) for j in 1:inst.nb_jobs]
    sum_processing_div_decile_processing = sum(processing_div_decile_processing)

    for j in 1:inst.nb_jobs
        X[18, j] = decile_rj[j] # Not equal
        X[19, j] = release_div_decile_release[j] / sum_release_div_decile_release #Not equal
        X[20,j] = decile_pj[j] # Not equal
        X[21,j] = processing_div_decile_processing[j] / sum_processing_div_decile_processing   #Not equal
    end

    position_in_srpt = invperm(sorted)

    for j in 1:inst.nb_jobs
        # Nb preemptions
        X[22,j] = total_number_preemptions ==0 ? 0 : number_of_preemptions[j] / total_number_preemptions

        # SRPT position
        X[23,j] = position_in_srpt[j] / inst.nb_jobs # Equal module a 1/nb_jobs constant
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
        X[24,j] = sum_nb_smaller_p_before == 0 ? 0 : nb_smaller_p_before[j] / sum_nb_smaller_p_before
        X[25,j] = sum_nb_smaller_r_before == 0 ? 0 : nb_smaller_r_before[j] / sum_nb_smaller_r_before
        X[26,j] = sum_nb_larger_p_before == 0 ? 0 : nb_larger_p_before[j] / sum_nb_larger_p_before
        X[27,j] = sum_nb_larger_r_before == 0 ? 0 : nb_larger_r_before[j] / sum_nb_larger_r_before
    end

    return X
end

nb_features_encoder(encoder::typeof(encoder_1_rj_sumCj)) = 27

# Function to convert solutions between different formats

"""
    function sequence_to_embedding(seq::Vector{I}) where {I <: Int}
"""
function sequence_to_embedding(seq::Vector{I}) where {I <: Int}
    return Vector{Float64}(invperm(seq))
end

"""
    function embedding_to_sequence(y::AbstractVector)
"""
function embedding_to_sequence(y::AbstractVector)
    y_int = Vector{Int}(y)
    @assert(sum(abs.(y - y_int)) < 0.000001)
    return invperm(y_int)
end

# Algorithms available to find good/optimal solution:

"""
    solver_name(solver)

returns the name of `solver` used for logging purpose
"""
solver_name(solver) = "unknown"
solver_name(sol::typeof(rdi_aptrf)) = "rdiaptrf"

function glpk_1_rj_sumCj(inst::Instance1_rj_sumCj)
    return milp_solve_1_rj_sumCj(inst,MILP_solver=GLPK.Optimizer)
end
solver_name(sol::typeof(glpk_1_rj_sumCj)) = "glpk"

import JSON

function build_load_or_solve(;seed=0,nb_jobs=50,range=0.8, solver=milp_solver, load_and_save=true)
    inst = SingleMachineScheduling.build_instance_1_rj_sumCj(seed= seed, nb_jobs=nb_jobs,range=range)
    seq = Int[]
    val = 0.0

    if load_and_save 
        inst_sol_name = "sol_" * solver_name(solver) * "_seed" * string(seed) * "_jobs" * string(nb_jobs) * "_range" * string(range) *".json"
        mkpath("data")
        filename = joinpath("data",inst_sol_name)
        
        if isfile(filename)
            # load
            stringdata = ""
            f = open(filename, "r") 
            stringdata = read(f,String)
            close(f)
            seq = Vector{Int}(JSON.parse(stringdata))
            val = evaluate_solution_1_rj_sumCj(inst,seq)
        else
            val, seq = solver(inst)
            # Save
            stringdata = JSON.json(seq)
            open(filename,"w") do f
                write(f,stringdata)
            end
        end
    else
        val, seq = milp_solve_1_rj_sumCj(inst,MILP_solver=MILP_solver)
    end

    return inst, val, seq
end

"""
    function build_solve_and_encode_instance(;seed=0,nb_jobs=50,range=0.8, solver=milp_solver, load_and_save=true)

Function that 

    - builds an instance `inst`
    - compute its embedding `x`
    - solve the instance with `solver` to compute a solution
    - computes the embedding `y` of the solution computed by `solver`
    - computes the value `val` of the solution
    - returns `x,y,inst,val`

If `load_and_save` is set to `true`, the solution is loaded from the disk (if the solution file already exists), or computed and saved to the disk (if the solution file doesn't exit)
"""
function build_solve_and_encode_instance(;seed=0,nb_jobs=50,range=0.8, encoder=encoder_1_rj_sumCj, solver=milp_solver, load_and_save=true)
    @assert solver_name(solver) != solver_name(1)
    inst, val, seq = build_load_or_solve(seed=seed,nb_jobs=nb_jobs,range=range, solver=solver, load_and_save=load_and_save)
    x = encoder(inst)


    y = sequence_to_embedding(seq)
    return x,y,inst,val
end


"""
    build_and_solve_instance(;seed=0,nb_jobs=50,range=0.8, solver=milp_solver, load_and_save=true)

Function that 

    - builds an instance `inst`
    - solve the instance with `solver` to compute a solution and its value `val`
    - returns `inst,val`

If `load_and_save` is set to `true`, the solution is loaded from the disk (if the solution file already exists), or computed and saved to the disk (if the solution file doesn't exit)
"""
function build_and_solve_instance(;seed=0,nb_jobs=50,range=0.8, solver=milp_solver, load_and_save=true)
    @assert solver_name(solver) != solver_name(1)

    inst = SingleMachineScheduling.build_instance_1_rj_sumCj(seed= seed, nb_jobs=nb_jobs,range=range)
    x = encoder(inst)

    val, seq = load_or_solve(inst,load_and_save,solver)

    return inst,val
end