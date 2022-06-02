using SingleMachineScheduling

# Test fast local fast_local_descent_1_rj_sumCj
nb_jobs=100
inst = SingleMachineScheduling.build_instance_1_rj_sumCj(seed= 7, nb_jobs=nb_jobs,range=0.8)
sol = [i for i in 1:nb_jobs]
val_before_local_descent = SingleMachineScheduling.evaluate_solution_1_rj_sumCj(inst,sol)

SingleMachineScheduling.fast_local_descent_1_rj_sumCj!(inst,sol)
val_after_local_descent = SingleMachineScheduling.evaluate_solution_1_rj_sumCj(inst,sol)

@test val_after_local_descent <= val_before_local_descent
@test sort(deepcopy(sol)) == [i for i in 1:nb_jobs]

# Test APRTF

val_aprtf, sol_aprtf = SingleMachineScheduling.aprtf(inst)
@test val_before_local_descent > val_aprtf

# Test RDI
dispatching_rule = SingleMachineScheduling.jobs_positions_in_sol(sol_aprtf)
sol_rdi = [i for i in 1:nb_jobs]
SingleMachineScheduling.rdi!(inst,dispatching_rule,sol_rdi)
val_rdi = SingleMachineScheduling.evaluate_solution_1_rj_sumCj(inst,sol_rdi)

@test val_rdi <= val_aprtf