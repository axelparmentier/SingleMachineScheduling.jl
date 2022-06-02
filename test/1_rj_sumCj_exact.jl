using SingleMachineScheduling
using JuMP, GLPK, Gurobi

env = Gurobi.Env()
gurobi_solver = () -> Gurobi.Optimizer(env)

# Build a small instance
inst = SingleMachineScheduling.build_instance_1_rj_sumCj(seed= 7, nb_jobs=20,range=1.0)

# Test that MILP objective is equal to objective recomputed (without srpt cuts)
val,sol = SingleMachineScheduling.milp_solve_1_rj_sumCj(inst,MILP_solver=gurobi_solver,srpt_cuts=false)
@test length(sol) == inst.nb_jobs
@test abs(val - SingleMachineScheduling.evaluate_solution_1_rj_sumCj(inst,sol)) < 0.00001

# Test that SRPT gives a smaller solutionn , and that SRPT cuts are satisfied
val_recomputed_using_srpt , _ = SingleMachineScheduling.evaluate_preemptive_solution_1_rj_sumCj(inst,sol,[inst.processing_times[j] for j in sol])
@test val == val_recomputed_using_srpt

srpt_sol = SingleMachineScheduling.srpt_1_rj_sumCj(inst)

@test srpt_sol.obj_value <= val

# Test that MILP with srpt cuts gives the same solution
val_cut,sol_cut = SingleMachineScheduling.milp_solve_1_rj_sumCj(inst,MILP_solver=gurobi_solver,srpt_cuts=true)
@test val_cut == val

