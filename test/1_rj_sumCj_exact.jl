using SingleMachineScheduling
using JuMP, GLPK, Gurobi
env = Gurobi.Env()
gurobi_solver = () -> Gurobi.Optimizer(env)

inst = SingleMachineScheduling.build_instance_1_rj_sumCj(nb_jobs=50,range=1.0)
val,sol = SingleMachineScheduling.milp_solve_1_rj_sumCj(inst,MILP_solver=gurobi_solver)
@assert length(sol) == inst.nb_jobs
@assert abs(val - SingleMachineScheduling.evaluate_solution_1_rj_sumCj(inst,sol)) < 0.00001