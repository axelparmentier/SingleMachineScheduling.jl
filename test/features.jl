using SingleMachineScheduling
using JuMP, GLPK, Gurobi
using Flux
using InferOpt
using UnicodePlots

env = Gurobi.Env()
gurobi_solver = () -> Gurobi.Optimizer(env)

# Training set
# seeds = 1:1
# nb_jobs = 5:1:5
# ranges = 1.0:0.1:1.0
seeds = 1:5
nb_jobs = 20:10:40
ranges = 0.6:0.2:1.0

function sequence_to_embedding(seq::Vector{I}) where {I <: Int}
    return Vector{Float64}(invperm(seq))
end

function embedding_to_sequence(y::AbstractVector)
    y_int = Vector{Int}(y)
    @assert(sum(abs.(y - y_int)) < 0.000001)
    return invperm(y_int)
end

function build_instance(;seed=0,nb_jobs=50,range=0.8, MILP_solver=GLPK.Optimizer)
    inst = SingleMachineScheduling.build_instance_1_rj_sumCj(seed= seed, nb_jobs=nb_jobs,range=range)
    x = encoder_1_rj_sumCj(inst)
    val , sequence = milp_solve_1_rj_sumCj(inst,MILP_solver=MILP_solver)
    y = sequence_to_embedding(sequence)
    # y = zeros(inst.nb_jobs)
    # for (pos,j) in enumerate(sol)
    #     y[j] = inst.nb_jobs - pos
    # end
    # y = reverse(sol)
    return x,y,inst,val
end


training_data = [build_instance(seed=s, nb_jobs=n ,range=r, MILP_solver=gurobi_solver) for s in seeds for n in nb_jobs for r in ranges]

# Save training set to json
import JSON
filename = "data/training_set.json"
stringdata = JSON.json(training_data)
open(filename,"w") do f
    write(f,stringdata)
end

# Load training set from json
function parse_json_instance(my_vec::AbstractVector)
    # inst
    inst_dict = my_vec[3]
    inst = SingleMachineScheduling.Instance1_rj_sumCj(
        Int(inst_dict["nb_jobs"]),
        Vector{Int}(inst_dict["release_times"]),
        Vector{Int}(inst_dict["processing_times"])
        )
    # x
    nb_jobs = inst.nb_jobs
    @assert nb_jobs == length(my_vec[2])
    nb_features = length(my_vec[1][1])
    x = Array{Float64}(undef, nb_features,nb_jobs)
    for j in 1:nb_jobs
        for f in 1:nb_features
            x[f,j] = my_vec[1][j][f]
        end
    end
    
    # y
    y = Vector{Float64}(my_vec[2])
    
    return x,y,inst,my_vec[4]
end
    
f = open(filename, "r")
dicttxt = read(f,String)  # file information to string
close(f)
training_data_read=JSON.parse(dicttxt)  # parse and transform data
training_data = [parse_json_instance(dict_read) for dict_read in training_data_read]

# Model
nb_features = size(training_data[1][1])[1]
model = Chain(Dense(nb_features,1),X->dropdims(X,dims=1))
pipeline = Chain(model,ranking,embedding_to_sequence)

# Loss 
regularized_predictor = Perturbed(ranking; ε = 0.0, M=1)
loss = FenchelYoungLoss(regularized_predictor)

# Training
# training_data = [training_data[1]]
opt = ADAM();
fyl_losses = Float64[]
obj_losses = Float64[]
for epoch in 1:200
    fyl_l = 0.
    obj_l = 0.
    for (x, y,inst,val) in training_data
        grads = gradient(Flux.params(model)) do
            fyl_l += loss(model(x), y)
        end
        obj_l += (evaluate_solution_1_rj_sumCj(inst,pipeline(x)) - val) / val
        Flux.update!(opt, Flux.params(model), grads)
    end
    obj_l /= length(training_data)
    push!(fyl_losses, fyl_l)
    push!(obj_losses, obj_l)
end;

lineplot(fyl_losses[10:200], xlabel="Epoch", ylabel="FY Loss")
lineplot(obj_losses[10:200], xlabel="Epoch", ylabel="Obj Loss")

