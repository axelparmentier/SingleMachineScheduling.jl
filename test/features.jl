# # Learning algorithms for ``1|r_j|\sum C_j``


using SingleMachineScheduling
using Flux
using InferOpt
using UnicodePlots
using ProgressMeter


# ## Solution pipeline

#=
Select the encoder. Encoders available

 - `encoder_1_rj_sumCj`
=#

encoder = encoder_1_rj_sumCj
nb_features = nb_features_encoder(encoder)

#=
Select the model
=#

model = Chain(Dense(nb_features,1,bias=false),X->dropdims(X,dims=1))

#=
Select the decoder.

Decoders available

 - `(inst,y) -> y`: no decoder
 - `decoder = fast_local_descent_1_rj_sumCj`
 - `decoder = (inst,y) -> rdi(inst,fast_local_descent_1_rj_sumCj(inst,y))`
=#

decoder = (inst,y) -> y;
#= 
Solution pipeline
=#

pipeline(inst) = decoder(inst,embedding_to_sequence(ranking(model(encoder(inst)))));

# ## Training set
#=
Instances in the training set 
=#
seeds = 1:10;
nb_jobs = 50:10:100;
ranges = 0.2:0.2:1.4;


#=
Utils, do not modify, can be commented if you don't want to use gurobi
=#

using Gurobi
env = Gurobi.Env()
gurobi_solver = () -> Gurobi.Optimizer(env)

function gurobi_1_rj_sumCj(inst::Instance1_rj_sumCj)
    return milp_solve_1_rj_sumCj(inst,MILP_solver=gurobi_solver)
end
SingleMachineScheduling.solver_name(sol::typeof(gurobi_1_rj_sumCj)) = "gurobi";

#=
Solution algorithm used to build the solution of instances in the training set. Algorithms available:

- `glpk_1_rj_sumCj`: exact
- `gurobi_1_rj_sumCj`: exact
- `rdi_aptrf`: heuristic
=#

solver = gurobi_1_rj_sumCj;

#=
Builds the training set
=#

training_data = [build_solve_and_encode_instance(seed=s, nb_jobs=n ,range=r, solver=solver, load_and_save=true) for s in seeds for n in nb_jobs for r in ranges];

# ## Test set
#=
Select the instances in the test set
=#
seeds = 50:60;
nb_jobs = 50:10:100;
ranges = 0.2:0.2:1.4;

#=
Select the benchmark algorithm used on the test set. Same algorithms available as for training set
=#

solver = rdi_aptrf;

#=
Build the test set
=#

test_data = [build_solve_and_encode_instance(seed=s, nb_jobs=n ,range=r, solver=solver, load_and_save=true) for s in seeds for n in nb_jobs for r in ranges];

# ## Learning

#=
Computes features sd
=#

features_mean = zeros(nb_features)
dim_count = 0
for (x,_,inst,_) in training_data
    dim_count += inst.nb_jobs
    for j in 1:inst.nb_jobs
        for f in 1:nb_features
            features_mean[f] += x[f,j]
        end
    end
end
features_mean /= dim_count
features_sd = zeros(nb_features)
for (x,_,inst,_) in training_data
    dim_count += inst.nb_jobs
    for j in 1:inst.nb_jobs
        for f in 1:nb_features
            features_sd[f] += (x[f,j] - features_mean[f])^2
        end
    end
end
features_sd = sqrt.(features_sd)

#=
Standardization layer
=#

struct Standardize_layer
    sd_inv::Vector{Float64}
end

function (sl::Standardize_layer)(x::AbstractMatrix)
    (nf,nj) = size(x)
    @assert nf == length(sl.sd_inv)
    res = zeros(nf,nj)
    for j in 1:nj
        for f in 1:nf
            res[f,j] = x[f,j] * sl.sd_inv[f]
        end
    end
    return res
end

features_sd_inv = (x->1/x).(features_sd)
sd_layer = Standardize_layer(ones(nb_features))
# sd_layer = Standardize_layer(features_sd_inv);

#=
Loss
=#

regularized_predictor = Perturbed(ranking; ε = 1.0, M=20)
loss = FenchelYoungLoss(regularized_predictor);

#=
Learning
=#
std_training_data = [(sd_layer(x),y,inst,val) for (x,y,inst,val) in training_data]
std_test_data = [(sd_layer(x),y,inst,val) for (x,y,inst,val) in test_data]

opt = ADAM();
fyl_losses = Float64[]
obj_train_losses = Float64[]
obj_test_losses = Float64[]
partial_pipeline = Chain(model,ranking,embedding_to_sequence)
@showprogress for epoch in 1:500
    fyl_l = 0.
    obj_train_l = 0.
    for (x_std, y,inst,val) in std_training_data
        grads = gradient(Flux.params(model)) do
            fyl_l += loss(model(x_std), y)
        end
        obj_train_l += (evaluate_solution_1_rj_sumCj(inst,partial_pipeline(x_std)) - val) / val
        Flux.update!(opt, Flux.params(model), grads)
    end
    obj_train_l /= length(training_data)
    push!(fyl_losses, fyl_l)
    push!(obj_train_losses, obj_train_l)

    obj_test_l = 0.0
    for (x,y,inst,val) in std_test_data 
        obj_test_l += (evaluate_solution_1_rj_sumCj(inst,partial_pipeline(x)) - val) / val
    end
    obj_test_l /= length(std_test_data)
    push!(obj_test_losses,obj_test_l)
end;

#=
Fenchel Young loss
=#

lineplot(fyl_losses[10:length(fyl_losses)], xlabel="Epoch", ylabel="FY Loss")

#=
Loss on the training set
=#

lineplot(obj_train_losses[10:length(fyl_losses)], xlabel="Epoch", ylabel="Obj train Loss")

#=
Loss on the test set
=#
lineplot(obj_test_losses[10:length(fyl_losses)], xlabel="Epoch", ylabel="Obj test Loss")

# ## benchmark

pipeline_without_decoder(inst) = embedding_to_sequence(ranking(model(sd_layer(encoder(inst)))))

decoders = [
    ("no_decoder",(inst,y) -> y),
    ("local",fast_local_descent_1_rj_sumCj),
    ("rdi",(inst,y) -> rdi(inst,fast_local_descent_1_rj_sumCj(inst,y)))
]

pipelines = Any[(name,inst -> decoder(inst, pipeline_without_decoder(inst))) for (name,decoder) in decoders]
function rdia(inst)
    _,sol = rdi_aptrf(inst)
    return sol
end
push!(pipelines, ("rdia", rdia))

data_sets = [("train",training_data),("test",test_data)];

for (data_name,data_set) in data_sets
    for (name, pipeline) in pipelines
        gaps = Float64[]
        gap = 0.
        for (_,_,inst,val) in data_set 
            gap = (evaluate_solution_1_rj_sumCj(inst,pipeline(inst)) - val) / val;
            push!(gaps, gap);
        end
        println(histogram(gaps,nbins=10,name=name * " " * data_name))
        println(sum(gaps)/length(gaps))
    end
end

# ## Check with values from paper

weights = [9.506266089662077, -1.3710315054206788,  0.1334585280839313, -12.717671717074401, -31.393832945142343, -65.99076384998047, 2727.5046035932914, 61.883341118377146, 20.013854704894786, -306.89057967968387, 11.016281079036249, -33.77663126876743, 2246.5767196831075, 75.12578950854285, -16.140917318465277, -10.391296995373096, 23.56958788377952,  0.2345640964855094, 73.68335584637983, -1.6562121307640043, -244.85450540859512, -41.84024227378858, 89.32668553827389, 14.394554937735686, -206.2433702076072, 46.13339975880264, -56.350659387437126]
model_paper = Chain(Dense(weights',false),X->dropdims(X,dims=1))

pipeline_without_decoder(inst) = embedding_to_sequence(ranking(model_paper(encoder(inst))))

pipelines = Any[(name,inst -> decoder(inst, pipeline_without_decoder(inst))) for (name,decoder) in decoders]
push!(pipelines, ("rdia", rdia))

for (name, pipeline) in pipelines
    gaps = Float64[]
    gap = 0.
    for (_,_,inst,val) in test_data 
        gap = (evaluate_solution_1_rj_sumCj(inst,pipeline(inst)) - val) / val;
        push!(gaps, gap);
    end
    println(histogram(gaps,nbins=10,name=name))
    println(sum(gaps)/length(gaps))
end




