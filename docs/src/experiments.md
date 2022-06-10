```@meta
EditURL = "<unknown>/test/experiments.jl"
```

# Learning algorithms for ``1|r_j|\sum C_j``

````julia
using SingleMachineScheduling
using Flux
using InferOpt
using UnicodePlots
using ProgressMeter
````

## Solution pipeline

Select the encoder. Encoders available

 - `encoder_1_rj_sumCj`

````julia
encoder = encoder_1_rj_sumCj
nb_features = nb_features_encoder(encoder);
````

Select the model

````julia
model = Chain(Dense(nb_features,1,bias=false),X->dropdims(X,dims=1));
````

Select the decoder.

Decoders available

 - `(inst,y) -> y`: no decoder
 - `decoder = fast_local_descent_1_rj_sumCj`
 - `decoder = (inst,y) -> rdi(inst,fast_local_descent_1_rj_sumCj(inst,y))`

````julia
decoder = (inst,y) -> y;

#=
Solution pipeline
=#

pipeline(inst) = decoder(inst,embedding_to_sequence(ranking(model(encoder(inst)))));
````

## Training set
Instances in the training set

````julia
seeds = 1:10;
nb_jobs = 50:10:100;
ranges = 0.2:0.2:1.4;
````

Utils, do not modify, can be commented if you don't want to use gurobi

````julia
using Gurobi
env = Gurobi.Env()
gurobi_solver = () -> Gurobi.Optimizer(env)

function gurobi_1_rj_sumCj(inst::Instance1_rj_sumCj)
    return milp_solve_1_rj_sumCj(inst,MILP_solver=gurobi_solver)
end
SingleMachineScheduling.solver_name(sol::typeof(gurobi_1_rj_sumCj)) = "gurobi";
````

````

--------------------------------------------
Warning: your license will expire in 13 days
--------------------------------------------

Academic license - for non-commercial use only

````

Solution algorithm used to build the solution of instances in the training set. Algorithms available:

- `glpk_1_rj_sumCj`: exact
- `gurobi_1_rj_sumCj`: exact
- `rdi_aptrf`: heuristic

````julia
solver = gurobi_1_rj_sumCj;
````

Builds the training set

````julia
training_data = [build_solve_and_encode_instance(seed=s, nb_jobs=n ,range=r, solver=solver, load_and_save=true) for s in seeds for n in nb_jobs for r in ranges];
````

## Test set
Select the instances in the test set

````julia
seeds = 50:60;
nb_jobs = 50:10:100;
ranges = 0.2:0.2:1.4;
````

Select the benchmark algorithm used on the test set. Same algorithms available as for training set

````julia
solver = rdi_aptrf;
````

Build the test set

````julia
test_data = [build_solve_and_encode_instance(seed=s, nb_jobs=n ,range=r, solver=solver, load_and_save=true) for s in seeds for n in nb_jobs for r in ranges];
````

## Learning

Computes features sd

````julia
count_dim = 0
features_mean = zeros(nb_features)
for (x,_,inst,_) in training_data
    global count_dim += inst.nb_jobs
    for j in 1:inst.nb_jobs
        for f in 1:nb_features
            features_mean[f] += x[f,j]
        end
    end
end
features_mean /= count_dim
features_sd = zeros(nb_features)
for (x,_,inst,_) in training_data
    for j in 1:inst.nb_jobs
        for f in 1:nb_features
            features_sd[f] += (x[f,j] - features_mean[f])^2
        end
    end
end
features_sd = sqrt.(features_sd);
````

Definition of standardization layer

````julia
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
````

Activate the standardization layer

````julia
features_sd_inv = (x->1/x).(features_sd)
sd_layer = Standardize_layer(features_sd_inv);
````

We desactivate the standardization layer for these specific experiments (comment the following block to activate it)

````julia
sd_layer = Standardize_layer(ones(nb_features));
````

Loss

````julia
regularized_predictor = Perturbed(ranking; ε = 1.0, M=20)
loss = FenchelYoungLoss(regularized_predictor);
````

Learning

````julia
std_training_data = [(sd_layer(x),y,inst,val) for (x,y,inst,val) in training_data]
std_test_data = [(sd_layer(x),y,inst,val) for (x,y,inst,val) in test_data]

opt = ADAM();
fyl_losses = Float64[]
obj_train_losses = Float64[]
obj_test_losses = Float64[]
partial_pipeline = Chain(model,ranking,embedding_to_sequence)
for epoch in 1:300
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
````

### Learning results.

Curves giving the Fenchel Young loss (convex loss used for learning) on the training set and the objective value on the training and test set

Fenchel Young loss

````julia
lineplot(fyl_losses[10:length(fyl_losses)], xlabel="Epoch", ylabel="FY Loss")
````

````
                  ┌────────────────────────────────────────┐ 
           300000 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                  │⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                  │⢇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                  │⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
   FY Loss        │⠈⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                  │⠀⠹⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                  │⠀⠀⢱⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                  │⠀⠀⠀⠙⠦⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                  │⠀⠀⠀⠀⠀⠈⠉⠙⠛⠓⠓⠓⠻⠒⠒⠶⠖⠖⠖⠖⠶⠖⠓⠶⠾⠶⠞⠖⠳⠶⠶⠟⠖⠲⠒⠶⠾⠲⠶⠀│ 
                  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                0 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                  └────────────────────────────────────────┘ 
                  ⠀0⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀300⠀ 
                  ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀Epoch⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ 
````

Loss on the training set

````julia
lineplot(obj_train_losses[10:length(fyl_losses)], xlabel="Epoch", ylabel="Obj train Loss")
````

````
                      ┌────────────────────────────────────────┐ 
                  0.2 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                      │⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                      │⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                      │⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                      │⢱⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                      │⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
   Obj train Loss     │⠘⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                      │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                      │⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                      │⠀⠀⢇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                      │⠀⠀⠈⢆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                      │⠀⠀⠀⠈⠳⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                      │⠀⠀⠀⠀⠀⠈⠓⠢⠤⣤⣀⣀⣀⣀⣀⣀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                    0 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠀│ 
                      └────────────────────────────────────────┘ 
                      ⠀0⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀300⠀ 
                      ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀Epoch⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ 
````

Loss on the test set

````julia
lineplot(obj_test_losses[10:length(fyl_losses)], xlabel="Epoch", ylabel="Obj test Loss")
````

````
                     ┌────────────────────────────────────────┐ 
                 0.2 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
   Obj test Loss     │⠘⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⠀⢸⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⠀⠀⢇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⠀⠀⠈⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⠀⠀⠀⠈⠳⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⠀⠀⠀⠀⠀⠈⠙⠒⠦⢤⣄⣀⣀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                   0 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠀│ 
                     └────────────────────────────────────────┘ 
                     ⠀0⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀300⠀ 
                     ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀Epoch⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ 
````

## Benchmark

### Learned model performance

````julia
function test_pipeline_on_training_and_test_set(name, pipeline)
    data_sets = [("train",training_data),("test",test_data)];

    for (data_name,data_set) in data_sets
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

function test_model_on_training_and_test_set(model_name, model)

    pipeline_without_decoder(inst) = embedding_to_sequence(ranking(model(sd_layer(encoder(inst)))))

    decoders = [
        (model_name * "no_decoder",(inst,y) -> y),
        (model_name * "local",fast_local_descent_1_rj_sumCj),
        (model_name * "rdi",(inst,y) -> rdi(inst,fast_local_descent_1_rj_sumCj(inst,y)))
    ]
    pipelines = [(name,inst -> decoder(inst, pipeline_without_decoder(inst))) for (name,decoder) in decoders]

    for (name, pipeline) in pipelines
        test_pipeline_on_training_and_test_set(name,pipeline)
    end
end

test_model_on_training_and_test_set("learned model " , model)
````

````
                ┌                                        ┐                               
   [0.0 , 0.01) ┤███████████████████████████████████  165  learned model no_decoder train
   [0.01, 0.02) ┤███████████████████████████████▎ 147                                    
   [0.02, 0.03) ┤███████████████▏ 71                                                     
   [0.03, 0.04) ┤█████▌ 26                                                               
   [0.04, 0.05) ┤█▊ 9                                                                    
   [0.05, 0.06) ┤  0                                                                     
   [0.06, 0.07) ┤  0                                                                     
   [0.07, 0.08) ┤▌ 2                                                                     
                └                                        ┘                               
                                 Frequency                                               
0.014557739228648524
                  ┌                                        ┐                              
   [-0.01,  0.0 ) ┤██▍ 14                                    learned model no_decoder test
   [ 0.0 ,  0.01) ┤███████████████████████████████████  214                               
   [ 0.01,  0.02) ┤████████████████████████▌ 150                                          
   [ 0.02,  0.03) ┤██████████▋ 65                                                         
   [ 0.03,  0.04) ┤█▊ 11                                                                  
   [ 0.04,  0.05) ┤▋ 4                                                                    
   [ 0.05,  0.06) ┤▌ 3                                                                    
   [ 0.06,  0.07) ┤  0                                                                    
   [ 0.07,  0.08) ┤  0                                                                    
   [ 0.08,  0.09) ┤  0                                                                    
   [ 0.09,  0.1 ) ┤▎ 1                                                                    
                  └                                        ┘                              
                                   Frequency                                              
0.01219254155420144
                ┌                                        ┐                          
   [0.0 , 0.01) ┤███████████████████████████████████  212  learned model local train
   [0.01, 0.02) ┤█████████████████████████▎ 152                                     
   [0.02, 0.03) ┤███████▎ 43                                                        
   [0.03, 0.04) ┤█▋ 10                                                              
   [0.04, 0.05) ┤▎ 1                                                                
   [0.05, 0.06) ┤  0                                                                
   [0.06, 0.07) ┤▍ 2                                                                
                └                                        ┘                          
                                 Frequency                                          
0.011084772514537483
                  ┌                                        ┐                         
   [-0.02, -0.01) ┤▍ 2                                       learned model local test
   [-0.01,  0.0 ) ┤███▏ 23                                                           
   [ 0.0 ,  0.01) ┤███████████████████████████████████  263                          
   [ 0.01,  0.02) ┤██████████████████▋ 140                                           
   [ 0.02,  0.03) ┤███▉ 30                                                           
   [ 0.03,  0.04) ┤▎ 1                                                               
   [ 0.04,  0.05) ┤▍ 3                                                               
                  └                                        ┘                         
                                   Frequency                                         
0.00868317232381912
                  ┌                                        ┐                        
   [0.0  , 0.005) ┤███████████████████████████████████  372  learned model rdi train
   [0.005, 0.01 ) ┤██▊ 31                                                           
   [0.01 , 0.015) ┤█▍ 15                                                            
   [0.015, 0.02 ) ┤  0                                                              
   [0.02 , 0.025) ┤▎ 2                                                              
                  └                                        ┘                        
                                   Frequency                                        
0.002515767000202535
                    ┌                                        ┐                       
   [-0.025, -0.02 ) ┤▎ 1                                       learned model rdi test
   [-0.02 , -0.015) ┤▎ 2                                                             
   [-0.015, -0.01 ) ┤▍ 3                                                             
   [-0.01 , -0.005) ┤█▍ 13                                                           
   [-0.005,  0.0  ) ┤███████▎ 74                                                     
   [ 0.0  ,  0.005) ┤███████████████████████████████████  360                        
   [ 0.005,  0.01 ) ┤▊ 8                                                             
   [ 0.01 ,  0.015) ┤  0                                                             
   [ 0.015,  0.02 ) ┤▎ 1                                                             
                    └                                        ┘                       
                                     Frequency                                       
0.00018478607912945626

````

### Benchmark againt RDI APTRF

````julia
function rdia(inst)
    _,sol = rdi_aptrf(inst)
    return sol
end

test_pipeline_on_training_and_test_set("RDI APTRF", rdia)
````

````
                  ┌                                        ┐                
   [0.0  , 0.005) ┤███████████████████████████████████  380  RDI APTRF train
   [0.005, 0.01 ) ┤█▌ 17                                                    
   [0.01 , 0.015) ┤█▏ 11                                                    
   [0.015, 0.02 ) ┤▊ 9                                                      
   [0.02 , 0.025) ┤▎ 1                                                      
   [0.025, 0.03 ) ┤▎ 2                                                      
                  └                                        ┘                
                                   Frequency                                
0.0023261444095230587
              ┌                                        ┐               
   [0.0, 1.0) ┤███████████████████████████████████  462  RDI APTRF test
              └                                        ┘               
                               Frequency                               
0.0

````

### Comparison to a random model

This enables to check that the post-processing used are not enough alone to get the performance we have.

````julia
model_random = Chain(Dense(nb_features,1,bias=false),X->dropdims(X,dims=1))
test_model_on_training_and_test_set("random model ", model_random)
````

````
              ┌                                        ┐                              
   [0.8, 1.0) ┤█▊ 6                                      random model no_decoder train
   [1.0, 1.2) ┤██████████  34                                                         
   [1.2, 1.4) ┤████████████████████▊ 71                                               
   [1.4, 1.6) ┤████████████████████████████████▉ 112                                  
   [1.6, 1.8) ┤███████████████████████████████████  119                               
   [1.8, 2.0) ┤██████████████████▌ 63                                                 
   [2.0, 2.2) ┤███▌ 12                                                                
   [2.2, 2.4) ┤▊ 3                                                                    
              └                                        ┘                              
                               Frequency                                              
1.5625934749446024
              ┌                                        ┐                             
   [0.8, 1.0) ┤███▎ 14                                   random model no_decoder test
   [1.0, 1.2) ┤██████████▌ 46                                                        
   [1.2, 1.4) ┤████████████████▍ 72                                                  
   [1.4, 1.6) ┤███████████████████████████████████  154                              
   [1.6, 1.8) ┤████████████████████████████████▌ 143                                 
   [1.8, 2.0) ┤██████▊ 30                                                            
   [2.0, 2.2) ┤▋ 3                                                                   
              └                                        ┘                             
                               Frequency                                             
1.5046112027535476
              ┌                                        ┐                         
   [0.0, 0.2) ┤▍ 1                                       random model local train
   [0.2, 0.4) ┤█████▋ 18                                                         
   [0.4, 0.6) ┤██████████████▊ 47                                                
   [0.6, 0.8) ┤█████████████▎ 42                                                 
   [0.8, 1.0) ┤████████████████▋ 53                                              
   [1.0, 1.2) ┤███████████████████████████████████  111                          
   [1.2, 1.4) ┤████████████████████████████████▊ 104                             
   [1.4, 1.6) ┤████████████▉ 41                                                  
   [1.6, 1.8) ┤▉ 3                                                               
              └                                        ┘                         
                               Frequency                                         
1.0163335483214315
              ┌                                        ┐                        
   [0.2, 0.4) ┤██████▎ 30                                random model local test
   [0.4, 0.6) ┤█████████▏ 44                                                    
   [0.6, 0.8) ┤██████████▍ 50                                                   
   [0.8, 1.0) ┤████████████▉ 63                                                 
   [1.0, 1.2) ┤███████████████████████████████████  170                         
   [1.2, 1.4) ┤███████████████████▌ 95                                          
   [1.4, 1.6) ┤██▏ 10                                                           
              └                                        ┘                        
                               Frequency                                        
0.9698502112215877
                ┌                                        ┐                       
   [0.0 , 0.02) ┤███████████████████████████████████  288  random model rdi train
   [0.02, 0.04) ┤████▋ 38                                                        
   [0.04, 0.06) ┤██▋ 22                                                          
   [0.06, 0.08) ┤██▊ 23                                                          
   [0.08, 0.1 ) ┤██▌ 20                                                          
   [0.1 , 0.12) ┤█▊ 15                                                           
   [0.12, 0.14) ┤▎ 2                                                             
   [0.14, 0.16) ┤▋ 5                                                             
   [0.16, 0.18) ┤▋ 6                                                             
   [0.18, 0.2 ) ┤▎ 1                                                             
                └                                        ┘                       
                                 Frequency                                       
0.02609663976506764
                ┌                                        ┐                      
   [-0.1,  0.0) ┤█████▌ 55                                 random model rdi test
   [ 0.0,  0.1) ┤███████████████████████████████████  349                       
   [ 0.1,  0.2) ┤███▍ 33                                                        
   [ 0.2,  0.3) ┤▎ 2                                                            
   [ 0.3,  0.4) ┤▊ 8                                                            
   [ 0.4,  0.5) ┤█▎ 11                                                          
   [ 0.5,  0.6) ┤▍ 4                                                            
                └                                        ┘                      
                                 Frequency                                      
0.04593212499919188

````

### Check with values from paper

Test the performance of the different pipelines on the test set and on the training set with statistical model weights coming from previous work. Enables to benchmark the weights learned above.

````julia
weights = [9.506266089662077, -1.3710315054206788,  0.1334585280839313, -12.717671717074401, -31.393832945142343, -65.99076384998047, 2727.5046035932914, 61.883341118377146, 20.013854704894786, -306.89057967968387, 11.016281079036249, -33.77663126876743, 2246.5767196831075, 75.12578950854285, -16.140917318465277, -10.391296995373096, 23.56958788377952,  0.2345640964855094, 73.68335584637983, -1.6562121307640043, -244.85450540859512, -41.84024227378858, 89.32668553827389, 14.394554937735686, -206.2433702076072, 46.13339975880264, -56.350659387437126]
model_paper = Chain(Dense(weights',false),X->dropdims(X,dims=1))

test_model_on_training_and_test_set("paper model ", model_paper)
````

````
                ┌                                        ┐                             
   [0.0 , 0.02) ┤███▊ 15                                   paper model no_decoder train
   [0.02, 0.04) ┤█████████▏ 35                                                         
   [0.04, 0.06) ┤███████████████▊ 61                                                   
   [0.06, 0.08) ┤███████████████████████████████████  135                              
   [0.08, 0.1 ) ┤█████████████████████████▋ 99                                         
   [0.1 , 0.12) ┤███████████▉ 46                                                       
   [0.12, 0.14) ┤████▍ 17                                                              
   [0.14, 0.16) ┤█▊ 7                                                                  
   [0.16, 0.18) ┤▊ 3                                                                   
   [0.18, 0.2 ) ┤▌ 2                                                                   
                └                                        ┘                             
                                 Frequency                                             
0.07576496837960733
                  ┌                                        ┐                            
   [-0.02,  0.0 ) ┤▊ 3                                       paper model no_decoder test
   [ 0.0 ,  0.02) ┤█████▍ 19                                                            
   [ 0.02,  0.04) ┤████████████▊ 46                                                     
   [ 0.04,  0.06) ┤████████████████████████▋ 88                                         
   [ 0.06,  0.08) ┤███████████████████████████████████  125                             
   [ 0.08,  0.1 ) ┤█████████████████████████████████▊ 121                               
   [ 0.1 ,  0.12) ┤██████████▉ 39                                                       
   [ 0.12,  0.14) ┤█████▍ 19                                                            
   [ 0.14,  0.16) ┤  0                                                                  
   [ 0.16,  0.18) ┤▌ 2                                                                  
                  └                                        ┘                            
                                   Frequency                                            
0.07044254023365457
                ┌                                        ┐                        
   [0.0 , 0.01) ┤███████████████████████████████████  369  paper model local train
   [0.01, 0.02) ┤██▊ 29                                                           
   [0.02, 0.03) ┤▌ 5                                                              
   [0.03, 0.04) ┤▊ 8                                                              
   [0.04, 0.05) ┤▌ 5                                                              
   [0.05, 0.06) ┤▎ 2                                                              
   [0.06, 0.07) ┤▎ 2                                                              
                └                                        ┘                        
                                 Frequency                                        
0.004613449375132551
                  ┌                                        ┐                       
   [-0.03, -0.02) ┤▎ 1                                       paper model local test
   [-0.02, -0.01) ┤█▋ 11                                                           
   [-0.01,  0.0 ) ┤█████████████████████████▌ 168                                  
   [ 0.0 ,  0.01) ┤███████████████████████████████████  231                        
   [ 0.01,  0.02) ┤████▎ 28                                                        
   [ 0.02,  0.03) ┤█▍ 9                                                            
   [ 0.03,  0.04) ┤▌ 3                                                             
   [ 0.04,  0.05) ┤▊ 6                                                             
   [ 0.05,  0.06) ┤▌ 3                                                             
   [ 0.06,  0.07) ┤▎ 1                                                             
   [ 0.07,  0.08) ┤▎ 1                                                             
                  └                                        ┘                       
                                   Frequency                                       
0.0027501374607042054
                  ┌                                        ┐                      
   [0.0  , 0.001) ┤███████████████████████████████████  235  paper model rdi train
   [0.001, 0.002) ┤████████████████▉ 114                                          
   [0.002, 0.003) ┤███████▍ 49                                                    
   [0.003, 0.004) ┤██▎ 14                                                         
   [0.004, 0.005) ┤▎ 1                                                            
   [0.005, 0.006) ┤▌ 3                                                            
   [0.006, 0.007) ┤▍ 2                                                            
   [0.007, 0.008) ┤  0                                                            
   [0.008, 0.009) ┤  0                                                            
   [0.009, 0.01 ) ┤▎ 1                                                            
   [0.01 , 0.011) ┤▎ 1                                                            
                  └                                        ┘                      
                                   Frequency                                      
0.0011762948716585167
                    ┌                                        ┐                     
   [-0.025, -0.02 ) ┤▎ 1                                       paper model rdi test
   [-0.02 , -0.015) ┤▋ 4                                                           
   [-0.015, -0.01 ) ┤█▍ 8                                                          
   [-0.01 , -0.005) ┤███▊ 23                                                       
   [-0.005,  0.0  ) ┤██████████████████████████████████▋ 210                       
   [ 0.0  ,  0.005) ┤███████████████████████████████████  212                      
   [ 0.005,  0.01 ) ┤▌ 3                                                           
   [ 0.01 ,  0.015) ┤  0                                                           
   [ 0.015,  0.02 ) ┤▎ 1                                                           
                    └                                        ┘                     
                                     Frequency                                     
-0.001012941597738036

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

