```@meta
EditURL = "<unknown>/test/features.jl"
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
nb_features = nb_features_encoder(encoder)
````

````
27
````

Select the model

````julia
model = Chain(Dense(nb_features,1,bias=false),X->dropdims(X,dims=1))
````

````
Chain(
  Dense(27, 1; bias=false),             # 27 parameters
  Main.##301.var"#1#2"(),
)
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
Warning: your license will expire in 14 days
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

count_dim = 0
features_mean = zeros(nb_features)
for (x,_,inst,_) in training_data
    count_dim += inst.nb_jobs
    for j in 1:inst.nb_jobs
        for f in 1:nb_features
            features_mean[f] += x[f,j]
        end
    end
end
features_mean /= count_dim

````julia
features_sd = zeros(nb_features)
````

````
27-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
````

for (x,_,inst,_) in training_data
    count_dim += inst.nb_jobs
    for j in 1:inst.nb_jobs
        for f in 1:nb_features
            features_sd[f] += (x[f,j] - features_mean[f])^2
        end
    end
end
features_sd = sqrt.(features_sd)

Standardization layer

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

features_sd_inv = (x->1/x).(features_sd)
sd_layer = Standardize_layer(ones(nb_features))
````

````
Main.##301.Standardize_layer([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
````

sd_layer = Standardize_layer(features_sd_inv);

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
@showprogress for epoch in 1:2000
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

````


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
                  │⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                  │⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                  │⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
   FY Loss        │⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                  │⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                  │⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                  │⢻⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                  │⠈⠻⠶⠿⠿⠾⠶⠷⠾⠾⠶⠾⠾⠶⠷⠶⢶⠿⠷⠿⠿⠶⠷⠶⠾⠾⠾⠷⠶⠷⠶⠿⠷⠾⠾⠷⠷⠶⠷⠶│ 
                  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                0 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                  └────────────────────────────────────────┘ 
                  ⠀0⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀2000⠀ 
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
                      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                      │⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                      │⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                      │⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                      │⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
   Obj train Loss     │⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                      │⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                      │⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                      │⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                      │⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                      │⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                      │⠘⣆⣀⡀⠀⠀⠀⠀⠀⠀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀│ 
                    0 │⠀⠀⠈⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                      └────────────────────────────────────────┘ 
                      ⠀0⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀2000⠀ 
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
                     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
   Obj test Loss     │⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     │⠘⣆⡀⠀⠀⠀⠀⠀⠀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤│ 
                   0 │⠀⠀⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
                     └────────────────────────────────────────┘ 
                     ⠀0⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀2000⠀ 
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
   [0.0 , 0.01) ┤███████████████████████████████████  152  learned model no_decoder train
   [0.01, 0.02) ┤█████████████████████████████▌ 128                                      
   [0.02, 0.03) ┤██████████████████▌ 80                                                  
   [0.03, 0.04) ┤█████████▌ 41                                                           
   [0.04, 0.05) ┤███▊ 17                                                                 
   [0.05, 0.06) ┤▎ 1                                                                     
   [0.06, 0.07) ┤  0                                                                     
   [0.07, 0.08) ┤▎ 1                                                                     
                └                                        ┘                               
                                 Frequency                                               
0.016394662046428057
                  ┌                                        ┐                              
   [-0.02,  0.0 ) ┤▊ 8                                       learned model no_decoder test
   [ 0.0 ,  0.02) ┤███████████████████████████████████  313                               
   [ 0.02,  0.04) ┤████████████▋ 113                                                      
   [ 0.04,  0.06) ┤██▍ 21                                                                 
   [ 0.06,  0.08) ┤▌ 4                                                                    
   [ 0.08,  0.1 ) ┤▎ 2                                                                    
   [ 0.1 ,  0.12) ┤▎ 1                                                                    
                  └                                        ┘                              
                                   Frequency                                              
0.016450248012374394
                ┌                                        ┐                          
   [0.0 , 0.01) ┤███████████████████████████████████  186  learned model local train
   [0.01, 0.02) ┤███████████████████████████▍ 145                                   
   [0.02, 0.03) ┤████████████▋ 67                                                   
   [0.03, 0.04) ┤███▌ 19                                                            
   [0.04, 0.05) ┤▍ 2                                                                
   [0.05, 0.06) ┤  0                                                                
   [0.06, 0.07) ┤▎ 1                                                                
                └                                        ┘                          
                                 Frequency                                          
0.01253850377930412
                  ┌                                        ┐                         
   [-0.01,  0.0 ) ┤███▍ 20                                   learned model local test
   [ 0.0 ,  0.01) ┤███████████████████████████████████  210                          
   [ 0.01,  0.02) ┤███████████████████████▋ 142                                      
   [ 0.02,  0.03) ┤██████████▋ 64                                                    
   [ 0.03,  0.04) ┤██▌ 15                                                            
   [ 0.04,  0.05) ┤█▎ 7                                                              
   [ 0.05,  0.06) ┤▍ 2                                                               
   [ 0.06,  0.07) ┤▎ 1                                                               
   [ 0.07,  0.08) ┤  0                                                               
   [ 0.08,  0.09) ┤▎ 1                                                               
                  └                                        ┘                         
                                   Frequency                                         
0.012038417788875455
                  ┌                                        ┐                        
   [0.0  , 0.005) ┤███████████████████████████████████  364  learned model rdi train
   [0.005, 0.01 ) ┤███▍ 35                                                          
   [0.01 , 0.015) ┤█▌ 16                                                            
   [0.015, 0.02 ) ┤▎ 2                                                              
   [0.02 , 0.025) ┤▎ 2                                                              
   [0.025, 0.03 ) ┤▎ 1                                                              
                  └                                        ┘                        
                                   Frequency                                        
0.0026638216380414155
                    ┌                                        ┐                       
   [-0.025, -0.02 ) ┤▎ 1                                       learned model rdi test
   [-0.02 , -0.015) ┤▎ 1                                                             
   [-0.015, -0.01 ) ┤▍ 3                                                             
   [-0.01 , -0.005) ┤▊ 9                                                             
   [-0.005,  0.0  ) ┤████████▎ 82                                                    
   [ 0.0  ,  0.005) ┤███████████████████████████████████  351                        
   [ 0.005,  0.01 ) ┤█▎ 12                                                           
   [ 0.01 ,  0.015) ┤▎ 2                                                             
   [ 0.015,  0.02 ) ┤▎ 1                                                             
                    └                                        ┘                       
                                     Frequency                                       
0.0003633883397266644

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

````julia
#=
This enables to check that the post-processing used are not enough alone to get the performance we have.
=#

model_random = Chain(Dense(nb_features,1,bias=false),X->dropdims(X,dims=1))
test_model_on_training_and_test_set("random model ", model)
````

````
                ┌                                        ┐                              
   [0.0 , 0.01) ┤███████████████████████████████████  152  random model no_decoder train
   [0.01, 0.02) ┤█████████████████████████████▌ 128                                     
   [0.02, 0.03) ┤██████████████████▌ 80                                                 
   [0.03, 0.04) ┤█████████▌ 41                                                          
   [0.04, 0.05) ┤███▊ 17                                                                
   [0.05, 0.06) ┤▎ 1                                                                    
   [0.06, 0.07) ┤  0                                                                    
   [0.07, 0.08) ┤▎ 1                                                                    
                └                                        ┘                              
                                 Frequency                                              
0.016394662046428057
                  ┌                                        ┐                             
   [-0.02,  0.0 ) ┤▊ 8                                       random model no_decoder test
   [ 0.0 ,  0.02) ┤███████████████████████████████████  313                              
   [ 0.02,  0.04) ┤████████████▋ 113                                                     
   [ 0.04,  0.06) ┤██▍ 21                                                                
   [ 0.06,  0.08) ┤▌ 4                                                                   
   [ 0.08,  0.1 ) ┤▎ 2                                                                   
   [ 0.1 ,  0.12) ┤▎ 1                                                                   
                  └                                        ┘                             
                                   Frequency                                             
0.016450248012374394
                ┌                                        ┐                         
   [0.0 , 0.01) ┤███████████████████████████████████  186  random model local train
   [0.01, 0.02) ┤███████████████████████████▍ 145                                  
   [0.02, 0.03) ┤████████████▋ 67                                                  
   [0.03, 0.04) ┤███▌ 19                                                           
   [0.04, 0.05) ┤▍ 2                                                               
   [0.05, 0.06) ┤  0                                                               
   [0.06, 0.07) ┤▎ 1                                                               
                └                                        ┘                         
                                 Frequency                                         
0.01253850377930412
                  ┌                                        ┐                        
   [-0.01,  0.0 ) ┤███▍ 20                                   random model local test
   [ 0.0 ,  0.01) ┤███████████████████████████████████  210                         
   [ 0.01,  0.02) ┤███████████████████████▋ 142                                     
   [ 0.02,  0.03) ┤██████████▋ 64                                                   
   [ 0.03,  0.04) ┤██▌ 15                                                           
   [ 0.04,  0.05) ┤█▎ 7                                                             
   [ 0.05,  0.06) ┤▍ 2                                                              
   [ 0.06,  0.07) ┤▎ 1                                                              
   [ 0.07,  0.08) ┤  0                                                              
   [ 0.08,  0.09) ┤▎ 1                                                              
                  └                                        ┘                        
                                   Frequency                                        
0.012038417788875455
                  ┌                                        ┐                       
   [0.0  , 0.005) ┤███████████████████████████████████  364  random model rdi train
   [0.005, 0.01 ) ┤███▍ 35                                                         
   [0.01 , 0.015) ┤█▌ 16                                                           
   [0.015, 0.02 ) ┤▎ 2                                                             
   [0.02 , 0.025) ┤▎ 2                                                             
   [0.025, 0.03 ) ┤▎ 1                                                             
                  └                                        ┘                       
                                   Frequency                                       
0.0026638216380414155
                    ┌                                        ┐                      
   [-0.025, -0.02 ) ┤▎ 1                                       random model rdi test
   [-0.02 , -0.015) ┤▎ 1                                                            
   [-0.015, -0.01 ) ┤▍ 3                                                            
   [-0.01 , -0.005) ┤▊ 9                                                            
   [-0.005,  0.0  ) ┤████████▎ 82                                                   
   [ 0.0  ,  0.005) ┤███████████████████████████████████  351                       
   [ 0.005,  0.01 ) ┤█▎ 12                                                          
   [ 0.01 ,  0.015) ┤▎ 2                                                            
   [ 0.015,  0.02 ) ┤▎ 1                                                            
                    └                                        ┘                      
                                     Frequency                                      
0.0003633883397266644

````

### Check with values from paper

````julia
#=
Test the performance of the different pipelines on the test set and on the training set with statistical model weights coming from previous work. Enables to benchmark the weights learned above.
=#

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
