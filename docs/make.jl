using SingleMachineScheduling
using Documenter
using Literate

DocMeta.setdocmeta!(SingleMachineScheduling, :DocTestSetup, :(using SingleMachineScheduling); recursive=true)

# expe_jl_file = joinpath(dirname(@__DIR__), "test", "experiments.jl")
# expe_md_dir = joinpath(@__DIR__, "src")
# Literate.markdown(expe_jl_file, expe_md_dir; documenter=true, execute=true)

makedocs(;
    modules=[SingleMachineScheduling],
    authors="Axel Parmentier <axel.parmentier@enpc.fr> and contributors",
    repo="https://github.com/axelparmentier/SingleMachineScheduling.jl/blob/{commit}{path}#{line}",
    sitename="SingleMachineScheduling.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://axelparmentier.github.io/SingleMachineScheduling.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Exact MILP" => "milp.md",
        "Heuristics" => "heuristics.md",
        # "Experiments" => "experiments.md",
        "API" => "api.md"
    ],
)

deploydocs(;
    repo="github.com/axelparmentier/SingleMachineScheduling.jl",
    devbranch="main",
)
