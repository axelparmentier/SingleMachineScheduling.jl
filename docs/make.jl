using SingleMachineScheduling
using Documenter

DocMeta.setdocmeta!(SingleMachineScheduling, :DocTestSetup, :(using SingleMachineScheduling); recursive=true)

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
        "API" => "api.md"
    ],
)

deploydocs(;
    repo="github.com/axelparmentier/SingleMachineScheduling.jl",
    devbranch="main",
)
