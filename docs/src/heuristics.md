# Combinatorial heuristics for ``1|r_j|\sum C_j``

## Dispatching rules

Dispatching rules are used within a greedy algorithm that, given the current

### Basic dispatching rules

Take into account only properties of the job.

### Priority Rule for Total Flowtime (PRTF)

Takes into account properties of the remaining jobs and current time.

Implemented in [`SingleMachineScheduling.prtf`](@ref)


### Altnernative Priority Rule for Total Flowtime (APRTF)

Takes into account properties of current time and all the remaining job.

#### APTRF

Implemented in [`SingleMachineScheduling.aprtf`](@ref)

> [Chu, 1992, Efficient heuristics to minimize total flowtime with release dates](https://www.sciencedirect.com/science/article/pii/016763779290092H)

## Local descent

Implemented in [`SingleMachineScheduling.fast_local_descent_1_rj_sumCj!`](@ref)


## Release Date Iteration heuristic

Implemented in [`SingleMachineScheduling.rdi!`](@ref)

> [Chand et al., 1996, An iterative heuristic for the single machine dynamic total completion time scheduling problem](https://linkinghub.elsevier.com/retrieve/pii/0305054895000712)

