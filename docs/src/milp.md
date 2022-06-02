# Exact MILP formulation for ``1|r_j|\sum_j C_j``


## MILP formulation

We introduce the following formulation for ``1|r_j|\sum_j C_j``

```math 
    \tag{MILP} 
    \begin{array}{rll}
        \displaystyle\min_{x,C} \, & \displaystyle \sum_{j=1}^n C_{[j]} \\
        \mathrm{s.t.} \, 
        & \displaystyle \sum_{j =1}^n x_{ij} = 1 & \text{for all } i \in \{1,\ldots,n\} \\
        & \displaystyle \sum_{i =1}^n x_{ij} = 1 & \text{for all } j \in \{1,\ldots,n\} \\
        & C_{[1]} \geq \displaystyle \sum_i (p_i + r_i) x_{i1} \\
        & C_{[j]} \geq C_{[{]}j-1} + \displaystyle \sum_i p_i x_{ij} & \text{for all } j \in \{2,\ldots,n\} \\
        & C_{[j]} \geq \displaystyle \sum_i (p_i + r_i) x_{ij} & \text{for all } j \in \{2,\ldots,n\} \\
        & x \in \{0,1\}
    \end{array}
```

where ``x_{ij}`` indicates if job ``i`` is in position ``j``, and ``C_{[j]}`` is the completion time of the job in position ``j``.

This MILP is implemented in function [`SingleMachineScheduling.Instance1_rj_sumCj
`](@ref)

## Valid cuts from SRPT

### Preemptive version ``1|r_j, preemp|\sum_j C_j`` 

In the preemptive version of ``1|r_j|\sum_j C_j``, a job can be interrupted at any moment, to operate another job, and finished later, restarting from where it has been interrupted.
A solution of the preemptive problem can therefore be encoded as a sequence of jobs ``j_1,\ldots,j_m`` which may contain repetitions (and do contain repetitions on many optimal solutions) and a sequence ``q_1,\ldots,q_m`` of processing times, where ``q_k`` gives the time spent on the machine by job ``j_k`` at that moment. We therefore have

```math
    \sum_{k\colon j_k = i} q_k = p_i \quad \text{for each job }i
```

Let ``S`` be such a solution.
Let ``D_k`` be that time at which operation in position ``k`` is finished. We have

```math
    \begin{array}{rl}
        D_1^{S} &= r_{j_1} + q_1 \\
        D_k^{S} &= \max(r_{j_k}, D_{k-1}) + q_{k} \quad \text{for }k>1
    \end{array}
```

The completion time of job ``i`` is then

```math
    C_{i}^{S} = D_{k_i} \quad \text{where} \quad k_i = \max
```
 
Given the ordering ``\tilde i_1,\ldots,\tilde i_n `` f the jobs by increasing ``C_{i}^{S}``, we have

```math
    C_{[j]}^{S} = C_{i_j}^{S}
```
### SRPT

The preemptive version can be solved with the *shortest remaining processing time* (SRPT) dispatching rule.
The solution ``(j_1,\ldots,j_k), (q_1,\ldots,q_k)`` is built iteratively
At time ``k``, the job ``i`` with the shortest *remaining processing time*

```math
    p_i - \sum_{h \in \{1,\ldots,k-1\} \colon j_h=i} q_h
```

is scheduled is position ``k``.
Preemption happens if a job with shorter remaining processing time is released before the completion of ``i``.

Let ``S`` be the solution of ``1|r_j, preemp|\sum_j C_j``  returned by SRPT ``q_1,\ldots,q_m``. 
Let ``C_[j]`` be the completion time of an optimal solution of the non-preemptive version ``1|r_j|\sum_j C_j``.
It can be shown that

```math
    C_{[j]} \geq C_{[j]}^S
```

Remark that ``[j]`` does not refer to the same job in ``C_{[j]}`` and ``C_{[j]}^S``.
This equation can be introduced as a valid cut in the MILP formulation [(MILP)](#milp-formulation).
Note that in that case ``C_{[j]}`` is a variable, while and ``C_{[j]}^S`` is a precomputed constant. This cut is implemented by default in This MILP is implemented in function [`SingleMachineScheduling.Instance1_rj_sumCj
`](@ref). It can be removed using the keyword argument `srpt_cuts=false`.

Improved SRPT cuts can be proposed, but we did not implement them 

> [Improving the preemptive bound for the one-machine dynamic total completion time scheduling problem](https://www.sciencedirect.com/science/article/pii/S0167637702002158?ref=pdf_download&fr=RR-2&rr=714ea2e0799e32b3)

> [A hybrid heuristic approach for single machine scheduling with release times](https://www.sciencedirect.com/science/article/pii/S0305054813003390)