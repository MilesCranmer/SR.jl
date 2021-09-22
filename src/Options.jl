using FromFile
using Distributed
using LossFunctions
using Zygote
#TODO - eventually move some of these
# into the SR call itself, rather than
# passing huge options at once.
@from "Operators.jl" import plus, pow, mult, sub, div, log_abs, log10_abs, log2_abs, log1p_abs, sqrt_abs, acosh_abs, atanh_clip
@from "Equation.jl" import Node

"""
         build_constraints(una_constraints, bin_constraints,
                           unary_operators, binary_operators)

Build constraints on operator-level complexity from a user-passed dict.
"""
function build_constraints(una_constraints, bin_constraints,
                           unary_operators, binary_operators,
                           nuna, nbin)::Tuple{Array{Int, 1}, Array{Tuple{Int,Int}, 1}}
    # Expect format ((*)=>(-1, 3)), etc.
    # TODO: Need to disable simplification if (*, -, +, /) are constrained?
    #  Or, just quit simplification is constraints violated.

    is_bin_constraints_already_done = typeof(bin_constraints) <: Array{Tuple{Int,Int},1}
    is_una_constraints_already_done = typeof(una_constraints) <: Array{Int,1}

    if typeof(bin_constraints) <: Array && !is_bin_constraints_already_done
        bin_constraints = Dict(bin_constraints)
    end
    if typeof(una_constraints) <: Array && !is_una_constraints_already_done
        una_constraints = Dict(una_constraints)
    end

    if una_constraints == nothing
        una_constraints = [-1 for i=1:nuna]
    elseif !is_una_constraints_already_done
        una_constraints::Dict
        _una_constraints = Int[]
        for (i, op) in enumerate(unary_operators)
            did_user_declare_constraints = haskey(una_constraints, op)
            if did_user_declare_constraints
                constraint::Int = una_constraints[op]
                push!(_una_constraints, constraint)
            else
                push!(_una_constraints, -1)
            end
        end
        una_constraints = _una_constraints
    end
    if bin_constraints == nothing
        bin_constraints = [(-1, -1) for i=1:nbin]
    elseif !is_bin_constraints_already_done
        bin_constraints::Dict
        _bin_constraints = Tuple{Int,Int}[]
        for (i, op) in enumerate(binary_operators)
            did_user_declare_constraints = haskey(bin_constraints, op)
            if did_user_declare_constraints
                constraint::Tuple{Int,Int} = bin_constraints[op]
                push!(_bin_constraints, constraint)
            else
                push!(_bin_constraints, (-1, -1))
            end
        end
        bin_constraints = _bin_constraints
    end

    return una_constraints, bin_constraints
end

function binopmap(op)
    if op == plus
        return +
    elseif op == mult
        return *
    elseif op == sub
        return -
    elseif op == div
        return /
    elseif op == ^
        return pow
    end
    return op
end

function unaopmap(op)
    if op == log
        return log_abs
    elseif op == log10
        return log10_abs
    elseif op == log2
        return log2_abs
    elseif op == log1p
        return log1p_abs
    elseif op == sqrt
        return sqrt_abs
    elseif op == acosh
        return acosh_abs
    elseif op == atanh
        return atanh_clip
    end
    return op
end

struct Options{A,B,dA,dB,C<:Union{SupervisedLoss,Function}}

    binops::A
    unaops::B
    diff_binops::dA
    diff_unaops::dB
    bin_constraints::Array{Tuple{Int,Int}, 1}
    una_constraints::Array{Int, 1}
    ns::Int
    parsimony::Float32
    alpha::Float32
    maxsize::Int
    maxdepth::Int
    fast_cycle::Bool
    migration::Bool
    hofMigration::Bool
    fractionReplacedHof::Float32
    shouldOptimizeConstants::Bool
    hofFile::String
    npopulations::Int
    perturbationFactor::Float32
    annealing::Bool
    batching::Bool
    batchSize::Int
    mutationWeights::Array{Float64, 1}
    warmupMaxsizeBy::Float32
    useFrequency::Bool
    npop::Int
    ncyclesperiteration::Int
    fractionReplaced::Float32
    topn::Int
    verbosity::Int
    probNegate::Float32
    nuna::Int
    nbin::Int
    seed::Union{Int, Nothing}
    loss::C
    progress::Bool
    terminal_width::Union{Int, Nothing}
    optimizer_algorithm::String
    optimize_probability::Float32
    optimizer_nrestarts::Int
    optimizer_iterations::Int
    recorder::Bool
    recorder_file::String
    probPickFirst::Float32

end

"""
    Options(;kws...)

Construct options for `EquationSearch` and other functions.

# Arguments
- `binary_operators=(div, plus, mult)`: Tuple of binary
    operators to use. Each operator should be defined for two input scalars,
    and one output scalar. All operators need to be defined over the entire
    real line (excluding infinity - these are stopped before they are input).
    Thus, `log` should be replaced with `log_abs`, etc.
    For speed, define it so it takes two reals
    of the same type as input, and outputs the same type. For the SymbolicUtils
    simplification backend, you will need to define a generic method of the
    operator so it takes arbitrary types.
- `unary_operators=(exp, cos)`: Same, but for
    unary operators (one input scalar, gives an output scalar).
- `constraints=nothing`: Array of pairs specifying size constraints
    for each operator. The constraints for a binary operator should be a 2-tuple
    (e.g., `(-1, -1)`) and the constraints for a unary operator should be an `Int`.
    A size constraint is a limit to the size of the subtree
    in each argument of an operator. e.g., `[(^)=>(-1, 3)]` means that the
    `^` operator can have arbitrary size (`-1`) in its left argument,
    but a maximum size of `3` in its right argument. Default is
    no constraints.
- `batching=false`: Whether to evolve based on small mini-batches of data,
    rather than the entire dataset.
- `batchSize=50`: What batch size to use if using batching.
- `loss=L2DistLoss()`: What loss function to use. Can be one of
    the following losses, or any other loss of type
    `SupervisedLoss`. You can also pass a function that takes
    a scalar target (left argument), and scalar predicted (right
    argument), and returns a scalar. This will be averaged
    over the predicted data. If weights are supplied, your
    function should take a third argument for the weight scalar.
    Included losses:
        Regression:
            - `LPDistLoss{P}()`,
            - `L1DistLoss()`,
            - `L2DistLoss()` (mean square),
            - `LogitDistLoss()`,
            - `HuberLoss(d)`,
            - `L1EpsilonInsLoss(ϵ)`,
            - `L2EpsilonInsLoss(ϵ)`,
            - `PeriodicLoss(c)`,
            - `QuantileLoss(τ)`,
        Classification:
            - `ZeroOneLoss()`,
            - `PerceptronLoss()`,
            - `L1HingeLoss()`,
            - `SmoothedL1HingeLoss(γ)`,
            - `ModifiedHuberLoss()`,
            - `L2MarginLoss()`,
            - `ExpLoss()`,
            - `SigmoidLoss()`,
            - `DWDMarginLoss(q)`.
- `npopulations=nothing`: How many populations of equations to use. By default
    this is set equal to the number of cores
- `npop=1000`: How many equations in each population.
- `ncyclesperiteration=300`: How many generations to consider per iteration.
- `ns=10`: Number of equations in each subsample during regularized evolution.
- `topn=10`: Number of equations to return to the host process, and to
    consider for the hall of fame.
- `alpha=0.100000f0`: The probability of accepting an equation mutation
    during regularized evolution is given by exp(-delta_loss/(alpha * T)),
    where T goes from 1 to 0. Thus, alpha=infinite is the same as no annealing.
- `maxsize=20`: Maximum size of equations during the search.
- `maxdepth=nothing`: Maximum depth of equations during the search, by default
    this is set equal to the maxsize.
- `parsimony=0.000100f0`: A multiplicative factor for how much complexity is
    punished.
- `useFrequency=false`: Whether to use a parsimony that adapts to the
    relative proportion of equations at each complexity; this will
    ensure that there are a balanced number of equations considered
    for every complexity.
- `fast_cycle=false`: Whether to thread over subsamples of equations during
    regularized evolution. Slightly improves performance, but is a different
    algorithm.
- `migration=true`: Whether to migrate equations between processes.
- `hofMigration=true`: Whether to migrate equations from the hall of fame
    to processes.
- `fractionReplaced=0.1f0`: What fraction of each population to replace with
    migrated equations at the end of each cycle.
- `fractionReplacedHof=0.1f0`: What fraction to replace with hall of fame
    equations at the end of each cycle.
- `shouldOptimizeConstants=true`: Whether to use NelderMead optimization
    to periodically optimize constants in equations.
- `optimizer_nrestarts=3`: How many different random starting positions to consider
    when using NelderMead optimization.
- `hofFile=nothing`: What file to store equations to, as a backup.
- `perturbationFactor=1.000000f0`: When mutating a constant, either
    multiply or divide by (1+perturbationFactor)^(rand()+1).
- `probNegate=0.01f0`: Probability of negating a constant in the equation
    when mutating it.
- `mutationWeights=[10.000000, 1.000000, 1.000000, 3.000000, 3.000000, 0.010000, 1.000000, 1.000000]`:
- `annealing=true`: Whether to use simulated annealing.
- `warmupMaxsize=0`: Whether to slowly increase the max size from 5 up to
    `maxsize`. If nonzero, specifies how many cycles (populations*iterations)
    before increasing by 1.
- `verbosity=convert(Int, 1e9)`: Whether to print debugging statements or
    not.
- `bin_constraints=nothing`:
- `una_constraints=nothing`:
- `seed=nothing`: What random seed to use. `nothing` uses no seed.
- `progress=false`: Whether to use a progress bar output (`verbosity` will
    have no effect).


- `probPickFirst=1.0`: Expressions in subsample are chosen based on, for
    p=probPickFirst: p, p*(1-p), p*(1-p)^2, and so on.
"""
function Options(;
    binary_operators::NTuple{nbin, Any}=(div, plus, mult),
    unary_operators::NTuple{nuna, Any}=(exp, cos),
    constraints=nothing,
    loss=L2DistLoss(),
    ns=10, #1 sampled from every ns per mutation
    topn=10, #samples to return per population
    parsimony=0.000100f0,
    alpha=0.100000f0,
    maxsize=20,
    maxdepth=nothing,
    fast_cycle=false,
    migration=true,
    hofMigration=true,
    fractionReplacedHof=0.1f0,
    shouldOptimizeConstants=true,
    hofFile=nothing,
    npopulations=nothing,
    perturbationFactor=1.000000f0,
    annealing=true,
    batching=false,
    batchSize=50,
    mutationWeights=[10.000000, 1.000000, 1.000000, 3.000000, 3.000000, 0.010000, 1.000000, 1.000000],
    warmupMaxsizeBy=0f0,
    useFrequency=false,
    npop=1000,
    ncyclesperiteration=300,
    fractionReplaced=0.1f0,
    verbosity=convert(Int, 1e9),
    probNegate=0.01f0,
    seed=nothing,
    bin_constraints=nothing,
    una_constraints=nothing,
    progress=false,
    terminal_width=nothing,
    warmupMaxsize=nothing,
    optimizer_algorithm="NelderMead",
    optimizer_nrestarts=3,
    optimize_probability=0.1f0,
    optimizer_iterations=100,
    nrestarts=nothing,
    recorder=nothing,
    recorder_file="pysr_recorder.json",
    probPickFirst=1.0,
   ) where {nuna,nbin}

    if nrestarts != nothing
        optimizer_nrestarts = nrestarts
    end
    if warmupMaxsize != nothing
        error("warmupMaxsize is deprecated. Please use warmupMaxsizeBy, and give the time at which the warmup will end as a fraction of the total search cycles.")
    end

    if hofFile == nothing
        hofFile = "hall_of_fame.csv" #TODO - put in date/time string here
    end

    @assert maxsize > 3
    @assert warmupMaxsizeBy >= 0f0

    constraints::Union{Tuple,Array{Pair{Any,Any}, 1},Nothing}


    if typeof(constraints) <: Tuple
        constraints = collect(constraints)
    end
    if constraints !== nothing
        @assert bin_constraints == nothing
        @assert una_constraints == nothing
        # TODO: This is redundant with the checks in EquationSearch
        for op in binary_operators
            @assert !(op in unary_operators)
        end
        for op in unary_operators
            @assert !(op in binary_operators)
        end
        bin_constraints = constraints
        una_constraints = constraints
    end

    una_constraints, bin_constraints = build_constraints(una_constraints, bin_constraints,
                                                         unary_operators, binary_operators,
                                                         nuna, nbin)

    if maxdepth == nothing
        maxdepth = maxsize
    end

    if npopulations == nothing
        npopulations = nworkers()
    end

    binary_operators = map(binopmap, binary_operators)
    unary_operators = map(unaopmap, unary_operators)

    diff_binary_operators = Any[]
    diff_unary_operators = Any[]

    for op in binary_operators
        diff_op(x,y) = gradient(op,x,y)
        push!(diff_binary_operators, diff_op)
    end

    for op in unary_operators
       diff_op(x) = gradient(op,x)[1]
       push!(diff_unary_operators, diff_op)
    end

    diff_binary_operators = Tuple(diff_binary_operators)
    diff_unary_operators = Tuple(diff_unary_operators)

    mutationWeights = map((x,)->convert(Float64, x), mutationWeights)
    if length(mutationWeights) != 8
        error("Not the right number of mutation probabilities given")
    end

    for (op, f) in enumerate(map(Symbol, binary_operators))
        _f = if f == Symbol(pow)
            Symbol(^)
        else
            f
        end
        if !isdefined(Base, _f)
            continue
        end
        @eval begin
            Base.$_f(l::Node, r::Node)::Node = (l.constant && r.constant) ? Node($f(l.val, r.val)::AbstractFloat) : Node($op, l, r)
            Base.$_f(l::Node, r::AbstractFloat)::Node =        l.constant ? Node($f(l.val, r)::AbstractFloat)     : Node($op, l, r)
            Base.$_f(l::AbstractFloat, r::Node)::Node =        r.constant ? Node($f(l, r.val)::AbstractFloat)     : Node($op, l, r)
        end
    end

    for (op, f) in enumerate(map(Symbol, unary_operators))
        if !isdefined(Base, f)
            continue
        end
        @eval begin
            Base.$f(l::Node)::Node = l.constant ? Node($f(l.val)::AbstractFloat) : Node($op, l)
        end
    end

    if progress
        verbosity = 0
    end

    if recorder == nothing
        recorder = haskey(ENV, "PYSR_RECORDER") && (ENV["PYSR_RECORDER"] == "1")
    end

    options = Options{typeof(binary_operators),typeof(unary_operators), typeof(diff_binary_operators), typeof(diff_unary_operators), typeof(loss)}(binary_operators, unary_operators, diff_binary_operators, diff_unary_operators, bin_constraints, una_constraints, ns, parsimony, alpha, maxsize, maxdepth, fast_cycle, migration, hofMigration, fractionReplacedHof, shouldOptimizeConstants, hofFile, npopulations, perturbationFactor, annealing, batching, batchSize, mutationWeights, warmupMaxsizeBy, useFrequency, npop, ncyclesperiteration, fractionReplaced, topn, verbosity, probNegate, nuna, nbin, seed, loss, progress, terminal_width, optimizer_algorithm, optimize_probability, optimizer_nrestarts, optimizer_iterations, recorder, recorder_file, probPickFirst)
    return options
end


