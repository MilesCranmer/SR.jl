module MutateModule

using DynamicExpressions:
    AbstractExpression,
    copy_into!,
    get_tree,
    preserve_sharing,
    count_scalar_constants,
    simplify_tree!,
    combine_operators,
    allocate_container
using ..CoreModule:
    AbstractOptions,
    AbstractMutationWeights,
    Dataset,
    RecordType,
    sample_mutation,
    max_features
using ..ComplexityModule: compute_complexity
using ..LossFunctionsModule: score_func, score_func_batched
using ..CheckConstraintsModule: check_constraints
using ..AdaptiveParsimonyModule: RunningSearchStatistics
using ..PopMemberModule: PopMember
using ..MutationFunctionsModule:
    mutate_constant,
    mutate_operator,
    swap_operands,
    append_random_op,
    prepend_random_op,
    insert_random_op,
    delete_random_op!,
    crossover_trees,
    form_random_connection!,
    break_random_connection!,
    randomly_rotate_tree!,
    randomize_tree
using ..ConstantOptimizationModule: optimize_constants
using ..RecorderModule: @recorder

abstract type AbstractMutationResult{N<:AbstractExpression,P<:PopMember} end

"""
    MutationResult{N<:AbstractExpression,P<:PopMember}

Represents the result of a mutation operation in the genetic programming algorithm. This struct is used to return values from `mutate!` functions.

# Fields

- `tree::Union{N, Nothing}`: The mutated expression tree, if applicable. Either `tree` or `member` must be set, but not both.
- `member::Union{P, Nothing}`: The mutated population member, if applicable. Either `member` or `tree` must be set, but not both.
- `num_evals::Float64`: The number of evaluations performed during the mutation, which is automatically set to `0.0`. Only used for things like `optimize`.
- `return_immediately::Bool`: If `true`, the mutation process should return immediately, bypassing further checks, used for things like `simplify` or `optimize` where you already know the loss value of the result.

# Usage

This struct encapsulates the result of a mutation operation. Either a new expression tree or a new population member is returned, but not both.

Return the `member` if you want to return immediately, and have
computed the loss value as part of the mutation.
"""
struct MutationResult{N<:AbstractExpression,P<:PopMember} <: AbstractMutationResult{N,P}
    tree::Union{N,Nothing}
    member::Union{P,Nothing}
    num_evals::Float64
    return_immediately::Bool

    # Explicit constructor with keyword arguments
    function MutationResult{_N,_P}(;
        tree::Union{_N,Nothing}=nothing,
        member::Union{_P,Nothing}=nothing,
        num_evals::Float64=0.0,
        return_immediately::Bool=false,
    ) where {_N<:AbstractExpression,_P<:PopMember}
        @assert(
            (tree === nothing) ⊻ (member === nothing),
            "Mutation result must return either a tree or a pop member, not both"
        )
        return new{_N,_P}(tree, member, num_evals, return_immediately)
    end
end

"""
    condition_mutation_weights!(weights::AbstractMutationWeights, member::PopMember, options::AbstractOptions, curmaxsize::Int)

Adjusts the mutation weights based on the properties of the current member and options.

This function modifies the mutation weights to ensure that the mutations applied to the member are appropriate given its current state and the provided options. It can be overloaded to customize the behavior for different types of expressions or members.

Note that the weights were already copied, so you don't need to worry about mutation.

# Arguments
- `weights::AbstractMutationWeights`: The mutation weights to be adjusted.
- `member::PopMember`: The current population member being mutated.
- `options::AbstractOptions`: The options that guide the mutation process.
- `curmaxsize::Int`: The current maximum size constraint for the member's expression tree.
"""
function condition_mutation_weights!(
    weights::AbstractMutationWeights, member::P, options::AbstractOptions, curmaxsize::Int
) where {T,L,N<:AbstractExpression,P<:PopMember{T,L,N}}
    tree = get_tree(member.tree)
    if !preserve_sharing(typeof(member.tree))
        weights.form_connection = 0.0
        weights.break_connection = 0.0
    end
    if tree.degree == 0
        # If equation is too small, don't delete operators
        # or simplify
        weights.mutate_operator = 0.0
        weights.swap_operands = 0.0
        weights.delete_node = 0.0
        weights.simplify = 0.0
        if !tree.constant
            weights.optimize = 0.0
            weights.mutate_constant = 0.0
        end
        return nothing
    end

    if !any(node -> node.degree == 2, tree)
        # swap is implemented only for binary ops
        weights.swap_operands = 0.0
    end

    condition_mutate_constant!(typeof(member.tree), weights, member, options, curmaxsize)

    complexity = compute_complexity(member, options)

    if complexity >= curmaxsize
        # If equation is too big, don't add new operators
        weights.add_node = 0.0
        weights.insert_node = 0.0
    end

    if !options.should_simplify
        weights.simplify = 0.0
    end

    return nothing
end

"""
Use this to modify how `mutate_constant` changes for an expression type.
"""
function condition_mutate_constant!(
    ::Type{<:AbstractExpression},
    weights::AbstractMutationWeights,
    member::PopMember,
    options::AbstractOptions,
    curmaxsize::Int,
)
    n_constants = count_scalar_constants(member.tree)
    weights.mutate_constant *= min(8, n_constants) / 8.0

    return nothing
end

# Go through one simulated options.annealing mutation cycle
#  exp(-delta/T) defines probability of accepting a change
function next_generation(
    dataset::D,
    member::P,
    temperature,
    curmaxsize::Int,
    running_search_statistics::RunningSearchStatistics,
    options::AbstractOptions;
    tmp_recorder::RecordType,
)::Tuple{
    P,Bool,Float64
} where {T,L,D<:Dataset{T,L},N<:AbstractExpression{T},P<:PopMember{T,L,N}}
    parent_ref = member.ref
    num_evals = 0.0

    #TODO - reconsider this
    beforeScore, beforeLoss = if options.batching
        num_evals += (options.batch_size / dataset.n)
        score_func_batched(dataset, member, options)
    else
        member.score, member.loss
    end

    nfeatures = max_features(dataset, options)

    weights = copy(options.mutation_weights)

    condition_mutation_weights!(weights, member, options, curmaxsize)

    mutation_choice = sample_mutation(weights)

    successful_mutation = false
    attempts = 0
    max_attempts = 10
    node_storage = allocate_container(member.tree)

    #############################################
    # Mutations
    #############################################
    # local tree
    rtree = Ref{N}()
    while (!successful_mutation) && attempts < max_attempts
        rtree[] = copy_into!(node_storage, member.tree)

        mutation_result = _dispatch_mutations!(
            rtree[],
            member,
            mutation_choice,
            options.mutation_weights,
            options;
            recorder=tmp_recorder,
            temperature,
            dataset,
            score=beforeScore,
            loss=beforeLoss,
            parent_ref,
            curmaxsize,
            nfeatures,
        )
        mutation_result::AbstractMutationResult{N,P}
        num_evals += mutation_result.num_evals::Float64

        if mutation_result.return_immediately
            @assert(
                mutation_result.member isa P,
                "Mutation result must return a `PopMember` if `return_immediately` is true"
            )
            return mutation_result.member::P, true, num_evals
        else
            @assert(
                mutation_result.tree isa N,
                "Mutation result must return a tree if `return_immediately` is false"
            )
            rtree[] = mutation_result.tree::N
            successful_mutation = check_constraints(rtree[], options, curmaxsize)
            attempts += 1
        end
    end

    tree = rtree[]

    if !successful_mutation
        @recorder begin
            tmp_recorder["result"] = "reject"
            tmp_recorder["reason"] = "failed_constraint_check"
        end
        mutation_accepted = false
        return (
            PopMember(
                copy_into!(node_storage, member.tree),
                beforeScore,
                beforeLoss,
                options,
                compute_complexity(member, options);
                parent=parent_ref,
                deterministic=options.deterministic,
            ),
            mutation_accepted,
            num_evals,
        )
    end

    if options.batching
        afterScore, afterLoss = score_func_batched(dataset, tree, options)
        num_evals += (options.batch_size / dataset.n)
    else
        afterScore, afterLoss = score_func(dataset, tree, options)
        num_evals += 1
    end

    if isnan(afterScore)
        @recorder begin
            tmp_recorder["result"] = "reject"
            tmp_recorder["reason"] = "nan_loss"
        end
        mutation_accepted = false
        return (
            PopMember(
                copy_into!(node_storage, member.tree),
                beforeScore,
                beforeLoss,
                options,
                compute_complexity(member, options);
                parent=parent_ref,
                deterministic=options.deterministic,
            ),
            mutation_accepted,
            num_evals,
        )
    end

    probChange = 1.0
    if options.annealing
        delta = afterScore - beforeScore
        probChange *= exp(-delta / (temperature * options.alpha))
    end
    newSize = -1
    if options.use_frequency
        oldSize = compute_complexity(member, options)
        newSize = compute_complexity(tree, options)
        old_frequency = if (0 < oldSize <= options.maxsize)
            running_search_statistics.normalized_frequencies[oldSize]
        else
            1e-6
        end
        new_frequency = if (0 < newSize <= options.maxsize)
            running_search_statistics.normalized_frequencies[newSize]
        else
            1e-6
        end
        probChange *= old_frequency / new_frequency
    end

    if probChange < rand()
        @recorder begin
            tmp_recorder["result"] = "reject"
            tmp_recorder["reason"] = "annealing_or_frequency"
        end
        mutation_accepted = false
        return (
            PopMember(
                copy_into!(node_storage, member.tree),
                beforeScore,
                beforeLoss,
                options,
                compute_complexity(member, options);
                parent=parent_ref,
                deterministic=options.deterministic,
            ),
            mutation_accepted,
            num_evals,
        )
    else
        @recorder begin
            tmp_recorder["result"] = "accept"
            tmp_recorder["reason"] = "pass"
        end
        mutation_accepted = true
        return (
            PopMember(
                tree,
                afterScore,
                afterLoss,
                options,
                newSize;
                parent=parent_ref,
                deterministic=options.deterministic,
            ),
            mutation_accepted,
            num_evals,
        )
    end
end

@generated function _dispatch_mutations!(
    tree::AbstractExpression,
    member::PopMember,
    mutation_choice::Symbol,
    weights::W,
    options::AbstractOptions;
    kws...,
) where {W<:AbstractMutationWeights}
    mutation_choices = fieldnames(W)
    quote
        Base.Cartesian.@nif(
            $(length(mutation_choices)),
            i -> mutation_choice == $(mutation_choices)[i],
            i -> begin
                @assert mutation_choice == $(mutation_choices)[i]
                mutate!(
                    tree, member, Val($(mutation_choices)[i]), weights, options; kws...
                )
            end,
        )
    end
end

"""
    mutate!(
        tree::N,
        member::P,
        ::Val{S},
        mutation_weights::AbstractMutationWeights,
        options::AbstractOptions;
        kws...,
    ) where {N<:AbstractExpression,P<:PopMember,S}

Perform a mutation on the given `tree` and `member` using the specified mutation type `S`.
Various `kws` are provided to access other data needed for some mutations.

You may overload this function to handle new mutation types for new `AbstractMutationWeights` types.

# Keywords

- `temperature`: The temperature parameter for annealing-based mutations.
- `dataset::Dataset`: The dataset used for scoring.
- `score`: The score of the member before mutation.
- `loss`: The loss of the member before mutation.
- `curmaxsize`: The current maximum size constraint, which may be different from `options.maxsize`.
- `nfeatures`: The number of features in the dataset.
- `parent_ref`: Reference to the mutated member's parent (only used for logging purposes).
- `recorder::RecordType`: A recorder to log mutation details.

# Returns

A `MutationResult{N,P}` object containing the mutated tree or member (but not both),
the number of evaluations performed, if any, and whether to return immediately from
the mutation function, or to let the `next_generation` function handle accepting or
rejecting the mutation. For example, a `simplify` operation will not change the loss,
so it can always return immediately.
"""
function mutate!(
    ::N, ::P, ::Val{S}, ::AbstractMutationWeights, ::AbstractOptions; kws...
) where {N<:AbstractExpression,P<:PopMember,S}
    return error("Unknown mutation choice: $S")
end

function mutate!(
    tree::N,
    member::P,
    ::Val{:mutate_constant},
    ::AbstractMutationWeights,
    options::AbstractOptions;
    recorder::RecordType,
    temperature,
    kws...,
) where {N<:AbstractExpression,P<:PopMember}
    tree = mutate_constant(tree, temperature, options)
    @recorder recorder["type"] = "mutate_constant"
    return MutationResult{N,P}(; tree=tree)
end

function mutate!(
    tree::N,
    member::P,
    ::Val{:mutate_operator},
    ::AbstractMutationWeights,
    options::AbstractOptions;
    recorder::RecordType,
    kws...,
) where {N<:AbstractExpression,P<:PopMember}
    tree = mutate_operator(tree, options)
    @recorder recorder["type"] = "mutate_operator"
    return MutationResult{N,P}(; tree=tree)
end

function mutate!(
    tree::N,
    member::P,
    ::Val{:swap_operands},
    ::AbstractMutationWeights,
    options::AbstractOptions;
    recorder::RecordType,
    kws...,
) where {N<:AbstractExpression,P<:PopMember}
    tree = swap_operands(tree)
    @recorder recorder["type"] = "swap_operands"
    return MutationResult{N,P}(; tree=tree)
end

function mutate!(
    tree::N,
    member::P,
    ::Val{:add_node},
    ::AbstractMutationWeights,
    options::AbstractOptions;
    recorder::RecordType,
    nfeatures,
    kws...,
) where {N<:AbstractExpression,P<:PopMember}
    if rand() < 0.5
        tree = append_random_op(tree, options, nfeatures)
        @recorder recorder["type"] = "add_node:append"
    else
        tree = prepend_random_op(tree, options, nfeatures)
        @recorder recorder["type"] = "add_node:prepend"
    end
    return MutationResult{N,P}(; tree=tree)
end

function mutate!(
    tree::N,
    member::P,
    ::Val{:insert_node},
    ::AbstractMutationWeights,
    options::AbstractOptions;
    recorder::RecordType,
    nfeatures,
    kws...,
) where {N<:AbstractExpression,P<:PopMember}
    tree = insert_random_op(tree, options, nfeatures)
    @recorder recorder["type"] = "insert_node"
    return MutationResult{N,P}(; tree=tree)
end

function mutate!(
    tree::N,
    member::P,
    ::Val{:delete_node},
    ::AbstractMutationWeights,
    options::AbstractOptions;
    recorder::RecordType,
    nfeatures,
    kws...,
) where {N<:AbstractExpression,P<:PopMember}
    tree = delete_random_op!(tree, options, nfeatures)
    @recorder recorder["type"] = "delete_node"
    return MutationResult{N,P}(; tree=tree)
end

function mutate!(
    tree::N,
    member::P,
    ::Val{:form_connection},
    ::AbstractMutationWeights,
    options::AbstractOptions;
    recorder::RecordType,
    kws...,
) where {N<:AbstractExpression,P<:PopMember}
    tree = form_random_connection!(tree)
    @recorder recorder["type"] = "form_connection"
    return MutationResult{N,P}(; tree=tree)
end

function mutate!(
    tree::N,
    member::P,
    ::Val{:break_connection},
    ::AbstractMutationWeights,
    options::AbstractOptions;
    recorder::RecordType,
    kws...,
) where {N<:AbstractExpression,P<:PopMember}
    tree = break_random_connection!(tree)
    @recorder recorder["type"] = "break_connection"
    return MutationResult{N,P}(; tree=tree)
end

function mutate!(
    tree::N,
    member::P,
    ::Val{:rotate_tree},
    ::AbstractMutationWeights,
    options::AbstractOptions;
    recorder::RecordType,
    kws...,
) where {N<:AbstractExpression,P<:PopMember}
    tree = randomly_rotate_tree!(tree)
    @recorder recorder["type"] = "rotate_tree"
    return MutationResult{N,P}(; tree=tree)
end

# Handle mutations that require early return
function mutate!(
    tree::N,
    member::P,
    ::Val{:simplify},
    ::AbstractMutationWeights,
    options::AbstractOptions;
    recorder::RecordType,
    parent_ref,
    kws...,
) where {N<:AbstractExpression,P<:PopMember}
    @assert options.should_simplify
    simplify_tree!(tree, options.operators)
    tree = combine_operators(tree, options.operators)
    @recorder recorder["type"] = "simplify"
    return MutationResult{N,P}(;
        member=PopMember(
            tree,
            member.score,
            member.loss,
            options;
            parent=parent_ref,
            deterministic=options.deterministic,
        ),
        return_immediately=true,
    )
end

function mutate!(
    tree::N,
    ::P,
    ::Val{:randomize},
    ::AbstractMutationWeights,
    options::AbstractOptions;
    recorder::RecordType,
    curmaxsize,
    nfeatures,
    kws...,
) where {T,N<:AbstractExpression{T},P<:PopMember}
    tree = randomize_tree(tree, curmaxsize, options, nfeatures)
    @recorder recorder["type"] = "randomize"
    return MutationResult{N,P}(; tree=tree)
end

function mutate!(
    tree::N,
    member::P,
    ::Val{:optimize},
    ::AbstractMutationWeights,
    options::AbstractOptions;
    recorder::RecordType,
    dataset::Dataset,
    kws...,
) where {N<:AbstractExpression,P<:PopMember}
    cur_member, new_num_evals = optimize_constants(dataset, member, options)
    @recorder recorder["type"] = "optimize"
    return MutationResult{N,P}(;
        member=cur_member, num_evals=new_num_evals, return_immediately=true
    )
end

function mutate!(
    tree::N,
    member::P,
    ::Val{:do_nothing},
    ::AbstractMutationWeights,
    options::AbstractOptions;
    recorder::RecordType,
    parent_ref,
    kws...,
) where {N<:AbstractExpression,P<:PopMember}
    @recorder begin
        recorder["type"] = "identity"
        recorder["result"] = "accept"
        recorder["reason"] = "identity"
    end
    return MutationResult{N,P}(;
        member=PopMember(
            tree,
            member.score,
            member.loss,
            options,
            compute_complexity(tree, options);
            parent=parent_ref,
            deterministic=options.deterministic,
        ),
        return_immediately=true,
    )
end

"""Generate a generation via crossover of two members."""
function crossover_generation(
    member1::P, member2::P, dataset::D, curmaxsize::Int, options::AbstractOptions
)::Tuple{P,P,Bool,Float64} where {T,L,D<:Dataset{T,L},N,P<:PopMember{T,L,N}}
    tree1 = member1.tree
    tree2 = member2.tree
    crossover_accepted = false

    # We breed these until constraints are no longer violated:
    child_tree1, child_tree2 = crossover_trees(tree1, tree2)
    num_tries = 1
    max_tries = 10
    num_evals = 0.0
    afterSize1 = -1
    afterSize2 = -1
    while true
        afterSize1 = compute_complexity(child_tree1, options)
        afterSize2 = compute_complexity(child_tree2, options)
        # Both trees satisfy constraints
        if check_constraints(child_tree1, options, curmaxsize, afterSize1) &&
            check_constraints(child_tree2, options, curmaxsize, afterSize2)
            break
        end
        if num_tries > max_tries
            crossover_accepted = false
            return member1, member2, crossover_accepted, num_evals  # Fail.
        end
        child_tree1, child_tree2 = crossover_trees(tree1, tree2)
        num_tries += 1
    end
    if options.batching
        afterScore1, afterLoss1 = score_func_batched(
            dataset, child_tree1, options; complexity=afterSize1
        )
        afterScore2, afterLoss2 = score_func_batched(
            dataset, child_tree2, options; complexity=afterSize2
        )
        num_evals += 2 * (options.batch_size / dataset.n)
    else
        afterScore1, afterLoss1 = score_func(
            dataset, child_tree1, options; complexity=afterSize1
        )
        afterScore2, afterLoss2 = score_func(
            dataset, child_tree2, options; complexity=afterSize2
        )
        num_evals += options.batch_size / dataset.n
    end

    baby1 = PopMember(
        child_tree1::AbstractExpression,
        afterScore1,
        afterLoss1,
        options,
        afterSize1;
        parent=member1.ref,
        deterministic=options.deterministic,
    )::P
    baby2 = PopMember(
        child_tree2::AbstractExpression,
        afterScore2,
        afterLoss2,
        options,
        afterSize2;
        parent=member2.ref,
        deterministic=options.deterministic,
    )::P

    crossover_accepted = true
    return baby1, baby2, crossover_accepted, num_evals
end

end  # module MutateModule
