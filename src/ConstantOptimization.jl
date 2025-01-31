module ConstantOptimizationModule

using LineSearches: LineSearches
using Optim: Optim
using ADTypes: AbstractADType, AutoEnzyme, AutoZygote
using DifferentiationInterface: value_and_gradient
using DynamicExpressions:
    AbstractExpression,
    Expression,
    count_scalar_constants,
    get_scalar_constants,
    set_scalar_constants!,
    extract_gradient
using ..CoreModule: AbstractOptions, Dataset, DATA_TYPE, LOSS_TYPE, specialized_options
using ..UtilsModule: get_birth_order
using ..LossFunctionsModule: eval_loss, loss_to_score, batch_sample
using ..PopMemberModule: PopMember

function optimize_constants(
    dataset::Dataset{T,L}, member::P, options::AbstractOptions
)::Tuple{P,Float64} where {T<:DATA_TYPE,L<:LOSS_TYPE,P<:PopMember{T,L}}
    if options.batching
        dispatch_optimize_constants(
            dataset, member, options, batch_sample(dataset, options)
        )
    else
        dispatch_optimize_constants(dataset, member, options, nothing)
    end
end
function dispatch_optimize_constants(
    dataset::Dataset{T,L}, member::P, options::AbstractOptions, idx
) where {T<:DATA_TYPE,L<:LOSS_TYPE,P<:PopMember{T,L}}
    nconst = count_constants_for_optimization(member.tree)
    nconst == 0 && return (member, 0.0)
    if nconst == 1 && !(T <: Complex)
        algorithm = Optim.Newton(; linesearch=LineSearches.BackTracking())
        return _optimize_constants(
            dataset,
            member,
            specialized_options(options),
            algorithm,
            options.optimizer_options,
            idx,
        )
    end
    return _optimize_constants(
        dataset,
        member,
        specialized_options(options),
        # We use specialized options here due to Enzyme being
        # more particular about dynamic dispatch
        options.optimizer_algorithm,
        options.optimizer_options,
        idx,
    )
end

"""How many constants will be optimized."""
count_constants_for_optimization(ex::Expression) = count_scalar_constants(ex)

function _optimize_constants(
    dataset, member::P, options, algorithm, optimizer_options, idx
)::Tuple{P,Float64} where {T,L,P<:PopMember{T,L}}
    tree = member.tree
    eval_fraction = options.batching ? (options.batch_size / dataset.n) : 1.0
    x0, refs = get_scalar_constants(tree)
    @assert count_constants_for_optimization(tree) == length(x0)
    f = Evaluator(tree, refs, dataset, options, idx)
    fg! = GradEvaluator(f, options.autodiff_backend)
    obj = if algorithm isa Optim.Newton || options.autodiff_backend === nothing
        f
    else
        Optim.only_fg!(fg!)
    end
    baseline = f(x0)
    result = Optim.optimize(obj, x0, algorithm, optimizer_options)
    num_evals = result.f_calls * eval_fraction
    # Try other initial conditions:
    for _ in 1:(options.optimizer_nrestarts)
        eps = randn(T, size(x0)...)
        xt = @. x0 * (T(1) + T(1//2) * eps)
        tmpresult = Optim.optimize(obj, xt, algorithm, optimizer_options)
        num_evals += tmpresult.f_calls * eval_fraction
        # TODO: Does this need to take into account h_calls?

        if tmpresult.minimum < result.minimum
            result = tmpresult
        end
    end

    if result.minimum < baseline
        member.tree = tree
        member.loss = f(result.minimizer; regularization=true)
        member.score = loss_to_score(
            member.loss, dataset.use_baseline, dataset.baseline_loss, member, options
        )
        member.birth = get_birth_order(; deterministic=options.deterministic)
        num_evals += eval_fraction
    else
        set_scalar_constants!(member.tree, x0, refs)
    end

    return member, num_evals
end

struct Evaluator{N<:AbstractExpression,R,D<:Dataset,O<:AbstractOptions,I} <: Function
    tree::N
    refs::R
    dataset::D
    options::O
    idx::I
end
function (e::Evaluator)(x::AbstractVector; regularization=false)
    set_scalar_constants!(e.tree, x, e.refs)
    return eval_loss(e.tree, e.dataset, e.options; regularization, e.idx)
end
struct GradEvaluator{F<:Evaluator,AD<:Union{Nothing,AbstractADType},EX} <: Function
    f::F
    backend::AD
    extra::EX
end
GradEvaluator(f::F, backend::AD) where {F,AD} = GradEvaluator(f, backend, nothing)

function (g::GradEvaluator{<:Any,AD})(_, G, x::AbstractVector) where {AD}
    AD isa AutoEnzyme && error("Please load the `Enzyme.jl` package.")
    set_scalar_constants!(g.f.tree, x, g.f.refs)
    (val, grad) = try
        value_and_gradient(g.backend, g.f.tree) do tree
            eval_loss(tree, g.f.dataset, g.f.options; regularization=false, idx=g.f.idx)
        end
    catch e
        (g.backend isa AutoZygote && occursin(r"ZygoteNothingError", string(e))) || throw(e)
        (
            eval_loss(
                g.f.tree, g.f.dataset, g.f.options; regularization=false, idx=g.f.idx
            ),
            nothing,
        )
    end
    if G !== nothing && grad !== nothing
        G .= extract_gradient(grad, g.f.tree)
    end
    return val
end

end
