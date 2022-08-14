module ConstantOptimizationModule

using LineSearches: LineSearches
using Optim: Optim
import ..CoreModule: CONST_TYPE, Node, Options, Dataset
import ..UtilsModule: get_birth_order
import ..EquationUtilsModule: get_constants, set_constants, count_constants
import ..LossCacheModule: LossCache
import ..LossFunctionsModule: score_func, eval_loss
import ..PopMemberModule: PopMember

# Proxy function for optimization
function opt_func(
    x::Vector{CONST_TYPE},
    dataset::Dataset{T},
    baseline::T,
    tree::Node,
    options::Options,
    cache::Union{Nothing,LossCache{T}},
)::T where {T<:Real}
    set_constants(tree, x)
    # TODO(mcranmer): This should use score_func batching.
    loss = eval_loss(tree, dataset, options; cache=cache)
    return loss
end

# Use Nelder-Mead to optimize the constants in an equation
function optimize_constants(
    dataset::Dataset{T},
    baseline::T,
    member::PopMember,
    options::Options;
    cache::Union{Nothing,LossCache{T}}=nothing,
)::Tuple{PopMember,Float64} where {T<:Real}
    nconst = count_constants(member.tree)
    num_evals = 0.0
    if nconst == 0
        return (member, 0.0)
    end
    x0 = get_constants(member.tree)
    f(x::Vector{CONST_TYPE})::T =
        opt_func(x, dataset, baseline, member.tree, options, cache)
    if nconst == 1
        algorithm = Optim.Newton(; linesearch=LineSearches.BackTracking())
    else
        if options.optimizer_algorithm == "NelderMead"
            algorithm = Optim.NelderMead(; linesearch=LineSearches.BackTracking())
        elseif options.optimizer_algorithm == "BFGS"
            algorithm = Optim.BFGS(; linesearch=LineSearches.BackTracking())#order=3))
        else
            error("Optimization function not implemented.")
        end
    end
    result = Optim.optimize(
        f, x0, algorithm, Optim.Options(; iterations=options.optimizer_iterations)
    )
    num_evals += result.f_calls
    # Try other initial conditions:
    for i in 1:(options.optimizer_nrestarts)
        new_start =
            x0 .* (
                convert(CONST_TYPE, 1) .+
                convert(CONST_TYPE, 1//2) * randn(CONST_TYPE, size(x0, 1))
            )
        tmpresult = Optim.optimize(
            f,
            new_start,
            algorithm,
            Optim.Options(; iterations=options.optimizer_iterations),
        )
        num_evals += tmpresult.f_calls

        if tmpresult.minimum < result.minimum
            result = tmpresult
        end
    end

    if Optim.converged(result)
        set_constants(member.tree, result.minimizer)
        member.score, member.loss = score_func(
            dataset, baseline, member.tree, options; cache=cache
        )
        num_evals += 1
        member.birth = get_birth_order(; deterministic=options.deterministic)
    else
        set_constants(member.tree, x0)
    end
    return member, num_evals
end

end
