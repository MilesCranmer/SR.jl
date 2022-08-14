module LossFunctionsModule

import Random: randperm
import LossFunctions: value, AggMode, SupervisedLoss
import ..CoreModule: Options, Dataset, Node
import ..EquationUtilsModule: compute_complexity
import ..EvaluateEquationModule: eval_tree_array, differentiable_eval_tree_array
import ..LossCacheModule: LossCache, maybe_get!

function loss(
    x::AbstractArray{T}, y::AbstractArray{T}, options::Options{A,B,dA,dB,C,D}
)::T where {T<:Real,A,B,dA,dB,C<:SupervisedLoss,D}
    return value(options.loss, y, x, AggMode.Mean())
end
function loss(
    x::AbstractArray{T}, y::AbstractArray{T}, options::Options{A,B,dA,dB,C,D}
)::T where {T<:Real,A,B,dA,dB,C<:Function,D}
    return sum(options.loss.(x, y)) / length(y)
end

function loss(
    x::AbstractArray{T},
    y::AbstractArray{T},
    w::AbstractArray{T},
    options::Options{A,B,dA,dB,C,D},
)::T where {T<:Real,A,B,dA,dB,C<:SupervisedLoss,D}
    return value(options.loss, y, x, AggMode.WeightedMean(w))
end
function loss(
    x::AbstractArray{T},
    y::AbstractArray{T},
    w::AbstractArray{T},
    options::Options{A,B,dA,dB,C,D},
)::T where {T<:Real,A,B,dA,dB,C<:Function,D}
    return sum(options.loss.(x, y, w)) / sum(w)
end

# Evaluate the loss of a particular expression on the input dataset.
function eval_loss(
    tree::Node,
    dataset::Dataset{T},
    options::Options;
    cache::Union{Nothing,LossCache{T}}=nothing,
)::T where {T<:Real}
    maybe_get!(cache, tree) do
        (prediction, completion) = eval_tree_array(tree, dataset.X, options)
        if !completion
            return T(1000000000)
        end

        if dataset.weighted
            return loss(prediction, dataset.y, dataset.weights, options)
        else
            return loss(prediction, dataset.y, options)
        end
    end
end

# Compute a score which includes a complexity penalty in the loss
function loss_to_score(
    loss::T, baseline::T, tree::Node, options::Options
)::T where {T<:Real}
    normalized_loss_term = loss / baseline
    size = compute_complexity(tree, options)
    parsimony_term = size * options.parsimony

    return normalized_loss_term + parsimony_term
end

# Score an equation
function score_func(
    dataset::Dataset{T},
    baseline::T,
    tree::Node,
    options::Options;
    cache::Union{Nothing,LossCache{T}}=nothing,
)::Tuple{T,T} where {T<:Real}
    result_loss = eval_loss(tree, dataset, options; cache=cache)
    score = loss_to_score(result_loss, baseline, tree, options)
    return score, result_loss
end

# Score an equation with a small batch
function score_func_batch(
    dataset::Dataset{T}, baseline::T, tree::Node, options::Options
)::Tuple{T,T} where {T<:Real}
    batch_idx = randperm(dataset.n)[1:(options.batchSize)]
    batch_X = dataset.X[:, batch_idx]
    batch_y = dataset.y[batch_idx]
    (prediction, completion) = eval_tree_array(tree, batch_X, options)
    if !completion
        return T(1000000000), T(1000000000)
    end

    if !dataset.weighted
        result_loss = loss(prediction, batch_y, options)
    else
        batch_w = dataset.weights[batch_idx]
        result_loss = loss(prediction, batch_y, batch_w, options)
    end
    score = loss_to_score(result_loss, baseline, tree, options)
    return score, result_loss
end

end
