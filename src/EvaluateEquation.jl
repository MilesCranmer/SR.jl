module EvaluateEquationModule

import ..CoreModule: Node, Options
import ..UtilsModule: @return_on_false, is_bad_array

macro return_on_check(val, T, n)
    # This will generate the following code:
    # if !isfinite(val)
    #     return (Array{T, 1}(undef, n), false)
    # end

    :(
        if !isfinite($(esc(val)))
            return (Array{$(esc(T)),1}(undef, $(esc(n))), false)
        end
    )
end

macro return_on_nonfinite_array(array, T, n)
    :(
        if is_bad_array($(esc(array)))
            return (Array{$(esc(T)),1}(undef, $(esc(n))), false)
        end
    )
end

"""
    eval_tree_array(tree::Node, cX::AbstractMatrix{T}, options::Options)

Evaluate a binary tree (equation) over a given input data matrix. The
options contain all of the operators used. This function fuses doublets
and triplets of operations for lower memory usage.

This function can be represented by the following pseudocode:

```
function eval(current_node)
    if current_node is leaf
        return current_node.value
    elif current_node is degree 1
        return current_node.operator(eval(current_node.left_child))
    else
        return current_node.operator(eval(current_node.left_child), eval(current_node.right_child))
```
The bulk of the code is for optimizations and pre-emptive NaN/Inf checks,
which speed up evaluation significantly.

# Returns

- `(output, complete)::Tuple{AbstractVector{T}, Bool}`: the result,
    which is a 1D array, as well as if the evaluation completed
    successfully (true/false). A `false` complete means an infinity
    or nan was encountered, and a large loss should be assigned
    to the equation.
"""
function eval_tree_array(
    tree::Node, cX::AbstractMatrix{T}, options::Options
)::Tuple{AbstractVector{T},Bool} where {T<:Real}
    n = size(cX, 2)
    result, finished = _eval_tree_array(tree, cX, options)
    @return_on_false finished result
    @return_on_nonfinite_array result T n
    return result, finished
end



function _eval_tree_array(
    tree::Node, cX::AbstractMatrix{T}, options::Options
)::Tuple{AbstractVector{T},Bool} where {T<:Real}
    if tree.degree == 0
        deg0_eval(tree, cX, options)
    elseif tree.degree == 1
        # TODO: We could all do Val(tree.l.degree) here, instead of having
        # different kernels for const vs data.

        # We fuse (and compile) the following:
        #  - op(op2(x, y)), where x, y, z are constants or variables.
        #  - op(op2(x)), where x is a constant or variable.
        #  - op(x), for any x.
        if tree.l.degree == 2 && tree.l.l.degree == 0 && tree.l.r.degree == 0
            deg1_l2_ll0_lr0_eval(tree, cX, Val(tree.op), Val(tree.l.op), options)
        elseif tree.l.degree == 1 && tree.l.l.degree == 0
            deg1_l1_ll0_eval(tree, cX, Val(tree.op), Val(tree.l.op), options)
        else
            deg1_eval(tree, cX, Val(tree.op), options)
        end
    else
        # We fuse (and compile) the following:
        #  - op(x, y), where x, y are constants or variables.
        #  - op(x, y), where x is a constant or variable but y is not.
        #  - op(x, y), where y is a constant or variable but x is not.
        #  - op(x, y), for any x or y
        # TODO - add op(op2(x, y), z) and op(x, op2(y, z))
        if tree.l.degree == 0 && tree.r.degree == 0
            deg2_l0_r0_eval(tree, cX, Val(tree.op), options)
        elseif tree.l.degree == 0
            deg2_l0_eval(tree, cX, Val(tree.op), options)
        elseif tree.r.degree == 0
            deg2_r0_eval(tree, cX, Val(tree.op), options)
        else
            deg2_eval(tree, cX, Val(tree.op), options)
        end
    end
end

function deg2_eval(
    tree::Node, cX::AbstractMatrix{T}, ::Val{op_idx}, options::Options
)::Tuple{AbstractVector{T},Bool} where {T<:Real,op_idx}
    n = size(cX, 2)
    (cumulator, complete) = _eval_tree_array(tree.l, cX, options)
    @return_on_false complete cumulator
    @return_on_nonfinite_array cumulator T n
    (array2, complete2) = _eval_tree_array(tree.r, cX, options)
    @return_on_false complete2 cumulator
    @return_on_nonfinite_array array2 T n
    op = options.binops[op_idx]

    # We check inputs (and intermediates), not outputs.
    @inbounds @simd for j in 1:n
        x = op(cumulator[j], array2[j])::T
        cumulator[j] = x
    end
    # return (cumulator, finished_loop) #
    return (cumulator, true)
end

function deg1_eval(
    tree::Node, cX::AbstractMatrix{T}, ::Val{op_idx}, options::Options
)::Tuple{AbstractVector{T},Bool} where {T<:Real,op_idx}
    n = size(cX, 2)
    (cumulator, complete) = _eval_tree_array(tree.l, cX, options)
    @return_on_false complete cumulator
    @return_on_nonfinite_array cumulator T n
    op = options.unaops[op_idx]
    @inbounds @simd for j in 1:n
        x = op(cumulator[j])::T
        cumulator[j] = x
    end
    return (cumulator, true) #
end

function deg0_eval(
    tree::Node, cX::AbstractMatrix{T}, options::Options
)::Tuple{AbstractVector{T},Bool} where {T<:Real}
    n = size(cX, 2)
    if tree.constant
        return (fill(convert(T, tree.val), n), true)
    else
        return (cX[tree.feature, :], true)
    end
end

function deg1_l2_ll0_lr0_eval(
    tree::Node, cX::AbstractMatrix{T}, ::Val{op_idx}, ::Val{op_l_idx}, options::Options
)::Tuple{AbstractVector{T},Bool} where {T<:Real,op_idx,op_l_idx}
    n = size(cX, 2)
    op = options.unaops[op_idx]
    op_l = options.binops[op_l_idx]
    if tree.l.l.constant && tree.l.r.constant
        val_ll = convert(T, tree.l.l.val)
        val_lr = convert(T, tree.l.r.val)
        @return_on_check val_ll T n
        @return_on_check val_lr T n
        x_l = op_l(val_ll, val_lr)::T
        @return_on_check x_l T n
        x = op(x_l)::T
        @return_on_check x T n
        return (fill(x, n), true)
    elseif tree.l.l.constant
        val_ll = convert(T, tree.l.l.val)
        @return_on_check val_ll T n
        feature_lr = tree.l.r.feature
        cumulator = Array{T,1}(undef, n)
        @inbounds @simd for j in 1:n
            x_l = op_l(val_ll, cX[feature_lr, j])::T
            x = isfinite(x_l) ? op(x_l)::T : T(Inf) # These will get discovered by _eval_tree_array at end.
            cumulator[j] = x
        end
        return (cumulator, true)
    elseif tree.l.r.constant
        feature_ll = tree.l.l.feature
        val_lr = convert(T, tree.l.r.val)
        @return_on_check val_lr T n
        cumulator = Array{T,1}(undef, n)
        @inbounds @simd for j in 1:n
            x_l = op_l(cX[feature_ll, j], val_lr)::T
            x = isfinite(x_l) ? op(x_l)::T : T(Inf)
            cumulator[j] = x
        end
        return (cumulator, true)
    else
        feature_ll = tree.l.l.feature
        feature_lr = tree.l.r.feature
        cumulator = Array{T,1}(undef, n)
        @inbounds @simd for j in 1:n
            x_l = op_l(cX[feature_ll, j], cX[feature_lr, j])::T
            x = isfinite(x_l) ? op(x_l)::T : T(Inf)
            cumulator[j] = x
        end
        return (cumulator, true)
    end
end

# op(op2(x)) for x variable or constant
function deg1_l1_ll0_eval(
    tree::Node, cX::AbstractMatrix{T}, ::Val{op_idx}, ::Val{op_l_idx}, options::Options
)::Tuple{AbstractVector{T},Bool} where {T<:Real,op_idx,op_l_idx}
    n = size(cX, 2)
    op = options.unaops[op_idx]
    op_l = options.unaops[op_l_idx]
    if tree.l.l.constant
        val_ll = convert(T, tree.l.l.val)
        @return_on_check val_ll T n
        x_l = op_l(val_ll)::T
        @return_on_check x_l T n
        x = op(x_l)::T
        @return_on_check x T n
        return (fill(x, n), true)
    else
        feature_ll = tree.l.l.feature
        cumulator = Array{T,1}(undef, n)
        @inbounds @simd for j in 1:n
            x_l = op_l(cX[feature_ll, j])::T
            x = isfinite(x_l) ? op(x_l)::T : T(Inf)
            cumulator[j] = x
        end
        return (cumulator, true)
    end
end

function deg2_l0_r0_eval(
    tree::Node, cX::AbstractMatrix{T}, ::Val{op_idx}, options::Options
)::Tuple{AbstractVector{T},Bool} where {T<:Real,op_idx}
    n = size(cX, 2)
    op = options.binops[op_idx]
    if tree.l.constant && tree.r.constant
        val_l = convert(T, tree.l.val)
        @return_on_check val_l T n
        val_r = convert(T, tree.r.val)
        @return_on_check val_r T n
        x = op(val_l, val_r)::T
        @return_on_check x T n
        return (fill(x, n), true)
    elseif tree.l.constant
        cumulator = Array{T,1}(undef, n)
        val_l = convert(T, tree.l.val)
        @return_on_check val_l T n
        feature_r = tree.r.feature
        @inbounds @simd for j in 1:n
            x = op(val_l, cX[feature_r, j])::T
            cumulator[j] = x
        end
    elseif tree.r.constant
        cumulator = Array{T,1}(undef, n)
        feature_l = tree.l.feature
        val_r = convert(T, tree.r.val)
        @return_on_check val_r T n
        @inbounds @simd for j in 1:n
            x = op(cX[feature_l, j], val_r)::T
            cumulator[j] = x
        end
    else
        cumulator = Array{T,1}(undef, n)
        feature_l = tree.l.feature
        feature_r = tree.r.feature
        @inbounds @simd for j in 1:n
            x = op(cX[feature_l, j], cX[feature_r, j])::T
            cumulator[j] = x
        end
    end
    return (cumulator, true)
end

function deg2_l0_eval(
    tree::Node, cX::AbstractMatrix{T}, ::Val{op_idx}, options::Options
)::Tuple{AbstractVector{T},Bool} where {T<:Real,op_idx}
    n = size(cX, 2)
    (cumulator, complete) = _eval_tree_array(tree.r, cX, options)
    @return_on_false complete cumulator
    @return_on_nonfinite_array cumulator T n
    op = options.binops[op_idx]
    if tree.l.constant
        val = convert(T, tree.l.val)
        @return_on_check val T n
        @inbounds @simd for j in 1:n
            x = op(val, cumulator[j])::T
            cumulator[j] = x
        end
    else
        feature = tree.l.feature
        @inbounds @simd for j in 1:n
            x = op(cX[feature, j], cumulator[j])::T
            cumulator[j] = x
        end
    end
    return (cumulator, true)
end

function deg2_r0_eval(
    tree::Node, cX::AbstractMatrix{T}, ::Val{op_idx}, options::Options
)::Tuple{AbstractVector{T},Bool} where {T<:Real,op_idx}
    n = size(cX, 2)
    (cumulator, complete) = _eval_tree_array(tree.l, cX, options)
    @return_on_false complete cumulator
    @return_on_nonfinite_array cumulator T n
    op = options.binops[op_idx]
    if tree.r.constant
        val = convert(T, tree.r.val)
        @return_on_check val T n
        @inbounds @simd for j in 1:n
            x = op(cumulator[j], val)::T
            cumulator[j] = x
        end
    else
        feature = tree.r.feature
        @inbounds @simd for j in 1:n
            x = op(cumulator[j], cX[feature, j])::T
            cumulator[j] = x
        end
    end
    return (cumulator, true)
end

# Evaluate an equation over an array of datapoints
# This one is just for reference. The fused one should be faster.
function differentiable_eval_tree_array(
    tree::Node, cX::AbstractMatrix{T}, options::Options
)::Tuple{AbstractVector{T},Bool} where {T<:Real}
    n = size(cX, 2)
    if tree.degree == 0
        if tree.constant
            return (ones(T, n) .* tree.val, true)
        else
            return (cX[tree.feature, :], true)
        end
    elseif tree.degree == 1
        return deg1_diff_eval(tree, cX, Val(tree.op), options)
    else
        return deg2_diff_eval(tree, cX, Val(tree.op), options)
    end
end

function deg1_diff_eval(
    tree::Node, cX::AbstractMatrix{T}, ::Val{op_idx}, options::Options
)::Tuple{AbstractVector{T},Bool} where {T<:Real,op_idx}
    (left, complete) = differentiable_eval_tree_array(tree.l, cX, options)
    @return_on_false complete left
    op = options.unaops[op_idx]
    out = op.(left)
    no_nans = !any(x -> (!isfinite(x)), out)
    return (out, no_nans)
end

function deg2_diff_eval(
    tree::Node, cX::AbstractMatrix{T}, ::Val{op_idx}, options::Options
)::Tuple{AbstractVector{T},Bool} where {T<:Real,op_idx}
    (left, complete) = differentiable_eval_tree_array(tree.l, cX, options)
    @return_on_false complete left
    (right, complete2) = differentiable_eval_tree_array(tree.r, cX, options)
    @return_on_false complete2 left
    op = options.binops[op_idx]
    out = op.(left, right)
    no_nans = !any(x -> (!isfinite(x)), out)
    return (out, no_nans)
end



## Stack-based evaluation:
function _eval_tree_array_stack(
    tree::Node, cX::AbstractMatrix{T}, options::Options
)::Tuple{AbstractVector{T},Bool} where {T<:Real}
    n = size(cX, 2)
    # tree_size = count_nodes(tree)
    stack = Node[]
    processed_stack = Node[]
    # TODO: Pre-allocate this array:
    stack_results = Vector{Float32}[]
    current = tree # 2
    push!(stack, current)
    while length(stack) > 0
        top = pop!(stack)
        if top.degree == 2
            push!(stack, top.l)
            push!(stack, top.r)
        elseif top.degree == 1
            push!(stack, top.l)
        end
        push!(processed_stack, top)
    end
    # Process stack:
    # for top in reverse(processed_stack)
    while length(processed_stack) > 0
        top = pop!(processed_stack)
        if top.degree == 0
            # Evaluate and store in stack_results.
            if top.constant
                result = fill(convert(T, top.val), n)
            else
                result = cX[top.feature, :]
            end
            push!(stack_results, result)
        elseif top.degree == 1
            stack_results[end] .= options.unaops[top.op].(stack_results[end])
        else # top.degree == 2
            child_r_result = pop!(stack_results)
            stack_results[end] .=
                options.binops[top.op].(stack_results[end], child_r_result)
        end
    end
    return stack_results[1], true
end

end
