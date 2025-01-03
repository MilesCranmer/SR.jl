module ComposableExpressionModule

using DispatchDoctor: @unstable
using Compat: Fix
using DynamicExpressions:
    AbstractExpression,
    Expression,
    AbstractExpressionNode,
    AbstractOperatorEnum,
    Metadata,
    constructorof,
    get_metadata,
    eval_tree_array,
    set_node!,
    get_contents,
    with_contents,
    DynamicExpressions as DE
using DynamicExpressions.InterfacesModule:
    ExpressionInterface, Interfaces, @implements, all_ei_methods_except, Arguments
using DynamicExpressions.ValueInterfaceModule: is_valid_array

using ..ConstantOptimizationModule: ConstantOptimizationModule as CO
using ..CoreModule: get_safe_op

abstract type AbstractComposableExpression{T,N} <: AbstractExpression{T,N} end

"""
    ComposableExpression{T,N,D} <: AbstractComposableExpression{T,N} <: AbstractExpression{T,N}

A symbolic expression representing a mathematical formula as an expression tree (`tree::N`) with associated metadata (`metadata::Metadata{D}`). Used to construct and manipulate expressions in symbolic regression tasks.

Example:

Create variables `x1` and `x2`, and build an expression `f = x1 * sin(x2)`:

```julia
operators = OperatorEnum(; binary_operators=(+, *, /, -), unary_operators=(sin, cos))
variable_names = ["x1", "x2"]
x1 = ComposableExpression(Node(Float64; feature=1); operators, variable_names)
x2 = ComposableExpression(Node(Float64; feature=2); operators, variable_names)
f = x1 * sin(x2)
# ^This now references the first and second arguments of things passed to it:

f(x1, x1) # == x1 * sin(x1)
f(randn(5), randn(5)) # == randn(5) .* sin.(randn(5))

# You can even pass it to itself:
f(f, f) # == (x1 * sin(x2)) * sin((x1 * sin(x2)))
```
"""
struct ComposableExpression{
    T,
    N<:AbstractExpressionNode{T},
    D<:@NamedTuple{operators::O, variable_names::V} where {O<:AbstractOperatorEnum,V},
} <: AbstractComposableExpression{T,N}
    tree::N
    metadata::Metadata{D}
end

@inline function ComposableExpression(
    tree::AbstractExpressionNode{T};
    operators::Union{AbstractOperatorEnum,Nothing}=nothing,
    variable_names::Union{AbstractVector{<:AbstractString},Nothing}=nothing,
) where {T}
    return ComposableExpression(tree, Metadata((; operators, variable_names)))
end

@unstable DE.constructorof(::Type{<:ComposableExpression}) = ComposableExpression

DE.get_metadata(ex::AbstractComposableExpression) = ex.metadata
DE.get_contents(ex::AbstractComposableExpression) = ex.tree
DE.get_tree(ex::AbstractComposableExpression) = ex.tree

function DE.get_operators(
    ex::AbstractComposableExpression, operators::Union{AbstractOperatorEnum,Nothing}=nothing
)
    return @something(operators, DE.get_metadata(ex).operators)
end
function DE.get_variable_names(
    ex::AbstractComposableExpression,
    variable_names::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
)
    return @something(variable_names, DE.get_metadata(ex).variable_names, Some(nothing))
end

function DE.get_scalar_constants(ex::AbstractComposableExpression)
    return DE.get_scalar_constants(DE.get_contents(ex))
end
function DE.set_scalar_constants!(ex::AbstractComposableExpression, constants, refs)
    return DE.set_scalar_constants!(DE.get_contents(ex), constants, refs)
end

function Base.copy(ex::AbstractComposableExpression)
    return ComposableExpression(copy(ex.tree), copy(ex.metadata))
end

function Base.convert(::Type{E}, ex::AbstractComposableExpression) where {E<:Expression}
    return constructorof(E)(get_contents(ex), get_metadata(ex))
end

for name in (:combine_operators, :simplify_tree!)
    @eval function DE.$name(
        ex::AbstractComposableExpression{T,N},
        operators::Union{AbstractOperatorEnum,Nothing}=nothing,
    ) where {T,N}
        inner_ex = DE.$name(convert(Expression, ex), operators)
        return with_contents(ex, inner_ex)
    end
end

function DE.count_scalar_constants(ex::AbstractComposableExpression)
    return DE.count_scalar_constants(convert(Expression, ex))
end
function CO.count_constants_for_optimization(ex::AbstractComposableExpression)
    return CO.count_constants_for_optimization(convert(Expression, ex))
end

struct PreallocatedComposableExpression{N}
    tree::N
end
function DE.allocate_container(
    prototype::ComposableExpression, n::Union{Nothing,Integer}=nothing
)
    return PreallocatedComposableExpression(
        DE.allocate_container(get_contents(prototype), n)
    )
end
function DE.copy_into!(dest::PreallocatedComposableExpression, src::ComposableExpression)
    new_tree = DE.copy_into!(dest.tree, get_contents(src))
    return DE.with_contents(src, new_tree)
end

@implements(
    ExpressionInterface{all_ei_methods_except(())}, ComposableExpression, [Arguments()]
)

"""
    ValidVector{A<:AbstractVector}

A wrapper for an AbstractVector paired with a validity flag (valid::Bool).
It represents a vector along with a boolean indicating whether the data is valid.
This is useful in computations where certain operations might produce invalid data
(e.g., division by zero), allowing the validity to propagate through calculations.
Operations on `ValidVector` instances automatically handle the valid flag: if all
operands are valid, the result is valid; if any operand is invalid, the result is
marked invalid.

You will need to work with this to do highly custom operations with
`ComposableExpression` and `TemplateExpression`.

# Fields:

- `x::A`: The vector data.
- `valid::Bool`: Indicates if the data is valid.
"""
struct ValidVector{A<:AbstractVector}
    x::A
    valid::Bool
end
ValidVector(x::Tuple{Vararg{Any,2}}) = ValidVector(x...)

function (ex::AbstractComposableExpression)(x)
    return error("ComposableExpression does not support input of type $(typeof(x))")
end
function (ex::AbstractComposableExpression)(
    x::AbstractVector, _xs::Vararg{AbstractVector,N}
) where {N}
    __xs = (x, _xs...)
    # Wrap it up for the recursive call
    xs = map(Base.Fix2(ValidVector, true), __xs)
    result = ex(xs...)
    # Unwrap it
    if _is_valid(result)
        return _get_value(result)
    else
        # TODO: Make this more general. Like checking if the eltype is numeric.
        x = _get_value(result)
        nan = convert(eltype(x), NaN)
        return x .* nan
    end
end
function (ex::AbstractComposableExpression)(
    x::ValidVector, _xs::Vararg{ValidVector,N}
) where {N}
    xs = (x, _xs...)
    valid = all(_is_valid, xs)
    if !valid
        return ValidVector(_get_value(first(xs)), false)
    else
        X = Matrix(stack(map(_get_value, xs))')
        return ValidVector(eval_tree_array(ex, X))
    end
end
function (ex::AbstractComposableExpression{T})() where {T}
    X = Matrix{T}(undef, 0, 1)  # Value is irrelevant as it won't be used
    out, complete = eval_tree_array(ex, X)  # TODO: The valid is not used; not sure how to incorporate
    y = only(out)
    return complete ? y::T : nan(y)::T
end
nan(::T) where {T<:AbstractFloat} = convert(T, NaN)
nan(x) = x

function (ex::AbstractComposableExpression)(
    x::AbstractComposableExpression, _xs::Vararg{AbstractComposableExpression,N}
) where {N}
    xs = (x, _xs...)
    # To do this, we basically want to put the tree of x
    # into the position of variable 1, and so on!
    tree = copy(get_contents(ex))
    xs_trees = map(get_contents, xs)
    # TODO: This is a bit dangerous, no? We are assuming
    # that `foreach` won't try to go down the copied trees
    foreach(tree) do node
        if node.degree == 0 && !node.constant
            set_node!(node, copy(xs_trees[node.feature]))
        end
    end
    return with_contents(ex, tree)
end

# Basically we want to vectorize every single operation on ValidVector,
# so that the user can use it easily.

function apply_operator(op::F, x::Vararg{Any,N}) where {F<:Function,N}
    if all(_is_valid, x)
        vx = map(_get_value, x)
        safe_op = get_safe_op(op)
        result = safe_op.(vx...)
        return ValidVector(result, is_valid_array(result))
    else
        example_vector =
            something(map(xi -> xi isa ValidVector ? xi : nothing, x)...)::ValidVector
        return ValidVector(_get_value(example_vector), false)
    end
end
_is_valid(x::ValidVector) = x.valid
_is_valid(x) = true
_get_value(x::ValidVector) = x.x
_get_value(x) = x

#! format: off
# First, binary operators:
for op in (
    :*, :/, :+, :-, :^, :÷, :mod, :log,
    :atan, :atand, :copysign, :flipsign,
    :&, :|, :⊻, ://, :\,
)
    @eval begin
        Base.$(op)(x::ValidVector, y::ValidVector) = apply_operator(Base.$(op), x, y)
        Base.$(op)(x::ValidVector, y::Number) = apply_operator(Base.$(op), x, y)
        Base.$(op)(x::Number, y::ValidVector) = apply_operator(Base.$(op), x, y)
    end
end
function Base.literal_pow(::typeof(^), x::ValidVector, ::Val{p}) where {p}
    return apply_operator(Fix{1}(Fix{3}(Base.literal_pow, Val(p)), ^), x)
end

for op in (
    :sin, :cos, :tan, :sinh, :cosh, :tanh, :asin, :acos,
    :asinh, :acosh, :atanh, :sec, :csc, :cot, :asec, :acsc, :acot, :sech, :csch,
    :coth, :asech, :acsch, :acoth, :sinc, :cosc, :cosd, :cotd, :cscd, :secd,
    :sinpi, :cospi, :sind, :tand, :acosd, :acotd, :acscd, :asecd, :asind,
    :log, :log2, :log10, :log1p, :exp, :exp2, :exp10, :expm1, :frexp, :exponent,
    :float, :abs, :real, :imag, :conj, :unsigned,
    :nextfloat, :prevfloat, :transpose, :significand,
    :modf, :rem, :floor, :ceil, :round, :trunc,
    :inv, :sqrt, :cbrt, :abs2, :angle, :factorial,
    :(!), :-, :+, :sign, :identity,
)
    @eval Base.$(op)(x::ValidVector) = apply_operator(Base.$(op), x)
end
#! format: on

end
