module ConstrainedExpressionModule

using Random: AbstractRNG
using DynamicExpressions:
    DynamicExpressions as DE,
    AbstractStructuredExpression,
    AbstractExpressionNode,
    AbstractExpression,
    AbstractOperatorEnum,
    OperatorEnum,
    Expression,
    Metadata,
    get_contents,
    with_contents,
    get_metadata,
    get_operators,
    get_variable_names,
    get_tree,
    node_type,
    eval_tree_array
using DynamicExpressions.InterfacesModule:
    ExpressionInterface, Interfaces, @implements, all_ei_methods_except, Arguments

using ..CoreModule: AbstractOptions, Dataset, CoreModule as CM
using ..ConstantOptimizationModule: ConstantOptimizationModule as CO
using ..InterfaceDynamicExpressionsModule: expected_array_type
using ..MutationFunctionsModule: MutationFunctionsModule as MF
using ..ExpressionBuilderModule: ExpressionBuilderModule as EB
using ..CheckConstraintsModule: CheckConstraintsModule as CC
using ..ComplexityModule: ComplexityModule
using ..LossFunctionsModule: LossFunctionsModule as LF

"""
    BlueprintExpression{T,F,N,E,TS,C,D} <: AbstractStructuredExpression{T,F,N,E,D}

A symbolic expression that allows the combination of multiple sub-expressions
in a structured way, with constraints on variable usage.

`BlueprintExpression` is designed for symbolic regression tasks where
domain-specific knowledge or constraints must be imposed on the model's structure.

# Constructor

- `BlueprintExpression(trees; structure, operators, variable_names, variable_mapping)`
    - `trees`: A `NamedTuple` holding the sub-expressions (e.g., `f = Expression(...)`, `g = Expression(...)`).
    - `structure`: A function that defines how the sub-expressions are combined. This should have one method
        that takes `trees` as input and returns a single `Expression` node, and another method which takes
        a `NamedTuple` of `Vector` (representing the numerical results of each sub-expression) and returns
        a single vector after combining them.
    - `operators`: An `OperatorEnum` that defines the allowed operators for the sub-expressions.
    - `variable_names`: An optional `Vector` of `String` that defines the names of the variables in the dataset.
    - `variable_mapping`: A `NamedTuple` that defines which variables each sub-expression is allowed to access.
        For example, requesting `f(x1, x2)` and `g(x3)` would be equivalent to `(; f=[1, 2], g=[3])`.

# Example

Let's create an example `BlueprintExpression` that combines two sub-expressions `f(x1, x2)` and `g(x3)`:

```julia
# Define operators and variable names
operators = OperatorEnum(; binary_operators=(+, *, /, -), unary_operators=(sin, cos))
variable_names = ["x1", "x2", "x3"]

# Create sub-expressions
x1 = Expression(Node{Float64}(; feature=1); operators, variable_names)
x2 = Expression(Node{Float64}(; feature=2); operators, variable_names)
x3 = Expression(Node{Float64}(; feature=3); operators, variable_names)

# Define structure function for symbolic and numerical evaluation
function my_structure(nt::NamedTuple{<:Any,<:Tuple{Vararg{<:Expression}}})
    return sin(nt.f) + nt.g * nt.g
end
function my_structure(nt::NamedTuple{<:Any,<:Tuple{Vararg{<:AbstractVector}}})
    return @. sin(nt.f) + nt.g * nt.g
end

# Define variable constraints (if desired)
variable_mapping = (; f=[1, 2], g=[3])

# Create BlueprintExpression
example_expr = (; f=x1, g=x3)
st_expr = BlueprintExpression(
    example_expr;
    structure=my_structure, operators, variable_names, variable_mapping
)
```

When fitting a model in SymbolicRegression.jl, you would provide the `BlueprintExpression`
as the `expression_type` argument, and then pass `expression_options=(; structure=my_structure, variable_mapping=variable_mapping)`
as additional options. The `variable_mapping` will constraint `f` to only have access to `x1` and `x2`,
and `g` to only have access to `x3`.
"""
struct BlueprintExpression{
    T,
    F<:Function,
    N<:AbstractExpressionNode{T},
    E<:Expression{T,N},  # TODO: Generalize this
    TS<:NamedTuple{<:Any,<:NTuple{<:Any,E}},
    C<:NamedTuple{<:Any,<:NTuple{<:Any,Vector{Int}}},  # The constraints
    # TODO: No need for this to be a parametric type
    D<:@NamedTuple{
        structure::F, operators::O, variable_names::V, variable_mapping::C
    } where {O,V},
} <: AbstractStructuredExpression{T,F,N,E,D}
    trees::TS
    metadata::Metadata{D}

    function BlueprintExpression(
        trees::TS, metadata::Metadata{D}
    ) where {
        TS,
        F<:Function,
        C<:NamedTuple{<:Any,<:NTuple{<:Any,Vector{Int}}},
        D<:@NamedTuple{
            structure::F, operators::O, variable_names::V, variable_mapping::C
        } where {O,V},
    }
        E = typeof(first(values(trees)))
        N = node_type(E)
        return new{eltype(N),F,N,E,TS,C,D}(trees, metadata)
    end
end

function BlueprintExpression(
    trees::NamedTuple{<:Any,<:NTuple{<:Any,<:AbstractExpression}};
    structure::F,
    operators::Union{AbstractOperatorEnum,Nothing}=nothing,
    variable_names::Union{AbstractVector{<:AbstractString},Nothing}=nothing,
    variable_mapping::NamedTuple{<:Any,<:NTuple{<:Any,Vector{Int}}},
) where {F<:Function}
    @assert length(trees) == length(variable_mapping)
    if variable_names !== nothing
        # TODO: Should this be removed?
        @assert Set(eachindex(variable_names)) ==
            Set(Iterators.flatten(values(variable_mapping)))
    end
    @assert keys(trees) == keys(variable_mapping)
    example_tree = first(values(trees))::AbstractExpression
    operators = get_operators(example_tree, operators)
    variable_names = get_variable_names(example_tree, variable_names)
    metadata = (; structure, operators, variable_names, variable_mapping)
    return BlueprintExpression(trees, Metadata(metadata))
end

DE.constructorof(::Type{<:BlueprintExpression}) = BlueprintExpression

@implements(
    ExpressionInterface{all_ei_methods_except(())}, BlueprintExpression, [Arguments()]
)

function EB.create_expression(
    t::AbstractExpressionNode{T},
    options::AbstractOptions,
    dataset::Dataset{T,L},
    ::Type{<:AbstractExpressionNode},
    ::Type{E},
    ::Val{embed}=Val(false),
) where {T,L,embed,E<:BlueprintExpression}
    function_keys = keys(options.expression_options.variable_mapping)

    # NOTE: We need to copy over the operators so we can call the structure function
    operators = options.operators
    variable_names = embed ? dataset.variable_names : nothing
    inner_expressions = ntuple(
        _ -> Expression(copy(t); operators, variable_names), length(function_keys)
    )
    # TODO: Generalize to other inner expression types
    return DE.constructorof(E)(
        NamedTuple{function_keys}(inner_expressions);
        EB.init_params(options, dataset, nothing, Val(embed))...,
    )
end
function EB.extra_init_params(
    ::Type{E},
    prototype::Union{Nothing,AbstractExpression},
    options::AbstractOptions,
    dataset::Dataset{T},
    ::Val{embed},
) where {T,embed,E<:BlueprintExpression}
    # We also need to include the operators here to be consistent with `create_expression`.
    return (; options.operators, options.expression_options...)
end
function EB.sort_params(params::NamedTuple, ::Type{<:BlueprintExpression})
    return (;
        params.structure, params.operators, params.variable_names, params.variable_mapping
    )
end

function ComplexityModule.compute_complexity(
    tree::BlueprintExpression, options::AbstractOptions; break_sharing=Val(false)
)
    # Rather than including the complexity of the combined tree,
    # we only sum the complexity of each inner expression, which will be smaller.
    return sum(
        ex -> ComplexityModule.compute_complexity(ex, options; break_sharing),
        values(get_contents(tree)),
    )
end

function LF.eval_tree_dispatch(
    tree::BlueprintExpression, dataset::Dataset, options::AbstractOptions, idx
)
    raw_contents = get_contents(tree)

    # Raw numerical results of each inner expression:
    outs = map(ex -> LF.eval_tree_dispatch(ex, dataset, options, idx), values(raw_contents))

    # Check for any invalid evaluations
    if !all(last, outs)
        # TODO: Would be nice to return early
        return first(first(outs)), false
    end

    # Combine them using the structure function:
    results = NamedTuple{keys(raw_contents)}(map(first, outs))
    return get_metadata(tree).structure(results), true
end

"""
We need full specialization for constrained expressions, as they rely on subexpressions being combined.
"""
CM.operator_specialization(
    ::Type{O}, ::Type{<:BlueprintExpression}
) where {O<:OperatorEnum} = O

"""
We pick a random subexpression to mutate,
and also return the symbol we mutated on so that we can put it back together later.
"""
function MF.get_contents_for_mutation(ex::BlueprintExpression, rng::AbstractRNG)
    raw_contents = get_contents(ex)
    function_keys = keys(raw_contents)
    key_to_mutate = rand(rng, function_keys)

    return raw_contents[key_to_mutate], key_to_mutate
end

"""See `get_contents_for_mutation(::BlueprintExpression, ::AbstractRNG)`."""
function MF.with_contents_for_mutation(
    ex::BlueprintExpression, new_inner_contents, context::Symbol
)
    raw_contents = get_contents(ex)
    raw_contents_keys = keys(raw_contents)
    new_contents = NamedTuple{raw_contents_keys}(
        ntuple(length(raw_contents_keys)) do i
            if raw_contents_keys[i] == context
                new_inner_contents
            else
                raw_contents[raw_contents_keys[i]]
            end
        end,
    )
    return with_contents(ex, new_contents)
end

"""We combine the operators of each inner expression."""
function DE.combine_operators(
    ex::BlueprintExpression{T,N}, operators::Union{AbstractOperatorEnum,Nothing}=nothing
) where {T,N}
    raw_contents = get_contents(ex)
    function_keys = keys(raw_contents)
    new_contents = NamedTuple{function_keys}(
        map(Base.Fix2(DE.combine_operators, operators), values(raw_contents))
    )
    return with_contents(ex, new_contents)
end

"""We simplify each inner expression."""
function DE.simplify_tree!(
    ex::BlueprintExpression{T,N}, operators::Union{AbstractOperatorEnum,Nothing}=nothing
) where {T,N}
    raw_contents = get_contents(ex)
    function_keys = keys(raw_contents)
    new_contents = NamedTuple{function_keys}(
        map(Base.Fix2(DE.simplify_tree!, operators), values(raw_contents))
    )
    return with_contents(ex, new_contents)
end

function CO.count_constants_for_optimization(ex::BlueprintExpression)
    return sum(CO.count_constants_for_optimization, values(get_contents(ex)))
end

function CC.check_constraints(
    ex::BlueprintExpression,
    options::AbstractOptions,
    maxsize::Int,
    cursize::Union{Int,Nothing}=nothing,
)::Bool
    raw_contents = get_contents(ex)
    variable_mapping = get_metadata(ex).variable_mapping

    # First, we check the variable constraints at the top level:
    has_invalid_variables = any(keys(raw_contents)) do key
        tree = raw_contents[key]
        allowed_variables = variable_mapping[key]
        contains_other_features_than(tree, allowed_variables)
    end
    if has_invalid_variables
        return false
    end

    # Then, we check other constraints for inner expressions:
    if any(t -> !CC.check_constraints(t, options, maxsize, cursize), values(raw_contents))
        return false
    end

    # Then, we check the constraints for the combined tree:
    return CC.check_constraints(get_tree(ex), options, maxsize, cursize)
end
function contains_other_features_than(tree::AbstractExpression, features)
    return contains_other_features_than(get_tree(tree), features)
end
function contains_other_features_than(tree::AbstractExpressionNode, features)
    any(tree) do node
        node.degree == 0 && !node.constant && node.feature ∉ features
    end
end

# TODO: Add custom behavior to adjust what feature nodes can be generated
# TODO: Add custom printing
# TODO: Better versions:
#  - Allow evaluation to call structure function - in which case the structure would simply combine the results.
#  - Maybe we want to do similar for string output as well. That way, the operators provided don't really matter.

end