module TemplateExpressionModule

using Random: AbstractRNG
using StatsBase: StatsBase
using Compat: Fix
using Random: default_rng
using DynamicDiff: DynamicDiff
using DispatchDoctor: @unstable, @stable
using StyledStrings: @styled_str, annotatedstring
using DynamicExpressions:
    DynamicExpressions as DE,
    AbstractStructuredExpression,
    AbstractExpressionNode,
    AbstractExpression,
    AbstractOperatorEnum,
    OperatorEnum,
    Metadata,
    get_contents,
    get_metadata,
    get_operators,
    get_variable_names,
    get_tree,
    with_metadata,
    with_contents,
    node_type,
    count_nodes,
    preserve_sharing
using DynamicExpressions.InterfacesModule:
    ExpressionInterface, Interfaces, @implements, all_ei_methods_except, Arguments
using DynamicExpressions.ExpressionModule: _copy

using ..UtilsModule: FixKws
using ..CoreModule:
    AbstractOptions,
    Options,
    Dataset,
    CoreModule as CM,
    AbstractMutationWeights,
    has_units,
    DATA_TYPE
using ..ConstantOptimizationModule: ConstantOptimizationModule as CO
using ..InterfaceDynamicExpressionsModule: InterfaceDynamicExpressionsModule as IDE
using ..MutationFunctionsModule: MutationFunctionsModule as MF
using ..ExpressionBuilderModule: ExpressionBuilderModule as EB
using ..HallOfFameModule: HallOfFameModule as HOF
using ..DimensionalAnalysisModule: DimensionalAnalysisModule as DA
using ..CheckConstraintsModule: CheckConstraintsModule as CC
using ..ComplexityModule: ComplexityModule
using ..LossFunctionsModule: LossFunctionsModule as LF
using ..MutateModule: MutateModule as MM
using ..PopMemberModule: PopMember
using ..ComposableExpressionModule: ComposableExpression, ValidVector

struct ParamVector{T} <: AbstractVector{T}
    _data::Vector{T}
end
Base.size(pv::ParamVector) = size(pv._data)
Base.getindex(pv::ParamVector, i::Integer) = pv._data[i]

# TODO: This likely slows down evaluation a bit. In the future
#       we might want to have integers passed explicitly.
Base.getindex(pv::ParamVector, i::Number) = pv[Int(i)]

function Base.setindex!(::ParamVector, ::Integer, _)
    return error(
        "ParamVector should be treated as read-only. Create a new ParamVector instead."
    )
end
function Base.getindex(pv::ParamVector, I::ValidVector)
    data = pv[I.x]
    return ValidVector(data, I.valid)
end
function Base.copy(pv::ParamVector)
    return ParamVector(copy(pv._data))
end

"""
    TemplateStructure{K,E,NF} <: Function

A struct that defines a prescribed structure for a `TemplateExpression`,
including functions that define the result in different contexts.

The `K` parameter is used to specify the symbols representing the inner expressions.
If not declared using the constructor `TemplateStructure{K}(...)`, the keys of the
`variable_constraints` `NamedTuple` will be used to infer this.

The `Kp` parameter is used to specify the symbols representing the parameters, if any.

# Fields
- `combine`: Required function taking a `NamedTuple` of `ComposableExpression`s (sharing the keys `K`),
    and then tuple representing the data of `ValidVector`s. For example,
    `((; f, g), (x1, x2, x3)) -> f(x1, x2) + g(x3)` would be a valid `combine` function. You may also
    re-use the callable expressions and use different inputs, such as
    `((; f, g), (x1, x2)) -> f(x1 + g(x2)) - g(x1)` is another valid choice.
- `num_features`: Optional `NamedTuple` of function keys => integers representing the number of
    features used by each expression. If not provided, it will be inferred using the `combine`
    function. For example, if `f` takes two arguments, and `g` takes one, then
    `num_features = (; f=2, g=1)`.
- `num_parameters`: Optional `NamedTuple` of parameter keys => integers representing the number of
    parameters required for each parameter vector.
"""
struct TemplateStructure{K,Kp,E<:Function,NF<:NamedTuple{K},NP<:NamedTuple{Kp}} <: Function
    combine::E
    num_features::NF
    num_parameters::NP
end

function TemplateStructure{K}(
    combine::E,
    _deprecated_num_features=nothing;
    num_features=nothing,
    num_parameters=nothing,
) where {K,E<:Function}
    return TemplateStructure{K,()}(
        combine, _deprecated_num_features; num_features, num_parameters
    )
end
function TemplateStructure{K,Kp}(
    combine::E,
    _deprecated_num_features=nothing;
    num_features::Union{NamedTuple{K},Nothing}=nothing,
    num_parameters::Union{NamedTuple{Kp},Nothing}=nothing,
) where {K,Kp,E<:Function}
    if _deprecated_num_features !== nothing
        Base.depwarn(
            "Passing `num_features` as an argument is deprecated, pass it explicitly as a keyword argument instead",
            :TemplateStructure,
        )
    end
    if !isempty(Kp)
        @assert(
            num_parameters !== nothing,
            "Expected `num_parameters` to be provided to indicate the number of parameters for each symbol in `$Kp`"
        )
    end
    num_parameters = @something(num_parameters, NamedTuple(),)
    num_features = @something(
        num_features,
        _deprecated_num_features,
        infer_variable_constraints(Val(K), num_parameters, combine)
    )
    return TemplateStructure{K,Kp,E,typeof(num_features),typeof(num_parameters)}(
        combine, num_features, num_parameters
    )
end

@unstable function combine(template::TemplateStructure, args...)
    return template.combine(args...)
end

# COV_EXCL_START
get_function_keys(::TemplateStructure{K}) where {K} = K
get_parameter_keys(::TemplateStructure{<:Any,Kp}) where {Kp} = Kp

has_params(s::TemplateStructure) = !isempty(get_parameter_keys(s))
# COV_EXCL_STOP

function _record_composable_expression!(variable_constraints, ::Val{k}, args...) where {k}
    vc = variable_constraints[k][]
    if vc == -1
        variable_constraints[k][] = length(args)
    elseif vc != length(args)
        throw(ArgumentError("Inconsistent number of arguments passed to $k"))
    end
    return isempty(args) ? 0.0 : first(args)
end

struct ArgumentRecorder{F} <: Function
    f::F
end
(f::ArgumentRecorder)(args...) = f.f(args...)

# We pass through the derivative operators, since
# we just want to record the number of arguments.
DynamicDiff.D(f::ArgumentRecorder, ::Integer) = f

function check_combiner_applicability(
    @nospecialize(combiner),
    @nospecialize(dummy_expressions),
    @nospecialize(dummy_params),
    @nospecialize(dummy_valid_vectors),
)
    if isempty(dummy_params)
        if !applicable(combiner, dummy_expressions, dummy_valid_vectors)
            throw(
                ArgumentError(
                    "Your template structure's `combine` function must accept\n" *
                    "\t1. A `NamedTuple` of `ComposableExpression`s (or `ArgumentRecorder`s)\n" *
                    "\t2. A tuple of `ValidVector`s",
                ),
            )
        end
    else
        if !applicable(combiner, dummy_expressions, dummy_params, dummy_valid_vectors)
            throw(
                ArgumentError(
                    "Your template structure's `combine` function must accept\n" *
                    "\t1. A `NamedTuple` of `ComposableExpression`s (or `ArgumentRecorder`s)\n" *
                    "\t2. A `NamedTuple` of `ParamVector`s\n" *
                    "\t3. A tuple of `ValidVector`s",
                ),
            )
        end
    end
    return nothing
end

"""Infers number of features used by each subexpression, by passing in test data."""
function infer_variable_constraints(
    ::Val{K}, @nospecialize(num_parameters::NamedTuple), @nospecialize(combiner)
) where {K}
    variable_constraints = NamedTuple{K}(map(_ -> Ref(-1), K))
    inner = Fix{1}(_record_composable_expression!, variable_constraints)
    dummy_expressions = NamedTuple{K}(map(k -> ArgumentRecorder(Fix{1}(inner, Val(k))), K))
    dummy_valid_vectors = Base.Iterators.repeated(ValidVector(ones(Float64, 1), true))
    dummy_params = NamedTuple{keys(num_parameters)}(
        map(n -> ParamVector(ones(Float64, n)), values(num_parameters))
    )

    check_combiner_applicability(
        combiner, dummy_expressions, dummy_params, dummy_valid_vectors
    )

    # Actually call the combiner
    if isempty(dummy_params)
        combiner(dummy_expressions, dummy_valid_vectors)
    else
        combiner(dummy_expressions, dummy_params, dummy_valid_vectors)
    end

    inferred = NamedTuple{K}(map(x -> x[], values(variable_constraints)))
    if any(==(-1), values(inferred))
        failed_keys = filter(k -> inferred[k] == -1, K)
        throw(ArgumentError("Failed to infer number of features used by $failed_keys"))
    end
    return inferred
end

"""
    TemplateExpression{T,F,N,E,TS,D} <: AbstractExpression{T,N}

A symbolic expression that allows the combination of multiple sub-expressions
in a structured way, with constraints on variable usage.

`TemplateExpression` is designed for symbolic regression tasks where
domain-specific knowledge or constraints must be imposed on the model's structure.

# Constructor

- `TemplateExpression(trees; structure, operators, variable_names)`
    - `trees`: A `NamedTuple` holding the sub-expressions (e.g., `f = Expression(...)`, `g = Expression(...)`).
    - `structure`: A `TemplateStructure` which holds functions that define how the sub-expressions are combined
        in different contexts.
    - `operators`: An `OperatorEnum` that defines the allowed operators for the sub-expressions.
    - `variable_names`: An optional `Vector` of `String` that defines the names of the variables in the dataset.

# Example

Let's create an example `TemplateExpression` that combines two sub-expressions `f(x1, x2)` and `g(x3)`:

```julia
# Define operators and variable names
options = Options(; binary_operators=(+, *, /, -), unary_operators=(sin, cos))
operators = options.operators
variable_names = ["x1", "x2", "x3"]

# Create sub-expressions
x1 = Expression(Node{Float64}(; feature=1); operators, variable_names)
x2 = Expression(Node{Float64}(; feature=2); operators, variable_names)
x3 = Expression(Node{Float64}(; feature=3); operators, variable_names)

# Create TemplateExpression
example_expr = (; f=x1, g=x3)
st_expr = TemplateExpression(
    example_expr;
    structure=TemplateStructure{(:f, :g)}(
        ((; f, g), (x1, x2, x3)) -> sin(f(x1, x2)) + g(x3)^2
    ),
    operators,
    variable_names,
)
```

When fitting a model in SymbolicRegression.jl, you would provide the `TemplateExpression`
as the `expression_type` argument, and then pass `expression_options=(; structure=TemplateStructure(...))`
as additional options. The `variable_constraints` will constraint `f` to only have access to `x1` and `x2`,
and `g` to only have access to `x3`.
"""
struct TemplateExpression{
    T,
    F<:TemplateStructure,
    N<:AbstractExpressionNode{T},
    E<:ComposableExpression{T,N},
    TS<:NamedTuple{<:Any,<:NTuple{<:Any,E}},
    D<:@NamedTuple{
        structure::F, operators::O, variable_names::V, parameters::P
    } where {O<:AbstractOperatorEnum,V,P<:NamedTuple{<:Any,<:NTuple{<:Any,ParamVector}}},
} <: AbstractExpression{T,N}
    trees::TS
    metadata::Metadata{D}

    function TemplateExpression(
        trees::TS, metadata::Metadata{D}
    ) where {
        TS,
        F<:TemplateStructure,
        D<:@NamedTuple{
            structure::F, operators::O, variable_names::V, parameters::P
        } where {O,V,P<:NamedTuple{<:Any,<:NTuple{<:Any,ParamVector}}},
    }
        @assert keys(trees) == get_function_keys(metadata.structure)
        @assert keys(metadata.parameters) == keys(metadata.structure.num_parameters)
        E = typeof(first(values(trees)))
        N = node_type(E)
        return new{eltype(N),F,N,E,TS,D}(trees, metadata)
    end
end

function TemplateExpression(
    trees::NamedTuple{<:Any,<:NTuple{<:Any,<:AbstractExpression}};
    structure::TemplateStructure,
    operators::Union{AbstractOperatorEnum,Nothing}=nothing,
    variable_names::Union{AbstractVector{<:AbstractString},Nothing}=nothing,
    parameters::Union{NamedTuple,Nothing}=nothing,
)
    example_tree = first(values(trees))::AbstractExpression
    operators = get_operators(example_tree, operators)
    variable_names = get_variable_names(example_tree, variable_names)
    parameters = if has_params(structure)
        @assert(
            parameters !== nothing,
            "Expected `parameters` to be provided for `structure.num_parameters=$(structure.num_parameters)`"
        )
        for k in keys(structure.num_parameters)
            @assert(
                length(parameters[k]) == structure.num_parameters[k],
                "Expected `parameters.$k` to have length $(structure.num_parameters[k]), got $(length(parameters[k]))"
            )
        end
        # TODO: Delete this extra check once we are confident that it works
        NamedTuple{keys(structure.num_parameters)}(
            map(p -> p isa ParamVector ? p : ParamVector(p::Vector), parameters)
        )
    else
        @assert(
            parameters === nothing || isempty(parameters),
            "Expected `parameters` to not be specified for `structure.num_parameters=$(structure.num_parameters)`"
        )
        NamedTuple()
    end
    metadata = (; structure, operators, variable_names, parameters)
    return TemplateExpression(trees, Metadata(metadata))
end

@unstable DE.constructorof(::Type{<:TemplateExpression}) = TemplateExpression  # COV_EXCL_LINE

@implements(
    ExpressionInterface{all_ei_methods_except(())}, TemplateExpression, [Arguments()]
)

has_params(ex::TemplateExpression) = has_params(get_metadata(ex).structure)

@unstable function combine(ex::TemplateExpression, args...)
    return combine(get_metadata(ex).structure, args...)
end

function Base.copy(e::TemplateExpression)
    ts = get_contents(e)
    meta = get_metadata(e)
    meta_inner = DE.ExpressionModule.unpack_metadata(meta)
    copy_ts = NamedTuple{keys(ts)}(map(copy, values(ts)))
    keys_except_structure = filter(!=(:structure), keys(meta_inner))
    copy_metadata = (;
        meta_inner.structure,
        # Note: this `_copy` is just `copy` but with handling for `nothing` and `NamedTuple`
        NamedTuple{keys_except_structure}(
            map(_copy, values(meta_inner[keys_except_structure]))
        )...,
    )
    return DE.constructorof(typeof(e))(copy_ts, Metadata(copy_metadata))
end
function DE.get_contents(e::TemplateExpression)
    return e.trees
end
function DE.get_metadata(e::TemplateExpression)
    return e.metadata
end
function DE.get_operators(
    e::TemplateExpression, operators::Union{AbstractOperatorEnum,Nothing}=nothing
)
    return @something(operators, get_metadata(e).operators)
end
function DE.get_variable_names(
    e::TemplateExpression,
    variable_names::Union{AbstractVector{<:AbstractString},Nothing}=nothing,
)
    return if variable_names !== nothing
        variable_names
    elseif hasproperty(get_metadata(e), :variable_names)
        get_metadata(e).variable_names
    else
        nothing
    end
end
function DE.get_scalar_constants(e::TemplateExpression)
    # Get constants for each inner expression
    consts_and_refs = map(DE.get_scalar_constants, values(get_contents(e)))
    parameters = get_metadata(e).parameters
    flat_constants = vcat(
        map(first, consts_and_refs)..., (has_params(e) ? values(parameters) : ())...
    )
    # Collect info so we can put them back in the right place,
    # like the indexes of the constants in the flattened array
    refs = map(c_ref -> (; n=length(first(c_ref)), ref=last(c_ref)), consts_and_refs)
    return flat_constants, refs
end
function DE.set_scalar_constants!(e::TemplateExpression, constants, refs)
    cursor = Ref(1)
    foreach(values(get_contents(e)), refs) do tree, r
        n = r.n
        i = cursor[]
        c = constants[i:(i + n - 1)]
        DE.set_scalar_constants!(tree, c, r.ref)
        cursor[] = i + n
    end
    if has_params(e)
        num_parameters = get_metadata(e).structure.num_parameters
        parameters = get_metadata(e).parameters
        for k in keys(num_parameters)
            n = num_parameters[k]
            i = cursor[]
            parameters[k]._data[:] = constants[i:(i + n - 1)]
            cursor[] = i + n
        end
    end
    return e
end

Base.@kwdef struct PreallocatedTemplateExpression{A,B}
    trees::A
    parameters::B
end

function DE.allocate_container(e::TemplateExpression, n::Union{Nothing,Integer}=nothing)
    ts = get_contents(e)
    parameters = get_metadata(e).parameters
    preallocated_trees = NamedTuple{keys(ts)}(
        map(t -> DE.allocate_container(t, n), values(ts))
    )
    preallocated_parameters = NamedTuple{keys(parameters)}(
        map(p -> similar(p), values(parameters))
    )
    return PreallocatedTemplateExpression(preallocated_trees, preallocated_parameters)
end
function DE.copy_into!(dest::PreallocatedTemplateExpression, src::TemplateExpression)
    ts = get_contents(src)
    parameters = get_metadata(src).parameters
    new_contents = NamedTuple{keys(ts)}(map(DE.copy_into!, values(dest.trees), values(ts)))
    for k in keys(parameters)
        dest.parameters[k][:] = (parameters[k]::ParamVector)[:]
    end
    new_parameters = NamedTuple{keys(parameters)}(
        map(p -> ParamVector(p), values(dest.parameters))
    )
    return with_metadata(with_contents(src, new_contents); parameters=new_parameters)
end

function DE.get_tree(ex::TemplateExpression{<:Any,<:Any,<:Any,E}) where {E}
    raw_contents = get_contents(ex)
    total_num_features = max(values(get_metadata(ex).structure.num_features)...)
    example_inner_ex = first(values(raw_contents))
    example_tree = get_contents(example_inner_ex)::AbstractExpressionNode

    variable_trees = [
        DE.constructorof(typeof(example_tree))(; feature=i) for i in 1:total_num_features
    ]
    variable_expressions = [
        with_contents(inner_ex, variable_tree) for
        (inner_ex, variable_tree) in zip(values(raw_contents), variable_trees)
    ]
    if has_params(ex)
        throw(
            ArgumentError(
                "`get_tree` is not implemented for TemplateExpression with parameters"
            ),
        )
    end

    return DE.get_tree(
        combine(get_metadata(ex).structure, raw_contents, variable_expressions)
    )
end

function EB.create_expression(
    t::AbstractExpressionNode{T},
    options::AbstractOptions,
    dataset::Dataset{T,L},
    ::Type{<:AbstractExpressionNode},
    ::Type{E},
    ::Val{embed}=Val(false),
) where {T,L,embed,E<:TemplateExpression}
    function_keys = get_function_keys(options.expression_options.structure)

    # NOTE: We need to copy over the operators so we can call the structure function
    operators = options.operators
    variable_names = embed ? dataset.variable_names : nothing
    inner_expressions = ntuple(
        _ -> ComposableExpression(copy(t); operators, variable_names),
        Val(length(function_keys)),
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
) where {T,embed,E<:TemplateExpression}
    num_parameters = options.expression_options.structure.num_parameters
    parameters = if isempty(num_parameters)
        NamedTuple()
    else
        # COV_EXCL_START
        if prototype === nothing
            NamedTuple{keys(num_parameters)}(
                map(n -> ParamVector(randn(T, (n,))), values(num_parameters))
            )
        else
            _copy(get_metadata(prototype).parameters::NamedTuple)
        end
        # COV_EXCL_STOP
    end
    # We also need to include the operators here to be consistent with `create_expression`.
    return (; options.operators, options.expression_options..., parameters)
end
function EB.sort_params(params::NamedTuple, ::Type{<:TemplateExpression})
    return (; params.structure, params.operators, params.variable_names, params.parameters)
end

function ComplexityModule.compute_complexity(
    tree::TemplateExpression, options::AbstractOptions; break_sharing=Val(false)
)
    # Rather than including the complexity of the combined tree,
    # we only sum the complexity of each inner expression, which will be smaller.
    return sum(
        ex -> ComplexityModule.compute_complexity(ex, options; break_sharing),
        values(get_contents(tree)),
    )
end

# Rather than using iterator with repeat, just make a tuple:
function _colors(::Val{n}) where {n}
    return ntuple(
        (i -> (:magenta, :green, :red, :blue, :yellow, :cyan)[mod1(i, n)]), Val(n)
    )
end
_color_string(s::AbstractString, c::Symbol) = styled"{$c:$s}"

function _format_component(ex::AbstractExpression, c::Symbol; operators, kws...)
    return _color_string(DE.string_tree(ex, operators; kws...), c)
end
function _format_component(param::ParamVector, c::Symbol; pretty, f_constant::FC) where {FC}
    p_str = if !pretty || length(param) <= 5
        join(map(f_constant, param), ", ")
    else
        string(join(map(f_constant, param[1:3]), ", "), ", ..., ", f_constant(param[end]))
    end

    return _color_string('[' * p_str * ']', c)
end
function _prefix_string_with_pipe(k::Symbol, s; all_keys, pretty)
    prefix = if !pretty || length(all_keys) == 1
        ""
    elseif k == first(all_keys)
        "╭ "
    elseif k == last(all_keys)
        "╰ "
    else
        "├ "
    end
    return annotatedstring(prefix, string(k), " = ", s)
end
function DE.string_tree(
    ex::TemplateExpression,
    operators::Union{AbstractOperatorEnum,Nothing}=nothing;
    pretty::Bool=false,
    variable_names=nothing,  # ignored
    f_constant::FC=string,
    kws...,
) where {FC}
    expressions = get_contents(ex)
    num_features = get_metadata(ex).structure.num_features
    total_num_features = max(values(num_features)...)
    variable_names = ['#' * string(i) for i in 1:total_num_features]
    parameters = has_params(ex) ? get_metadata(ex).parameters : NamedTuple()
    all_keys = (keys(num_features)..., keys(parameters)...)
    colors = _colors(Val(length(all_keys)))

    strings = NamedTuple{all_keys}((
        map(
            FixKws(
                _format_component; operators, pretty, variable_names, f_constant, kws...
            ),
            values(expressions),
            colors[1:length(expressions)],
        )...,
        map(
            FixKws(_format_component; pretty, f_constant),
            values(parameters),
            colors[(length(expressions) + 1):end],
        )...,
    ))
    prefixed_strings = NamedTuple{all_keys}(
        map(FixKws(_prefix_string_with_pipe; all_keys, pretty), all_keys, values(strings))
    )
    return annotatedstring(join(prefixed_strings, pretty ? styled"\n" : styled"; "))
end
function HOF.make_prefix(::TemplateExpression, ::AbstractOptions, ::Dataset)
    return ""
end

@stable(
    default_mode = "disable",
    default_union_limit = 2,
    begin
        function DE.eval_tree_array(
            tree::TemplateExpression,
            cX::AbstractMatrix,
            operators::Union{AbstractOperatorEnum,Nothing}=nothing;
            kws...,
        )
            raw_contents = get_contents(tree)
            metadata = get_metadata(tree)
            if has_invalid_variables(tree)
                return (nothing, false)
            end
            extra_args = if has_params(tree)
                (metadata.parameters,)
            else
                ()
            end
            result = combine(
                tree,
                raw_contents,
                extra_args...,
                map(x -> ValidVector(copy(x), true), eachrow(cX)),
            )
            return result.x, result.valid
        end
        function (ex::TemplateExpression)(
            X, operators::Union{AbstractOperatorEnum,Nothing}=nothing; kws...
        )
            result, valid = DE.eval_tree_array(ex, X, operators; kws...)
            if valid
                return result
            else
                return nothing
            end
        end
    end
)
@unstable begin
    IDE.expected_array_type(::AbstractArray, ::Type{<:TemplateExpression}) = Any
    IDE.expected_array_type(::Matrix{T}, ::Type{<:TemplateExpression}) where {T} = Any
end

function DA.violates_dimensional_constraints(
    @nospecialize(tree::TemplateExpression),
    dataset::Dataset,
    @nospecialize(options::AbstractOptions)
)
    @assert !has_units(dataset)
    return false
end
function MM.condition_mutation_weights!(
    @nospecialize(weights::AbstractMutationWeights),
    @nospecialize(member::P),
    @nospecialize(options::AbstractOptions),
    curmaxsize::Int,
) where {T,L,N<:TemplateExpression,P<:PopMember{T,L,N}}
    if !preserve_sharing(typeof(member.tree))
        weights.form_connection = 0.0
        weights.break_connection = 0.0
    end

    MM.condition_mutate_constant!(typeof(member.tree), weights, member, options, curmaxsize)

    complexity = ComplexityModule.compute_complexity(member, options)

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
We need full specialization for constrained expressions, as they rely on subexpressions being combined.
"""
function CM.operator_specialization(
    ::Type{O}, ::Type{<:TemplateExpression}
) where {O<:OperatorEnum}
    return O
end

function CM.max_features(
    dataset::Dataset, options::Options{<:Any,<:Any,<:Any,<:TemplateExpression}
)
    num_features = options.expression_options.structure.num_features
    return max(values(num_features)...)
end

"""
We pick a random subexpression to mutate,
and also return the symbol we mutated on so that we can put it back together later.
"""
function MF.get_contents_for_mutation(ex::TemplateExpression, rng::AbstractRNG)
    raw_contents = get_contents(ex)
    function_keys = keys(raw_contents)

    key_to_mutate = rand(rng, function_keys)
    return raw_contents[key_to_mutate], key_to_mutate
end

"""See `get_contents_for_mutation(::TemplateExpression, ::AbstractRNG)`."""
function MF.with_contents_for_mutation(
    ex::TemplateExpression, new_inner_contents, context::Symbol
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
function MM.condition_mutate_constant!(
    ::Type{<:TemplateExpression},
    weights::AbstractMutationWeights,
    member::PopMember,
    options::AbstractOptions,
    curmaxsize::Int,
)
    # Avoid modifying the mutate_constant weight, since
    # otherwise we would be mutating constants all the time!
    return nothing
end

"""We combine the operators of each inner expression."""
function DE.combine_operators(
    ex::TemplateExpression{T,N}, operators::Union{AbstractOperatorEnum,Nothing}=nothing
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
    ex::TemplateExpression{T,N}, operators::Union{AbstractOperatorEnum,Nothing}=nothing
) where {T,N}
    raw_contents = get_contents(ex)
    function_keys = keys(raw_contents)
    new_contents = NamedTuple{function_keys}(
        map(Base.Fix2(DE.simplify_tree!, operators), values(raw_contents))
    )
    return with_contents(ex, new_contents)
end
function MF.mutate_constant(
    ex::TemplateExpression{T},
    temperature,
    options::AbstractOptions,
    rng::AbstractRNG=default_rng(),
) where {T<:DATA_TYPE}
    regular_constant_mutation = !has_params(ex) || rand(rng, Bool)
    if regular_constant_mutation
        # Normal mutation of inner constant
        tree, context = MF.get_contents_for_mutation(ex, rng)
        new_tree = MF.mutate_constant(tree, temperature, options, rng)
        return MF.with_contents_for_mutation(ex, new_tree, context)
    else # Mutate parameters

        # We mutate between 1 and all of the parameter vector
        key_to_mutate = rand(rng, keys(get_metadata(ex).parameters))
        num_params = get_metadata(ex).structure.num_parameters[key_to_mutate]::Integer
        num_params_to_mutate = rand(rng, 1:num_params)
        # TODO: I feel we should mutate all keys at once, and only randomize which
        # parameters (of the combined list) to mutate.

        idx_to_mutate = StatsBase.sample(
            rng, 1:num_params, num_params_to_mutate; replace=false
        )
        parameters = get_metadata(ex).parameters[key_to_mutate]::ParamVector
        factors = [MF.mutate_factor(T, temperature, options, rng) for _ in idx_to_mutate]
        @inbounds for (i, f) in zip(idx_to_mutate, factors)
            parameters._data[i] *= f
        end
        return ex
    end
end
# TODO: Look at other ParametricExpression behavior

function CO.count_constants_for_optimization(ex::TemplateExpression)
    return (
        sum(CO.count_constants_for_optimization, values(get_contents(ex))) +
        (has_params(ex) ? sum(values(get_metadata(ex).structure.num_parameters)) : 0)
    )
end

function CC.check_constraints(
    ex::TemplateExpression,
    options::AbstractOptions,
    maxsize::Int,
    cursize::Union{Int,Nothing}=nothing,
)::Bool
    # First, we check the variable constraints at the top level:
    if has_invalid_variables(ex)
        return false
    end

    # We also check the combined complexity:
    @something(cursize, ComplexityModule.compute_complexity(ex, options)) > maxsize &&
        return false

    # Then, we check other constraints for inner expressions:
    raw_contents = get_contents(ex)
    for t in values(raw_contents)
        if !CC.check_constraints(t, options, maxsize, nothing)
            return false
        end
    end
    return true
    # TODO: The concept of `cursize` doesn't really make sense here.
end
function has_invalid_variables(ex::TemplateExpression)
    raw_contents = get_contents(ex)
    num_features = get_metadata(ex).structure.num_features
    any(keys(raw_contents)) do key
        tree = raw_contents[key]
        max_feature = num_features[key]
        contains_features_greater_than(tree, max_feature)
    end
end
function contains_features_greater_than(tree::AbstractExpression, max_feature)
    return contains_features_greater_than(get_tree(tree), max_feature)
end
function contains_features_greater_than(tree::AbstractExpressionNode, max_feature)
    any(tree) do node
        node.degree == 0 && !node.constant && node.feature > max_feature
    end
end

function Base.isempty(ex::TemplateExpression)
    return all(isempty, values(get_contents(ex)))
end

# TODO: Add custom behavior to adjust what feature nodes can be generated

end
