module MLJInterfaceModule

using Optim: Optim
using LineSearches: LineSearches
using Logging: AbstractLogger
using MLJModelInterface: MLJModelInterface as MMI
using ADTypes: AbstractADType
using DynamicExpressions:
    eval_tree_array,
    string_tree,
    AbstractExpressionNode,
    AbstractExpression,
    Node,
    Expression,
    default_node_type,
    get_tree
using DynamicQuantities:
    QuantityArray,
    UnionAbstractQuantity,
    AbstractDimensions,
    SymbolicDimensions,
    Quantity,
    DEFAULT_DIM_BASE_TYPE,
    ustrip,
    dimension
using LossFunctions: SupervisedLoss
using ..InterfaceDynamicQuantitiesModule: get_dimensions_type
using ..InterfaceDynamicExpressionsModule: InterfaceDynamicExpressionsModule as IDE
using ..CoreModule:
    AbstractOptions,
    Options,
    Dataset,
    AbstractMutationWeights,
    MutationWeights,
    LOSS_TYPE,
    ComplexityMapping
using ..CoreModule.OptionsModule: DEFAULT_OPTIONS, OPTION_DESCRIPTIONS
using ..ComplexityModule: compute_complexity
using ..HallOfFameModule: HallOfFame, format_hall_of_fame
using ..UtilsModule: subscriptify, @ignore
using ..LoggingModule: AbstractSRLogger
using ..TemplateExpressionModule: TemplateExpression

import ..equation_search

abstract type AbstractSymbolicRegressor <: MMI.Deterministic end

abstract type AbstractSingletargetSRRegressor <: AbstractSymbolicRegressor end
abstract type AbstractMultitargetSRRegressor <: AbstractSymbolicRegressor end

# For static analysis tools:
@ignore mutable struct SRRegressor <: AbstractSingletargetSRRegressor
    selection_method::Function
end
@ignore mutable struct MultitargetSRRegressor <: AbstractMultitargetSRRegressor
    selection_method::Function
end

# TODO: To reduce code re-use, we could forward these defaults from
#       `equation_search`, similar to what we do for `Options`.

"""Generate an `SRRegressor` struct containing all the fields in `Options`."""
function modelexpr(model_name::Symbol, parent_type::Symbol=:AbstractSymbolicRegressor)
    struct_def = :(Base.@kwdef mutable struct $(model_name){
        D<:AbstractDimensions,L,E<:AbstractExpression
    } <: $parent_type
        niterations::Int = 100
        parallelism::Symbol = :multithreading
        numprocs::Union{Int,Nothing} = nothing
        procs::Union{Vector{Int},Nothing} = nothing
        addprocs_function::Union{Function,Nothing} = nothing
        heap_size_hint_in_bytes::Union{Integer,Nothing} = nothing
        worker_imports::Union{Vector{Symbol},Nothing} = nothing
        logger::Union{AbstractSRLogger,Nothing} = nothing
        runtests::Bool = true
        run_id::Union{String,Nothing} = nothing
        loss_type::Type{L} = Nothing
        selection_method::Function = choose_best
        dimensions_type::Type{D} = SymbolicDimensions{DEFAULT_DIM_BASE_TYPE}
    end)
    # TODO: store `procs` from initial run if parallelism is `:multiprocessing`
    fields = last(last(struct_def.args).args).args

    # Add everything from `Options` constructor directly to struct:
    for (i, option) in enumerate(DEFAULT_OPTIONS)
        if getsymb(first(option.args)) == :expression_type
            continue
        end
        insert!(fields, i, Expr(:(=), option.args...))
        if getsymb(first(option.args)) == :node_type
            # Manually add `expression_type` above, so it can be depended on by `node_type`
            insert!(fields, i - 1, :(expression_type::Type{E} = Expression))
        end
    end

    # We also need to create the `get_options` function, based on this:
    constructor = :(Options(;))
    constructor_fields = last(constructor.args).args
    for option in DEFAULT_OPTIONS
        symb = getsymb(first(option.args))
        push!(constructor_fields, Expr(:kw, symb, Expr(:(.), :m, Core.QuoteNode(symb))))
    end

    return quote
        $struct_def
        function get_options(m::$(model_name))
            return $constructor
        end
    end
end
function getsymb(ex::Symbol)
    return ex
end
function getsymb(ex::Expr)
    for arg in ex.args
        isa(arg, Symbol) && return arg
        s = getsymb(arg)
        isa(s, Symbol) && return s
    end
    return nothing
end

"""Get an equivalent `Options()` object for a particular regressor."""
function get_options(::AbstractSymbolicRegressor) end

eval(modelexpr(:SRRegressor, :AbstractSingletargetSRRegressor))
eval(modelexpr(:MultitargetSRRegressor, :AbstractMultitargetSRRegressor))

"""
    SRFitResultTypes

A struct referencing types in the `SRFitResult` struct,
to be used in type inference during MLJ.update to speed up iterative fits.
"""
Base.@kwdef struct SRFitResultTypes{
    _T,_X_t,_y_t,_w_t,_state,_X_units,_y_units,_X_units_clean,_y_units_clean
}
    T::Type{_T} = Any
    X_t::Type{_X_t} = Any
    y_t::Type{_y_t} = Any
    w_t::Type{_w_t} = Any
    state::Type{_state} = Any
    X_units::Type{_X_units} = Any
    y_units::Type{_y_units} = Any
    X_units_clean::Type{_X_units_clean} = Any
    y_units_clean::Type{_y_units_clean} = Any
end

"""
    SRFitResult

A struct containing the result of a fit of an `SRRegressor` or `MultitargetSRRegressor`.
"""
Base.@kwdef struct SRFitResult{
    M<:AbstractSymbolicRegressor,
    S,
    O<:AbstractOptions,
    XD<:Union{Vector{<:AbstractDimensions},Nothing},
    YD<:Union{AbstractDimensions,Vector{<:AbstractDimensions},Nothing},
    TYPES<:SRFitResultTypes,
}
    model::M
    state::S
    niterations::Int
    num_targets::Int
    options::O
    variable_names::Vector{String}
    y_variable_names::Union{Vector{String},Nothing}
    y_is_table::Bool
    has_class::Bool
    X_units::XD
    y_units::YD
    types::TYPES
end

# Cleaning already taken care of by `Options` and `equation_search`
function full_report(
    m::AbstractSymbolicRegressor,
    fitresult::SRFitResult;
    v_with_strings::Val{with_strings}=Val(true),
) where {with_strings}
    _, hof = fitresult.state
    # TODO: Adjust baseline loss
    formatted = format_hall_of_fame(hof, fitresult.options)
    equation_strings = if with_strings
        get_equation_strings_for(
            m, formatted.trees, fitresult.options, fitresult.variable_names
        )
    else
        nothing
    end
    best_idx = dispatch_selection_for(
        m, formatted.trees, formatted.losses, formatted.scores, formatted.complexities
    )
    return (;
        best_idx=best_idx,
        equations=formatted.trees,
        equation_strings=equation_strings,
        losses=formatted.losses,
        complexities=formatted.complexities,
        scores=formatted.scores,
    )
end

MMI.clean!(::AbstractSymbolicRegressor) = ""

# TODO: Enable `verbosity` being passed to `equation_search`
function MMI.fit(m::AbstractSymbolicRegressor, verbosity, X, y, w=nothing)
    return MMI.update(m, verbosity, nothing, nothing, X, y, w)
end
function MMI.update(
    m::AbstractSymbolicRegressor,
    verbosity,
    old_fitresult::Union{SRFitResult,Nothing},
    old_cache,
    X,
    y,
    w=nothing,
)
    options = old_fitresult === nothing ? get_options(m) : old_fitresult.options
    return _update(m, verbosity, old_fitresult, old_cache, X, y, w, options, nothing)
end
function _update(
    m,
    verbosity,
    old_fitresult::Union{SRFitResult,Nothing},
    old_cache,
    X,
    y,
    w,
    options,
    class,
)
    if (
        IDE.handles_class_column(m.expression_type) &&
        isnothing(class) &&
        MMI.istable(X) &&
        :class in MMI.schema(X).names
    )
        names_without_class = filter(!=(:class), MMI.schema(X).names)
        new_X = MMI.selectcols(X, collect(names_without_class))
        new_class = MMI.selectcols(X, :class)
        return _update(
            m, verbosity, old_fitresult, old_cache, new_X, y, w, options, new_class
        )
    end
    if !isnothing(old_fitresult)
        @assert(
            old_fitresult.has_class == !isnothing(class),
            "If the first fit used class, the second fit must also use class."
        )
    end
    # To speed up iterative fits, we cache the types:
    types = if isnothing(old_fitresult)
        SRFitResultTypes()
    else
        old_fitresult.types
    end
    X_t::types.X_t, variable_names, display_variable_names, X_units::types.X_units = get_matrix_and_info(
        X, m.dimensions_type
    )
    y_t::types.y_t, y_variable_names, y_units::types.y_units = format_input_for(
        m, y, m.dimensions_type
    )
    X_units_clean::types.X_units_clean = clean_units(X_units)
    y_units_clean::types.y_units_clean = clean_units(y_units)
    w_t::types.w_t = if w !== nothing && isa(m, AbstractMultitargetSRRegressor)
        @assert(isa(w, AbstractVector) && ndims(w) == 1, "Unexpected input for `w`.")
        repeat(w', size(y_t, 1))
    else
        w
    end
    niterations =
        m.niterations - (old_fitresult === nothing ? 0 : old_fitresult.niterations)
    @assert niterations >= 0
    search_state::types.state = equation_search(
        X_t,
        y_t;
        niterations=niterations,
        weights=w_t,
        variable_names=variable_names,
        display_variable_names=display_variable_names,
        options=options,
        parallelism=m.parallelism,
        numprocs=m.numprocs,
        procs=m.procs,
        addprocs_function=m.addprocs_function,
        heap_size_hint_in_bytes=m.heap_size_hint_in_bytes,
        worker_imports=m.worker_imports,
        runtests=m.runtests,
        saved_state=(old_fitresult === nothing ? nothing : old_fitresult.state),
        return_state=true,
        run_id=m.run_id,
        loss_type=m.loss_type,
        X_units=X_units_clean,
        y_units=y_units_clean,
        verbosity=verbosity,
        extra=isnothing(class) ? (;) : (; class),
        logger=m.logger,
        # Help out with inference:
        v_dim_out=isa(m, AbstractSingletargetSRRegressor) ? Val(1) : Val(2),
    )
    fitresult = SRFitResult(;
        model=m,
        state=search_state,
        niterations=niterations +
                    (old_fitresult === nothing ? 0 : old_fitresult.niterations),
        num_targets=isa(m, AbstractSingletargetSRRegressor) ? 1 : size(y_t, 1),
        options=options,
        variable_names=variable_names,
        y_variable_names=y_variable_names,
        y_is_table=MMI.istable(y),
        has_class=!isnothing(class),
        X_units=X_units_clean,
        y_units=y_units_clean,
        types=SRFitResultTypes(;
            T=hof_eltype(search_state[2]),
            X_t=typeof(X_t),
            y_t=typeof(y_t),
            w_t=typeof(w_t),
            state=typeof(search_state),
            X_units=typeof(X_units),
            y_units=typeof(y_units),
            X_units_clean=typeof(X_units_clean),
            y_units_clean=typeof(y_units_clean),
        ),
    )::(old_fitresult === nothing ? SRFitResult : typeof(old_fitresult))
    return (fitresult, nothing, full_report(m, fitresult))
end
hof_eltype(::Type{H}) where {T,H<:HallOfFame{T}} = T
hof_eltype(::Type{V}) where {V<:Vector} = hof_eltype(eltype(V))
hof_eltype(h) = hof_eltype(typeof(h))

function clean_units(units)
    !isa(units, AbstractDimensions) && error("Unexpected units.")
    iszero(units) && return nothing
    return units
end
function clean_units(units::Vector)
    !all(Base.Fix2(isa, AbstractDimensions), units) && error("Unexpected units.")
    all(iszero, units) && return nothing
    return units
end

function get_matrix_and_info(X, ::Type{D}) where {D}
    sch = MMI.istable(X) ? MMI.schema(X) : nothing
    Xm_t = MMI.matrix(X; transpose=true)
    colnames, display_colnames = if sch === nothing
        (
            ["x$(i)" for i in eachindex(axes(Xm_t, 1))],
            ["x$(subscriptify(i))" for i in eachindex(axes(Xm_t, 1))],
        )
    else
        ([string(name) for name in sch.names], [string(name) for name in sch.names])
    end
    D_promoted = get_dimensions_type(Xm_t, D)
    Xm_t_strip, X_units = unwrap_units_single(Xm_t, D_promoted)
    return Xm_t_strip, colnames, display_colnames, X_units
end

function format_input_for(::AbstractSingletargetSRRegressor, y, ::Type{D}) where {D}
    @assert(
        !(MMI.istable(y) || (length(size(y)) == 2 && size(y, 2) > 1)),
        "For multi-output regression, please use `MultitargetSRRegressor`."
    )
    y_t = vec(y)
    colnames = nothing
    D_promoted = get_dimensions_type(y_t, D)
    y_t_strip, y_units = unwrap_units_single(y_t, D_promoted)
    return y_t_strip, colnames, y_units
end
function format_input_for(::AbstractMultitargetSRRegressor, y, ::Type{D}) where {D}
    @assert(
        MMI.istable(y) || (length(size(y)) == 2 && size(y, 2) > 1),
        "For single-output regression, please use `SRRegressor`."
    )
    out = get_matrix_and_info(y, D)
    return out[1], out[2], out[4]
end
function validate_variable_names(variable_names, fitresult::SRFitResult)
    @assert(
        variable_names == fitresult.variable_names,
        "Variable names do not match fitted regressor."
    )
    return nothing
end
function validate_units(X_units, old_X_units)
    @assert(
        all(X_units .== old_X_units),
        "Units of new data do not match units of fitted regressor."
    )
    return nothing
end

# TODO: Test whether this conversion poses any issues in data normalization...
function dimension_with_fallback(q::UnionAbstractQuantity{T}, ::Type{D}) where {T,D}
    return dimension(convert(Quantity{T,D}, q))::D
end
function dimension_with_fallback(_, ::Type{D}) where {D}
    return D()
end
function prediction_warn()
    @warn "Evaluation failed either due to NaNs detected or due to unfinished search. Using 0s for prediction."
end

@inline function wrap_units(v, y_units, i::Integer)
    if y_units === nothing
        return v
    else
        return (yi -> Quantity(yi, y_units[i])).(v)
    end
end
@inline function wrap_units(v, y_units, ::Nothing)
    if y_units === nothing
        return v
    else
        return (yi -> Quantity(yi, y_units)).(v)
    end
end

function prediction_fallback(
    ::Type{T}, m::AbstractSingletargetSRRegressor, Xnew_t, fitresult::SRFitResult, _
) where {T}
    prediction_warn()
    out = fill!(similar(Xnew_t, T, axes(Xnew_t, 2)), zero(T))
    return wrap_units(out, fitresult.y_units, nothing)
end
function prediction_fallback(
    ::Type{T}, ::AbstractMultitargetSRRegressor, Xnew_t, fitresult::SRFitResult, prototype
) where {T}
    prediction_warn()
    out_cols = [
        wrap_units(
            fill!(similar(Xnew_t, T, axes(Xnew_t, 2)), zero(T)), fitresult.y_units, i
        ) for i in 1:(fitresult.num_targets)
    ]
    out_matrix = reduce(hcat, out_cols)
    if !fitresult.y_is_table
        return out_matrix
    else
        return MMI.table(out_matrix; names=fitresult.y_variable_names, prototype=prototype)
    end
end

compat_ustrip(A::QuantityArray) = ustrip(A)
compat_ustrip(A) = ustrip.(A)

"""
    unwrap_units_single(::AbstractArray, ::Type{<:AbstractDimensions})

Remove units from some features in a matrix, and return, as a tuple,
(1) the matrix with stripped units, and (2) the dimensions for those features.
"""
function unwrap_units_single(A::AbstractMatrix, ::Type{D}) where {D}
    dims = D[dimension_with_fallback(first(row), D) for row in eachrow(A)]
    @inbounds for (i, row) in enumerate(eachrow(A))
        all(xi -> dimension_with_fallback(xi, D) == dims[i], row) ||
            error("Inconsistent units in feature $i of matrix.")
    end
    return stack(compat_ustrip, eachrow(A); dims=1)::AbstractMatrix, dims
end
function unwrap_units_single(v::AbstractVector, ::Type{D}) where {D}
    dims = dimension_with_fallback(first(v), D)
    all(xi -> dimension_with_fallback(xi, D) == dims, v) ||
        error("Inconsistent units in vector.")
    return compat_ustrip(v)::AbstractVector, dims
end

function MMI.fitted_params(m::AbstractSymbolicRegressor, fitresult::SRFitResult)
    report = full_report(m, fitresult)
    return (;
        best_idx=report.best_idx,
        equations=report.equations,
        equation_strings=report.equation_strings,
    )
end

function eval_tree_mlj(
    tree::AbstractExpression,
    X_t,
    class,
    m::AbstractSymbolicRegressor,
    ::Type{T},
    fitresult,
    i,
    prototype,
) where {T}
    out, completed = if isnothing(class)
        eval_tree_array(tree, X_t, fitresult.options)
    else
        eval_tree_array(tree, X_t, class, fitresult.options)
    end
    if completed
        return wrap_units(out, fitresult.y_units, i)
    else
        return prediction_fallback(T, m, X_t, fitresult, prototype)
    end
end

function MMI.predict(
    m::M, fitresult, Xnew; idx=nothing, class=nothing
) where {M<:AbstractSymbolicRegressor}
    return _predict(m, fitresult, Xnew, idx, class)
end
function _predict(m::M, fitresult, Xnew, idx, class) where {M<:AbstractSymbolicRegressor}
    if Xnew isa NamedTuple && (haskey(Xnew, :idx) || haskey(Xnew, :data))
        @assert(
            haskey(Xnew, :idx) && haskey(Xnew, :data) && length(keys(Xnew)) == 2,
            "If specifying an equation index during prediction, you must use a named tuple with keys `idx` and `data`."
        )
        return _predict(m, fitresult, Xnew.data, Xnew.idx, class)
    end
    if (
        IDE.handles_class_column(m.expression_type) &&
        isnothing(class) &&
        MMI.istable(Xnew) &&
        :class in MMI.schema(Xnew).names
    )
        names_without_class = filter(!=(:class), MMI.schema(Xnew).names)
        Xnew2 = MMI.selectcols(Xnew, collect(names_without_class))
        class = MMI.selectcols(Xnew, :class)
        return _predict(m, fitresult, Xnew2, idx, class)
    end

    if fitresult.has_class
        @assert(
            !isnothing(class), "Classes must be specified if the model was fit with class."
        )
    end

    params = full_report(m, fitresult; v_with_strings=Val(false))
    prototype = MMI.istable(Xnew) ? Xnew : nothing
    Xnew_t, variable_names, _, X_units = get_matrix_and_info(Xnew, m.dimensions_type)
    T = promote_type(eltype(Xnew_t), fitresult.types.T)

    if isempty(params.equations) || any(isempty, params.equations)
        @warn "Equations not found. Returning 0s for prediction."
        return prediction_fallback(T, m, Xnew_t, fitresult, prototype)
    end

    X_units_clean = clean_units(X_units)
    validate_variable_names(variable_names, fitresult)
    validate_units(X_units_clean, fitresult.X_units)

    _idx = something(idx, params.best_idx)

    if M <: AbstractSingletargetSRRegressor
        return eval_tree_mlj(
            params.equations[_idx], Xnew_t, class, m, T, fitresult, nothing, prototype
        )
    elseif M <: AbstractMultitargetSRRegressor
        outs = [
            eval_tree_mlj(
                params.equations[i][_idx[i]], Xnew_t, class, m, T, fitresult, i, prototype
            ) for i in eachindex(_idx, params.equations)
        ]
        out_matrix = reduce(hcat, outs)
        if !fitresult.y_is_table
            return out_matrix
        else
            return MMI.table(out_matrix; names=fitresult.y_variable_names, prototype)
        end
    end
end

function get_equation_strings_for(
    ::AbstractSingletargetSRRegressor, trees, options, variable_names
)
    return (
        t -> string_tree(t, options; variable_names=variable_names, pretty=false)
    ).(trees)
end
function get_equation_strings_for(
    ::AbstractMultitargetSRRegressor, trees, options, variable_names
)
    return [
        (t -> string_tree(t, options; variable_names=variable_names, pretty=false)).(ts) for
        ts in trees
    ]
end

function choose_best(; trees, losses::Vector{L}, scores, complexities) where {L<:LOSS_TYPE}
    # Same as in PySR:
    # https://github.com/MilesCranmer/PySR/blob/e74b8ad46b163c799908b3aa4d851cf8457c79ef/pysr/sr.py#L2318-L2332
    # threshold = 1.5 * minimum_loss
    # Then, we get max score of those below the threshold.
    threshold = 1.5 * minimum(losses)
    return argmax([
        (losses[i] <= threshold) ? scores[i] : typemin(L) for i in eachindex(losses)
    ])
end

function dispatch_selection_for(
    m::AbstractSingletargetSRRegressor, trees, losses, scores, complexities
)::Int
    length(trees) == 0 && return 0
    return m.selection_method(;
        trees=trees, losses=losses, scores=scores, complexities=complexities
    )
end
function dispatch_selection_for(
    m::AbstractMultitargetSRRegressor, trees, losses, scores, complexities
)
    any(t -> length(t) == 0, trees) && return fill(0, length(trees))
    return [
        m.selection_method(;
            trees=trees[i], losses=losses[i], scores=scores[i], complexities=complexities[i]
        ) for i in eachindex(trees)
    ]
end

MMI.metadata_pkg(
    AbstractSymbolicRegressor;
    name="SymbolicRegression",
    uuid="8254be44-1295-4e6a-a16d-46603ac705cb",
    url="https://github.com/MilesCranmer/SymbolicRegression.jl",
    julia=true,
    license="Apache-2.0",
    is_wrapper=false,
)

const input_scitype = Union{
    MMI.Table(MMI.Continuous),
    AbstractMatrix{<:MMI.Continuous},
    MMI.Table(MMI.Continuous, MMI.Count),
}

# TODO: Allow for Count data, and coerce it into Continuous as needed.
MMI.metadata_model(
    SRRegressor;
    input_scitype,
    target_scitype=AbstractVector{<:MMI.Continuous},
    supports_weights=true,
    reports_feature_importances=false,
    load_path="SymbolicRegression.MLJInterfaceModule.SRRegressor",
    human_name="Symbolic Regression via Evolutionary Search",
)
MMI.metadata_model(
    MultitargetSRRegressor;
    input_scitype,
    target_scitype=Union{MMI.Table(MMI.Continuous),AbstractMatrix{<:MMI.Continuous}},
    supports_weights=true,
    reports_feature_importances=false,
    load_path="SymbolicRegression.MLJInterfaceModule.MultitargetSRRegressor",
    human_name="Multi-Target Symbolic Regression via Evolutionary Search",
)

function MMI.target_scitype(
    ::Type{<:SRRegressor{D,L,E}}
) where {D<:AbstractDimensions,L,E<:TemplateExpression}
    return AbstractVector{<:MMI.Unknown}
end
function MMI.target_scitype(
    ::Type{<:MultitargetSRRegressor{D,L,E}}
) where {D<:AbstractDimensions,L,E<:TemplateExpression}
    return Union{MMI.Table(MMI.Unknown),AbstractMatrix{<:MMI.Unknown}}
end

function tag_with_docstring(model_name::Symbol, description::String, bottom_matter::String)
    docstring = """$(MMI.doc_header(eval(model_name)))

    $(description)

    # Hyper-parameters
    """

    # TODO: These ones are copied (or written) manually:
    append_arguments = """- `niterations::Int=10`: The number of iterations to perform the search.
        More iterations will improve the results.
    - `parallelism=:multithreading`: What parallelism mode to use.
        The options are `:multithreading`, `:multiprocessing`, and `:serial`.
        By default, multithreading will be used. Multithreading uses less memory,
        but multiprocessing can handle multi-node compute. If using `:multithreading`
        mode, the number of threads available to julia are used. If using
        `:multiprocessing`, `numprocs` processes will be created dynamically if
        `procs` is unset. If you have already allocated processes, pass them
        to the `procs` argument and they will be used.
        You may also pass a string instead of a symbol, like `"multithreading"`.
    - `numprocs::Union{Int, Nothing}=nothing`:  The number of processes to use,
        if you want `equation_search` to set this up automatically. By default
        this will be `4`, but can be any number (you should pick a number <=
        the number of cores available).
    - `procs::Union{Vector{Int}, Nothing}=nothing`: If you have set up
        a distributed run manually with `procs = addprocs()` and `@everywhere`,
        pass the `procs` to this keyword argument.
    - `addprocs_function::Union{Function, Nothing}=nothing`: If using multiprocessing
        (`parallelism=:multithreading`), and are not passing `procs` manually,
        then they will be allocated dynamically using `addprocs`. However,
        you may also pass a custom function to use instead of `addprocs`.
        This function should take a single positional argument,
        which is the number of processes to use, as well as the `lazy` keyword argument.
        For example, if set up on a slurm cluster, you could pass
        `addprocs_function = addprocs_slurm`, which will set up slurm processes.
    - `heap_size_hint_in_bytes::Union{Int,Nothing}=nothing`: On Julia 1.9+, you may set the `--heap-size-hint`
        flag on Julia processes, recommending garbage collection once a process
        is close to the recommended size. This is important for long-running distributed
        jobs where each process has an independent memory, and can help avoid
        out-of-memory errors. By default, this is set to `Sys.free_memory() / numprocs`.
    - `worker_imports::Union{Vector{Symbol},Nothing}=nothing`: If you want to import
        additional modules on each worker, pass them here as a vector of symbols.
        By default some of the extensions will automatically be loaded when needed.
    - `runtests::Bool=true`: Whether to run (quick) tests before starting the
        search, to see if there will be any problems during the equation search
        related to the host environment.
    - `run_id::Union{String,Nothing}=nothing`: A unique identifier for the run.
        This will be used to store outputs from the run in the `outputs` directory.
        If not specified, a unique ID will be generated.
    - `loss_type::Type=Nothing`: If you would like to use a different type
        for the loss than for the data you passed, specify the type here.
        Note that if you pass complex data `::Complex{L}`, then the loss
        type will automatically be set to `L`.
    - `selection_method::Function`: Function to selection expression from
        the Pareto frontier for use in `predict`.
        See `SymbolicRegression.MLJInterfaceModule.choose_best` for an example.
        This function should return a single integer specifying
        the index of the expression to use. By default, this maximizes
        the score (a pound-for-pound rating) of expressions reaching the threshold
        of 1.5x the minimum loss. To override this at prediction time, you can pass
        a named tuple with keys `data` and `idx` to `predict`. See the Operations
        section for details.
    - `dimensions_type::AbstractDimensions`: The type of dimensions to use when storing
        the units of the data. By default this is `DynamicQuantities.SymbolicDimensions`.
    """

    bottom = """
    # Operations

    - `predict(mach, Xnew)`: Return predictions of the target given features `Xnew`, which
        should have same scitype as `X` above. The expression used for prediction is defined
        by the `selection_method` function, which can be seen by viewing `report(mach).best_idx`.
    - `predict(mach, (data=Xnew, idx=i))`: Return predictions of the target given features
        `Xnew`, which should have same scitype as `X` above. By passing a named tuple with keys
        `data` and `idx`, you are able to specify the equation you wish to evaluate in `idx`.

    $(bottom_matter)
    """

    # Remove common indentation:
    docstring = replace(docstring, r"^    " => "")
    extra_arguments = replace(append_arguments, r"^    " => "")
    bottom = replace(bottom, r"^    " => "")

    # Add parameter descriptions:
    docstring = docstring * OPTION_DESCRIPTIONS
    docstring = docstring * extra_arguments
    docstring = docstring * bottom
    return quote
        @doc $docstring $model_name
    end
end

#https://arxiv.org/abs/2305.01582
eval(
    tag_with_docstring(
        :SRRegressor,
        replace(
            """
    Single-target Symbolic Regression regressor (`SRRegressor`) searches
    for symbolic expressions that predict a single target variable from
    a set of input variables. All data is assumed to be `Continuous`.
    The search is performed using an evolutionary algorithm.
    This algorithm is described in the paper
    https://arxiv.org/abs/2305.01582.

    # Training data

    In MLJ or MLJBase, bind an instance `model` to data with

        mach = machine(model, X, y)

    OR

        mach = machine(model, X, y, w)

    Here:

    - `X` is any table of input features (eg, a `DataFrame`) whose columns are of scitype
      `Continuous`; check column scitypes with `schema(X)`. Variable names in discovered
      expressions will be taken from the column names of `X`, if available. Units in columns
      of `X` (use `DynamicQuantities` for units) will trigger dimensional analysis to be used.

    - `y` is the target, which can be any `AbstractVector` whose element scitype is
        `Continuous`; check the scitype with `scitype(y)`. Units in `y` (use `DynamicQuantities`
        for units) will trigger dimensional analysis to be used.

    - `w` is the observation weights which can either be `nothing` (default) or an
      `AbstractVector` whose element scitype is `Count` or `Continuous`.

    Train the machine using `fit!(mach)`, inspect the discovered expressions with
    `report(mach)`, and predict on new data with `predict(mach, Xnew)`.
    Note that unlike other regressors, symbolic regression stores a list of
    trained models. The model chosen from this list is defined by the function
    `selection_method` keyword argument, which by default balances accuracy
    and complexity. You can override this at prediction time by passing a named
    tuple with keys `data` and `idx`.

    """,
            r"^    " => "",
        ),
        replace(
            """
    # Fitted parameters

    The fields of `fitted_params(mach)` are:

    - `best_idx::Int`: The index of the best expression in the Pareto frontier,
       as determined by the `selection_method` function. Override in `predict` by passing
        a named tuple with keys `data` and `idx`.
    - `equations::Vector{Node{T}}`: The expressions discovered by the search, represented
      in a dominating Pareto frontier (i.e., the best expressions found for
      each complexity). `T` is equal to the element type
      of the passed data.
    - `equation_strings::Vector{String}`: The expressions discovered by the search,
      represented as strings for easy inspection.

    # Report

    The fields of `report(mach)` are:

    - `best_idx::Int`: The index of the best expression in the Pareto frontier,
       as determined by the `selection_method` function. Override in `predict` by passing
       a named tuple with keys `data` and `idx`.
    - `equations::Vector{Node{T}}`: The expressions discovered by the search, represented
      in a dominating Pareto frontier (i.e., the best expressions found for
      each complexity).
    - `equation_strings::Vector{String}`: The expressions discovered by the search,
      represented as strings for easy inspection.
    - `complexities::Vector{Int}`: The complexity of each expression in the Pareto frontier.
    - `losses::Vector{L}`: The loss of each expression in the Pareto frontier, according
      to the loss function specified in the model. The type `L` is the loss type, which
      is usually the same as the element type of data passed (i.e., `T`), but can differ
      if complex data types are passed.
    - `scores::Vector{L}`: A metric which considers both the complexity and loss of an expression,
      equal to the change in the log-loss divided by the change in complexity, relative to
      the previous expression along the Pareto frontier. A larger score aims to indicate
      an expression is more likely to be the true expression generating the data, but
      this is very problem-dependent and generally several other factors should be considered.

    # Examples

    ```julia
    using MLJ
    SRRegressor = @load SRRegressor pkg=SymbolicRegression
    X, y = @load_boston
    model = SRRegressor(binary_operators=[+, -, *], unary_operators=[exp], niterations=100)
    mach = machine(model, X, y)
    fit!(mach)
    y_hat = predict(mach, X)
    # View the equation used:
    r = report(mach)
    println("Equation used:", r.equation_strings[r.best_idx])
    ```

    With units and variable names:

    ```julia
    using MLJ
    using DynamicQuantities
    SRegressor = @load SRRegressor pkg=SymbolicRegression

    X = (; x1=rand(32) .* us"km/h", x2=rand(32) .* us"km")
    y = @. X.x2 / X.x1 + 0.5us"h"
    model = SRRegressor(binary_operators=[+, -, *, /])
    mach = machine(model, X, y)
    fit!(mach)
    y_hat = predict(mach, X)
    # View the equation used:
    r = report(mach)
    println("Equation used:", r.equation_strings[r.best_idx])
    ```

    See also [`MultitargetSRRegressor`](@ref).
    """,
            r"^    " => "",
        ),
    ),
)
eval(
    tag_with_docstring(
        :MultitargetSRRegressor,
        replace(
            """
    Multi-target Symbolic Regression regressor (`MultitargetSRRegressor`)
    conducts several searches for expressions that predict each target variable
    from a set of input variables. All data is assumed to be `Continuous`.
    The search is performed using an evolutionary algorithm.
    This algorithm is described in the paper
    https://arxiv.org/abs/2305.01582.

    # Training data
    In MLJ or MLJBase, bind an instance `model` to data with

        mach = machine(model, X, y)

    OR

        mach = machine(model, X, y, w)

    Here:

    - `X` is any table of input features (eg, a `DataFrame`) whose columns are of scitype
    `Continuous`; check column scitypes with `schema(X)`. Variable names in discovered
    expressions will be taken from the column names of `X`, if available. Units in columns
    of `X` (use `DynamicQuantities` for units) will trigger dimensional analysis to be used.

    - `y` is the target, which can be any table of target variables whose element
      scitype is `Continuous`; check the scitype with `schema(y)`. Units in columns of
      `y` (use `DynamicQuantities` for units) will trigger dimensional analysis to be used.

    - `w` is the observation weights which can either be `nothing` (default) or an
      `AbstractVector` whose element scitype is `Count` or `Continuous`. The same
      weights are used for all targets.

    Train the machine using `fit!(mach)`, inspect the discovered expressions with
    `report(mach)`, and predict on new data with `predict(mach, Xnew)`.
    Note that unlike other regressors, symbolic regression stores a list of lists of
    trained models. The models chosen from each of these lists is defined by the function
    `selection_method` keyword argument, which by default balances accuracy
    and complexity. You can override this at prediction time by passing a named
    tuple with keys `data` and `idx`.

    """,
            r"^    " => "",
        ),
        replace(
            """
    # Fitted parameters

    The fields of `fitted_params(mach)` are:

    - `best_idx::Vector{Int}`: The index of the best expression in each Pareto frontier,
      as determined by the `selection_method` function. Override in `predict` by passing
      a named tuple with keys `data` and `idx`.
    - `equations::Vector{Vector{Node{T}}}`: The expressions discovered by the search, represented
      in a dominating Pareto frontier (i.e., the best expressions found for
      each complexity). The outer vector is indexed by target variable, and the inner
      vector is ordered by increasing complexity. `T` is equal to the element type
      of the passed data.
    - `equation_strings::Vector{Vector{String}}`: The expressions discovered by the search,
      represented as strings for easy inspection.

    # Report

    The fields of `report(mach)` are:

    - `best_idx::Vector{Int}`: The index of the best expression in each Pareto frontier,
       as determined by the `selection_method` function. Override in `predict` by passing
       a named tuple with keys `data` and `idx`.
    - `equations::Vector{Vector{Node{T}}}`: The expressions discovered by the search, represented
      in a dominating Pareto frontier (i.e., the best expressions found for
      each complexity). The outer vector is indexed by target variable, and the inner
      vector is ordered by increasing complexity.
    - `equation_strings::Vector{Vector{String}}`: The expressions discovered by the search,
      represented as strings for easy inspection.
    - `complexities::Vector{Vector{Int}}`: The complexity of each expression in each Pareto frontier.
    - `losses::Vector{Vector{L}}`: The loss of each expression in each Pareto frontier, according
      to the loss function specified in the model. The type `L` is the loss type, which
      is usually the same as the element type of data passed (i.e., `T`), but can differ
      if complex data types are passed.
    - `scores::Vector{Vector{L}}`: A metric which considers both the complexity and loss of an expression,
      equal to the change in the log-loss divided by the change in complexity, relative to
      the previous expression along the Pareto frontier. A larger score aims to indicate
      an expression is more likely to be the true expression generating the data, but
      this is very problem-dependent and generally several other factors should be considered.

    # Examples

    ```julia
    using MLJ
    MultitargetSRRegressor = @load MultitargetSRRegressor pkg=SymbolicRegression
    X = (a=rand(100), b=rand(100), c=rand(100))
    Y = (y1=(@. cos(X.c) * 2.1 - 0.9), y2=(@. X.a * X.b + X.c))
    model = MultitargetSRRegressor(binary_operators=[+, -, *], unary_operators=[exp], niterations=100)
    mach = machine(model, X, Y)
    fit!(mach)
    y_hat = predict(mach, X)
    # View the equations used:
    r = report(mach)
    for (output_index, (eq, i)) in enumerate(zip(r.equation_strings, r.best_idx))
        println("Equation used for ", output_index, ": ", eq[i])
    end
    ```

    See also [`SRRegressor`](@ref).
    """,
            r"^    " => "",
        ),
    ),
)

end
