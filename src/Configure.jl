@from "Core.jl" import RecordType

# Check for errors before they happen
function testOptionConfiguration(T, options::Options)
    
    for op in (options.binops..., options.unaops...)
		if is_anonymous_function(op)
			throw(AssertionError("Anonymous functions can't be used as operators for SymbolicRegression.jl"))
        end
    end

	test_input = map(x->convert(T, x), LinRange(-100, 100, 99))
	cur_op = nothing
    try
        for left in test_input
            for right in test_input
                for binop in options.binops
					cur_op = binop
                    test_output = binop.(left, right)
                end
            end
            for unaop in options.unaops
				cur_op = unaop
                test_output = unaop.(left)
            end
        end
    catch error
        throw(AssertionError("Your configuration is invalid - one of your operators ($cur_op) is not well-defined over the real line."))
    end

    for binop in options.binops
        if binop in options.unaops
            throw(AssertionError("Your configuration is invalid - one operator ($binop) appears in both the binary operators and unary operators."))
        end
    end
end

# Check for errors before they happen
function testDatasetConfiguration(dataset::Dataset{T}, options::Options) where {T<:Real}
    n = dataset.n
    if n != size(dataset.X, 2) || n != size(dataset.y, 1)
        throw(AssertionError("Dataset dimensions are invalid. Make sure X is of shape [features, rows], y is of shape [rows] and if there are weights, they are of shape [rows]."))
    end

    if size(dataset.X, 2) > 10000
        if !options.batching
            debug((options.verbosity > 0 || options.progress), "Note: you are running with more than 10,000 datapoints. You should consider turning on batching (`options.batching`), and also if you need that many datapoints. Unless you have a large amount of noise (in which case you should smooth your dataset first), generally < 10,000 datapoints is enough to find a functional form.")
        end
    end

    if !(typeof(options.loss) <: SupervisedLoss)
        if dataset.weighted
            if !(3 in [m.nargs for m in methods(options.loss)])
                throw(AssertionError("When you create a custom loss function, and are using weights, you need to define your loss function with three scalar arguments: f(prediction, target, weight)."))
            end
        end
    end
end

""" Move custom operators and loss functions to workers, if undefined """
function move_functions_to_workers(procs, options::Options, dataset::Dataset{T}) where {T}
    for function_set=1:5
        if function_set == 1
            ops = options.unaops
            nargs = 1
        elseif function_set == 2
            ops = options.binops
            nargs = 2
        elseif function_set == 3
            ops = options.diff_unaops
            nargs = 1
        elseif function_set == 4
            ops = options.diff_binops
            nargs = 2
        elseif function_set == 5
            if typeof(options.loss) <: SupervisedLoss
                continue
            end
            ops = (options.loss,)
            nargs = dataset.weighted ? 3 : 2
        end
        for op in ops
            try
                test_function_on_workers(T, nargs, op, procs)
            catch e
                undefined_on_workers = isa(e.captured.ex, UndefVarError)
                if undefined_on_workers
                    copy_definition_to_workers(op, procs, options)
                else
                    throw(e)
                end
            end
            test_function_on_workers(T, nargs, op, procs)
        end
    end
end

function copy_definition_to_workers(op, procs, options::Options)
    name = nameof(op)
    debug_inline((options.verbosity > 0 || options.progress), "Copying definition of $op to workers...")
    src_ms = methods(op).ms
    # Thanks https://discourse.julialang.org/t/easy-way-to-send-custom-function-to-distributed-workers/22118/2
    @everywhere procs @eval function $name end
    for m in src_ms
        @everywhere procs @eval $m
    end
    debug((options.verbosity > 0 || options.progress), "Finished!")
end

function test_function_on_workers(T, nargs, op, procs)
    futures = []
    for proc in procs
        if nargs == 1
            push!(futures,
                  @spawnat proc op(convert(T, 0)))
        elseif nargs == 2 #2D ops, and loss function
            push!(futures,
                  @spawnat proc op(convert(T, 0), convert(T, 0)))
        elseif nargs == 3 #weighted loss function
            push!(futures,
                  @spawnat proc op(convert(T, 0), convert(T, 0), convert(T, 0)))
        end
    end
    for future in futures
        fetch(future)
    end
end

function activate_env_on_workers(procs, project_path::String, options::Options)
    debug((options.verbosity > 0 || options.progress), "Activating environment on workers.")
    @everywhere procs begin
        Base.MainInclude.eval(quote
            using Pkg
            Pkg.activate($$project_path)
        end)
    end
end

function import_module_on_workers(procs, filename::String, options::Options)
    included_local = !("SymbolicRegression" in [k.name for (k, v) in Base.loaded_modules])
    if included_local
        debug_inline((options.verbosity > 0 || options.progress), "Importing local module ($filename) on workers...")
        @everywhere procs begin
            # Parse functions on every worker node
            Base.MainInclude.eval(quote
                include($$filename)
                using .SymbolicRegression
            end)
        end
        debug((options.verbosity > 0 || options.progress), "Finished!")
    else
        debug_inline((options.verbosity > 0 || options.progress), "Importing installed module on workers...")
        @everywhere procs begin
            Base.MainInclude.eval(using SymbolicRegression)
        end
        debug((options.verbosity > 0 || options.progress), "Finished!")
    end
end

function test_module_on_workers(procs, options::Options)
    debug_inline((options.verbosity > 0 || options.progress), "Testing module on workers...")
    futures = []
    for proc in procs
        push!(futures,
              @spawnat proc SymbolicRegression.genRandomTree(3, options, 5))
    end
    for future in futures
        fetch(future)
    end
    debug((options.verbosity > 0 || options.progress), "Finished!")
end

function test_entire_pipeline(procs, dataset::Dataset{T}, options::Options) where {T<:Real}
    futures = []
    debug_inline((options.verbosity > 0 || options.progress), "Testing entire pipeline on workers...")
    for proc in procs
        push!(futures, @spawnat proc begin
            tmp_pop = Population(dataset, convert(T, 1), npop=20, nlength=3, options=options, nfeatures=dataset.nfeatures)
            tmp_pop = SRCycle(dataset, convert(T, 1), tmp_pop, 5, 5, ones(T, options.maxsize), verbosity=options.verbosity, options=options, record=RecordType())[1]
            tmp_pop = OptimizeAndSimplifyPopulation(dataset, T(1.0), tmp_pop, options, options.maxsize, RecordType())
        end)
    end
    for future in futures
        fetch(future)
    end
    debug((options.verbosity > 0 || options.progress), "Finished!")
end
