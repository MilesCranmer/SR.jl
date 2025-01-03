@testitem "Generic interface tests" tags = [:part3] begin
    using SymbolicRegression
    using MLJTestInterface: MLJTestInterface as MTI
    include("test_params.jl")

    failures, summary = MTI.test(
        [SRRegressor], MTI.make_regression()...; mod=@__MODULE__, verbosity=0, throw=true
    )
    @test isempty(failures)

    X = randn(100, 3)
    Y = @. cos(X^2) * 3.2 - 0.5
    (X, Y) = MTI.table.((X, Y))
    w = ones(100)
    failures, summary = MTI.test(
        [MultitargetSRRegressor], X, Y, w; mod=@__MODULE__, verbosity=0, throw=true
    )
    @test isempty(failures)
end

@testitem "Variable names - single outputs" tags = [:part3] begin
    using SymbolicRegression
    using SymbolicRegression: Node
    using MLJBase
    using SymbolicUtils
    using Random: MersenneTwister

    include("test_params.jl")

    stop_kws = (; early_stop_condition=(loss, complexity) -> loss < 1e-5)

    rng = MersenneTwister(0)
    X = (a=rand(rng, 32), b=rand(rng, 32))
    y = X.a .^ 2.1
    # We also make sure the deprecated npop and npopulations still work:
    model = SRRegressor(; niterations=10, npop=1000, npopulations=15, stop_kws...)
    mach = machine(model, X, y)
    fit!(mach)
    rep = report(mach)
    @test occursin("a", rep.equation_strings[rep.best_idx])
    ypred_good = predict(mach, X)
    @test sum(abs2, predict(mach, X) .- y) / length(y) < 1e-4

    # Check that we can choose the equation
    ypred_same = predict(mach, (data=X, idx=rep.best_idx))
    @test ypred_good == ypred_same

    ypred_bad = predict(mach, (data=X, idx=1))
    @test ypred_good != ypred_bad

    # Smoke test SymbolicUtils
    eqn = node_to_symbolic(rep.equations[rep.best_idx], model)
    n = symbolic_to_node(eqn, model)
    eqn2 = convert(SymbolicUtils.Symbolic, n, model)
    n2 = convert(Node, eqn2, model)
end

@testitem "Variable names - multiple outputs" tags = [:part1] begin
    using SymbolicRegression
    using MLJBase
    using Random: MersenneTwister

    include("test_params.jl")

    stop_kws = (; early_stop_condition=(loss, complexity) -> loss < 1e-5)

    rng = MersenneTwister(0)
    X = (a=rand(rng, 32), b=rand(rng, 32))
    y = X.a .^ 2.1
    model = MultitargetSRRegressor(; niterations=10, stop_kws...)
    mach = machine(model, X, reduce(hcat, [reshape(y, :, 1) for i in 1:3]))
    fit!(mach)
    rep = report(mach)
    @test all(
        eq -> occursin("a", eq), [rep.equation_strings[i][rep.best_idx[i]] for i in 1:3]
    )
    ypred_good = predict(mach, X)

    # Test that we can choose the equation
    ypred_same = predict(mach, (data=X, idx=rep.best_idx))
    @test ypred_good == ypred_same

    ypred_bad = predict(mach, (data=X, idx=[1, 1, 1]))
    @test ypred_good != ypred_bad

    ypred_mixed = predict(mach, (data=X, idx=[rep.best_idx[1], 1, rep.best_idx[3]]))
    @test ypred_mixed == hcat(ypred_good[:, 1], ypred_bad[:, 2], ypred_good[:, 3])

    @test_throws AssertionError predict(mach, (data=X,))
    @test_throws "If specifying an equation index during" predict(mach, (data=X,))
    @test_throws "If specifying an equation index during" predict(mach, (X=X, idx=1))
end

@testitem "Variable names - named outputs" tags = [:part1] begin
    using SymbolicRegression
    using MLJBase
    using Random: MersenneTwister

    include("test_params.jl")

    stop_kws = (; early_stop_condition=(loss, complexity) -> loss < 1e-5)

    rng = MersenneTwister(0)
    X = (b1=randn(rng, 32), b2=randn(rng, 32))
    Y = (c1=X.b1 .* X.b2, c2=X.b1 .+ X.b2)
    w = ones(32)
    model = MultitargetSRRegressor(; niterations=10, stop_kws...)
    mach = machine(model, X, Y, w)
    fit!(mach)
    test_outs = predict(mach, X)
    @test isempty(setdiff((:c1, :c2), keys(test_outs)))
    @test_throws AssertionError predict(mach, (a1=randn(32), b2=randn(32)))
    @test_throws "Variable names do not match fitted" predict(
        mach, (b1=randn(32), a2=randn(32))
    )
end

@testitem "Good predictions" tags = [:part1] begin
    using SymbolicRegression
    using MLJBase
    using Random: MersenneTwister

    include("test_params.jl")

    stop_kws = (; early_stop_condition=(loss, complexity) -> loss < 1e-5)

    rng = MersenneTwister(0)
    X = randn(rng, 100, 3)
    Y = X

    # Create a temporary directory
    temp_dir = mktempdir()

    # Set the run_id and output_directory
    run_id = "test_run"
    output_directory = temp_dir

    # Instantiate the model with the specified run_id and output_directory
    model = MultitargetSRRegressor(;
        niterations=10, run_id=run_id, output_directory=output_directory, stop_kws...
    )

    mach = machine(model, X, Y)
    fit!(mach)

    # Check predictions
    @test sum(abs2, predict(mach, X) .- Y) / length(X) < 1e-5

    # Load the output CSV file
    for i in 1:3
        csv_file = joinpath(output_directory, run_id, "hall_of_fame_output$(i).csv")
        csv_content = read(csv_file, String)

        # Parse the CSV content using regex
        lines = split(csv_content, '\n')
        header = split(lines[1], ',')
        data_lines = lines[2:end]

        @test header[1] == "Complexity"
        @test header[2] == "Loss"
        @test header[3] == "Equation"

        complexities = Int[]
        losses = Float64[]
        equations = String[]

        for line in data_lines
            if isempty(line)
                continue
            end
            cols = split(line, ',')
            push!(complexities, parse(Int, cols[1]))
            push!(losses, parse(Float64, cols[2]))
            @show cols
            push!(equations, cols[3])
        end

        @test !isempty(complexities)
        @test complexities == report(mach).complexities[i]
        @test losses == report(mach).losses[i]
        for (eq, eq_str) in zip(equations, report(mach).equation_strings[i])
            @test eq[(begin + 1):(end - 1)] == eq_str
        end
    end
end

@testitem "Helpful errors" tags = [:part3] begin
    using SymbolicRegression
    using MLJBase
    using Random: MersenneTwister

    include("test_params.jl")

    model = MultitargetSRRegressor()
    rng = MersenneTwister(0)
    mach = machine(model, randn(rng, 32, 3), randn(rng, 32); scitype_check_level=0)
    @test_throws AssertionError @quiet(fit!(mach))
    @test_throws "For single-output regression, please" @quiet(fit!(mach))

    model = SRRegressor()
    rng = MersenneTwister(0)
    mach = machine(model, randn(rng, 32, 3), randn(rng, 32, 2); scitype_check_level=0)
    @test_throws AssertionError @quiet(fit!(mach))
    @test_throws "For multi-output regression, please" @quiet(fit!(mach))

    model = SRRegressor(; verbosity=0)
    rng = MersenneTwister(0)
    mach = machine(model, randn(rng, 32, 3), randn(rng, 32))
    @test_throws ErrorException @quiet(fit!(mach; verbosity=0))
end

@testitem "Unfinished search" tags = [:part3] begin
    using SymbolicRegression
    using MLJBase
    using Suppressor
    using Random: MersenneTwister

    model = SRRegressor(; timeout_in_seconds=1e-10)
    rng = MersenneTwister(0)
    mach = machine(model, randn(rng, 32, 3), randn(rng, 32))
    fit!(mach)
    # Ensure that the hall of fame is empty:
    _, hof = mach.fitresult.state
    hof.exists .= false
    # Recompute the report:
    mach.report[:fit] = SymbolicRegression.MLJInterfaceModule.full_report(
        model, mach.fitresult
    )
    @test report(mach).best_idx == 0
    @test predict(mach, randn(32, 3)) == zeros(32)
    msg = @capture_err begin
        predict(mach, randn(32, 3))
    end
    @test occursin("Evaluation failed either due to", msg)

    model = MultitargetSRRegressor(; timeout_in_seconds=1e-10)
    rng = MersenneTwister(0)
    mach = machine(model, randn(rng, 32, 3), randn(rng, 32, 3))
    fit!(mach)
    # Ensure that the hall of fame is empty:
    _, hofs = mach.fitresult.state
    foreach(hofs) do hof
        hof.exists .= false
    end
    mach.report[:fit] = SymbolicRegression.MLJInterfaceModule.full_report(
        model, mach.fitresult
    )
    @test report(mach).best_idx == [0, 0, 0]
    @test predict(mach, randn(32, 3)) == zeros(32, 3)
    msg = @capture_err begin
        predict(mach, randn(32, 3))
    end
    @test occursin("Evaluation failed either due to", msg)
end
