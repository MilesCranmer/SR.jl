@testitem "Generic operator tests" tags = [:part2] begin
    using SymbolicRegression
    using SymbolicRegression:
        plus,
        sub,
        mult,
        square,
        cube,
        safe_pow,
        safe_log,
        safe_log2,
        safe_log10,
        safe_sqrt,
        safe_acosh,
        safe_atanh,
        safe_asin,
        safe_acos,
        neg,
        greater,
        cond,
        relu,
        logical_or,
        logical_and,
        gamma

    types_to_test = [Float16, Float32, Float64, BigFloat]
    for T in types_to_test
        val = T(0.5)
        val2 = T(3.2)
        @test abs(safe_log(val) - log(val)) < 1e-6
        @test isnan(safe_log(-val))
        @test abs(safe_log2(val) - log2(val)) < 1e-6
        @test isnan(safe_log2(-val))
        @test abs(safe_log10(val) - log10(val)) < 1e-6
        @test isnan(safe_log10(-val))
        @test abs(safe_log1p(val) - log1p(val)) < 1e-6
        @test abs(safe_acosh(val2) - acosh(val2)) < 1e-6
        @test isnan(safe_acosh(-val2))
        @test abs(safe_asin(val) - asin(val)) < 1e-6
        @test isnan(safe_asin(val2))
        @test abs(safe_acos(val) - acos(val)) < 1e-6
        @test isnan(safe_acos(val2))
        @test abs(safe_atanh(val) - atanh(val)) < 1e-6
        @test isnan(safe_atanh(val2))
        @test neg(-val) == val
        @test safe_sqrt(val) == sqrt(val)
        @test isnan(safe_sqrt(-val))
        @test mult(val, val2) == val * val2
        @test plus(val, val2) == val + val2
        @test sub(val, val2) == val - val2
        @test square(val) == val * val
        @test cube(val) == val * val * val
        @test isnan(safe_pow(T(0.0), -T(1.0)))
        @test isnan(safe_pow(-val, val2))
        @test all(isnan.([safe_pow(-val, -val2), safe_pow(T(0.0), -val2)]))
        @test abs(safe_pow(val, val2) - val^val2) < 1e-6
        @test abs(safe_pow(val, -val2) - val^(-val2)) < 1e-6
        @test !isnan(safe_pow(T(-1.0), T(2.0)))
        @test isnan(safe_pow(T(-1.0), T(2.1)))
        @test isnan(safe_log(zero(T)))
        @test isnan(safe_log2(zero(T)))
        @test isnan(safe_log10(zero(T)))
        @test isnan(safe_log1p(T(-2.0)))
        @test greater(val, val2) == T(0.0)
        @test greater(val2, val) == T(1.0)
        @test relu(-val) == T(0.0)
        @test relu(val) == val
        @test logical_or(val, val2) == T(1.0)
        @test logical_or(T(0.0), val2) == T(1.0)
        @test logical_and(T(0.0), val2) == T(0.0)

        @inferred cond(val, val2)
        @test cond(val, val2) == val2
        @test cond(-val, val2) == zero(T)
    end
end

@testitem "Built-in operators pass validation" tags = [:part3] begin
    using SymbolicRegression
    using SymbolicRegression:
        plus, sub, mult, square, cube, neg, relu, greater, logical_or, logical_and, cond

    types_to_test = [Float16, Float32, Float64, BigFloat]
    options = Options(;
        binary_operators=[plus, sub, mult, /, ^, greater, logical_or, logical_and, cond],
        unary_operators=[
            square, cube, log, log2, log10, log1p, sqrt, atanh, acosh, neg, relu
        ],
    )
    for T in types_to_test
        @test_nowarn SymbolicRegression.assert_operators_well_defined(T, options)
    end
end

@testitem "Built-in operators pass validation for complex numbers" tags = [:part2] begin
    using SymbolicRegression
    using SymbolicRegression: plus, sub, mult, square, cube, neg

    types_to_test = [ComplexF16, ComplexF32, ComplexF64]
    options = Options(;
        binary_operators=[plus, sub, mult, /, ^],
        unary_operators=[square, cube, log, log2, log10, log1p, sqrt, acosh, neg],
    )
    for T in types_to_test
        @test_nowarn SymbolicRegression.assert_operators_well_defined(T, options)
    end
end

@testitem "Incompatibilities are caught" tags = [:part3] begin
    using SymbolicRegression
    using SymbolicRegression: greater

    options = Options(; binary_operators=[greater])
    @test_throws ErrorException SymbolicRegression.assert_operators_well_defined(
        ComplexF64, options
    )
    @test_throws "complex plane" SymbolicRegression.assert_operators_well_defined(
        ComplexF64, options
    )
end

@testitem "Operators with wrong type fail" tags = [:part2] begin
    using SymbolicRegression

    my_bad_op(x) = 1.0f0
    options = Options(; binary_operators=[], unary_operators=[my_bad_op])
    @test_throws ErrorException SymbolicRegression.assert_operators_well_defined(
        Float64, options
    )
    @test_throws "returned an output of type" SymbolicRegression.assert_operators_well_defined(
        Float64, options
    )
    @test_nowarn SymbolicRegression.assert_operators_well_defined(Float32, options)
end

@testitem "Turbo mode matches regular mode" tags = [:part1] begin
    using SymbolicRegression
    using SymbolicRegression:
        plus, sub, mult, square, cube, neg, relu, greater, logical_or, logical_and, cond
    using Random: MersenneTwister
    using Suppressor: @capture_err
    using LoopVectorization: LoopVectorization as _
    include("test_params.jl")

    binary_operators = [plus, sub, mult, /, ^, greater, logical_or, logical_and, cond]
    unary_operators = [square, cube, log, log2, log10, log1p, sqrt, atanh, acosh, neg, relu]
    options = Options(; binary_operators, unary_operators)

    function test_part(tree, Xpart, options)
        y, completed = eval_tree_array(tree, Xpart, options)
        completed || return nothing
        # We capture any warnings about the LoopVectorization not working
        local y_turbo
        eval_warnings = @capture_err begin
            y_turbo, _ = eval_tree_array(tree, Xpart, options; turbo=true)
        end
        test_info(@test(y[1] ≈ y_turbo[1] && eval_warnings == "")) do
            @info T tree X[:, seed] y y_turbo eval_warnings
        end
    end

    for T in (Float32, Float64),
        index_bin in 1:length(binary_operators),
        index_una in 1:length(unary_operators)

        x1, x2 = Node(T; feature=1), Node(T; feature=2)
        tree = Node(index_bin, x1, Node(index_una, x2))
        X = rand(MersenneTwister(0), T, 2, 20)
        for seed in 1:20
            Xpart = X[:, [seed]]
            test_part(tree, Xpart, options)
        end
    end
end

@testitem "Safe operators are compatible with ForwardDiff" tags = [:part2] begin
    using SymbolicRegression
    using SymbolicRegression:
        safe_log,
        safe_log2,
        safe_log10,
        safe_log1p,
        safe_sqrt,
        safe_asin,
        safe_acos,
        safe_atanh,
        safe_acosh,
        safe_pow
    using ForwardDiff

    # Test all safe operators
    safe_operators = [
        (safe_log, 2.0, -1.0),  # (operator, valid_input, invalid_input)
        (safe_log2, 2.0, -1.0),
        (safe_log10, 2.0, -1.0),
        (safe_log1p, 0.5, -2.0),
        (safe_sqrt, 2.0, -1.0),
        (safe_asin, 0.5, 2.0),
        (safe_acos, 0.5, 2.0),
        (safe_atanh, 0.5, 2.0),
        (safe_acosh, 2.0, 0.5),
    ]

    for (op, valid_x, invalid_x) in safe_operators
        # Test derivative exists and is correct for valid input
        deriv = ForwardDiff.derivative(op, valid_x)
        @test !isnan(deriv)
        @test !iszero(deriv)  # All these operators should have non-zero derivatives at test points

        # Test derivative is 0.0 for invalid input
        deriv_invalid = ForwardDiff.derivative(op, invalid_x)
        @test iszero(deriv_invalid)
    end

    # Test safe_pow separately since it's binary
    for x in [0.5, 2.0], y in [2.0, 0.5]
        # Test valid derivatives
        deriv_x = ForwardDiff.derivative(x -> safe_pow(x, y), x)
        deriv_y = ForwardDiff.derivative(y -> safe_pow(x, y), y)
        @test !isnan(deriv_x)
        @test !isnan(deriv_y)
        @test !iszero(deriv_x)  # Should be non-zero for our test points

        # Test invalid cases return 0.0 derivatives
        @test iszero(ForwardDiff.derivative(x -> safe_pow(x, -1.0), 0.0))  # 0^(-1)
        @test iszero(ForwardDiff.derivative(x -> safe_pow(-x, 0.5), 1.0))  # (-x)^0.5
        @test iszero(ForwardDiff.derivative(x -> safe_pow(x, -0.5), 0.0))  # 0^(-0.5)
    end
end
