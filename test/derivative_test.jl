using Test
using SymbolicRegression
using SymbolicRegression: evalDiffTreeArray, evalGradTreeArray
using Zygote
using LinearAlgebra

input_data = [1.0 2.0 3.0]
input_matrix = transpose(input_data)

custom_cos(x) = cos(x)^2

options = Options(;
    binary_operators=(+, *, -, /, ^),
    unary_operators=(custom_cos, exp),
)
equation(x1, x2, x3) =  (abs(abs(3.0 * custom_cos(x1)) ^ 2.3) ^ (-1.2)) + x2 - x3
true_grad = gradient(equation, input_data[1, :]...)

tree = Node(5, (Node(3.0) * Node(1, Node("x1"))) ^ 2.3, -1.2) + Node("x2") - Node("x3")

for i=1:3
    @test isapprox(evalDiffTreeArray(tree, input_matrix, options, i)[2][1], true_grad[i], rtol=1e-5)
end

for i=1:3
    @test isapprox(evalGradTreeArray(tree, input_matrix, options; variable=true)[2][i, 1], true_grad[i], rtol=1e-5)
end


options = Options(;
    binary_operators=(+, *, -, /, ^),
    unary_operators=(cos, exp),
)
equation(x1, x2, x3) =  (abs(abs(3.0 * cos(x1)) ^ 2.3) ^ (-1.2)) + x2 - x3
true_grad = gradient(equation, input_data[1, :]...)

tree = Node(5, (Node(3.0) * Node(1, Node("x1"))) ^ 2.3, -1.2) + Node("x2") - Node("x3")

for i=1:3
    @test isapprox(evalDiffTreeArray(tree, input_matrix, options, i)[2][1], true_grad[i], rtol=1e-5)
end

for i=1:3
    @test isapprox(evalGradTreeArray(tree, input_matrix, options; variable=true)[2][i, 1], true_grad[i], rtol=1e-5)
end
