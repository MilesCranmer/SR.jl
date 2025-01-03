if !(VERSION >= v"1.10.0" && VERSION < v"1.12.0-DEV.0")
    exit(0)
end

dir = mktempdir()

@info "Starting test_jet.jl" dir

using Pkg
@info "Creating environment..."
Pkg.activate(dir; io=devnull)
Pkg.develop(; path=dirname(@__DIR__), io=devnull)
Pkg.add(["JET", "Preferences", "DynamicExpressions"]; io=devnull)
@info "Done!"

using Preferences
cd(dir)
Preferences.set_preferences!(
    "SymbolicRegression", "instability_check" => "disable"; force=true
)
Preferences.set_preferences!(
    "DynamicExpressions", "instability_check" => "disable"; force=true
)

using SymbolicRegression
using JET

@info "Running tests..."
JET.test_package(SymbolicRegression; target_defined_modules=true)
@info "Done!"

@info "test_jet.jl finished"
