module CoreModule

include("ProgramConstants.jl")
include("Dataset.jl")
include("OptionsStruct.jl")
include("Equation.jl")
include("Operators.jl")
include("Options.jl")

import .ProgramConstantsModule:
    CONST_TYPE,
    MAX_DEGREE,
    BATCH_DIM,
    FEATURE_DIM,
    RecordType,
    SRConcurrency,
    SRSerial,
    SRThreaded,
    SRDistributed
import .DatasetModule: Dataset
import .OptionsStructModule: Options
import .EquationModule: Node, copy_node, string_tree, print_tree
import .OptionsModule: Options
import .OperatorsModule:
    plus,
    sub,
    mult,
    square,
    cube,
    pow,
    safe_pow,
    div,
    safe_log,
    safe_log2,
    safe_log10,
    safe_log1p,
    safe_sqrt,
    safe_acosh,
    neg,
    greater,
    greater,
    relu,
    logical_or,
    logical_and,
    gamma,
    erf,
    erfc,
    atanh_clip

end