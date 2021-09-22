import Printf: @printf
using Distributed
using FromFile
@from "Core.jl" import SRThreaded, SRSerial, SRDistributed

function debug(verbosity, string...)
    if verbosity > 0
        println(string...)
    end
end

function debug_inline(verbosity, string...)
    if verbosity > 0
        print(string...)
    end
end

function getTime()::Int
    return round(Int, 1e3*(time()-1.6e9))
end


function check_numeric(n)
    return tryparse(Float64, n) !== nothing
end

function is_anonymous_function(op)
	op_string = string(nameof(op))
	return length(op_string) > 1 && op_string[1] == '#' && check_numeric(op_string[2:2])
end

function recursive_merge(x::AbstractVector...)
    cat(x...; dims=1)
end

function recursive_merge(x::AbstractDict...)
    merge(recursive_merge, x...)
end

function recursive_merge(x...)
    x[end]
end

isgood(x::T) where {T<:Number} = !(isnan(x) || !isfinite(x))
isgood(x) = true
isbad(x) = !isgood(x)

macro return_on_false(flag, retval)
    :(if !$(esc(flag))
          return ($(esc(retval)), false)
    end)
end

macro return_on_false2(flag, retval, retval2)
    :(if !$(esc(flag))
          return ($(esc(retval)), $(esc(retval2)), false)
    end)
end

function next_worker(worker_assignment::Dict{Tuple{Int,Int}, Int}, procs::Vector{Int})::Int
    job_counts = Dict(proc=>0 for proc in procs)
    for (key, value) in worker_assignment
        @assert haskey(job_counts, value)
        job_counts[value] += 1
    end
    least_busy_worker = reduce(
        (proc1, proc2) -> (
            job_counts[proc1] <= job_counts[proc2] ? proc1 : proc2
        ),
        procs
    )
    return least_busy_worker
end

function next_worker(worker_assignment::Dict{Tuple{Int,Int}, Int}, procs::Nothing)::Int
    return 0
end

macro sr_spawner(parallel, p, expr)
    quote
        if $(esc(parallel)) == SRSerial
            $(esc(expr))
        elseif $(esc(parallel)) == SRDistributed
            @spawnat($(esc(p)), $(esc(expr)))
        else
            Threads.@spawn($(esc(expr)))
        end
    end
end
