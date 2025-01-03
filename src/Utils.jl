"""Useful functions to be used throughout the library."""
module UtilsModule

using Printf: @printf
using MacroTools: splitdef
using StyledStrings: StyledStrings

macro ignore(args...) end

const pseudo_time = Ref(0)

function get_birth_order(; deterministic::Bool=false)::Int
    """deterministic gives a birth time with perfect resolution, but is not thread safe."""
    if deterministic
        global pseudo_time
        pseudo_time[] += 1
        return pseudo_time[]
    else
        resolution = 1e7
        return round(Int, resolution * time())
    end
end

function is_anonymous_function(op)
    op_string = string(nameof(op))
    return length(op_string) > 1 &&
           op_string[1] == '#' &&
           op_string[2] in ('1', '2', '3', '4', '5', '6', '7', '8', '9')
end
precompile(Tuple{typeof(is_anonymous_function),Function})

recursive_merge(x::AbstractVector...) = cat(x...; dims=1)
recursive_merge(x::AbstractDict...) = merge(recursive_merge, x...)
recursive_merge(x...) = x[end]
recursive_merge() = error("Unexpected input.")

get_base_type(::Type{Complex{BT}}) where {BT} = BT

const subscripts = ('₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉')
function subscriptify(number::Integer)
    return join([subscripts[i + 1] for i in reverse(digits(number))])
end

"""
    split_string(s::AbstractString, n::Integer)

```jldoctest
julia> split_string("abcdefgh", 3)
["abc", "def", "gh"]
```
"""
function split_string(s::AbstractString, n::Integer)
    length(s) <= n && return [s]
    # Due to unicode characters, need to split only at valid indices:
    I = eachindex(s) |> collect
    return [s[I[i]:I[min(i + n - 1, end)]] for i in 1:n:length(s)]
end

"""
Tiny equivalent to StaticArrays.MVector

This is so we don't have to load StaticArrays, which takes a long time.
"""
mutable struct MutableTuple{S,T,N} <: AbstractVector{T}
    data::N

    MutableTuple(::Val{_S}, ::Type{_T}, data::_N) where {_S,_T,_N} = new{_S,_T,_N}(data)
end
@inline Base.eltype(::MutableTuple{S,T}) where {S,T} = T
Base.@propagate_inbounds function Base.getindex(v::MutableTuple, i::Integer)
    T = eltype(v)
    # Trick from MArray.jl
    return GC.@preserve v unsafe_load(
        Base.unsafe_convert(Ptr{T}, pointer_from_objref(v)), i
    )
end
Base.@propagate_inbounds function Base.setindex!(v::MutableTuple, x, i::Integer)
    T = eltype(v)
    GC.@preserve v unsafe_store!(Base.unsafe_convert(Ptr{T}, pointer_from_objref(v)), x, i)
    return x
end
@inline Base.lastindex(::MutableTuple{S}) where {S} = S
@inline Base.firstindex(v::MutableTuple) = 1
Base.dataids(v::MutableTuple) = (UInt(pointer(v)),)
function _to_vec(v::MutableTuple{S,T}) where {S,T}
    x = Vector{T}(undef, S)
    @inbounds for i in 1:S
        x[i] = v[i]
    end
    return x
end

"""Return the bottom k elements of x, and their indices."""
bottomk_fast(x::AbstractVector{T}, k) where {T} = Base.Cartesian.@nif(
    32,
    d -> d == k,
    d -> _bottomk_dispatch(x, Val(d))::Tuple{Vector{T},Vector{Int}},
    _ -> _bottomk_dispatch(x, Val(k))::Tuple{Vector{T},Vector{Int}}
)

function _bottomk_dispatch(x::AbstractVector{T}, ::Val{k}) where {T,k}
    if k == 1
        return (p -> [p]).(findmin_fast(x))
    end
    indmin = MutableTuple(Val(k), Int, ntuple(_ -> 1, Val(k)))
    minval = MutableTuple(Val(k), T, ntuple(_ -> typemax(T), Val(k)))
    _bottomk!(x, minval, indmin)
    return _to_vec(minval), _to_vec(indmin)
end
function _bottomk!(x, minval, indmin)
    @inbounds for i in eachindex(x)
        new_min = x[i] < minval[end]
        if new_min
            minval[end] = x[i]
            indmin[end] = i
            for ki in lastindex(minval):-1:(firstindex(minval) + 1)
                need_swap = minval[ki] < minval[ki - 1]
                if need_swap
                    minval[ki], minval[ki - 1] = minval[ki - 1], minval[ki]
                    indmin[ki], indmin[ki - 1] = indmin[ki - 1], indmin[ki]
                end
            end
        end
    end
    return nothing
end

# Thanks Chris Elrod
# https://discourse.julialang.org/t/why-is-minimum-so-much-faster-than-argmin/66814/9
function findmin_fast(x::AbstractVector{T}) where {T}
    indmin = 1
    minval = typemax(T)
    @inbounds @simd for i in eachindex(x)
        newmin = x[i] < minval
        minval = newmin ? x[i] : minval
        indmin = newmin ? i : indmin
    end
    return minval, indmin
end

function argmin_fast(x::AbstractVector{T}) where {T}
    return findmin_fast(x)[2]
end

function poisson_sample(λ::T) where {T}
    k, p, L = 0, one(T), exp(-λ)
    while p > L
        k += 1
        p *= rand(T)
    end
    return k - 1
end

macro threads_if(flag, ex)
    return quote
        if $flag
            Threads.@threads $ex
        else
            $ex
        end
    end |> esc
end

"""
    @save_kwargs variable function ... end

Save the kwargs and their default values to a variable as a constant.
This is to be used to create these same kwargs in other locations.
"""
macro save_kwargs(log_variable, fdef)
    return esc(_save_kwargs(log_variable, fdef))
end
function _save_kwargs(log_variable::Symbol, fdef::Expr)
    def = splitdef(fdef)
    # Get kwargs:
    kwargs = copy(def[:kwargs])
    kwargs = map(kwargs) do k
        # If it's a macrocall for @nospecialize
        if k.head == :macrocall && string(k.args[1]) == "@nospecialize"
            # Find the actual argument - it's the last non-LineNumberNode argument
            inner_arg = last(filter(arg -> !(arg isa LineNumberNode), k.args))
            return inner_arg
        end
        return k
    end
    kwargs = filter(kwargs) do k
        # Filter ...:
        k.head == :... && return false
        # Filter other deprecated kwargs:
        startswith(string(first(k.args)), "deprecated") && return false
        return true
    end
    return quote
        $(Base).@__doc__ $fdef
        const $log_variable = $kwargs
    end
end

json3_write(args...) = error("Please load the JSON3.jl package.")

"""
    PerThreadCache{T}

A cache that is efficient for multithreaded code, and works
by having a separate cache for each thread. This allows
us to avoid repeated locking. We only need to lock the cache
when resizing to the number of threads.
"""
struct PerThreadCache{T}
    x::Vector{T}
    num_threads::Ref{Int}
    lock::Threads.SpinLock

    PerThreadCache{T}() where {T} = new(Vector{T}(undef, 1), Ref(1), Threads.SpinLock())
end

function _get_thread_cache(cache::PerThreadCache{T}) where {T}
    if cache.num_threads[] < Threads.nthreads()
        Base.@lock cache.lock begin
            # The reason we have this extra `.num_threads[]` parameter is to avoid
            # a race condition between a thread resizing the array concurrent
            # to the check above. Basically we want to make sure the array is
            # always big enough by the time we get to using it. Since `.num_threads[]`
            # is set last, we can safely use the array.
            if cache.num_threads[] < Threads.nthreads()
                resize!(cache.x, Threads.nthreads())
                cache.num_threads[] = Threads.nthreads()
            end
        end
    end
    threadid = Threads.threadid()
    if !isassigned(cache.x, threadid)
        cache.x[threadid] = eltype(cache.x)()
    end
    return cache.x[threadid]
end
function Base.get!(f::F, cache::PerThreadCache, key) where {F<:Function}
    thread_cache = _get_thread_cache(cache)
    return get!(f, thread_cache, key)
end

# https://discourse.julialang.org/t/performance-of-hasmethod-vs-try-catch-on-methoderror/99827/14
# Faster way to catch method errors:
@enum IsGood::Int8 begin
    Good
    Bad
    Undefined
end
const SafeFunctions = PerThreadCache{Dict{Type,IsGood}}()

function safe_call(f::F, x::T, default::D) where {F,T<:Tuple,D}
    thread_cache = _get_thread_cache(SafeFunctions)
    status = get(thread_cache, Tuple{F,T}, Undefined)
    status == Good && return (f(x...)::D, true)
    status == Bad && return (default, false)

    output = try
        (f(x...)::D, true)
    catch e
        !isa(e, MethodError) && rethrow(e)
        (default, false)
    end
    if output[2]
        thread_cache[Tuple{F,T}] = Good
    else
        thread_cache[Tuple{F,T}] = Bad
    end
    return output
end

@static if VERSION >= v"1.11.0-"
    @eval begin
        const AnnotatedIOBuffer = Base.AnnotatedIOBuffer
        const AnnotatedString = Base.AnnotatedString
    end
else
    @eval begin
        const AnnotatedIOBuffer = StyledStrings.AnnotatedStrings.AnnotatedIOBuffer
        const AnnotatedString = StyledStrings.AnnotatedStrings.AnnotatedString
    end
end

dump_buffer(buffer::IOBuffer) = String(take!(buffer))
function dump_buffer(buffer::AnnotatedIOBuffer)
    return AnnotatedString(dump_buffer(buffer.io), buffer.annotations)
end

struct FixKws{F,KWS}
    f::F
    kws::KWS
end
function FixKws(f::F; kws...) where {F}
    return FixKws{F,typeof(kws)}(f, kws)
end
function (f::FixKws{F,KWS})(args::Vararg{Any,N}) where {F,KWS,N}
    return f.f(args...; f.kws...)
end

end
