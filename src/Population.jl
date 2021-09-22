using FromFile
@from "Core.jl" import Options, Dataset, RecordType
@from "EquationUtils.jl" import stringTree, countNodes
@from "LossFunctions.jl" import scoreFunc
@from "MutationFunctions.jl" import genRandomTree
@from "PopMember.jl" import PopMember
# A list of members of the population, with easy constructors,
#  which allow for random generation of new populations
mutable struct Population{T<:Real}
    members::Array{PopMember{T}, 1}
    n::Int
end
"""
    Population(pop::Array{PopMember{T}, 1})

Create population from list of PopMembers.
"""
Population(pop::Array{PopMember{T}, 1}) where {T<:Real} = Population{T}(pop, size(pop, 1))

"""
    Population(dataset::Dataset{T}, baseline::T;
               npop::Int, nlength::Int=3, options::Options,
               nfeatures::Int)

Create random population from number of population, need to initialize score.
"""

Population(npop::Int,
           options::Options,
           nfeatures::Int, nlength::Int=3,) where {T<:Real} = Population([PopMember(genRandomTree(nlength, options, nfeatures), 0.0) for i=1:npop], npop)



"""
    Population(dataset::Dataset{T}, baseline::T;
               npop::Int, nlength::Int=3, options::Options,
               nfeatures::Int)

Create random population and with baseline score.
"""

Population(baseline::T;
           npop::Int, nlength::Int=3,
           options::Options,
           nfeatures::Int) where {T<:Real} = Population([PopMember(genRandomTree(nlength, options, nfeatures), baseline) for i=1:npop], npop)


"""
    Population(dataset::Dataset{T}, baseline::T;
               npop::Int, nlength::Int=3, options::Options,
               nfeatures::Int)

Create random population and score them on the dataset.
"""
Population(dataset::Dataset{T}, baseline::T;
           npop::Int, nlength::Int=3,
           options::Options,
           nfeatures::Int) where {T<:Real} = Population([PopMember(dataset, baseline, genRandomTree(nlength, options, nfeatures), options) for i=1:npop], npop)
"""
    Population(X::AbstractMatrix{T}, y::AbstractVector{T},
               baseline::T; npop::Int, nlength::Int=3,
               options::Options, nfeatures::Int)

Create random population and score them on the dataset.
"""
Population(X::AbstractMatrix{T}, y::AbstractVector{T}, baseline::T;
           npop::Int, nlength::Int=3,
           options::Options,
           nfeatures::Int) where {T<:Real} = Population(Dataset(X, y), baseline, npop=npop, options=options, nfeatures=nfeatures)

# Sample 10 random members of the population, and make a new one
function samplePop(pop::Population, options::Options)::Population
    idx = rand(1:pop.n, options.ns)
    return Population(pop.members[idx])
end

# Sample the population, and get the best member from that sample
function bestOfSample(pop::Population, options::Options)::PopMember
    sample = samplePop(pop, options)
    if options.probPickFirst == 1.0
        best_idx = argmin([sample.members[member].score for member=1:options.ns])
        return sample.members[best_idx]
    else
        sort_idx = sortperm([sample.members[member].score for member=1:options.ns])
        # Lowest comes first
        k = range(0.0, stop=options.ns-1, step=1.0) |> collect
        p = options.probPickFirst

        # Weighted choice:
        prob_each = p * (1 - p) .^ k
        prob_each /= sum(prob_each)
        cumprob = cumsum(prob_each)
        chosen_idx = findfirst(cumprob .> rand(Float32))

        return sample.members[chosen_idx]
    end
end

function finalizeScores(dataset::Dataset{T},
                        baseline::T, pop::Population,
                        options::Options)::Population where {T<:Real}
    need_recalculate = options.batching
    if need_recalculate
        @inbounds @simd for member=1:pop.n
            pop.members[member].score = scoreFunc(dataset, baseline,
                                                  pop.members[member].tree,
                                                  options)
        end
    end
    return pop
end

# Return best 10 examples
function bestSubPop(pop::Population; topn::Int=10)::Population
    best_idx = sortperm([pop.members[member].score for member=1:pop.n])
    return Population(pop.members[best_idx[1:topn]])
end


function record_population(pop::Population{T}, options::Options)::RecordType where {T<:Real}
    RecordType("population"=>[RecordType("tree"=>stringTree(member.tree, options),
                                         "loss"=>member.score,
                                         "complexity"=>countNodes(member.tree),
                                         "birth"=>member.birth,
                                         "ref"=>member.ref,
                                         "parent"=>member.parent)
                             for member in pop.members],
               "time"=>time()
    )
end
