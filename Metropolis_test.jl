"""
Testing metropolis at finite temperature in the MC updates
Created: 10.03.2026
"""

include("/Users/jamwalk/Desktop/Research/Roderich/Discrete CSL/exp_charge_consv/ecc_func.jl")
using DataFrames
using CSV
using Printf
using Colors

L=2
N=2
beta=2.0

lattice=Lattice(L,L)
bc=construct_bonds(lattice, N) # initialize the bond structure
δB_0s=δB_0_tuple(bc)

rng = MersenneTwister(1234)



plot_bondsnv(bc)
MC_T_flip!(bc, rng, beta)


#MC_T_worm!(bc, rng, δB_0s, beta, 10.0)


#Count up the number of configurations for N=2 on a 2x2 lattice
L=3
N=2
beta=2.0
lattice=Lattice(L,L)
bc=construct_bonds(lattice, N) # initialize the bond structure
rng= MersenneTwister()

# store counts and energies by bond configuration
dict_counts   = Dict{Tuple{Vararg{Int}}, Int}()
dict_energies = Dict{Tuple{Vararg{Int}}, Float64}()

for i in 1:10^8
    MC_T_flip!(bc, rng, beta)

    key = Tuple(bc.bond)

    # increment visit count
    dict_counts[key] = get(dict_counts, key, 0) + 1

    # store energy only the first time this configuration is seen
    get!(dict_energies, key, sum(bc.charges .^ 2))

    # MC_T_worm!(bc, rng, δB_0s, beta, 25.0)
    # key = Tuple(bc.bond)

    # # increment visit count
    # dict_counts[key] = get(dict_counts, key, 0) + 1

    # # store energy only the first time this configuration is seen
    # get!(dict_energies, key, sum(bc.charges .^ 2))

end


println(length(keys(dict_counts))) #Number of unique configurations found
counts=collect(values(dict_counts))
energies=collect(values(dict_energies))

p2=scatter(energies, -log.(counts/maximum(counts))/beta, markersize=4, markerstrokewidth=0, c=:grays, xlabel="Energy", ylabel="-Log(P)/beta (normalized)", legend=false)
plot!(p2,energies, energies, c=:red, label="E=ΔF line")
title!(p2, "Actual Energy vs. Estimated Energy beta=$beta")
display(p2)

#p=histogram(energies, nbins=50, xlabel="Energy Change ΔE", ylabel="Frequency", title="Histogram of Energy Changes at beta=$beta")
#display(p)

