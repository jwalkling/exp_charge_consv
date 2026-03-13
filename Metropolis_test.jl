"""
Testing metropolis at finite temperature in the MC updates
Created: 10.03.2026
"""

include("/Users/jamwalk/Desktop/Research/Roderich/Discrete CSL/exp_charge_consv/ecc_func.jl")
using DataFrames
using CSV
using Printf
using Colors

betas = [1.0]#[0.5, 1.0, 2.0, 0.75, 1.5, 3.0]
ratios = [1,10,100,1000] 
directory = "../ECC_data/T>0/Metro_Test/3x3/Maxcutoffbeta1.0/"

#Data for just plotting the line Eestimate=E
reddata=[0,2,6,8,10,16,20]

i=1
for j in 3:4
    filename  = joinpath(directory, "metro_data$(betas[i])_$(ratios[j]).csv")
    df = CSV.read(filename, DataFrame)
    beta=betas[i]
    energies = df.energy
    counts = df.count

    #factor=10^2
    #energies,counts = thindata_random_bernoulli(energies, counts,factor)
    p2=scatter(energies, -log.(counts/maximum(counts))/beta, markersize=4, markerstrokewidth=0, c=:grays, xlabel="Energy", ylabel="-log(p)/beta", legend=false)
    xlims!(p2,0,20)
    ylims!(p2,0,20)
    plot!(p2,reddata,reddata, c=:red, label="E=ΔF line")
    title!(p2, "E (red) vs. Estimate [beta=$beta for 3x3 w/ Ratio=$(ratios[j])]")
    display(p2)
end


filename  = joinpath(directory, "dist_data1.0_10.csv")
df = CSV.read(filename, DataFrame)
beta=1.0
energies = df.energy
counts = df.count
p2=scatter(energies, -log.(counts/maximum(counts))/beta, markersize=4, markerstrokewidth=0, c=:grays, xlabel="Energy", ylabel="-log(p)/beta", legend=false)
xlims!(p2,0,10)
ylims!(p2,0,10)







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

