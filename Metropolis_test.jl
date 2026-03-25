"""
Testing metropolis at finite temperature in the MC updates
Created: 10.03.2026
"""

include("/Users/jamwalk/Desktop/Research/Roderich/Discrete CSL/exp_charge_consv/ecc_func.jl")
using DataFrames
using CSV
using Printf
using Colors

betas = [0.01]#[1.0]#[0.5, 1.0, 2.0, 0.75, 1.5, 3.0]
ratios = [1,10,100,1000] 
directory = "../ECC_data/T>0/Metro_Test/3x3/maxcutoff,beta=$(betas[1])/"


function thindata_random_bernoulli(data1, data2, factor; rng=Random.default_rng())
    N = length(data1)
    N == length(data2) || throw(DimensionMismatch("data1 and data2 must have same length"))

    p = 1 / factor

    out1 = Vector{eltype(data1)}()
    out2 = Vector{eltype(data2)}()
    sizehint!(out1, max(1, N ÷ factor))
    sizehint!(out2, max(1, N ÷ factor))

    @inbounds for i in 1:N
        if rand(rng) < p
            push!(out1, data1[i])
            push!(out2, data2[i])
        end
    end
    return out1, out2
end


#Data for just plotting the line Eestimate=E
reddata=[0,2,6,8,10,16,20]

i=1
for j in 4:4
    filename  = joinpath(directory, "metro_data$(betas[i])_$(ratios[j]).csv")
    df = CSV.read(filename, DataFrame)
    beta=betas[i]
    energies = df.energy
    counts = df.count
    factor=10^2
    energies,counts = thindata_random_bernoulli(energies, counts,factor)

    p2=scatter(energies, -log.(counts/maximum(counts))/beta, markersize=4, markerstrokewidth=0, c=:grays, xlabel="Energy", ylabel="-log(p)/beta", legend=false)
    xlims!(p2,0,100)
    ylims!(p2,0,100)
    plot!(p2,reddata,reddata, c=:red, label="E=ΔF line")
    title!(p2, "E (red) vs. Estimate [beta=$beta for 3x3 w/ ratio=$(ratios[j])]")
    display(p2)
end

i=1
j=2
filename  = joinpath(directory, "metro_data$(betas[i])_$(ratios[j]).csv")
df = CSV.read(filename, DataFrame)
beta=betas[i]
energies = df.energy
counts = df.count
total1=0
num=0
for (i,E) in enumerate(energies)
    if E == 2
        total1+=counts[i]
        num+=1
    end
end
println(total1/num)

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


filename  = joinpath(directory, "metro_data1.0_1.csv")
df = CSV.read(filename, DataFrame)
beta=1.0
energies = df.energy
counts = df.count
p2=scatter(energies, -log.(counts/maximum(counts))/beta, markersize=4, markerstrokewidth=0, c=:grays, xlabel="Energy", ylabel="-log(p)/beta", legend=false)
plot!(p2,reddata,reddata, c=:red, label="E=ΔF line")
xlims!(p2,0,20)
ylims!(p2,0,20)


#Study the distribution of states at a given energy
i=1
j=3


filename  = joinpath(directory, "metro_data$(betas[i])_$(ratios[j]).csv")
df = CSV.read(filename, DataFrame)
beta=betas[i]
energies = df.energy
counts = df.count

E=10
counts_E = []

for k in 1:length(energies)
    if energies[k] == E
        push!(counts_E, counts[k])
    end
end 
histogram(counts_E, nbins=50, xlabel="Count", 
ylabel="Frequency", title="Histogram of counts for energy E=$(E) at beta=$(beta)")


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

