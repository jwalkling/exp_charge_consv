"""
Testing 
Created: 04.02.2026
"""

include("/Users/jamwalk/Desktop/Research/Roderich/Discrete CSL/exp_charge_consv/ecc_func.jl")
using DataFrames
using CSV
using Printf
using Colors


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
#--------------------------------
# Varying Ecutoff and beta for 3x3 lattice
#--------------------------------

#Import the locally stored data
#Parameters
betas = [0.01,0.1,1.0]#[0.5, 1.0, 2.0, 0.75, 1.5, 3.0]
Ecutoffs = [[60,50,40],[50,40,30],[20,15,10]]#[20, 10, 5, 20, 10, 5]
directory = "../ECC_data/T>0/Dist_Test/3x3/Diff_Cutoffs2/"

reddata=[0,2,6,8,10,16,20]
i=1
for j in 1:3
    filename  = joinpath(directory, "dist_data$(betas[i])_$(Ecutoffs[i][j]).csv")
    df = CSV.read(filename, DataFrame)
    beta=betas[i]
    energies = df.energy
    counts = df.count

    factor=10^2
    energies,counts = thindata_random_bernoulli(energies, counts,factor)
    p2=scatter(energies, -log.(counts/maximum(counts))/beta, markersize=4, markerstrokewidth=0, c=:grays, xlabel="Energy", ylabel="-log(p)/beta", legend=false)
    xlims!(p2,0,10)
    ylims!(p2,0,10)
    plot!(p2,reddata,reddata, c=:red, label="E=ΔF line")
    title!(p2, "E (red) vs. Estimate [beta=$beta for 3x3 w/ Ecut=$(Ecutoffs[i][j])]")
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

for Echoose in 6:15
    counts1=[]
    for i in 1:length(energies)
        Ei=energies[i]
        if Ei == Echoose
            push!(counts1, counts[i])
        end
    end

    p=histogram(counts1, nbins=30, xlabel="Counts at E=$(Echoose)", ylabel="Frequency", title="Histogram of freq at E=$(Echoose) for beta=1.0 & Ecut=10")
    display(p)
end


filename  = joinpath(directory, "dist_data0.01_60.csv")
df = CSV.read(filename, DataFrame)
beta=0.01
energies = df.energy
counts = df.count
p=plot()
histogram!(p,energies, nbins=200, xlabel="Energy", ylabel="Frequency", title="Histogram of Counts at beta=$beta")
#xlims!(p, 0, 100)
display(p)


#Import the locally stored data
#Parameters
Ecutoffs = [40,20,20,10,10,5]#[20, 10, 5, 20, 10, 5]
betas = [0.5,0.5,1.0,1.0,2.0,2.0]#[0.5, 1.0, 2.0, 0.75, 1.5, 3.0]
directory = "../ECC_data/T>0/Dist_Test/3x3/Diff_Cuttoffs/"

for i in 4:6
    beta=betas[i]
    filename  = joinpath(directory, "Dist_data$(i).csv")
    df = CSV.read(filename, DataFrame)
    energies = df.energy
    counts = df.count
    p2=scatter(energies, -log.(counts/maximum(counts))/beta, markersize=4, markerstrokewidth=0, c=:grays, xlabel="Energy", ylabel="Log Frequency (normalized)", legend=false)
    plot!(p2,energies, energies, c=:red, label="E=ΔF line")
    title!(p2, "Actual Energy vs. Estimated Energy beta=$beta")
    display(p2)
end

i=3
filename  = joinpath(directory, "Dist_data$(i)_$(Ecutoffs[i]).csv")
df = CSV.read(filename, DataFrame)
beta=betas[i]
energies = df.energy
counts = df.count
p2=scatter(energies, -log.(counts/maximum(counts))/beta, markersize=4, markerstrokewidth=0, c=:grays, xlabel="Energy", ylabel="-log(p)/beta", legend=false)
xlims!(p2,0,20)
ylims!(p2,0,20)
plot!(p2,energies, energies, c=:red, label="E=ΔF line")
title!(p2, "E (red) vs. Estimate [beta=$beta for 3x3 w/ Ecut=$(Ecutoffs[i])]")
display(p2)
savefig(p2, joinpath(directory, "Energy_vs_Estimate_beta$(betas[i])_Ecut$(Ecutoffs[i]).png"))

#Histogram of the counts
i=6
filename  = joinpath(directory, "Dist_data$(i).csv")
df = CSV.read(filename, DataFrame)
beta=betas[i]
energies = df.energy
counts = df.count
p=plot(xlims=(0, 100))
histogram!(p,energies, nbins=200, xlabel="Energy", ylabel="Frequency", title="Histogram of Counts at beta=$beta")
#xlims!(p, 0, 100)
display(p)

p=plot()
histogram!(p,log10.(counts), nbins=200, xlabel="Counts", ylabel="Frequency", title="Histogram of Counts at beta=$beta")
#xlims!(p, 0, 100)
display(p)


p=scatter(energies,log10.(counts))
xlims!(p,0,5)


#Import the locally stored data
#Parameters
Ecutoffs = [20, 10, 5, 20, 10, 5]
betas = [0.5, 1.0, 2.0, 0.75, 1.5, 3.0]
directory = "../ECC_data/T>0/Dist_Test/3x3/"

for i in 4:6
    beta=betas[i]
    filename  = joinpath(directory, "Dist_data$(i).csv")
    df = CSV.read(filename, DataFrame)
    energies = df.energy
    counts = df.count
    p2=scatter(energies, -log.(counts/maximum(counts))/beta, markersize=4, markerstrokewidth=0, c=:grays, xlabel="Energy", ylabel="Log Frequency (normalized)", legend=false)
    plot!(p2,energies, energies, c=:red, label="E=ΔF line")
    title!(p2, "Actual Energy vs. Estimated Energy beta=$beta")
    display(p2)
end

p2=scatter(energies, -log.(counts/maximum(counts))/beta, markersize=4, markerstrokewidth=0, c=:grays, xlabel="Energy", ylabel="Log Frequency (normalized)", legend=false)
plot!(p2,energies, energies, c=:red, label="E=ΔF line")
title!(p2, "Actual Energy vs. Estimated Energy beta=$beta")
display(p2)


p = scatter(
    counts,
    marker_z = energies,        # color by energy
    c = :grays,                 # low=black, high=white
    markersize = 4,
    markerstrokewidth = 0,
    xlabel = "Count",
    ylabel = "Frequency",
    colorbar_title = "Energy",
    legend = false,
)

ylims!(p, maximum(counts) * 0.01, maximum(counts) * 1.1)
display(p)


## Sort by frequency and plot top 10
sorted_configs = sort(collect(dict), by=x->x[2], rev=true)
top_configs = sorted_configs[1:10]
for (config, count) in top_configs
    println("Config: ", config, " Count: ", count)
    bc = Bonds(lattice, N, config, zeros(Int, lattice.Lx*lattice.Ly), zeros(Int, lattice.Lx*lattice.Ly))
    # p=plot_bondsnv(bc)
    # title!(p, "Config with count: $count")
    # display(p)
end






# Single beta and study distribution
L=3
N=2
lattice = Lattice(L,L)
bc = construct_bonds(lattice,N) #Initialize charges to zero
rng = MersenneTwister()#MersenneTwister(1234)
δB_0s=δB_0_tuple(bc)

beta=0.01 #0.01#log(4)/5#log(4)/5
iterations=10^6 #10^8
#Store counts in a dictionary
dict=Dict{Vector, Int64}()

for i in 1:iterations
    #bonds_0=copy(bond_config.bond)
    MC_T_worm!(bc, rng, δB_0s, beta, exp(beta*32*N^2))
    # if bond_config.bond == bonds_0
    #     continue
    # end
    dict[copy(bc.bond)] = get(dict, copy(bc.bond), 0) + 1
end
#Find the energies
energies = zeros(Float64, length(keys(dict)))
bonds = collect(keys(dict))               # make a stable indexable list
counts = Float64.(collect(values(dict)))  # z-values for scatter

for (i, bond) in enumerate(bonds)
    bc = construct_bonds(lattice, N)
    energies[i] = sum(vertex_charges(bc).^2)
end




#Testing each part of the code
lattice=Lattice(3,3)
N=2
bond_config=construct_bonds(lattice, N) # initialize the bond structure
rng=MersenneTwister(1234)

in_bounds(2,1,lattice)

bond_label(5,6) #just returns the bond label

multiplier(-1,2)
δB_0_tuple(bond_config)

v=[]
for i in 1:1000
    push!(v, allowed_step_first(1.0, bond_config, 5, rng))
end
histogram(v, nbins=3, xlabel="Allowed Step", ylabel="Frequency", title="Histogram of Allowed Steps")





"""
    test_detailed_balance_uniform!(
        bond_config,
        rng::AbstractRNG;
        burnin::Int = 10_000,
        nsteps::Int = 200_000,
        min_pair_count::Int = 50,
        topk::Int = 20,
        store_states::Bool = true,
        mc_step! = MC_T0_loop!
    )

Detailed-balance diagnostic for a Markov chain whose stationary distribution is *uniform*
over configurations (all visited states are assumed degenerate).

Correct DB condition for uniform π is:
    P(a->b) = P(b->a)
Empirically:
    N_ab / visits[a]  ≈  N_ba / visits[b]

This function:
- keys states by *contents* using Tuple(bond_config.bond)
- records directed transition counts N_ab and visit counts
- reports the worst symmetry violations by relative difference in transition probabilities

Returns:
    trans, visits, diffs, states

where diffs entries are tuples:
    (relprob, total, a, b, nab, nba, pab, pba, flux_rel)
"""
lattice=Lattice(3,3)
N=2 #max_bond value
bond_config=construct_bonds(lattice, N) # initialize the bond structure

function test_detailed_balance_uniform!(
    bond_config,
    rng::AbstractRNG;
    burnin::Int = 10_000,
    nsteps::Int = 200_000,
    min_pair_count::Int = 50,
    topk::Int = 20,
    store_states::Bool = true,
    mc_step! = MC_T0_loop!
)
    δB_0s=δB_0_tuple(bond_config) # Precompute δB_0s for worm updates (not needed for T=0 loop)
    # Content-based immutable key (diagnostic-safe; allocates)
    statekey() = Tuple(bond_config.bond)

    # Directed transition counts: (a,b) -> N_ab
    trans  = Dict{Tuple{Any,Any}, Int}()
    visits = Dict{Any, Int}()

    # Optional snapshots for printing
    states = Dict{Any, Any}()

    # Burn-in
    for _ in 1:burnin
        mc_step!(bond_config, rng, δB_0s)
    end

    # Initialize
    a = statekey()
    visits[a] = get(visits, a, 0) + 1
    if store_states && !haskey(states, a)
        states[a] = copy(bond_config.bond)
    end

    # Main sampling
    for _ in 1:nsteps
        mc_step!(bond_config, rng, δB_0s)
        b = statekey()

        trans[(a, b)] = get(trans, (a, b), 0) + 1
        visits[b] = get(visits, b, 0) + 1

        if store_states && !haskey(states, b)
            states[b] = copy(bond_config.bond)
        end

        a = b
    end

    # Analyze symmetry on *transition probabilities*:
    #   pab = N_ab / visits[a]
    #   pba = N_ba / visits[b]
    #
    # Also report a "flux" symmetry check:
    #   flux_ab = visits[a] * pab = N_ab
    # so flux symmetry reduces to N_ab ≈ N_ba (but it is noisier / less informative).
    diffs = Vector{Tuple{Float64, Int, Any, Any, Int, Int, Float64, Float64, Float64}}()
    # (relprob, total, a, b, nab, nba, pab, pba, flux_rel)

    for ((ka, kb), nab) in trans
        nba = get(trans, (kb, ka), 0)
        total = nab + nba
        total < min_pair_count && continue

        va = get(visits, ka, 0)
        vb = get(visits, kb, 0)
        (va == 0 || vb == 0) && continue

        pab = nab / va
        pba = nba / vb

        denom = pab + pba
        relprob = denom == 0 ? 0.0 : abs(pab - pba) / denom

        # Optional flux symmetry metric (for uniform π, flux is proportional to N_ab)
        flux_denom = nab + nba
        flux_rel = flux_denom == 0 ? 0.0 : abs(nab - nba) / flux_denom

        push!(diffs, (relprob, total, ka, kb, nab, nba, pab, pba, flux_rel))
    end

    sort!(diffs; by = x -> (x[1], x[2]), rev = true)

    println("Unique states visited:          ", length(visits))
    println("Unique directed transitions:    ", length(trans))
    println("Pairs checked (min total = $min_pair_count): ", length(diffs))
    println()
    println("Worst detailed-balance violations (probability symmetry):")
    for i in 1:min(topk, length(diffs))
        relprob, total, ka, kb, nab, nba, pab, pba, flux_rel = diffs[i]
        println(rpad("[$i]", 4),
                " relP=", round(relprob, digits=4),
                "  pab=", round(pab, digits=6),
                "  pba=", round(pba, digits=6),
                "  total=", total,
                "  N_ab=", nab,
                "  N_ba=", nba,
                "  fluxRel=", round(flux_rel, digits=4))

        if store_states
            bond_a = states[ka]
            bond_b = states[kb]
            #println("   a_key=", ka)
            println("   bond_a=", bond_a)
            #println("   b_key=", kb)
            println("   bond_b=", bond_b)
        end
    end

    return trans, visits, diffs, states
end



# Example usage:
rng = MersenneTwister(1236)
trans, visits, diffs, states = test_detailed_balance_uniform!(
    bond_config, rng;
    burnin=10_000,
    nsteps=100_000_000,
    min_pair_count=50,
    topk=20,
    mc_step! = MC_T0_loop!
)



function test_detailed_balance_worm!(
    bond_config,
    rng::AbstractRNG;
    burnin::Int = 10_000,
    nsteps::Int = 200_000,
    min_pair_count::Int = 50,
    topk::Int = 20,
    beta::Float64 = 0.01,
    store_states::Bool = true,
)
    
    normalization_factor = exp(beta * 32 * N^2) # Precompute normalization factor for acceptance probability
    # Content-based immutable key (diagnostic-safe; allocates)
    statekey() = Tuple(bond_config.bond)

    # Directed transition counts: (a,b) -> N_ab
    trans  = Dict{Tuple{Any,Any}, Int}()
    visits = Dict{Any, Int}()

    # Optional snapshots for printing
    states = Dict{Any, Any}()

    # Burn-in
    for _ in 1:burnin
        MC_T_worm!(bc, rng, δB_0s, beta, normalization_factor)
    end

    # Initialize
    a = statekey()
    visits[a] = get(visits, a, 0) + 1
    if store_states && !haskey(states, a)
        states[a] = copy(bond_config.bond)
    end

    # Main sampling
    for _ in 1:nsteps
        MC_T_worm!(bc, rng, δB_0s, beta, normalization_factor)
        b = statekey()

        trans[(a, b)] = get(trans, (a, b), 0) + 1
        visits[b] = get(visits, b, 0) + 1

        if store_states && !haskey(states, b)
            states[b] = copy(bond_config.bond)
        end

        a = b
    end

    # Analyze symmetry on *transition probabilities*:
    #   pab = N_ab / visits[a]
    #   pba = N_ba / visits[b]
    #
    # Also report a "flux" symmetry check:
    #   flux_ab = visits[a] * pab = N_ab
    # so flux symmetry reduces to N_ab ≈ N_ba (but it is noisier / less informative).
    diffs = Vector{Tuple{Float64, Int, Any, Any, Int, Int, Float64, Float64, Float64}}()
    # (relprob, total, a, b, nab, nba, pab, pba, flux_rel)

    for ((ka, kb), nab) in trans
        nba = get(trans, (kb, ka), 0)
        total = nab + nba
        total < min_pair_count && continue

        va = get(visits, ka, 0)
        vb = get(visits, kb, 0)
        (va == 0 || vb == 0) && continue

        pab = nab / va
        pba = nba / vb

        denom = pab + pba
        relprob = denom == 0 ? 0.0 : abs(pab - pba) / denom

        # Optional flux symmetry metric (for uniform π, flux is proportional to N_ab)
        flux_denom = nab + nba
        flux_rel = flux_denom == 0 ? 0.0 : abs(nab - nba) / flux_denom

        push!(diffs, (relprob, total, ka, kb, nab, nba, pab, pba, flux_rel))
    end

    sort!(diffs; by = x -> (x[1], x[2]), rev = true)

    println("Unique states visited:          ", length(visits))
    println("Unique directed transitions:    ", length(trans))
    println("Pairs checked (min total = $min_pair_count): ", length(diffs))
    println()
    println("Worst detailed-balance violations (probability symmetry):")
    for i in 1:min(topk, length(diffs))
        relprob, total, ka, kb, nab, nba, pab, pba, flux_rel = diffs[i]
        println(rpad("[$i]", 4),
                " relP=", round(relprob, digits=4),
                "  pab=", round(pab, digits=6),
                "  pba=", round(pba, digits=6),
                "  total=", total,
                "  N_ab=", nab,
                "  N_ba=", nba,
                "  fluxRel=", round(flux_rel, digits=4))

        if store_states
            bond_a = states[ka]
            bond_b = states[kb]
            #println("   a_key=", ka)
            println("   bond_a=", bond_a)
            #println("   b_key=", kb)
            println("   bond_b=", bond_b)
        end
    end

    return trans, visits, diffs, states
end

function test_detailed_balance_loop!(
    bond_config,
    rng::AbstractRNG;
    burnin::Int = 10_000,
    nsteps::Int = 200_000,
    min_pair_count::Int = 50,
    topk::Int = 20,
    store_states::Bool = true,
    mc_step! = MC_T0_loop!
)
    δB_0s=δB_0_tuple(bond_config)
    # Content-based immutable key (diagnostic-safe; allocates)
    statekey() = Tuple(bond_config.bond)

    # Directed transition counts: (a,b) -> N_ab
    trans  = Dict{Tuple{Any,Any}, Int}()
    visits = Dict{Any, Int}()

    # Optional snapshots for printing
    states = Dict{Any, Any}()

    # Burn-in
    for _ in 1:burnin
        mc_step!(bond_config, rng,δB_0s)
    end

    # Initialize
    a = statekey()
    visits[a] = get(visits, a, 0) + 1
    if store_states && !haskey(states, a)
        states[a] = copy(bond_config.bond)
    end

    # Main sampling
    for _ in 1:nsteps
        mc_step!(bond_config, rng,δB_0s)
        b = statekey()

        trans[(a, b)] = get(trans, (a, b), 0) + 1
        visits[b] = get(visits, b, 0) + 1

        if store_states && !haskey(states, b)
            states[b] = copy(bond_config.bond)
        end

        a = b
    end

    # Analyze symmetry on *transition probabilities*:
    #   pab = N_ab / visits[a]
    #   pba = N_ba / visits[b]
    #
    # Also report a "flux" symmetry check:
    #   flux_ab = visits[a] * pab = N_ab
    # so flux symmetry reduces to N_ab ≈ N_ba (but it is noisier / less informative).
    diffs = Vector{Tuple{Float64, Int, Any, Any, Int, Int, Float64, Float64, Float64}}()
    # (relprob, total, a, b, nab, nba, pab, pba, flux_rel)

    for ((ka, kb), nab) in trans
        nba = get(trans, (kb, ka), 0)
        total = nab + nba
        total < min_pair_count && continue

        va = get(visits, ka, 0)
        vb = get(visits, kb, 0)
        (va == 0 || vb == 0) && continue

        pab = nab / va
        pba = nba / vb

        denom = pab + pba
        relprob = denom == 0 ? 0.0 : abs(pab - pba) / denom

        # Optional flux symmetry metric (for uniform π, flux is proportional to N_ab)
        flux_denom = nab + nba
        flux_rel = flux_denom == 0 ? 0.0 : abs(nab - nba) / flux_denom

        push!(diffs, (relprob, total, ka, kb, nab, nba, pab, pba, flux_rel))
    end

    sort!(diffs; by = x -> (x[1], x[2]), rev = true)

    println("Unique states visited:          ", length(visits))
    println("Unique directed transitions:    ", length(trans))
    println("Pairs checked (min total = $min_pair_count): ", length(diffs))
    println()
    println("Worst detailed-balance violations (probability symmetry):")
    for i in 1:min(topk, length(diffs))
        relprob, total, ka, kb, nab, nba, pab, pba, flux_rel = diffs[i]
        println(rpad("[$i]", 4),
                " relP=", round(relprob, digits=4),
                "  pab=", round(pab, digits=6),
                "  pba=", round(pba, digits=6),
                "  total=", total,
                "  N_ab=", nab,
                "  N_ba=", nba,
                "  fluxRel=", round(flux_rel, digits=4))

        if store_states
            bond_a = states[ka]
            bond_b = states[kb]
            #println("   a_key=", ka)
            println("   bond_a=", bond_a)
            #println("   b_key=", kb)
            println("   bond_b=", bond_b)
        end
    end

    return trans, visits, diffs, states
end


L = 3
N = 2
lattice = Lattice(L, L)
bc = construct_bonds(lattice, N)
rng = MersenneTwister(1234)
δB_0s=δB_0_tuple(bc)

# Example usage:
rng = MersenneTwister(1234)
trans, visits, diffs, states = test_detailed_balance_loop!(
    bc, rng;
    burnin=10_000,
    nsteps=10_000_000,
    min_pair_count=50,
    topk=20,
    mc_step! = MC_T0_loop!
)

L = 2
N = 2
lattice = Lattice(L, L)
bc = construct_bonds(lattice, N)
rng = MersenneTwister(1234)

beta = 0.1

trans, visits, diffs, states = test_detailed_balance_worm!(
    bc,
    rng;
    beta = beta,
    burnin = 1_00,
    nsteps = 20_000_000,
    min_pair_count = 50,
    topk = 20,
)

visits

#-------------------------
# Testing explicit transition probabilities
#-------------------------

# Pair that violates detailed balance
#--------------------------
state_A=[1, -1, 1, 1, 0, 2, -1, -1, -2, 2, 0, 0, -2, 0, 0, 0, 0, 0] #[0, 0, 1, -1, 0, 2, 0, 0, -1, -1, 0, 2, 0, 0, -2, 0, 0, 0]
state_B=[0, 0, -1, 1, 0, -2, -1, 1, 2, -2, 0, 0, 2, 0, 0, 0, 0, 0]#[0, 0, -1, 1, 0, -2, -1, 1, 1, -1, 0, -2, 2, 0, 2, 0, 0, 0] #[0, 0, -1, 1, 0, -2, -1, 1, 1, -1, 0, -2, 2, 0, 2, 0, 0, 0]
# state_A=[0, 0, 1, -1, 0, 2, 1, -1, -1, 1, 0, 2, -2, 0, -2, 0, 0, 0]
# state_B=[0, 0, 0, 0, 0, 0, -1, 1, -1, -1, 0, -2, 2, 0, 2, 0, 0, 0]
# Plot them
bond_config_A=Bonds(lattice, N, copy(state_A), zeros(Int, lattice.Lx*lattice.Ly), zeros(Int, lattice.Lx*lattice.Ly))
p=plot_bondsnv(bond_config_A)
display(p)
bond_config_B=Bonds(lattice, N, copy(state_B), zeros(Int, lattice.Lx*lattice.Ly), zeros(Int, lattice.Lx*lattice.Ly))
p=plot_bondsnv(bond_config_B)
display(p)








# Study detailed balance between two states
state_A = #copy(sorted_configs[4][1])
state_B = #[0, 0, 0, 1, -1, 0, 0, 0]

beta = 0.5
iterations = 10^6
index_0 = 2
rng = MersenneTwister(1234)

A_to_B, B_to_A = count_AB_transitions(
    lattice, N, state_A, state_B, δB_0s, beta, iterations, index_0;
    rng=rng, worm! =MC_T_worm_TEST!, precompute_charges=true
)

println("Transitions from A to B: ", A_to_B)
println("Transitions from B to A: ", B_to_A)



function MC_T_worm_TEST!(bond_config::Bonds, rng::AbstractRNG, δB_0s::Tuple{Vararg{Float64}}, beta::Float64, index_0::Int)
    charges = bond_config.charges

    δB_0    = rand(rng, δB_0s)
    δB_prev = δB_0

    step_0 = allowed_step_first(δB_0, bond_config, index_0, rng)
    step_prev=step_0
    if step_0 == 0
        return
    end

    # apply first move
    index_prev = index_0
    index_curr = index_0 + step_0

    

    bond0 = step_bond(index_prev, step_0)
    bond_config.bond[bond0] += δB_prev

    # Calculate charges
    q0 = charges[index_0] + charge_factor(-step_0)*δB_0 # charge left on the initial site
    #println("q0: ", q0, " at index ", index_0, "\n")
    #println("Initial move: step ", step_0, " from index ", index_0, " to index ", index_curr, " with δB ", δB_0, "\n")
    #println("qinitial: ", charges[index_0], " qfinal: ", charges[index_curr], "\n")
    while index_curr != index_0
        #Stop with a probability given by delta_j
        #Energy of the vertex to be left
        q1 = charges[index_curr] + charge_factor(step_prev)*δB_prev
        ΔE = q1*q1 + q0*q0 #Energy is square of charges
        delta = exp(-beta * ΔE) #stopping probability
        if rand(rng) < delta
            charges[index_0]    = q0
            #println("bond_config.charges", bond_config.charges)
            charges[index_curr] = q1
            #println("bond_config.charges", bond_config.charges)
            return
        end

        #If no stop, simply sample the next step uniformly
        step, bond, δB_curr = allowed_step(δB_prev, bond_config, rng, index_curr, index_prev)

        index_prev = index_curr
        index_curr = index_curr + step

        bond_config.bond[bond] += δB_curr
        δB_prev = δB_curr
        step_prev = step
    end
end

"""
    count_AB_transitions(
        lattice, N,
        fluxA::AbstractVector{<:Integer},
        fluxB::AbstractVector{<:Integer},
        δB_0s::Tuple{Vararg{Float64}},
        beta::Real,
        iterations::Integer,
        index_0::Integer;
        rng::AbstractRNG = MersenneTwister(),
        worm! = MC_T_worm_TEST!,
        precompute_charges::Bool = true,
        return_dicts::Bool = false,
    )

Run `iterations` independent worm updates, each time restarting exactly from the same
frozen configuration A (resp. B), and count the number of times the output configuration
equals the other one.

Uses tuple keys `Tuple(bond_config.bond)` so Dict keys are immutable and hashed by content.

Returns `(A_to_B, B_to_A)` by default; if `return_dicts=true`, returns
`(A_to_B, B_to_A, transitions_A, transitions_B)`.

Required worm signature:
`worm!(bond_config::Bonds, rng::AbstractRNG, δB_0s::Tuple{Vararg{Float64}}, beta::Float64, index_0::Int)`
"""
function count_AB_transitions(
    lattice, N,
    fluxA::AbstractVector{<:Integer},
    fluxB::AbstractVector{<:Integer},
    δB_0s::Tuple{Vararg{Float64}},
    beta::Real,
    iterations::Integer,
    index_0::Integer;
    rng::AbstractRNG = MersenneTwister(),
    worm! = MC_T_worm_TEST!,
    precompute_charges::Bool = true,
    return_dicts::Bool = false,
)
    # ---- sanity checks ----
    iterations ≥ 1 || throw(ArgumentError("iterations must be ≥ 1"))
    index_0 ≥ 1    || throw(ArgumentError("index_0 must be ≥ 1"))

    # ---- freeze inputs (avoid accidental mutation) ----
    fluxA0 = collect(Int, fluxA)
    fluxB0 = collect(Int, fluxB)

    # ---- (optional) precompute start charges once ----
    qA0 = if precompute_charges
        tmp = Bonds(lattice, N, copy(fluxA0), zeros(Int, lattice.Lx * lattice.Ly), zeros(Int, lattice.Lx * lattice.Ly))
        vertex_charges(tmp)
    else
        nothing
    end

    qB0 = if precompute_charges
        tmp = Bonds(lattice, N, copy(fluxB0), zeros(Int, lattice.Lx * lattice.Ly), zeros(Int, lattice.Lx * lattice.Ly))
        vertex_charges(tmp)
    else
        nothing
    end

    # ---- Dicts: keys are immutable tuples of bonds ----
    transitions_A = Dict{Tuple{Vararg{Int}}, Int64}()
    transitions_B = Dict{Tuple{Vararg{Int}}, Int64}()

    # ---- sweep from A ----
    for _ in 1:iterations
        bond_config = if precompute_charges
            Bonds(lattice, N, copy(fluxA0), qA0)
        else
            tmp = Bonds(lattice, N, copy(fluxA0), zeros(Int, lattice.Lx * lattice.Ly))
            Bonds(lattice, N, copy(fluxA0), vertex_charges(tmp))
        end

        worm!(bond_config, rng, δB_0s, Float64(beta), Int(index_0))
        key = Tuple(bond_config.bond)
        transitions_A[key] = get(transitions_A, key, 0) + 1
    end

    # ---- sweep from B ----
    for _ in 1:iterations
        bond_config = if precompute_charges
            Bonds(lattice, N, copy(fluxB0), qB0)
        else
            tmp = Bonds(lattice, N, copy(fluxB0), zeros(Int, lattice.Lx * lattice.Ly))
            Bonds(lattice, N, copy(fluxB0), vertex_charges(tmp))
        end

        worm!(bond_config, rng, δB_0s, Float64(beta), Int(index_0))
        key = Tuple(bond_config.bond)
        transitions_B[key] = get(transitions_B, key, 0) + 1
    end

    key_A = Tuple(fluxA0)
    key_B = Tuple(fluxB0)

    A_to_B = get(transitions_A, key_B, 0)
    B_to_A = get(transitions_B, key_A, 0)

    return return_dicts ? (A_to_B, B_to_A, transitions_A, transitions_B) : (A_to_B, B_to_A)
end




function MC_T_typicalE!(bond_config::Bonds, rng::AbstractRNG, δB_0s::Tuple{Vararg{Float64}}, beta::Float64, Norm::Float64)
    lat     = bond_config.lattice
    charges = bond_config.charges
    Lx      = lat.Lx
    Ly      = lat.Ly
    Nsites  = Lx * Ly

    index_0 = rand(rng, 1:Nsites)

    δB_0    = rand(rng, δB_0s)
    δB_prev = δB_0

    step_0 = allowed_step_first(δB_0, bond_config, index_0, rng)
    step_prev=step_0
    if step_0 == 0
        return (false, 0.0)
    end

    # apply first move
    index_prev = index_0
    index_curr = index_0 + step_0

    

    bond0 = step_bond(index_prev, step_0)
    bond_config.bond[bond0] += δB_prev

    # Calculate charges
    #Emin_sq = -32 * bond_config.max_bond^2
    qi1 = charges[index_0] + charge_factor(-step_0)*δB_0
    while index_curr != index_0
        #Stop with a probability given by delta_j
        #Energy of the vertex to be left
        qe0=charges[index_curr]
        qe1 = qe0 + charge_factor(step_prev)*δB_prev
        ΔE = qe1*qe1 - qe0*qe0  #Energy is square of charges
        delta = exp(-beta * (ΔE))/Norm#-Emin_sq)) #stopping probability #Norm*
        if rand(rng) < delta
            charges[index_0]    = qi1
            #println("bond_config.charges", bond_config.charges)
            charges[index_curr] = qe1
            #println("bond_config.charges", bond_config.charges)
            # println("triggered")
            # println("ΔE: ", ΔE)
            return (true, ΔE)
        end

        #If no stop, simply sample the next step uniformly
        step, bond, δB_curr = allowed_step(δB_prev, bond_config, rng, index_curr, index_prev)

        index_prev = index_curr
        index_curr = index_curr + step

        bond_config.bond[bond] += δB_curr
        δB_prev = δB_curr
        step_prev = step
    end
    return (true, 0.0)
end

L=4
N=2
lattice = Lattice(L,L)

rng = MersenneTwister()#MersenneTwister(1234)
δB_0s=δB_0_tuple(bc)

betas=[0.00001,0.0001,0.001,0.01]
for beta in betas
    bc = Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly), zeros(Int, lattice.Lx*lattice.Ly)) #Initialize charges to zero
    #beta=0.01#log(4)/5#log(4)/5
    iterations=10^6
    #Store counts in a dictionary
    dict=Dict{Vector, Int64}()

    energies = []

    for i in 1:iterations
        #bonds_0=copy(bond_config.bond)
        vals= MC_T_typicalE!(bc, rng, δB_0s, beta, exp(beta*32*N^2))
        if vals[1] == false
            continue
        end
        push!(energies, vals[2])
        # if bond_config.bond == bonds_0
        #     continue
        # end
        dict[copy(bc.bond)] = get(dict, copy(bc.bond), 0) + 1
    end
    #Find the energies

    p=histogram(energies, nbins=50, xlabel="Energy Change ΔE", ylabel="Frequency", title="Histogram of Energy Changes at beta=$beta")
    display(p)
end