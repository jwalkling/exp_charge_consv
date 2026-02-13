"""
Testing 
Created: 04.02.2026
"""

include("/Users/jamwalk/Desktop/Research/Roderich/Discrete CSL/exp_charge_consv/ecc_func.jl")
using DataFrames
using CSV
using Printf
using Colors


L=3
N=2
lattice = Lattice(L,L)
bc = Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly), zeros(Int, lattice.Lx*lattice.Ly)) #Initialize charges to zero
rng = MersenneTwister(1234)
δB_0s=δB_0_tuple(bc)

iterations=10_000

@btime MC_T0_loop!(bc, rng, δB_0s)

plot_bondsnv(bc)


MC_T_worm!(bc, rng, δB_0s, 1.0, 10^8.0)
MC_T_worm!(bc, rng, δB_0s, 0.1)
plot_bondsnv(bc)

# Range of betas and study charge frequencies
betas = [0.001,0.01, 0.1, 1.0]
iterations = 1_000_000

charge_dicts = [Dict{Int,Int}() for _ in betas]  # one dict per beta

for (k, beta) in pairs(betas)
    #bc = Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly), zeros(Int, lattice.Lx*lattice.Ly))
    d = charge_dicts[k]
    empty!(d)  # optional; ensures clean even if re-running
    p=plot_bondsnv(bc)
    display(p)
    for _ in 1:iterations
        MC_T_worm!(bc, rng, δB_0s, beta, 10^6.0)

        for q in bc.charges
            d[q] = get(d, q, 0) + 1
        end
    end
end


# Single beta and study ergodicity
L=3
N=2
lattice = Lattice(L,L)
bc = Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly), zeros(Int, lattice.Lx*lattice.Ly)) #Initialize charges to zero
rng = MersenneTwister()
δB_0s=δB_0_tuple(bc)

beta=1.0
#Store counts in a dictionary
dict=Dict{Vector, Int64}()

for i in 1:10^5
    #bonds_0=copy(bond_config.bond)
     MC_T_worm!(bc, rng, δB_0s, beta)
    # if bond_config.bond == bonds_0
    #     continue
    # end
    dict[copy(bc.bond)] = get(dict, copy(bc.bond), 0) + 1
end
println(length(keys(dict)))

plot_bondsnv(bc)


# Single beta and study distribution
L=2
N=2
lattice = Lattice(L,L)
bc = Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly), zeros(Int, lattice.Lx*lattice.Ly)) #Initialize charges to zero
rng = MersenneTwister()#MersenneTwister(1234)
δB_0s=δB_0_tuple(bc)

beta=0.01#log(4)/5#log(4)/5
iterations=10^8
#Store counts in a dictionary
dict=Dict{Vector, Int64}()

for i in 1:iterations
    #bonds_0=copy(bond_config.bond)
    MC_T_worm!(bc, rng, δB_0s, beta, exp(beta*75))
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
    bc = Bonds(lattice, N, bond, zeros(Int, lattice.Lx * lattice.Ly))
    energies[i] = sum(vertex_charges(bc).^2)
end


p2=scatter(energies, -log.(counts/maximum(counts))/beta, markersize=4, markerstrokewidth=0, c=:grays, xlabel="Energy", ylabel="Log Frequency (normalized)", legend=false)
plot!(p2,energies, energies, c=:red, label="E=ΔF line")


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

#plot of the calculated energies vs. estimated energy from frequency of appearance
p2=scatter(energies, -log.(counts/maximum(counts))/beta, markersize=4, markerstrokewidth=0, c=:grays, xlabel="Energy", ylabel="Log Frequency (normalized)", legend=false)
plot!(p2,energies, energies, c=:red, label="E=ΔF line")
title!(p2, "Actual Energy vs. Estimated Energy beta=$beta")
display(p2)




plot_bondsnv(bc)

## Sort by frequency and plot top 10
sorted_configs = sort(collect(dict), by=x->x[2], rev=true)
top_configs = sorted_configs[1:10]
for (config, count) in top_configs
    println("Config: ", config, " Count: ", count)
    bc = Bonds(lattice, N, config, zeros(Int, lattice.Lx*lattice.Ly))
    # p=plot_bondsnv(bc)
    # title!(p, "Config with count: $count")
    # display(p)
end

-log(sorted_configs[4][2]/sorted_configs[3][2])/beta

# Study detailed balance between two states
state_A = copy(sorted_configs[4][1])
state_B = [0, 0, 0, 1, -1, 0, 0, 0]

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


#At zero temperature
L=2
N=2
lattice = Lattice(L,L)
bc = Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly), zeros(Int, lattice.Lx*lattice.Ly)) #Initialize charges to zero
rng = MersenneTwister(1234)
δB_0s=δB_0_tuple(bc)

beta=log(2)/5 #log(4)/5
#Store counts in a dictionary
dict=Dict{Vector, Int64}()

for i in 1:10^7
    #bonds_0=copy(bond_config.bond)
     MC_T_worm!(bc, rng, δB_0s, beta)
    # if bond_config.bond == bonds_0
    #     continue
    # end
    dict[copy(bc.bond)] = get(dict, copy(bc.bond), 0) + 1
end
println(length(keys(dict)))



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
        tmp = Bonds(lattice, N, copy(fluxA0), zeros(Int, lattice.Lx * lattice.Ly))
        vertex_charges(tmp)
    else
        nothing
    end

    qB0 = if precompute_charges
        tmp = Bonds(lattice, N, copy(fluxB0), zeros(Int, lattice.Lx * lattice.Ly))
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

L=2
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