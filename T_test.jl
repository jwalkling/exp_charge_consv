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

MC_T0_loop!(bc, rng, δB_0s)

plot_bondsnv(bc)


MC_T_worm!(bc, rng, δB_0s, 1.0)

plot_bondsnv(bc)

# Range of betas and study charge frequencies
betas = [0.1, 0.5, 1.0, 2.0, 5.0]
iterations = 100_000

charge_dicts = [Dict{Int,Int}() for _ in betas]  # one dict per beta

for (k, beta) in pairs(betas)
    d = charge_dicts[k]
    empty!(d)  # optional; ensures clean even if re-running

    for _ in 1:iterations
        MC_T_worm!(bc, rng, δB_0s, beta)

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
rng = MersenneTwister(1234)
δB_0s=δB_0_tuple(bc)

beta=0.0001
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

beta=0.5#log(4)/5#log(4)/5
#Store counts in a dictionary
dict=Dict{Vector, Int64}()

for i in 1:10^6
    #bonds_0=copy(bond_config.bond)
    MC_T_worm!(bc, rng, δB_0s, beta)
    # if bond_config.bond == bonds_0
    #     continue
    # end
    dict[copy(bc.bond)] = get(dict, copy(bc.bond), 0) + 1
end
println(length(keys(dict)))
p=plot(collect(values(dict)))#, seriestype=:histogram, bins=50, title="Distribution of bond configurations for beta=$(beta)", xlabel="Count", ylabel="Frequency")
ylims!(p, 0,maximum(collect(values(dict)))*1.1) #log10(maximum(collect(values(dict)))*1.1))
display(p)
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
fluxp0 = copy(sorted_configs[4][1])  # frozen start bonds for A
flux00 = [0, 0, 0, 1, -1, 0, 0, 0]#copy(sorted_configs[6][1])  # frozen start bonds for B

# Precompute start charges once (optional but faster)
state_A0 = Bonds(lattice, N, copy(fluxp0), zeros(Int, lattice.Lx * lattice.Ly))
state_B0 = Bonds(lattice, N, copy(flux00), zeros(Int, lattice.Lx * lattice.Ly))
qA0 = vertex_charges(state_A0)
qB0 = vertex_charges(state_B0)
state_A0 = Bonds(lattice, N, copy(fluxp0), qA0)
state_B0 = Bonds(lattice, N, copy(flux00), qB0)

plot_bondsnv(state_A0)
plot_bondsnv(state_B0)

beta = 0.5
iterations = 10^7
rng=MersenneTwister()

# Better Dict keys: Tuple of bond values (immutable, hashable by content)
transitions_A = Dict{Tuple{Vararg{Int}}, Int64}()
for i in 1:iterations
    bond_config = Bonds(lattice, N, copy(fluxp0), qA0)   # <- exact same start every time
    MC_T_worm!(bond_config, rng, δB_0s, beta)
    key = Tuple(bond_config.bond)
    transitions_A[key] = get(transitions_A, key, 0) + 1
end

transitions_B = Dict{Tuple{Vararg{Int}}, Int64}()
for i in 1:iterations
    bond_config = Bonds(lattice, N, copy(flux00), qB0)   # <- exact same start every time
    MC_T_worm!(bond_config, rng, δB_0s, beta)
    key = Tuple(bond_config.bond)
    transitions_B[key] = get(transitions_B, key, 0) + 1
end

key_B = Tuple(flux00)   # B configuration
key_A = Tuple(fluxp0)   # A configuration

println("Transitions from A to B: ", get(transitions_A, key_B, 0))
println("Transitions from B to A: ", get(transitions_B, key_A, 0))


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