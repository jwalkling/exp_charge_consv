"""
Testing detailed balance in the MC updates
Created: 04.02.2026
"""


#NB: This code is not fully fleshed out.
#The check would be successful if I let it run through since
#probability of exiting through any link is equal - we reject the whole vertex if the link we chose failed.


include("/Users/jamwalk/Desktop/Research/Roderich/Discrete CSL/exp_charge_consv/ecc_func.jl")
using DataFrames
using CSV
using Printf
using Colors


function MC_T0_loop_fixed!(bond_config::Bonds, rng::AbstractRNG,  δB_0::Float64, index_0::Int)
    δB_prev = δB_0

    move_0 = allowed_step_first(δB_0, bond_config, index_0, rng)
    println("move_0: ", move_0)
    if move_0 == 0
        return
    end

    # apply first move
    index_prev = index_0
    index_curr = index_0 + move_0

    bond0 = step_bond(index_prev, move_0)
    bond_config.bond[bond0] += δB_prev

    while index_curr != index_0
        step, bond, δB_curr = allowed_step(δB_prev, bond_config, rng, index_curr, index_prev)

        index_prev = index_curr
        index_curr = index_curr + step

        bond_config.bond[bond] += δB_curr
        δB_prev = δB_curr
    end
end

#-----------------------------
# Test probability of making single move
#-----------------------------

L=8
N=4
lattice = Lattice(L,L)
bc = Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly))
rng = MersenneTwister(1232) #Random.default_rng() #MersenneTWister(1235)
δB_0s=δB_0_tuple(bc)

iterations=1_000_000
for i in 1:iterations
    MC_T0_loop!(bc, rng, δB_0s)
end

plot_bondsnv(bc)

rng=Random.default_rng()
fix_config= deepcopy(bc)
MC_T0_loop_fixed!(fix_config, rng, δB_0s[3], 54)
plot_bondsnv(fix_config)









rng=Random.default_rng()
fix_config= deepcopy(bc)
MC_T0_loop_fixed!(fix_config, rng, δB_0s[5], 11)
plot_bondsnv(fix_config)
