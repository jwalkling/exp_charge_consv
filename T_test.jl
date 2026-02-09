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

for i in 1:iterations
    MC_T0_loop!(bc, rng, δB_0s)
    for b in 1:(2*lattice.Lx*lattice.Ly)
        average_bond[b] += bc.bond[b]
    end
end

average_bond ./= iterations
plot(average_bond) #