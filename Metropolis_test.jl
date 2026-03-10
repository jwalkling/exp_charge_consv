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


MC_T_worm!(bc, rng, δB_0s, beta, 10.0)
plot_bondsnv(bc)
MC_T_flip!(bc, rng, beta)
