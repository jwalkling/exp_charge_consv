"""
Testing the distribution for the metropolis step mixed in with loops
Created: 12.03.2026
"""

using Plots
using Random
using LinearAlgebra
using DataFrames
using CSV
using Printf
using Colors



#-------------------------
# Define Lattice Structure
#-------------------------

#Struct that stores the lattice dimensions
struct Lattice
    Lx::Int
    Ly::Int
end

"""
Define a struct to store the config of the lattice along with properties.
lattice::Lattice -> Lattice struct defined above that stores dimensions
max_bond::Int -> bond can take integer values -max_bond, ..., max_bond
bond::Vector{Int} -> A vector of all the different values on the bonds
"""
struct Bonds
    lattice::Lattice #Lattice struct for dimensions
    max_bond::Int #Maximum allowed bond value (symmetric range [-max_bond, max_bond])
    bond::Vector{Int} #Vector to store the values of each bond
    charges::Vector{Int} #Vector to store the charges at each vertex
    allowed_bonds::Vector{Int} #Vector to store the allowed bonds (for OBC)
end

function construct_bonds(lattice::Lattice, max_bond::Int)
    Lx = lattice.Lx
    Ly = lattice.Ly
    Nsites = Lx * Ly
    bond_vector = zeros(Int, 2 * Nsites) # 2 bonds per site (x and y)
    charge_vector = zeros(Int, Nsites)     # 1 charge per site
    allowed_bonds = allowed_bonds_OBC(lattice) # Precompute allowed bonds for OBC
    return Bonds(lattice, max_bond, bond_vector, charge_vector, allowed_bonds)
end

"""
in_bounds: open boundary conditions.
No global bound check; only the geometric edge checks.
"""
@inline function in_bounds(index_curr::Int, step::Int, lat::Lattice)::Bool
    Lx = lat.Lx
    Ly = lat.Ly

    # --- Extract x,y from flattened coordinate (1-indexed) ---
    x = ((index_curr - 1) % Lx) + 1
    y = ((index_curr - 1) ÷ Lx) + 1

    # Horizontal moves
    if step == 1
        return x != Lx
    elseif step == -1
        return x != 1
    end

    # Vertical moves
    if step == Lx
        return y != Ly
    elseif step == -Lx
        return y != 1
    end

    # If you ever pass any other step, treat as invalid
    return false
end

function allowed_bonds_OBC(lat::Lattice)
    Lx=lat.Lx
    Ly=lat.Ly

    bonds_allowed = Int[]
    sizehint!(bonds_allowed, 2Lx*Ly - Lx - Ly)  # preallocate for all bonds except those dangling off the right and top edges

    for s in 1:(Lx*Ly)
        x = ((s - 1) % Lx) + 1
        y = ((s - 1) ÷ Lx) + 1

        hb = 2s - 1
        vb = 2s

        if x < Lx
            push!(bonds_allowed, hb)
        end
        if y < Ly
            push!(bonds_allowed, vb)
        end
    end
    return bonds_allowed
end

# Function to convert index to coordinate of centre of link
@inline function index_to_coord(lat::Lattice, i::Int)
    h = isodd(i)
    v = ((i + (h ? 1 : 0)) ÷ 2) - 1
    vx = v % lat.Lx
    vy = v ÷ lat.Lx
    (vx + (h ? 0.5 : 0.0), vy + (h ? 0.0 : 0.5))
end


@inline function bond_distance(lat::Lattice, i::Int, j::Int)
    x1, y1 = index_to_coord(lat, i)
    x2, y2 = index_to_coord(lat, j)
    sqrt((x2 - x1)^2 + (y2 - y1)^2)
end

function vertex_charges(bonds::Bonds)
    Lx = bonds.lattice.Lx
    Ly = bonds.lattice.Ly
    vals = bonds.bond
    Nsites = Lx * Ly

    q = zeros(Float64, Nsites)

    for i in 1:Nsites
        x = ((i-1) % Lx) + 1
        y = ((i-1) ÷ Lx) + 1

        # +x bond (right)
        σ_px = (x < Lx) ? vals[2i - 1] : 0.0

        # +y bond (up)
        σ_py = (y < Ly) ? vals[2i] : 0.0

        # -x bond is the right bond of the site to the left
        if x > 1
            i_left = i - 1
            σ_mx = vals[2i_left - 1]
        else
            σ_mx = 0.0
        end

        # -y bond is the up bond of the site below
        if y > 1
            i_down = i - Lx
            σ_my = vals[2i_down]
        else
            σ_my = 0.0
        end

        q[i] = σ_px + σ_py - 2σ_mx - 2σ_my
    end

    return q
end


#Return the index of a vertex given x and y coordinates
@inline idx(lat::Lattice, x::Int, y::Int) = (y-1)*lat.Lx + x

"""
step_bond: label for bond between index_curr and index_curr+step.
Assumes step is a nearest-neighbour step (±1 or ±Lx); caller enforces.
"""
@inline function step_bond(index_curr::Int, step::Int)::Int
    # i = min(index_curr, index_curr + step)
    i = index_curr + ((step < 0) ? step : 0)

    # x-step iff step == ±1
    isx = (step == 1) | (step == -1)

    # 2i-1 for x-step, else 2i
    return 2*i - (isx ? 1 : 0)
end


"""
step_bond: label for bond between index_curr and index_next.
Assumes step is a nearest-neighbour step (±1 or ±Lx); caller enforces.
"""
@inline function bond_label(index_curr::Int, index_next::Int)::Int
    Δ = index_next - index_curr              # ±1 or ±Lx (for NN moves)

    # i = min(index_curr, index_next) without calling min
    i = ifelse(Δ < 0, index_next, index_curr)

    # x-step iff Δ == ±1
    isx = (Δ == 1) | (Δ == -1)

    # 2i-1 for x-step, else 2i
    return 2*i - (isx ? 1 : 0)
end

# Returns value for charge neutrality (fast, branch-light)
@inline function multiplier(Δ_curr::Int, Δ_prev::Int)::Float64
    sameabs = (Δ_prev == Δ_curr) | (Δ_prev == -Δ_curr)
    oppsign = (Δ_curr > 0) != (Δ_prev > 0)
    partner = (!sameabs) & oppsign
    return ifelse(partner, -1.0, ifelse(Δ_curr > 0, 2.0, 0.5))
end

#Generates the list of allowed δB_0 values based on max_bond
@inline function δB_0_tuple(bond_config::Bonds)
    N = Int(bond_config.max_bond)
    N < 1 && return ()

    m = 8*sizeof(Int) - leading_zeros(N) - 1   # floor(log2(N))
    L = m + 2                                  # exponents n = 0..(m+1)

    return (ntuple(i -> -Float64(1 << (i-1)), L)...,
            ntuple(i ->  Float64(1 << (i-1)), L)...)
end

"""
allowed_step_first -> returns integer step or 0 if no step accepted this call
"""
@inline function allowed_step_first(
    δB_firstmove::Float64,
    bond_config::Bonds,
    index_curr::Int,
    rng::AbstractRNG,
)
    lat   = bond_config.lattice
    Lx    = lat.Lx
    bonds = bond_config.bond

    # no allocation
    steps = (-1, 1, -Lx, Lx)
    step  = steps[rand(rng, 1:4)]

    if !in_bounds(index_curr, step, lat)
        return 0
    end

    bond = step_bond(index_curr, step)

    if abs(bonds[bond] + δB_firstmove) <= bond_config.max_bond && abs(δB_firstmove) >= 1
        return step
    end

    return 0
end

"""
allowed_step: returns (step, bond_label, δB_curr)
Backtracking is allowed.
"""
@inline function allowed_step(
    δB::Float64,
    bond_config::Bonds,
    rng::AbstractRNG,
    index_curr::Int,
    index_prev::Int,
)
    lat    = bond_config.lattice
    Lx     = lat.Lx
    Δ_prev = index_curr - index_prev
    bonds  = bond_config.bond

    # no allocation
    steps = (-1, 1, -Lx, Lx)

    # random starting offset gives unbiased random order
    k0 = rand(rng, 0:3)

    @inbounds for t = 0:3
        step = steps[mod1(k0 + t + 1, 4)]

        # backtracking allowed
        if step == -Δ_prev
            bond = step_bond(index_curr, step)
            return step, bond, -δB
        end

        if !in_bounds(index_curr, step, lat)
            continue
        end

        δB_curr = δB * multiplier(step, Δ_prev)
        bond    = step_bond(index_curr, step)

        if abs(bonds[bond] + δB_curr) <= bond_config.max_bond && abs(δB_curr) >= 1
            return step, bond, δB_curr
        end
    end

    error("No allowed steps for (curr=$index_curr, prev=$index_prev)")
end

function MC_T0_loop!(bond_config::Bonds, rng::AbstractRNG, δB_0s::Tuple{Vararg{Float64}})
    lat    = bond_config.lattice
    Lx     = lat.Lx
    Ly     = lat.Ly
    Nsites = Lx * Ly

    index_0 = rand(rng, 1:Nsites)

    δB_0    = rand(rng, δB_0s)
    δB_prev = δB_0

    move_0 = allowed_step_first(δB_0, bond_config, index_0, rng)
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

@inline charge_factor(Δ::Int) = ifelse(Δ > 0, -2, 1)


function MC_T_worm!(bond_config::Bonds, rng::AbstractRNG, δB_0s::Tuple{Vararg{Float64}}, beta::Float64, Norm::Float64)
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
        return
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

function MC_T_flip!(bond_config::Bonds, rng::AbstractRNG, beta::Float64)
    lat     = bond_config.lattice
    charges = bond_config.charges
    allowed_bonds = bond_config.allowed_bonds
    Lx      = lat.Lx
    Ly      = lat.Ly
    N       = bond_config.max_bond

    #Pick a random bond to flip. Use allowed_bonds since OBC
    bond_0 = rand(rng, allowed_bonds)
    

    #±1 sufficient for detailed balance for single spin flips.
    δB_0 = rand(rng, Bool) ? 1 : -1


    bond_config.bond[bond_0] += δB_0

    #Reject if the bond value goes out of bounds
    if abs(bond_config.bond[bond_0]) > N
        bond_config.bond[bond_0] -= δB_0 #undo change
        return
    end

    #TK CAN BE CACHED LATER FOR SPEED
    if isodd(bond_0)
        #Horizontal bond: affects charges of the two vertices it connects
        i1 = (bond_0 + 1) ÷ 2
        i2 = i1 + 1
    else # for vertical (even bonds)
        #Vertical bond: affects charges of the two vertices it connects
        i1 = bond_0 ÷ 2
        i2 = i1 + lat.Lx
    end

    #Calculate new charges and energies
    q1i=charges[i1]
    q2i=charges[i2]

    q1_new = q1i+ δB_0
    q2_new = q2i -2*δB_0

    if exp(-beta*(q1_new*q1_new + q2_new*q2_new - q1i*q1i - q2i*q2i)) < rand(rng)
        #Reject move, undo change
        bond_config.bond[bond_0] -= δB_0
    else
        #Accept move, update charges
        charges[i1] = q1_new
        charges[i2] = q2_new
    end
end



#-----------------------------
# Running the code
#-----------------------------
t0=time()

L=3
N=2
lattice=Lattice(L,L)
bc=construct_bonds(lattice, N) # initialize the bond structure
rng = MersenneTwister()
δB_0s=δB_0_tuple(bc)

iterations=10^5