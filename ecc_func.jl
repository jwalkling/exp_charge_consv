"""
Library of functions for Exponential Charge Conservation
Created: 04.12.2025
"""

using Plots
using Random
using BenchmarkTools
using LinearAlgebra

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

# Function to convert index to coordinate of centre of link
@inline function index_to_coord(lat::Lattice, index::Int)
    Ly = lat.Ly
    q, r = divrem(index - 1, Ly)
    offset = 0.5 * (r & 0x1)
    x = q + 1 + offset
    y = 1.0 + 0.5 * r
    return (x, y)
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


function MC_T_worm!(bond_config::Bonds, rng::AbstractRNG, δB_0s::Tuple{Vararg{Float64}}, beta::Float64)
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
    Emin_sq = -32 * bond_config.max_bond^2
    qi1 = charges[index_0] + charge_factor(-step_0)*δB_0
    while index_curr != index_0
        #Stop with a probability given by delta_j
        #Energy of the vertex to be left
        qe0=charges[index_curr]
        qe1 = qe0 + charge_factor(step_prev)*δB_prev
        ΔE = qe1*qe1 - qe0*qe0  #Energy is square of charges
        delta = exp(-beta * (ΔE-Emin_sq)) #stopping probability
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


#-------------------------
# Measurements
#-------------------------


# --- core: correlation of a 2D link-field A at displacement (dx,dy), open boundaries ---
@inline function corr_offset(A, dx::Int, dy::Int, mean_s::Float64, connected::Bool)
    nx, ny = size(A)
    nx2 = nx - dx
    ny2 = ny - dy
    (nx2 <= 0 || ny2 <= 0) && return 0.0, 0

    @views A1 = A[1:nx2, 1:ny2]
    @views A2 = A[1+dx:nx, 1+dy:ny]

    n = nx2 * ny2
    ss = dot(vec(A1), vec(A2))

    if !connected
        return ss / n, n
    end

    s1 = sum(A1)
    s2 = sum(A2)
    # (s-μ)(s'-μ) = ss - μ(s1+s2) + μ^2 n
    return (ss - mean_s*(s1 + s2) + mean_s*mean_s*n) / n, n
end

@inline function mean_links(A)
    # mean over all entries in A (your "bulk" mean over that link set)
    s = sum(A)
    n = length(A)
    return s / max(n, 1)
end

"""
Snapshot bond-bond correlator at (dx,dy) from bc.bond with open boundaries.

orientation = :h, :v, :both
connected   = subtracts a SINGLE mean over the chosen link-set (as in your code)
"""
function bond_bond_corr_snapshot_fast(
    bc::Bonds, dx::Int, dy::Int;
    orientation::Symbol = :both,
    connected::Bool = false,
)
    lat = bc.lattice
    Lx, Ly = lat.Lx, lat.Ly
    b = bc.bond

    # Views into your native storage: x-link is odd entries, y-link is even entries
    @views Hfull = reshape(b[1:2:end], Lx, Ly)  # x-links "from" each vertex (x→x+1)
    @views Vfull = reshape(b[2:2:end], Lx, Ly)  # y-links "from" each vertex (y→y+1)

    if orientation === :h
        @views H = Hfull[1:Lx-1, 1:Ly]          # valid x-links
        μ = connected ? mean_links(H) : 0.0
        return corr_offset(H, dx, dy, μ, connected)

    elseif orientation === :v
        @views V = Vfull[1:Lx, 1:Ly-1]          # valid y-links
        μ = connected ? mean_links(V) : 0.0
        return corr_offset(V, dx, dy, μ, connected)

    else # :both
        @views H = Hfull[1:Lx-1, 1:Ly]
        @views V = Vfull[1:Lx, 1:Ly-1]

        μH = connected ? mean_links(H) : 0.0
        μV = connected ? mean_links(V) : 0.0

        CH, nH = corr_offset(H, dx, dy, μH, connected)
        CV, nV = corr_offset(V, dx, dy, μV, connected)

        # Weighted by number of pairs, like your explicit accumulation did.
        nT = nH + nV
        nT == 0 && return 0.0, 0
        return (CH*nH + CV*nV) / nT, nT
    end
end

"""
One-chain thermal average of C(dx,dy) on a full grid dx,dy ∈ 0:dmax.

Returns:
    Cmean::Matrix, Cstderr::Matrix, npairs_mean::Matrix
All are (dmax+1)×(dmax+1).
"""
function bond_bond_corr_thermal_grid_fast!(
    bc::Bonds,
    rng::AbstractRNG,
    dmax::Int;
    orientation::Symbol = :both,
    connected::Bool = false,
    burnin::Int = 1_000,
    nsamples::Int = 2_000,
    thin::Int = 10,
    mc_step! = MC_T0_loop!,
)
    δB_0s = δB_0_tuple(bc)
    n = dmax + 1

    meanC = zeros(Float64, n, n)
    M2C   = zeros(Float64, n, n)
    npacc = zeros(Float64, n, n)

    # burnin
    @inbounds for _ in 1:burnin
        mc_step!(bc, rng, δB_0s)
    end

    # sample
    @inbounds for s in 1:nsamples
        for _ in 1:thin
            mc_step!(bc, rng, δB_0s)
        end

        for dx in 0:dmax, dy in 0:dmax
            C, np = bond_bond_corr_snapshot_fast(bc, dx, dy; orientation=orientation, connected=connected)

            i = dx + 1
            j = dy + 1

            δ = C - meanC[i, j]
            meanC[i, j] += δ / s
            M2C[i, j]   += δ * (C - meanC[i, j])

            npacc[i, j] += np
        end
    end

    Cstderr = fill(NaN, n, n)
    if nsamples > 1
        @inbounds Cstderr .= sqrt.( (M2C ./ (nsamples - 1)) ./ nsamples )
    end
    npmean = npacc ./ nsamples

    return meanC, Cstderr, npmean
end

# --- radial binning (short + safe) ---
function radialize_corr(Cmean::AbstractMatrix, Cstderr::AbstractMatrix; dmax::Int)
    shells = Dict{Int, Tuple{Float64,Float64,Int}}() # r2 => (sumC, sumVar, count) with var ~ dC^2
    @inbounds for dx in 0:dmax, dy in 0:dmax
        r2 = dx*dx + dy*dy
        C  = Cmean[dx+1, dy+1]
        dC = Cstderr[dx+1, dy+1]
        sumC, sumVar, cnt = get(shells, r2, (0.0, 0.0, 0))
        shells[r2] = (sumC + C, sumVar + dC*dC, cnt + 1)
    end

    r2s = sort!(collect(keys(shells)))
    rs  = Vector{Float64}(undef, length(r2s))
    Cr  = similar(rs)
    dCr = similar(rs)
    nr  = Vector{Int}(undef, length(r2s))

    @inbounds for k in eachindex(r2s)
        r2 = r2s[k]
        sumC, sumVar, cnt = shells[r2]
        rs[k]  = sqrt(r2)
        Cr[k]  = sumC / cnt
        dCr[k] = sqrt(sumVar) / cnt   # crude combine (matches your “naive” spirit)
        nr[k]  = cnt
    end
    return rs, Cr, dCr, nr
end

#-------------------------
# Plotting
#-------------------------

"""
    plot_bonds(bonds; cmap=:RdBu, lw=4)

Plot a Bonds object on a square lattice using colored links.

Arguments
---------
bonds::Bonds
    A struct with fields:
        - lattice::Lattice (having Lx, Ly)
        - max_bond::Int     -> symmetric color range [-max_bond, max_bond]
        - bond::Vector{Int} -> values to display on the links

Keyword Arguments
-----------------
cmap  : colormap (default :RdBu)
lw    : line width of bond segments
"""

function plot_bondsnv(bonds::Bonds; cmap=:RdBu, lw=4)
    Lx   = bonds.lattice.Lx
    Ly   = bonds.lattice.Ly
    vals = bonds.bond
    Nmax = bonds.max_bond
    Nsites = Lx * Ly

    @assert length(vals) ≥ 2Nsites "Bond vector too short for 2 bonds per site."

    # -----------------------
    # Build bond segments
    # -----------------------
    xs = Float64[]
    ys = Float64[]
    zs = Float64[]

    for i in 1:Nsites
        x = ((i-1) % Lx) + 1
        y = ((i-1) ÷ Lx) + 1

        # Right bond: (x, y) -> (x+1, y)
        if x < Lx
            b = vals[2i - 1]
            append!(xs, (x, x+1, NaN))
            append!(ys, (y, y,   NaN))
            append!(zs, (b, b,   NaN))
        end

        # Up bond: (x, y) -> (x, y+1)
        if y < Ly
            b = vals[2i]
            append!(xs, (x, x,   NaN))
            append!(ys, (y, y+1, NaN))
            append!(zs, (b, b,   NaN))
        end
    end

    # -----------------------
    # Base plot: bonds with colorbar
    # -----------------------
    p = plot(xs, ys;
             seriestype   = :path,
             line_z       = zs,
             c            = cmap,
             clim         = (-Nmax, Nmax),
             linewidth    = lw,
             aspect_ratio = :equal,
             xlabel       = "x",
             ylabel       = "y",
             colorbar     = true,
             legend       = false)

    # -----------------------
    # Vertex positions & charges
    # -----------------------
    q = vertex_charges(bonds)

    xs_v = [((i-1) % Lx) + 1 for i in 1:Nsites]
    ys_v = [((i-1) ÷ Lx) + 1 for i in 1:Nsites]

    # draw vertices as small black dots
    scatter!(p, xs_v, ys_v;
             markersize = 4,
             marker     = :circle,
             color      = :black)

    # numeric labels for charges at each vertex
    ann = [ (xs_v[i], ys_v[i], text(string(round(q[i], digits=1)), 8, :black, :center))
            for i in 1:Nsites ]

    annotate!(p, ann)

    return p
end
