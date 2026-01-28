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
    lattice::Lattice
    max_bond::Int
    bond::Vector{Int}
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

L=8
lattice = Lattice(L,L)
bc = Bonds(lattice, 2, zeros(Int, 2*lattice.Lx*lattice.Ly))
rng = MersenneTwister(1234)

dmax = 6
Cmean, Cstderr, npairs = bond_bond_corr_thermal_grid_fast!(bc, rng, dmax;
    orientation = :both,
    connected   = false,
    burnin      = 1_000,
    nsamples    = 900_000,
    thin        = 20,
    mc_step!    = MC_T0_loop!,
)

# dy=0 cut:
Cs = @view Cmean[:, 1]
plot(0:dmax, log10.(abs.(Cs)); xlabel="dx", ylabel="C(dx,0)", title="Bond-Bond Correlator Cut (dy=0)")


rs, Cr, dCr, nr = radialize_corr(Cmean, Cstderr; dmax=dmax)
p=plot(rs, log10.(abs.(Cr) .+ eps(Float64)); marker=:circle, xlabel="r", ylabel="log10|C(r)|", label="")
xlims!(p, (0, dmax))
display(p)
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




#-------------------------
# Running the Monte Carlo
#-------------------------

lattice=Lattice(2,2)
N=2 #max_bond value
bond_config=Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly))
rng=MersenneTwister(1234)
δB_0s = (-4.0,-2.0, -1.0, 1.0, 2.0,4.0)
MC_T0_loop!(bond_config, rng, δB_0s)
plot_bondsnv(bond_config)


direc= allowed_step_first(1.0, bond_config, 14)
#bond_config.bond[8]=2

@time MC_T0_loop!(bond_config, MersenneTwister(1234), δB_0s)

#-------------------------
# Testing ergodicity
#-------------------------
#Count up the number of configurations for N=2 on a 2x2 lattice
lattice=Lattice(2,2)
N=2 #max_bond value
bond_config=Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly))

#Only 3 configs, can store counts in vector which is 3 long
vector=[0,0,0]
for i in 1:10^6
    MC_T0_loop!(bond_config, rng, δB_0s)
    vector[bond_config.bond[1]+2]+=1 #Configs uniquely determined by value at 1.
    #Added on +2 since -1 -> +1, etc. for the indices.
end
println(vector)

#Count up the number of configurations for N=3 on a LxL lattice
lattice=Lattice(3,3)
N=3 #max_bond value
bond_config=Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly))
δB_0s=δB_0_tuple(bond_config)
#Store counts in a dictionary
dict=Dict{Vector, Int64}()

for i in 1:10^6
    #bonds_0=copy(bond_config.bond)
    MC_T0_loop!(bond_config, rng, δB_0s)
    # if bond_config.bond == bonds_0
    #     continue
    # end
    dict[copy(bond_config.bond)] = get(dict, copy(bond_config.bond), 0) + 1
end
println(length(keys(dict)))

@btime MC_T0_loop!($bond_config, $rng, $δB_0s)
# Plot the first 15 unique configs found
count=0
for (i,v) in dict
    bond_config_dummy=Bonds(lattice, N, i)
    println("Configuration: ", i, " Count: ", v)
    p=plot_bondsnv(bond_config_dummy)
    title!(p, "Count: $v")
    display(p)
    count+=1
    if count>=15
        break
    end
end

#Count up the number of configurations for N=2 on a 3x3 lattice
lattice=Lattice(3,3)
N=2 #max_bond value
bond_config=Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly))
δB_0s=δB_0_tuple(bond_config)
#Store counts in a dictionary
dict=Dict{Vector, Int64}()

for i in 1:10^8
    #bonds_0=copy(bond_config.bond)
    MC_T0_loop!(bond_config, rng, δB_0s)
    # if bond_config.bond == bonds_0
    #     continue
    # end
    dict[copy(bond_config.bond)] = get(dict, copy(bond_config.bond), 0) + 1
end


println(length(keys(dict))) #Number of unique configurations found
println(collect(values(dict))) #List of unique configurations found


#Count up the number of configurations for N=4 on a 3x3 lattice
lattice=Lattice(3,3)
N=4 #max_bond value
bond_config=Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly))
δB_0s=δB_0_tuple(bond_config)
#Store counts in a dictionary
dict=Dict{Vector, Int64}()

for i in 1:10^8
    #bonds_0=copy(bond_config.bond)
    MC_T0_loop!(bond_config, rng, δB_0s)
    # if bond_config.bond == bonds_0
    #     continue
    # end
    dict[copy(bond_config.bond)] = get(dict, copy(bond_config.bond), 0) + 1
end

println(length(keys(dict))) #Number of unique configurations found
println(collect(values(dict))) #List of unique configurations found

results, slope, dict = uniformity_scaling_test!(
    bond_config, rng;
    Nmax=100_000_000,
    topK=200,
    burnin=10_000,
    mc_step! =MC_T0_loop!,
)

println("Fitted slope log(CV) vs log(n): ", slope, "  (shot noise target: -0.5)")
for r in results
    println("n=$(r.n)  K=$(r.K)  mean=$(round(r.mean,digits=2))  std=$(round(r.std,digits=2))  CV=$(round(r.cv,digits=4))")
end



for (i,v) in dict
    bond_config_dummy=Bonds(lattice, N, i)
    println("Configuration: ", i, " Count: ", v)
    p=plot_bondsnv(bond_config_dummy)
    title!(p, "Count: $v")
    display(p)
end
plot_bondsnv(bond_config)

MC_T0_loop!(bond_config, rng)
plot_bondsnv(bond_config)



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
    
    # Content-based immutable key (diagnostic-safe; allocates)
    statekey() = Tuple(bond_config.bond)

    # Directed transition counts: (a,b) -> N_ab
    trans  = Dict{Tuple{Any,Any}, Int}()
    visits = Dict{Any, Int}()

    # Optional snapshots for printing
    states = Dict{Any, Any}()

    # Burn-in
    for _ in 1:burnin
        mc_step!(bond_config, rng)
    end

    # Initialize
    a = statekey()
    visits[a] = get(visits, a, 0) + 1
    if store_states && !haskey(states, a)
        states[a] = copy(bond_config.bond)
    end

    # Main sampling
    for _ in 1:nsteps
        mc_step!(bond_config, rng)
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
rng = MersenneTwister(1234)
trans, visits, diffs, states = test_detailed_balance_uniform!(
    bond_config, rng;
    burnin=10_000,
    nsteps=1000_000_000,
    min_pair_count=50,
    topk=20,
    mc_step! = MC_T0_loop!
)

#-------------------------
# Testing explicit transition probabilities
#-------------------------

# Pair that violates detailed balance
#--------------------------
#state_A=[0, 0, 1, -1, 0, 2, 0, 0, -1, -1, 0, 2, 0, 0, -2, 0, 0, 0] #[0, 0, 1, -1, 0, 2, 0, 0, -1, -1, 0, 2, 0, 0, -2, 0, 0, 0]
#state_B=[0, 0, -1, 1, 0, -2, -1, 1, 1, -1, 0, -2, 2, 0, 2, 0, 0, 0] #[0, 0, -1, 1, 0, -2, -1, 1, 1, -1, 0, -2, 2, 0, 2, 0, 0, 0]
# state_A=[0, 0, 1, -1, 0, 2, 1, -1, -1, 1, 0, 2, -2, 0, -2, 0, 0, 0]
# state_B=[0, 0, 0, 0, 0, 0, -1, 1, -1, -1, 0, -2, 2, 0, 2, 0, 0, 0]
# Plot them
bond_config_A=Bonds(lattice, N, copy(state_A))
p=plot_bondsnv(bond_config_A)
display(p)
bond_config_B=Bonds(lattice, N, copy(state_B))
p=plot_bondsnv(bond_config_B)
display(p)




# Pair that obeys detailed balance
#--------------------------
state_A=[0, 0, 1, -1, 0, 2, 1, -1, -1, 1, 0, 2, -2, 0, -2, 0, 0, 0] #[0,0,0,0,0,0,-1,1,-1,-1,0,-2,2,0,2,0,0,0]
state_B=[0, 0, 0, 0, 0, 0, -1, 1, -1, -1, 0, -2, 2, 0, 2, 0, 0, 0]#[0,0,-1,1,0,-2,-1,1,1,-1,0,-2,2,0,2,0,0,0]
v0=6
rng=MersenneTwister(1234)
function CTRL_MC_T0_loop!(bond_config::Bonds, rng::AbstractRNG, v0::Int, δB_0::Float64 = rand(rng, (-4.0,-2.0, -1.0, 1.0, 2.0,4.0)))
    made_moves = Int[]

    lattice = bond_config.lattice
    Lx      = lattice.Lx
    Ly      = lattice.Ly
    Nsites  = Lx * Ly

    # INITIALISE
    index_0 = v0  # deterministic starting point

    # Value to change spin by (controlled)
    #δB_0    = +2.0#rand(rng, (-2.0, -1.0, 1.0, 2.0))
    δB_prev = δB_0

    # First move (still uses your existing first-move logic)
    move_0 = allowed_step_first(δB_0, bond_config, index_0)
    if move_0 == false
        return made_moves
    end

    index_curr = index_0 + move_0
    index_prev = index_0

    bond_prev, _ = bond_label(index_curr, index_prev)
    bond_config.bond[bond_prev] += δB_prev

    push!(made_moves, move_0)

    # Safety valve
    max_attempts = 100_000
    attempts = 0

    while index_curr != index_0
        attempts += 1
        if attempts > max_attempts
            print("Triggered max_moves")
            return made_moves
        end

        # New allowed_step: 1/4 directions; invalid => (0,0,0.0) self-loop
        Δmove, bond_curr, δB_curr = allowed_step(δB_prev, bond_config, rng, index_curr, index_prev)

       
        # Record the attempted move. (You can choose to record only nonzero moves.)
        push!(made_moves, Δmove)

        # Self-loop: do nothing, try again
        if Δmove == 0
            continue
        end


        # Apply move
        index_prev = index_curr
        index_curr = index_curr + Δmove

        bond_config.bond[bond_curr] += δB_curr

        # Update for next iteration
        δB_prev = δB_curr
    end

    return made_moves
end

#Count up the number of transitions to each state when we have a fixed start for each A and B.
transitions_A=Dict{Vector, Int64}()
bond_config=Bonds(lattice, N, copy(state_A))
for i in 1:10^6
    bond_config=Bonds(lattice, N, copy(state_A))
    made_moves=CTRL_MC_T0_loop!(bond_config, rng, v0)
    transitions_A[copy(bond_config.bond)] = get(transitions_A, copy(bond_config.bond), 0) + 1
end

transitions_B=Dict{Vector, Int64}()
bond_config=Bonds(lattice, N, copy(state_B))
for i in 1:10^6
    bond_config=Bonds(lattice, N, copy(state_B))
    CTRL_MC_T0_loop!(bond_config, rng, v0)
    transitions_B[copy(bond_config.bond)] = get(transitions_B, copy(bond_config.bond), 0) + 1
end

# Check the difference in the number of transitions. Should be the same if detailed balance is obeyed.
println("Transitions from A to B: ", get(transitions_A, state_B, 0))
println("Transitions from B to A: ", get(transitions_B, state_A, 0))

desired_moves=[3,1,-3,-1]#[3,-1,3,1,-3,-1,3,1,1,-3,-3,-1]
counts=zeros(Int, length(desired_moves))
for i in 1:10^6
    bond_config=Bonds(lattice, N, copy(state_A))
    moves=CTRL_MC_T0_loop!(bond_config, rng, v0, 1.0)
    for k in 1:length(moves)
        if moves[k] == desired_moves[k]
            counts[k]+=1
        else
            break
        end
    end
end

for k in 1:length(counts)-1
    println(counts[k+1]/counts[k])
end



desired_moves_rev=-reverse(desired_moves)
counts=zeros(Int, length(desired_moves_rev))
for i in 1:10^6
    bond_config=Bonds(lattice, N, copy(state_B))
    moves=CTRL_MC_T0_loop!(bond_config, rng, v0, 1.0)
    for k in 1:length(moves)
        if moves[k] == desired_moves_rev[k]
            counts[k]+=1
        else
            break
        end
    end
end

for k in 1:length(counts)-1
    println(counts[k+1]/counts[k])
end


state_intermediate=[0, 0, 1, 0, 0, 2, 0,0, -1, 1, 0, 2, -2, 0, -2, 0, 0, 0]
bond_config_int=Bonds(lattice, N, copy(state_intermediate))
p=plot_bondsnv(bond_config_int)
display(p)
count_1=0
for k in 1:10^5
    if allowed_step(1.0, bond_config, rng, 7, 4)[1] == 1
        count_1+=1
    end
end
println("Probability of first move being +1 from v0=2: ", count_1/10^5)
# count1=0
# count2=0
# count3=0
# count4=0
# desired_moves = [1,-3,-1,3]#[-1,3,1,-3]#[1, 3, -1, -3]
# for i in 1:10^6
#     bond_config=Bonds(lattice, N, copy(state_A))
#     moves=CTRL_MC_T0_loop!(bond_config, rng, v0)
    
#     if moves[1] == desired_moves[1]
#         count1+=1
#         if moves[2] == desired_moves[2]
#             count2+=1
#             if moves[3] == desired_moves[3]
#                 count3+=1
#                 if moves[4] == desired_moves[4]
#                     count4+=1
#                 end
#             end
#         end
#     end
# end

# #Find the probability of the loop occurring
# println("Results for A -> B transition in single loop flip channel:")
# println("Numerical result, ", count1/10^6*count2/count1*count3/count2*count4/count3)
# println("Analytic result, ", (1/4)^4)


# count1=0
# count2=0
# count3=0
# count4=0
# desired_moves = [1,-3,-1,3]#[-3,1,3,-1]#[-1,3,1,-3]#[1, 3, -1, -3]
# for i in 1:10^6
#     bond_config=Bonds(lattice, N, copy(state_B))
#     moves=CTRL_MC_T0_loop!(bond_config, rng, v0)
#     if length(moves) < 4
#         continue
#     end
#     if moves[1] == desired_moves[1]
#         count1+=1
#         if moves[2] == desired_moves[2]
#             count2+=1
#             if moves[3] == desired_moves[3]
#                 count3+=1
#                 if moves[4] == desired_moves[4]
#                     count4+=1
#                 end
#             end
#         end
#     end
# end

# #Find the probability of the loop occurring
# println("Results for B -> A transition in single loop flip channel:")
# println("Numerical result, ", count1/10^6*count2/count1*count3/count2*count4/count3)
# println("Analytic result, ", (1/4)^4)



total=0
for k in 1:10^4
    # Uncontrolled looping to get move probabilities
    transitions_A=Dict{Vector, Int64}()
    bond_config=Bonds(lattice, N, copy(state_A))
    for i in 1:10^4
        bond_config=Bonds(lattice, N, copy(state_A))
        MC_T0_loop!(bond_config, rng)
        transitions_A[copy(bond_config.bond)] = get(transitions_A, copy(bond_config.bond), 0) + 1
    end

    transitions_B=Dict{Vector, Int64}()
    bond_config=Bonds(lattice, N, copy(state_B))
    for i in 1:10^4
        bond_config=Bonds(lattice, N, copy(state_B))
        MC_T0_loop!(bond_config, rng)
        transitions_B[copy(bond_config.bond)] = get(transitions_B, copy(bond_config.bond), 0) + 1
    end
    total+=get(transitions_A, state_B, 0)-get(transitions_B, state_A, 0)
end
println(total/10^4)
# Check the difference in the number of transitions. Should be the same if detailed balance is obeyed.
println("Transitions from A to B: ", get(transitions_A, state_B, 0))
println("Transitions from B to A: ", get(transitions_B, state_A, 0))


using Statistics

function measure_delta(lattice, N, state_A, state_B;
                       n_inner::Int,
                       n_outer::Int,
                       rng::AbstractRNG)

    deltas = zeros(Float64, n_outer)

    for k in 1:n_outer
        hits_AB = 0
        hits_BA = 0

        for i in 1:n_inner
            bc = Bonds(lattice, N, copy(state_A))
            MC_T0_loop!(bc, rng)
            if bc.bond == state_B
                hits_AB += 1
            end

            bc = Bonds(lattice, N, copy(state_B))
            MC_T0_loop!(bc, rng)
            if bc.bond == state_A
                hits_BA += 1
            end
        end

        deltas[k] = hits_AB - hits_BA
    end

    return mean(deltas), std(deltas)
end


# Check for shot noise
rng = MersenneTwister(1234)

n_outer = 200          # repetitions for variance
n_list  = [500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000, 1000_000]  # inner loop sizes

println(" n_inner    mean(Δ)    std(Δ)    std(Δ)/sqrt(n)")
println("------------------------------------------------")

# If the measure plateaus at the end of the table, sqrt behaviour, hence shot noise. 
for n in n_list
    μ, σ = measure_delta(lattice, N, state_A, state_B;
                         n_inner = n,
                         n_outer = n_outer,
                         rng = rng)

    println(rpad(n,8), " ",
            round(μ, digits=4), "   ",
            round(σ, digits=4), "   ",
            round(σ / sqrt(n), digits=4))

end

# # Dictionary with probabilities for each different initial move from v0
# # First case is where we just make a step, don't need to resolve the move.
# bond_config=Bonds(lattice, N, copy(state_A))
# lattice=bond_config.lattice
# Lx=lattice.Lx
# Ly=lattice.Ly
# Nsites=lattice.Lx*lattice.Ly

# δB_0=1.0
# δB_prev = δB_0

# Niterations = 100000
# move_counts = Dict{Int, Int}()
# n_fail = 0

# for i in 1:Niterations
#     mv = allowed_step_first(δB_0, bond_config, v0)
#     if mv === false
#         n_fail += 1
#         continue
#     end
#     move_counts[mv] = get(move_counts, mv, 0) + 1
# end

# total_success = sum(values(move_counts))

# # Probabilities relative to all trials, and conditional on successful moves
# move_prob_total = Dict(k => v / Niterations for (k, v) in move_counts)
# move_prob_success = total_success > 0 ? Dict(k => v / total_success for (k, v) in move_counts) : Dict{Int, Float64}()

# println("Counts: ", move_counts)
# println("Prob (per trial): ", move_prob_total)
# println("Prob (conditional on success): ", move_prob_success)
# println("Failures: $n_fail / $Niterations")


# #Second case make the full move.
# Niterations = 100000
# move_counts = Dict{Int, Int}()
# n_fail = 0

# for i in 1:Niterations
#     bond_config=Bonds(lattice, N, copy(state_B))
#     mv = CTRL_MC_T0_loop!(bond_config, rng, v0)
#     if mv === false
#         n_fail += 1
#         continue
#     end
#     move_counts[mv] = get(move_counts, mv, 0) + 1
# end

# total_success = sum(values(move_counts))

# # Probabilities relative to all trials, and conditional on successful moves
# move_prob_total = Dict(k => v / Niterations for (k, v) in move_counts)
# move_prob_success = total_success > 0 ? Dict(k => v / total_success for (k, v) in move_counts) : Dict{Int, Float64}()

# println("Counts: ", move_counts)
# println("Prob (per trial): ", move_prob_total)
# println("Prob (conditional on success): ", move_prob_success)
# println("Failures: $n_fail / $Niterations")




#-------------------------
# Profiling
#-------------------------
using Profile
rng = MersenneTwister(1234)

# warm-up
MC_T0_loop!(bond_config, rng)
plot_bondsnv(bond_config)

Profile.clear()
@profile begin
    for _ in 1:10^6  # increase if needed
        MC_T0_loop!(bond_config, rng)
    end
end

Profile.print()











#-------------------------
# Checks on code
#-------------------------

# Check to see the lattice plots correctly
lattice_1=Lattice(3,3)
p=plot(legend=false, aspect_ratio=1,title="Lattice Link Centres")
for index in 1:(lattice_1.Lx*lattice_1.Ly)
    coord = index_to_coord(lattice_1, index)
    scatter!(p,coord, color=:blue, markersize=4)
    #println("Index: $index -> Coord: $coord")
end
display(p)


lattice_1 = Lattice(3,3)

p = plot(
    legend = false,
    aspect_ratio = 1,
    title = "Lattice Link Centres",
    xlim = (0.5, lattice_1.Lx + 1.5),
    ylim = (0.5, lattice_1.Ly + 1.5),
)

# sites
for index in 0:(lattice_1.Lx * lattice_1.Ly - 1)
    x, y = divrem(index, lattice_1.Lx)
    scatter!(p, (x+1, y+1), color=:blue, markersize=4)
end

Lx, Ly = lattice_1.Lx, lattice_1.Ly
N      = Lx * Ly

# bonds (2i-1 to the right, 2i upwards), with dangling edges
for i in 1:N
    x = ((i-1) % Lx) + 1
    y = ((i-1) ÷ Lx) + 1

    # --- right bond (2i-1) ---
    if x < Lx
        xr = x + 1
    else
        xr = x + 1        # dangles off to the right (x = Lx+1)
    end
    b_right = 2i - 1
    plot!(p, [x, xr], [y, y], color=:gray)
    annotate!(p, ((x + xr)/2, y, text("$b_right", 6)))

    # --- up bond (2i) ---
    if y < Ly
        yu = y + 1
    else
        yu = y + 1        # dangles off top (y = Ly+1)
    end
    b_up = 2i
    plot!(p, [x, x], [y, yu], color=:gray)
    annotate!(p, (x, (y + yu)/2, text("$b_up", 6)))
end
# sites + red labels
for index in 0:(lattice_1.Lx * lattice_1.Ly - 1)
    y, x = divrem(index, lattice_1.Lx)
    xp, yp = x + 1, y + 1
    scatter!(p, (xp, yp), color=:blue, markersize=4)
    annotate!(p, (xp, yp, text("$(index+1)", 8, :red)))
end
display(p)



Lx = 5

@inline function bond_between(index_prev::Int, index_curr::Int, Lx::Int)
    i = index_prev < index_curr ? index_prev : index_curr
    Δ = index_curr - index_prev
    dx = mod(Δ, Lx)
    horizontal = (dx == 1) || (dx == Lx - 1)
    return horizontal ? (2i - 1) : (2i)
end

@inline function bond_betweenPBC(index_prev::Int, index_curr::Int, Lx::Int)
    i = index_prev < index_curr ? index_prev : index_curr
    Δ = index_curr - index_prev

    if abs(Δ) == 1 #x hop
        return 2i - 1 
    elseif abs(Δ) == Lx-1 #PBC in x
        i = index_prev > index_curr ? index_prev : index_curr # Find max index
        return 2i - 1  #TK horizontal bond (PBC)
    elseif abs(Δ) == Lx #y hop
        return 2i      # vertical bond
    else #PBC in y
        i = index_prev > index_curr ? index_prev : index_curr # Find max index
        return 2i      #TK vertical bond (PBC) #Find max index
    end
end

@inline function bond_betweenPBC_fast(index_prev::Int, index_curr::Int, Lx::Int)
    Δ   = index_curr - index_prev
    aΔ  = abs(Δ)
    i_min = index_prev < index_curr ? index_prev : index_curr

    if aΔ == 1             # x hop (internal)      → use min
        return 2*i_min - 1
    elseif aΔ == Lx        # y hop (internal)      → use min
        return 2*i_min
    elseif aΔ == Lx - 1    # x hop (PBC)           → use max
        i_max = index_prev > index_curr ? index_prev : index_curr
        return 2*i_max - 1
    else                   # y hop (PBC)           → use max
        i_max = index_prev > index_curr ? index_prev : index_curr
        return 2*i_max
    end
end


Lx = 7
function test_bonds()
    pairs = [
        (1, 2),   # right
        (2, 1),   # left
        (1, Lx+1),   # up
        (Lx+1, 1),   # down
        (Lx, 1),   # wrap right (x:5→1)
        (1, Lx),   # wrap left  (x:1→5)
        ((Ly-1)*Lx+1, 1),  # wrap down (y:5→1)
        (1, (Ly-1)*Lx+1)   # wrap up   (y:1→5)
    ]

    for (a,b) in pairs
        i = min(a,b)
        bidx = bond_PBC(a, b, Lx)
        println("($a, $b)  ->  owning site i = $i, bond = $bidx")
    end
end

test_bonds()



#Modified step function here

# @inline function allowed_step_first_mod(bond_config::Bonds, index_curr::Int, rng::AbstractRNG)
#     Lx = bond_config.lattice.Lx
#     Ly = bond_config.lattice.Ly
#     bonds = bond_config.bond
#     N = bond_config.max_bond

#     steps = (-1, 1, -Lx, Lx)

#     # Build list of all allowed (step, δB) pairs
#     pairs_step = Int[]
#     pairs_dB   = Int[]
#     for step in steps
#         in_bounds(index_curr, step, Lx, Ly) || continue
#         index_next = index_curr + step
#         bond = bond_label(index_curr, index_next)[1]

#         b = bonds[bond]

#         # Allowed integer δB such that |b + δB| <= N and δB != 0
#         lo = max(-N - b, -N)
#         hi = min( N - b,  N)

#         # δB in [lo,hi] excluding 0
#         for δB in lo:hi
#             δB == 0 && continue
#             push!(pairs_step, step)
#             push!(pairs_dB, δB)
#         end
#     end

#     isempty(pairs_step) && return (0, 0)

#     k = rand(rng, 1:length(pairs_step))
#     return pairs_step[k], pairs_dB[k]
# end


"""
Shot-noise scaling test for uniformity.

- mc_step!(bond_config, rng) should perform one MCMC update (your MC_T0_loop!)
- The dict key is Tuple(bond_config.bond), which is content-based and safe.
- We analyze CV over the topK most visited states at each checkpoint to keep K fixed.

Returns:
    results: Vector of named tuples with fields
        n, K, mean, std, cv
    slope: fitted slope of log(cv) vs log(n) (target ~ -0.5)
"""
function uniformity_scaling_test!(
    bond_config,
    rng;
    Nmax::Int = 100_000_000,
    checkpoints::Vector{Int} = unique(round.(Int, 10 .^ range(3, log10(Nmax), length=20))),
    topK::Int = 200,          # choose based on how many states you typically revisit
    burnin::Int = 10_000,
    mc_step! = MC_T0_loop!,
)

    # Burn-in (optional but recommended)
    for _ in 1:burnin
        mc_step!(bond_config, rng)
    end

    dict = Dict{Tuple, Int}()

    results = NamedTuple[]
    cp_idx = 1
    next_cp = checkpoints[cp_idx]

    for i in 1:Nmax
        mc_step!(bond_config, rng)

        key = Tuple(bond_config.bond)
        dict[key] = get(dict, key, 0) + 1

        if i == next_cp
            # Extract counts and take topK (fixed-K analysis)
            vals = collect(values(dict))
            sort!(vals; rev=true)

            K = min(topK, length(vals))
            top = @view vals[1:K]

            μ = mean(top)
            σ = std(top)
            cv = σ / μ

            push!(results, (n=i, K=K, mean=μ, std=σ, cv=cv))

            # Advance checkpoint
            cp_idx += 1
            if cp_idx > length(checkpoints)
                break
            end
            next_cp = checkpoints[cp_idx]
        end
    end

    # Fit slope of log(cv) vs log(n): expected ≈ -0.5 for shot noise
    xs = log.(Float64[r.n for r in results])
    ys = log.(Float64[r.cv for r in results])

    x̄ = mean(xs); ȳ = mean(ys)
    slope = sum((xs .- x̄) .* (ys .- ȳ)) / sum((xs .- x̄).^2)
    p=plot(xs, ys, seriestype=:scatter, label="Data", xlabel="log(N_iterations)", ylabel="log(Error)")
    display(p)
    return results, slope, dict
end