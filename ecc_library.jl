"""
Library of functions for Exponential Charge Conservation
Created: 04.12.2025
"""

using Plots
using Random
using BenchmarkTools

#-------------------------
# Define Lattice Structure
#-------------------------

#Struct that stores the lattice dimensions
struct Lattice
    #Dimensions of the lattice
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
    bond::Vector{Int}  #Bond variables
end

"""
function in_bounds: returns a bool, tells you if you step outside the lattice
index_curr::Int -> starting index
step::Int -> step size
Lx, Ly::Int -> Lattice dimensions
"""
#TK can maybe make this more efficient by taking lattice as an input.
@inline function in_bounds(index_curr::Int, step::Int, Lx::Int, Ly::Int)
    # Attempted new index
    index_next = index_curr + step

    # Global bound check
    if index_next < 1 || index_next > Lx * Ly
        return false
    end

    # --- Extract x,y from flattened coordinate (1-indexed) ---
    # x runs 1..Lx, y runs 1..Ly
    x = ((index_curr - 1) % Lx) + 1
    y = ((index_curr - 1) ÷ Lx) + 1

    # Horizontal moves
    if step == 1      && x == Lx
        return false
    elseif step == -1 && x == 1
        return false
    end

    # Vertical moves
    if step == Lx     && y == Ly
        return false
    elseif step == -Lx && y == 1
        return false
    end

    return true
end



# Function to convert index to coordinate of centre of link
@inline function index_to_coord(lattice::Lattice, index::Int)
    Ly = lattice.Ly
    # q: which "column" (0-based), r: position within column (0-based)
    q, r = divrem(index - 1, Ly)  # q,r are Int

    # 0.0 if r even, 0.5 if r odd — branchless
    offset = 0.5 * (r & 0x1)

    x = q + 1 + offset
    y = 1.0 + 0.5 * r

    return (x, y)
end

#Return the index of a vertex given x and y coordinates
@inline idx(lat::Lattice, x::Int, y::Int) = (y-1)*lat.Lx + x

#Efficient random move generation using bools
#Picks a random move from (-Lx,+Lx,-1,+1) without storing memory
@inline function rand_step(lat::Lattice, rng::AbstractRNG)
    axis = rand(rng, Bool)           # false → ±1, true → ±Lx
    sign = rand(rng, Bool) ? 1 : -1
    step = axis ? lat.Lx : 1
    return sign * step #Returns a random move from (-Lx,+Lx,-1,+1)
end



#Function to grab the bond label between two indices.
#TK need to be careful with this function, can give a wrong bond
# if indices are not nearest neighbours.
@inline function bond_label(index_prev::Int, index_curr::Int)
    i = index_prev < index_curr ? index_prev : index_curr #Find the smallest index
    Δ = index_curr - index_prev 

    # |Δ| == 1  → vertical → 2i-1
    # |Δ| >  1  → horizontal → 2i
    return (abs(Δ) == 1) ? (2i - 1) : (2i), Δ
end

#Takes the effective Δ as input
#Returns value for charge neutrality
@inline function multiplier(Δ_curr, Δ_prev)
    var=sign(Δ_curr)
    if abs(Δ_curr) != abs(Δ_prev) && var != sign(Δ_prev)
        return -1.0 #If we go to partner bond, -1
    else
        return 2.0^var #Any other legal move should by x2 or x1/2. 
    end
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
allowed_step_first -> returns integer step or returns false if no steps allowed
δB_firstmove::Float64 -> chosen size of first step
bond_config::Bonds -> bond object 
index_curr::Int -> starting randomly chosen index
"""

@inline function allowed_step_first(δB_firstmove::Float64, bond_config::Bonds, index_curr::Int, rng::AbstractRNG)
    Lx = bond_config.lattice.Lx
    Ly = bond_config.lattice.Ly
    bonds = bond_config.bond

    #TK these lines are one of the most inefficient, need to find a better way
    steps = [-1, 1, -Lx, Lx]
    step  = rand(rng, steps) # Pick a random direction to move

    #If the step takes us out of bounds, try again next time
    if !in_bounds(index_curr, step, Lx, Ly) 
        return false
    end

    index_next = index_curr + step
    bond = bond_label(index_curr, index_next)[1] #bond value of the move

    #Allow when within the bond bounds set by max_bond and do not add a fractional value
    if abs(bonds[bond] + δB_firstmove) <= bond_config.max_bond &&
        abs(δB_firstmove) >= 1
        return step
    end

    #Return just false
    return false #error("No allowed steps for first worm move")
end

"""
allowed_step
"""
@inline function allowed_step(
    δB::Float64,
    bond_config::Bonds,
    rng::AbstractRNG,
    index_curr::Int,
    index_prev::Int,
)
    Lx     = bond_config.lattice.Lx
    Ly     = bond_config.lattice.Ly
    Δ_prev = index_curr - index_prev
    bonds  = bond_config.bond

    # 4 geometric directions (each with probability 1/4)
    steps   = [-1, 1, -Lx, Lx]
    shuffle!(steps) # Randomise order of steps

    for step in steps # Accept the first valid step
        # Backtracking allowed to prevent getting stuck
        if step == -Δ_prev
            bond = bond_label(index_curr, index_prev)[1]
            return step, bond, -δB #need to take away what we added
        end

        # open boundary conditions
        if !in_bounds(index_curr, step, Lx, bond_config.lattice.Ly)
            continue
        end

        # Physical constraint on bond variables
        index_next = index_curr + step
        Δ_curr     = step
        #print("last increment: ", Δ_prev, ", current increment: ", Δ_curr, "\n")
        δB_curr    = δB * multiplier(Δ_curr, Δ_prev)
        bond       = bond_label(index_curr, index_next)[1]

        # Can't exceed max bond, and must be integer.
        if abs(bonds[bond] + δB_curr) <= bond_config.max_bond && abs(δB_curr) >= 1
            return step, bond, δB_curr
        end
    end

    error("No allowed steps for (curr=$index_curr, prev=$index_prev)")
end

function MC_T0_loop!(bond_config::Bonds, rng::AbstractRNG, δB_0s::Tuple{Vararg{Float64}})
    lattice = bond_config.lattice
    Lx      = lattice.Lx
    Ly      = lattice.Ly
    Nsites  = Lx * Ly

    # Pick a random vertex starting point on the grid
    index_0 = rand(rng, 1:Nsites)

    # Value to change spin by
    δB_0    = rand(rng, δB_0s)
    δB_prev = δB_0

    # First move (keep your existing logic; you can later refactor similarly)
    move_0 = allowed_step_first(δB_0, bond_config, index_0, rng)
    if move_0 == false
        return
    end


    index_curr = index_0 + move_0
    index_prev = index_0

    bond_prev, Δmove = bond_label(index_curr, index_prev)
    bond_config.bond[bond_prev] += δB_prev

    while index_curr != index_0

        # 1/4 over directions; invalid => (0,0,0.0) self-loop
        Δmove, bond_curr, δB_curr = allowed_step(δB_prev, bond_config, rng, index_curr, index_prev)

        # Self-loop: do nothing, try again
        if Δmove == 0
            continue
        end

        # Apply move
        index_prev = index_curr
        index_curr = index_curr + Δmove

        bond_config.bond[bond_curr] += δB_curr

        # Update variables for next iteration
        δB_prev = δB_curr
    end
end

#-------------------------
# Measurements
#-------------------------

"""
Spin–spin correlator on links (bonds) at displacement (dx,dy).

- orientation = :both  -> include both horizontal and vertical links
              = :h     -> only horizontal links (step = +1)
              = :v     -> only vertical links   (step = +Lx)

- connected=true subtracts mean: <s_i s_j> - <s>^2 (computed over the same link set)
Returns: (C, npairs)
"""
function bond_bond_corr_snapshot(
    bc::Bonds,
    dx::Int,
    dy::Int;
    orientation::Symbol = :both,
    connected::Bool = false,
)
    lat = bc.lattice
    Lx, Ly = lat.Lx, lat.Ly
    bonds = bc.bond

    # choose which link orientations to include
    steps = orientation === :h    ? (1,) :
            orientation === :v    ? (Lx,) :
            (1, Lx)  # :both

    # Optional mean for connected correlator (over same link set in the bulk)
    mean_s = 0.0
    nlinks = 0

    if connected
        for y in 1:Ly, x in 1:Lx
            i = idx(lat, x, y)
            for step in steps
                in_bounds(i, step, Lx, Ly) || continue
                j = i + step
                b, _ = bond_label(i, j)
                mean_s += bonds[b]
                nlinks += 1
            end
        end
        mean_s /= max(nlinks, 1)
    end

    # Accumulate correlator
    sum_ss = 0.0
    npairs = 0

    for y in 1:Ly, x in 1:Lx
        i = idx(lat, x, y)
        x2 = x + dx
        y2 = y + dy
        (1 <= x2 <= Lx && 1 <= y2 <= Ly) || continue
        i2 = idx(lat, x2, y2)

        for step in steps
            # link at origin exists?
            in_bounds(i, step, Lx, Ly) || continue
            j  = i + step
            b1, _ = bond_label(i, j)
            s1 = bonds[b1]

            # corresponding link at displaced vertex exists?
            in_bounds(i2, step, Lx, Ly) || continue
            j2 = i2 + step
            b2, _ = bond_label(i2, j2)
            s2 = bonds[b2]

            if connected
                sum_ss += (s1 - mean_s) * (s2 - mean_s)
            else
                sum_ss += s1 * s2
            end
            npairs += 1
        end
    end

    C = sum_ss / max(npairs, 1)
    return C, npairs
end

"""
Thermal/MCMC average of bond-bond correlator at (dx,dy).

mc_step! should update bc in-place (e.g., MC_T0_loop!)
Returns: (C_mean, C_stderr, n_pairs_mean)

Note: stderr here is naive (assumes weak correlations). For rigorous error bars, use blocking.
"""
function bond_bond_corr_thermal!(
    bc::Bonds,
    rng::AbstractRNG,
    dx::Int,
    dy::Int;
    orientation::Symbol = :both,
    connected::Bool = false,
    burnin::Int = 10_000,
    nsamples::Int = 2_000,
    thin::Int = 10,
    mc_step! = MC_T0_loop!,
)
    # Allowed starting ΔB values for this bond_config
    δB_0s = δB_0_tuple(bc)

    # Burn-in
    for _ in 1:burnin
        mc_step!(bc, rng, δB_0s)
    end

    # Sample
    vals = Vector{Float64}(undef, nsamples)
    npair_acc = 0.0

    for s in 1:nsamples
        for _ in 1:thin
            mc_step!(bc, rng, δB_0s)
        end
        Csnap, np = bond_bond_corr_snapshot(
            bc, dx, dy;
            orientation = orientation,
            connected = connected
        )
        vals[s] = Csnap
        npair_acc += np
    end

    Cmean = mean(vals)
    Cstderr = std(vals) / sqrt(nsamples)   # naive stderr
    npairs_mean = npair_acc / nsamples

    return Cmean, Cstderr, npairs_mean
end



lattice=Lattice(12,12)
N=6 #max_bond value
bond_config=Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly))
δB_0s=δB_0_tuple(bond_config)
rng=MersenneTwister(1234)
MC_T0_loop!(bond_config, rng, δB_0s)
lat = bond_config.lattice

#C, dC, np = bond_bond_corr_thermal!(bond_config, rng, 3, 0; orientation=:h, nsamples=100_000, thin=20)

#MC_T0_loop!(bond_config, rng, δB_0s)
#plot_bondsnv(bond_config)


# --- parameters ---
dmax     = 11                 # for Lx=7, the largest meaningful dx is 6
nsamples = 10_000_000
thin     = 20

# --- measure C(dx,0) for dx = 0..dmax ---
ds  = collect(0:dmax)
Cs  = similar(Float64[], length(ds))
dCs = similar(Float64[], length(ds))

for (i, dx) in enumerate(ds)
    C, dC, np = bond_bond_corr_thermal!(bond_config, rng, dx, 0;
        orientation = :h,
        nsamples    = nsamples,
        thin        = thin,
        connected   = false,
    )
    Cs[i]  = C
    dCs[i] = dC
    println("d=$d  C=$(Cs[i]) ± $(dCs[i])   (npairs≈$np)")
end

y = max.(abs.(Cs), eps(Float64))   # replace 0 with tiny positive number for log-scale

# --- semi-log plot (log y axis) ---
# If C can be negative at some distances, log-scale won't work directly.
# Common choice: plot abs(C) on log scale.
p = plot(ds, log10.(y);
    marker = :circle,
    ylims = (-3.5,1),
    yticks = -3:1:1,
    xlabel = "Distance dx",
    ylabel = "log10|C(dx,0)|"
)
#Label the yticks and xticks
#yticks!(p, -3:1:1, string.(10 .^ (-3.0:1.0:1.0)))
#xticks!(p, 0:1:dmax)
# Optional error bars (on |C|). For small errors relative to |C| this is fine.
#plot!(p, ds, y; yerror = dCs, label = "")

display(p)



"""
    bond_bond_corr_thermal_grid!(
        bc::Bonds,
        rng::AbstractRNG,
        dmax::Int;
        orientation::Symbol = :both,
        connected::Bool = false,
        burnin::Int = 10_000,
        nsamples::Int = 2_000,
        thin::Int = 10,
        mc_step! = MC_T0_loop!,
    )

Runs ONE MCMC trajectory and, for each sampled configuration, computes C(dx,dy)
for all dx,dy ∈ 0:dmax. Returns matrices (dmax+1)×(dmax+1):

    Cmean[dx+1,dy+1],  Cstderr[dx+1,dy+1],  npairs_mean[dx+1,dy+1]

Compatible with mc_step!(bc, rng, δB_0s) where δB_0s = δB_0_tuple(bc).
"""
function bond_bond_corr_thermal_grid!(
    bc::Bonds,
    rng::AbstractRNG,
    dmax::Int;
    orientation::Symbol = :both,
    connected::Bool = false,
    burnin::Int = 10_000,
    nsamples::Int = 2_000,
    thin::Int = 10,
    mc_step! = MC_T0_loop!,
)
    # Allowed starting ΔB values for this bond_config (per your new API)
    δB_0s = δB_0_tuple(bc)
    if isempty(δB_0s)
        n = dmax + 1
        return fill(NaN, n, n), fill(NaN, n, n), fill(0.0, n, n)
    end

    n = dmax + 1

    # Online mean/variance accumulators for each (dx,dy)
    meanC = zeros(Float64, n, n)
    M2C   = zeros(Float64, n, n)     # sum of squared deviations
    npair_acc = zeros(Float64, n, n) # accumulate npairs per snapshot

    # Burn-in
    for _ in 1:burnin
        mc_step!(bc, rng, δB_0s)
    end

    # Sampling loop: one chain, many observables per snapshot
    for s in 1:nsamples
        for _ in 1:thin
            mc_step!(bc, rng, δB_0s)
        end

        # Compute ALL offsets on this snapshot
        for dx in 0:dmax
            for dy in 0:dmax
                Csnap, np = bond_bond_corr_snapshot(
                    bc, dx, dy;
                    orientation = orientation,
                    connected = connected,
                )

                i = dx + 1
                j = dy + 1

                # Welford update for variance per entry
                δ = Csnap - meanC[i, j]
                meanC[i, j] += δ / s
                M2C[i, j]   += δ * (Csnap - meanC[i, j])

                npair_acc[i, j] += np
            end
        end
    end

    # Finalize stderr and mean npairs
    Cmean = meanC
    Cstderr = similar(Cmean)
    if nsamples > 1
        varC = M2C ./ (nsamples - 1)           # sample variance
        Cstderr .= sqrt.(varC ./ nsamples)     # naive stderr (same as your scalar version)
    else
        Cstderr .= NaN
    end
    npairs_mean = npair_acc ./ nsamples

    return Cmean, Cstderr, npairs_mean
end

"""
    radialize_corr(Cmean, Cstderr; dmax)

Convert (dx,dy)-grid correlator statistics into radial shells keyed by r = sqrt(dx^2+dy^2).

Returns:
    rs, Cr, dCr, nr
where nr is the number of (dx,dy) offsets in each shell.
"""
function radialize_corr(Cmean::AbstractMatrix, Cstderr::AbstractMatrix; dmax::Int)
    shells = Dict{Int, Vector{Tuple{Float64,Float64}}}()  # r2 => [(C, dC), ...]

    for dx in 0:dmax
        for dy in 0:dmax
            r2 = dx*dx + dy*dy
            push!(get!(shells, r2, Tuple{Float64,Float64}[]),
                  (Cmean[dx+1, dy+1], Cstderr[dx+1, dy+1]))
        end
    end

    r2s = sort(collect(keys(shells)))
    rs  = Float64[]
    Cr  = Float64[]
    dCr = Float64[]
    nr  = Int[]

    for r2 in r2s
        vals = shells[r2]
        Cs   = first.(vals)
        dCs  = last.(vals)

        # Inverse-variance weighted average across offsets in the shell
        good = all(isfinite.(dCs)) && all(dCs .> 0)
        if good
            w = 1.0 ./ (dCs .^ 2)
            Cbar = sum(w .* Cs) / sum(w)
            dCbar = sqrt(1.0 / sum(w))
        else
            Cbar = mean(Cs)
            dCbar = std(Cs) / sqrt(length(Cs))
        end

        push!(rs, sqrt(r2))
        push!(Cr, Cbar)
        push!(dCr, dCbar)
        push!(nr, length(vals))
    end

    return rs, Cr, dCr, nr
end
# --- parameters ---
dmax     = 7                 # for Lx=7, the largest meaningful dx is 6
nsamples = 10_00_000
thin     = 20

Cmean, Cstderr, npairs = bond_bond_corr_thermal_grid!(bond_config, rng, dmax;
    orientation = :h,
    nsamples = nsamples,
    thin = thin,
    connected = false,
)

# # Example: recover your old loop output for dy = 0
# Plot Cmean(dx,0) with error bars (dy = 0 is column 1)
ds = collect(0:dmax)
Cs_dx = Cmean[:, 1]
dCs_dx = Cstderr[:, 1]

p = plot(ds, log10.(abs.(Cs_dx));
         yerror = dCs_dx,
         marker = :circle,
         xlabel = "dx",
         ylabel = "Cmean(dx, 0)",
         title  = "Bond–bond correlator C(dx,0)",
         legend = false)
display(p)

# Plot C(dx,0) for several N = 2..6 (overlayed)
lattice = Lattice(20,20)
dmax = 8                   # radial range used earlier
ds = collect(0:dmax)

# Sampling parameters (adjust for runtime / precision)
burnin = 1_000
nsamples_plot = 1_000
thin_plot = 20

p = plot(marker = :circle,
         xlabel = "dx",
         ylabel = "log10|C(dx,0)|",
         title = "Bond–bond correlator C(dx,0) for various N",
         legend = :topright)

for N in 2:6
    bc = Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly))
    rngN = MersenneTwister(1234 + N)

    # compute grid and take dy=0 column (dy index 1)
    Cmean, Cstderr, npairs = bond_bond_corr_thermal_grid!(
        bc, rngN, dmax;
        orientation = :h,
        connected   = false,
        burnin      = burnin,
        nsamples    = nsamples_plot,
        thin        = thin_plot,
        mc_step!    = MC_T0_loop!,
    )

    Cs_dx = Cmean[:, 1]
    # avoid log10(0) by adding tiny epsilon
    plot!(p, ds, log10.(abs.(Cs_dx/N^2) .+ eps(Float64)), label = "N=$N")
end

display(p)



function make_plot()
    lattice = Lattice(20,20)
    dmax = 8
    ds = collect(0:dmax)

    burnin = 1_000
    nsamples_plot = 1_0000
    thin_plot = 20

    p = plot(marker = :circle,
             xlabel = "dx",
             ylabel = "log10|C(dx,0)|",
             title = "Bond–bond correlator C(dx,0) for various N",
             legend = :topright)

    for N in 2:6
        bc = Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly))
        rngN = MersenneTwister(1234 + N)

        Cmean, Cstderr, npairs = bond_bond_corr_thermal_grid!(
            bc, rngN, dmax;
            orientation = :h,
            connected   = false,
            burnin      = burnin,
            nsamples    = nsamples_plot,
            thin        = thin_plot,
            mc_step!    = MC_T0_loop!,
        )

        Cs_dx = @view Cmean[:, 1]
        plot!(p, ds, log10.(abs.(Cs_dx ./ N^2) .+ eps(Float64)), label = "N=$N")
    end

    return p
end
make_plot();  # warmup
@time make_plot();

rs, Cr, dCr, nr = radialize_corr(Cmean, Cstderr; dmax=dmax)
plot(rs,log10.(abs.(Cr)), marker= :circle)
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


#Count up the number of configurations for N=3 on a LxL lattice
lattice=Lattice(3,3)
N=2 #max_bond value
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