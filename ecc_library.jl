```
Library of functions for Exponential Charge Conservation
Created: 04.12.2025
```

using Plots
using Random
using BenchmarkTools

#-------------------------
# Define Lattice Structure
#-------------------------

#Struct that stores the lattie properties
struct Lattice
    #Dimensions of the lattice
    Lx::Int
    Ly::Int
end

struct Bonds
    lattice::Lattice
    max_bond::Int
    bond::Vector{Int}  #Bond variables
end

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


#Efficient random move generation using bools
@inline function rand_step(lat::Lattice, rng::AbstractRNG)
    axis = rand(rng, Bool)           # false → ±1, true → ±Ly
    sign = rand(rng, Bool) ? 1 : -1
    step = axis ? lat.Lx : 1
    return sign * step
end



#Function to grab the bond label between two indices.
#TK need to be careful with this function, can give a wrong bond
# if indices are not nearest neighbours.
@inline function bond_label(index_prev::Int, index_curr::Int)
    i = index_prev < index_curr ? index_prev : index_curr
    Δ = index_curr - index_prev

    # |Δ| == 1  → vertical → 2i-1
    # |Δ| >  1  → horizontal → 2i
    return (abs(Δ) == 1) ? (2i - 1) : (2i), Δ
end

#Takes the effective Δ as input (PBC adjusted)
#Returns value for charge neutrality
@inline function multiplier(Δ_curr, Δ_prev)
    var=sign(Δ_curr)
    if abs(Δ_curr) != abs(Δ_prev) && var != sign(Δ_prev)
        return 
    else
        return 2.0^var
    end
end

#Return a random allowed move
@inline function allowed_step_first(δB_firstmove::Float64, bond_config::Bonds, index_curr::Int)
    Lx = bond_config.lattice.Lx
    Ly = bond_config.lattice.Ly
    bonds = bond_config.bond

    steps = [-1, 1, -Lx, Lx]
    shuffle!(steps)

    for step in steps
        if !in_bounds(index_curr, step, Lx, Ly)
            continue
        end

        index_next = index_curr + step
        bond = bond_label(index_curr, index_next)[1]

        if abs(bonds[bond] + δB_firstmove) <= bond_config.max_bond &&
           abs(δB_firstmove) >= 1
            return step
        end
    end

    error("No allowed steps for first worm move")
end

@inline function allowed_step(δB::Float64, bond_config::Bonds, index_curr::Int, index_prev::Int)
    Lx      = bond_config.lattice.Lx
    Δ_prev  = index_curr - index_prev
    bonds   = bond_config.bond
    steps   = [-1, 1, -Lx, Lx]

    shuffle!(steps) # Randomise order of steps

    for step in steps # Accept the first valid step
        # no backtracking
        if step == -Δ_prev
            continue
        end

        # open boundary conditions
        if !in_bounds(index_curr, step, Lx, bond_config.lattice.Ly)
            continue
        end

        # Physical constraint on bond variables
        index_next = index_curr + step
        Δ_curr     = step
        δB_curr    = δB * multiplier(Δ_curr, Δ_prev)
        bond       = bond_label(index_curr, index_next)[1]

        # Can't exceed max bond, and must be integer.
        if abs(bonds[bond] + δB_curr) <= bond_config.max_bond && abs(δB_curr) >= 1
            println("Allowed step: ", step, "\n")
            println("  from index ", index_curr, " to ", index_next, "\n")
            println(" bond ", bond, ": current value ", bonds[bond], ", δB_curr=", δB_curr, "\n")
            return step, bond, δB_curr
        end
    end

    error("No allowed steps for (curr=$index_curr, prev=$index_prev)")
end

function MC_T0_loop!(bond_config::Bonds, rng::AbstractRNG)
    #Count total number of sites
    lattice=bond_config.lattice
    Lx=lattice.Lx
    Ly=lattice.Ly
    Nsites=lattice.Lx*lattice.Ly
    
    #INITIALISE
    rng = MersenneTwister(1234) #Choose rng
    #Pick a random vertex starting point on the grid
    index_0 = rand(1:Nsites)

    #Value to change spin by
    δB_0=2.0 #TK this can be modified
    δB_prev = δB_0

    #Pick a direction to move
    move_0=allowed_step_first(δB_0, bond_config, index_0)
    println("Initial move: ", move_0, "\n")
    println("  from index ", index_0, " to ", index_0 + move_0, "\n")
    index_curr=index_0+move_0
    index_prev=index_0

    bond_prev, Δmove=bond_label(index_curr, index_prev) #bond label
    bond_config.bond[bond_prev]+=δB_prev
    Δmove_prev=Δmove

    while index_curr != index_0
        #Pick a new direction to move
        Δmove, bond_curr, δB_curr=allowed_step(δB_prev, bond_config, index_curr, index_prev)

        #Moves with PBCs included
        index_prev=index_curr
        index_curr=index_curr+Δmove

        print(index_prev, " to ", index_curr, ", δB_curr=", δB_curr, "\n")
        bond_config.bond[bond_curr]+=δB_curr #Update bond variable

        #Update variables for next loop
        Δmove_prev=Δmove
        δB_prev=δB_curr
    end

    # bond_curr, Δmove=bond_label(index_curr, index_prev) #bond label
    # δB_curr=δB_prev*multiplier(Δmove,Δmove_prev)
    # bond_config.bond[bond_curr]+=δB_curr
end


#-------------------------
# Running the Monte Carlo
#-------------------------

lattice=Lattice(10,10)
N=10 #max_bond value
bond_config=Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly))

MC_T0_loop!(bond_config, MersenneTwister(1234))
plot_bondsnv(bond_config)

direc= allowed_step_first(1.0, bond_config, 14)
#bond_config.bond[8]=2



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
# Checks on code
#-------------------------

# Check to see the lattice plots correctly
lattice_1=Lattice(10,10)
p=plot(legend=false, aspect_ratio=1,title="Lattice Link Centres")
for index in 1:(lattice_1.Lx*lattice_1.Ly)
    coord = index_to_coord(lattice_1, index)
    scatter!(p,coord, color=:blue, markersize=4)
    #println("Index: $index -> Coord: $coord")
end
display(p)


lattice_1 = Lattice(7,7)

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
