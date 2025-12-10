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
    bond::Vector{Int}  #Bond variables
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

@inline function allowed_step(δB::Float, bond_config::Bonds, rng::AbstractRNG)
    #Find all the values that result from adding δB


    #Choose a move randomly
end

@inline function step_index(index_curr::Int, move::Int, Lx::Int, Ly::Int)
    if index_curr % Lx == 0 && move == 1 #Right edge moving right
        return index_curr - Lx + 1
    elseif (index_curr - 1) % Lx == 0 && move == -1 #Left edge moving left
        return index_curr + Lx - 1
    elseif index_curr <= Lx && move == -Lx #Bottom edge moving down
        return index_curr + Lx*(Ly-1)
    elseif index_curr >= Lx*(Ly-1)+1 && move == Lx #Top edge moving up
        return index_curr - Lx*(Ly-1)
    else
        return index_curr + move
    end
end


#Function to grab the bond label between two indices.
#TK need to be careful with this function, can give a wrong bond
# if indices are not nearest neighbours.
@inline function bond_between(index_prev::Int, index_curr::Int)
    i = index_prev < index_curr ? index_prev : index_curr
    Δ = index_curr - index_prev

    # |Δ| == 1  → vertical → 2i-1
    # |Δ| >  1  → horizontal → 2i
    return (abs(Δ) == 1) ? (2i - 1) : (2i)
end

@inline function bond_PBC(index_prev::Int, index_curr::Int, Lx::Int)
    Δ   = index_curr - index_prev
    aΔ  = abs(Δ)
    i_min = index_prev < index_curr ? index_prev : index_curr
    i_max = index_prev > index_curr ? index_prev : index_curr
    
    if aΔ == 1             # x hop (internal)      → use min
        return 2*i_min - 1, Δ
    elseif aΔ == Lx        # y hop (internal)      → use min
        return 2*i_min, Δ
    #For the PBC cases, return effective delta.
    elseif aΔ == Lx - 1    # x hop (PBC)           → use max
        return 2*i_max - 1, -sign(Δ)
    else                   # y hop (PBC)           → use max
        return 2*i_max, -sign(Δ)
    end
end

#Takes the effective Δ as input (PBC adjusted)
#Returns value for charge neutrality
@inline function multiplier(Δ_curr, Δ_prev)
    var=sign(Δ_curr)
    if abs(Δ_curr) == abs(Δ_prev)
        return 2.0^var
    else
        return -1
    end
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

    #Pick a direction to move
    move_0=rand_step(lattice, rng)

    index_curr=step_index(index_0,move_0,Lx,Ly)
    index_prev=index_0

    #Value to change spin by
    δB_0=1 #TK this can be modified
    δB_prev = δB_0

    bond_prev, Δ_eff=bond_PBC(index_curr, index_prev, Lx) #bond label
    bond_config.bond[bond_prev]+=δB_prev
    Δ_eff_prev=Δ_eff
    k=0
    while k < 100#index_curr != index_0
        #Pick a new direction to move
        move=rand_step(lattice, rng) #TK need to fix that we don't bounce back.
        
        #Moves with PBCs included
        index_prev=index_curr
        index_curr=step_index(index_curr,move,Lx,Ly)

        bond_curr, Δ_eff_curr=bond_PBC(index_curr, index_prev, Lx) #bond label
        δB_curr=δB_prev*multiplier(Δ_eff_curr,Δ_eff_prev)
        bond_config.bond[bond_curr]+=δB_curr #Update bond variable

        #Update variables for next loop
        Δ_eff_prev=Δ_eff_curr
        δB_prev=δB_curr
        k+=1
    end

    bond_curr, Δ_eff_curr=bond_PBC(index_curr, index_prev, Lx) #bond label
    δB_curr=δB_prev*multiplier(Δ_eff_curr,Δ_eff_prev)
    bond_config.bond[bond_curr]+=δB_curr
end


#-------------------------
# Running the Monte Carlo
#-------------------------

lattice=Lattice(10,10)
bond_config=Bonds(lattice, zeros(Int, 2*lattice.Lx*lattice.Ly))

MC_T0_loop!(bond_config, MersenneTwister(1234))




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
