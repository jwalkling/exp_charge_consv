"""
Testing correlations between the bonds
Created: 28.01.2026
"""

include("/Users/jamwalk/Desktop/Research/Roderich/Discrete CSL/exp_charge_consv/ecc_func.jl")



"""
    paired_diagonal_bonds(indexc::Int; inc::Int, max_index::Int)

Return two vectors of bond indices along the line x=-y
- `same`: indices with the same orientation (parity) as `indexc`
- `other`: paired indices with opposite orientation (index±1)

Assumes orientation is encoded by parity: odd vs even.
In minus xy direction
"""
function paired_diagonal_bonds_xym(indexc::Int; max_index::Int)
    inc=2L-2
    # paired bond of opposite orientation
    indexc2 = isodd(indexc) ? indexc + 1 : indexc - 1

    same  = Int[]
    other = Int[]

    # helper: append current pair if in bounds
    @inline function maybe_push!(p1::Int, p2::Int)
        if 1 <= p1 <= max_index && 1 <= p2 <= max_index
            if p1 != indexc
                push!(same, p1)
            end
            push!(other, p2)
            return true
        end
        return false
    end

    # negative direction (excluding center to avoid duplicates)
    p1, p2 = indexc - inc, indexc2 - inc
    while maybe_push!(p1, p2)
        p1 -= inc
        p2 -= inc
    end
    
    maybe_push!(indexc, indexc2)  # include the center bond itself in the "other" list

    # positive direction (excluding center to avoid duplicates)
    p1, p2 = indexc + inc, indexc2 + inc
    while maybe_push!(p1, p2)
        p1 += inc
        p2 += inc
    end

    return same, other
end


function paired_bonds(indexc::Int; inc::Int, max_index::Int)
    same  = Int[]

    # helper: append current pair if in bounds
    @inline function maybe_push!(p1::Int)
        if 1 <= p1 <= max_index
            if p1 != indexc
                push!(same, p1)
            end
            return true
        end
        return false
    end

    # negative direction (excluding center to avoid duplicates)
    p1 = indexc - inc
    while maybe_push!(p1)
        p1 -= inc
    end
    
    maybe_push!(indexc)  # include the center bond itself in the "other" list

    # positive direction (excluding center to avoid duplicates)
    p1=indexc + inc
    while maybe_push!(p1)
        p1 += inc
    end

    return same #Return bonds of the same kind. Works in general. 
end


#-----------------------------
# Find the average bond values
#-----------------------------
L=8
N=2
lattice = Lattice(L,L)
bc = construct_bonds(lattice, N)
rng = MersenneTwister(1234)
δB_0s=δB_0_tuple(bond_config)

iterations=100_000
average_bond = zeros(Float64, 2*lattice.Lx*lattice.Ly)

for i in 1:iterations
    MC_T0_loop!(bc, rng, δB_0s)
    for b in 1:(2*lattice.Lx*lattice.Ly)
        average_bond[b] += bc.bond[b]
    end
end

average_bond ./= iterations
plot(average_bond) #see that the average gets smaller for a large number of iterations.


#-----------------------------------
# Bond-Bond Correlator in Real Space
#-----------------------------------

function plot_bond_corr_realspace(
    bc::Bonds,
    b0::Int,
    meanC::AbstractVector;
    cmap = :RdBu,
    lw::Real = 4,
    clim = nothing,
    show_ref::Bool = true,
    shrink::Real = 0.15,                 # new: fraction to cut off each end
    title::Union{Nothing,String} = nothing,
)
    b = bc.bond
    lat = bc.lattice
    Lx, Ly = lat.Lx, lat.Ly
    Nsites = Lx * Ly

    length(meanC) == length(b) || throw(DimensionMismatch("meanC must have length length(bc.bond)"))
    (1 <= b0 <= length(b)) || throw(ArgumentError("b0 must satisfy 1 ≤ b0 ≤ length(bc.bond)"))
    (0 ≤ shrink < 0.5) || throw(ArgumentError("shrink must satisfy 0 ≤ shrink < 0.5"))

    @inline site_of_bond(bi::Int) = (bi + 1) >>> 1
    @inline is_hbond(bi::Int)     = isodd(bi)
    @inline x_of_site(i::Int)     = ((i - 1) % Lx) + 1
    @inline y_of_site(i::Int)     = ((i - 1) ÷ Lx) + 1

    # helper: shrink [a,b] toward midpoint by `shrink` on each side
    @inline function shrink_seg(a::Real, b::Real, s::Real)
        d = b - a
        return (a + s*d, b - s*d)
    end

    xs = Float64[]; ys = Float64[]; zs = Float64[]

    @inbounds for i in 1:Nsites
        x = x_of_site(i)
        y = y_of_site(i)

        if x < Lx
            bi = 2i - 1
            C  = meanC[bi]
            x1, x2 = shrink_seg(x, x+1, shrink)
            append!(xs, (x1, x2, NaN))
            append!(ys, (y,  y,  NaN))
            append!(zs, (C,  C,  NaN))
        end

        if y < Ly
            bi = 2i
            C  = meanC[bi]
            y1, y2 = shrink_seg(y, y+1, shrink)
            append!(xs, (x,  x,  NaN))
            append!(ys, (y1, y2, NaN))
            append!(zs, (C,  C,  NaN))
        end
    end

    if clim === nothing
        m = maximum(abs, meanC)
        clim = (-m, m)
    end

    p = plot(xs, ys;
        seriestype   = :path,
        line_z       = zs,
        c            = cmap,
        clim         = clim,
        linewidth    = lw,
        aspect_ratio = :equal,
        xlabel       = "x",
        ylabel       = "y",
        colorbar     = true,
        legend       = false,
        title        = title === nothing ? "Bond–bond correlator map, b0=$b0" : title,
    )

    if show_ref
        i0 = site_of_bond(b0)
        x0, y0 = x_of_site(i0), y_of_site(i0)

        if is_hbond(b0)
            x0 < Lx || throw(ArgumentError("b0=$b0 is a horizontal bond on the right boundary (invalid link)."))
            x1, x2 = shrink_seg(x0, x0+1, shrink)
            plot!(p, [x1, x2], [y0, y0]; linewidth=lw+2, color=:black, label="")
        else
            y0 < Ly || throw(ArgumentError("b0=$b0 is a vertical bond on the top boundary (invalid link)."))
            y1, y2 = shrink_seg(y0, y0+1, shrink)
            plot!(p, [x0, x0], [y1, y2]; linewidth=lw+2, color=:black, label="")
        end
    end

    return p
end

#----------------------------------
# Import data for larger iterations
#----------------------------------

Ns = [2,6,10,14,20]
Cdict  = Dict{Int, Matrix{Float64}}()
directory = "../ECC_data/T=0/Bond_C_Ns/it=10^9L=20/"

for N in Ns
    Cfile  = joinpath(directory, "Cmat_N$(N).csv")

    dfC  = CSV.read(Cfile,  DataFrame)

    Cmat  = Matrix{Float64}(dfC)

    Cdict[N]  = Cmat
end

for N in Ns
    L=20
    lattice = Lattice(L,L)
    indexc=Int((L-1)*L) #Comparison index of the bond
    bc = construct_bonds(lattice, N)

    Cbonds=Cdict[N][indexc, :]
    p = plot_bond_corr_realspace(bc, indexc, log10.(abs.(Cbonds)); clim=(-4,1.05), shrink=0.3, title="Bond–bond correlator map for N=$(N), L=20, it=10^9")
    display(p)
    savefig(p, joinpath(homedir(), "Downloads",  "BondCorr_N$(N)_L20_it1e9.png"))
end


#-----------------------------------------
# Plot correlators in the direction -x=y
#-----------------------------------------
# Get the list of bond indices along the -x=y direction for the reference bond at indexc
L=20
lattice = Lattice(L,L)
indexc=Int((L-1)*L)#Int((L-1)*L) #Comparison index of the bond (Int(2L))

#Find the paired vertical/horizontal bond index
same, other = paired_diagonal_bonds_xym(indexc; max_index=2L*L)



N=2
bc = construct_bonds(lattice, N)

Cbonds=Cdict[N][indexc, :]

rs=[]
Cm=[]
for bond in same
    push!(rs, bond_distance(lattice, indexc, bond))
    push!(Cm,Cbonds[bond])
end
p = scatter(rs,(abs.(Cm)./N^2).+10^(-10), title="Correlator along -x=y for N=$(N), L=20, it=10^9",
    xscale=:log10,
    yscale=:log10)
ylims!(p, 10^(-3), 10^(-1.5))
xlabel!(p, "x")
ylabel!(p, "C(x,-x)")
display(p)
savefig(p, joinpath(homedir(), "Downloads",  "BondCorr_N$(N)_diag-.png"))

N=20
bc = construct_bonds(lattice, N)

Cbonds=Cdict[N][indexc, :]

rs=[]
Cm=[]
for bond in same
    push!(rs, bond_distance(lattice, indexc, bond))
    push!(Cm,Cbonds[bond])
end
p = scatter(rs,(abs.(Cm)./N^2).+10^(-10), title="Correlator along -x=y for N=$(N), L=20, it=10^9",
    #xscale=:log10,
    yscale=:log10)
ylims!(p, 10^(-3.5), 10^(-1.5))
xlabel!(p, "x")
ylabel!(p, "C(x,-x)")
display(p)
savefig(p, joinpath(homedir(), "Downloads",  "BondCorr_N$(N)_diag-.png"))

#-------------------------------
# Plot correlators in the direction x=y
#-------------------------------
L=20
lattice = Lattice(L,L)
indexc=1#Int((L-1)*L) #Int((L-1)*L) #Comparison index of the bond

#Find the paired vertical/horizontal bond index
same = paired_bonds(indexc; inc=2*L+2, max_index=2L*L)



N=2
bc = construct_bonds(lattice, N)

Cbonds=Cdict[N][indexc, :]

rs=[]
Cm=[]
for bond in same
    push!(rs, bond_distance(lattice, indexc, bond))
    push!(Cm,Cbonds[bond])
end
p = scatter(rs,(abs.(Cm)./N^2).+10^(-10), title="Correlator along x=y for N=$(N), L=20, it=10^9",
    yscale=:log10)
ylims!(p, 10^(-5), 10^(-1))
display(p)
savefig(p, joinpath(homedir(), "Downloads",  "BondCorr_N$(N)_diag+.png"))



N=20
bc = construct_bonds(lattice, N)

Cbonds=Cdict[N][indexc, :]

rs=[]
Cm=[]
for bond in same
    push!(rs, bond_distance(lattice, indexc, bond))
    push!(Cm,Cbonds[bond])
end
p = scatter(rs,(abs.(Cm)./N^2).+10^(-10), title="Correlator along x=y for N=$(N), L=20, it=10^9",
    yscale=:log10)
ylims!(p, 10^(-6), 10^(-1))
display(p)
savefig(p, joinpath(homedir(), "Downloads",  "BondCorr_N$(N)_diag+.png"))



#--------------------------------
# Testing loop length cutoff
#--------------------------------

L=10
N=4

lattice = Lattice(L,L)
bc = construct_bonds(lattice, N)

rng = MersenneTwister(1234)
δB_0s=δB_0_tuple(bc)

m=10^6

#Time the code without the cutoff to run m loops
t1=time()
bc = construct_bonds(lattice, N)
for i in 1:m
    MC_T0_loop!(bc, rng, δB_0s)
end

#Time the code with a cutoff of L to run m loops
t2=time()
bc = construct_bonds(lattice, N)
for i in 1:m
    MC_T0_loop_cutoff!(bc, rng, δB_0s, L)
end
t3=time()
println("Time without cutoff: ", t2-t1, " seconds")
println("Time with cutoff: ", t3-t2, " seconds")


using Random
using Statistics
using LinearAlgebra
using Plots

# ============================================================
# settings
# ============================================================

L = 10
N = 4
lattice = Lattice(L, L)

ms = unique(round.(Int, 10 .^ range(1, 5; length=12)))   # up to 10^4
rngseed = 1234
time_budget = 1.0   # seconds for convergence test

# ============================================================
# helper: runtime scan
# ============================================================

function runtime_scan(ms; L=10, N=4, rngseed=1234)
    lattice = Lattice(L, L)

    times_plain  = zeros(Float64, length(ms))
    times_cutoff = zeros(Float64, length(ms))

    for (k, m) in enumerate(ms)

        # plain
        rng = MersenneTwister(rngseed)
        bc = construct_bonds(lattice, N)
        δB_0s = δB_0_tuple(bc)

        t1 = time()
        @inbounds for _ in 1:m
            MC_T0_loop!(bc, rng, δB_0s)
        end
        times_plain[k] = time() - t1

        # cutoff
        rng = MersenneTwister(rngseed)
        bc = construct_bonds(lattice, N)
        δB_0s = δB_0_tuple(bc)

        t2 = time()
        @inbounds for _ in 1:m
            MC_T0_loop_cutoff!(bc, rng, δB_0s, L)
        end
        times_cutoff[k] = time() - t2
    end

    return times_plain, times_cutoff
end

# ============================================================
# helper: convergence of running bond average to zero
# ============================================================

"""
Run for approximately `time_budget` seconds.
After each MC step, update the cumulative bond sum:

    running_sum += bc.bond

and measure closeness of the running average to zero via

    sum(abs.(running_sum / nsteps))

Returns:
- closeness trace
- number of steps performed
"""
function convergence_fixed_time_plain(; L=10, N=4, rngseed=1234, time_budget=1.0)
    lattice = Lattice(L, L)
    rng = MersenneTwister(rngseed)

    bc = construct_bonds(lattice, N)
    δB_0s = δB_0_tuple(bc)

    running_sum = zeros(Float64, length(bc.bond))
    closeness = Float64[]

    nsteps = 0
    t0 = time()
    while (time() - t0) < time_budget
        MC_T0_loop!(bc, rng, δB_0s)
        nsteps += 1

        running_sum .+= bc.bond
        push!(closeness, sum(abs.(running_sum)) / nsteps)   # L1 norm of running average
    end

    return closeness, nsteps
end


function convergence_fixed_time_cutoff(; L=10, N=4, rngseed=1234, time_budget=1.0)
    lattice = Lattice(L, L)
    rng = MersenneTwister(rngseed)

    bc = construct_bonds(lattice, N)
    δB_0s = δB_0_tuple(bc)

    running_sum = zeros(Float64, length(bc.bond))
    closeness = Float64[]

    nsteps = 0
    t0 = time()
    while (time() - t0) < time_budget
        MC_T0_loop_cutoff!(bc, rng, δB_0s, L)
        nsteps += 1

        running_sum .+= bc.bond
        push!(closeness, sum(abs.(running_sum)) / nsteps)   # L1 norm of running average
    end

    return closeness, nsteps
end

# ============================================================
# 1. runtime scaling
# ============================================================

times_plain, times_cutoff = runtime_scan(ms; L=L, N=N, rngseed=rngseed)

p1 = plot(
    ms, times_plain;
    xscale = :log10,
    yscale = :log10,
    marker = :circle,
    lw = 2,
    xlabel = "m",
    ylabel = "runtime (s)",
    label = "MC_T0_loop!",
    title = "Runtime scaling"
)

plot!(
    p1,
    ms, times_cutoff;
    marker = :square,
    lw = 2,
    label = "MC_T0_loop_cutoff!"
)

display(p1)

println("Runtime scan:")
for i in eachindex(ms)
    println("m = $(ms[i]): plain = $(times_plain[i]) s, cutoff = $(times_cutoff[i]) s, speedup = $(times_plain[i] / times_cutoff[i])")
end

# ============================================================
# 2. convergence over fixed runtime
# ============================================================

closeness_plain, n_plain = convergence_fixed_time_plain(; L=L, N=N, rngseed=rngseed, time_budget=time_budget)
closeness_cutoff, n_cutoff = convergence_fixed_time_cutoff(; L=L, N=N, rngseed=rngseed, time_budget=time_budget)

p2 = plot(
    1:length(closeness_plain), log10.(closeness_plain);
    lw = 2,
    xlabel = "MC step",
    ylabel = "sum(abs.(running average of bc.bond))",
    label = "MC_T0_loop!",
    title = "Convergence to zero over fixed wall-clock time = $(time_budget) s"
)

plot!(
    p2,
    1:length(closeness_cutoff), log10.(closeness_cutoff);
    lw = 2,
    label = "MC_T0_loop_cutoff!"
)

display(p2)

println("\nConvergence test:")
println("plain steps performed  = $n_plain")
println("cutoff steps performed = $n_cutoff")
println("final closeness plain  = $(closeness_plain[end])")
println("final closeness cutoff = $(closeness_cutoff[end])")









#-------------------------------
# Testing the coordinates function
#-------------------------------
p = plot()
for i in 1:(2*L^2)
    x, y = index_to_coord(lattice, i)
    scatter!(p, [x], [y])   # vectors, so Plots knows it's one point
end
display(p)


index=3
vertex=Int((index+1)/2) #site index
vx = ((vertex - 1) % 4)
vy = ((vertex - 1) ÷ 4)
println("Vertex: ", vertex, " vx: ", vx, " vy: ", vy)