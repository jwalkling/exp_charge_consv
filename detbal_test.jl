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

#Added little simple line to count the steps and return Nsteps.
function MC_T0_loop_size!(bond_config::Bonds, rng::AbstractRNG, δB_0s::Tuple{Vararg{Float64}})
    lat    = bond_config.lattice
    Lx     = lat.Lx
    Ly     = lat.Ly
    Nsites = Lx * Ly

    Nsteps=0
    index_0 = rand(rng, 1:Nsites)

    δB_0    = rand(rng, δB_0s)
    δB_prev = δB_0

    move_0 = allowed_step_first(δB_0, bond_config, index_0, rng)
    if move_0 == 0
        return 0
    end
    
    # apply first move
    index_prev = index_0
    index_curr = index_0 + move_0

    bond0 = step_bond(index_prev, move_0)
    bond_config.bond[bond0] += δB_prev
    Nsteps +=1

    while index_curr != index_0
        step, bond, δB_curr = allowed_step(δB_prev, bond_config, rng, index_curr, index_prev)

        index_prev = index_curr
        index_curr = index_curr + step

        bond_config.bond[bond] += δB_curr
        δB_prev = δB_curr
        Nsteps+=1
    end
    return Nsteps
end

#-----------------------------
# Test probability of making single move
#-----------------------------

L=8
N=4
lattice = Lattice(L,L)
bc = Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly), zeros(Int, lattice.Lx*lattice.Ly))
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


#-----------------------------
# Test distribution of loop sizes
#-----------------------------
L=100 #8 #20 used later
N=2 #4
lattice = Lattice(L,L)
bc = Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly), zeros(Int, lattice.Lx*lattice.Ly))
rng = MersenneTwister() #Random.default_rng() #MersenneTWister(1235)
δB_0s=δB_0_tuple(bc)

iterations=10000000
loop_lengths=[]
count=0
for i in 1:iterations
    LengthLoop=MC_T0_loop_size!(bc, rng, δB_0s)
    #push!(loop_lengths,LengthLoop)
    if LengthLoop>0
        push!(loop_lengths,LengthLoop)
        count+=1
    end
    if i % 100000 == 0
        println(i/iterations)
        println("Iteration: ", i, " Loop Length: ", LengthLoop, " Count: ", count)
    end
end

power=1/4
p=histogram(loop_lengths.^power, nbins=1000, xlabel="Loop Length", ylabel="Frequency", yscale=:log10, title="Distribution of Loop Lengths")
xlims!(p,10^power, 20000^power)

maxL = maximum(loop_lengths)
freq = zeros(Int, maxL + 1)

for ℓ in loop_lengths
    freq[ℓ÷2 + 1] += 1
end

length_values = 0:maxL
maxval=400
p=scatter(length_values[2:maxval], freq[2:maxval], xlabel="Loop Length", ylabel="Frequency", title="Distribution of Loop Lengths", 
    yscale=:log10,
    xscale=:log10)

using Plots
using LsqFit
using Statistics
# -------------------------
# Inputs you already have:
#   length_values = 0:maxL
#   freq::Vector{<:Integer}  (same length as length_values)
# -------------------------

# ===== User knobs (set these) =====
maxval = 500          # how far to plot (in terms of length index/value)
ℓmin  = 5             # lower cutoff for fitting (exclude microscopic regime)
ℓmax  = min(maxval-1, 300)  # upper cutoff for fitting (exclude finite-size tail if needed)

# ===== Prepare data (drop zeros, apply window) =====
# We fit only lengths ℓ in [ℓmin, ℓmax] with freq>0 and ℓ>=1 (log scale).
mask = (length_values .>= 1) .& (length_values .>= ℓmin) .& (length_values .<= ℓmax) .& (freq .> 0)

ℓ = Float64.(collect(length_values[mask]))
y = Float64.(freq[mask])

# ===== Model 1: pure power law y = A * ℓ^(-τ) =====
model_pow(ℓ, p) = p[1] .* ℓ.^(-p[2])          # p = [A, τ]

# good initial guess from log-log slope
logℓ = log.(ℓ)
logy = log.(y)
τ0 = - (cov(logℓ, logy) / var(logℓ))         # slope = cov/var
A0 = exp(mean(logy) + τ0 * mean(logℓ))

p0_pow = [A0, τ0]
fit_pow = curve_fit(model_pow, ℓ, y, p0_pow)
A_pow, τ_pow = fit_pow.param

# ===== Model 2: power law with exponential cutoff y = A * ℓ^(-τ) * exp(-ℓ/ℓc) =====
model_cut(ℓ, p) = p[1] .* ℓ.^(-p[2]) .* exp.(-ℓ ./ p[3])   # p = [A, τ, ℓc]

# initial ℓc guess: a fraction of your fitting window width
ℓc0 = max(10.0, 0.5 * (ℓmax - ℓmin))
p0_cut = [A_pow, τ_pow, ℓc0]

fit_cut = curve_fit(model_cut, ℓ, y, p0_cut)
A_cut, τ_cut, ℓc_cut = fit_cut.param

# ===== Compare fits (simple AIC via RSS; assumes same Gaussian noise scale) =====
rss_pow = sum(abs2, y .- model_pow(ℓ, fit_pow.param))
rss_cut = sum(abs2, y .- model_cut(ℓ, fit_cut.param))

k_pow, k_cut = 2, 3
n = length(y)
aic_pow = n * log(rss_pow/n) + 2k_pow
aic_cut = n * log(rss_cut/n) + 2k_cut

println("Fit window: ℓ ∈ [$ℓmin, $ℓmax], N = $n points")
println("Power-law fit:             τ = $(τ_pow),   A = $(A_pow)")
println("Power-law + cutoff fit:    τ = $(τ_cut),   A = $(A_cut),   ℓc = $(ℓc_cut)")
println("AIC (lower is better):     AIC_pow = $(aic_pow),  AIC_cut = $(aic_cut)")

# ===== Plot data and both fits on log-log =====
# Plot range (for visualization) up to maxval
plot_mask = (length_values .>= 1) .& (length_values .< maxval) .& (freq .> 0)
ℓp = Float64.(collect(length_values[plot_mask]))
yp = Float64.(freq[plot_mask])

p = scatter(ℓp, yp,
    xscale=:log10, yscale=:log10,
    xlabel="Loop Length", ylabel="Frequency",
    title="Distribution of Loop Lengths (with fits)",
    label="data", markersize=3)

# draw fitted curves over the fitting window and across the plotted range
ℓgrid = range(max(1.0, minimum(ℓp)), stop=maximum(ℓp), length=600)

plot!(p, ℓgrid, model_pow(ℓgrid, fit_pow.param),
    lw=3, label="power law (τ=$(round(τ_pow, digits=3)))")

plot!(p, ℓgrid, model_cut(ℓgrid, fit_cut.param),
    lw=3, label="power law + cutoff (τ=$(round(τ_cut, digits=3)), ℓc=$(round(ℓc_cut, digits=1)))")

# highlight the fitting window (optional vertical guides)
vline!(p, [ℓmin, ℓmax], lw=2, label=false)

display(p)




p=histogram(loop_lengths, nbins=500000, xlabel="Loop Length", ylabel="Frequency", title="Distribution of Loop Lengths")
xlims!(p,0,100)


p=histogram(log10.(loop_lengths), nbins=1000, xlabel="log10(Loop Length)", ylabel="Frequency", yscale=:log10, title="Distribution of Loop Lengths")
xlims!(p,1,log10.(5000))


