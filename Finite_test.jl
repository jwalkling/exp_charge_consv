"""
Testing finite size effects on correlations between the bonds
Created: 29.01.2026
"""

include("/Users/jamwalk/Desktop/Research/Roderich/Discrete CSL/exp_charge_consv/ecc_func.jl")
using DataFrames
using CSV
using Printf
using Colors

#-----------------------------
# Import Data
#-----------------------------

sizes = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
Cdict  = Dict{Int, Matrix{Float64}}()
dCdict = Dict{Int, Matrix{Float64}}()
#Data has N=2 fixed with 10^8 iterations
directory = "../ECC_data/T=0/Bond_C_Ls/"

for size in sizes
    L = size
    Cfile  = joinpath(directory, "Cmat_L$(L).csv")
    dCfile = joinpath(directory, "dCmat_L$(L).csv")

    dfC  = CSV.read(Cfile,  DataFrame)
    dfdC = CSV.read(dCfile, DataFrame)

    Cmat  = Matrix{Float64}(dfC)
    dCmat = Matrix{Float64}(dfdC)

    Cdict[L]  = Cmat
    dCdict[L] = dCmat
end

#-----------------------------------
# Bulk pair-averaged correlator
#-----------------------------------
#The labelling means bonds alternate between vert and horizont (A vs. B sublattice)
plotname="ECC_bulkC_T=0"
# First plot: A-A sublattice correlations
p=plot()
for L in [10, 12, 14, 16, 18, 20, 22, 24]
    println("L = $L")
    N=2
    lattice = Lattice(L,L)
    indexc=Int((L-1)*L) #Comparison index of the bond
    Cmat=Cdict[L]
    bc = Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly), zeros(Int, lattice.Lx*lattice.Ly), [])
    plot!(p,
    [((r, C) = Cbulk_r(Cmat, bc, 2*Δq); (r, log10(abs(C))))
     for Δq in 1:Int(L÷2)+3],
    label = "{L=$L}")  # exclude onsite

end
xlabel!(p, "Δr (bond midpoint grid)")
ylabel!(p, "log10 ⟨C(Δr,0)⟩_bulk")
ylims!(p, (-4.5,-1.5))
title!(p, "⟨C(Δr,0)⟩_bulk in x-direction for A-A sublattices")
display(p)
savefig(p, joinpath(homedir(), "Downloads", plotname*"_AA.png"))

# Second plot: A-B sublattice correlations
p=plot()
for L in [10, 12, 14, 16, 18, 20, 22, 24]
    println("L = $L")
    N=2
    lattice = Lattice(L,L)
    indexc=Int((L-1)*L) #Comparison index of the bond
    Cmat=Cdict[L]
    bc = Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly), zeros(Int, lattice.Lx*lattice.Ly), [])
    plot!(p,
    [((r, C) = Cbulk_r(Cmat, bc, 2*Δq+1); (r, log10(abs(C))))
     for Δq in 0:Int(L÷2)+3], #L/2-1 cuts off before they go into noise/finite-size effects.
    label = "{L=$L}",
) 
end
xlabel!(p, "Δr (bond midpoint grid)")
ylabel!(p, "log10 ⟨C(Δr,0)⟩_bulk")
ylims!(p, (-4,-0.5))
title!(p, "⟨C(Δr,0)⟩_bulk in x-direction for A-B sublattices")
display(p)
savefig(p, joinpath(homedir(), "Downloads", plotname*"_AB.png"))

# Third plot: All correlations together
p=plot()
for L in [10, 12, 14, 16, 18, 20, 22, 24]
    println("L = $L")
    N=2
    lattice = Lattice(L,L)
    indexc=Int((L-1)*L) #Comparison index of the bond
    Cmat=Cdict[L]
    bc = Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly), zeros(Int, lattice.Lx*lattice.Ly), [])
    scatter!(p,
    [((r, C) = Cbulk_r(Cmat, bc, Δq); (r, log10(abs(C))))
     for Δq in 1:L-1],
    label = "{L=$L}",
    
)  # exclude onsite
end
xlabel!(p, "Δr (bond midpoint grid)")
ylabel!(p, "log10 ⟨C(Δr,0)⟩_bulk")
ylims!(p, (-4,-0.5))
title!(p, "⟨C(Δr,0)⟩_bulk in x-direction for all points")
display(p)
savefig(p, joinpath(homedir(), "Downloads", plotname*"_all.png"))
#-----------------------------------
# Antiquated plots of real-space correlators
#-----------------------------------
L=22
N=2
lattice = Lattice(L,L)
indexc=Int((L-1)*L) #Comparison index of the bond

Cmat=Cdict[L]
bc = Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly), zeros(Int, lattice.Lx*lattice.Ly), [])
p = plot_bond_corr_realspace(bc, indexc, log10.(abs.(Cmat[:,indexc])); clim=(-6,1.05))
display(p)
p = plot_bond_corr_realspace_signlogshade(bc, indexc, Cmat[indexc,:]; eps_mag=1e-12, γ=1.0, vmin=0.01, vmax=4)
display(p)




L=8
N=2
lattice = Lattice(L,L)
bc = Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly))
rng = MersenneTwister(1234)
δB_0s=δB_0_tuple(bond_config)


indexc=Int((L-1)*L) #Comparison index of the bond

function plot_bond_corr_realspace(
    bc::Bonds,
    b0::Int,
    meanC::AbstractVector;
    cmap = :RdBu,
    lw::Real = 4,
    clim = nothing,
    show_ref::Bool = true,
    title::Union{Nothing,String} = nothing,
)
    b = bc.bond
    lat = bc.lattice
    Lx, Ly = lat.Lx, lat.Ly
    Nsites = Lx * Ly

    length(meanC) == length(b) || throw(DimensionMismatch("meanC must have length length(bc.bond)"))
    (1 <= b0 <= length(b)) || throw(ArgumentError("b0 must satisfy 1 ≤ b0 ≤ length(bc.bond)"))

    @inline site_of_bond(bi::Int) = (bi + 1) >>> 1
    @inline is_hbond(bi::Int)     = isodd(bi)
    @inline x_of_site(i::Int)     = ((i - 1) % Lx) + 1
    @inline y_of_site(i::Int)     = ((i - 1) ÷ Lx) + 1

    xs = Float64[]; ys = Float64[]; zs = Float64[]

    @inbounds for i in 1:Nsites
        x = x_of_site(i)
        y = y_of_site(i)

        if x < Lx
            bi = 2i - 1
            C  = meanC[bi]
            append!(xs, (x, x+1, NaN))
            append!(ys, (y, y,   NaN))
            append!(zs, (C, C,   NaN))
        end

        if y < Ly
            bi = 2i
            C  = meanC[bi]
            append!(xs, (x, x,   NaN))
            append!(ys, (y, y+1, NaN))
            append!(zs, (C, C,   NaN))
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

    xs_v = [((i-1) % Lx) + 1 for i in 1:Nsites]
    ys_v = [((i-1) ÷ Lx) + 1 for i in 1:Nsites]
    scatter!(p, xs_v, ys_v; markersize=3, marker=:circle, color=:black)

    if show_ref
        i0 = site_of_bond(b0)
        x0, y0 = x_of_site(i0), y_of_site(i0)

        if is_hbond(b0)
            x0 < Lx || throw(ArgumentError("b0=$b0 is a horizontal bond on the right boundary (invalid link)."))
            plot!(p, [x0, x0+1], [y0, y0]; linewidth=lw+2, color=:black, label="")
        else
            y0 < Ly || throw(ArgumentError("b0=$b0 is a vertical bond on the top boundary (invalid link)."))
            plot!(p, [x0, x0], [y0, y0+1]; linewidth=lw+2, color=:black, label="")
        end
    end

    return p
end

function plot_bond_corr_realspace_signlogshade(
    bc::Bonds,
    b0::Int,
    meanC::AbstractVector;
    lw::Real = 4,
    show_ref::Bool = true,
    title::Union{Nothing,String} = nothing,
    # magnitude -> darkness mapping
    eps_mag::Real = 1e-15,          # floor for |C| to avoid log(0)
    vmin::Union{Nothing,Real} = nothing,  # min magnitude for scaling (in |C|)
    vmax::Union{Nothing,Real} = nothing,  # max magnitude for scaling (in |C|)
    γ::Real = 1.0,                  # gamma for contrast: >1 emphasizes extremes
    light_min::Real = 0.15,         # darkest (for largest |C|)
    light_max::Real = 0.95,         # lightest (for smallest nonzero |C|)
)
    b = bc.bond
    lat = bc.lattice
    Lx, Ly = lat.Lx, lat.Ly
    Nsites = Lx * Ly

    length(meanC) == length(b) || throw(DimensionMismatch("meanC must have length length(bc.bond)"))
    (1 <= b0 <= length(b)) || throw(ArgumentError("b0 must satisfy 1 ≤ b0 ≤ length(bc.bond)"))
    eps_mag > 0 || throw(ArgumentError("eps_mag must be > 0"))
    (0 < light_min < light_max < 1) || throw(ArgumentError("Require 0 < light_min < light_max < 1"))
    γ > 0 || throw(ArgumentError("γ must be > 0"))

    @inline site_of_bond(bi::Int) = (bi + 1) >>> 1
    @inline is_hbond(bi::Int)     = isodd(bi)
    @inline x_of_site(i::Int)     = ((i - 1) % Lx) + 1
    @inline y_of_site(i::Int)     = ((i - 1) ÷ Lx) + 1

    # --- determine scaling range for log-magnitude ---
    # We'll map log10(|C|) in [log10(vmin), log10(vmax)] -> lightness in [light_max, light_min]
    absvals = Float64[]
    sizehint!(absvals, length(meanC))
    @inbounds for c in meanC
        a = abs(float(c))
        if a > 0
            push!(absvals, a)
        end
    end
    isempty(absvals) && error("meanC appears to be identically zero.")

    vmin_eff = vmin === nothing ? maximum([minimum(absvals), float(eps_mag)]) : max(float(vmin), float(eps_mag))
    vmax_eff = vmax === nothing ? maximum(absvals) : max(float(vmax), vmin_eff)

    logmin = log10(vmin_eff)
    logmax = log10(vmax_eff)
    denom  = max(logmax - logmin, 1e-30)  # avoid divide by zero

    # --- mapping magnitude -> lightness (HSL) ---
    # t=0 for smallest mag => light_max (nearly white)
    # t=1 for largest  mag => light_min (dark)
    @inline function lightness_from_mag(a::Float64)::Float64
        t = (log10(max(a, float(eps_mag))) - logmin) / denom
        t = clamp(t, 0.0, 1.0)
        t = t^float(γ)
        return light_max - t*(light_max - light_min)
    end

    # Fixed hues: blue for negative, red for positive. Saturation high.
    HRED  = 0.0
    HBLUE = 220.0  # "deep blue"
    SAT   = 0.90

    @inline function color_from_C(c::Real)
        c == 0 && return RGB(1,1,1)  # white
        a = abs(float(c))
        L = lightness_from_mag(a)
        h = (c > 0) ? HRED : HBLUE
        return convert(RGB, HSL(h, SAT, L))
    end

    # We'll build segments and plot them as one series per segment so each gets its own color.
    # This avoids relying on line_z colormap normalization entirely.
    p = plot(; aspect_ratio=:equal, xlabel="x", ylabel="y",
             legend=false,
             title = title === nothing ? "Sign-hue + log(|C|) shade map, b0=$b0" : title)

    @inbounds for i in 1:Nsites
        x = x_of_site(i)
        y = y_of_site(i)

        if x < Lx
            bi = 2i - 1
            col = color_from_C(meanC[bi])
            plot!(p, [x, x+1], [y, y]; linewidth=lw, color=col, label="")
        end

        if y < Ly
            bi = 2i
            col = color_from_C(meanC[bi])
            plot!(p, [x, x], [y, y+1]; linewidth=lw, color=col, label="")
        end
    end

    # vertices
    xs_v = [((i-1) % Lx) + 1 for i in 1:Nsites]
    ys_v = [((i-1) ÷ Lx) + 1 for i in 1:Nsites]
    scatter!(p, xs_v, ys_v; markersize=3, marker=:circle, color=:black)

    # reference bond highlight
    if show_ref
        i0 = site_of_bond(b0)
        x0, y0 = x_of_site(i0), y_of_site(i0)

        if is_hbond(b0)
            x0 < Lx || throw(ArgumentError("b0=$b0 is a horizontal bond on the right boundary (invalid link)."))
            plot!(p, [x0, x0+1], [y0, y0]; linewidth=lw+2, color=:black, label="")
        else
            y0 < Ly || throw(ArgumentError("b0=$b0 is a vertical bond on the top boundary (invalid link)."))
            plot!(p, [x0, x0], [y0, y0+1]; linewidth=lw+2, color=:black, label="")
        end
    end

    return p
end






#-------------------------------------
# Visualize lattice with bond indices
#-------------------------------------
L=10
lattice_image = Lattice(L,L)

p = plot(
    legend = false,
    aspect_ratio = 1,
    title = "Lattice Link Centres",
    xlim = (0.5, lattice_image.Lx + 1.5),
    ylim = (0.5, lattice_image.Ly + 1.5),
)

# sites
for index in 0:(lattice_image.Lx * lattice_image.Ly - 1)
    x, y = divrem(index, lattice_image.Lx)
    scatter!(p, (x+1, y+1), color=:blue, markersize=4)
end

Lx, Ly = lattice_image.Lx, lattice_image.Ly
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
for index in 0:(lattice_image.Lx * lattice_image.Ly - 1)
    y, x = divrem(index, lattice_image.Lx)
    xp, yp = x + 1, y + 1
    scatter!(p, (xp, yp), color=:blue, markersize=4)
    annotate!(p, (xp, yp, text("$(index+1)", 8, :red)))
end
display(p)



p=plot(xlims=(0, L+1), ylims=(0, L+1), aspect_ratio=1, title="Bond Centres")

for i in 1:4
    plot!(p, index_to_coord(lattice_image, i), seriestype=:scatter, label="Bond $i", aspect_ratio=1)
end

display(p)