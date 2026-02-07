"""
Testing effect of N on the correlations
Created: 04.02.2026
"""

include("/Users/jamwalk/Desktop/Research/Roderich/Discrete CSL/exp_charge_consv/ecc_func.jl")
using DataFrames
using CSV
using Printf
using Colors

#-----------------------------
# Import Data
#-----------------------------
#The data is taken for fixed L=24 and N=2 to see how correlations scale with the iterations
Ns = [2,6,10,14,20]
Cdict  = Dict{Int, Matrix{Float64}}()
directory = "../ECC_data/T=0/Bond_C_Ns/it=10^9L=20/"

for N in Ns
    Cfile  = joinpath(directory, "Cmat_N$(N).csv")

    dfC  = CSV.read(Cfile,  DataFrame)

    Cmat  = Matrix{Float64}(dfC)

    Cdict[N]  = Cmat
end

#-----------------------------------
# Bulk pair-averaged correlator
#-----------------------------------
function Cbulk_r(Cmat, bc, shift::Int)
    lat= bc.lattice
    Lx, Ly = lat.Lx, lat.Ly
    Nbonds = 2*Lx * Ly
    total = 0.0
    count = 0
    stored_dr = false
    dr=0

    for i in 1:Nbonds
        (x0,y0) = index_to_coord2(lat, i)
        (x1,y1) = index_to_coord2(lat, i+shift)
        dx = x1 - x0
        dy = y1 - y0
        if dx < 0 || dy < 0 || dx >= Lx || dy >= Ly
            continue
        end
        if stored_dr == false
            dr=sqrt(dx^2+dy^2)
            stored_dr = true
        end
        #println(dx^2+dy^2)
        value=Cmat[i, i+shift]
        if value == 0.0 # Ignore the zero values corresponding to the "dead" bonds used for PBC.
            continue
        end
        total+=Cmat[i, i+shift] #TK stop the bonds where they go out of bounds.
        count+=1
    end
    return (dr,count > 0 ? total / count : NaN)
end

L=20
lattice = Lattice(L,L)
indexc=Int((L-1)*L) #Comparison index of the bond

p=plot()
for N in Ns
    Cmat=Cdict[N]
    bc = Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly))
    plot!(p,log10.(abs.(bulk_pair_mean_r(Cmat, lattice, 10; margin=1)[2]/N^2)), label="N=$N")
    xlims!(2,11)
    ylims!(-6,0)
end
title!(p, "Bulk Pair-Averaged Correlator vs N (L=20, 10^9 iterations)")
xlabel!(p, "Δr (bond midpoint grid)")
ylabel!(p, "log10 ⟨C(Δr,0)/N^2⟩_bulk")
display(p)
savefig(p, joinpath(homedir(), "Downloads", plotname*"_general.png"))

#The labelling means bonds alternate between vert and horizont (A vs. B sublattice)
plotname="ECC_bulkC_T=0_Ns"
# First plot: A-A sublattice correlations
p=plot()
for L in [10, 12, 14, 16, 18, 20, 22, 24]
    println("L = $L")
    N=2
    lattice = Lattice(L,L)
    indexc=Int((L-1)*L) #Comparison index of the bond
    Cmat=Cdict[L]
    bc = Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly))
    plot!(p,
    [((r, C) = Cbulk_r(Cmat, bc, 2*Δq); (r, log10(abs(C))))
     for Δq in 1:Int(L÷2)+3],
    label = "{L=$L}",
)  # exclude onsite

end
xlabel!(p, "Δr (bond midpoint grid)")
ylabel!(p, "log10 ⟨C(Δr,0)⟩_bulk")
ylims!(p, (-4,-0.5))
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
    bc = Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly))
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
    bc = Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly))
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
# Bond-Bond Correlator in Real Space
#-----------------------------------
L=22
N=2
lattice = Lattice(L,L)
indexc=Int((L-1)*L) #Comparison index of the bond

Cmat=Cdict[L]
bc = Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly))
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

function bond_corr_realspace_thermal!(
    bc::Bonds,
    rng::AbstractRNG,
    b0::Int;
    connected::Bool = false,
    burnin::Int = 1_000,
    nsamples::Int = 2_000,
    thin::Int = 10,
    mc_step! = MC_T0_loop!,
)
    b = bc.bond
    (1 <= b0 <= length(b)) || throw(ArgumentError("b0 must satisfy 1 ≤ b0 ≤ length(bc.bond)"))

    δB_0s = δB_0_tuple(bc)

    @inbounds for _ in 1:burnin
        mc_step!(bc, rng, δB_0s)
    end

    meanC = zeros(Float64, length(b))
    M2C   = zeros(Float64, length(b))

    @inbounds for s in 1:nsamples
        for _ in 1:thin
            mc_step!(bc, rng, δB_0s)
        end

        s0 = b[b0]
        μ  = connected ? (sum(b) / length(b)) : 0.0

        @inbounds for bi in eachindex(b)
            C = connected ? (s0 - μ) * (b[bi] - μ) : (s0 * b[bi])
            δ = C - meanC[bi]
            meanC[bi] += δ / s
            M2C[bi]   += δ * (C - meanC[bi])
        end
    end

    Cstderr = fill(NaN, length(b))
    if nsamples > 1
        @inbounds Cstderr .= sqrt.((M2C ./ (nsamples - 1)) ./ nsamples)
    end

    return meanC, Cstderr
end

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


Cb, dCb = bond_corr_realspace_thermal!(bc, rng, indexc;
    connected=true, burnin=1_000, nsamples=100_000_000, thin=20)

p = plot_bond_corr_realspace(bc, indexc, log10.(abs.(Cb)); clim=(-4,1.05))
display(p)

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
    plot!(p, index_to_coord2(lattice_image, i), seriestype=:scatter, label="Bond $i", aspect_ratio=1)
end

display(p)