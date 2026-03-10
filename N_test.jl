"""
Testing effect of N on the correlations
Created: 04.02.2026
"""

include("/Users/jamwalk/Desktop/Research/Roderich/Discrete CSL/exp_charge_consv/ecc_func.jl")
using DataFrames
using CSV
using Printf
using Colors

# ============================================================
# Helpers
# ============================================================

make_bc(lattice, N) = Bonds(
    lattice,
    N,
    zeros(Int, 2 * lattice.Lx * lattice.Ly),
    zeros(Int, lattice.Lx * lattice.Ly),
)

logcorr(C; norm=1.0) = log10(abs(C) / norm)

function shift_curve(Cmat, bc, shifts; norm=1.0)
    pts = Tuple{Float64,Float64}[]
    for sh in shifts
        r, C = Cbulk_r(Cmat, bc, sh)
        if isfinite(C) && C != 0.0
            push!(pts, (r, logcorr(C; norm=norm)))
        end
    end
    return pts
end

function add_shift_curve!(
    p,
    Cmat,
    bc,
    shifts;
    norm=1.0,
    label="",
    seriestype=:path,
    kwargs...,
)
    pts = shift_curve(Cmat, bc, shifts; norm=norm)

    if seriestype == :scatter
        scatter!(p, pts; label=label, kwargs...)
    else
        plot!(p, pts; label=label, kwargs...)
    end
    return p
end

function finish_plot!(
    p;
    xlabel_str="Δr (bond midpoint grid)",
    ylabel_str="log10 ⟨C(Δr,0)⟩_bulk",
    title_str="",
    ylims_tuple=nothing,
    savepath=nothing,
)
    xlabel!(p, xlabel_str)
    ylabel!(p, ylabel_str)
    title!(p, title_str)
    isnothing(ylims_tuple) || ylims!(p, ylims_tuple)
    display(p)
    isnothing(savepath) || savefig(p, savepath)
    return p
end
#-----------------------------
# Import Data
#-----------------------------
#The data is taken for fixed L=20 to see how correlations scale with N
Ns = [2,10,20]#[2,6,10,14,20]
Cdict  = Dict{Int, Matrix{Float64}}()
directory = "../ECC_data/T=0/Bond_C_Ns/it=10^7L=20/"

for N in Ns
    Cfile  = joinpath(directory, "Cmat_N$(N).csv")

    dfC  = CSV.read(Cfile,  DataFrame)

    Cmat  = Matrix{Float64}(dfC)

    Cdict[N]  = Cmat
end

#---------------------------------------------------------------------
# --- plotting bulk pair-averaged correlator vs. r for different N ---
#---------------------------------------------------------------------
# ============================================================
# Plot 1: general bulk pair-averaged correlator vs N
# ============================================================

L = 20
lattice = Lattice(L, L)
plotname = "ECC_bulkC_T=0_Ns"
shift_max = 23

p = plot(xlabel="r", ylabel="log10 |C(r)|")

for N in Ns
    Cmat = Cdict[N]
    bc   = make_bc(lattice, N)

    rs, Cs = Cbulk_vs_r(Cmat, bc; shift_max=shift_max, shift_min=1)
    ys = log10.(abs.(Cs) ./ N^2)

    plot!(p, rs, ys; marker=:circle, ms=3, label="N=$N")
end
display(p)
#xlims!(p,0, 20)
finish_plot!(
    p;
    xlabel_str="Δr (bond midpoint grid)",
    ylabel_str="log10 ⟨C(Δr,0)/N^2⟩_bulk",
    title_str="Bulk Pair-Averaged Correlator vs N (L=20, 10^9 iterations)",
    savepath=joinpath(homedir(), "Downloads", plotname * "_general.png"),
)

# ============================================================
# Plot 2: A-A sublattice correlations
# ============================================================

p = plot()

for N in Ns
    println("N = $N")
    Cmat = Cdict[N]
    bc   = make_bc(lattice, N)

    aa_shifts = (2 * Δq for Δq in 1:(Int(L ÷ 2) + 3))
    add_shift_curve!(
        p, Cmat, bc, aa_shifts;
        norm=N^2,
        label="N=$N",
    )
end

Cxx_asymp(r) = r == 0.0 ? NaN : 0.005 * sqrt(r) * exp(-r) * (1 + 3/(8r) - 15/(2 * (8r)^2))
rgrid = collect(range(2.0, stop=10, length=400))
plot!(p, rgrid, log10.(abs.(Cxx_asymp.(rgrid))); lw=2, ls=:dash, label="asymp ∝ r K₁(r)")

finish_plot!(
    p;
    xlabel_str="Δr (bond midpoint grid)",
    ylabel_str="log10 ⟨C(Δr,0)⟩_bulk",
    title_str="⟨C(Δr,0)⟩_bulk in x-direction for A-A sublattices",
    ylims_tuple=(-5.5, -1.5),
    savepath=joinpath(homedir(), "Downloads", plotname * "_AA.png"),
)

# ============================================================
# Plot 3: A-B sublattice correlations
# ============================================================

p = plot()

for N in Ns
    println("N = $N")
    Cmat = Cdict[N]
    bc   = make_bc(lattice, N)

    ab_shifts = (2 * Δq + 1 for Δq in 0:(Int(L ÷ 2) + 3))
    add_shift_curve!(
        p, Cmat, bc, ab_shifts;
        norm=N^2,
        label="N=$N",
    )
end

finish_plot!(
    p;
    xlabel_str="Δr (bond midpoint grid)",
    ylabel_str="log10 ⟨C(Δr,0)⟩_bulk",
    title_str="⟨C(Δr,0)⟩_bulk in x-direction for A-B sublattices",
    ylims_tuple=(-7.0, -1.0),
)








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