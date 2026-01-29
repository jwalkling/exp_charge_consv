"""
Testing finite size effects on correlations between the bonds
Created: 29.01.2026
"""

include("/Users/jamwalk/Desktop/Research/Roderich/Discrete CSL/exp_charge_consv/ecc_func.jl")
using DataFrames
using CSV


"""
    bond_corr_matrix_thermal!(
        bc::Bonds,
        rng::AbstractRNG;
        connected::Bool = false,
        burnin::Int = 1_000,
        nsamples::Int = 2_000,
        thin::Int = 10,
        mc_step! = MC_T0_loop!,
    ) -> meanC::Matrix{Float64}, Cstderr::Matrix{Float64}

Compute the full bond–bond correlator matrix

    C[b1, b2] = ⟨ s[b1] s[b2] ⟩                    (connected=false)
    C[b1, b2] = ⟨ (s[b1]-μ)(s[b2]-μ) ⟩            (connected=true)

where μ = average bond value in the *current* configuration (per sample),
matching your existing estimator logic.

Returns:
- `meanC`: Nb×Nb matrix of correlator means
- `Cstderr`: Nb×Nb matrix of standard errors (NaN if nsamples ≤ 1)

Notes:
- This is O(Nb^2) memory and O(nsamples * Nb^2) work. For large lattices, this will be heavy.
- Uses only the upper triangle during accumulation, then mirrors to enforce symmetry.
"""
function bond_corr_matrix_thermal!(
    bc::Bonds,
    rng::AbstractRNG;
    connected::Bool = false,
    burnin::Int = 1_000,
    nsamples::Int = 2_000,
    thin::Int = 10,
    mc_step! = MC_T0_loop!,
)
    b = bc.bond
    Nb = length(b)

    δB_0s = δB_0_tuple(bc)

    @inbounds for _ in 1:burnin
        mc_step!(bc, rng, δB_0s)
    end

    meanC = zeros(Float64, Nb, Nb)  # store full for convenience; update only upper triangle
    M2C   = zeros(Float64, Nb, Nb)

    # work buffer for centered/un-centered bond values
    v = Vector{Float64}(undef, Nb)

    @inbounds for s in 1:nsamples
        for _ in 1:thin
            mc_step!(bc, rng, δB_0s)
        end

        if connected
            μ = sum(b) / Nb
            @inbounds for i in 1:Nb
                v[i] = b[i] - μ
            end
        else
            @inbounds for i in 1:Nb
                v[i] = b[i]
            end
        end

        # Welford update, upper triangle only (symmetry)
        @inbounds for j in 1:Nb
            vj = v[j]
            for i in 1:j
                Cij = v[i] * vj
                δ   = Cij - meanC[i, j]
                meanC[i, j] += δ / s
                M2C[i, j]   += δ * (Cij - meanC[i, j])
            end
        end
    end

    # standard error
    Cstderr = fill(NaN, Nb, Nb)
    if nsamples > 1
        @inbounds for j in 1:Nb
            for i in 1:j
                Cstderr[i, j] = sqrt((M2C[i, j] / (nsamples - 1)) / nsamples)
            end
        end
    end

    # mirror to lower triangle (enforce exact symmetry)
    @inbounds for j in 1:Nb
        for i in 1:j-1
            meanC[j, i]   = meanC[i, j]
            Cstderr[j, i] = Cstderr[i, j]
        end
    end

    return meanC, Cstderr
end


#-----------------------------
# Find the correlator matrix between all pairs
#-----------------------------
sizes = [6, 8, 10, 14, 18, 22]

Cdict  = Dict{Int, Matrix{Float64}}()
dCdict = Dict{Int, Matrix{Float64}}()

iterations = 10_000

outdir = "bond_corr_mats"
isdir(outdir) || mkpath(outdir)


for L in sizes
    println("Computing bond correlator for L = $L")
    N = 2
    lattice = Lattice(L, L)
    bc = Bonds(lattice, N, zeros(Int, 2 * lattice.Lx * lattice.Ly))
    rng = MersenneTwister(1234 + L)   # vary seed per size if desired

    Cmat, dCmat = bond_corr_matrix_thermal!(bc, rng;
        connected = true,
        burnin    = 1_000,
        nsamples  = iterations,
        thin      = 10,
        mc_step!  = MC_T0_loop!,
    )

    Cdict[L]  = Cmat
    dCdict[L] = dCmat

    # --- export ---
    dfC  = DataFrame(Cmat, :auto)
    dfdC = DataFrame(dCmat, :auto)

    CSV.write(joinpath(outdir, "Cmat_L$(L).csv"),  dfC)
    CSV.write(joinpath(outdir, "dCmat_L$(L).csv"), dfdC)
end



# Cdict and dCdict now map L -> correlator matrix / stderr matrix

#-----------------------------------
# Bulk pair-averaged correlator
#-----------------------------------



#-----------------------------------
# Bond-Bond Correlator in Real Space
#-----------------------------------

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
