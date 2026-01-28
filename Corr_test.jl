"""
Testing correlations between the bonds
Created: 28.01.2026
"""

include("/Users/jamwalk/Desktop/Research/Roderich/Discrete CSL/exp_charge_consv/ecc_func.jl")


#-----------------------------
# Find the average bond values
#-----------------------------
L=8
N=2
lattice = Lattice(L,L)
bc = Bonds(lattice, N, zeros(Int, 2*lattice.Lx*lattice.Ly))
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

L=8
N=4
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

p = plot_bond_corr_realspace(bc, indexc, log10.(abs.(Cb)); clim=nothing)
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
