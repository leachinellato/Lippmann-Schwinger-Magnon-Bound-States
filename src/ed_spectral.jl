# ═══════════════════════════════════════════════════════════════════════════
#  ed_spectral.jl — channel-projected spectral functions from exact
#  diagonalization, i.e. the ED counterpart of `spectral` in ls_dos.jl.
#
#  In the Γ two-magnon sector a basis state is labelled by the relative
#  separation r, so |⟨a|n⟩|² is directly the weight the exact eigenstate n
#  places on the LS channel a.  Broadening that weight histogram with the
#  same η as the LS sweep gives a curve that should overlay A_a(E).
# ═══════════════════════════════════════════════════════════════════════════

using .GammaMagnons

"""
    channel_index(b, r) -> Int

Index of the Γ basis state carrying relative separation `r`.  The symmetrized
basis identifies `r` with `-r`, so both are looked up.
"""
function channel_index(b::GammaBasis{2}, r::NTuple{3,Int})
    f(v) = mod(v[1], b.Nx) + b.Nx*mod(v[2], b.Ny) + b.Nx*b.Ny*mod(v[3], b.Nz) + 1
    s, s2 = f(r), f((-r[1], -r[2], -r[3]))
    α = findfirst(t -> Int(t[2]) == s || Int(t[2]) == s2, b.reps)
    α === nothing && error("separation $r absent from the basis")
    return α
end

"""Weight |⟨a|n⟩|² of channel `a` in each eigenvector; sums to 1 over n."""
w_channel(evecs::AbstractMatrix, ia::Integer) = abs2.(@view evecs[ia, :])

"""
    A_channel(Egrid, evals, evecs, ia; η) -> Vector

Channel-projected spectral function

    A_a(E) = -Im G_aa(E+iη)/π = Σ_n |⟨a|n⟩|² (η/π)/((E-E_n)² + η²),

the Lorentzian-broadened weight histogram.  This is identically the quantity
`spectral` returns at the same η, so the two curves should overlay.
"""
function A_channel(Egrid::AbstractVector, evals::AbstractVector,
                   evecs::AbstractMatrix, ia::Integer; η::Real = 0.05)
    w = w_channel(evecs, ia)
    A = zeros(float(eltype(Egrid)), length(Egrid))
    @inbounds for n in eachindex(evals)
        wn = w[n]
        wn < 1e-14 && continue
        En = evals[n]
        for (i, E) in enumerate(Egrid)
            d = E - En
            A[i] += wn * η / (π * (d*d + η*η))
        end
    end
    return A
end

"""
    ipr(evecs, mult) -> Vector

Inverse participation ratio in the FULL relative-coordinate lattice,
Σ_r|ψ(r)|⁴ / (Σ_r|ψ(r)|²)².  `mult[i]` is the multiplicity of representative
`i` in the symmetrized basis |r,+⟩ = (|r⟩+|-r⟩)/√2, i.e. 1 for r = 0 and 2
otherwise, obtained from the stabilizer as `mult = 2 ./ b.stab`.  Without this
correction the IPR is wrong by an r-dependent factor, because the physical
amplitude on each site of a star is c_i/√m_i and not c_i.
"""
function ipr(evecs::AbstractMatrix, mult::AbstractVector)
    out = zeros(Float64, size(evecs, 2))
    @inbounds for n in axes(evecs, 2)
        s = 0.0; nrm = 0.0
        for i in axes(evecs, 1)
            p = abs2(evecs[i, n])
            nrm += p
            s += p*p/mult[i]
        end
        out[n] = s/(nrm*nrm)
    end
    return out
end

"""Multiplicity vector for `ipr`, read off the stabilizer orders."""
star_multiplicity(b::GammaBasis{2}) = [2.0/Int(s) for s in b.stab]

"""Mean of `y` in each energy bin: smooths the per-state IPR into a curve."""
function binned(E::AbstractVector, y::AbstractVector, edges::AbstractVector)
    nb = length(edges) - 1
    s = zeros(Float64, nb); c = zeros(Int, nb)
    @inbounds for k in eachindex(E)
        j = searchsortedlast(edges, E[k])
        (1 <= j <= nb) || continue
        s[j] += y[k]; c[j] += 1
    end
    return [c[j] > 0 ? s[j]/c[j] : NaN for j in 1:nb]
end

"""
    separations(b) -> Vector{NTuple{3,Int}}

Minimum-image relative separation of each Γ two-magnon representative, for
computing pair radii directly from an ED eigenvector.
"""
function separations(b::GammaBasis{2})
    mi(r, N) = (r + N ÷ 2) % N - N ÷ 2
    map(1:length(b)) do α
        x, y, z = site_coords(Int(b.reps[α][2]), b.Nx, b.Ny)
        (mi(x, b.Nx), mi(y, b.Ny), mi(z, b.Nz))
    end
end
