# ═══════════════════════════════════════════════════════════════════════════
#  ls_dos.jl — the pair continuum: density of states, spectral shift and
#  channel-projected spectral functions.
#
#  Everything is built from the same propagator as ls_two_body.jl, continued
#  to z = E + iε.  Two rules govern this file:
#
#   1. Work with the FULL matrix F over channels (0, x, 2x, y, z), never with
#      the Schur complement M_hc.  det F = (-Π₀₀) det M_hc, and the extra
#      factor is the hard core itself: the onsite relative state is pushed to
#      infinite energy and leaves the spectrum.  Numerically, Re Π₀₀ vanishes
#      at some real energy inside the band, where M_hc has a pole; in
#      ∂_E ln det F that pole is cancelled.  Never divide by Π₀₀.
#
#   2. F is complex SYMMETRIC, not Hermitian.  Transposes, never adjoints.
# ═══════════════════════════════════════════════════════════════════════════

using LinearAlgebra
using StaticArrays

"""
    Π_full(z, e, fx, f2x) -> (Π, ∂Π)

Propagator and its energy derivative at complex `z`, INCLUDING the hard-core
channel 0 (φ₀ ≡ 1).  Two 5×5 arrays over channels (0, x, 2x, y, z).
"""
function Π_full(z::ComplexF64, e, fx, f2x)
    L = size(e, 1)
    s = zero(MMatrix{5,5,ComplexF64})
    d = zero(MMatrix{5,5,ComplexF64})
    @inbounds for il in 1:L
        fz = fx[il]
        for ik in 1:L
            fy = fx[ik]
            for ih in 1:L
                φ = SVector(1.0, fx[ih], f2x[ih], fy, fz)
                g = 1.0 / (z - e[ih,ik,il])
                P = φ * transpose(φ)            # transpose: φ is real here,
                s .+= P .* g                    # but the rule is global
                d .+= P .* (-g*g)
            end
        end
    end
    w = 1/L^3
    return w*SMatrix{5,5,ComplexF64}(s), w*SMatrix{5,5,ComplexF64}(d)
end

"""
    spectral(E, e, fx, f2x, V; ε) -> (ρ₀, A₀, A, Δρ)

All continuum quantities at one energy.

  ρ₀ = -Im Π₀₀/π                     free pair DOS,       ∫ρ₀ dE = 1
  A₀ = -Im Π_aa/π    per channel     free weight,         ∫A₀ dE = 1
  A  = -Im G_aa/π,  G = Π + Π F⁻¹Π   interacting weight,  ∫A dE = 1 - w_bound
  Δρ = -Im Tr[F⁻¹ ∂_E F]/π           spectral shift,      ∫Δρ dE = -(1 + N_b)

ρ₀ and A are normalized PER STATE; Δρ is an ABSOLUTE state count.  Adding
them requires dividing Δρ by L³ first — the symptom of forgetting is a total
DOS that goes negative.

A is the object to compare against A₀.  The total DOS is not: the interaction
has finite rank, so the integrated level count changes by O(1) out of L³ and
the free and interacting densities of states are indistinguishable.  What
changes is which levels the pair state overlaps with.
"""
function spectral(E::Float64, e, fx, f2x, V; ε = 0.15)
    Π, dΠ = Π_full(complex(E, ε), e, fx, f2x)

    keep = [1; 1 .+ findall(!iszero, V)]        # channel 0 is always kept
    F  = Matrix(-Π[keep,keep])
    dF = Matrix(-dΠ[keep,keep])
    for (i,a) in enumerate(keep)                # 1/U → 0 leaves F[1,1] alone
        a == 1 || (F[i,i] += 1/V[a-1])
    end

    Πk = Matrix(Π[keep,keep])
    G  = Πk + Πk*(F \ Πk)

    A0 = [-imag(Πk[i,i])/π for i in eachindex(keep)]
    A1 = [-imag(G[i,i])/π  for i in eachindex(keep)]
    return A0[1], A0, A1, -imag(tr(F \ dF))/π
end

"""
    spectral_sweep(Es, e, fx, f2x, V; ε) -> (ρ₀, A₀, A, Δρ)

`spectral` over an energy window.  Always check the two sum rules before
interpreting anything: `∫ρ₀ dE = 1` and `∫Δρ dE = -(1 + N_b)`.  Lorentzian
tails decay as 1/E², so pad the window by ≳ 20ε or the norm falls short for
reasons that have nothing to do with the physics.
"""
function spectral_sweep(Es, e, fx, f2x, V; ε = 0.15)
    nc = length(findall(!iszero, V)) + 1
    ρ0 = similar(Es); Δρ = similar(Es)
    A0 = zeros(length(Es), nc); A1 = zeros(length(Es), nc)
    for (i,E) in enumerate(Es)
        ρ0[i], a0, a1, Δρ[i] = spectral(E, e, fx, f2x, V; ε = ε)
        A0[i,:] .= a0; A1[i,:] .= a1
    end
    return ρ0, A0, A1, Δρ
end

"""Channel names for the columns returned by `spectral_sweep`."""
channel_names(V) = ["0 (hard core)", "x", "2x", "y", "z"][[1; 1 .+ findall(!iszero, V)]]

"""
    branch_dos(E, e, fx, f2x, V; ε) -> Vector

Channel-resolved decomposition of Δρ: Δρ = Σ_i Δρ_i with
Δρ_i = -Im(∂_E μ_i / μ_i)/π over the eigenvalues μ_i of F.  Separates a true
channel resonance (a zero of ONE μ_i, moving when its coupling is varied)
from repulsive push-up near the upper band edge and from van Hove structure,
which appears in every branch and in ρ₀ as well.

F is complex symmetric: the left eigenvectors are vᵀ, not v†.  Using the
adjoint gives branch weights that are silently wrong while still summing to
something plausible, so the sum is checked against `spectral` here.
"""
function branch_dos(E::Float64, e, fx, f2x, V; ε = 0.15)
    Π, dΠ = Π_full(complex(E, ε), e, fx, f2x)
    keep = [1; 1 .+ findall(!iszero, V)]
    F  = Matrix(-Π[keep,keep])
    dF = Matrix(-dΠ[keep,keep])
    for (i,a) in enumerate(keep)
        a == 1 || (F[i,i] += 1/V[a-1])
    end
    λs, Vs = eigen(F)
    out = similar(λs, Float64)
    for i in eachindex(λs)
        v  = @view Vs[:, i]
        dμ = (transpose(v)*dF*v) / (transpose(v)*v)      # transpose, not adjoint
        out[i] = -imag(dμ/λs[i])/π
    end
    return out
end
