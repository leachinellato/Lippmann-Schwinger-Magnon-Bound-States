# ═══════════════════════════════════════════════════════════════════════════
#  ls_two_body.jl — Lippmann–Schwinger solver for two hard-core bosons.
#
#  Below the continuum edge the projected propagator
#
#      Π_ab(E,K) = (1/L³) Σ_q φ_a(q) φ_b(q) / (E - ℰ_K(q))
#
#  is real, the hard-core matrix is
#
#      [M_hc]_ab = δ_ab/V_a - Π_ab + Π_a0 Π_0b / Π_00 ,
#
#  and every bound state is a root of det M_hc = 0.  Channels are the
#  relative separations (x, 2x, y, z) with form factors
#  φ_a(q) = √2 cos(α q·δ); the onsite channel is ELIMINATED (Schur
#  complement), never deleted.
# ═══════════════════════════════════════════════════════════════════════════

using LinearAlgebra
using StaticArrays
using FFTW

"""
    mesh_cache(L, ε; K = (0,0,0)) -> (e, fx, f2x)

Cache `ℰ_K(q)` on an `L³` mesh together with the two form-factor tables.
`ε` is the SINGLE-magnon dispersion; the pair energy is built here.  Every
`φ_a` depends on one Cartesian component only, so `fx = √2cos(2πq)` serves
the x, y and z channels and `f2x = √2cos(4πq)` serves 2x.
"""
function mesh_cache(L::Int, ε; K = (0.0, 0.0, 0.0))
    grid = collect(-0.5:1/L:(0.5 - 1/L))
    fx   = [√2*cos(2π*q) for q in grid]
    f2x  = [√2*cos(4π*q) for q in grid]
    e    = Array{Float64}(undef, L, L, L)
    @inbounds for il in 1:L, ik in 1:L, ih in 1:L
        e[ih,ik,il] = pair_energy(ε, grid[ih], grid[ik], grid[il]; K = K)
    end
    return e, fx, f2x
end

"""Continuum edge `E_th` and the gap to the next mesh level (resolution floor)."""
function edge(e)
    Emin, Enext = minimum(e), Inf
    @inbounds for x in e
        x - Emin > 1e-10 && x < Enext && (Enext = x)
    end
    return Emin, Enext - Emin
end

"""Smallest `L` whose mesh resolves an offset `η` at the band edge (gap ∝ 1/L²)."""
function L_for(η, ε; L0 = 48, safety = 10.0)
    _, gap = edge(first(mesh_cache(L0, ε)))
    return ceil(Int, L0*sqrt(safety*gap/η))
end

"""
    Π_all(E, e, fx, f2x) -> (Π00, Π0, Π)

Projected pair propagator at real `E` below threshold, channels (x, 2x, y, z),
with `Π0[a] = Π_a0` and `Π[a,b] = Π_ab`.  One sweep, all components.
"""
function Π_all(E::Float64, e, fx, f2x)
    L   = size(e, 1)
    s00 = 0.0
    s0  = zero(MVector{4,Float64})
    sab = zero(MMatrix{4,4,Float64})
    @inbounds for il in 1:L
        fz = fx[il]
        for ik in 1:L
            fy = fx[ik]
            for ih in 1:L
                φ = SVector(fx[ih], f2x[ih], fy, fz)
                g = 1.0 / (E - e[ih,ik,il])
                s00 += g
                s0  .+= φ .* g
                sab .+= (φ * φ') .* g
            end
        end
    end
    w = 1/L^3
    return w*s00, w*SVector(s0), w*SMatrix{4,4,Float64}(sab)
end

"""Hard-core matrix, restricted to the channels with `V_a ≠ 0`."""
function M_hc(Π00, Π0, Π, V)
    keep = findall(!iszero, V)
    A    = Matrix(-Π + (Π0*Π0')/Π00)
    return Symmetric(A[keep,keep] + Diagonal([1/V[a] for a in keep])), keep
end

"""
    λ_critical(Π00, Π0, Π, V̄; c = C)

Critical coupling from the rank-one identity: only the diagonal entry of the
tuned channel carries λ, so `det M_hc` is linear in 1/λ and no root-finding
in λ is needed.  Evaluate at `E = E_th - η` and extrapolate in `√η`, not in
`η`: the band edge is quadratic, so the leading correction is `O(√η)`.
"""
function λ_critical(Π00, Π0, Π, V̄; c = C)
    keep = findall(!iszero, V̄)
    A    = Matrix(-Π + (Π0*Π0')/Π00)
    A    = A[keep,keep] + Diagonal([a == c ? 0.0 : 1/V̄[a] for a in keep])
    ic   = findfirst(==(c), keep)
    rest = [i for i in eachindex(keep) if i != ic]
    return -det(A[rest,rest]) / (V̄[c] * det(A))
end

function bisect(f, a, b; tol = 1e-10, itmax = 200)
    fa, fb = f(a), f(b)
    fa*fb > 0 && error("no sign change in [$a, $b]")
    for _ in 1:itmax
        m  = 0.5*(a+b)
        fm = f(m)
        if fa*fm ≤ 0
            b, fb = m, fm
        else
            a, fa = m, fm
        end
        b - a < tol && break
    end
    return 0.5*(a+b)
end

"""
    E_bound(V, e, fx, f2x) -> (E_b, Δ_b) or nothing

Lowest two-body bound state at fixed couplings `V`.

The criterion is `det M_hc = 0`, NOT `eigmin = 0`: a weakly coupled channel
contributes a large fixed `1/V_a` on the diagonal that dominates the smallest
eigenvalue at every energy and hides the crossing.  The sign of the
determinant deep below the band is `(-1)^n_attr`, so the bracket is found by
looking for any sign change rather than assuming one.  `η` sets only the upper
bracket endpoint; the mesh gap is what limits how small a `Δ_b` is meaningful.
"""
function E_bound(V, e, fx, f2x; η = 1e-8, Emin = nothing, gap = nothing,
                 itmax = 60, warn_gaps = 50)
    (Emin === nothing || gap === nothing) && ((Emin, gap) = edge(e))
    f(E) = det(first(M_hc(Π_all(E, e, fx, f2x)..., V)))

    Ehi = Emin - η
    fhi = f(Ehi)
    depth = 4*(maximum(e) - Emin)
    step, Elo, ok = 1e-5*abs(Emin), Emin, false
    for _ in 1:itmax
        Elo = Emin - step
        if f(Elo)*fhi < 0
            ok = true
            break
        end
        step *= 2
        step > depth && break
    end
    ok || return nothing                       # no crossing below the edge

    Eb = bisect(f, Elo, Ehi; tol = 1e-12*max(1, abs(Emin)))
    Δb = Emin - Eb
    Δb < warn_gaps*gap &&
        @warn "Δb = $Δb is only $(round(Δb/gap, digits=1))× the mesh gap $gap — raise L"
    return Eb, Δb
end

"""Convenience wrapper: bound state at tuning parameter `λ`."""
E_bound(λ::Real, e, fx, f2x, V̄::SVector; c = C, kw...) =
    E_bound(Vλ(λ; c = c), e, fx, f2x; kw...)

# ── wave function ──────────────────────────────────────────────────────────

"""
    pair_wavefunction(Eb, V, e, fx, f2x) -> (Ψ, c, d0)

Relative wave function of the bound state at `Eb`.  The null vector of
`M_hc` gives the finite-range amplitudes `c_a`; the eliminated onsite source
is recovered as `d0 = -Σ_b Π_0b V_b c_b / Π_00`, which stays finite as
`U → ∞` while `c_0 → 0`.  Then

    ψ(q) = [d0 + Σ_a V_a φ_a(q) c_a] / (Eb - ℰ_K(q))

and `Ψ(r)` is its Fourier transform on the same mesh.  `Ψ[1,1,1] ≈ 0` is the
sharpest check that the hard core was eliminated and not truncated.
"""
function pair_wavefunction(Eb::Float64, V, e, fx, f2x)
    L = size(e, 1)
    Π00, Π0, Π = Π_all(Eb, e, fx, f2x)
    M, keep = M_hc(Π00, Π0, Π, V)
    F = eigen(M)
    c = F.vectors[:, argmin(abs.(F.values))]

    d0 = -sum(Π0[a]*V[a]*c[i] for (i,a) in enumerate(keep)) / Π00

    ψ = Array{Float64}(undef, L, L, L)
    @inbounds for il in 1:L, ik in 1:L, ih in 1:L
        φ = SVector(fx[ih], f2x[ih], fx[ik], fx[il])
        s = d0
        for (i,a) in enumerate(keep)
            s += V[a]*φ[a]*c[i]
        end
        ψ[ih,ik,il] = s / (Eb - e[ih,ik,il])
    end

    Ψ = real.(ifft_shifted(ψ))
    Ψ ./= sqrt(sum(abs2, Ψ))
    return Ψ, c, d0
end

"""
Real-space transform of a mesh function defined on the symmetric BZ grid
q = -1/2 + m/L.  Writing that shift out, e^{i2πq·r} = (-1)^{r_x+r_y+r_z}
e^{i2πm·r/L}, so this is one `ifft` plus a checkerboard sign — O(L³ log L),
not the O(L⁶) double loop the definition suggests.
"""
function ifft_shifted(ψ)
    L = size(ψ, 1)
    Ψ = ifft(ComplexF64.(ψ))
    @inbounds for rz in 0:L-1, ry in 0:L-1, rx in 0:L-1
        isodd(rx + ry + rz) && (Ψ[rx+1, ry+1, rz+1] *= -1)
    end
    return Ψ
end

"""
    pair_radius(Ψ) -> (ξx, ξy, ξz)

Root-mean-square extent of the pair along each axis, in lattice units, using
the minimum image on the torus.  Diverges as `Δ_b → 0`; the calculation is
converged only when every `ξ_d ≪ L_d`.
"""
function pair_radius(Ψ)
    L = size(Ψ, 1)
    mi(r) = (r + L÷2) % L - L÷2
    s = zeros(3); n = 0.0
    @inbounds for rz in 0:L-1, ry in 0:L-1, rx in 0:L-1
        p = abs2(Ψ[rx+1, ry+1, rz+1])
        n += p
        s[1] += p*mi(rx)^2; s[2] += p*mi(ry)^2; s[3] += p*mi(rz)^2
    end
    return Tuple(sqrt.(s ./ n))
end
