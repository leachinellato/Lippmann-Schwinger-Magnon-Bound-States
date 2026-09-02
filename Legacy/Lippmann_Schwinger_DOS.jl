using LinearAlgebra
using StaticArrays
using Plots

# ═══════════════════════════════════════════════════════════════════════
#  Parameters
# ═══════════════════════════════════════════════════════════════════════

# Couplings in the sign convention of Eq. (Hint):  H_int = Σ V_a n n,
# so V_a < 0 is attractive, V_a > 0 repulsive, V_a = 0 removes the channel.
# Channel order (x, 2x, y, z) of Eq. (Mhc-explicit).
const V̄ = SVector(-18.734, 5.7908, 0.478, -0.0646)
const C = 1                              # channel carrying λ:  V_C = λ V̄_C

kB      = 8.617333262e-5 * 1000          # meV/K
J010_yz = -213.3 * kB
J010_x  = -217.4 * kB
J020    =   67.2 * kB
J100    =   5.55 * kB
J001    =  -0.75 * kB
J131    =   7.44 * kB
Δ       = J010_x / J010_yz
 
function ω(h, k, l; J010_yz, J020, J100, J001, J131, Δ)
    S = 1/2
    A = 2S*( J010_yz*(cos(2π*k) - Δ) + J020*(cos(4π*k) - 1)
           + J100*(cos(2π*h) - 1)    + J001*(cos(2π*l) - 1) ) + 8S*J131
    B = 8S*J131*cos(π*h)*cos(3π*k)*cos(π*l)
    return sqrt(A^2 - B^2)
end
 
ε(h,k,l) = ω(k, h, l; J010_yz, J020, J100, J001, J131, Δ)
#ϵ(h,k,l) = ε(h,k,l) + ε(-h,-k,-l)        # ℰ_K(q) at K = (0,0,0)
 

# ═══════════════════════════════════════════════════════════════════════
#  Kernels
# ═══════════════════════════════════════════════════════════════════════


"""Continuum edge E_th and the gap to the next mesh level (resolution floor)."""
function edge(e)
    Emin, Enext = minimum(e), Inf
    @inbounds for x in e
        x - Emin > 1e-10 && x < Enext && (Enext = x)
    end
    return Emin, Enext - Emin
end

"""Smallest L whose mesh resolves η at the band edge (gap ∝ 1/L²)."""
function L_for(η, ϵ; L0 = 48, safety = 10.0)
    _, gap = edge(first(mesh_cache(L0, ϵ)))
    return ceil(Int, L0*sqrt(safety*gap/η))
end

"""
Projected pair propagator, Eq. (Pi), channels (x, 2x, y, z).
Returns `(Π00, Π0, Π)` with `Π0[a] = Π_a0`, `Π[a,b] = Π_ab`.
"""


"""Hard-core matrix of Eq. (Mhc), restricted to the channels with V_a ≠ 0."""
function M_hc(Π00, Π0, Π, V)
    keep = findall(!iszero, V)
    A    = Matrix(-Π + (Π0*Π0')/Π00)
    return Symmetric(A[keep,keep] + Diagonal([1/V[a] for a in keep])), keep
end

"""Critical coupling from the rank-one identity, Eq. (lambda-c)."""
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
Bound-state energy at fixed λ.

The condition is det M_hc = 0, NOT eigmin = 0: a weakly coupled channel
contributes a large fixed 1/V_a on the diagonal that dominates the smallest
eigenvalue at every energy and hides the crossing.

Sign of det at the edge depends on how many channels are attractive, so the
bracket is found by looking for any sign change rather than assuming one.
η is only the upper bracket endpoint; the mesh gap is what limits Δb.
"""
function E_bound(λ, e, fx, f2x, V̄; c = C, η = 1e-8, Emin = nothing,
                 gap = nothing, itmax = 60)
    V = SVector(ntuple(a -> a == c ? λ*V̄[a] : V̄[a], 4))
    (Emin === nothing || gap === nothing) && ((Emin, gap) = edge(e))
    f(E) = det(first(M_hc(Π_all(E, e, fx, f2x)..., V)))

    Ehi, fhi = Emin - η, 0.0
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
    ok || return nothing                 # no crossing below the edge: unbound

    Eb = bisect(f, Elo, Ehi; tol = 1e-12*max(1, abs(Emin)))
    Δb = Emin - Eb
    Δb < 50gap && @warn "Δb = $Δb is only $(round(Δb/gap, digits=1))× the mesh gap $gap — raise L"
    return Eb, Δb
end


# ═══════════════════════════════════════════════════════════════════════
#  Additions for the pair density of states.
#  Assumes ϵ_1, V̄, C, M_hc, edge, bisect are already defined.
# ═══════════════════════════════════════════════════════════════════════

"""
Cache ℰ_K(q) = ε(K/2+q) + ε(K/2-q) on the L³ mesh, plus the two form-factor
profiles.  `K` in reduced units; the form factors are K-independent because
they belong to the relative coordinate.
"""
function mesh_cache(L::Int, ϵ_1::Function; K = (0.0, 0.0, 0.0))
    grid = collect(-0.5:1/L:(0.5-1/L))
    fx   = [√2*cos(2π*q) for q in grid]
    f2x  = [√2*cos(4π*q) for q in grid]
    e    = Array{Float64}(undef, L, L, L)
    Kh, Kk, Kl = K
    @inbounds for il in 1:L, ik in 1:L, ih in 1:L
        h, k, l = grid[ih], grid[ik], grid[il]
        e[ih,ik,il] = ϵ_1(Kh/2 + h, Kk/2 + k, Kl/2 + l) +
                      ϵ_1(Kh/2 - h, Kk/2 - k, Kl/2 - l)
    end
    return e, fx, f2x
end

#═══════════════════════════════════════════════════════════════════════
#  Pair density of states.  Replaces the earlier Π_all(::ComplexF64),
#  dos and dos_sweep.  mesh_cache is unchanged.
# ═══════════════════════════════════════════════════════════════════════
 
"""
Propagator and its energy derivative at complex z, INCLUDING the hard-core
channel 0 (φ_0 ≡ 1).  Returns the two 5×5 arrays over channels (0, x, 2x, y, z).
 
Working with the full matrix rather than the Schur complement avoids dividing
by Π_00, which vanishes at some real energy inside the band and puts a
spurious pole in M_hc there.
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
                φ  = SVector(1.0, fx[ih], f2x[ih], fy, fz)
                g  = 1.0 / (z - e[ih,ik,il])
                P  = φ * transpose(φ)
                s .+= P .* g
                d .+= P .* (-g*g)
            end
        end
    end
    w = 1/L^3
    return w*SMatrix{5,5,ComplexF64}(s), w*SMatrix{5,5,ComplexF64}(d)
end


# ═══════════════════════════════════════════════════════════════════════
#  Replaces dos / dos_sweep.  Π_full and mesh_cache are unchanged.
# ═══════════════════════════════════════════════════════════════════════
 
"""
Free and interacting quantities at fixed K, all normalised PER STATE
(∫ dE = 1), so that free and interacting curves are directly comparable.
 
Returns `(ρ_free, A_free, A_int, Δρ)`:
 
  ρ_free  = -(1/π) Im Π_00                     ∫ρ_free dE = 1
  A_free  = -(1/π) Im Π_aa   per channel       ∫A_free dE = 1
  A_int   = -(1/π) Im G_aa,  G = Π + Π F⁻¹ Π   ∫A_int  dE = 1 - (bound weight)
  Δρ      = -(1/π) Im Tr[F⁻¹ ∂_E F]            ∫Δρ dE = -(1 + N_bound), ABSOLUTE
 
A_int is the object to compare against A_free.  The total DOS is not: the
interaction has rank ~(number of interacting sites), so the integrated level
count changes by at most that, i.e. by O(1) out of N.  The spectral weight in
a channel, by contrast, is redistributed across the whole continuum.
 
F is complex symmetric, not Hermitian — transposes, never adjoints.
"""
function spectral(E::Float64, e, fx, f2x, V; ε = 0.15)
    Π, dΠ = Π_full(complex(E, ε), e, fx, f2x)
 
    keep = [1; 1 .+ findall(!iszero, V)]        # channel 0 always kept
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
Sweep over an energy window.  Checks:
  ∫A_free dE = 1 per channel   (Lorentzian tails converge slowly — pad the
                                window by ≳ 20ε or the norm falls short)
  ∫Δρ    dE = -(1 + N_bound)
"""
function spectral_sweep(Es, e, fx, f2x, V; ε = 0.15)
    nc = length(findall(!iszero, V)) + 1
    ρf = similar(Es); Δρ = similar(Es)
    A0 = zeros(length(Es), nc); A1 = zeros(length(Es), nc)
    for (i,E) in enumerate(Es)
        ρf[i], a0, a1, Δρ[i] = spectral(E, e, fx, f2x, V; ε = ε)
        A0[i,:] .= a0; A1[i,:] .= a1
    end
    return ρf, A0, A1, Δρ
end
 
# ═══════════════════════════════════════════════════════════════════════
#  Run: free vs interacting continuum, per channel
# ═══════════════════════════════════════════════════════════════════════
 
L = 110
λ = 1.01
V = SVector(ntuple(a -> a == C ? λ*V̄[a] : V̄[a], 4))
ε_tol = 0.008
names = ["0 (hard core)", "x", "2x", "y", "z"][[1; 1 .+ findall(!iszero, V)]]
 

K = (0,0,0)
    e, fx, f2x = mesh_cache(L, ε; K = K)
    Eth, _     = edge(e)
    Es         = collect(range(Eth - 20ε_tol, maximum(e) + 20ε_tol, length = 10_000))
    ρf, A0, A1, Δρ = spectral_sweep(Es, e, fx, f2x, V; ε = ε_tol)
    dE = Es[2] - Es[1]
 
    println("K = ", K, "   E_th = ", round(Eth, digits=4),
            "   ∫ρ_free = ", round(sum(ρf)*dE, digits=4),
            "   ∫Δρ = ", round(sum(Δρ)*dE, digits=4))
    for j in eachindex(names)
        println("    channel ", names[j],
                ":  ∫A_free = ", round(sum(A0[:,j])*dE, digits=4),
                "   ∫A_int = ", round(sum(A1[:,j])*dE, digits=4),
                "   (weight pulled out: ",
                round(1 - sum(A1[:,j])/sum(A0[:,j]), digits=4), ")")
    end
 
    
        p = Plots.plot(Es, A0[:,2], label = "free", xlabel = "E",
                       ylabel = "A(E)", title = "K = $K, channel $(names[2])", xlims=(2.5,5.5), ylims=(0,0.5))
        Plots.plot!(p, Es, A1[:,2], label = "interacting" )
        
        
        Plots.plot(Es, A0[:,2]+A0[:,3]+A0[:,4]+A0[:,5], label = "free", xlabel = "E",
                       ylabel = "A(E)", title = "K = $K, channel $(names[2])", xlims=(2.5,5.5), ylims=(0,2.0))
        Plots.plot!(Es, A1[:,2]+A1[:,3]+A1[:,4]+A1[:,5], label = "interacting" )

 
 
