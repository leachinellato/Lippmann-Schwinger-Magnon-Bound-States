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
ϵ(h,k,l) = ε(h,k,l) + ε(-h,-k,-l)        # ℰ_K(q) at K = (0,0,0)
 



tx1, tx2, tx3, tx4 = -18.822, 5.479, -0.308, -0.1586
ty1, ty2 = -0.2867, 0.022
tz1, tz2 = -0.8131, -0.074

ϵ_1(h,k,l) = (tx1*cos(2π*h) + tx2*cos(4π*h) + tx3*cos(6π*h) + tx4*cos(8π*h)
            + ty1*cos(2π*k) + ty2*cos(4π*k)
            + tz1*cos(2π*l) + tz2*cos(4π*l))

ϵ(h,k,l) = ϵ_1(h,k,l) + ϵ_1(-h,-k,-l)    # ℰ_K(q) at K = (0,0,0)

# ═══════════════════════════════════════════════════════════════════════
#  Kernels
# ═══════════════════════════════════════════════════════════════════════

"""
Cache ℰ_K(q) on the L³ mesh, plus the two distinct form-factor profiles.
φ_x and φ_y and φ_z are all √2cos(2πq) of their own component, so one
length-L table serves three channels; `f2x` serves the 2x channel.

Building this once turns every later Π_all into a pure array sweep.
"""
function mesh_cache(L::Int, ϵ::Function)
    grid = collect(-0.5:1/L:(0.5-1/L))
    fx  = [√2*cos(2π*q) for q in grid]
    f2x = [√2*cos(4π*q) for q in grid]
    e   = Array{Float64}(undef, L, L, L)
    @inbounds for il in 1:L, ik in 1:L, ih in 1:L
        e[ih,ik,il] = ϵ(grid[ih], grid[ik], grid[il])
    end
    return e, fx, f2x
end

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
#  Runs
# ═══════════════════════════════════════════════════════════════════════

# (a) λc → 0 limit.  ONE L for the whole η sequence, large enough for the
#     smallest η, so the fit isolates the η dependence from the mesh error.
#ηs = [1e-1, 1e-2, 1e-3, 0.5e-3]
ηs = [1e-2, 1e-3, 0.5e-3]
L  = L_for(minimum(ηs), ϵ)
e, fx, f2x = mesh_cache(L, ϵ)
Emin, gap  = edge(e)
println("L = ", L, "   E_th = ", Emin, "   gap = ", gap)

λcηs = [λ_critical(Π_all(Emin - η, e, fx, f2x)..., V̄) for η in ηs]
for (η, λ) in zip(ηs, λcηs)
    println("  η = ", η, "   gap/η = ", round(gap/η, sigdigits=3), "   λc = ", λ)
end

λc0 = ([ones(length(ηs)) sqrt.(ηs)] \ λcηs)[1]     # least squares in √η
println("λc(η→0) = ", λc0)

Plots.plot(sqrt.(ηs), λcηs, marker = :square, xlabel = "√η", ylabel = "λ_c")
Plots.scatter!([0.0], [λc0], marker = :diamond, label = "extrapolated")


# (b) Δb ∝ (λ - λc)² :  √Δb linear in λ, vanishing at λc
λs  = λc0 .* (1 .+ [0.005,0.01,0.02,0.03 ])
res = [E_bound(λ, e, fx, f2x, V̄; Emin = Emin, gap = gap) for λ in λs]
for (λ, r) in zip(λs, res)
    println("  λ = ", λ, r === nothing ? "   unbound" : "   Δb = $(r[2])")
end

ok = findall(!isnothing, res)
x  = λs[ok] .- λc0
y  = sqrt.([res[i][2] for i in ok])


Plots.plot(x, y, marker = :square, xlabel = "λ - λc", ylabel = "√Δ_b",
           label = "data", xlims = (0, maximum(x)*1.05), ylims = (0, maximum(y)*1.05))



