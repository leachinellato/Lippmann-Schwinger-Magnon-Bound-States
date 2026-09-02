# ═══════════════════════════════════════════════════════════════════════════
#  run_magnon_bound_states.jl
#
#  Two- and three-magnon bound states at Γ for the anisotropic XXZ chain-
#  coupled model, using the O(1)-canonicalization Γ engine.
#
#  Adapted from the original driver.  Physics, couplings, and the LSWT
#  coefficient fit are unchanged.  What changed:
#
#    old                                     new
#    ─────────────────────────────────────   ────────────────────────────────
#    build_rep_table_fast_3d(_threemagnon)   (gone — no O(D) table at all)
#    MomentumBasis3D_{Two,Three}Magnon       build_gamma_basis(Nx,Ny,Nz,n)
#    make_XXZ_k3d_{two,three}magnon          xxz_gamma / GammaOperator
#    spzeros(ComplexF64) + H[i,j] +=         direct threaded CSC assembly
#    sum over per-direction Hamiltonians     one pass over all BondTerms
#    H_k .+= h*(L/2-n)*I                     constant, folded into E_rel
#
#  Energies are returned RELATIVE TO THE FM STATE, i.e. the old
#      E2 = vals_2 .- E0()
#  becomes
#      E_rel = vals .- ht.E_vac .- n*h
#  which is algebraically identical:
#      E0() = 0.25*L*(Jyz1+Jyz2) + 0.5*h*L = ht.E_vac + h*L/2
#      vals(old) = vals(new) + h*(L/2 - n)
#
#  Run:  julia --project -t auto run_magnon_bound_states.jl
# ═══════════════════════════════════════════════════════════════════════════

using Printf
using LinearAlgebra
using SparseArrays
using KrylovKit
using Plots
using JLD2
using MKL 
using LaTeXStrings

include("gamma_magnons_3D.jl")
# Only needed for the bond generators (bonds_from_offsets_3d,
# cubic_bonds_by_direction) and for the optional cross-validation below.
include("two_magnon_basis_3D.jl")
using .GammaMagnons



λ = 1.725


λ = 1.5
λ = 1.0

λ = 2.1

tx1 = -18.822
tx2 = 5.479
tx3 = -0.308
tx4 = -0.1586

ty1 = -0.2867
ty2 = 0.022

tz1 = -0.8131
tz2 = -0.074

V1 = 12.943 * λ
V2 = 5.7908


ϵ_1(h,k,l) = (tx1 * cos(2π*h) + tx2 * cos(2π*h*2) + tx3 * cos(2π*h*3) + tx4 * cos(2π*h*4)
            + ty1 * cos(2π*k) + ty2 * cos(2π*k*2)
            + tz1 * cos(2π*l) + tz2 * cos(2π*l*2) )

ϵ(h,k,l) = ϵ_1(h,k,l) + ϵ_1(-h,-k,-l) #Assuming that K = (0,0,0)



function model_terms(Nx::Int, Ny::Int, Nz::Int)
    bx,bx2, by, by2, by3, by4, bz, bz2 = cubic_bonds_by_direction(Nx, Ny, Nz)
    return [BondTerm(bx,  ty1,  0.0),
            BondTerm(bx2,  ty2,  0.0),
            BondTerm(by,  tx1, -V1),
            BondTerm(by2, tx2,  V2),
            BondTerm(by3, tx3,  0.0),
            BondTerm(by4, tx4,  0.0),
            BondTerm(bz,  tz1,  0.0),
            BondTerm(bz2,  tz2,  0.0)]
end

function gamma_spectrum(Nx::Int, Ny::Int, Nz::Int, n::Integer,
                        terms::Vector{BondTerm}, h::Real;
                        nev::Int = 8,
                        matrixfree = :auto,
                        mem_limit_GB::Real = 4.0,
                        krylovdim::Int = max(150, 2nev + 10),
                        tol::Real = 1e-12,
                        verbose::Bool = true)

    t0 = time()
    b  = build_gamma_basis(Nx, Ny, Nz, n)
    ht = build_hop_table(b.L, terms)
    t_basis = time() - t0

    dim = length(b)
    nev = min(nev, max(dim - 1, 1))

    est_GB = dim * (n * ht.maxdeg + 1) * 12 / 2^30
    mf = matrixfree === :auto ? (est_GB > mem_limit_GB) : Bool(matrixfree)

    if verbose
        @printf("\n── %d-magnon Γ, %d×%d×%d (L = %d) ──\n", n, Nx, Ny, Nz, b.L)
        @printf("   dim(Γ)   = %d\n", dim)
        @printf("   basis    = %.2f s\n", t_basis)
        @printf("   H est.   = %.2f GB  → %s\n", est_GB,
                mf ? "matrix-free" : "stored CSC")
    end

    t1 = time()
    x0 = randn(dim)
    if mf
        A = GammaOperator(b, ht)
        vals, vecs, info = eigsolve(A, x0, nev, :SR;
                                    ishermitian = true, krylovdim = krylovdim, tol = tol)
        Hs = nothing
    else
        Hs = xxz_gamma(b, ht)
        vals, vecs, info = eigsolve(Hs, x0, nev, :SR;
                                    ishermitian = true, krylovdim = krylovdim, tol = tol)
    end
    t_solve = time() - t1

    if verbose
        mf || @printf("   nnz      = %d  (%.2f GB actual)\n", nnz(Hs), nnz(Hs) * 12 / 2^30)
        @printf("   eigsolve = %.2f s   converged = %d / %d\n",
                t_solve, info.converged, nev)
        info.converged < nev &&
            @warn "only $(info.converged)/$nev eigenpairs converged — raise krylovdim or tol"
    end

    # E_FM = ht.E_vac + h*L/2 ;  Zeeman on the n-magnon sector = h*(L/2 - n)
    E_rel = real.(vals) .- ht.E_vac .- n * h

    return (E = E_rel, vals = vals, vecs = vecs, basis = b, ht = ht,
            H = Hs, dim = dim, info = info,
            t_basis = t_basis, t_solve = t_solve, n = n,
            Nx = Nx, Ny = Ny, Nz = Nz, L = b.L)
end

function Full_diagonalization_spectrum(Nx::Int, Ny::Int, Nz::Int, n::Integer,
                        terms::Vector{BondTerm}, h::Real;
                        verbose::Bool = true)

    b  = build_gamma_basis(Nx, Ny, Nz, n)
    ht = build_hop_table(b.L, terms)
    dim = length(b)
    

    if verbose
        @printf("\n── %d-magnon Γ, %d×%d×%d (L = %d) ──\n", n, Nx, Ny, Nz, b.L)
        @printf("   dim(Γ)   = %d\n", dim)
    end


    
    Hs = xxz_gamma(b, ht)
    vals, vects = eigen(Hermitian(Matrix(Hs)))
    
    
    E_rel = real.(vals) .- ht.E_vac .- n * h

    return E_rel, vects
end

"""Binding energy report against the n-magnon continuum threshold n·ω(Γ)."""
function report(res, ω_gamma)
    thr  = res.n * ω_gamma
    Emin = minimum(res.E)
    Ebin = thr - Emin
    @printf("   threshold %dω(Γ) = %.6f meV\n", res.n, thr)
    @printf("   E_min            = %.6f meV\n", Emin)
    @printf("   E_bind           = %+.6f meV   %s\n", Ebin,
            Ebin > 0 ? "← BOUND" : "(unbound)")
    return (thr = thr, Emin = Emin, Ebin = Ebin)
end


function Π_all(E::Float64, L::Int, ϵ::Function)
    s00 = s_x0 = s_2x0 = s_xx = s_2x2x = s_x2x = 0.0
    grid = -0.5:1/L:(0.5-1/L)          # one BZ, exactly L points
    for h in grid
        fx  = √2*cos(2π*h)
        f2x = √2*cos(2π*2h)
        G = 0.0
        for k in grid, l in grid
            G += 1.0 / (E - ϵ(h,k,l))
        end
        s00 += G; s_x0 += fx*G; s_2x0 += f2x*G
        s_xx += fx*fx*G; s_2x2x += f2x*f2x*G; s_x2x += fx*f2x*G
    end
    w = 1/L^3
    return (Π_00=w*s00, Π_x0=w*s_x0, Π_2x0=w*s_2x0,
            Π_xx=w*s_xx, Π_2x2x=w*s_2x2x, Π_x2x=w*s_x2x)
end

function E_bound(λ, L, ϵ; η = 1e-3, itmax = 60)
    Emin = ϵ(0,0,0)
    f(E) = minimum(eigvals(M_from_Π(λ, Π_all(E, L, ϵ))))

    Ehi = Emin - η
    f(Ehi) ≤ 0 && return nothing          # no root below threshold: unbound at this λ

    Elo = Emin - 2η                        # walk down until the sign flips
    ok = false
    for _ in 1:itmax
        if f(Elo) < 0
            ok = true
            break
        end
        Elo = Emin - 2*(Emin - Elo)        # double the distance below threshold
    end
    ok || error("no bracket found down to E = $Elo")

    Eb = bisect(f, Elo, Ehi; tol = 1e-10)
    return Eb, Emin - Eb                   # (bound energy, binding energy Δb)
end


function M_from_Π(λ, Π; V1 = 12.943, V2 = 5.7908)
    M11 = -1/(λ*V1) - Π[4] + Π[2]^2  / Π[1]
    M21 = -Π[6] + Π[2]*Π[3] / Π[1]
    M22 =  1/(V2) - Π[5] + Π[3]^2 / Π[1]
    return [M11 M21; M21 M22]
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
    w_channel(evecs, ia)
 
Weight |⟨a|n⟩|² of channel `a` in each eigenvector, where `ia` is the row index
of |a⟩ in the symmetrized relative-coordinate basis. Sums to 1 over all n.
"""
w_channel(evecs::AbstractMatrix, ia::Integer) = abs2.(@view evecs[ia, :])
 
"""
    A_channel(Egrid, evals, evecs, ia; η=0.05)
 
Channel-projected spectral function
 
    A_a(E) = -Im G_aa(E+iη)/π = Σ_n |⟨a|n⟩|² (η/π) / ((E-E_n)² + η²)
 
i.e. the Lorentzian-broadened *weight* histogram. This is identically the
LS quantity evaluated at the same η, so the two curves should overlay.
"""
function A_channel(Egrid::AbstractVector, evals::AbstractVector,
                   evecs::AbstractMatrix, ia::Integer; η::Real=0.05)
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
    ipr(evecs, mult)
 
Inverse participation ratio of each eigenvector in the *full* relative-coordinate
lattice, Σ_r |ψ(r)|⁴ / (Σ_r |ψ(r)|²)².
 
`mult[i]` is the multiplicity of representative i in the symmetrized basis
|r,+⟩ = (|r⟩+|-r⟩)/√2, i.e. 1 for r = 0 and 2 otherwise:
 
    mult = [all(iszero, r) ? 1.0 : 2.0 for r in reps]
 
Without this correction the IPR is wrong by an r-dependent factor, because the
physical amplitude on each site of a star is c_i/√m_i, not c_i.
"""
function ipr(evecs::AbstractMatrix, mult::AbstractVector)
    out = zeros(Float64, size(evecs, 2))
    @inbounds for n in axes(evecs, 2)
        s = 0.0
        nrm = 0.0
        for i in axes(evecs, 1)
            p = abs2(evecs[i, n])
            nrm += p
            s += p * p / mult[i]
        end
        out[n] = s / (nrm * nrm)
    end
    return out
end
 
"""
    binned(E, y, edges)
 
Mean of `y` in each energy bin. Use to smooth the (noisy) per-state IPR into a
readable IPR-vs-energy curve.
"""
function binned(E::AbstractVector, y::AbstractVector, edges::AbstractVector)
    nb = length(edges) - 1
    s = zeros(Float64, nb)
    c = zeros(Int, nb)
    @inbounds for k in eachindex(E)
        j = searchsortedlast(edges, E[k])
        (1 <= j <= nb) || continue
        s[j] += y[k]
        c[j] += 1
    end
    return [c[j] > 0 ? s[j]/c[j] : NaN for j in 1:nb]
end


"""Index of the Γ basis state carrying relative separation `r`."""
function channel_index(b::GammaBasis{2}, r::NTuple{3,Int})
    f(v) = mod(v[1],b.Nx) + b.Nx*mod(v[2],b.Ny) + b.Nx*b.Ny*mod(v[3],b.Nz) + 1
    s, s2 = f(r), f((-r[1], -r[2], -r[3]))
    α = findfirst(t -> Int(t[2]) == s || Int(t[2]) == s2, b.reps)
    α === nothing && error("separation $r absent from basis")
    return α
end



#LS approach

L = 24
res = E_bound(λ, L, ϵ)
if res === nothing
    println("λ = $λ: unbound")
else
    Eb, Δb = res
    println("λ = $λ:  E_b = $Eb   Δ_b = $Δb")
end

#ED approach

#ωΓ = ϵ_1(0.0, 0.0, 0.0)
ωΓ = ϵ_1(0.0, 0.0, 0.0) + V1 - V2

# ── two magnons, 24³ ──
terms2 = model_terms(L, L, L)
#res2 = gamma_spectrum(L,L,L, 2, terms2, 0.0; nev = 8)
#r2   = report(res2, ωΓ)


b, H = xxz_gamma(L,L,L, 2, terms2)
ht   = build_hop_table(b.L, terms2)
F    = eigen(Hermitian(Matrix(H)))
Ev   = F.values .- ht.E_vac .- 2*ωΓ
Vv   = F.vectors

mult = [2.0 / Int(s) for s in b.stab]
ix   = channel_index(b, (1,0,0))     # your four LS channels:
i2x  = channel_index(b, (2,0,0))
iy   = channel_index(b, (0,1,0))
iz   = channel_index(b, (0,0,1))

Egrid = range(minimum(Ev).-0.2, maximum(Ev); length=4000)
A_x   = A_channel(Egrid, Ev, Vv, ix; η=0.1)
A_2x   = A_channel(Egrid, Ev, Vv, i2x; η=0.05)
A_y   = A_channel(Egrid, Ev, Vv, iy; η=0.05)
A_z   = A_channel(Egrid, Ev, Vv, iz; η=0.05)
I_all = ipr(Vv, mult)

plot(Egrid, 10*A_x, label=L"A_x")
scatter!(Ev, I_all, label=L"IPR")

plot(Egrid, 10*A_x, label=L"A_x", xlims=(minimum(Egrid), abs(ωΓ)))
scatter!(Ev, I_all, label=L"IPR")
vline!([ωΓ])

plot(Egrid, 10*A_2x, label=L"A_2x")
scatter!(Ev, I_all, label=L"IPR")

plot(Egrid, A_y, label=L"A_y")
scatter!(Ev, I_all, label=L"IPR")

plot(Egrid, A_z, label=L"A_z")
scatter!(Ev, I_all, label=L"IPR")

sum(w_channel(Vv, iy)) #Checked for all! (= 1)


r_of(α) = (v = collect(site_coords(Int(b.reps[α][2]), b.Nx, b.Ny));
           ((v[1]+b.Nx÷2)%b.Nx - b.Nx÷2,
            (v[2]+b.Ny÷2)%b.Ny - b.Ny÷2,
            (v[3]+b.Nz÷2)%b.Nz - b.Nz÷2))

n = 1                                        # bound state
p = abs2.(Vv[:, n])
rr = r_of.(1:length(b))
@show sum(p .* [r[1]^2 for r in rr]), sum(p .* [r[2]^2 for r in rr]), sum(p .* [r[3]^2 for r in rr])



#Three bodies
L=12
terms2 = model_terms(L, L, L)
res3 = gamma_spectrum(L,L,L, 3, terms2, 0.0; nev = 8)
r3   = report(res3, ωΓ)

E3 = res3.n * ωΓ .- res3.E

sqrt(E3[1] / E3[2])
sqrt(E3[2] / E3[3])
sqrt(E3[3] / E3[4])

scatter(E3)







