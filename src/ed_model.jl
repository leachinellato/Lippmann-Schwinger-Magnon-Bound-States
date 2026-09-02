# ═══════════════════════════════════════════════════════════════════════════
#  ed_model.jl — bridge between the model of model.jl and the Γ-point exact
#  diagonalization engine of ed_gamma.jl.
#
#  The engine works with the XXZ form
#
#      H = Σ_shells Σ_bonds  J_xy (S⁺S⁻ + S⁻S⁺)/2  +  J_z S^z S^z ,
#
#  which on the polarized background is, in hard-core boson language,
#
#      H = Σ_shells Σ_bonds  (J_xy/2)(b†b + h.c.)  +  J_z n n   +  const,
#
#  plus a one-body piece -J_z per shell per magnon (see `interaction_shift`).
#  Therefore
#
#      J_xy = coefficient of cos(q·δ) in eps_tb_bare      (NOT twice it)
#      J_z  = V_a                                          (same sign convention)
#
#  and the single-magnon energy of the resulting Hamiltonian, relative to the
#  polarized state, is eps_tb_bare(q) - Σ_a V_a = eps_tb(V)(q).  Every test
#  script asserts this before comparing anything.
# ═══════════════════════════════════════════════════════════════════════════

using .GammaMagnons
using LinearAlgebra
using SparseArrays
using KrylovKit
using Printf

"""
    model_terms(Nx, Ny, Nz, V; hop = hoppings_tb()) -> Vector{BondTerm}

Bond terms for the tight-binding model at coupling vector `V`, channel order
(x, 2x, y, z).  The chain is along **x**: the four hopping shells `TX` sit on
the x bonds and `V[1]`, `V[2]` on the first and second x shells.
"""
function model_terms(Nx::Int, Ny::Int, Nz::Int, V; hop = hoppings_tb())
    Vmap = Dict((1,0,0) => V[1], (2,0,0) => V[2],
                (0,1,0) => V[3], (0,0,1) => V[4])
    terms = BondTerm[]
    for (δ, t) in sort(collect(hop); by = first)
        push!(terms, BondTerm(shell_bonds(Nx, Ny, Nz, δ), t, get(Vmap, δ, 0.0)))
    end
    for (δ, v) in sort(collect(Vmap); by = first)     # channels with V but no hopping
        (v == 0 || haskey(hop, δ)) && continue
        push!(terms, BondTerm(shell_bonds(Nx, Ny, Nz, δ), 0.0, v))
    end
    return terms
end

"""
    shell_offset(bond, Nx, Ny, Nz) -> NTuple{3,Int}

Minimum-image displacement of one bond, used to recover a shell's offset from
its bond list.
"""
function shell_offset(bond::Tuple{Int,Int}, Nx::Int, Ny::Int, Nz::Int)
    ax, ay, az = site_coords(bond[1], Nx, Ny)
    bx, by, bz = site_coords(bond[2], Nx, Ny)
    mi(d, N) = (d + N ÷ 2 + N) % N - N ÷ 2
    return (mi(bx - ax, Nx), mi(by - ay, Ny), mi(bz - az, Nz))
end

"""
    one_magnon_dispersion(Nx, Ny, Nz, terms) -> Function

Single-magnon energy of the ED Hamiltonian, relative to the polarized state,
derived from the bond terms themselves rather than assumed.  Each shell `±δ`
contributes `J_xy cos(q·δ)` from the two hops and `-J_z` from the Ising
one-body piece.

Comparing this against `eps_tb(V)` is the convention check: it catches a wrong
hopping normalization, a wrong Ising shift, and — because it is a function of
q rather than a single number — a chain assigned to the wrong axis.
"""
function one_magnon_dispersion(Nx::Int, Ny::Int, Nz::Int, terms::Vector{BondTerm})
    shells = [(shell_offset(first(t.bonds), Nx, Ny, Nz), t.Jxy, t.Jz) for t in terms]
    return (h, k, l) -> sum(Jxy*cos(2π*(d[1]*h + d[2]*k + d[3]*l)) - Jz
                            for (d, Jxy, Jz) in shells)
end

"""
    check_conventions(Nx, Ny, Nz, V; L = 8, tol = 1e-10) -> Float64

Assert that the ED Hamiltonian's single-magnon dispersion agrees with
`eps_tb(V)` on an `L³` sample of the Brillouin zone.  Returns the maximum
deviation; throws if it exceeds `tol`.  Run this before any comparison.
"""
function check_conventions(Nx::Int, Ny::Int, Nz::Int, V; L::Int = 8, tol = 1e-10)
    terms = model_terms(Nx, Ny, Nz, V)
    εED   = one_magnon_dispersion(Nx, Ny, Nz, terms)
    εLS   = eps_tb(V)
    err   = 0.0
    for n3 in 0:L-1, n2 in 0:L-1, n1 in 0:L-1
        q = (n1/L, n2/L, n3/L)
        err = max(err, abs(εED(q...) - εLS(q...)))
    end
    err > tol && error("ED and LS single-magnon dispersions disagree by $err " *
                       "— check the chain axis and the J_xy / J_z conventions")
    return err
end

"""
    gamma_spectrum(Nx, Ny, Nz, n, terms; nev, ...) -> NamedTuple

Lowest `nev` energies of the `n`-magnon Γ sector, returned RELATIVE TO THE
POLARIZED STATE (`E = vals - E_vac`), which is the same zero as the LS
solvers use once they are fed `eps_tb(V)`.
"""
function gamma_spectrum(Nx::Int, Ny::Int, Nz::Int, n::Integer,
                        terms::Vector{BondTerm};
                        nev::Int = 8, matrixfree = :auto,
                        mem_limit_GB::Real = 4.0,
                        krylovdim::Int = max(150, 2nev + 10),
                        tol::Real = 1e-12, verbose::Bool = true)
    t0 = time()
    b  = build_gamma_basis(Nx, Ny, Nz, n)
    ht = build_hop_table(b.L, terms)
    t_basis = time() - t0

    dim = length(b)
    nev = min(nev, max(dim - 1, 1))
    est_GB = dim * (n * ht.maxdeg + 1) * 12 / 2^30
    mf = matrixfree === :auto ? (est_GB > mem_limit_GB) : Bool(matrixfree)

    if verbose
        @printf("  %d-magnon Γ, %d×%d×%d (L = %d)   dim = %d   H ≈ %.2f GB → %s\n",
                n, Nx, Ny, Nz, b.L, dim, est_GB, mf ? "matrix-free" : "stored")
    end

    t1 = time()
    x0 = randn(dim)
    if mf
        vals, vecs, info = eigsolve(GammaOperator(b, ht), x0, nev, :SR;
                                    ishermitian = true, krylovdim = krylovdim, tol = tol)
        Hs = nothing
    else
        Hs = xxz_gamma(b, ht)
        vals, vecs, info = eigsolve(Hs, x0, nev, :SR;
                                    ishermitian = true, krylovdim = krylovdim, tol = tol)
    end
    verbose && @printf("  eigsolve %.2f s, converged %d/%d\n",
                       time() - t1, info.converged, nev)
    info.converged < nev &&
        @warn "only $(info.converged)/$nev eigenpairs converged — raise krylovdim or tol"

    return (E = real.(vals) .- ht.E_vac, vecs = vecs, basis = b, ht = ht,
            H = Hs, dim = dim, info = info, n = n, L = b.L)
end

"""
    gamma_full_spectrum(Nx, Ny, Nz, n, terms) -> (E, vectors, basis)

Dense diagonalization of the whole Γ sector.  Needed for the spectral
functions of ed_spectral.jl, which require every eigenvector; only feasible
for small lattices in the two-magnon sector.
"""
function gamma_full_spectrum(Nx::Int, Ny::Int, Nz::Int, n::Integer,
                             terms::Vector{BondTerm}; verbose::Bool = true)
    b  = build_gamma_basis(Nx, Ny, Nz, n)
    ht = build_hop_table(b.L, terms)
    verbose && @printf("  dense %d-magnon Γ, dim = %d\n", n, length(b))
    F = eigen(Symmetric(Matrix(xxz_gamma(b, ht))))
    return F.values .- ht.E_vac, F.vectors, b
end

"""Binding energy report against the `n`-magnon threshold `n·ε(Γ)`."""
function report(E, n, εΓ)
    thr, Emin = n*εΓ, minimum(E)
    @printf("  threshold %dε(Γ) = %.10f\n", n, thr)
    @printf("  E_min            = %.10f\n", Emin)
    @printf("  binding          = %+.10f   %s\n", thr - Emin,
            thr - Emin > 0 ? "BOUND" : "(unbound)")
    return (thr = thr, Emin = Emin, Ebind = thr - Emin)
end
