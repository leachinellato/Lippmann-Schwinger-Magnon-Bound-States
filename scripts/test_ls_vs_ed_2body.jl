# ═══════════════════════════════════════════════════════════════════════════
#  Two-body test: Lippmann–Schwinger against exact diagonalization.
#
#  Both sides must see the SAME model, so this script uses the tight-binding
#  dispersion `eps_tb`, which has a finite-range real-space Hamiltonian.  The
#  spin-wave dispersion cannot be used here — it is not of finite range and
#  no lattice Hamiltonian reproduces it.  For physics, use
#  scripts/bound_states_ls.jl instead.
#
#  The LS mesh and the ED lattice are the same L³ torus, so the agreement
#  should be at the level of the bisection tolerance, NOT at the level of a
#  thermodynamic-limit extrapolation.  A disagreement here is a bug; a
#  disagreement with experiment is physics.
#
#  Run:  julia --project scripts/test_ls_vs_ed_2body.jl
# ═══════════════════════════════════════════════════════════════════════════

include(joinpath(@__DIR__, "..", "src", "setup.jl"))
using Printf

L = 9          # cubic: the LS two-body mesh is L³ and must match the ED torus.
               # L ≥ 9 so the 4th-neighbour x hopping does not alias (2·4 < 9).
λ = 1.20       # above λc so that a pair is bound and there is something to compare

V  = Vλ(λ)
ε  = eps_tb(V)

println("── two-body LS vs ED,  L = $L³,  λ = $λ ──")

# 1. conventions: the ED Hamiltonian's one-magnon dispersion must be ε itself
err = check_conventions(L, L, L, V)
@printf("convention check: max |ε_ED - ε_LS| = %.2e\n", err)

# 2. Lippmann–Schwinger
e, fx, f2x = mesh_cache(L, ε)
Eth, gap   = edge(e)
res = E_bound(V, e, fx, f2x; Emin = Eth, gap = gap, warn_gaps = 0)
res === nothing && error("no bound state at λ = $λ; raise λ")
Eb_LS, Δb_LS = res
@printf("LS:  E_th = %.10f   E_b = %.10f   Δ_b = %.10f\n", Eth, Eb_LS, Δb_LS)

# 3. exact diagonalization of the same lattice
terms = model_terms(L, L, L, V)
r2    = gamma_spectrum(L, L, L, 2, terms; nev = 4)
Eb_ED = minimum(r2.E)
@printf("ED:  E_b = %.10f   (threshold 2ε(Γ) = %.10f)\n", Eb_ED, 2*ε(0.0,0.0,0.0))

# 4. verdict
d = abs(Eb_LS - Eb_ED)
@printf("|E_LS - E_ED| = %.3e   %s\n", d, d < 1e-8 ? "PASS" : "FAIL")

# 5. wave function checks, which the energy alone does not exercise
Ψ, c, d0 = pair_wavefunction(Eb_LS, V, e, fx, f2x)
@printf("hard core   Ψ(0) = %.3e  (must vanish: the onsite channel is\n", Ψ[1,1,1])
@printf("                          eliminated by the Schur complement,\n")
@printf("                          not deleted from the basis)\n")
ξ = pair_radius(Ψ)
@printf("pair radius ξ    = (%.2f, %.2f, %.2f) lattice units", ξ[1], ξ[2], ξ[3])
println("   [converged only if every ξ ≪ $L]")
