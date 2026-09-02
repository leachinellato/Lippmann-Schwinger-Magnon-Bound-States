# ═══════════════════════════════════════════════════════════════════════════
#  Three-body test: Faddeev / Skornyakov–Ter-Martirosyan against exact
#  diagonalization of three hard-core magnons at Γ.
#
#  As in the two-body test, both sides use the tight-binding dispersion so
#  that a common real-space Hamiltonian exists.  The mesh may be anisotropic
#  here, which is what makes the test affordable: the x direction needs
#  L_x ≥ 9 for the fourth-neighbour hopping, but y and z need only L ≥ 5.
#
#  Run:  julia --project scripts/test_ls_vs_ed_3body.jl
# ═══════════════════════════════════════════════════════════════════════════

include(joinpath(@__DIR__, "..", "src", "setup.jl"))
using Printf

L = (9, 5, 5)   # L_x ≥ 9: 4th-neighbour hopping must not alias (2·4 < 9)
                # L_y, L_z ≥ 5: the 2x channel needs 2·2 < L along its axis,
                # and the y,z channels need 2·1 < L
λ = 1.20

V = Vλ(λ)
ε = eps_tb(V)
Nx, Ny, Nz = L

println("── three-body LS vs ED,  L = $L,  λ = $λ ──")

# 0. a fast independent check of the Faddeev algebra on a toy model, using the
#    self-contained ED inside ls_three_body.jl
println("\nself-test of the Faddeev construction (toy couplings, small lattices):")
selftest3()

# 1. conventions
err = check_conventions(Nx, Ny, Nz, V)
@printf("\nconvention check: max |ε_ED - ε_LS| = %.2e\n", err)

# 2. Lippmann–Schwinger
ms = mesh3(L, ε, collect(V))
@printf("LS:  N_s = %d   n_c = %d   dim A₃ = %d   E_th = %.10f\n",
        ms.Ns, ms.nc, ms.nc*ms.Ns, ms.Eth)
r = E3_bound(ms)
r === nothing && error("no three-body bound state on this mesh at λ = $λ")
E3_LS, Δ3 = r
@printf("     E₃ = %.10f   Δ₃ = %.10f   states below threshold: %d\n",
        E3_LS, Δ3, n_bound(ms.Eth - 1e-12, ms))

# 3. exact diagonalization of the same lattice
terms = model_terms(Nx, Ny, Nz, V)
r3    = gamma_spectrum(Nx, Ny, Nz, 3, terms; nev = 6)
E3_ED = minimum(r3.E)
@printf("ED:  E₃ = %.10f   (threshold 3ε(Γ) = %.10f)\n", E3_ED, 3*ε(0.0,0.0,0.0))

# 4. verdict
d = abs(E3_LS - E3_ED)
@printf("|E_LS - E_ED| = %.3e   %s\n", d, d < 1e-7 ? "PASS" : "FAIL")

# 5. the counting function should reproduce the number of exact states below
#    threshold, not merely the lowest one
nb_LS = n_bound(ms.Eth - 1e-12, ms)
nb_ED = count(<(ms.Eth), r3.E)
@printf("bound-state count below E_th:  LS %d   ED ≥ %d (of %d computed)\n",
        nb_LS, nb_ED, length(r3.E))
