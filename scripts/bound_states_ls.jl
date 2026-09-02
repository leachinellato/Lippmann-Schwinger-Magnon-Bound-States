# ═══════════════════════════════════════════════════════════════════════════
#  Bound states of two and three magnons with the PHYSICAL dispersion.
#
#  Here the spin-wave dispersion ε = sqrt(A²-B²) is used directly.  It has no
#  finite-range real-space Hamiltonian, so there is no ED to compare against;
#  the algebra is validated separately by the two test scripts.
#
#  Run:  julia --project scripts/bound_states_ls.jl
# ═══════════════════════════════════════════════════════════════════════════

include(joinpath(@__DIR__, "..", "src", "setup.jl"))
using Printf, Plots

ε = eps_sw
@printf("single-magnon minimum ε(Γ) = %.8f meV\n\n", ε(0.0, 0.0, 0.0))

# ═══════════════════════════════════════════════════════════════════════════
#  (a) critical coupling.  One L for the whole η sequence, large enough for
#      the SMALLEST η, so the fit isolates the η dependence from the mesh
#      error.  Extrapolate in √η, not in η: the band edge is quadratic, so the
#      leading correction to Π is O(√η) and a fit in η lands on the wrong
#      intercept.
# ═══════════════════════════════════════════════════════════════════════════

ηs = [1e-2, 1e-3, 0.5e-3]
L2 = L_for(minimum(ηs), ε)
e, fx, f2x = mesh_cache(L2, ε)
Eth, gap   = edge(e)
@printf("two-body mesh L = %d   E_th = %.8f   mesh gap = %.2e\n", L2, Eth, gap)

λcs = [λ_critical(Π_all(Eth - η, e, fx, f2x)..., V̄) for η in ηs]
for (η, λ) in zip(ηs, λcs)
    @printf("   η = %.1e   gap/η = %6.2f   λ_c = %.8f\n", η, gap/η, λ)
end
λc0 = ([ones(length(ηs)) sqrt.(ηs)] \ λcs)[1]
@printf("λ_c(η → 0) = %.8f\n\n", λc0)

p1 = plot(sqrt.(ηs), λcs, marker = :square, xlabel = "√η", ylabel = "λ_c",
          label = "data", title = "critical coupling")
scatter!(p1, [0.0], [λc0], marker = :diamond, label = "extrapolated")

# ═══════════════════════════════════════════════════════════════════════════
#  (b) two-body binding above λc.  √Δ_b is linear in λ - λc and vanishes at
#      λc: an independent determination of the same number as (a), and the
#      internal consistency check on it.
# ═══════════════════════════════════════════════════════════════════════════

λs  = λc0 .* (1 .+ [0.005, 0.01, 0.02, 0.03])
res = [E_bound(Vλ(λ), e, fx, f2x; Emin = Eth, gap = gap) for λ in λs]
for (λ, r) in zip(λs, res)
    r === nothing ? @printf("   λ = %.6f   unbound\n", λ) :
                    @printf("   λ = %.6f   E_b = %.8f   Δ_b = %.3e\n", λ, r[1], r[2])
end
ok = findall(!isnothing, res)
p2 = plot(λs[ok] .- λc0, sqrt.([res[i][2] for i in ok]), marker = :square,
          xlabel = "λ - λ_c", ylabel = "√Δ_b", label = "data",
          title = "square-root law")

# ═══════════════════════════════════════════════════════════════════════════
#  (c) three magnons at a chosen coupling.  The relevant threshold is the
#      LOWER of complete breakup, 3ε_min, and breakup into one magnon plus a
#      bound pair, E_{1+2}; at λ = λc the two coincide.
#
#      Cost is one symmetric factorization of a (5·N_s)² matrix per bisection
#      step, so bracket on a coarse mesh and pass the bracket to a fine one.
#      An anisotropic mesh with L_x ≠ L_y = L_z buys accuracy at fixed N_s and
#      is preferable to raising a cubic N: the dispersion is strongly
#      anisotropic and a uniform mesh resolves the directions unequally, which
#      shows up as non-monotonic convergence.
# ═══════════════════════════════════════════════════════════════════════════

λ3 = λc0                       # at the pair resonance
L3 = (12, 8, 8)
V3 = Vλ(λ3)

ms = mesh3(L3, ε, collect(V3))
@printf("\nthree-body mesh %s   N_s = %d   dim A₃ = %d   (%.2f GB dense)\n",
        string(L3), ms.Ns, ms.nc*ms.Ns, 8*(ms.nc*ms.Ns)^2/2^30)
@printf("   E_th = 3ε_min = %.8f\n", ms.Eth)

Epair = E_pair_threshold(ms; pset = 1:max(1, ms.Ns ÷ 64):ms.Ns)
@printf("   E_{1+2}       = %s\n",
        Epair === nothing ? "no bound pair (Borromean regime)" : string(Epair))

r3 = E3_bound(ms)
if r3 === nothing
    println("   no three-magnon bound state below threshold")
else
    @printf("   E₃ = %.8f   Δ₃ = %.8f   states below E_th: %d\n",
            r3[1], r3[2], n_bound(ms.Eth - 1e-12, ms))
    if Epair !== nothing && r3[1] > Epair
        @warn "E₃ lies above E_{1+2}: this is a magnon scattering off a bound " *
              "pair, not a genuine three-body bound state"
    end
end

plot(p1, p2, layout = (1, 2), size = (900, 380))
