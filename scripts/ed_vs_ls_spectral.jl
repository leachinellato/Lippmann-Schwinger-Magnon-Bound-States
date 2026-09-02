# ═══════════════════════════════════════════════════════════════════════════
#  Channel spectral function: Lippmann–Schwinger against exact diagonalization.
#
#  A_a(E) computed from the LS propagator and from the ED weight histogram are
#  the same quantity, so at equal broadening and on the same lattice the two
#  curves must overlay.  This is the sharpest test of the continuum machinery:
#  it exercises the retarded continuation, the hard-core factor in F, and the
#  channel projection all at once — none of which the bound-state tests touch.
#
#  Uses the tight-binding dispersion, since ED is involved.  Dense
#  diagonalization is needed (every eigenvector), so keep the lattice small.
#
#  Run:  julia --project scripts/ed_vs_ls_spectral.jl
# ═══════════════════════════════════════════════════════════════════════════

include(joinpath(@__DIR__, "..", "src", "setup.jl"))
using Printf, Plots

L = 9
λ = 1.20
η = 0.15        # must be ≫ the mesh level spacing on a lattice this small,
                # or both curves resolve individual poles

V = Vλ(λ)
ε = eps_tb(V)
names = channel_names(V)

check_conventions(L, L, L, V)

# ── ED side ───────────────────────────────────────────────────────────────
terms  = model_terms(L, L, L, V)
Ev, Vv, b = gamma_full_spectrum(L, L, L, 2, terms)
idx = Dict("x" => channel_index(b, (1,0,0)), "2x" => channel_index(b, (2,0,0)),
           "y" => channel_index(b, (0,1,0)), "z" => channel_index(b, (0,0,1)))

# ── LS side, same lattice as the mesh ─────────────────────────────────────
e, fx, f2x = mesh_cache(L, ε)
Es = collect(range(minimum(Ev) - 20η, maximum(Ev) + 20η, length = 3000))
_, A0, A1, _ = spectral_sweep(Es, e, fx, f2x, V; ε = η)

# ── overlay ───────────────────────────────────────────────────────────────
ps = []
for (j, nm) in enumerate(names[2:end])
    A_ed = A_channel(Es, Ev, Vv, idx[nm]; η = η)
    p = plot(Es, A1[:, j+1], label = "LS", xlabel = "E (meV)", ylabel = "A(E)",
             title = "channel $nm")
    plot!(p, Es, A_ed, label = "ED", linestyle = :dash)
    @printf("channel %-3s   max |A_LS - A_ED| = %.3e   (peak A ≈ %.3f)\n",
            nm, maximum(abs.(A1[:, j+1] .- A_ed)), maximum(A_ed))
    push!(ps, p)
end
plot(ps..., layout = (2, 2), size = (950, 650))
