# ═══════════════════════════════════════════════════════════════════════════
#  Two-magnon continuum with the PHYSICAL dispersion: free density of states,
#  Krein spectral shift, and the channel-projected spectral functions.
#
#  Two rules, both enforced in src/ls_dos.jl and both worth restating:
#
#   * Never divide by Π₀₀.  The full matrix F over channels (0, x, 2x, y, z)
#     is the right object; det F = (-Π₀₀) det M_hc, and the extra factor is
#     the hard core leaving the spectrum.  Re Π₀₀ vanishes somewhere inside
#     the band, where M_hc alone has a spurious pole.
#
#   * Compare A_a, not the total DOS.  The interaction has finite rank, so the
#     integrated level count changes by O(1) out of L³ and the free and
#     interacting densities of states are indistinguishable.  What the
#     interaction changes is which levels the pair state overlaps with, and
#     that is what a neutron measurement sees.
#
#  Run:  julia --project scripts/continuum_dos.jl
# ═══════════════════════════════════════════════════════════════════════════

include(joinpath(@__DIR__, "..", "src", "setup.jl"))
using Printf, Plots

ε   = eps_sw
L   = 110
λ   = 1.01
K   = (0.0, 0.0, 0.0)
η   = 0.008          # broadening: implements the i0⁺ AND smooths the mesh
                     # poles.  These compete — reduce η and 1/L together, not
                     # independently.  η below the mesh level spacing ~ W/L³
                     # resolves individual poles and returns noise.

V     = Vλ(λ)
names = channel_names(V)

e, fx, f2x = mesh_cache(L, ε; K = K)
Eth, gap   = edge(e)

# pad well beyond the band edges: Lorentzian tails decay as 1/E², so a tight
# window makes the norms fall short for reasons unrelated to the physics
Es = collect(range(Eth - 20η, maximum(e) + 20η, length = 10_000))
ρ0, A0, A1, Δρ = spectral_sweep(Es, e, fx, f2x, V; ε = η)
dE = Es[2] - Es[1]

@printf("K = %s   E_th = %.6f   mesh gap = %.2e   η = %.3f\n", string(K), Eth, gap, η)
@printf("∫ρ₀ dE = %.4f   (sum rule: 1)\n", sum(ρ0)*dE)
@printf("∫Δρ dE = %.4f   (sum rule: -(1 + N_b), the 1 being the hard core)\n",
        sum(Δρ)*dE)
for j in eachindex(names)
    @printf("   channel %-14s  ∫A₀ = %.4f   ∫A = %.4f   weight removed %.4f\n",
            names[j], sum(A0[:,j])*dE, sum(A1[:,j])*dE,
            1 - sum(A1[:,j])/sum(A0[:,j]))
end

# free versus interacting, per channel
ps = [plot(Es, A0[:,j], label = "free", xlabel = "E (meV)", ylabel = "A(E)",
           title = "channel $(names[j])", legend = :topright)
      for j in 2:length(names)]
for (i, j) in enumerate(2:length(names))
    plot!(ps[i], Es, A1[:,j], label = "interacting")
end
p_channels = plot(ps..., layout = (2, 2), size = (950, 650))

# spectral shift, with ρ₀ on the same axes: van Hove structure appears in both
# and is not a resonance
p_shift = plot(Es, Δρ, xlabel = "E (meV)", ylabel = "Δρ (states)",
               label = "Δρ", title = "spectral shift")
plot!(twinx(), Es, ρ0, color = :grey, alpha = 0.5, label = "ρ₀", legend = :topleft)

display(p_channels)
display(p_shift)
