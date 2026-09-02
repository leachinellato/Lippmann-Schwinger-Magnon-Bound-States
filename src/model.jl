# ═══════════════════════════════════════════════════════════════════════════
#  model.jl — couplings, dispersions and conventions shared by every solver.
#
#  CONVENTIONS (all code in this repository follows them)
#
#  Axes.  In the crystal the chains run along y.  Everywhere in this
#  repository the axes are rotated so that THE CHAIN IS ALONG x, i.e. the
#  first momentum slot h and the first lattice direction.  The only place the
#  crystal axes survive is inside `omega`, which is written as published; the
#  rotation is applied once, in `eps_sw`.
#
#  Momenta.  Fractional coordinates (h,k,l); a physical momentum component is
#  2π times a fractional one.
#
#  Interactions.  H_int = Σ_a V_a n n, so V_a < 0 attracts, V_a > 0 repels,
#  and V_a = 0 removes the channel entirely.  Channel order is (x, 2x, y, z)
#  everywhere, matching RS4 in ls_three_body.jl.
#
#  Two dispersions are defined here and they are NOT interchangeable:
#
#    eps_sw   linear spin-wave, ε = sqrt(A²-B²).  Not of finite range in real
#             space, so it cannot be fed to exact diagonalization.  This is
#             the physical dispersion; use it for every production run.
#
#    eps_tb   tight-binding fit to eps_sw, a finite cosine series.  A real
#             space Hamiltonian exists for it, so ED and the LS solvers can
#             be run on exactly the same model.  Use it only for the
#             LS-versus-ED tests.
# ═══════════════════════════════════════════════════════════════════════════

using StaticArrays

const kB = 8.617333262e-2                # meV/K

# ── exchange constants (Zoghlin et al.), meV ───────────────────────────────
const J010_yz = -213.3 * kB
const J010_x  = -217.4 * kB
const J020    =   67.2 * kB
const J100    =    5.55 * kB
const J001    =   -0.75 * kB
const J131    =    7.44 * kB
const Δ       = J010_x / J010_yz

# ── pair couplings, channel order (x, 2x, y, z) ────────────────────────────
const V̄ = SVector(-18.734, 5.7908, 0.478, -0.0646)
const C = 1                              # channel carrying λ:  V_C = λ V̄_C

"""Coupling vector at tuning parameter `λ`, channel order (x, 2x, y, z)."""
Vλ(λ; c = C) = SVector(ntuple(a -> a == c ? λ * V̄[a] : V̄[a], 4))

# ── linear spin-wave dispersion ────────────────────────────────────────────

"""
    omega(h, k, l) -> Float64

Single-magnon energy in the **crystal** axes, chains along y, exactly as
published.  Do not call this directly; call `eps_sw`.
"""
function omega(h, k, l; J010_yz = J010_yz, J020 = J020, J100 = J100,
                        J001 = J001, J131 = J131, Δ = Δ)
    S = 1/2
    A = 2S*( J010_yz*(cos(2π*k) - Δ) + J020*(cos(4π*k) - 1)
           + J100*(cos(2π*h) - 1)    + J001*(cos(2π*l) - 1) ) + 8S*J131
    B = 8S*J131*cos(π*h)*cos(3π*k)*cos(π*l)
    return sqrt(A^2 - B^2)
end

"""
    eps_sw(h, k, l) -> Float64

Single-magnon dispersion in the **repository** axes: chain along x, i.e. the
first argument.  This is `omega` with its first two arguments swapped.
"""
eps_sw(h, k, l) = omega(k, h, l)

# ── tight-binding fit (ED-compatible) ──────────────────────────────────────

const TX = (-18.822, 5.479, -0.308, -0.1586)   # shells 1..4 along x (chain)
const TY = (-0.2867, 0.022)                    # shells 1..2 along y
const TZ = (-0.8131, -0.074)                   # shells 1..2 along z

"""
    eps_tb_bare(h, k, l) -> Float64

The bare cosine series fitted to `eps_sw`, chain along x.  This is the
hopping part only: it does not contain the one-body shift that the pair
couplings induce in the spin model, so it is NOT the single-magnon energy of
the ED Hamiltonian.  Use `eps_tb(V)` for that.
"""
eps_tb_bare(h, k, l) =
    ( TX[1]*cos(2π*h) + TX[2]*cos(4π*h) + TX[3]*cos(6π*h) + TX[4]*cos(8π*h)
    + TY[1]*cos(2π*k) + TY[2]*cos(4π*k)
    + TZ[1]*cos(2π*l) + TZ[2]*cos(4π*l) )

"""
    hoppings_tb() -> Dict{NTuple{3,Int},Float64}

Real-space shells of the tight-binding fit, offset => J_xy.  Each entry is one
shell ±δ, and `J_xy` is the coefficient of `cos(q·δ)` in `eps_tb_bare`,
because `BondTerm(bonds, J_xy, J_z)` contributes `J_xy cos(q·δ)` to the
single-magnon energy (see `interaction_shift`).
"""
hoppings_tb() = Dict(
    (1,0,0) => TX[1], (2,0,0) => TX[2], (3,0,0) => TX[3], (4,0,0) => TX[4],
    (0,1,0) => TY[1], (0,2,0) => TY[2],
    (0,0,1) => TZ[1], (0,0,2) => TZ[2])

"""
    interaction_shift(V) -> Float64

One-body energy that the Ising part of the spin Hamiltonian adds to a single
magnon.  With `S^z = 1/2 - n`, a bond term `J_z S^z S^z` contributes
`J_z/4 - J_z(n_a+n_b)/2 + J_z n_a n_b`, so `V_a = J_z` and each magnon picks
up `-V_a` per shell it sits in (two bonds, `-V_a/2` each).  Hence the ED
single-magnon energy is `eps_tb_bare(q) - Σ_a V_a`.
"""
interaction_shift(V) = sum(V)

"""
    eps_tb(V) -> Function

Single-magnon dispersion of the ED Hamiltonian built by `model_terms(V)`, as
a function `(h,k,l) -> ε`.  Equal to the tight-binding fit shifted by
`interaction_shift`, so that LS energies computed with it are directly
comparable with the `E_rel` returned by the ED driver — no offset, no rescaling.
"""
eps_tb(V) = (h, k, l) -> eps_tb_bare(h, k, l) - interaction_shift(V)

# ── pair energy ────────────────────────────────────────────────────────────

"""
    pair_energy(ε, h, k, l; K = (0,0,0)) -> Float64

Two-particle energy `ℰ_K(q) = ε(K/2+q) + ε(K/2-q)` at relative momentum
`q = (h,k,l)`.
"""
pair_energy(ε, h, k, l; K = (0.0, 0.0, 0.0)) =
    ε(K[1]/2 + h, K[2]/2 + k, K[3]/2 + l) +
    ε(K[1]/2 - h, K[2]/2 - k, K[3]/2 - l)
