# Lippmann–Schwinger magnon bound states

Two- and three-magnon bound states of an anisotropic XXZ model, treated as
hard-core bosons on a cubic lattice, solved by the Lippmann–Schwinger equation
and validated against exact diagonalization.

## Layout

```
src/       library
scripts/   entry points
Legacy/    the original unstructured files, kept for reference
```

## Conventions

These hold everywhere; violating one silently produces plausible wrong
numbers, so they are asserted at runtime wherever possible.

* **The chain is along x.** In the crystal it runs along y; the rotation is
  applied once, inside `eps_sw`. Channel order is `(x, 2x, y, z)`.
* **Interactions.** `H_int = Σ_a V_a n n`, so `V_a < 0` attracts and `V_a = 0`
  removes the channel rather than weakening it.
* **Momenta** are fractional; a physical component is `2π` times a fractional
  one.
* **Two dispersions, not interchangeable.** `eps_sw` is the physical spin-wave
  dispersion `√(A²-B²)`; it is not of finite range, so no lattice Hamiltonian
  reproduces it and ED cannot see it. `eps_tb(V)` is the tight-binding fit
  *plus* the one-body shift the Ising term induces, and is what makes LS and
  ED comparable term by term. Tests use `eps_tb`; physics uses `eps_sw`.

## Scripts

| script | what it does | dispersion |
|---|---|---|
| `test_ls_vs_ed_2body.jl` | two-body LS vs ED on the same torus | `eps_tb` |
| `test_ls_vs_ed_3body.jl` | three-body Faddeev/STM vs ED, plus a toy self-test | `eps_tb` |
| `bound_states_ls.jl` | λ_c, two-body binding, three-magnon state | `eps_sw` |
| `continuum_dos.jl` | free DOS, Krein shift, channel spectral functions | `eps_sw` |
| `ed_vs_ls_spectral.jl` | channel spectral function, LS vs ED | `eps_tb` |

Run any of them with `julia --project scripts/<name>.jl`.

## Library

| file | contents |
|---|---|
| `model.jl` | couplings, `eps_sw`, `eps_tb`, `Vλ`, pair energy |
| `bonds.jl` | bond lists, with the aliasing condition `2\|δ\| < L` enforced |
| `ed_gamma.jl` | `GammaMagnons`: Γ-point ED engine for 2 and 3 magnons |
| `ed_model.jl` | `BondTerm`s from the model, ED drivers, `check_conventions` |
| `ed_spectral.jl` | channel weights, `A_channel`, IPR |
| `ls_two_body.jl` | `Π_all`, `M_hc`, `λ_critical`, `E_bound`, wave function |
| `ls_dos.jl` | `Π_full`, `spectral`, `branch_dos` |
| `ls_three_body.jl` | `mesh3`, `A3!`, inertia counting, `E3_bound` |

`include("src/setup.jl")` loads everything in dependency order.

## Things that will bite

* **Root criterion.** Two-body: `det M_hc = 0`, never `eigmin`; a weakly
  coupled channel puts a large fixed `1/V_a` on the diagonal that dominates
  the smallest eigenvalue at every energy. Three-body: neither — the
  determinant of a `5N_s` matrix overflows, so `E3_bound` counts the negative
  inertia instead, `N₃(E) = ν(-∞) - ν(E)` with `ν(-∞) = N_s·n_attr` known in
  closed form from the signs of the couplings.
* **Extrapolate in √η, not η.** The band edge is quadratic.
* **Mesh aliasing.** Every solver needs `2|δ_d| < L_d`, so the 2x channel
  needs `L_x ≥ 5` and the 4th-neighbour hopping needs `L_x ≥ 9`.
* **Never divide by `Π₀₀`** inside the band; use the full `F`.
* **`F` is complex symmetric, not Hermitian.** Transposes, never adjoints.
* **Normalization.** `ρ₀` and `A_a` are per state, `Δρ` is an absolute count.
  Adding them without dividing `Δρ` by `L³` makes the total DOS go negative.
* **Anisotropy.** A uniform mesh resolves x and y,z unequally and converges
  non-monotonically; prefer `L_x ≠ L_y = L_z`. Agreement between the two
  largest meshes is not an error bar.
