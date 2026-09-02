# =============================================================================
#  two_magnon_basis_3D.jl
#
#  Two-magnon sector on a 3D lattice with translational symmetry.
#
#  A two-magnon state = two flipped spins on a fully polarized background,
#  fully specified by a pair of site indices (m₁, m₂) with m₁ < m₂.
#
#  Translations act on site INDICES via modular arithmetic — no bit-encoded
#  states, no BasisInt.  Direct generalization of two_magnon_basis.jl from
#  2D (Nx × Ny) to 3D (Nx × Ny × Nz).
#
#  For a tetragonal lattice, bonds in x, y, z directions carry different
#  couplings Jx, Jy, Jz:
#
#      E0, mx, my, mz, kb, vec = find_ground_state_twomagnon_3d(
#          Nx, Ny, Nz,
#          [(bonds_x, Jx), (bonds_y, Jy), (bonds_z, Jz)]
#      )
#
#  Key data structures:
#    TwoMagnonBasis3D_Lean      — zero-storage basis (pairs computed on the fly)
#    RepTable3D_Fast            — Vector-based lookup table (no Dict overhead)
#    MomentumBasis3D_TwoMagnon  — representatives for a (kx, ky, kz) sector
#
#  Site ordering (consistent with Lattices.jl and TranslationBasis_3D.jl):
#    site(ix, iy, iz) = ix + (iy-1)*Nx + (iz-1)*Nx*Ny
#    where ix ∈ 1:Nx, iy ∈ 1:Ny, iz ∈ 1:Nz
#
#  Dependencies: BondGenerators.jl (for 2D site_index, bond generators)
#                Lattices.jl (for cubic bond generation, optional)
#  Depended on by: user scripts (two-magnon sector calculations on 3D lattices)
# =============================================================================

using SparseArrays
using LinearAlgebra


# ─────────────────────────────────────────────────────────────────
#  3D site-level translations (pure integer arithmetic)
# ─────────────────────────────────────────────────────────────────

"""Map linear site index to (ix, iy, iz), all 1-indexed."""
@inline function site_to_xyz(s::Int, Nx::Int, Ny::Int)
    s0 = s - 1
    iz, rem_xy = divrem(s0, Nx * Ny)
    iy, ix = divrem(rem_xy, Nx)
    return ix + 1, iy + 1, iz + 1
end

"""Map (ix, iy, iz) to linear site index with periodic wrapping."""
@inline function xyz_to_site(ix::Int, iy::Int, iz::Int, Nx::Int, Ny::Int, Nz::Int)
    return mod1(ix, Nx) + (mod1(iy, Ny) - 1) * Nx + (mod1(iz, Nz) - 1) * Nx * Ny
end

"""Translate a single site by (dx, dy, dz) on a 3D torus."""
@inline function translate_site_3d(s::Int, dx::Int, dy::Int, dz::Int,
                                    Nx::Int, Ny::Int, Nz::Int)
    ix, iy, iz = site_to_xyz(s, Nx, Ny)
    return xyz_to_site(ix + dx, iy + dy, iz + dz, Nx, Ny, Nz)
end

"""Translate a pair (i,j) by (dx,dy,dz) and return in canonical order (min, max)."""
@inline function translate_pair_3d(i::Int, j::Int, dx::Int, dy::Int, dz::Int,
                                    Nx::Int, Ny::Int, Nz::Int)
    ti = translate_site_3d(i, dx, dy, dz, Nx, Ny, Nz)
    tj = translate_site_3d(j, dx, dy, dz, Nx, Ny, Nz)
    return ti < tj ? (ti, tj) : (tj, ti)
end


# ─────────────────────────────────────────────────────────────────
#  Pair ↔ linear index mapping  (depends only on L = Nx*Ny*Nz)
#
#  An ordered pair (m1, m2) with 1 ≤ m1 < m2 ≤ L maps to a unique
#  linear index in 1:L(L-1)/2.
# ─────────────────────────────────────────────────────────────────

"""Map an ordered pair (m1 < m2) to a unique linear index in 1:L(L-1)/2."""
@inline function pair_to_linear_3d(m1::Int, m2::Int, L::Int)
    return (m2 - 1) * (m2 - 2) ÷ 2 + m1
end

"""Recover (m1, m2) from a linear index."""
@inline function linear_to_pair_3d(idx::Int, L::Int)
    m2 = floor(Int, (3 + sqrt(8 * idx - 7)) / 2)
    m1 = idx - (m2 - 1) * (m2 - 2) ÷ 2
    return m1, m2
end


# ─────────────────────────────────────────────────────────────────
#  3D bond generators
# ─────────────────────────────────────────────────────────────────

"""
    bonds_from_offsets_3d(Nx, Ny, Nz, offsets) -> Vector{Tuple{Int,Int}}

Generate translationally-invariant bonds on an Nx × Ny × Nz torus from
a list of grid-coordinate offsets (Δix, Δiy, Δiz).

Each offset generates L = Nx×Ny×Nz bonds (one per site), all automatically
periodic.  Self-bonds trigger an error.

# Example: cubic NN bonds
```julia
bonds_x = bonds_from_offsets_3d(Nx, Ny, Nz, [(1, 0, 0)])
bonds_y = bonds_from_offsets_3d(Nx, Ny, Nz, [(0, 1, 0)])
bonds_z = bonds_from_offsets_3d(Nx, Ny, Nz, [(0, 0, 1)])
```
"""
function bonds_from_offsets_3d(Nx::Int, Ny::Int, Nz::Int, offsets)
    bonds = Tuple{Int,Int}[]
    L = Nx * Ny * Nz
    sizehint!(bonds, length(offsets) * L)

    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx
        s = xyz_to_site(ix, iy, iz, Nx, Ny, Nz)
        for (Δix, Δiy, Δiz) in offsets
            t = xyz_to_site(ix + Δix, iy + Δiy, iz + Δiz, Nx, Ny, Nz)
            if s == t
                error("Self-bond detected: offset ($Δix, $Δiy, $Δiz) maps site " *
                      "($ix, $iy, $iz) to itself on a $Nx × $Ny × $Nz torus. " *
                      "Use a larger lattice.")
            end
            push!(bonds, (s, t))
        end
    end
    return bonds
end


"""
    cubic_bonds_by_direction(Nx, Ny, Nz) -> (bonds_x, bonds_y, bonds_z)

Generate NN bonds for a cubic/tetragonal lattice, separated by direction.
Periodic boundary conditions in all directions.

# Example: tetragonal Heisenberg
```julia
bonds_x, bonds_y, bonds_z = cubic_bonds_by_direction(4, 4, 4)
E0, ... = find_ground_state_twomagnon_3d(4, 4, 4,
    [(bonds_x, Jx), (bonds_y, Jy), (bonds_z, Jz)])
```
"""
function cubic_bonds_by_direction(Nx::Int, Ny::Int, Nz::Int)
    bonds_x = bonds_from_offsets_3d(Nx, Ny, Nz, [(1, 0, 0)])
    bonds_x2 = bonds_from_offsets_3d(Nx, Ny, Nz, [(2, 0, 0)])
    bonds_y = bonds_from_offsets_3d(Nx, Ny, Nz, [(0, 1, 0)])
    bonds_y2 = bonds_from_offsets_3d(Nx, Ny, Nz, [(0, 2, 0)])
    bonds_y3 = bonds_from_offsets_3d(Nx, Ny, Nz, [(0, 3, 0)])
    bonds_y4 = bonds_from_offsets_3d(Nx, Ny, Nz, [(0, 4, 0)])
    bonds_z = bonds_from_offsets_3d(Nx, Ny, Nz, [(0, 0, 1)])
    bonds_z2 = bonds_from_offsets_3d(Nx, Ny, Nz, [(0, 0, 2)])
    return bonds_x,bonds_x2, bonds_y,bonds_y2, bonds_y3,bonds_y4 , bonds_z, bonds_z2
end


"""
    cubic_bonds_all(Nx, Ny, Nz) -> Vector{Tuple{Int,Int}}

Generate ALL NN bonds for a cubic lattice (isotropic case).
Total: 3 × Nx × Ny × Nz bonds.
"""
function cubic_bonds_all(Nx::Int, Ny::Int, Nz::Int)
    return bonds_from_offsets_3d(Nx, Ny, Nz, [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
end


# ─────────────────────────────────────────────────────────────────
#  Lean two-magnon basis (zero storage) — 3D version
# ─────────────────────────────────────────────────────────────────

"""
    TwoMagnonBasis3D_Lean

Two-magnon basis for a 3D lattice with no stored pairs and no Dict.
Pairs are computed on the fly via `linear_to_pair_3d`.

Memory: 40 bytes (5 Ints), regardless of system size.
"""
struct TwoMagnonBasis3D_Lean
    Nx :: Int
    Ny :: Int
    Nz :: Int
    L  :: Int
    D  :: Int     # = L*(L-1) ÷ 2
end

function TwoMagnonBasis3D_Lean(Nx::Int, Ny::Int, Nz::Int)
    L = Nx * Ny * Nz
    TwoMagnonBasis3D_Lean(Nx, Ny, Nz, L, L * (L - 1) ÷ 2)
end

Base.length(b::TwoMagnonBasis3D_Lean) = b.D

@inline function get_pair_3d(b::TwoMagnonBasis3D_Lean, idx::Int)
    return linear_to_pair_3d(idx, b.L)
end

function Base.show(io::IO, ::MIME"text/plain", b::TwoMagnonBasis3D_Lean)
    printstyled(io, "TwoMagnonBasis3D_Lean"; bold=true, color=:cyan)
    println(io)
    println(io, "  Lattice    $(b.Nx) × $(b.Ny) × $(b.Nz)  (L = $(b.L))")
    println(io, "  Dimension  $(b.D)")
    println(io, "  Memory     40 bytes")
end


# ─────────────────────────────────────────────────────────────────
#  Vector-based representative lookup table — 3D version
# ─────────────────────────────────────────────────────────────────

"""
    RepTable3D_Fast

Precomputed lookup: each pair (m1, m2) → (representative, nx, ny, nz).
Uses flat Vectors indexed by `pair_to_linear_3d`, with Int16 for
translation distances.

Memory: ~22 bytes per pair ≈ 11 L² bytes total.
"""
struct RepTable3D_Fast
    L      :: Int
    rep_m1 :: Vector{Int}       # representative first index
    rep_m2 :: Vector{Int}       # representative second index
    nx     :: Vector{Int16}     # translation distance x
    ny     :: Vector{Int16}     # translation distance y
    nz     :: Vector{Int16}     # translation distance z
end

"""
    build_rep_table_fast_3d(basis::TwoMagnonBasis3D_Lean) -> RepTable3D_Fast

Build the lookup table by traversing each orbit exactly once.
Translation distances are computed via modular arithmetic.

Cost: O(D × L) where D = L(L-1)/2, L = Nx*Ny*Nz.
"""
function build_rep_table_fast_3d(basis::TwoMagnonBasis3D_Lean)
    Nx, Ny, Nz, L, D = basis.Nx, basis.Ny, basis.Nz, basis.L, basis.D

    rep_m1 = Vector{Int}(undef, D)
    rep_m2 = Vector{Int}(undef, D)
    nx_vec = Vector{Int16}(undef, D)
    ny_vec = Vector{Int16}(undef, D)
    nz_vec = Vector{Int16}(undef, D)

    processed = falses(D)

    for idx in 1:D
        processed[idx] && continue
        m1, m2 = linear_to_pair_3d(idx, L)

        # Generate orbit, find representative (= lexicographic minimum)
        best = (m1, m2)
        ax, ay, az = 0, 0, 0
        orbit = Vector{Tuple{Tuple{Int,Int}, Int, Int, Int}}()
        sizehint!(orbit, L)

        for dz in 0:Nz-1, dy in 0:Ny-1, dx in 0:Nx-1
            p = translate_pair_3d(m1, m2, dx, dy, dz, Nx, Ny, Nz)
            push!(orbit, (p, dx, dy, dz))
            if p < best
                best = p
                ax, ay, az = dx, dy, dz
            end
        end

        # Fill table: T^(nx,ny,nz)|s⟩ = |rep⟩
        for (p, sx, sy, sz) in orbit
            lin = pair_to_linear_3d(p[1], p[2], L)
            processed[lin] && continue
            processed[lin] = true

            rep_m1[lin] = best[1]
            rep_m2[lin] = best[2]
            nx_vec[lin] = mod(ax - sx, Nx)
            ny_vec[lin] = mod(ay - sy, Ny)
            nz_vec[lin] = mod(az - sz, Nz)
        end
    end

    return RepTable3D_Fast(L, rep_m1, rep_m2, nx_vec, ny_vec, nz_vec)
end

"""Look up representative and translation distances from the table. O(1)."""
@inline function lookup_rep_3d(table::RepTable3D_Fast, m1::Int, m2::Int)
    idx = pair_to_linear_3d(m1, m2, table.L)
    return (table.rep_m1[idx], table.rep_m2[idx]),
           Int(table.nx[idx]), Int(table.ny[idx]), Int(table.nz[idx])
end


# ─────────────────────────────────────────────────────────────────
#  Find representative (without table, for basis construction)
# ─────────────────────────────────────────────────────────────────

"""
    find_representative_pair_3d(i, j, Nx, Ny, Nz) -> (rep_pair, dx, dy, dz)

Find the lexicographically smallest pair in the orbit under all
Nx×Ny×Nz translations.  O(L) integer arithmetic.
"""
function find_representative_pair_3d(i::Int, j::Int, Nx::Int, Ny::Int, Nz::Int)
    best = (i, j)
    best_dx, best_dy, best_dz = 0, 0, 0

    for dz in 0:Nz-1, dy in 0:Ny-1, dx in 0:Nx-1
        p = translate_pair_3d(i, j, dx, dy, dz, Nx, Ny, Nz)
        if p < best
            best = p
            best_dx, best_dy, best_dz = dx, dy, dz
        end
    end
    return best, best_dx, best_dy, best_dz
end


# ─────────────────────────────────────────────────────────────────
#  Momentum basis for two-magnon sector — 3D version
# ─────────────────────────────────────────────────────────────────

"""
    MomentumBasis3D_TwoMagnon

Representatives for one (kx, ky, kz) sector of the two-magnon Hilbert space.
"""
struct MomentumBasis3D_TwoMagnon
    parent      :: TwoMagnonBasis3D_Lean
    reps        :: Vector{Tuple{Int,Int}}
    orbit_sizes :: Vector{Int}
    norms       :: Vector{Float64}
    mx          :: Int
    my          :: Int
    mz          :: Int
    kx          :: Float64
    ky          :: Float64
    kz          :: Float64
    Nx          :: Int
    Ny          :: Int
    Nz          :: Int
    L           :: Int
    rep_index   :: Dict{Tuple{Int,Int}, Int}
end

Base.length(kb::MomentumBasis3D_TwoMagnon) = length(kb.reps)

function Base.show(io::IO, ::MIME"text/plain", kb::MomentumBasis3D_TwoMagnon)
    printstyled(io, "MomentumBasis3D_TwoMagnon"; bold=true, color=:cyan)
    println(io)
    println(io, "  Lattice       $(kb.Nx) × $(kb.Ny) × $(kb.Nz)  (L = $(kb.L))")
    println(io, "  (mx, my, mz) = ($(kb.mx), $(kb.my), $(kb.mz))")
    println(io, "  # representatives = $(length(kb.reps))")
end


"""
    MomentumBasis3D_TwoMagnon(basis, mx, my, mz)

Build the k-sector basis for two magnons on a 3D lattice.

Compatibility condition (exact integer arithmetic):
    For every stabilizer element (sx, sy, sz) ∈ S(m1,m2),
    mx·sx·Ny·Nz + my·sy·Nx·Nz + mz·sz·Nx·Ny ≡ 0  (mod L)
"""
function MomentumBasis3D_TwoMagnon(basis::TwoMagnonBasis3D_Lean, mx::Int, my::Int, mz::Int)
    Nx, Ny, Nz, L, D = basis.Nx, basis.Ny, basis.Nz, basis.L, basis.D
    kx = 2π * mx / Nx
    ky = 2π * my / Ny
    kz = 2π * mz / Nz

    processed = falses(D)
    reps        = Tuple{Int,Int}[]
    orbit_sizes = Int[]

    for idx in 1:D
        processed[idx] && continue
        m1, m2 = linear_to_pair_3d(idx, L)

        # Generate orbit
        orbit = Vector{Tuple{Int,Int}}()
        sizehint!(orbit, L)
        for dz in 0:Nz-1, dy in 0:Ny-1, dx in 0:Nx-1
            push!(orbit, translate_pair_3d(m1, m2, dx, dy, dz, Nx, Ny, Nz))
        end

        # Mark all orbit members as processed
        for p in orbit
            lin = pair_to_linear_3d(p[1], p[2], L)
            processed[lin] = true
        end

        rep = minimum(orbit)
        orb_size = length(unique(orbit))

        # Compatibility check: for every stabilizer element, phase must be 1
        # Stabilizer = {(dx,dy,dz) : T^(dx,dy,dz)|s⟩ = |s⟩}
        compatible = true
        for dz in 0:Nz-1, dy in 0:Ny-1, dx in 0:Nx-1
            (dx == 0 && dy == 0 && dz == 0) && continue
            p = translate_pair_3d(m1, m2, dx, dy, dz, Nx, Ny, Nz)
            if p == (m1, m2)   # stabilizer: maps to ITSELF (not to rep!)
                numerator = mx * dx * Ny * Nz +
                            my * dy * Nx * Nz +
                            mz * dz * Nx * Ny
                if mod(numerator, L) != 0
                    compatible = false
                    break
                end
            end
        end

        if compatible
            push!(reps, rep)
            push!(orbit_sizes, orb_size)
        end
    end

    perm = sortperm(reps)
    reps        = reps[perm]
    orbit_sizes = orbit_sizes[perm]
    norms       = sqrt.(Float64.(orbit_sizes))

    rep_index = Dict{Tuple{Int,Int}, Int}()
    for (i, r) in enumerate(reps)
        rep_index[r] = i
    end

    MomentumBasis3D_TwoMagnon(basis, reps, orbit_sizes, norms,
                                mx, my, mz, kx, ky, kz,
                                Nx, Ny, Nz, L, rep_index)
end


# ─────────────────────────────────────────────────────────────────
#  Hamiltonian builder for two-magnon sector — 3D version
#
#  H = Σ_bonds J S⃗ᵢ·S⃗ⱼ  acting on states with exactly 2 flipped
#  spins on a fully-polarized background |↑↑...↑⟩.
#
#  Diagonal (Sz·Sz):
#    +J/4  if both or neither end is flipped
#    -J/4  if exactly one end is flipped
#
#  Off-diagonal (S+S- + S-S+)/2:
#    magnon hops from one end of the bond to the other
#    (only when exactly one end is flipped)
#
#  Phase factor: e^{i(kx·nx + ky·ny + kz·nz)}
# ─────────────────────────────────────────────────────────────────

"""
    make_Heisenberg_k3d_twomagnon(kb, bonds, table; J=1.0) -> SparseMatrix

Build the Heisenberg Hamiltonian in a 3D momentum sector for the
two-magnon problem.  Uses precomputed `RepTable3D_Fast` for O(1)
representative lookups.

`bonds` = Vector{Tuple{Int,Int}} — bond list for one direction.

For anisotropic couplings, call this once per direction and sum:
```julia
H_k  = make_Heisenberg_k3d_twomagnon(kb, bonds_x, table; J=Jx)
H_k .+= make_Heisenberg_k3d_twomagnon(kb, bonds_y, table; J=Jy)
H_k .+= make_Heisenberg_k3d_twomagnon(kb, bonds_z, table; J=Jz)
```
"""
function make_Heisenberg_k3d_twomagnon(kb::MomentumBasis3D_TwoMagnon,
                                        bonds,
                                        table::RepTable3D_Fast;
                                        J=1.0)
    L = kb.L
    dim = length(kb)
    H_k = spzeros(ComplexF64, dim, dim)

    for α in 1:dim
        m1, m2 = kb.reps[α]
        O_a = kb.orbit_sizes[α]

        for (a, b) in bonds

            # ── Diagonal: Sz_a Sz_b ──
            a_flip = (a == m1 || a == m2)
            b_flip = (b == m1 || b == m2)

            if a_flip && b_flip
                H_k[α, α] += J * 0.25
            elseif a_flip || b_flip
                H_k[α, α] += J * (-0.25)
            else
                H_k[α, α] += J * 0.25
            end

            # ── Off-diagonal: (S+S- + S-S+)/2 ──
            if a_flip ⊻ b_flip
                if a_flip
                    # Magnon at a hops to b
                    new_m = b
                    old_m = (a == m1) ? m2 : m1
                else
                    # Magnon at b hops to a
                    new_m = a
                    old_m = (b == m1) ? m2 : m1
                end

                # New pair in canonical order
                new_pair = old_m < new_m ? (old_m, new_m) : (new_m, old_m)

                # Skip if both magnons land on the same site
                if new_pair[1] != new_pair[2]
                    rep, nx, ny, nz = lookup_rep_3d(table, new_pair[1], new_pair[2])
                    β = get(kb.rep_index, rep, nothing)
                    if β !== nothing
                        O_b = kb.orbit_sizes[β]
                        phase = exp(im * (kb.kx * nx + kb.ky * ny + kb.kz * nz))
                        H_k[β, α] += J * 0.5 * phase * sqrt(O_a / O_b)
                    end
                end
            end
        end
    end

    return H_k
end


# ─────────────────────────────────────────────────────────────────
#  Convenience: ground state search across all k-sectors (3D)
# ─────────────────────────────────────────────────────────────────

"""
    find_ground_state_twomagnon_3d(Nx, Ny, Nz, bond_terms; verbose=true)

Sweep over all (kx, ky, kz) sectors, build and diagonalize H in each,
and return the global ground state.

`bond_terms` = [(bonds, J), ...] where bonds = Vector{Tuple{Int,Int}}.

# Example: tetragonal lattice
```julia
bonds_x, bonds_y, bonds_z = cubic_bonds_by_direction(4, 4, 4)
E0, mx, my, mz, kb, vec = find_ground_state_twomagnon_3d(4, 4, 4,
    [(bonds_x, 1.0), (bonds_y, 1.0), (bonds_z, 0.5)])
```

Returns: (energy, mx, my, mz, MomentumBasis3D_TwoMagnon, eigenvector)
"""
function find_ground_state_twomagnon_3d(Nx::Int, Ny::Int, Nz::Int, bond_terms;
                                         verbose=true)
    basis = TwoMagnonBasis3D_Lean(Nx, Ny, Nz)
    table = build_rep_table_fast_3d(basis)

    if verbose
        println("Two-magnon 3D basis: L=$(basis.L), D=$(basis.D) pairs")
    end

    best_E   = Inf
    best_mx  = 0
    best_my  = 0
    best_mz  = 0
    best_kb  = nothing
    best_vec = nothing

    for mz in 0:Nz-1, my in 0:Ny-1, mx in 0:Nx-1
        kb = MomentumBasis3D_TwoMagnon(basis, mx, my, mz)
        dim = length(kb)
        dim == 0 && continue

        H_k = spzeros(ComplexF64, dim, dim)
        for (bonds, J) in bond_terms
            H_k .+= make_Heisenberg_k3d_twomagnon(kb, bonds, table; J=J)
        end

        vals, vecs = eigen(Hermitian(Matrix(H_k)))
        E0 = vals[1]

        if verbose && E0 < best_E
            println("  ★ (mx=$mx, my=$my, mz=$mz): dim=$dim, E₀ = $(round(E0; digits=10))")
        end

        if E0 < best_E
            best_E   = E0
            best_mx  = mx
            best_my  = my
            best_mz  = mz
            best_kb  = kb
            best_vec = vecs[:, 1]
        end
    end

    return best_E, best_mx, best_my, best_mz, best_kb, best_vec
end
