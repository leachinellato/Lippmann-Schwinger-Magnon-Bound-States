# ═══════════════════════════════════════════════════════════════════════════
#  gamma_magnons_3D.jl
#
#  Γ-point (k = 0) exact diagonalization engine for the hard-core two- and
#  three-magnon sectors of an XXZ model on a 3D periodic lattice.
#
#  ─────────────────────────────────────────────────────────────────────────
#  WHY THIS IS FAST
#  ─────────────────────────────────────────────────────────────────────────
#  Lattice translations act *freely* on the L sites.  Therefore, for an
#  n-magnon configuration, each orbit contains exactly n translates that
#  place a magnon on the origin (site 1) — fewer only when the stabilizer
#  is nontrivial.  This gives an O(1) canonical form:
#
#      canonical rep = lex-min over i of  sort( T_{-r(m_i)} {m_1..m_n} )
#
#  Consequences:
#
#    1. No orbit enumeration.  Canonicalization is O(n), not O(L).
#    2. Every representative has the form (1, a) or (1, a, b), so the basis
#       is enumerated by looping over C(L-1, n-1) candidates — the full
#       D = C(L, n) space is NEVER touched or allocated.
#    3. The stabilizer acts freely on the n magnons, so |S| divides n.
#       n = 2 -> |S| ∈ {1,2};  n = 3 -> |S| ∈ {1,3}.  Orbit size O = L/|S|.
#       Two Int8 values instead of an O(D) orbit-size table.
#    4. At Γ every orbit is compatible (the phase condition is 0 ≡ 0),
#       so there is no stabilizer/coset compatibility test at all, and
#       every lookup is guaranteed to hit a valid representative.
#    5. All phases are 1  =>  H is REAL SYMMETRIC.  Float64, not ComplexF64.
#
#  Memory (three-magnon):  the O(D)-sized lookup table is eliminated
#  entirely; the only large array is `index`, of length C(L-1, 2).
#
#      lattice   L      dim(Γ)        index      H (CSC, Int32)
#      6^3       216       7,668       0.1 MB          1.8 MB
#      10^3     1000     166,167       2.0 MB           38 MB
#      20^3     8000  10,662,667     128.0 MB          1.6 GB
#
#  ─────────────────────────────────────────────────────────────────────────
#  CONVENTIONS (stated explicitly — check these against your bond generator)
#  ─────────────────────────────────────────────────────────────────────────
#  Sites are 1-based and indexed as
#
#      i = x + Nx*y + Nx*Ny*z + 1,     x ∈ 0:Nx-1, y ∈ 0:Ny-1, z ∈ 0:Nz-1
#
#  so that site 1 = (0,0,0) is the origin and is the global minimum index.
#  >>> This MUST agree with `translate_site_3d` in two_magnon_basis_3D.jl.
#  >>> Use `gamma_check_site_convention()` to verify (see bottom of file).
#
#  Hamiltonian, on a fully polarized |↑↑…↑⟩ background:
#
#      H = Σ_terms Σ_{(a,b) ∈ bonds}  Jxy (S⁺_a S⁻_b + S⁻_a S⁺_b)/2
#                                   + Jz  S^z_a S^z_b
#
#  The Jz term includes the vacuum offset +Jz/4 per bond (both spins up),
#  matching make_XXZ_k3d_threemagnon.  Subtract `ht.E_vac` for the
#  magnon-only energy.
#
#  Momentum-basis matrix elements use the same normalization as the
#  existing general-k code:
#
#      H[β,α] += Jxy/2 · e^{i k·n} · sqrt(O_α / O_β),   with O = L/|S|
#
#  and at Γ the phase is 1, so sqrt(O_α/O_β) = sqrt(|S_β| / |S_α|).
#  NOTE: individual matrix elements are NOT symmetric; H becomes symmetric
#  only after summing over all bonds, because O_α N_{αβ} = O_β N_{βα}.
#  The matrix-free gather in `apply_xxz_gamma!` relies on this.
#
#  Bond lists: each unordered bond must appear EXACTLY ONCE per term.
#
#  Validated against the orbit isometry V (V† H_full V == H_Γ) — see
#  test_gamma_magnons_3D.jl.
# ═══════════════════════════════════════════════════════════════════════════

module GammaMagnons

using SparseArrays
using LinearAlgebra

export GammaBasis, build_gamma_basis, orbit_size, gamma_rep,
       BondTerm, HopTable, build_hop_table,
       xxz_gamma, apply_xxz_gamma!, GammaOperator,
       gamma_memory_report, gamma_check_site_convention, site_coords


# ───────────────────────────────────────────────────────────────────────────
#  Site utilities
# ───────────────────────────────────────────────────────────────────────────

"""
    site_index(x, y, z, Nx, Ny) -> Int

1-based site index for the 0-based coordinate triple (x,y,z).
"""
@inline site_index(x::Int, y::Int, z::Int, Nx::Int, Ny::Int) =
    x + Nx * y + Nx * Ny * z + 1

"""
    site_coords(i, Nx, Ny) -> (x, y, z)

0-based coordinates of the 1-based site index `i`.
"""
@inline function site_coords(i::Int, Nx::Int, Ny::Int)
    i0 = i - 1
    x  = i0 % Nx
    r  = i0 ÷ Nx
    return x, r % Ny, r ÷ Ny
end

"""
    site_translate(i, dx, dy, dz, Nx, Ny, Nz) -> Int

Translate site `i` by (dx,dy,dz) with periodic boundary conditions.
Provided for cross-checking against existing code; not used in hot loops.
"""
@inline function site_translate(i::Int, dx::Int, dy::Int, dz::Int,
                                Nx::Int, Ny::Int, Nz::Int)
    x, y, z = site_coords(i, Nx, Ny)
    return site_index(mod(x + dx, Nx), mod(y + dy, Ny), mod(z + dz, Nz), Nx, Ny)
end


# ───────────────────────────────────────────────────────────────────────────
#  Tuple sorting (fully unrolled for n = 2, 3)
# ───────────────────────────────────────────────────────────────────────────

@inline function _sorted(t::NTuple{2,Int})
    a, b = t
    return a <= b ? (a, b) : (b, a)
end

@inline function _sorted(t::NTuple{3,Int})
    a, b, c = t
    if a > b
        a, b = b, a
    end
    if b > c
        b, c = c, b
    end
    if a > b
        a, b = b, a
    end
    return (a, b, c)
end


# ───────────────────────────────────────────────────────────────────────────
#  O(1) canonicalization
# ───────────────────────────────────────────────────────────────────────────

"""
    canon(flip, Nx, Ny, Nz) -> (rep, (dx, dy, dz))

Canonical representative of the translation orbit of the magnon
configuration `flip` (an `NTuple{N,Int}` of site indices), together with the
translation that reaches it:

    T_{(dx,dy,dz)} |flip⟩ = |rep⟩

`rep` is sorted and always begins with site 1.  Cost is O(N), independent
of L — no orbit is ever enumerated.

At Γ the translation is unused, but it is returned anyway so the same
canonicalization serves a general-k implementation without modification.
"""
@inline function canon(flip::NTuple{N,Int}, Nx::Int, Ny::Int, Nz::Int) where {N}
    NxNy = Nx * Ny
    best = ntuple(_ -> typemax(Int), Val(N))
    bdx = 0; bdy = 0; bdz = 0

    @inbounds for i in 1:N
        ox, oy, oz = site_coords(flip[i], Nx, Ny)

        cand = ntuple(Val(N)) do j
            x, y, z = site_coords(flip[j], Nx, Ny)
            xx = x - ox; xx < 0 && (xx += Nx)
            yy = y - oy; yy < 0 && (yy += Ny)
            zz = z - oz; zz < 0 && (zz += Nz)
            xx + Nx * yy + NxNy * zz + 1
        end

        s = _sorted(cand)
        if isless(s, best)
            best = s
            bdx = ox == 0 ? 0 : Nx - ox
            bdy = oy == 0 ? 0 : Ny - oy
            bdz = oz == 0 ? 0 : Nz - oz
        end
    end

    return best, (bdx, bdy, bdz)
end

"""
    stab_order(rep, Nx, Ny, Nz) -> Int

Order |S| of the translation stabilizer of the (sorted, canonical)
configuration `rep`.

Because translations act freely on sites, any stabilizer element is
uniquely determined by the magnon it sends the origin to; so |S| is just
the number of magnons `m` for which `T_{r(m)}` preserves the set.  The
action of S on the n magnons is free, hence |S| divides n:

    n = 2  ->  |S| ∈ {1, 2}   (2 requires some Nᵢ even)
    n = 3  ->  |S| ∈ {1, 3}   (3 requires some Nᵢ divisible by 3)

Orbit size is O = L / |S|.
"""
@inline function stab_order(rep::NTuple{N,Int}, Nx::Int, Ny::Int, Nz::Int) where {N}
    NxNy = Nx * Ny
    cnt = 0
    @inbounds for i in 1:N
        ox, oy, oz = site_coords(rep[i], Nx, Ny)
        t = ntuple(Val(N)) do j
            x, y, z = site_coords(rep[j], Nx, Ny)
            xx = x + ox; xx >= Nx && (xx -= Nx)
            yy = y + oy; yy >= Ny && (yy -= Ny)
            zz = z + oz; zz >= Nz && (zz -= Nz)
            xx + Nx * yy + NxNy * zz + 1
        end
        _sorted(t) === rep && (cnt += 1)
    end
    return cnt
end


# ───────────────────────────────────────────────────────────────────────────
#  Direct-address key for a representative
#
#  A representative is (1, a) or (1, a, b).  Writing A = a-1, B = b-1 with
#  1 ≤ A < B ≤ L-1, the combinatorial number system gives a dense key.
# ───────────────────────────────────────────────────────────────────────────

@inline _rep_key(rep::NTuple{2,Int}) = rep[2] - 1

@inline function _rep_key(rep::NTuple{3,Int})
    A = rep[2] - 1
    B = rep[3] - 1
    return ((B - 1) * (B - 2)) ÷ 2 + A
end

@inline _nkeys(L::Int, ::Val{2}) = L - 1
@inline _nkeys(L::Int, ::Val{3}) = ((L - 1) * (L - 2)) ÷ 2

# Candidate representatives: all sets containing the origin.
_candidates(L::Int, ::Val{2}) = ((1, a) for a in 2:L)
_candidates(L::Int, ::Val{3}) = ((1, a, b) for b in 3:L for a in 2:(b - 1))


# ───────────────────────────────────────────────────────────────────────────
#  Γ-sector basis
# ───────────────────────────────────────────────────────────────────────────

"""
    GammaBasis{N}

Complete k = 0 basis for the hard-core N-magnon sector on an Nx×Ny×Nz
periodic lattice.

Fields
  `reps`  : one canonical configuration per translation orbit, sorted,
            each beginning with site 1.
  `stab`  : |S_α| ∈ {1, N}.  Orbit size is L ÷ stab[α].
  `index` : direct-address map, key -> α, of length C(L-1, N-1).
            Every entry is nonzero: at Γ every orbit is a representative.

`reps` stores the leading `1` explicitly for clarity; dropping it would
save a further 1/N of that array if you ever need it.
"""
struct GammaBasis{N}
    Nx    :: Int
    Ny    :: Int
    Nz    :: Int
    L     :: Int
    reps  :: Vector{NTuple{N,Int32}}
    stab  :: Vector{Int8}
    index :: Vector{Int32}
end

Base.length(b::GammaBasis) = length(b.reps)

"""Orbit size O_α = L / |S_α|."""
@inline orbit_size(b::GammaBasis, α::Integer) = b.L ÷ Int(b.stab[α])

"""Representative α as an `NTuple{N,Int}`."""
@inline gamma_rep(b::GammaBasis{N}, α::Integer) where {N} =
    ntuple(j -> Int(@inbounds b.reps[α][j]), Val(N))

"""
    lookup(basis, flip) -> (α, (dx,dy,dz))

Sector index of the orbit containing configuration `flip`, plus the
translation carrying `flip` to the representative.  O(1), no hashing.
"""
@inline function lookup(b::GammaBasis{N}, flip::NTuple{N,Int}) where {N}
    rep, d = canon(flip, b.Nx, b.Ny, b.Nz)
    @inbounds α = Int(b.index[_rep_key(rep)])
    return α, d
end

function Base.show(io::IO, ::MIME"text/plain", b::GammaBasis{N}) where {N}
    printstyled(io, "GammaBasis{$N}"; bold = true, color = :cyan)
    println(io)
    println(io, "  Lattice          $(b.Nx) × $(b.Ny) × $(b.Nz)   (L = $(b.L))")
    println(io, "  Magnons          $N   (k = 0, Γ point)")
    D = binomial(big(b.L), N)
    println(io, "  Full sector      C(L,$N) = $D")
    println(io, "  Γ dimension      $(length(b.reps))   (reduction $(round(Float64(D) / length(b.reps), digits = 2))×)")
    nred = count(s -> Int(s) != 1, b.stab)
    println(io, "  Reduced orbits   $nred  (|S| = $N)")
    println(io, "  Memory           $(_fmt_bytes(_basis_bytes(b)))")
end

_basis_bytes(b::GammaBasis{N}) where {N} =
    length(b.reps) * 4N + length(b.stab) + length(b.index) * 4

function _fmt_bytes(n::Integer)
    n < 1024        && return "$n B"
    n < 1024^2      && return string(round(n / 1024, digits = 1), " KB")
    n < 1024^3      && return string(round(n / 1024^2, digits = 1), " MB")
    return string(round(n / 1024^3, digits = 2), " GB")
end

"""
    build_gamma_basis(Nx, Ny, Nz, n) -> GammaBasis{n}

Enumerate the Γ-sector basis for `n` = 2 or 3 magnons.

Cost is O(C(L-1, n-1)) canonicalizations, each O(1).  The full
D = C(L,n) space is never allocated or visited.
"""
function build_gamma_basis(Nx::Int, Ny::Int, Nz::Int, n::Integer)
    n == 2 && return _build_gamma_basis(Nx, Ny, Nz, Val(2))
    n == 3 && return _build_gamma_basis(Nx, Ny, Nz, Val(3))
    error("build_gamma_basis: only n = 2 and n = 3 are supported (got $n)")
end

function _build_gamma_basis(Nx::Int, Ny::Int, Nz::Int, ::Val{N}) where {N}
    (Nx > 0 && Ny > 0 && Nz > 0) || error("lattice dimensions must be positive")
    L = Nx * Ny * Nz
    L >= N + 1 || error("need L ≥ $(N+1) sites for $N magnons (got L = $L)")

    nk    = _nkeys(L, Val(N))
    index = zeros(Int32, nk)
    reps  = NTuple{N,Int32}[]
    stab  = Int8[]

    # Upper bound: the orbit count is C(L,N)/L rounded up; reserve that.
    sizehint!(reps, cld(nk, N))
    sizehint!(stab, cld(nk, N))

    for cand in _candidates(L, Val(N))
        rep, _ = canon(cand, Nx, Ny, Nz)
        rep === cand || continue
        s = stab_order(cand, Nx, Ny, Nz)
        # |S| acts freely on N magnons, so it must divide N.  N ∈ {2,3} is
        # prime, hence |S| ∈ {1,N}.  A violation means the site convention
        # or the free-action assumption is broken.
        (s == 1 || s == N) ||
            error("stabilizer order $s ∉ {1,$N} at rep $cand — site convention broken?")
        push!(reps, map(Int32, cand))
        push!(stab, Int8(s))
        index[_rep_key(cand)] = Int32(length(reps))
    end

    return GammaBasis{N}(Nx, Ny, Nz, L, reps, stab, index)
end


# ───────────────────────────────────────────────────────────────────────────
#  Bond terms and the weighted adjacency (hop) table
# ───────────────────────────────────────────────────────────────────────────

"""
    BondTerm(bonds, Jxy, Jz)

One coupling shell.  `bonds` lists each unordered bond EXACTLY ONCE as
`(a, b)` with `a != b`.  Combine terms for J1–J2–J3 models.
"""
struct BondTerm
    bonds :: Vector{Tuple{Int,Int}}
    Jxy   :: Float64
    Jz    :: Float64
end

BondTerm(bonds::AbstractVector, Jxy::Real, Jz::Real) =
    BondTerm(collect(Tuple{Int,Int}, bonds), Float64(Jxy), Float64(Jz))

"""
    HopTable

Weighted CSR adjacency over sites: for site `s`, the neighbours are
`nbr[ptr[s] : ptr[s+1]-1]` with couplings `Jxy[·]`, `Jz[·]`.

`E_vac = (1/4) Σ_terms Jz · (#bonds)` is the fully polarized reference
energy, i.e. the Σ_bonds Jz/4 that every diagonal element carries.
"""
struct HopTable
    L      :: Int
    ptr    :: Vector{Int32}
    nbr    :: Vector{Int32}
    Jxy    :: Vector{Float64}
    Jz     :: Vector{Float64}
    E_vac  :: Float64
    maxdeg :: Int
end

Base.show(io::IO, ::MIME"text/plain", ht::HopTable) = print(io,
    "HopTable(L = $(ht.L), entries = $(length(ht.nbr)), maxdeg = $(ht.maxdeg), E_vac = $(ht.E_vac))")

"""
    build_hop_table(L, terms) -> HopTable

Build the weighted adjacency from a vector of `BondTerm`s.  Each bond is
inserted in both directions so a magnon on either endpoint sees it.
"""
function build_hop_table(L::Int, terms::AbstractVector{BondTerm})
    deg = zeros(Int, L)
    E_vac = 0.0
    for t in terms
        E_vac += 0.25 * t.Jz * length(t.bonds)
        for (a, b) in t.bonds
            (1 <= a <= L && 1 <= b <= L) ||
                error("bond ($a,$b) out of range 1:$L")
            a == b && error("self-bond ($a,$a) in bond list")
            deg[a] += 1
            deg[b] += 1
        end
    end

    ptr = Vector{Int32}(undef, L + 1)
    ptr[1] = 1
    @inbounds for s in 1:L
        ptr[s + 1] = ptr[s] + deg[s]
    end

    ne  = Int(ptr[L + 1]) - 1
    nbr = Vector{Int32}(undef, ne)
    jxy = Vector{Float64}(undef, ne)
    jz  = Vector{Float64}(undef, ne)

    fill = copy(ptr)
    @inbounds for t in terms
        for (a, b) in t.bonds
            p = fill[a]; nbr[p] = b; jxy[p] = t.Jxy; jz[p] = t.Jz; fill[a] = p + 1
            q = fill[b]; nbr[q] = a; jxy[q] = t.Jxy; jz[q] = t.Jz; fill[b] = q + 1
        end
    end

    return HopTable(L, ptr, nbr, jxy, jz, E_vac, maximum(deg; init = 0))
end

build_hop_table(L::Int, bonds::AbstractVector{<:Tuple}, Jxy::Real, Jz::Real) =
    build_hop_table(L, [BondTerm(bonds, Jxy, Jz)])


# ───────────────────────────────────────────────────────────────────────────
#  The column kernel — the single hot loop
#
#  For representative α we walk only the ≤ N flipped sites and their
#  neighbours: O(N · z) work, independent of L.  Bonds with exactly one
#  flipped endpoint are enumerated exactly once (from the flipped side),
#  which simultaneously gives
#
#    - the off-diagonal hops, and
#    - the diagonal, since  Σ_bonds Jz SᶻSᶻ = E_vac - (1/2) Σ_{1-flip} Jz.
#
#  Bonds with zero or two flipped endpoints contribute +Jz/4 each, already
#  contained in E_vac.
# ───────────────────────────────────────────────────────────────────────────

@inline function _column!(rows::Vector{Int32}, vals::Vector{Float64},
                          b::GammaBasis{N}, ht::HopTable, α::Int) where {N}
    flip = gamma_rep(b, α)
    sα   = Int(@inbounds b.stab[α])
    diag = ht.E_vac
    n    = 0

    @inbounds for i in 1:N
        s = flip[i]
        for e in Int(ht.ptr[s]):(Int(ht.ptr[s + 1]) - 1)
            v = Int(ht.nbr[e])

            occupied = false
            for j in 1:N
                flip[j] == v && (occupied = true)
            end
            occupied && continue

            diag -= 0.5 * ht.Jz[e]

            jxy = ht.Jxy[e]
            jxy == 0.0 && continue

            # Hard-core is automatic: v is unoccupied by the xor above.
            newflip = ntuple(j -> (j == i ? v : flip[j]), Val(N))
            β, _ = lookup(b, newflip)
            sβ = Int(b.stab[β])

            n += 1
            rows[n] = Int32(β)
            vals[n] = 0.5 * jxy * sqrt(sβ / sα)
        end
    end

    n += 1
    @inbounds rows[n] = Int32(α)
    @inbounds vals[n] = diag
    return n
end

"""Insertion-sort `rows[1:n]` (carrying `vals`) then sum duplicates. Returns
the compacted length.  `n` is at most N·maxdeg+1, so insertion sort wins."""
@inline function _sort_merge!(rows::Vector{Int32}, vals::Vector{Float64}, n::Int)
    @inbounds for i in 2:n
        r = rows[i]; v = vals[i]; j = i - 1
        while j >= 1 && rows[j] > r
            rows[j + 1] = rows[j]; vals[j + 1] = vals[j]; j -= 1
        end
        rows[j + 1] = r; vals[j + 1] = v
    end
    m = 0
    @inbounds for i in 1:n
        if m > 0 && rows[m] == rows[i]
            vals[m] += vals[i]
        else
            m += 1
            rows[m] = rows[i]
            vals[m] = vals[i]
        end
    end
    return m
end

_chunks(dim::Int, nt::Int) =
    [(1 + ((t - 1) * dim) ÷ nt):(((t * dim) ÷ nt)) for t in 1:nt]


# ───────────────────────────────────────────────────────────────────────────
#  Sparse assembly (two-pass, threaded, direct CSC — no COO staging)
# ───────────────────────────────────────────────────────────────────────────

"""
    xxz_gamma(basis, ht) -> SparseMatrixCSC{Float64,Int32}

Real symmetric XXZ Hamiltonian in the Γ sector.

Assembled directly into CSC in two threaded passes (count, then fill), so
peak memory is exactly the final matrix — no triplet staging, no
`sparse()` deduplication pass, and no O(nnz)-per-insertion `H[i,j] +=`.

Column-major generation plus the symmetry of H means this is also the
transpose-consistent layout used by `apply_xxz_gamma!`.
"""
function xxz_gamma(b::GammaBasis{N}, ht::HopTable) where {N}
    b.L == ht.L || error("basis L = $(b.L) but hop table L = $(ht.L)")
    dim = length(b)
    cap = N * ht.maxdeg + 1
    nt  = min(Threads.nthreads(), max(dim, 1))
    rngs = _chunks(dim, nt)

    # ── Pass 1: count entries per column ──
    cnt = Vector{Int32}(undef, dim)
    @sync for r in rngs
        Threads.@spawn begin
            rows = Vector{Int32}(undef, cap)
            vals = Vector{Float64}(undef, cap)
            @inbounds for α in r
                m = _sort_merge!(rows, vals, _column!(rows, vals, b, ht, α))
                cnt[α] = Int32(m)
            end
        end
    end

    colptr = Vector{Int32}(undef, dim + 1)
    colptr[1] = 1
    total = 0
    @inbounds for α in 1:dim
        total += Int(cnt[α])
        total + 1 <= typemax(Int32) ||
            error("nnz ≥ 2^31; rebuild with Ti = Int64 or use the matrix-free GammaOperator")
        colptr[α + 1] = Int32(total + 1)
    end

    rowval = Vector{Int32}(undef, total)
    nzval  = Vector{Float64}(undef, total)

    # ── Pass 2: fill ──
    @sync for r in rngs
        Threads.@spawn begin
            rows = Vector{Int32}(undef, cap)
            vals = Vector{Float64}(undef, cap)
            @inbounds for α in r
                m = _sort_merge!(rows, vals, _column!(rows, vals, b, ht, α))
                p = Int(colptr[α]) - 1
                for t in 1:m
                    rowval[p + t] = rows[t]
                    nzval[p + t]  = vals[t]
                end
            end
        end
    end

    return SparseMatrixCSC(dim, dim, colptr, rowval, nzval)
end

xxz_gamma(Nx::Int, Ny::Int, Nz::Int, n::Integer, terms::AbstractVector{BondTerm}) =
    let b = build_gamma_basis(Nx, Ny, Nz, n)
        (b, xxz_gamma(b, build_hop_table(b.L, terms)))
    end


# ───────────────────────────────────────────────────────────────────────────
#  Matrix-free application
#
#  Generating column α yields pairs (β, H[β,α]).  Since H is real symmetric
#  at Γ, H[β,α] = H[α,β], so accumulating  y[α] += H[β,α]·x[β]  computes
#  (Hx)[α] as a pure GATHER: each task owns its output element, so the loop
#  is race-free with no atomics and no graph colouring.
#
#  (Individual matrix elements carry sqrt(O_α/O_β) and look asymmetric —
#  symmetry holds only after the bond sum, which is exactly what a full
#  column is.  See the header note.)
# ───────────────────────────────────────────────────────────────────────────

"""
    apply_xxz_gamma!(y, x, basis, ht) -> y

Compute `y = H x` without storing H.  Same kernel as `xxz_gamma`, no
sorting and no allocation in the inner loop.
"""
function apply_xxz_gamma!(y::AbstractVector{Float64}, x::AbstractVector{Float64},
                          b::GammaBasis{N}, ht::HopTable) where {N}
    dim = length(b)
    length(y) == dim && length(x) == dim ||
        error("dimension mismatch: dim = $dim, |x| = $(length(x)), |y| = $(length(y))")
    cap = N * ht.maxdeg + 1
    nt  = min(Threads.nthreads(), max(dim, 1))

    @sync for r in _chunks(dim, nt)
        Threads.@spawn begin
            rows = Vector{Int32}(undef, cap)
            vals = Vector{Float64}(undef, cap)
            @inbounds for α in r
                m = _column!(rows, vals, b, ht, α)
                acc = 0.0
                for t in 1:m
                    acc += vals[t] * x[Int(rows[t])]
                end
                y[α] = acc
            end
        end
    end
    return y
end

"""
    GammaOperator(basis, ht)

Matrix-free real symmetric linear operator.  Callable and `mul!`-able, so
it plugs straight into KrylovKit:

    H = GammaOperator(basis, ht)
    vals, vecs, info = eigsolve(H, rand(size(H,1)), 4, :SR; ishermitian = true)
"""
struct GammaOperator{N}
    basis :: GammaBasis{N}
    ht    :: HopTable
end

Base.size(A::GammaOperator) = (length(A.basis), length(A.basis))
Base.size(A::GammaOperator, d::Integer) = d <= 2 ? length(A.basis) : 1
Base.eltype(::GammaOperator) = Float64
LinearAlgebra.issymmetric(::GammaOperator) = true
LinearAlgebra.ishermitian(::GammaOperator) = true

LinearAlgebra.mul!(y::AbstractVector{Float64}, A::GammaOperator,
                   x::AbstractVector{Float64}) =
    apply_xxz_gamma!(y, x, A.basis, A.ht)

Base.:*(A::GammaOperator, x::AbstractVector{Float64}) =
    apply_xxz_gamma!(similar(x), x, A.basis, A.ht)

(A::GammaOperator)(x::AbstractVector{Float64}) = A * x


# ───────────────────────────────────────────────────────────────────────────
#  Diagnostics
# ───────────────────────────────────────────────────────────────────────────

"""
    gamma_memory_report(Nx, Ny, Nz, n; z = 6)

A-priori memory estimate without building anything.  `z` is the
coordination number (total bonds per site over all terms).
"""
function gamma_memory_report(Nx::Int, Ny::Int, Nz::Int, n::Integer; z::Int = 6)
    L   = Nx * Ny * Nz
    D   = binomial(big(L), n)
    nk  = n == 2 ? L - 1 : ((L - 1) * (L - 2)) ÷ 2
    dim = Float64(D) / L                     # exact when no reduced orbits
    nnz = dim * (n * z + 1)
    println("Γ sector, $Nx×$Ny×$Nz, n = $n magnons")
    println("  L                  $L")
    println("  C(L,$n)             $D")
    println("  dim(Γ) ≈           $(round(Int, dim))")
    println("  index array        $(_fmt_bytes(4 * nk))")
    println("  reps + stab        $(_fmt_bytes(round(Int, dim) * (4n + 1)))")
    println("  H (CSC, Int32)     $(_fmt_bytes(round(Int, nnz * 12)))")
    println("  matrix-free vec    $(_fmt_bytes(round(Int, dim * 8))) per Krylov vector")
    return nothing
end

"""
    gamma_check_site_convention(Nx, Ny, Nz, f) -> Bool

Verify that an external translation function `f(i, dx, dy, dz, Nx, Ny, Nz)`
agrees with this file's site indexing on every site and every shift.  Run
this once against `translate_site_3d` from two_magnon_basis_3D.jl before
mixing bond lists between the two code paths.
"""
function gamma_check_site_convention(Nx::Int, Ny::Int, Nz::Int, f)
    L = Nx * Ny * Nz
    for i in 1:L, dz in 0:Nz-1, dy in 0:Ny-1, dx in 0:Nx-1
        f(i, dx, dy, dz, Nx, Ny, Nz) == site_translate(i, dx, dy, dz, Nx, Ny, Nz) ||
            return false
    end
    return true
end

end # module
