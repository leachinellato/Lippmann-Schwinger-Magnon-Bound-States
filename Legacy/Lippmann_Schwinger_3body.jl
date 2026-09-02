using LinearAlgebra
using SparseArrays

# ═══════════════════════════════════════════════════════════════════════
#  Three-body sector: Faddeev / Skornyakov–Ter-Martirosyan form of the
#  Lippmann–Schwinger equation at total momentum K = 0.
#
#  After the Bose projection the whole kernel collapses onto one object,
#
#      α_a(p,k) = ( 1 , √2 cos[(k + p/2)·r_a] ),        a = onsite, r
#
#  because  Q(p)ᵀg(k) = Q(p)ᵀg(−p−k) = α(p,k)  and
#           Q(k)ᵀg(−p−k) = Q(k)ᵀg(p) = α(k,p).  Hence
#
#      [B₃(E;p)]_ab = δ_ab/v_a − (1/Ns) Σ_k α_a(p,k) α_b(p,k) / D_E(p,k)
#      [C₃(E;p,k)]_ab = 2 α_a(p,k) α_b(k,p) / D_E(p,k)          (rank one)
#      [A₃(E)]_(pa),(kb) = δ_pk [B₃(E;p)]_ab − (1/Ns)[C₃(E;p,k)]_ab
#
#  with D_E(p,k) = E − ε(p) − ε(k) − ε(−p−k).  A₃ is real symmetric below
#  threshold.  Hard core = 1/v_onsite = 0: the channel is kept, not deleted.
# ═══════════════════════════════════════════════════════════════════════

const RS4 = ((1,0,0), (2,0,0), (0,1,0), (0,0,1))   # channels (x, 2x, y, z)

struct Mesh3
    L    :: NTuple{3,Int}
    Ns   :: Int
    nidx :: Matrix{Int}              # nidx[p,d] ∈ 0:L[d]-1
    e    :: Vector{Float64}          # single-particle ε on the mesh
    free :: Matrix{Float64}          # ε(p)+ε(k)+ε(−p−k), symmetric
    dir  :: Vector{Int}              # channel → direction (0 = onsite)
    tab  :: Vector{Matrix{Float64}}  # channel → α table, tab[a][n_p+1, n_k+1]
    vinv :: Vector{Float64}          # (0, 1/V_a …)
    nc   :: Int
    Eth  :: Float64                  # breakup threshold on this mesh
end

@inline flat3(L, ax, ay, az) =
    1 + mod(ax, L[1]) + L[1]*(mod(ay, L[2]) + L[2]*mod(az, L[3]))

"""
    mesh3(L, ε1, V; rs = RS4)

Three-body mesh cache.  `L = (Lx,Ly,Lz)` need not be cubic — an anisotropic
mesh is the cheap way to handle the mass anisotropy.  `ε1` is the
**single-particle** dispersion in fractional coordinates (not the pair energy
`ϵ` of the two-body code).  `V[j]` is the coupling on the bond pair `±rs[j]`;
channels with `V[j] == 0` are dropped.  A constant in `ε1` shifts `E₃` and
`E_th` alike and cancels in `Δ₃`.
"""
function mesh3(L::NTuple{3,Int}, ε1, V; rs = RS4)
    Lx, Ly, Lz = L
    Ns = Lx*Ly*Lz
    nidx = Matrix{Int}(undef, Ns, 3)
    e    = Vector{Float64}(undef, Ns)
    i = 0
    for iz in 0:Lz-1, iy in 0:Ly-1, ix in 0:Lx-1
        i += 1
        nidx[i,1], nidx[i,2], nidx[i,3] = ix, iy, iz
        e[i] = ε1(ix/Lx, iy/Ly, iz/Lz)
    end

    free = Matrix{Float64}(undef, Ns, Ns)
    @inbounds for k in 1:Ns, p in 1:Ns
        m = flat3(L, -nidx[p,1]-nidx[k,1], -nidx[p,2]-nidx[k,2], -nidx[p,3]-nidx[k,3])
        free[p,k] = e[p] + e[k] + e[m]
    end

    keep = [j for j in eachindex(rs) if V[j] != 0]
    nc   = 1 + length(keep)
    dir  = zeros(Int, nc)
    tab  = Vector{Matrix{Float64}}(undef, nc)
    vinv = zeros(nc)
    tab[1] = ones(1, 1)                       # onsite: α ≡ 1, 1/U = 0
    for (a, j) in enumerate(keep)
        r = rs[j]
        count(!iszero, r) == 1 || error("channel $r is not axis aligned")
        d  = findfirst(!iszero, r)
        rd = r[d]; Ld = L[d]
        2*abs(rd) < Ld || error("separation $r aliases on a mesh of length $Ld")
        T = Matrix{Float64}(undef, Ld, Ld)
        for nk in 0:Ld-1, np in 0:Ld-1
            T[np+1, nk+1] = √2*cos(π*rd*(2nk + np)/Ld)      # √2 cos[(k+p/2)·r]
        end
        dir[a+1] = d; tab[a+1] = T; vinv[a+1] = 1/V[j]
    end
    return Mesh3(L, Ns, nidx, e, free, dir, tab, vinv, nc, minimum(free))
end

"""Inertia of A₃ at E → −∞: one negative eigenvalue per attractive channel per p."""
ν_inf(ms::Mesh3) = ms.Ns*count(<(0), @view ms.vinv[2:end])

# ───────────────────────────────────────────────────────────────────────
#  Kernels
# ───────────────────────────────────────────────────────────────────────

"""Two-body block `B₃(E;p)` — the two-body matrix at pair momentum −p and pair
energy `E − ε(p)`, in the hard-core projected basis."""
function B3(E::Float64, ms::Mesh3, p::Int)
    nc, Ns = ms.nc, ms.Ns
    B = zeros(nc, nc); α = ones(nc)
    w = 1/Ns
    @inbounds for k in 1:Ns
        g = w/(E - ms.free[k,p])
        for a in 2:nc
            d = ms.dir[a]
            α[a] = ms.tab[a][ms.nidx[p,d]+1, ms.nidx[k,d]+1]
        end
        for b in 1:nc, a in 1:nc
            B[a,b] -= g*α[a]*α[b]
        end
    end
    @inbounds for a in 1:nc
        B[a,a] += ms.vinv[a]
    end
    return B
end

"""Fill the preallocated symmetric matrix `A` with `A₃(E)` (dimension `nc·Ns`)."""
function A3!(A::Matrix{Float64}, E::Float64, ms::Mesh3)
    Ns, nc = ms.Ns, ms.nc
    fill!(A, 0.0)
    Bl = zeros(nc, nc, Ns)
    αp = ones(nc); αk = ones(nc)
    w  = 1/Ns
    nidx, dir, tab, free = ms.nidx, ms.dir, ms.tab, ms.free
    @inbounds for k in 1:Ns
        ck = (k-1)*nc
        for p in 1:Ns
            g = 1.0/(E - free[p,k])
            for a in 2:nc
                d = dir[a]; T = tab[a]
                ip = nidx[p,d] + 1; ik = nidx[k,d] + 1
                αp[a] = T[ip,ik]                    # α(p,k)
                αk[a] = T[ik,ip]                    # α(k,p)
            end
            rp = (p-1)*nc
            gw = 2w*g
            for b in 1:nc
                cb = ck + b; s = gw*αk[b]
                for a in 1:nc
                    A[rp+a, cb] -= s*αp[a]
                end
            end
            wg = w*g
            for b in 1:nc, a in 1:nc
                Bl[a,b,p] -= wg*αp[a]*αp[b]
            end
        end
    end
    @inbounds for p in 1:Ns
        rp = (p-1)*nc
        for a in 1:nc
            Bl[a,a,p] += ms.vinv[a]
        end
        for b in 1:nc, a in 1:nc
            A[rp+a, rp+b] += Bl[a,b,p]
        end
    end
    return A
end

A3(E::Float64, ms::Mesh3) = A3!(zeros(ms.nc*ms.Ns, ms.nc*ms.Ns), E, ms)

# ───────────────────────────────────────────────────────────────────────
#  Inertia counting.
#
#  ν(E) = #negative eigenvalues of A₃(E);  n_bound(E) = ν(−∞) − ν(E).
#  Counting rather than det (which overflows at dimension 10⁴ and is blind to
#  simultaneous crossings) and rather than eigmin (which tracks a spectator
#  channel, exactly as in the two-body problem).
# ───────────────────────────────────────────────────────────────────────

"""Negative inertia of a symmetric matrix.  Destroys `A`."""
function inertia_neg!(A::Matrix{Float64}; exact::Bool = false)
    exact && return count(<(0), eigvals(Symmetric(A)))
    F = bunchkaufman!(Symmetric(A, :U); check = false)
    D = F.D; d = D.d; du = D.du
    m = length(d); i = 1; neg = 0
    @inbounds while i ≤ m
        if i < m && du[i] != 0                      # 2×2 block
            a, b, c = d[i], du[i], d[i+1]
            det2 = a*c - b*b
            if det2 < 0
                neg += 1
            elseif a < 0
                neg += 2
            end
            i += 2
        else
            d[i] < 0 && (neg += 1)
            i += 1
        end
    end
    return neg
end

νcount(E, ms::Mesh3, A = zeros(ms.nc*ms.Ns, ms.nc*ms.Ns); exact = false) =
    inertia_neg!(A3!(A, E, ms); exact = exact)

"""Number of three-body bound states below `E`."""
n_bound(E, ms::Mesh3; kw...) = ν_inf(ms) - νcount(E, ms; kw...)

"""
    E3_bound(ms; bracket = nothing, tol = 1e-10)

Lowest three-boson energy at `K = 0`, returned as `(E₃, Δ₃ = E_th − E₃)`, or
`nothing` if unbound.  Bisection is on the count `ν(E)`, so the root found is
the deepest one.  Pass `bracket = (Elo, Ehi)` from a coarser mesh to skip the
search phase: every evaluation is an O((nc·Ns)³) factorization.
"""
function E3_bound(ms::Mesh3; bracket = nothing, tol = 1e-10, dmax = 60,
                  exact = false, verbose = false)
    ν∞  = ν_inf(ms)
    Eth = ms.Eth
    s   = max(1.0, abs(Eth))
    A   = zeros(ms.nc*ms.Ns, ms.nc*ms.Ns)
    f(E) = νcount(E, ms, A; exact = exact) == ν∞     # true ⇔ E below every state

    if bracket === nothing
        hi = Eth - 1e-12*s
        f(hi) && return nothing                       # no bound state at all
        lo = NaN; δ = 1e-3*s; ok = false
        for _ in 1:dmax
            if f(Eth - δ)
                lo = Eth - δ; ok = true; break
            end
            δ *= 2
        end
        ok || error("could not bracket from below; raise dmax")
    else
        lo, hi = bracket
        f(lo)  || error("lower bracket is not below the lowest state")
        !f(hi) || error("upper bracket is below the lowest state")
    end

    while hi - lo > tol*s
        mid = 0.5*(lo + hi)
        if f(mid)
            lo = mid
        else
            hi = mid
        end
        verbose && println("    [", lo, ", ", hi, "]")
    end
    E3 = 0.5*(lo + hi)
    return E3, Eth - E3
end

"""
    E_atomdimer(ms; pset = 1:ms.Ns)

Atom–dimer threshold `min_p [ε(p) + E₂(−p)]`, from `det B₃(E;p) = 0`.  Returns
`nothing` if no dimer is bound at any spectator momentum in `pset` (λ < λc), in
which case the relevant threshold is the breakup one, `ms.Eth`.  Cost is
O(|pset| · Ns) per bisection step; subsample `pset` on large meshes.
"""
function E_atomdimer(ms::Mesh3; pset = 1:ms.Ns, tol = 1e-12, dmax = 50)
    best = nothing
    for p in pset
        top = minimum(@view ms.free[:,p])
        s   = max(1.0, abs(top))
        g(E) = det(B3(E, ms, p))
        hi = top - 1e-12*s; fhi = g(hi)
        lo = NaN; ok = false; δ = 1e-4*s
        for _ in 1:dmax
            if g(top - δ)*fhi < 0
                lo = top - δ; ok = true; break
            end
            δ *= 2
        end
        ok || continue
        flo = g(lo)
        while hi - lo > tol*s
            mid = 0.5*(lo + hi); fm = g(mid)
            if flo*fm ≤ 0
                hi, fhi = mid, fm
            else
                lo, flo = mid, fm
            end
        end
        E = 0.5*(lo + hi)
        (best === nothing || E < best) && (best = E)
    end
    return best
end

# ───────────────────────────────────────────────────────────────────────
#  Exact diagonalization check (small meshes, finite-range hopping only)
# ───────────────────────────────────────────────────────────────────────

"""Single-particle dispersion from a real-space hopping table `τ :: Dict(δ => t)`,
i.e. `H = Σ_i Σ_δ t (b†_{i+δ} b_i + h.c.)`."""
disp_from_hoppings(τ) =
    (h, k, l) -> sum(2t*cos(2π*(d[1]*h + d[2]*k + d[3]*l)) for (d, t) in τ)

"""
    ed3_K0(L, τ, V; rs = RS4, nev = 4)

Three hard-core bosons on the `L` lattice, exactly diagonalized in the `K = 0`
sector.  Small meshes only: the basis is `binomial(Ns,3)`.
"""
function ed3_K0(L::NTuple{3,Int}, τ, V; rs = RS4, nev = 4)
    Ns = L[1]*L[2]*L[3]
    nx = [mod(i-1, L[1]) for i in 1:Ns]
    ny = [mod(div(i-1, L[1]), L[2]) for i in 1:Ns]
    nz = [div(i-1, L[1]*L[2]) for i in 1:Ns]
    fl(a, b, c) = flat3(L, a, b, c)

    Vsep = zeros(Ns)
    for (j, r) in enumerate(rs)
        V[j] == 0 && continue
        jp = fl(r[1], r[2], r[3]); jm = fl(-r[1], -r[2], -r[3])
        jp == jm && error("separation $r aliases on lattice $L")
        Vsep[jp] += V[j]; Vsep[jm] += V[j]
    end
    hop = Dict{Int,Float64}()
    for (d, t) in τ, sgn in (1, -1)
        j = fl(sgn*d[1], sgn*d[2], sgn*d[3])
        hop[j] = get(hop, j, 0.0) + t
    end

    states = NTuple{3,Int}[]
    for a in 1:Ns-2, b in a+1:Ns-1, c in b+1:Ns
        push!(states, (a, b, c))
    end
    pos = Dict(s => n for (n, s) in enumerate(states))
    Dm  = length(states)

    Ii = Int[]; Jj = Int[]; Ww = Float64[]
    for (n, s) in enumerate(states)
        dg = 0.0
        for i in 1:2, j in i+1:3
            dg += Vsep[fl(nx[s[i]]-nx[s[j]], ny[s[i]]-ny[s[j]], nz[s[i]]-nz[s[j]])]
        end
        push!(Ii, n); push!(Jj, n); push!(Ww, dg)
        for i in 1:3
            s0 = s[i]; o1 = s[mod1(i+1,3)]; o2 = s[mod1(i+2,3)]
            for (j, t) in hop
                s1 = fl(nx[s0]+nx[j], ny[s0]+ny[j], nz[s0]+nz[j])
                (s1 == o1 || s1 == o2) && continue
                st1 = Tuple(sort([o1, o2, s1]))
                push!(Ii, pos[st1]); push!(Jj, n); push!(Ww, t)
            end
        end
    end
    H = sparse(Ii, Jj, Ww, Dm, Dm)

    orb = Dict{NTuple{3,Int},Vector{Int}}()
    for (n, s) in enumerate(states)
        rep = s
        for R in 1:Ns
            img = Tuple(sort([fl(nx[x]+nx[R], ny[x]+ny[R], nz[x]+nz[R]) for x in s]))
            img < rep && (rep = img)
        end
        push!(get!(orb, rep, Int[]), n)
    end
    I2 = Int[]; J2 = Int[]; W2 = Float64[]
    for (c, pr) in enumerate(sort(collect(orb), by = first))
        mem = pr[2]; w = 1/√length(mem)
        for n in mem
            push!(I2, n); push!(J2, c); push!(W2, w)
        end
    end
    P  = sparse(I2, J2, W2, Dm, length(orb))
    Hk = Matrix(P' * (H * P)); Hk = 0.5*(Hk + Hk')
    return sort(eigvals(Symmetric(Hk)))[1:min(nev, size(Hk, 1))]
end
 
"""Check the three-body construction against ED on two small lattices."""
function selftest3()
    for (L, τ, V) in (
        ((7,1,1), Dict((1,0,0) => -1.0, (2,0,0) => 0.35),
                  [-2.3, 0.7, 0.0, 0.0]),
        ((5,3,3), Dict((1,0,0) => -1.0, (2,0,0) => 0.35,
                       (0,1,0) => -0.22, (0,0,1) => -0.13),
                  [-2.6, 0.7, 0.31, -0.18]))
        ms = mesh3(L, disp_from_hoppings(τ), V)
        r  = E3_bound(ms; tol = 1e-14)
        ed = ed3_K0(L, τ, V; nev = 1)[1]
        println("L = $L   LS E₃ = ", r[1], "   ED = ", ed,
                "   |diff| = ", abs(r[1] - ed))
    end
end

# ═══════════════════════════════════════════════════════════════════════
#  Physical run
# ═══════════════════════════════════════════════════════════════════════

# Single-particle dispersion.  ϵ_1 is the tight-binding fit used in the
# two-body run (there ϵ = ϵ_1(q) + ϵ_1(−q)); shift it so its absolute value
# matches spin-wave theory at the band minimum.  The shift cancels in Δ₃.
const ε0_shift = ω(0.0, 0.0, 0.0; J010_yz, J020, J100, J001, J131, Δ) -
                 ϵ_1(0.0, 0.0, 0.0)
ε1(h, k, l) = ϵ_1(h, k, l) #+ ε0_shift

Vλ(λ; c = C) = [a == c ? λ*V̄[a] : V̄[a] for a in 1:4]

function run3(λ, L::NTuple{3,Int}; bracket = nothing, tol = 1e-10,
              atomdimer = false, pstride = 1)
    ms  = mesh3(L, ε1, Vλ(λ))
    dim = ms.nc*ms.Ns
    println("L = $L   Ns = $(ms.Ns)   dim A₃ = $dim   (",
            round(8*dim^2/2^30, digits = 2), " GB dense)")
    println("  ε_min       = ", minimum(ms.e))
    println("  E_th  = 3ε   = ", ms.Eth)
    if atomdimer
        ead = E_atomdimer(ms; pset = 1:pstride:ms.Ns)
        println("  E_atomdimer = ", ead === nothing ? "dimer unbound" : ead)
    end
    r = E3_bound(ms; bracket = bracket, tol = tol)
    if r === nothing
        println("  no three-body bound state below E_th")
    else
        println("  E₃ = ", r[1], "   Δ₃ = ", r[2],
                "   n_bound = ", n_bound(ms.Eth - 1e-12, ms))
    end
    return ms, r
end

λ = 1.19

out = run3(λ, (12,12,12);  tol = 1e-10,
              atomdimer = false, pstride = 1)

out[2]