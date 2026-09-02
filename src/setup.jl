# ═══════════════════════════════════════════════════════════════════════════
#  setup.jl — load the whole library in dependency order.
#
#      include(joinpath(@__DIR__, "..", "src", "setup.jl"))
#
#  is all any script in scripts/ needs.
# ═══════════════════════════════════════════════════════════════════════════

const SRC = @__DIR__

include(joinpath(SRC, "model.jl"))          # couplings, dispersions, conventions
include(joinpath(SRC, "bonds.jl"))          # bond lists for ED
include(joinpath(SRC, "ed_gamma.jl"))       # module GammaMagnons: Γ-point ED engine
include(joinpath(SRC, "ed_model.jl"))       # BondTerms, drivers, convention check
include(joinpath(SRC, "ed_spectral.jl"))    # channel weights and spectral functions
include(joinpath(SRC, "ls_two_body.jl"))    # two-body Lippmann–Schwinger
include(joinpath(SRC, "ls_dos.jl"))         # continuum: DOS, Krein shift, A_a(E)
include(joinpath(SRC, "ls_three_body.jl"))  # three-body Faddeev / STM
