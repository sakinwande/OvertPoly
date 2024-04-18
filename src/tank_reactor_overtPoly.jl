include("overtPoly_helpers.jl")
include("nn_mip_encoding.jl")
include("overtPoly_to_mip.jl")
include("overt_to_pwa.jl")
include("problems.jl")
include("reachability.jl")
using LazySets

ρ = 1000 #[g/L]
Cₚ = 0.239 #[J/g K]
ΔH = -5e-4 #[J/mol]
ER = 8750 #[K]
k₀ = 7.2e10 #[1/min]
UA = 5e4 #[J/min K]
q = 100 #[l/min]
Tf = 350 #[K]
Caf = 1 #[mol/L]
V = 100 #[l]
τ = 0.015 #time step size

Ca₀ = 0.5
T₀ = 350
Tc₀ = 300

u = :(-3*x1 - 6.9*x2 - $Tc₀)

#States are x2 = T and x1 = Ca. Need to offset by Ca₀ and T₀
#Offset control by Tc₀
#x = [x1, x2] - [x1_0, x2_0]


x1Func = :((1-$((q*τ)/(2*V)) - $(k₀ * τ) * exp(-$(ER)/x2)* x1 + $(q/V * Caf * τ))/(1 + $(q*τ)/$(2*V)+$τ*w1))
x2Func = :(((x2*$(1 - τ/2 - (τ*UA)/(2*V*ρ*Cₚ)) + $τ*($Tf*$(q/V) + ($UA*u)/$(ρ*V*Cₚ)))/(1 + $((τ/2)*(q/V)) + $(τ*UA)/$(2*V*ρ*Cₚ)))- (x1*$((ΔH * k₀ * τ)/(ρ*Cₚ)) * exp(-$(ER)/x2))/(1 + $((q*τ)/(2*V))+$(τ*UA/(2*V*ρ*Cₚ))) + $τ*w2)

x1Range = [0.34, 0.36]
x2Range = [303, 307]
w1Range = [-0.1,0.1]
w2Range = [-2,2]



parse_and_reduce(x1Func)