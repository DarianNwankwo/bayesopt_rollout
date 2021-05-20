import Base.@kwdef

include("Types.jl")


# Squared Exponential Kernel Start
@kwdef mutable struct SquaredExponential <: Kernel
    variance::Real # provided as ν²
    lengthscale::Real # provided as l
    noise::Real # provided as σ
    
    function SquaredExponential(variance, lengthscale, noise)
        @assert variance >= 0 "variance must be a positive real number"
        
        return new(variance, lengthscale, noise)
    end
end

SquaredExponential() = SquaredExponential(variance=1.0, lengthscale=1.0, noise=0.0)
SE() = SquaredExponential()

function (se::SquaredExponential)(x::Array{Float64, 1}, y::Array{Float64, 1})::Real
    δxx = Int8(x == y)
    r = x - y
    ρ = norm(r, 2)
    return se.variance * exp(-ρ^2 / 2*se.lengthscale^2) + δxx*se.noise^2
end
# Squared Exponential Kernel End