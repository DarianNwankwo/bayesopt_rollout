import Base.@kwdef

include("Types.jl")
include("GaussianProcess.jl")


@kwdef mutable struct BayesianOptimization
    gp::GaussianProcess
    X::Array{Float64, 2}
    y::Array{Float64, 1}
    ybest::Float64
    xi::Float64 # exploration parameter
    
    function BayesianOptimization(gp::GaussianProcess, X::Array{Float64, 2}, y::Array{Float64, 1})
        @assert size(X)[2] == length(y) "each variate must have a corresponding covariate:" # may cause issues when generalizing to vector valued functions
        
        return new(gp, X, y, maximum(y)) 
    end
end


function z(x, bopt::BayesianOptimization)
    σ̂(xa) = sqrt(bopt.gp.varianceFunction(xa))
    μ̂(xa) = bopt.gp.meanFunction(xa)
    f⁺ = bopt.ybest
    ξ = .1
    return σ̂(x) > 0 ? (μ̂(x) - f⁺ - ξ) / σ̂(x) : 0
end


function ei(x, bopt::BayesianOptimization)
    z_eval = z(x, bopt)
    normal = Normal()
    normal_cdf_at_z = cdf(normal, z_eval)
    normal_pdf_at_z = pdf(normal, z_eval)
    σ̂(xa) = sqrt(bopt.gp.varianceFunction(xa))
    μ̂(xa) = bopt.gp.meanFunction(xa)
    f⁺ = bopt.ybest
    ξ = .1
    
    return σ̂(x) > 0 ? (μ̂(x) - f⁺ - ξ) * normal_cdf_at_z + σ̂(x)*normal_pdf_at_z : 0
end