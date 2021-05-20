import Base.@kwdef

include("Types.jl")
include("Utils.jl")


@kwdef mutable struct GaussianProcess
    meanFunction
    covarianceFunction::Kernel
    choleskyFactor::DataOrNothing{AbstractMatrix{Float64}}
    choleskySolve::DataOrNothing{AbstractVector{Float64}}
    varianceFunction
    
    function GaussianProcess(mean, kernel)
        @assert isCallable(mean) == true
        @assert isCallable(kernel) == true

        return new(
            mean,
            kernel,
            nothing,
            nothing,
            nothing
        )
    end
end


GP(mean, kernel) = GaussianProcess(mean, kernel)


function conditional(gp::GaussianProcess, X::Array{Float64, 2}, y::Array{Float64, 1}, σnoise::Real)
    K = gp.choleskyFactor
    
    # Update GP
    if any(isnothing.( [gp.choleskyFactor, gp.choleskySolve, gp.varianceFunction] ))
        K = covarianceMatrix(gp.covarianceFunction, X, X; σnoise=σnoise)
        K = Matrix(Hermitian(K))
        gp.choleskyFactor = L = cholesky(K).U'
        gp.choleskySolve = L'\(L\y)
    end
    
    # Construct mean function
    function μ(xtest)
        xtest = toMatrix(xtest)
        kxs = covarianceMatrix(gp.covarianceFunction, X, xtest; σnoise=σnoise)
        return dot(kxs, gp.choleskySolve)
    end
    
    gp.meanFunction = μ
    
    # Construct variance function
    function σsquared(xtest)
        xtest = toMatrix(xtest)
        kxs = covarianceMatrix(gp.covarianceFunction, X, xtest; σnoise=σnoise)
        v = gp.choleskyFactor \ kxs
        kss = covarianceMatrix(gp.covarianceFunction, xtest, xtest; σnoise=σnoise)[1][1]
        return kss - dot(v, v)
    end
    
    gp.varianceFunction = σsquared
    
    return gp
end