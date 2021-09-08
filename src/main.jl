using Distributions
using LinearAlgebra
using Plots
using Statistics

include("CovarianceFunctions.jl")
include("GaussianProcess.jl")
include("BayesianOptimization.jl")


function main()
    ## STEP 1: Gather Initial Samples
    domain = -10:.1:10

    # Suppose we have some process and we have n=4 known function evaluations
    n = 6
    d = 1
    σnoise = 0.05
    
    # D x n, D: dimensions, n: number of observations
    X = round.(
        rand(Uniform(-10.0, 10.0), d, n), 
        digits=4
    ) 
    y = vec(blackBoxProcess.(X; σnoise=σnoise))
    
    # Let's sample from our predictive distribution conditioned on our observations
    # across the interval [-1, 2] to observe some reasonable functions
    Xtrue = collect(-10.0:.1:10.0)
    Xtrue = reshape(Xtrue, (1, length(Xtrue)))
    ytrue = blackBoxProcess.(Xtrue; σnoise=0.0)

    ## STEP 2: Initialize Our Model
    zero_mean(x) = 0.
    sekernel = SquaredExponential()
    gp = GaussianProcess(zero_mean, sekernel)
    bopt = BayesianOptimization(gp, X, y)
    conditional(bopt.gp, X, y, σnoise)

    
    ## STEP 3: Get The Acquisition Function α(x)
    

    # scatter(X', y)
    # plot!(
    #     domain,
    #     cgp.meanFunction.(domain),
    #     ribbon=2*sqrt.(cgp.varianceFunction.(domain)),
    #     label="Predictions"
    # ) # mean function from GP
    # plot!(domain, blackBoxProcess.(domain; σnoise=0.0), labels="Objective") # objective function

    plot(domain, ei.(domain, Ref(bopt)), label="Acquisition Function") # acquistion function
    scatter!(X', y, label="Observations") # observations
    plot!(
        domain,
        bopt.gp.meanFunction.(domain),
        ribbon=2*sqrt.(bopt.gp.varianceFunction.(domain)),
        label="Mean of GP"
    ) # mean function from GP
    plot!(domain, blackBoxProcess.(domain; σnoise=0.0), label="Objective") # true objective
    savefig("test.png")
end


main()