include("Types.jl")

function blackBoxProcess(xloc::Float64; σnoise::Float64=.1)
    # Domain of interest (for f) is restricted to [-1, 2]
    scale_factor = .1
    f(x; σ=σnoise) = scale_factor * (-sin(3x) - x^2 + .7x) + σ*randn()
    return f(xloc)
end

# Matrix will be xlength x ylength: every row corresponds to an element from x
# More generally, the covariance between two entities can be vector valued
function covarianceMatrix(kernel::Kernel, Xa::Array{Float64, 2}, Xb::Array{Float64, 2}; σnoise::Real=0.0)
    @assert size(Xa)[1] == size(Xb)[1]
    @assert isempty(methods(kernel)) == false
    
    # Kernel should be a function of two arguments
    xalength = size(Xa)[2] # rows here correspond to dimensionality
    xblength = size(Xb)[2]
    covMatrix = zeros(xalength, xblength)

    for xacol in 1:xalength
        for xbcol in 1:xblength
            δkronecker = xacol == xbcol ? σnoise^2 * 1 : 0
            covMatrix[xacol, xbcol] = kernel(Xa[:, xacol], Xb[:, xbcol]) + δkronecker
        end
    end

    return covMatrix
end

function toMatrix(x::Float64)
    return reshape([x], (1, 1))
end

isCallable(f) = !isempty(methods(f))

const DataOrNothing{Type} = Union{Type, Nothing}