module MyFirstPackage
# import packages
using LinearAlgebra

# export interfaces
export Lorenz, integrate_step
export Point, Point2D, Point3D
export RungeKutta, Euclidean
export D2Q9, LatticeBoltzmann, step!, equilibrium_density, momentum, curl, example_d2q9, density
export D3Q19, example_d3q19, _curl, _LatticeBoltzmann, _step!

include("lorenz.jl")
include("fluid.jl")
include("fluid3D.jl")

end