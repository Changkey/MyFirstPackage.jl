#= 

Ref: 
1. https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/divergence-and-curl-articles/a/curl
2. https://www.bilibili.com/read/cv22678528/

=#

"""
    D3Q19 <: AbstractLBConfig{3, 19}

A lattice Boltzmann configuration for 3D, 19-velocity model.
"""
struct D3Q19 <: AbstractLBConfig{3,19} end
directions(::D3Q19) = (
    Point(0, 0, 1), Point(0, 1, 0), Point(0, 1, 1),
    Point(1, 0, 0), Point(1, 0, 1), Point(1, 1, 0),
    Point(1, -1, 0), Point(1, 0, -1), Point(0, 1, -1),
    Point(0, 0, 0),
    Point(0, -1, 1), Point(-1, 0, 1), Point(-1, 1, 0),
    Point(-1, -1, 0), Point(-1, 0, -1), Point(-1, 0, 0),
    Point(0, -1, -1), Point(0, -1, 0), Point(0, 0, -1),
)

# REWRITE directions[k] is the opposite of directions[flip_direction_index(k)]
function flip_direction_index(::D3Q19, i::Int)
    return 20 - i
end

# REWRITE streaming step
function stream!(
    lb::AbstractLBConfig{3,N},  # lattice configuration
    newgrid::AbstractArray{D,3}, # the updated grid
    grid::AbstractArray{D,3}, # the original grid
    barrier::AbstractArray{Bool,3} # the barrier configuration
) where {N,T,D<:Cell{N,T}}
    ds = directions(lb)
    @inbounds for ci in CartesianIndices(newgrid)
        i, j, k = ci.I
        newgrid[ci] = Cell(ntuple(N) do q # collect the densities
            ei = ds[q]
            m, n, o = size(grid)
            i2, j2, k2 = mod1(i - ei[1], m), mod1(j - ei[2], n), mod1(q - ei[3], o)
            if barrier[i2, j2, k2]
                # if the cell is a barrier, the fluid flows back
                density(grid[i, j, k], flip_direction_index(lb, q))
            else
                # otherwise, the fluid flows to the neighboring cell
                density(grid[i2, j2, k2], q)
            end
        end)
    end
end

# the distribution of the 19 velocities at the equilibrium state
weights(::D3Q19) = (1 / 18, 1 / 18, 1 / 36, 1 / 18, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 3, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 18, 1 / 36, 1 / 18, 1 / 18)

"""
    _LatticeBoltzmann{D, N, T, CFG, MT, BT}

A lattice Boltzmann simulation with D dimensions, N velocities, and lattice configuration CFG.
"""

struct _LatticeBoltzmann{D,N,T,CFG<:AbstractLBConfig{D,N},MT<:AbstractArray{Cell{N,T},D},BT<:AbstractArray{Bool,D}}
    config::CFG # lattice configuration
    grid::MT    # density of the fluid
    gridcache::MT # cache for the density of the fluid
    barrier::BT # barrier configuration
end

function _LatticeBoltzmann(config::AbstractLBConfig{D,N}, grid::AbstractArray{<:Cell,D}, barrier::AbstractArray{Bool,D}) where {D,N}
    @assert size(grid) == size(barrier)
    return _LatticeBoltzmann(config, grid, similar(grid), barrier)
end

"""
    step!(lb::LatticeBoltzmann)

Perform a single step of the lattice Boltzmann simulation.
"""
function _step!(lb::_LatticeBoltzmann)
    copyto!(lb.gridcache, lb.grid)
    stream!(lb.config, lb.grid, lb.gridcache, lb.barrier)
    lb.grid .= collide.(Ref(lb.config), lb.grid)
    return lb
end

"""
_curl(u::Array{Point3D{T},3})

Compute the curl of the momentum field in 3D, which is defined as:
```math
∂u_y/∂x−∂u_x/∂y
```
"""
function _curl(u::Array{Point3D{T},3}) where {T}
    return map(CartesianIndices(u)) do ci
        i, j, k = ci.I
        m, n, o = size(u)
        uy = u[mod1(i + 1, m), j, k][2] - u[mod1(i - 1, m), j, k][2]
        ux = u[i, mod1(j + 1, n), k][1] - u[i, mod1(j - 1, n), k][1]
        uz = u[i, j, mod1(k + 1, o)][3] - u[i, j, mod1(k - 1, o)][3]
        return (uy - ux, ux - uz, uz - uy)
    end
end

function example_d3q19(;
    height=80, width=100, length=80,
    u0=Point(0.0, 0.1, 0.2)) # initial and in-flow speed
    # Initialize all the arrays to steady rightward flow:
    rho = equilibrium_density(D3Q19(), 1.0, u0)
    # println(rho)
    rgrid = fill(rho, height, width, length)

    # Initialize barriers:
    barrier = falses(height, width, length)  # True wherever there's a barrier
    mid = div(height, 2)
    barrier[mid-8:mid+8, div(height, 2), div(length, 2)] .= true              # simple linear barrier

    return _LatticeBoltzmann(D3Q19(), rgrid, barrier)
end