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
    newgrid::AbstractMatrix{D}, # the updated grid
    grid::AbstractMatrix{D}, # the original grid
    barrier::AbstractMatrix{Bool} # the barrier configuration
) where {N,T,D<:Cell{N,T}}
    ds = directions(lb)
    @inbounds for ci in CartesianIndices(newgrid)
        i, j = ci.I
        newgrid[ci] = Cell(ntuple(N) do k # collect the densities
            ei = ds[k]
            m, n = size(grid)
            i2, j2 = mod1(i - ei[1], m), mod1(j - ei[2], n)
            if barrier[i2, j2]
                # if the cell is a barrier, the fluid flows back
                density(grid[i, j], flip_direction_index(lb, k))
            else
                # otherwise, the fluid flows to the neighboring cell
                density(grid[i2, j2], k)
            end
        end)
    end
end

"""
    equilibrium_density(lb::AbstractLBConfig, ρ, u)

Compute the equilibrium density of the fluid from the total density and the momentum.
"""
function equilibrium_density(lb::AbstractLBConfig{D,N}, ρ, u) where {D,N}
    ws, ds = weights(lb), directions(lb)
    return Cell(
        ntuple(i -> ρ * ws[i] * _equilibrium_density(u, ds[i]), N)
    )
end

# the distribution of the 19 velocities at the equilibrium state
weights(::D3Q19) = (1 / 18, 1 / 18, 1 / 36, 1 / 18, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 3, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 18, 1 / 36, 1 / 18, 1 / 18)
function _equilibrium_density(u, ei)
    # the equilibrium density of the fluid with a specific mean momentum
    return (1 + 3 * dot(ei, u) + 9 / 2 * dot(ei, u)^2 - 3 / 2 * dot(u, u))
end

# collision step, applied on a single cell
function collide(lb::AbstractLBConfig{D,N}, rho; viscosity=0.02) where {D,N}
    omega = 1 / (3 * viscosity + 0.5)   # "relaxation" parameter
    # Recompute macroscopic quantities:
    v = momentum(lb, rho)
    return (1 - omega) * rho + omega * equilibrium_density(lb, density(rho), v)
end


"""
    curl(u::AbstractMatrix{Point2D{T}})

Compute the curl of the momentum field in 2D, which is defined as:
```math
∂u_y/∂x−∂u_x/∂y
```
"""
function curl(u::Matrix{Point3D{T}}) where {T}
    return map(CartesianIndices(u)) do ci
        i, j, k = ci.I
        m, n, o = size(u)
        uy = u[mod1(i + 1, m), j, k][2] - u[mod1(i - 1, m), j, k][2]
        ux = u[i, mod1(j + 1, n), k][1] - u[i, mod1(j - 1, n), k][1]
        uz = u[i, j, mod1(k + 1, o)][3] - u[i, j, mod1(k - 1, o)][3]
        return (uy - ux, ux - uz, uz - uy) # a factor of 1/2 is missing here?
    end
end

function example_d2q9(;
    height=80, width=200,
    u0=Point(0.0, 0.1)) # initial and in-flow speed
    # Initialize all the arrays to steady rightward flow:
    rho = equilibrium_density(D2Q9(), 1.0, u0)
    rgrid = fill(rho, height, width)

    # Initialize barriers:
    barrier = falses(height, width)  # True wherever there's a barrier
    mid = div(height, 2)
    barrier[mid-8:mid+8, div(height, 2)] .= true              # simple linear barrier

    return LatticeBoltzmann(D2Q9(), rgrid, barrier)
end