using Makie: RGBA # for visualization
using Makie, GLMakie
using MyFirstPackage # our package
using CSV

# Set up the visualization with Makie:
lb = example_d3q19()
println(typeof(lb.grid))
CSV.write("data.csv",lb.grid)
println(Ref(lb.config))
vorticity = Observable(collect.(permutedims(_curl(momentum.(Ref(lb.config), lb.grid)),(2,3,1))))
fig, ax, plot = image(vorticity, colormap = :jet, colorrange = (-0.1, 0.1))

# Show barrier
barrier_img = map(x -> x ? RGBA(0, 0, 0, 1) : RGBA(0, 0, 0, 0), lb.barrier)
image!(ax, barrier_img')

# Recording the simulation
record(fig, joinpath(@__DIR__, "barrier.mp4"), 1:100; framerate = 10) do i
    for i=1:20
        _step!(lb)
    end
    vorticity[] = collect.(permutedims(_curl(momentum.(Ref(lb.config), lb.grid)), (2, 3, 1)))
end