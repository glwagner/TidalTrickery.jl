using Oceananigans
using Oceananigans.Units
using IterativeSolvers
using Statistics
using GLMakie

include("shallow_water_tidal_operators.jl")

grid = LatitudeLongitudeGrid(size = (180, 60),
                             latitude = (-60, 60),
                             longitude = (-180, 180),
                             topology = (Periodic, Bounded, Flat))

op = ShallowWaterTidalOperator(grid)

start_time = time_ns()
@info "Solving the tidal operator problem with IterativeSolvers.idrs..."
solution = idrs(op, op.right_hand_side) #op * solution
elapsed = 1e-9 * (time_ns() - start_time)
@info "    ... done (" * prettytime(elapsed) * ")."

@info "Does A * x = b?"
@info op * solution ≈ op.right_hand_side

#####
##### Visualize results
#####

U, V, η = vector_to_shallow_water_fields(solution, grid)

fig = Figure()

axh = Axis(fig[1, 1], title="Equilibrium tide")
axu = Axis(fig[1, 2], title="RHS of tidal operator u-component")
axv = Axis(fig[1, 3], title="RHS of tidal operator v-component")

heatmap!(axh, real.(interior(op.equilibrium_tide, :, :, 1)))
heatmap!(axu, real.(interior(op.u_right_hand_side, :, :, 1)))
heatmap!(axv, real.(interior(op.v_right_hand_side, :, :, 1)))

axh = Axis(fig[2, 1], title="real(η̂) stationary tide solution")
axu = Axis(fig[2, 2], title="real(Û) stationary tide solution")
axv = Axis(fig[2, 3], title="real(V̂) stationary tide solution")

heatmap!(axh, real.(interior(η, :, :, 1)))
heatmap!(axu, real.(interior(U, :, :, 1)))
heatmap!(axv, real.(interior(V, :, :, 1)))

display(fig)

