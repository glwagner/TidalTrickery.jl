using Oceananigans
using Oceananigans.Units
using Oceananigans.Operators
using Oceananigans.Fields: ConstantField
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Coriolis: x_f_cross_U, y_f_cross_U
using Oceananigans.Grids: total_size, offset_data, xnode, ynode
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using LinearAlgebra
using IterativeSolvers
using Statistics
using GLMakie

const U_loc = (Face, Center, Center)
const V_loc = (Center, Face, Center)
const η_loc = (Center, Center, Center)

struct ShallowWaterTidalOperator{G, C, D, E, R, RU, RV}
    grid :: G
    coriolis :: C
    gravitational_acceleration :: Float64
    depth :: D
    damping_timescale :: Float64
    tidal_frequency :: Float64
    equilibrium_tide :: E
    right_hand_side :: R
    u_right_hand_side :: RU
    v_right_hand_side :: RV
end

Base.eltype(A::ShallowWaterTidalOperator) = eltype(A.grid)

function Base.size(A::ShallowWaterTidalOperator, d)
    U_size = total_size(U_loc, A.grid)
    V_size = total_size(V_loc, A.grid)
    η_size = total_size(η_loc, A.grid)

    NU = prod(U_size)
    NV = prod(V_size)
    Nη = prod(η_size)

    return NU + NV + Nη
end

function ShallowWaterTidalOperator(grid;
                                   coriolis = HydrostaticSphericalCoriolis(),
                                   gravitational_acceleration = 9.81,
                                   depth = ConstantField(3000),
                                   damping_timescale = 1days,
                                   γ₂ = 0.69,
                                   tidal_frequency = 2π / 12.421hours)

    equilibrium_tide = ηₑ = CenterField(grid, Complex{Float64})
    compute_equilibrium_tide!(ηₑ, γ₂, gravitational_acceleration)
    fill_halo_regions!(ηₑ)

    U_size = total_size(U_loc, grid)
    V_size = total_size(V_loc, grid)
    η_size = total_size(η_loc, grid)

    NU = prod(U_size)
    NV = prod(V_size)
    Nη = prod(η_size)
    N = NU + NV + Nη
    right_hand_side = zeros(Complex{Float64}, N)

    iU = 1:NU
    iV = NU+1:NU+NV

    RU, RV, _ = vector_to_shallow_water_fields(right_hand_side, grid)

    Nx, Ny, Nz = size(grid)
    g = gravitational_acceleration
    H = depth
    k = 1
    for j = 1:Ny
        for i = 1:Nx
            @inbounds begin
                RU[i, j, k] = - x_equilibrium_tide_gradient(i, j, k, grid, g, H, ηₑ)
                RV[i, j, k] = - y_equilibrium_tide_gradient(i, j, k, grid, g, H, ηₑ)
            end
        end
    end

    fill_halo_regions!(RU)
    fill_halo_regions!(RV)

    return ShallowWaterTidalOperator(grid,
                         coriolis,
                         gravitational_acceleration,
                         depth,
                         damping_timescale,
                         tidal_frequency,
                         equilibrium_tide,
                         right_hand_side, RU, RV)
end

function compute_equilibrium_tide!(ηₑ, γ₂, g)
    grid = ηₑ.grid
    Nx, Ny, Nz = size(grid)

    k = 1
    for j = 1:Ny, i = 1:Nx
        λ = xnode(Center(), i, grid)
        φ = ynode(Center(), j, grid)
        @inbounds ηₑ[i, j, k] = γ₂ / g * exp(2im * deg2rad(λ)) * cosd(φ)^2
    end
end

function x_equilibrium_tide_gradient(i, j, k, grid, g, H, ηₑ)
    Hᶠᶜᶜ = @inbounds min(H[i, j, k], H[i-1, j, k])
    return g * Hᶠᶜᶜ * ∂xᶠᶜᶜ(i, j, k, grid, ηₑ)
end

function y_equilibrium_tide_gradient(i, j, k, grid, g, H, ηₑ)
    Hᶜᶠᶜ = @inbounds min(H[i, j, k], H[i, j-1, k])
    return g * Hᶜᶠᶜ * ∂yᶜᶠᶜ(i, j, k, grid, ηₑ)
end

import Base: *

function *(A::ShallowWaterTidalOperator, solution)
    result = similar(solution)
    mul!(result, A, solution)
    return result
end

function LinearAlgebra.mul!(result, A::ShallowWaterTidalOperator, solution)
    grid = A.grid
    result .= 0

    U,  V,  η  = vector_to_shallow_water_fields(solution, grid)
    RU, RV, Rη = vector_to_shallow_water_fields(result, grid)

    mask_immersed_field!(U)
    mask_immersed_field!(V)
    mask_immersed_field!(η)

    fill_halo_regions!(U)
    fill_halo_regions!(V)
    fill_halo_regions!(η)

    Nx, Ny, Nz = size(grid)

    k = 1
    for j = 1:Ny, i = 1:Nx
        @inbounds begin
            RU[i, j, k] = u_tidal_operator(i, j, k, grid, U, V, η, A)
            RV[i, j, k] = v_tidal_operator(i, j, k, grid, U, V, η, A)
            Rη[i, j, k] = η_tidal_operator(i, j, k, grid, U, V, η, A)
        end
    end

    mask_immersed_field!(RU)
    mask_immersed_field!(RV)
    mask_immersed_field!(Rη)

    fill_halo_regions!(RU)
    fill_halo_regions!(RV)
    fill_halo_regions!(Rη)

    return result
end

@inline function u_tidal_operator(i, j, k, grid, U, V, η, tidal_operator)
    coriolis = tidal_operator.coriolis
    g = tidal_operator.gravitational_acceleration
    τ = tidal_operator.damping_timescale
    H = tidal_operator.depth
    ω = tidal_operator.tidal_frequency
    velocities = (u=U, v=V, w=0)
    Hᶠᶜᶜ = @inbounds min(H[i, j, k], H[i-1, j, k])

    return @inbounds (- im * ω * U[i, j, k]
                      + x_f_cross_U(i, j, k, grid, coriolis, velocities)
                      + g * Hᶠᶜᶜ * ∂xᶠᶜᶜ(i, j, k, grid, η)
                      + U[i, j, k] / τ)
end

@inline function v_tidal_operator(i, j, k, grid, U, V, η, tidal_operator)
    coriolis = tidal_operator.coriolis
    g = tidal_operator.gravitational_acceleration
    τ = tidal_operator.damping_timescale
    H = tidal_operator.depth
    ω = tidal_operator.tidal_frequency
    velocities = (u=U, v=V, w=0)
    Hᶜᶠᶜ = @inbounds min(H[i, j, k], H[i, j-1, k])

    return @inbounds (- im * ω * V[i, j, k]
                      + y_f_cross_U(i, j, k, grid, coriolis, velocities)
                      + g * Hᶜᶠᶜ * ∂yᶜᶠᶜ(i, j, k, grid, η)
                      + V[i, j, k] / τ)
end

@inline function η_tidal_operator(i, j, k, grid, U, V, η, tidal_operator)
    ω = tidal_operator.tidal_frequency
    return @inbounds - im * ω * η[i, j, k] + div_xyᶜᶜᶜ(i, j, k, grid, U, V)
end

function vector_to_shallow_water_fields(solution, grid)
    U_size = total_size(U_loc, grid)
    V_size = total_size(V_loc, grid)
    η_size = total_size(η_loc, grid)

    U_size = total_size(U_loc, grid)
    V_size = total_size(V_loc, grid)
    η_size = total_size(η_loc, grid)

    NU = prod(U_size)
    NV = prod(V_size)
    Nη = prod(η_size)

    iU = 1:NU
    iV = NU+1:NU+NV
    iη = NU+NV+1:NU+NV+Nη

    U_parent = reshape(view(solution, iU), U_size...)
    V_parent = reshape(view(solution, iV), V_size...)
    η_parent = reshape(view(solution, iη), η_size...)

    U_data = offset_data(U_parent, grid, U_loc) 
    V_data = offset_data(V_parent, grid, V_loc) 
    η_data = offset_data(η_parent, grid, η_loc) 

    v_boundary_conditions = FieldBoundaryConditions(grid, (Center, Face, Center);
                                                    north = OpenBoundaryCondition(nothing),
                                                    south = OpenBoundaryCondition(nothing))

    U = XFaceField(grid, data=U_data)
    V = YFaceField(grid, data=V_data, boundary_conditions=v_boundary_conditions)
    η = CenterField(grid, data=η_data)

    return U, V, η
end
