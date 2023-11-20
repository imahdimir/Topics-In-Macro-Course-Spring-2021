

## *****************************************************************************
#=
    Importing Packages
=#
using Random
using LinearAlgebra
using GLM
using QuantEcon
using Interpolations
using Optim
using Plots
pyplot()


## *****************************************************************************
#*** parameters
β = 0.99
α = 0.36
δ = 0.025
θ = 1
μ = 0.15

#*** unemployment and aggregate state duration
ug = 0.04
ub = 0.10

zg_ave_dur = 8
zb_ave_dur = 8

ug_ave_dur = 1.5
ub_ave_dur = 2.5

puu_rel_gb2bb =1.25
puu_rel_bg2gg =0.75


#*** model state space gride attributes
k_min = 0
k_max = 1000
k_size = 100

aggK_min = 30
aggK_max = 50
aggK_size = 4



## *****************************************************************************
#=
    create_transition_matrix
=#

pgg = 1-1/zg_ave_dur
pbb = 1-1/zb_ave_dur
pgb = 1-pgg
pbg = 1-pbb

p00_gg = 1-1/ug_ave_dur
p00_bb = 1-1/ub_ave_dur
p01_gg = 1-p00_gg
p01_bb = 1-p00_bb

p00_gb=puu_rel_gb2bb*p00_bb
p00_bg=puu_rel_bg2gg*p00_gg
p01_gb=1-p00_gb
p01_bg=1-p00_bg

p10_gg=(ug - ug*p00_gg)/(1-ug)
p10_bb=(ub - ub*p00_bb)/(1-ub)
p10_gb=(ub - ug*p00_gb)/(1-ug)
p10_bg=(ug - ub*p00_bg)/(1-ub)
p11_gg= 1-p10_gg
p11_bb= 1-p10_bb
p11_gb= 1-p10_gb
p11_bg= 1-p10_bg

Pz=[pgg pgb;
    pbg pbb]

Pϵ = fill(zeros(2,2), 2, 2)

# indexing the aggtegate productivity in the code is as follows:
bad  = 1
good = 2

Pϵ[good, good] = [p11_gg  p10_gg
                  p01_gg  p00_gg]
Pϵ[good, bad]  = [p11_gb  p10_gb
                  p01_gb  p00_gb]
Pϵ[bad, good]  = [p11_bg  p10_bg
                  p01_bg  p00_bg]
Pϵ[bad, bad]   = [p11_bb  p10_bb
                  p01_bb  p00_bb]

# indexing the employment status in the code is as follows:
emp   = 1
unemp = 2

P = zeros((2, 2, 2, 2))
P[bad,  bad, :,:] = Pϵ[bad, bad]   .* pbb
P[bad,  good,:,:] = Pϵ[bad, good]  .* pbg
P[good, good,:,:] = Pϵ[good, good] .* pgg
P[good, bad, :,:] = Pϵ[good, bad]  .* pgb


if θ == 1
    u = LogUtility()
else
    u = CRRAUtility(θ)
end

l_bar= 1/(1-ub)

# individual capital grid
k_grid= range(k_min, k_max, length=k_size)

# aggregate capital grid
aggK_grid= range(aggK_min, aggK_max, length=aggK_size)


## *****************************************************************************

struct TransitionMatrix
    P   ::Array{Float64,4}    # 2x2x2x2
    Pz  ::Matrix{Float64}     # 2x2 aggregate shock
    Pϵ  ::Array{Array{Float64,2},2}
end

# collection of transition matrices
transmat=TransitionMatrix(P, Pz, Pϵ);

p =(
    u            = u,
    β            = β,
    α            = α,
    δ            = δ,
    θ            = θ,
    l_bar        = l_bar,
    k_min        = k_min,
    k_max        = k_max,
    aggK_min     = aggK_min,
    aggK_max     = aggK_max,
    k_grid       = k_grid,
    aggK_grid    = aggK_grid,
    k_size       = k_size,
    aggK_size    = aggK_size,
    ug           = ug,
    ub           = ub,
    transmat     = transmat,
    μ            = μ
    );

## *****************************************************************************

rent(α::Real, z::Real, aggK::Real, aggL::Real)= α*z*(aggK/aggL)^(α-1)
wage(α::Real, z::Real, aggK::Real, aggL::Real)= (1-α)*z*(aggK/aggL)^(α)


initial_policy  = 0.9 * repeat(k_grid, outer=[1, 2, aggK_size, 2])
initial_policy .= clamp.(policy, k_min, k_max)
initial_value   = p.u.(0.1/0.9*policy)/(1-β)
law_of_motion   = [1.4914, 0.9648, 1.3804, 0.9623]

law_of_motion   = [0.14914, 0.9648, 0.13804, 0.9623]



mutable struct modelSolution
    policy  ::Array{Float64,4}
    value   ::Array{Float64,4}
    coeff   ::Vector{Float64}
end

sol = modelSolution(initial_policy, initial_value, law_of_motion) ;



## *****************************************************************************

function solution!(
    p           ::NamedTuple,
    sol         ::modelSolution
    ;
    max_iter    ::Integer          = 200,
    tol         ::AbstractFloat    = 1e-5,
    )

    counter_VFI = 0

    while true
        counter_VFI += 1

        value_old = copy(sol.value)

        # maximize value for all state
        for (index_k_i, k_i) in enumerate(p.k_grid)
            for (index_ϵ_i, ϵ_i) in enumerate([1,0])
                for (index_aggK_i, aggK_i) in enumerate(p.aggK_grid)
                    for (index_z_i, z_i) in enumerate([0.99, 1.01])

                        maximization!(
                            index_k_i, k_i,
                            index_ϵ_i, ϵ_i ,
                            index_aggK_i, aggK_i,
                            index_z_i, z_i,
                            p, sol, value_old)

                    end
                end
            end
        end

        # difference of guessed and new value
        dif = maximum(abs, value_old-sol.value)


        # if difference is sufficiently small
        if (dif < tol)
            println(" converged successfully. dif = $dif")
            break
        elseif (counter_VFI >= max_iter)
            println("maximum iteration reacged : $max_iter")
            break
        end
    end
end


function maximization!(
            index_k,
            k,
            index_ϵ,
            ϵ,
            index_aggK,
            aggK,
            index_z,
            z,
            p::NamedTuple,
            sol::modelSolution,
            value_old
            )

    # obtain minimum and maximum of grid
    k_min, k_max = p.k_grid[1], p.k_grid[end]

    # unpack parameters
    α, δ, l_bar, μ = p.α, p.δ, p.l_bar, p.μ

    # aggK_next, aggL = compute_Kp_L(aggK, z, sol.coeff, p) # next aggregate capital and current aggregate labor
    if (z > 1) # good state of the economy
        aggK_next = exp(sol.coeff[1] + sol.coeff[2]*log(aggK))
        aggL      = p.l_bar*(1-p.ug)
        τ         = (μ * p.ug)/(aggL)
    else     # bad state of the economy
        aggK_next = exp(sol.coeff[3] + sol.coeff[4]*log(aggK))
        aggL      = p.l_bar * (1-p.ub)
        τ         = (μ * p.ub)/(aggL)
    end
    aggK_next = clamp(aggK_next, p.aggK_min, p.aggK_max)


    # if kp>k_c_pos, consumption is negative
    r = rent(α, z, aggK, aggL)
    w = wage(α, z, aggK, aggL)
    k_c_pos = (r+1-δ)*k + w*((1-τ)ϵ*l_bar+(1-ϵ)*μ)

    objective(k_next)= -bellman_function(p, k_next, value_old, k, aggK, ϵ, z, index_ϵ, index_z, aggK_next, aggL, τ) # objective function

    result = optimize(objective, k_min, min(k_c_pos,k_max))   # maximize value

    # obtain result
    sol.policy[index_k, index_ϵ, index_aggK, index_z] =   Optim.minimizer(result)
    sol.value[index_k, index_ϵ, index_aggK, index_z]  = - Optim.minimum(result)

    return nothing
end



function bellman_function(
    p         :: NamedTuple,
    k_next    :: Real,
    value     :: Array{Float64, 4},
    k         :: Real,
    aggK      :: Real,
    ϵ         :: Real,
    z         :: Real,
    index_ϵ,
    index_z,
    aggK_next,
    aggL,
    τ
    )


    r = rent(α, z, aggK, aggL)
    w = wage(α, z, aggK, aggL)
    cons = (r+1-p.δ)*k + w*((1-τ)ϵ*l_bar+(1.0-ϵ)*p.μ) - k_next # current consumption

    # compute expectations by summing up
    expec = 0.0

    for (index_ϵ_next, ϵ_next) in enumerate([1,0])
        for (index_z_next, z_next) in enumerate([0.99, 1.01])

            value_itp = interpolate((p.k_grid, p.aggK_grid), value[:, index_ϵ_next, :, index_z_next], Gridded(Linear()))
            expec += p.transmat.P[index_z, index_z_next, index_ϵ, index_ϵ_next]*value_itp(k_next, aggK_next)

        end
    end

    return (u(cons) + β*expec)
end




## *****************************************************************************

solution!(
    p ,
    sol,
    max_iter   = 250,
    tol        = 1e-4
    )


plot(k_grid, sol.value[:,1,1,1])
plot!(k_grid, sol.value[:,1,2,1])



## *****************************************************************************


function simulation!(
    p               ::NamedTuple,
    sol             ::modelSolution;
    T               ::Integer = 1100,
    N               ::Integer = 10000
    )

    aggK_series = Vector{Float64}(undef, T)

    k_agents = rand(p.k_grid, N)
    k_agents .= 40

    # draw aggregate shock
    z_shocks = simulate(MarkovChain(p.transmat.Pz), T)

    ### Let's draw individual shock ###
    ϵ_shocks = Array{Int}(undef, T, N)

    # first period
    rand_draw = rand(N)
    # recall: index 1 of eps is employed, index 2 of eps is unemployed
    if (z_shocks[1] == 1)      # if bad
        ϵ_shocks[1, :] .= (rand_draw .< p.ub) .+ 1
    elseif (z_shocks[1] == 2)  # if good
        ϵ_shocks[1, :] .= (rand_draw .< p.ug) .+ 1
    else
        error("the value of z_shocks[1] (=$(z_shocks[1])) is strange")
    end

    # from second period
    for t = 2:T
        # loop over entire population
        for i=1:N

            z_this_period   = z_shocks[t]
            z_before        = z_shocks[t-1]

            Pϵ_z_zprime   = p.transmat.Pϵ[z_this_period, z_before]

            rand_draw = rand()
            ϵi_before = ϵ_shocks[t-1, i]

            ϵ_shocks[t, i] = (rand_draw < Pϵ_z_zprime[ϵi_before,2]) + 1
        end
    end


    for (t, z_i) = enumerate(z_shocks)
        # Aggregate capital hold by agents in period t
        aggK_series[t] = mean(k_agents)

        # loop over individuals
        for (i, k) in enumerate(k_agents)
            ϵ_i = ϵ_shocks[t, i]   # idiosyncratic shock

            # next captial agent i optimizes to hold is :
            policy_interpolate = interpolate((p.k_grid, p.aggK_grid), sol.policy[:, ϵ_i, :, z_i], Gridded(Linear()))
            aggregate_capital = clamp(aggK_series[t], p.aggK_grid[1], p.aggK_grid[end])
            k_agents[i] = policy_interpolate(k, aggregate_capital)
        end
    end

    return aggK_series, z_shocks
end


aggK_series, z_shocks = simulation!(p, sol, T=1100, N=10000)
plot(100:1100, aggK_series[100:end])



function simulated_law_of_motion!(
    p                ::NamedTuple,
    sol              ::modelSolution,
    z_shocks         ::Vector,
    aggK_series      ::Vector;
    T_discard        ::Integer=100
    )

    n_g=count(z_shocks[T_discard+1:end-1] .== 2)
    n_b=count(z_shocks[T_discard+1:end-1] .== 1)

    coeff_simulated = Vector{Float64}(undef, 4)
    x_g = Vector{Float64}(undef, n_g)
    y_g = Vector{Float64}(undef, n_g)
    x_b = Vector{Float64}(undef, n_b)
    y_b = Vector{Float64}(undef, n_b)

    global i_g = 0
    global i_b = 0
    for t = T_discard+1:length(z_shocks)-1
        if z_shocks[t] == 2
            global i_g = i_g+1
            x_g[i_g] = log(aggK_series[t])
            y_g[i_g] = log(aggK_series[t+1])
        else
            global i_b = i_b+1
            x_b[i_b] = log(aggK_series[t])
            y_b[i_b] = log(aggK_series[t+1])
        end
    end
    resg = lm([ones(n_g) x_g], y_g)
    resb = lm([ones(n_b) x_b], y_b)
    # kss.R2 = [r2(resg), r2(resb)]

    coeff_simulated[1], coeff_simulated[2] = coef(resg)
    coeff_simulated[3], coeff_simulated[4] = coef(resb)
    dif_B = maximum(abs, coeff_simulated-sol.coeff)
    println("difference of ALM coefficient is $dif_B and B = $coeff_simulated")
    return coeff_simulated, dif_B
end



simulated_law_of_motion!(p, sol, z_shocks, aggK_series,T_discard=100)
