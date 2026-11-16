#=
Endogenous grid method (EGM) for one-dimensional consumption-savings
problems with a post-decision state.

References
----------
Carroll, C. D. (2006). The method of endogenous gridpoints for solving
dynamic stochastic optimization problems. Economics Letters, 91(3), 312-320.

https://julia.quantecon.org/dynamic_programming/egm_policy_iter.html
=#

"""
    EGMProblem{TU,TUI,TG,TGS}

Consumption-savings problem that can be solved by the endogenous grid
method. The household maximizes the expected discounted sum of `u(c)`
subject to `s = x - c ≥ s_min` and `x′ = Γ(s, z′, ϵ′)`, where `x` is the
endogenous state, `s` is savings (the post-decision state), `z` follows a
finite Markov chain and `ϵ` is an IID shock. The lower bound `s_min` is the
first point of the savings grid. The shock `ϵ` may be a number or a tuple,
for example `(ζ, η)` for a return shock and an income shock. At interior choices the Euler
equation is `u′(c) = β E[Γ_s(s, z′, ϵ′) u′(σ(x′, z′)) | z]`.

# Fields

- `β::Float64`: Discount factor.
- `u_prime::TU`: Marginal utility `u′(c)`.
- `u_prime_inv::TUI`: Inverse of the marginal utility.
- `Γ::TG`: Next period state `Γ(s, z′, ϵ′)`.
- `Γ_s::TGS`: Derivative of `Γ` with respect to `s`, called as `Γ_s(s, z′, ϵ′)`.
- `Π::Matrix{Float64}`: Transition matrix of the exogenous state.
- `z_vals::Vector{Float64}`: Values of the exogenous state.
- `ϵ_vals::Vector{TE}`: Nodes of the IID shock. Each element is a number or a tuple.
- `ϵ_weights::Vector{Float64}`: Weights of the nodes, summing to one.
- `s_grid::Vector{Float64}`: Exogenous grid for savings. `s_grid[1]` is the lower bound of savings.
"""
struct EGMProblem{TU,TUI,TG,TGS,TE}
    β::Float64
    u_prime::TU
    u_prime_inv::TUI
    Γ::TG
    Γ_s::TGS
    Π::Matrix{Float64}
    z_vals::Vector{Float64}
    ϵ_vals::Vector{TE}
    ϵ_weights::Vector{Float64}
    s_grid::Vector{Float64}
end

"""
    EGMProblem(; β, u_prime, u_prime_inv, Γ, Γ_s, s_grid, mc, ϵ_vals, ϵ_weights)

Construct an `EGMProblem` from keyword arguments.

# Arguments

- `;β::Real`: Discount factor.
- `;u_prime`: Marginal utility, called as `u_prime(c)`.
- `;u_prime_inv`: Inverse of the marginal utility.
- `;Γ`: Next period state, called as `Γ(s, z′, ϵ′)`.
- `;Γ_s`: Derivative of `Γ` with respect to `s`, called as `Γ_s(s, z′, ϵ′)`.
- `;s_grid::AbstractVector`: Increasing grid for savings. `s_grid[1]` is the lower bound of savings, for example `-b` for a borrowing limit `b`.
- `;mc::MarkovChain`: Markov chain of the exogenous state. The default is a chain with a single state.
- `;ϵ_vals::AbstractVector([0.0])`: Nodes of the IID shock. Each element is passed to `Γ` and `Γ_s` as is, so it can be a tuple when there are several shocks.
- `;ϵ_weights::AbstractVector`: Weights of the nodes. The default is equal weights.

# Returns

- `p::EGMProblem`: The problem instance.
"""
function EGMProblem(; β, u_prime, u_prime_inv, Γ, Γ_s, s_grid,
                    mc::MarkovChain=MarkovChain(ones(1, 1), [0.0]),
                    ϵ_vals=[0.0],
                    ϵ_weights=fill(1 / length(ϵ_vals), length(ϵ_vals)))
    Π = Matrix{Float64}(mc.p)
    z = collect(Float64, mc.state_values)
    ϵ = collect(ϵ_vals)
    w = collect(Float64, ϵ_weights)
    s = collect(Float64, s_grid)
    if length(w) != length(ϵ) || !isapprox(sum(w), 1.0)
        throw(ArgumentError("ϵ_weights must match ϵ_vals and sum to 1"))
    end
    if length(s) < 2 || !all(diff(s) .> 0)
        throw(ArgumentError("s_grid must be increasing with at least two points"))
    end
    return EGMProblem(Float64(β), u_prime, u_prime_inv, Γ, Γ_s, Π, z, ϵ, w, s)
end

"""
    EGMSolution

Solution of an `EGMProblem`. Calling `sol(x, j)` returns consumption at the
endogenous state `x` and the `j`-th exogenous state.

# Fields

- `a::Matrix{Float64}`: Endogenous grid, one column for each exogenous state. The first row is the lower bound of savings `s_min`, where consumption is zero.
- `c::Matrix{Float64}`: Consumption on the endogenous grid.
- `num_iter::Int`: Number of iterations.
- `converged::Bool`: Whether the iteration converged.
"""
struct EGMSolution
    a::Matrix{Float64}
    c::Matrix{Float64}
    num_iter::Int
    converged::Bool
end

(sol::EGMSolution)(x::Real, j::Integer) =
    _egm_interp(view(sol.a, :, j), view(sol.c, :, j), x)

# Linear interpolation. Returns `ys[1]` below the grid and extrapolates
# linearly from the last two points above it.
function _egm_interp(xs::AbstractVector, ys::AbstractVector, x::Real)
    n = length(xs)
    x <= xs[1] && return float(ys[1])
    i = min(searchsortedlast(xs, x), n - 1)
    t = (x - xs[i]) / (xs[i+1] - xs[i])
    return (1 - t) * ys[i] + t * ys[i+1]
end

# The policy that saves the minimum, c = x - s_min, as a pair (a, c)
function _egm_initial_policy(p::EGMProblem)
    n_s, n_z = length(p.s_grid), length(p.z_vals)
    s_min = p.s_grid[1]
    g = collect(range(s_min, max(p.s_grid[end], s_min + 1.0), length=n_s + 1))
    a = repeat(g, 1, n_z)
    return a, a .- s_min
end

# Next period state must not fall below the lower bound of savings when
# savings are at the lower bound; otherwise consumption would be negative.
function _egm_check_feasible(p::EGMProblem)
    s_min = p.s_grid[1]
    for z in p.z_vals, ϵ in p.ϵ_vals
        if p.Γ(s_min, z, ϵ) < s_min - 1e-12
            throw(ArgumentError("lower bound of savings is too loose: " *
                                "Γ(s_min, z′, ϵ′) < s_min for some states"))
        end
    end
    return nothing
end

"""
    power_grid(s_max, n; k=3.0, s_min=0.0)

Grid on `[s_min, s_max]` with points concentrated near `s_min`, given by
`s_min + (s_max - s_min) * ((i - 1) / (n - 1))^k` for `i = 1, ..., n`.

# Arguments

- `s_max::Real`: Largest grid point.
- `n::Integer`: Number of grid points.
- `;k::Real(3.0)`: Curvature. `k = 1` gives an evenly spaced grid.
- `;s_min::Real(0.0)`: Smallest grid point.

# Returns

- `grid::Vector{Float64}`: The grid.
"""
power_grid(s_max::Real, n::Integer; k::Real=3.0, s_min::Real=0.0) =
    [s_min + (s_max - s_min) * ((i - 1) / (n - 1))^k for i in 1:n]

"""
    egm_step(p, a_in, c_in)

Apply the Coleman-Reffett operator once, using the endogenous grid method.
The current policy is given by consumption `c_in` on the endogenous grid
`a_in`, with one column for each exogenous state.

# Arguments

- `p::EGMProblem`: The problem.
- `a_in::AbstractMatrix`: Current endogenous grid.
- `c_in::AbstractMatrix`: Current consumption on `a_in`.

# Returns

- `a_out::Matrix{Float64}`: Updated endogenous grid. The first row is the lower bound of savings.
- `c_out::Matrix{Float64}`: Updated consumption on `a_out`.
"""
function egm_step(p::EGMProblem, a_in::AbstractMatrix, c_in::AbstractMatrix)
    n_s, n_z, n_ϵ = length(p.s_grid), length(p.z_vals), length(p.ϵ_vals)

    # m[i, k] = E_ϵ[Γ_s(s_i, z_k, ϵ) u′(σ(Γ(s_i, z_k, ϵ), z_k))]
    m = zeros(n_s, n_z)
    for k in 1:n_z
        z_next = p.z_vals[k]
        a_k, c_k = view(a_in, :, k), view(c_in, :, k)
        for i in 1:n_s
            s = p.s_grid[i]
            acc = 0.0
            for l in 1:n_ϵ
                ϵ_next = p.ϵ_vals[l]
                c_next = _egm_interp(a_k, c_k, p.Γ(s, z_next, ϵ_next))
                acc += p.ϵ_weights[l] * p.Γ_s(s, z_next, ϵ_next) *
                       p.u_prime(c_next)
            end
            m[i, k] = acc
        end
    end

    # first row: at x = s_min consumption is zero, and the straight line to the
    # next point gives c = x - s_min where the constraint binds
    a_out = fill(p.s_grid[1], n_s + 1, n_z)
    c_out = zeros(n_s + 1, n_z)
    for j in 1:n_z, i in 1:n_s
        expectation = 0.0
        for k in 1:n_z
            expectation += p.Π[j, k] * m[i, k]
        end
        c = p.u_prime_inv(p.β * expectation)
        c_out[i+1, j] = c
        a_out[i+1, j] = p.s_grid[i] + c
    end
    return a_out, c_out
end

"""
    solve(p; tol=1e-5, max_iter=1000)

Solve an `EGMProblem` by iterating `egm_step` from the policy that saves the
minimum, until the largest change in consumption is below `tol`.

# Arguments

- `p::EGMProblem`: The problem.
- `;tol::Real(1e-5)`: Convergence tolerance.
- `;max_iter::Integer(1000)`: Maximum number of iterations.

# Returns

- `sol::EGMSolution`: The solution.
"""
function solve(p::EGMProblem; tol::Real=1e-5, max_iter::Integer=1000)
    _egm_check_feasible(p)
    a, c = _egm_initial_policy(p)
    for it in 1:max_iter
        a_new, c_new = egm_step(p, a, c)
        for j in axes(a_new, 2)
            if !issorted(view(a_new, :, j))
                throw(ArgumentError("endogenous grid is not monotone; " *
                                    "the problem is outside the scope of EGM"))
            end
        end
        err = maximum(abs.(c_new .- c))
        a, c = a_new, c_new
        err < tol && return EGMSolution(a, c, it, true)
    end
    return EGMSolution(a, c, max_iter, false)
end

"""
    ifp_problem(; R, β, γ, mc, y, b, s_grid)

Income fluctuation problem with CRRA utility. The state `x` is cash on hand,
savings satisfy `s = x - c ≥ -b`, and `x′ = R * s + y(z′)`. In the timing
`c + a′ ≤ R * a + y(z)` with `a′ ≥ -b`, this corresponds to
`x = R * a + y(z)` and `s = a′`.

# Arguments

- `;R::Real(1.01)`: Gross interest rate.
- `;β::Real(0.96)`: Discount factor.
- `;γ::Real(1.5)`: Coefficient of relative risk aversion. `γ = 1` gives log utility.
- `;mc::MarkovChain`: Markov chain of the exogenous state.
- `;y(exp)`: Income as a function of the exogenous state.
- `;b::Real(0.0)`: Borrowing limit.
- `;s_grid::AbstractVector`: Grid for savings, starting at `-b`. The default is 200 evenly spaced points on `[-b, 16]`.

# Returns

- `p::EGMProblem`: The problem instance.
"""
function ifp_problem(; R::Real=1.01, β::Real=0.96, γ::Real=1.5,
                     mc::MarkovChain=MarkovChain([0.6 0.4; 0.05 0.95],
                                                 [-10.0, log(2.0)]),
                     y=exp, b::Real=0.0,
                     s_grid=range(-b, 16.0, length=200))
    β * R < 1 || throw(ArgumentError("stability condition βR < 1 violated"))
    first(s_grid) ≈ -b ||
        throw(ArgumentError("the first point of s_grid must be -b"))
    if γ == 1
        u_prime, u_prime_inv = c -> 1 / c, v -> 1 / v
    else
        u_prime, u_prime_inv = c -> c^(-γ), v -> v^(-1 / γ)
    end
    return EGMProblem(; β=β, u_prime=u_prime, u_prime_inv=u_prime_inv,
                      Γ=(s, z, ϵ) -> R * s + y(z),
                      Γ_s=(s, z, ϵ) -> R,
                      s_grid=s_grid, mc=mc)
end
