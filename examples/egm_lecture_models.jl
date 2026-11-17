# Solve the models of four QuantEcon lectures with the endogenous grid method,
# using the default parameters of each lecture.
#
#   1. Optimal Growth II: Time Iteration (Julia lecture)
#   2. The Income Fluctuation Problem III (Python lecture)
#   3. The Income Fluctuation Problem V (Python lecture)
#   4. Optimal Savings III: Occasionally Binding Constraints (Julia lecture)
#
# Run from the repository root:
#     julia --project=. examples/egm_lecture_models.jl

using QuantEcon
using Printf
using Random

# Largest relative Euler residual at interior choices, and whether the Euler
# inequality holds where the constraint binds
function euler_check(p, sol, xs)
    s_min = p.s_grid[1]
    worst, ineq_ok = 0.0, true
    for j in eachindex(p.z_vals), x in xs
        c = sol(x, j)
        s = x - c
        rhs = 0.0
        for k in eachindex(p.z_vals), l in eachindex(p.ϵ_vals)
            ϵ = p.ϵ_vals[l]
            rhs += p.Π[j, k] * p.ϵ_weights[l] * p.Γ_s(s, p.z_vals[k], ϵ) *
                   p.u_prime(sol(p.Γ(s, p.z_vals[k], ϵ), k))
        end
        rhs *= p.β
        lhs = p.u_prime(c)
        if s > s_min + 1e-10
            worst = max(worst, abs(lhs - rhs) / lhs)
        else
            ineq_ok = ineq_ok && lhs >= rhs * (1 - 1e-9)
        end
    end
    return worst, ineq_ok
end

function report(name, sol)
    @printf("%s\n  converged: %s, iterations: %d\n", name, sol.converged,
            sol.num_iter)
end

function print_policy(sol, xs, labels)
    @printf("  %8s", "x")
    for lab in labels
        @printf("  %12s", lab)
    end
    println()
    for x in xs
        @printf("  %8.2f", x)
        for j in eachindex(labels)
            @printf("  %12.4f", sol(x, j))
        end
        println()
    end
end

# == 1. Optimal Growth II =================================================
# x′ = f(s) ξ′ with f(k) = k^α, log utility, lognormal shocks
let α = 0.65, β = 0.95
    shocks = exp.(0.0 .+ 0.1 .* randn(MersenneTwister(42), 250))
    p = EGMProblem(; β=β, u_prime=c -> 1 / c, u_prime_inv=v -> 1 / v,
                   Γ=(s, z, ϵ) -> s^α * ϵ,
                   Γ_s=(s, z, ϵ) -> α * s^(α - 1) * ϵ,
                   s_grid=range(1e-6, 4.0, length=200), ϵ_vals=shocks)
    sol = solve(p)
    report("1. Optimal Growth II (log utility)", sol)
    err = maximum(abs(sol(y, 1) - (1 - α * β) * y)
                  for y in range(0.05, 4.0, length=200))
    @printf("  max |σ(y) - (1 - αβ) y| = %.2e\n", err)

    γ = 1.5      # exercise in the lecture: CRRA utility
    p = EGMProblem(; β=β, u_prime=c -> c^(-γ), u_prime_inv=v -> v^(-1 / γ),
                   Γ=(s, z, ϵ) -> s^α * ϵ,
                   Γ_s=(s, z, ϵ) -> α * s^(α - 1) * ϵ,
                   s_grid=range(1e-6, 4.0, length=200), ϵ_vals=shocks)
    sol = solve(p)
    report("   Optimal Growth II (CRRA, γ = 1.5)", sol)
    worst, _ = euler_check(p, sol, range(0.05, 3.5, length=100))
    @printf("  max Euler residual = %.2e\n", worst)
end

# == 2. Income Fluctuation Problem III ====================================
# a′ = R (a - c) + y(z′), CRRA utility, two-state Markov income
let
    p = ifp_problem(; s_grid=power_grid(16.0, 200))   # lecture defaults
    sol = solve(p)
    report("2. Income Fluctuation Problem III", sol)
    worst, ineq_ok = euler_check(p, sol, range(0.01, 15.0, length=300))
    @printf("  max Euler residual = %.2e, inequality holds: %s\n", worst, ineq_ok)
    print_policy(sol, (0.5, 1.0, 5.0, 10.0), ("bad state", "good state"))
end

# == 3. Income Fluctuation Problem V ======================================
# a′ = R(ζ′) (a - c) + Y(z′, η′), 100 × 100 shock draws as in the lecture
let γ = 1.5, β = 0.96, a_r = 0.16, b_r = 0.0, a_y = 0.2, b_y = 0.5
    rng = MersenneTwister(1234)
    η_draws, ζ_draws = randn(rng, 100), randn(rng, 100)
    pairs = vec([(ζ, η) for ζ in ζ_draws, η in η_draws])
    p = EGMProblem(; β=β, u_prime=c -> c^(-γ), u_prime_inv=v -> v^(-1 / γ),
                   Γ=(s, z, ϵ) -> exp(a_r * ϵ[1] + b_r) * s +
                                  exp(a_y * ϵ[2] + z * b_y),
                   Γ_s=(s, z, ϵ) -> exp(a_r * ϵ[1] + b_r),
                   s_grid=power_grid(100.0, 100),
                   mc=MarkovChain([0.9 0.1; 0.1 0.9], [0.0, 1.0]),
                   ϵ_vals=pairs)
    sol = solve(p)
    report("3. Income Fluctuation Problem V", sol)
    worst, ineq_ok = euler_check(p, sol, range(0.05, 60.0, length=60))
    @printf("  max Euler residual = %.2e, inequality holds: %s\n", worst, ineq_ok)
    @printf("  constraint binds for a ≤ %.3f (z = 0), %.3f (z = 1)\n",
            sol.a[2, 1], sol.a[2, 2])
    print_policy(sol, (1.0, 5.0, 20.0, 50.0), ("z = 0", "z = 1"))
end

# == 4. Optimal Savings III: Occasionally Binding Constraints =============
# c + a′ ≤ R a + z and a′ ≥ -b. With cash on hand x = R a + z this is
# s = x - c ≥ -b and x′ = R s + z′.
let
    mc = MarkovChain([0.6 0.4; 0.05 0.95], log.([0.5, 1.0]))
    for (r, b) in ((0.01, 0.0), (0.03, 0.0), (0.01, 1.0), (0.01, 3.0))
        R = 1 + r
        p = ifp_problem(; R=R, γ=1.0, mc=mc, b=b,
                        s_grid=power_grid(16.0, 200; s_min=-b))
        sol = solve(p)
        report(@sprintf("4. Optimal Savings III (r = %.2f, b = %.1f)", r, b), sol)
        worst, ineq_ok = euler_check(p, sol, range(-b + 0.05, 15.0, length=300))
        @printf("  max Euler residual = %.2e, inequality holds: %s\n", worst, ineq_ok)
        # consumption as a function of assets a in the timing of the lecture
        println("  consumption c(a, z) = σ(R a + z, z):")
        @printf("  %8s  %12s  %12s\n", "a", "z = 0.5", "z = 1.0")
        for a in (-b, 0.0, 2.0, 5.0, 10.0)
            @printf("  %8.2f  %12.4f  %12.4f\n", a,
                    sol(R * a + 0.5, 1), sol(R * a + 1.0, 2))
        end
    end
end
