@testset "Testing egm.jl" begin

    @testset "interpolation" begin
        xs_egm, ys_egm = [0.0, 1.0, 2.0], [0.0, 1.0, 3.0]
        @test QuantEcon._egm_interp(xs_egm, ys_egm, 0.5) ≈ 0.5
        @test QuantEcon._egm_interp(xs_egm, ys_egm, 1.5) ≈ 2.0
        @test QuantEcon._egm_interp(xs_egm, ys_egm, 3.0) ≈ 5.0   # extrapolation
        @test QuantEcon._egm_interp(xs_egm, ys_egm, -1.0) ≈ 0.0
        @test QuantEcon._egm_interp([0.0, 0.0, 1.0], [0.0, 0.0, 2.0], 0.5) ≈ 1.0
    end

    @testset "power_grid" begin
        g_egm = power_grid(16.0, 5; k=2.0)
        @test g_egm ≈ [0.0, 1.0, 4.0, 9.0, 16.0]
        g2_egm = power_grid(3.0, 3; k=1.0, s_min=-1.0)
        @test g2_egm ≈ [-1.0, 1.0, 3.0]
    end

    @testset "constructor checks" begin
        f_egm = c -> 1 / c
        @test_throws ArgumentError EGMProblem(; β=0.9, u_prime=f_egm,
            u_prime_inv=f_egm, Γ=(s, z, ϵ) -> s, Γ_s=(s, z, ϵ) -> 1.0,
            s_grid=[1.0, 0.5])
        @test_throws ArgumentError ifp_problem(; R=1.1, β=0.96)
        @test_throws ArgumentError ifp_problem(; b=1.0, s_grid=0.0:0.1:16.0)
        # income 0.5 cannot repay the interest on a debt of 60 at r = 0.01
        p_loose_egm = ifp_problem(; γ=1.0, b=60.0,
            mc=MarkovChain([0.6 0.4; 0.05 0.95], log.([0.5, 1.0])))
        @test_throws ArgumentError solve(p_loose_egm)
    end

    @testset "optimal growth with log utility" begin
        α_egm, β_egm = 0.4, 0.96
        shocks_egm = exp.(0.1 .* randn(MersenneTwister(1234), 250))
        p_egm = EGMProblem(; β=β_egm,
                           u_prime=c -> 1 / c, u_prime_inv=v -> 1 / v,
                           Γ=(s, z, ϵ) -> s^α_egm * ϵ,
                           Γ_s=(s, z, ϵ) -> α_egm * s^(α_egm - 1) * ϵ,
                           s_grid=range(1e-4, 4.0, length=120),
                           ϵ_vals=shocks_egm)
        sol_egm = solve(p_egm)
        @test sol_egm.converged
        x_egm, c_egm = sol_egm.a[2:end, 1], sol_egm.c[2:end, 1]
        # analytical solution σ(x) = (1 - αβ) x
        @test maximum(abs.(c_egm .- (1 - α_egm * β_egm) .* x_egm)) < 1e-4
    end

    @testset "cake eating as a special case of the IFP" begin
        for γ_egm in (1.0, 1.5)
            β_egm = 0.94
            p_egm = ifp_problem(; R=1.0, β=β_egm, γ=γ_egm, y=z -> 0.0)
            sol_egm = solve(p_egm)
            @test sol_egm.converged
            # analytical solution c = (1 - β^(1/γ)) x
            errs_egm = [abs(sol_egm(x, 1) - (1 - β_egm^(1 / γ_egm)) * x)
                        for x in range(0.1, 15.0, length=50)]
            @test maximum(errs_egm) < 1e-3
        end
    end

    @testset "Euler equation in the IFP" begin
        R_egm, γ_egm = 1.01, 1.5
        p_egm = ifp_problem(; s_grid=power_grid(16.0, 200))
        sol_egm = solve(p_egm)
        @test sol_egm.converged
        max_res_egm = 0.0
        ineq_ok_egm = true
        for j in 1:2, a in range(0.01, 15.0, length=400)
            c = sol_egm(a, j)
            rhs = p_egm.β * R_egm * sum(p_egm.Π[j, k] *
                  sol_egm(R_egm * (a - c) + exp(p_egm.z_vals[k]), k)^(-γ_egm)
                  for k in 1:2)
            if c < a - 1e-10
                max_res_egm = max(max_res_egm, abs(c^(-γ_egm) - rhs) / c^(-γ_egm))
            else
                ineq_ok_egm = ineq_ok_egm && (c^(-γ_egm) >= rhs - 1e-9)
            end
        end
        @test max_res_egm < 5e-3     # equality at interior choices
        @test ineq_ok_egm            # inequality where the constraint binds
    end

    @testset "stochastic returns" begin
        ζ_egm = exp.(0.16 .* randn(MersenneTwister(0), 200))
        p_egm = EGMProblem(; β=0.9,
                           u_prime=c -> c^(-1.5), u_prime_inv=v -> v^(-1 / 1.5),
                           Γ=(s, z, ϵ) -> ϵ * s + exp(z),
                           Γ_s=(s, z, ϵ) -> ϵ,
                           s_grid=range(0.0, 30.0, length=200),
                           mc=MarkovChain([0.9 0.1; 0.1 0.9], [0.0, 0.5]),
                           ϵ_vals=ζ_egm)
        sol_egm = solve(p_egm)
        @test sol_egm.converged
        @test all(issorted(sol_egm.a[:, j]) for j in 1:2)
    end

    # Largest relative Euler residual at interior choices, and whether the
    # Euler inequality holds where the constraint binds
    function euler_check_egm(p, sol, xs)
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

    @testset "borrowing limit" begin
        for b_egm in (1.0, 3.0)
            mc_b_egm = MarkovChain([0.6 0.4; 0.05 0.95], log.([0.5, 1.0]))
            p_egm = ifp_problem(; γ=1.0, mc=mc_b_egm, b=b_egm,
                                s_grid=power_grid(16.0, 200; s_min=-b_egm))
            sol_egm = solve(p_egm)
            @test sol_egm.converged
            @test all(sol_egm.a[1, :] .== -b_egm)
            # spend down to the limit where the constraint binds
            x_low_egm = (-b_egm + sol_egm.a[2, 1]) / 2
            @test sol_egm(x_low_egm, 1) ≈ x_low_egm + b_egm
            worst_egm, ineq_ok_egm = euler_check_egm(
                p_egm, sol_egm, range(-b_egm + 0.05, 15.0, length=300))
            @test worst_egm < 1e-3
            @test ineq_ok_egm
        end
    end

    @testset "two IID shocks given as tuples" begin
        rng_egm = MersenneTwister(1234)
        ζ_egm = exp.(0.16 .* randn(rng_egm, 10))   # return shock
        η_egm = randn(rng_egm, 10)                 # income shock
        pairs_egm = vec([(ζ, η) for ζ in ζ_egm, η in η_egm])
        p_egm = EGMProblem(; β=0.9,
                           u_prime=c -> c^(-1.5), u_prime_inv=v -> v^(-1 / 1.5),
                           Γ=(s, z, ϵ) -> ϵ[1] * s + exp(0.5 + 0.2 * ϵ[2] + z),
                           Γ_s=(s, z, ϵ) -> ϵ[1],
                           s_grid=power_grid(40.0, 200),
                           mc=MarkovChain([0.9 0.1; 0.1 0.9], [0.0, 0.5]),
                           ϵ_vals=pairs_egm)
        sol_egm = solve(p_egm)
        @test sol_egm.converged
        worst_egm, ineq_ok_egm = euler_check_egm(
            p_egm, sol_egm, range(0.1, 30.0, length=300))
        @test worst_egm < 1e-3
        @test ineq_ok_egm
    end

end  # @testset
