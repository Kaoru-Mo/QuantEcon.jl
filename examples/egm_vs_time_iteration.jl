# EGM と，求根による time iteration の比較
#
# 同じ所得変動問題  a′ = R (a - c) + y(z′)  を 2 通りに解き，
# 政策関数の差と実行時間を表示する．
#
# 実行（QuantEcon.jl のフォルダで）：
#     julia --project=. examples/egm_vs_time_iteration.jl

using QuantEcon
using Printf

const R_cmp, β_cmp, γ_cmp = 1.01, 0.96, 1.5
const mc_cmp = MarkovChain([0.6 0.4; 0.05 0.95], [log(0.5), log(2.0)])

# --- EGM ---------------------------------------------------------------
p = ifp_problem(; R=R_cmp, β=β_cmp, γ=γ_cmp, mc=mc_cmp,
                s_grid=power_grid(16.0, 200))

# --- 求根による time iteration（固定した資産の格子の上で解く） ----------
function K_root(c::Matrix{Float64}, grid::Vector{Float64}, Π, y_vals)
    n_a, n_z = size(c)
    out = zeros(n_a, n_z)
    for j in 1:n_z, i in 2:n_a          # i = 1 は a = 0 で c = 0
        a = grid[i]
        # オイラー方程式の残差  u′(t) - βR E[u′(σ(R(a - t) + y′, z′))]
        h = t -> begin
            acc = 0.0
            for k in 1:n_z
                c_next = QuantEcon._egm_interp(grid, view(c, :, k),
                                               R_cmp * (a - t) + y_vals[k])
                acc += Π[j, k] * max(c_next, 1e-300)^(-γ_cmp)
            end
            t^(-γ_cmp) - β_cmp * R_cmp * acc
        end
        # 全部使っても左辺 ≥ 右辺なら，借入制約が効いていて c = a
        out[i, j] = h(a) >= 0 ? a : brent(h, 1e-10, a)
    end
    return out
end

function solve_root(grid, Π, y_vals; tol=1e-5, max_iter=1000)
    c = repeat(grid, 1, length(y_vals))       # 全部消費する政策から始める
    for it in 1:max_iter
        c_new = K_root(c, grid, Π, y_vals)
        err = maximum(abs.(c_new .- c))
        c = c_new
        err < tol && return c, it
    end
    return c, max_iter
end

grid = power_grid(20.0, 200)
Π, y_vals = mc_cmp.p, exp.(mc_cmp.state_values)

# 1 回目はコンパイルを含むので，2 回目の時間を測る
solve(p); solve_root(grid, Π, y_vals)
t0 = time()
sol = solve(p)
t_egm = time() - t0
t0 = time()
c_root, it_root = solve_root(grid, Π, y_vals)
t_root = time() - t0

# --- 比較 --------------------------------------------------------------
max_diff = 0.0
for j in 1:2, a in range(0.05, 15.0, length=300)
    global max_diff = max(max_diff,
        abs(sol(a, j) - QuantEcon._egm_interp(grid, view(c_root, :, j), a)))
end

@printf("EGM            : %3d iterations, %.4f seconds\n", sol.num_iter, t_egm)
@printf("root finding   : %3d iterations, %.4f seconds\n", it_root, t_root)
@printf("speed ratio    : %.1f\n", t_root / t_egm)
@printf("max difference in consumption: %.2e\n", max_diff)
@printf("constraint binds for a ≤ %.3f (bad state), %.3f (good state)\n",
        sol.c[2, 1], sol.c[2, 2])
