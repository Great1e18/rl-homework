import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
from pathlib import Path
from contextlib import contextmanager

# 中文字体配置（matplotlib 默认的 DejaVu Sans 没有中文字形，会画成方框）
plt.rcParams['font.sans-serif'] = [
    'PingFang HK', 'PingFang SC', 'Hiragino Sans GB',
    'Arial Unicode MS', 'Heiti TC', 'Microsoft YaHei', 'SimHei',
    'DejaVu Sans',  # 兜底：箭头 ↑↓←→ 由它提供
]
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示，避免显示成方块

# ==================== 网格与参数定义 ====================
ROWS, COLS = 4, 4
START = (0, 0)
GOAL = (3, 3)
OBSTACLES = {(1, 1), (2, 1)}

# 动作定义：0-上, 1-下, 2-左, 3-右
ACTIONS = [0, 1, 2, 3]
ACTION_NAMES = {0: '上', 1: '下', 2: '左', 3: '右'}

# 各动作的意图移动方向（行偏移, 列偏移）
MOVE = {
    0: (-1, 0),  # 上
    1: (1, 0),   # 下
    2: (0, -1),  # 左
    3: (0, 1)    # 右
}

# 垂直方向（用于打滑偏移）：每个动作对应的两个垂直动作
PERPENDICULAR = {
    0: [2, 3],  # 上 -> 左, 右
    1: [2, 3],  # 下 -> 左, 右
    2: [0, 1],  # 左 -> 上, 下
    3: [0, 1]   # 右 -> 上, 下
}

GAMMA = 0.9
STEP_REWARD = -1
GOAL_REWARD = 10
# 打滑概率：按意图方向移动的概率为 1 - SLIP_PROB，剩下 SLIP_PROB 平分为左右偏
SLIP_PROB = 0.2

# 图片输出目录：固定在脚本旁边的 figures/，与从哪里运行无关
FIG_DIR = Path(__file__).resolve().parent / "figures"


@contextmanager
def mdp_config(gamma=None, slip_prob=None):
    """临时修改 GAMMA / SLIP_PROB，用于消融实验；退出时自动还原。"""
    global GAMMA, SLIP_PROB
    saved = (GAMMA, SLIP_PROB)
    if gamma is not None:
        GAMMA = gamma
    if slip_prob is not None:
        SLIP_PROB = slip_prob
    try:
        yield
    finally:
        GAMMA, SLIP_PROB = saved

# 构建状态空间：所有非障碍格子
def build_states():
    states = []
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) not in OBSTACLES:
                states.append((r, c))
    return states

STATES = build_states()
print(f"状态数：{len(STATES)}")
print(f"状态列表：{STATES}")

# ==================== 转移函数 ====================
def get_transition(state, action):
    """
    返回在 state 执行 action 后的转移列表：
    [(prob, next_state, reward), ...]

    规则：
    - 80% 按意图方向移动
    - 10% 向垂直方向偏移（左偏）
    - 10% 向垂直方向偏移（右偏）
    - 若移动后超出网格或进入障碍物，则停留在原地
    - 若 next_state 是目标，奖励为 GOAL_REWARD；否则为 STEP_REWARD
    """
    transitions = []
    r, c = state

    # TODO: 实现转移逻辑
    # 步骤提示：
    # 1. 获取意图方向 MOVE[action]，记为 (dr_int, dc_int)
    # 2. 获取垂直方向 PERPENDICULAR[action]，得到两个动作 a1, a2
    # 3. 构造三种移动：
    #    moves = [(0.8, MOVE[action]), (0.1, MOVE[a1]), (0.1, MOVE[a2])]
    # 4. 对每种移动 (prob, (dr, dc))：
    #    - 计算新位置 nr, nc = r + dr, c + dc
    #    - 若越界或 (nr, nc) 在 OBSTACLES 中，则 nr, nc = r, c（原地）
    #    - next_state = (nr, nc)
    #    - reward = GOAL_REWARD if next_state == GOAL else STEP_REWARD
    #    - 将 (prob, next_state, reward) 加入列表
    # 5. 合并相同 next_state 的概率（可选但推荐）

    dr_int, dc_int = MOVE[action]
    a1, a2 = PERPENDICULAR[action]
    p_slip = SLIP_PROB / 2  # 两个垂直方向各分一半
    moves = [(1.0 - SLIP_PROB, MOVE[action]), (p_slip, MOVE[a1]), (p_slip, MOVE[a2])]
    merged = {}
    for prob, (dr, dc) in moves:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < ROWS and 0 <= nc < COLS) or (nr, nc) in OBSTACLES:
            nr, nc = r, c
        next_state = (nr, nc)
        merged[next_state] = merged.get(next_state, 0) + prob
    for next_state, prob in merged.items():
        reward = GOAL_REWARD if next_state == GOAL else STEP_REWARD
        transitions.append((prob, next_state, reward))
    
    return transitions

# ==================== 值迭代 ====================
def value_iteration(theta=1e-8, max_iter=1000):
    """
    值迭代算法。
    返回：
        V            : 状态价值字典 {(r, c): value}
        policy       : 最优策略字典 {(r, c): action}
        n_iter       : 迭代次数（= 全状态扫描次数）
        elapsed      : 耗时（秒）
        delta_history: 每次扫描的 max|ΔV|，用于画收敛曲线
    """
    V = {s: 0.0 for s in STATES}
    policy = {s: 0 for s in STATES}
    delta_history = []

    start_time = time.time()
    n_iter = 0

    for it in range(max_iter):
        delta = 0.0
        new_V = V.copy()

        for s in STATES:
            if s == GOAL:
                continue  # 目标状态价值固定为 0

            # TODO: 计算每个动作的期望价值 Q(s, a)
            # 对每个动作 a：
            #   q = Σ_{s'} P(s'|s,a) * [R(s,a,s') + γ * V(s')]
            # 其中 V(GOAL) = 0
            # 取最大的 q 作为 new_V[s]，并记录最优动作到 policy[s]
            # 更新 delta = max(delta, |new_V[s] - V[s]|)
            
            qsa_list = []
            for a in range(4):
                q = 0
                transitions = get_transition(s, a)
                for (prob, next_state, reward) in transitions:
                    q += prob * (reward + GAMMA * V[next_state])
                qsa_list.append(q)
            policy[s] = np.argmax(qsa_list)
            new_V[s] = qsa_list[policy[s]]
            delta = max(delta, abs(new_V[s] - V[s]))

        V = new_V
        n_iter = it + 1
        delta_history.append(delta)
        if delta < theta:
            break

    elapsed = time.time() - start_time
    return V, policy, n_iter, elapsed, delta_history

# ==================== 策略评估 ====================
def policy_evaluation(policy, theta=1e-8, max_iter=1000):
    """给定策略，迭代评估其状态价值。返回 (V, n_sweeps)"""
    V = {s: 0.0 for s in STATES}

    for it in range(max_iter):
        delta = 0.0
        new_V = V.copy()

        for s in STATES:
            if s == GOAL:
                continue

            # TODO: 根据策略 policy[s] 计算 V(s)
            # v = Σ_{s'} P(s'|s, π(s)) * [R(s,a,s') + γ * V(s')]
            # 更新 new_V[s]，并更新 delta
            
            a = policy[s]
            v = 0
            transitions = get_transition(s, a)
            for prob, next_state, reward in transitions:
                v += prob * (reward + GAMMA * V[next_state])
            new_V[s] = v
            delta = max(delta, abs(new_V[s] - V[s]))

        V = new_V
        n_sweeps = it + 1
        if delta < theta:
            break

    return V, n_sweeps

# ==================== 策略改进 ====================
def policy_improvement(V):
    """根据 V 进行策略改进，返回新策略"""
    new_policy = {}

    for s in STATES:
        if s == GOAL:
            continue

        # TODO: 对每个动作 a 计算 Q(s, a)，选择 Q 最大的动作
        # new_policy[s] = argmax_a Q(s, a)
        
        qsa_list = []
        for a in range(4):
            q = 0
            transitions = get_transition(s, a)
            for (prob, next_state, reward) in transitions:
                q += prob * (reward + GAMMA * V[next_state])
            qsa_list.append(q)
        new_policy[s] = np.argmax(qsa_list)

    return new_policy

# ==================== 策略迭代 ====================
def policy_iteration(theta=1e-8, max_iter=1000):
    """
    策略迭代算法。
    返回：
        V          : 状态价值字典
        policy     : 最优策略字典
        n_iter     : 外层迭代次数
        elapsed    : 耗时（秒）
        sweep_hist : 每次外层迭代中策略评估的扫描次数
    """
    # 初始化策略：所有非终止状态都选动作 0（上）
    policy = {s: 0 for s in STATES if s != GOAL}

    start_time = time.time()
    n_iter = 0
    sweep_hist = []

    for it in range(max_iter):
        # TODO: 实现策略迭代主循环
        # 1. 策略评估：V = policy_evaluation(policy)
        # 2. 策略改进：new_policy = policy_improvement(V)
        # 3. 检查策略是否稳定：所有状态的动作是否与旧策略相同
        # 4. 若稳定则跳出；否则 policy = new_policy
        V, n_sweeps = policy_evaluation(policy, theta)
        sweep_hist.append(n_sweeps)
        new_policy = policy_improvement(V)
        n_iter += 1
        if policy == new_policy:
            break
        else:
            policy = new_policy

    elapsed = time.time() - start_time
    return V, policy, n_iter, elapsed, sweep_hist

# ==================== 绘图工具 ====================
ARROWS = {0: '↑', 1: '↓', 2: '←', 3: '→'}


def value_grid(V):
    """把 V 字典转成 ROWS×COLS 矩阵，障碍物位置填 nan"""
    grid = np.full((ROWS, COLS), np.nan)
    for (r, c), v in V.items():
        grid[r, c] = v
    return grid


def near_obstacle_states():
    """与障碍物上下左右相邻的非障碍状态（用于讨论分析题 1）"""
    result = set()
    for (r, c) in STATES:
        for dr, dc in MOVE.values():
            if (r + dr, c + dc) in OBSTACLES:
                result.add((r, c))
    return result


NEAR_OBSTACLE = near_obstacle_states()


def full_policy(policy):
    """补全缺失的终止状态（策略迭代返回的 policy 不包含 GOAL）"""
    return {s: policy.get(s, 0) for s in STATES}


def draw_value(ax, V, title, vmin=None, vmax=None):
    """在 ax 上画价值函数热力图（蓝虚线框 = 与障碍物相邻的状态）"""
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad('0.6')
    im = ax.imshow(value_grid(V), cmap=cmap, interpolation='nearest',
                   vmin=vmin, vmax=vmax)
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) in NEAR_OBSTACLE:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                       ec='#1f77b4', lw=2.0, ls='--'))
            if (r, c) in OBSTACLES:
                ax.text(c, r, '障碍', ha='center', va='center',
                        fontsize=11, color='white', fontweight='bold')
            elif (r, c) == GOAL:
                ax.text(c, r, '终点\n0.00', ha='center', va='center', fontsize=9)
            else:
                ax.text(c, r, f'{V[(r, c)]:.2f}', ha='center', va='center', fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.set_xlabel('列')
    ax.set_ylabel('行')
    return im


def draw_policy(ax, policy, title, ref_policy=None):
    """在 ax 上画策略箭头图；与 ref_policy 不同的动作用红圈标出"""
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) in NEAR_OBSTACLE:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                       ec='#1f77b4', lw=2.0, ls='--'))
            if (r, c) in OBSTACLES:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, color='0.6'))
                ax.text(c, r, '障碍', ha='center', va='center',
                        fontsize=11, color='white', fontweight='bold')
            elif (r, c) == GOAL:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, color='#b7f0b1'))
                ax.text(c, r, '终点', ha='center', va='center', fontsize=10)
            else:
                a = policy[(r, c)]
                changed = ref_policy is not None and ref_policy.get((r, c)) != a
                ax.text(c, r, ARROWS[a], ha='center', va='center', fontsize=20,
                        color='crimson' if changed else 'black',
                        bbox=dict(boxstyle='circle,pad=0.18', fc='#ffe0e0',
                                  ec='crimson', lw=1.5) if changed else None)
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('列')
    ax.set_ylabel('行')
    ax.grid(True, color='0.85')


def solve_by_value_iteration(gamma=None, slip_prob=None, theta=1e-8):
    with mdp_config(gamma, slip_prob):
        V, policy, n_iter, elapsed, hist = value_iteration(theta)
    return V, full_policy(policy), n_iter, elapsed, hist


def solve_by_policy_iteration(gamma=None, slip_prob=None, theta=1e-8):
    with mdp_config(gamma, slip_prob):
        V, policy, n_iter, elapsed, hist = policy_iteration(theta)
    return V, full_policy(policy), n_iter, elapsed, hist


# ==================== 图 1：两种方法的 V 与 π ====================
def fig1_methods_overview(res_vi, res_pi):
    V_vi, pol_vi, iter_vi, time_vi, _ = res_vi
    V_pi, pol_pi, iter_pi, time_pi, _ = res_pi

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), layout='constrained')
    vmin = min(min(V_vi.values()), min(V_pi.values()))
    vmax = max(max(V_vi.values()), max(V_pi.values()))

    im = draw_value(axes[0, 0], V_vi,
                    f'值迭代：V(s)（{iter_vi} 次全状态扫描，{time_vi * 1e3:.1f} ms）',
                    vmin, vmax)
    draw_value(axes[0, 1], V_pi,
               f'策略迭代：V(s)（{iter_pi} 次外层迭代，{time_pi * 1e3:.1f} ms）',
               vmin, vmax)
    fig.colorbar(im, ax=axes[0, :].tolist(), fraction=0.03, pad=0.02, label='V(s)')

    draw_policy(axes[1, 0], pol_vi, '值迭代：最优策略 π*(s)')
    draw_policy(axes[1, 1], pol_pi, '策略迭代：最优策略 π*(s)（红圈 = 与值迭代不同）',
                ref_policy=pol_vi)

    same = all(pol_vi[s] == pol_pi[s] for s in STATES)
    fig.suptitle(f'值迭代 vs 策略迭代（γ={GAMMA}，打滑概率={SLIP_PROB}）\n'
                 f'两法最优策略{"完全一致" if same else "不一致"}'
                 f'　|　蓝虚线框 = 与障碍物相邻的状态', fontsize=12)
    return fig


# ==================== 图 2：收敛过程 ====================
def fig2_convergence(res_vi, res_pi, theta=1e-8):
    _, _, _, _, delta_hist = res_vi
    _, _, n_iter_pi, _, sweep_hist = res_pi

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')

    ax = axes[0]
    ax.semilogy(np.arange(1, len(delta_hist) + 1), delta_hist, 'o-',
                ms=4, lw=1.6, color='#1f77b4')
    ax.axhline(theta, color='crimson', ls='--', lw=1.2, label=f'收敛阈值 θ={theta:g}')
    ax.set_xlabel('全状态扫描次数（迭代轮数）')
    ax.set_ylabel('max|ΔV|（对数轴）')
    ax.set_title('值迭代：每轮价值变化量单调衰减')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    x = np.arange(1, n_iter_pi + 1)
    bars = ax.bar(x, sweep_hist, color='#ff7f0e')
    for b, v in zip(bars, sweep_hist):
        ax.text(b.get_x() + b.get_width() / 2, v, f'{v}', ha='center',
                va='bottom', fontsize=9)
    ax.set_xlabel('外层迭代轮次')
    ax.set_xticks(x)
    ax.set_ylabel('该轮策略评估的扫描次数')
    ax.set_title(f'策略迭代：每轮策略评估开销（共 {sum(sweep_hist)} 次扫描）')
    ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('两种方法的收敛过程', fontsize=12)
    return fig


# ==================== 图 3：计算量对比 ====================
def fig3_cost_comparison(res_vi, res_pi):
    _, _, iter_vi, time_vi, _ = res_vi
    _, _, iter_pi, time_pi, sweep_hist = res_pi

    items = [
        ('迭代次数\n（值迭代=扫描次数，策略迭代=外层轮次）',
         [iter_vi, iter_pi], '次数'),
        ('总状态扫描次数\n（每次扫描都要对 4 个动作算 Q）',
         [iter_vi, sum(sweep_hist)], '次数'),
        ('实际耗时', [time_vi * 1e3, time_pi * 1e3], '毫秒'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout='constrained')
    for ax, (title, vals, unit) in zip(axes, items):
        bars = ax.bar(['值迭代', '策略迭代'], vals,
                      color=['#1f77b4', '#ff7f0e'], width=0.55)
        for b, v in zip(bars, vals):
            label = f'{v:.1f}' if unit == '毫秒' else f'{v}'
            ax.text(b.get_x() + b.get_width() / 2, v, label,
                    ha='center', va='bottom', fontsize=10)
        ax.set_ylabel(unit)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(vals) * 1.18 if max(vals) > 0 else 1)
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('计算量对比（策略迭代单轮更贵，但外层轮次少得多）', fontsize=12)
    return fig


# ==================== 图 4：折扣因子消融（分析题 2）====================
def fig4_gamma_ablation(default_policy):
    gammas = [0.5, GAMMA, 0.99]
    results = {g: solve_by_value_iteration(gamma=g) for g in gammas}

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.6), layout='constrained')
    all_v = [v for g in gammas for v in results[g][0].values()]
    vmin, vmax = min(all_v), max(all_v)

    for j, g in enumerate(gammas):
        V, pol, n_iter, _, _ = results[g]
        tag = '（默认）' if g == GAMMA else ''
        im = draw_value(axes[0, j], V, f'γ={g}{tag}：V(s)', vmin, vmax)
        changed = sum(1 for s in STATES if s != GOAL and pol[s] != default_policy[s])
        draw_policy(axes[1, j], pol,
                    f'γ={g}{tag}：π*(s)　与默认策略差 {changed} 个状态',
                    ref_policy=default_policy)
    fig.colorbar(im, ax=axes[0, :].tolist(), fraction=0.025, pad=0.02, label='V(s)')

    fig.suptitle('分析题 2：折扣因子 γ 对价值与策略的影响'
                 f'（红圈 = 与 γ={GAMMA} 的策略不同；障碍/终点格是灰色/绿色）',
                 fontsize=12)
    return fig, results


# ==================== 图 5：打滑概率消融（分析题 4）====================
def fig5_slip_ablation(default_policy):
    slips = [0.0, SLIP_PROB]
    results = {p: solve_by_value_iteration(slip_prob=p) for p in slips}

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout='constrained')
    all_v = [v for p in slips for v in results[p][0].values()]
    vmin, vmax = min(all_v), max(all_v)

    for j, p in enumerate(slips):
        V, pol, n_iter, _, _ = results[p]
        label = '打滑概率=0（完全确定）' if p == 0 else f'打滑概率={p}（随机）'
        im = draw_value(axes[0, j], V, f'{label}：V(s)', vmin, vmax)
        changed = sum(1 for s in STATES if s != GOAL and pol[s] != default_policy[s])
        draw_policy(axes[1, j], pol,
                    f'{label}：π*(s)　与默认策略差 {changed} 个状态',
                    ref_policy=default_policy)
    fig.colorbar(im, ax=axes[0, :].tolist(), fraction=0.03, pad=0.02, label='V(s)')

    fig.suptitle('分析题 4：打滑概率对最优策略的影响（红圈 = 与默认策略不同）',
                 fontsize=12)
    return fig, results


# ==================== 主流程 ====================
def main():
    print("=" * 60)
    print(f"默认参数：γ={GAMMA}, 打滑概率={SLIP_PROB}, 每步奖励={STEP_REWARD}, "
          f"目标奖励={GOAL_REWARD}, 障碍={sorted(OBSTACLES)}")
    print("=" * 60)

    print("\n--- 值迭代 ---")
    res_vi = solve_by_value_iteration()
    V_vi, pol_vi, iter_vi, time_vi, _ = res_vi
    print(f"迭代次数：{iter_vi}")
    print(f"耗时：{time_vi:.4f} 秒")
    print("V*(s) 矩阵（行=行号，列=列号，障碍显示 None）：")
    print(np.round(value_grid(V_vi), 3))

    print("\n--- 策略迭代 ---")
    res_pi = solve_by_policy_iteration()
    V_pi, pol_pi, iter_pi, time_pi, sweep_hist = res_pi
    print(f"外层迭代次数：{iter_pi}")
    print(f"每轮策略评估扫描次数：{sweep_hist}（合计 {sum(sweep_hist)}）")
    print(f"耗时：{time_pi:.4f} 秒")

    print("\n--- 对比 ---")
    print(f"值迭代  : 迭代 {iter_vi} 次，耗时 {time_vi:.4f} 秒")
    print(f"策略迭代: 迭代 {iter_pi} 次，耗时 {time_pi:.4f} 秒")
    same = all(pol_vi[s] == pol_pi[s] for s in STATES)
    print(f"两法最优策略是否一致：{same}")
    max_diff = max(abs(V_vi[s] - V_pi[s]) for s in STATES)
    print(f"两法价值函数最大差异：{max_diff:.2e}")

    print("\n--- 分析题 1 的线索：与障碍物相邻的状态 ---")
    for s in sorted(NEAR_OBSTACLE):
        print(f"  {s}: V={V_vi[s]:8.3f}, π={ACTION_NAMES[pol_vi[s]]}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figures = [
        ("fig1_methods_overview.png", fig1_methods_overview(res_vi, res_pi)),
        ("fig2_convergence.png", fig2_convergence(res_vi, res_pi)),
        ("fig3_cost_comparison.png", fig3_cost_comparison(res_vi, res_pi)),
    ]
    fig4, gamma_results = fig4_gamma_ablation(pol_vi)
    figures.append(("fig4_gamma_ablation.png", fig4))
    fig5, slip_results = fig5_slip_ablation(pol_vi)
    figures.append(("fig5_slip_ablation.png", fig5))

    print("\n--- 消融实验摘要（分析题 2 / 4）---")
    for g, (V, pol, n_iter, _, _) in gamma_results.items():
        diff = [(s, ACTION_NAMES[pol_vi[s]], ACTION_NAMES[pol[s]])
                for s in sorted(STATES) if s != GOAL and pol[s] != pol_vi[s]]
        print(f"γ={g}: V(0,0)={V[START]:.3f}, 与默认策略不同的状态 {len(diff)} 个 {diff}")
    for p, (V, pol, n_iter, _, _) in slip_results.items():
        diff = [(s, ACTION_NAMES[pol_vi[s]], ACTION_NAMES[pol[s]])
                for s in sorted(STATES) if s != GOAL and pol[s] != pol_vi[s]]
        print(f"打滑={p}: V(0,0)={V[START]:.3f}, 与默认策略不同的状态 {len(diff)} 个 {diff}")

    for name, fig in figures:
        fig.savefig(FIG_DIR / name, dpi=180, bbox_inches='tight')
        print(f"已保存：{FIG_DIR / name}")

    plt.show()


if __name__ == "__main__":
    main()
