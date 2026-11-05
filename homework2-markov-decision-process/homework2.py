import numpy as np
import matplotlib.pyplot as plt

# 使用中文字体，并兼容不同系统上的常见字体。
plt.rcParams['font.sans-serif'] = [
    'Arial Unicode MS', 'PingFang SC', 'Heiti TC',
    'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False

# ==================== MDP 参数定义 ====================
# 状态：0-到达, 1-分拣, 2-运输, 3-派送, 4-签收(终止), 5-异常(终止)
states = ['到达', '分拣', '运输', '派送', '签收', '异常']
non_terminal_states = [0, 1, 2, 3]
terminal_states = [4, 5]

# 动作：0-正常处理, 1-加急处理
actions = [0, 1]
action_names = {0: '正常', 1: '加急'}

# 折扣因子
gamma = 0.9

# 即时成本（每步执行动作的成本）
cost = {0: -1, 1: -3}   # 正常成本 -1，加急成本 -3

# 终止奖励
reward_terminal = {4: 10, 5: -10}  # 签收 +10，异常 -10

# 转移概率：P[s][a] = [(prob, next_state), ...]
# 注意：奖励在采样时根据 next_state 和动作计算
P = {
    0: {  # 到达
        0: [(0.9, 1), (0.1, 5)],   # 正常：0.9到分拣，0.1到异常
        1: [(0.95, 1), (0.05, 5)]  # 加急：0.95到分拣，0.05到异常
    },
    1: {  # 分拣
        0: [(0.8, 2), (0.1, 3), (0.1, 5)],
        1: [(0.9, 2), (0.05, 3), (0.05, 5)]
    },
    2: {  # 运输
        0: [(0.85, 3), (0.15, 5)],
        1: [(0.95, 3), (0.05, 5)]
    },
    3: {  # 派送
        0: [(0.9, 4), (0.1, 5)],
        1: [(0.98, 4), (0.02, 5)]
    }
}

# 给定策略 π：π[s][a] 表示在状态 s 选择动作 a 的概率
pi = {
    0: {0: 0.8, 1: 0.2},
    1: {0: 0.6, 1: 0.4},
    2: {0: 0.7, 1: 0.3},
    3: {0: 0.9, 1: 0.1}
}

print("MDP 参数与策略已定义。")

# ==================== 奖励函数 ====================
def get_reward(s, a, next_s):
    """
    计算从状态 s 采取动作 a 转移到 next_s 时获得的即时奖励。
    奖励 = 动作成本 + 终止奖励（如果 next_s 是终止状态）
    """
    r = cost[a]
    if next_s in terminal_states:
        r += reward_terminal[next_s]
    return r

# ==================== 单次轨迹采样 ====================
def generate_episode(start_state, pi, P, gamma):
    """
    从 start_state 出发，根据策略 pi 和环境 P 生成一条完整轨迹。
    返回：states_list, actions_list, rewards_list
    """
    s = start_state
    states_list = [s]
    actions_list = []
    rewards_list = []
    
    while s not in terminal_states:
        # TODO: 根据 pi[s] 随机选择动作 a
        a = np.random.choice(list(pi[s].keys()), p=list(pi[s].values()))
        
        # TODO: 根据 P[s][a] 随机选择下一个状态 next_s
        # 提示：probs = [p for p, _ in P[s][a]]; next_states = [ns for _, ns in P[s][a]]
        probs = [p for p, _ in P[s][a]]
        next_states = [ns for _, ns in P[s][a]]
        next_s = np.random.choice(next_states, p=probs)
        
        # 计算奖励
        r = get_reward(s, a, next_s)
        
        actions_list.append(a)
        rewards_list.append(r)
        states_list.append(next_s)
        
        s = next_s
    
    return states_list, actions_list, rewards_list

# ==================== 蒙特卡洛估计状态价值 ====================
def monte_carlo_value_estimation(start_state, pi, P, gamma, N=10000):
    """
    使用首次访问蒙特卡洛方法估计从 start_state 出发的状态价值 V(start_state)。
    对每条轨迹，计算从 start_state 开始的折扣回报，然后取平均。
    """
    returns = []
    for _ in range(N):
        states_list, actions_list, rewards_list = generate_episode(start_state, pi, P, gamma)
        # TODO: 计算折扣回报 G = sum_{k=0}^{len(rewards)-1} gamma^k * rewards[k]
        G = 0.0
        # 请补全循环计算 G
        # 提示：for k, r in enumerate(rewards_list): G += (gamma ** k) * r
        # 然后将 G 加入 returns
        for k, r in enumerate(rewards_list):
            G += (gamma ** k) * r
        returns.append(G)
    
    # TODO: 返回 returns 的均值作为 V(start_state) 的估计
    # 提示：np.mean(returns)
    return np.mean(returns)

# ==================== 对所有非终止状态进行估计 ====================
N = 10000
V_est = {}

for s in non_terminal_states:
    # TODO: 调用 monte_carlo_value_estimation 估计 V(s)
    # V_est[s] = ...
    V_est[s] = monte_carlo_value_estimation(start_state=s, pi=pi, P=P, gamma=gamma, N=N)

# 打印结果
print("蒙特卡洛估计的状态价值 V(s)：")
for s in non_terminal_states:
    if s in V_est:
        print(f"{states[s]}: {V_est[s]:.4f}")

# ==================== 绘制从“到达”出发的回报分布直方图 ====================
def get_returns_for_state(start_state, pi, P, gamma, N=10000):
    returns = []
    for _ in range(N):
        states_list, actions_list, rewards_list = generate_episode(start_state, pi, P, gamma)
        G = sum((gamma ** k) * r for k, r in enumerate(rewards_list))
        returns.append(G)
    return returns

# 获取从“到达”出发的回报样本
returns_arrival = get_returns_for_state(0, pi, P, gamma, N=5000)

plt.figure(figsize=(10, 5))
plt.hist(returns_arrival, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('折扣回报 G')
plt.ylabel('频数')
plt.title('从“到达”状态出发的折扣回报分布（策略 π 下）')
plt.grid(True, alpha=0.3)
plt.show()
