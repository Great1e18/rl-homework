import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class PickingBandit:
    def __init__(self, K: int, probs: list, seed: int):
        self.probs = np.array(probs)
        self.best_id = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_id]
        self.K = K
        self.rng = np.random.default_rng(seed)
    def step(self, k: int) -> int:
        if self.rng.random() < self.probs[k]:
            return 1
        else:
            return 0

class Solver:
    def __init__(self, bandit: PickingBandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K, dtype=int)  # 每个窗口的访问次数
        self.success_counts = np.zeros(self.bandit.K, dtype=int)  # 每个窗口的成功次数
        self.reward = 0
        self.regret = 0
        self.rewards = []
        self.regrets = []
        self.actions = []
        self.last_r = 0
    def update_regret(self, k: int):
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)
    def run_one_step(self):
        # 接口声明：规定每个子类都要提供 run_one_step
        raise NotImplementedError
    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.success_counts[k] += self.last_r
            self.actions.append(k)
            self.reward += self.last_r
            self.rewards.append(self.reward)
            self.update_regret(k)

class EpsilonGreedy(Solver):
    """ epsilon贪婪算法,继承Solver类 """
    def __init__(self, bandit: PickingBandit, epsilon: float = 0.01, init_prob: float = 1.0):
        super().__init__(bandit) # 调用 Solver 的 __init__，并把 bandit 交给他
        self.epsilon = epsilon
        self.estimates = np.array([init_prob]*bandit.K)
        self.rng = bandit.rng
    def run_one_step(self) -> int:
        if self.rng.random() < self.epsilon:
            k = self.rng.integers(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        self.last_r = self.bandit.step(k)
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (self.last_r - self.estimates[k])
        return k

class DecayingEpsilonGreedy(Solver):
    """ epsilon值随时间衰减的epsilon-贪婪算法,继承Solver类 """
    def __init__(self, bandit, init_prob=1.0):
        super().__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0
        self.rng = bandit.rng
    def run_one_step(self) -> int:
        self.total_count += 1
        epsilon = 1 / self.total_count
        if self.rng.random() < epsilon:
            k = self.rng.integers(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        self.last_r = self.bandit.step(k)
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (self.last_r - self.estimates[k])
        return k

class ThompsonSampling(Solver):
    """ 汤普森采样算法,继承Solver类 """
    def __init__(self, bandit):
        super().__init__(bandit)
        self.rng = bandit.rng
        self._a = np.ones(self.bandit.K)
        self._b = np.ones(self.bandit.K)
    def run_one_step(self):
        samples = self.rng.beta(self._a, self._b)  # 按照Beta分布采样一组奖励样本
        k = np.argmax(samples)
        self.last_r = self.bandit.step(k)
        self._a[k] += self.last_r
        self._b[k] += (1 - self.last_r)
        return k

def plot_result(solver: Solver, solver_name: str):
    # 固定图编号，让不同算法的曲线画在同一张图上。
    time_steps = range(1, len(solver.rewards) + 1)
    plt.figure(1)
    plt.plot(time_steps, solver.rewards, label=solver_name)
    plt.xlabel("Time steps")
    plt.ylabel("Mean cumulative reward")
    plt.legend()
    plt.figure(2)
    plt.plot(time_steps, solver.regrets, label=solver_name)
    plt.xlabel("Time steps")
    plt.ylabel("Mean cumulative regret")
    plt.legend()

    # 每种算法单独开一张图，左右分别显示访问次数和成功率。
    window_ids = range(1, solver.bandit.K + 1)
    success_rates = []
    rate_labels = []
    for k in range(solver.bandit.K):
        if solver.counts[k] > 0:
            rate = solver.success_counts[k] / solver.counts[k] * 100
            success_rates.append(rate)
            rate_labels.append(f"{rate:.2f}%")
        else:
            # 用 N/A 标明未访问，区别于访问过但成功率为 0%。
            success_rates.append(0)
            rate_labels.append("N/A")

    plt.figure(f"{solver_name} - window statistics", figsize=(10, 4))
    plt.suptitle(solver_name)

    plt.subplot(1, 2, 1)  # 1 行、2 列，选择第 1 个位置。
    bars = plt.bar(window_ids, solver.counts)
    plt.bar_label(bars, padding=3)
    plt.xticks(window_ids)
    plt.xlabel("Window")
    plt.ylabel("Total visits")
    plt.margins(y=0.15)

    plt.subplot(1, 2, 2)  # 选择第 2 个位置。
    bars = plt.bar(window_ids, success_rates)
    plt.bar_label(bars, labels=rate_labels, padding=3)
    plt.xticks(window_ids)
    plt.xlabel("Window")
    plt.ylabel("Success rate (%)")
    plt.ylim(0, 115)  # 给柱子上方的百分比文字留出空间。
    plt.tight_layout()


def print_result(solver: Solver, solver_name: str):
    print(f"\n算法：{solver_name}（所有实验合计）")
    for k in range(solver.bandit.K):
        count = solver.counts[k]
        success_count = solver.success_counts[k]
        if count > 0:
            success_rate = f"{success_count / count:.2%}"
        else:
            success_rate = "未访问"
        print(f"窗口 {k + 1}：总访问次数={count}，成功次数={success_count}，成功率={success_rate}")


def main():
    num_experiments = 200
    num_steps = 1000
    base_seed = 42  # 固定整组实验；修改它就能换一组实验。
    probs = [0.3, 0.5, 0.7, 0.4, 0.6]

    # 键是画图时显示的名称，值是算法类
    methods = {
        "eps-greedy": EpsilonGreedy,  # 使用默认参数 epsilon=0.01
        "decay-eps": DecayingEpsilonGreedy,
        "thompson-sampling": ThompsonSampling,
    }

    for name, solver_class in methods.items():
        # 这个 Solver 只用来汇总数据，不运行算法，也不继承上一轮的学习结果。
        summary_bandit = PickingBandit(K=5, probs=probs, seed=base_seed)
        summary_solver = Solver(summary_bandit)
        summary_solver.rewards = np.zeros(num_steps)
        summary_solver.regrets = np.zeros(num_steps)

        for exp_id in range(num_experiments):
            # 200 轮分别用 42、43、……、241，每个算法使用同一组种子。
            seed = base_seed + exp_id
            bandit = PickingBandit(K=5, probs=probs, seed=seed)
            solver = solver_class(bandit)  # 每轮重新学习。
            solver.run(num_steps=num_steps)

            summary_solver.rewards += solver.rewards
            summary_solver.regrets += solver.regrets
            summary_solver.counts += solver.counts
            summary_solver.success_counts += solver.success_counts

        # 曲线取 200 轮的平均；窗口访问次数和成功次数保留合计值。
        summary_solver.rewards /= num_experiments
        summary_solver.regrets /= num_experiments
        summary_solver.reward = summary_solver.rewards[-1]
        summary_solver.regret = summary_solver.regrets[-1]
        print_result(summary_solver, name)
        plot_result(summary_solver, name)

    # 保存到 Python 文件旁边的 figures 文件夹，与从哪里运行程序无关。
    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(exist_ok=True)
    plt.figure(1).savefig(output_dir / "mean_cumulative_reward.png", dpi=200, bbox_inches="tight")
    plt.figure(2).savefig(output_dir / "mean_cumulative_regret.png", dpi=200, bbox_inches="tight")
    for name in methods:
        plt.figure(f"{name} - window statistics").savefig(
            output_dir / f"{name}_window_statistics.png", dpi=200, bbox_inches="tight"
        )
    print(f"\n图片已保存到：{output_dir}")

    # 先保存再显示，关闭图窗不会影响已保存的图片。
    plt.show()


if __name__ == "__main__":
    main()
