# 强化学习作业与实验

学习《动手学强化学习》过程中的作业代码与实验记录，使用 Python 和 uv 管理环境。

## 当前进展

- **作业一｜多臂老虎机**：实现 ε-greedy、衰减 ε-greedy 和汤普森采样，对比累计奖励与遗憾，图片见 `homework1-multi-armed-bandit/figures/`。
- **作业二｜马尔可夫决策过程**：用首次访问蒙特卡洛估计快递分拨中心 MDP 的状态价值。
- **作业三｜动态规划**：在带障碍和打滑的 4×4 网格上实现值迭代与策略迭代，对比最优策略、收敛过程和计算开销，并完成 γ 与打滑概率消融实验，图片见 `homework3-dynamic-programming/figures/`。

## 目录结构

```
.
├── homework1-multi-armed-bandit/         # 多臂老虎机
├── homework2-markov-decision-process/    # 马尔可夫过程
├── homework3-dynamic-programming/        # 动态规划
└── final-project-smart-logistics/        # 大作业：基于强化学习的智慧物流决策系统
```

## 运行

```bash
uv sync --locked
uv run homework1-multi-armed-bandit/homework1.py
uv run homework3-dynamic-programming/homework3.py
```

学习资料与题目来源：[《动手学强化学习》](https://github.com/boyu-ai/Hands-on-RL)。
