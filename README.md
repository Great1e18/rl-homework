# 强化学习作业与实验

学习《动手学强化学习》过程中的作业代码与实验记录，使用 Python 和 uv 管理环境。

## 当前进展

- 已实现多臂老虎机实验：ε-greedy、衰减 ε-greedy 和汤普森采样。
- 每种算法运行 200 轮，每轮 1000 步，使用固定的一组随机种子复现实验。
- 已绘制平均累计奖励、平均累计遗憾，以及各窗口的总访问次数和成功率，图片保存在 `homework1-multi-armed-bandit/figures/`。
- 后续继续补充马尔可夫过程、动态规划等作业。

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
```

学习资料与题目来源：[《动手学强化学习》](https://github.com/boyu-ai/Hands-on-RL)。
