# 强化学习作业

《动手学强化学习》学习过程中的作业代码与实验记录，使用 Python 和 uv 管理环境。

## 作业

- **作业一｜多臂老虎机**：对比 ε-greedy、衰减 ε-greedy 和汤普森采样。
- **作业二｜马尔可夫决策过程**：用首次访问蒙特卡洛估计快递分拨中心 MDP 的状态价值。
- **作业三｜动态规划**：在 4×4 网格上实现值迭代与策略迭代，并完成 γ 与打滑概率消融实验。
- **大作业｜智慧物流**：基于强化学习的智慧物流决策系统。

## 运行

```bash
uv sync --locked
uv run homework1-multi-armed-bandit/homework1.py
uv run homework2-markov-decision-process/homework2.py
uv run homework3-dynamic-programming/homework3.py
```

学习资料来自[《动手学强化学习》](https://github.com/boyu-ai/Hands-on-RL)，作业题目来自[王洪苹](https://teacher.bupt.edu.cn/wanghongping/zh_CN/index.htm)老师的课程。
