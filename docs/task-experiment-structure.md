# Zbot 任务与实验分层

Zbot 强化学习代码采用两层组织方式。

## Task

Task 定义机器人要完成的行为目标和 MDP 语义，包括：

- 观测内容及其含义；
- 动作空间；
- 脚步相位和目标频率；
- 奖励项的行为目标；
- 终止条件；
- 机器人和传感器配置。

当前任务：

| Task | 任务目标 |
| --- | --- |
| `Mjlab-Zbot-6dof-Periodic-Stepping` | 按目标频率左右交替原地踏步 |
| `Mjlab-Zbot-6dof-Walking` | 周期踏步基础上增加前进速度奖励 |

## Experiment

Experiment 定义同一个 Task 的训练条件，包括：

- 是否加入外力/力矩扰动；
- 域随机化范围；
- 频率采样范围；
- PPO 超参数；
- 环境数量和训练迭代数；
- checkpoint 和日志目录。

当前实验：

| Task | Experiment | 实验目录 |
| --- | --- | --- |
| Periodic Stepping | baseline | `zbot_periodic_stepping` |
| Walking | baseline | `zbot_walking` |
| Walking | disturbed base wrench | `zbot_disturbed_walking` |

抗扰动实验默认让 75% 的并行环境参与扰动，25% 完全保持无扰动；参与扰动的环境中，每个事件独立从 `force_range` 和 `torque_range` 逐分量均匀采样，因此强度仍连续分布，包含接近零的小扰动和较强扰动。

抗扰动实验只能增加扰动或随机化配置，必须复用 Walking 的观测、动作、奖励和坐标定义。不要为了增加一个实验复制一套任务实现。实验 task ID 是 MJLab 注册层面的兼容入口，内部语义仍属于同一个 Task。

未来建议使用如下入口形式：

```text
python train.py <task> --experiment <experiment> ...
```

在完成统一实验解析前，现有命令仍可使用注册 task ID 启动对应实验。
