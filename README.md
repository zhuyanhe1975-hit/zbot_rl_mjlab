# Zbot RL MJLab

Zbot 六自由度双足 walking 项目。项目只保留 Zbot 任务源代码、MJCF/网格资产，以及训练、回放和评估入口；MJLab、MuJoCo Warp、PyTorch 和 RSL-RL 直接复用官方 MJLab 虚拟环境。

默认环境：`/home/yhzhu/AI/mjlab/.venv`。也可以通过 `MJLAB_ROOT` 或 `ZBOT_PYTHON` 覆盖。

```bash
cd /home/yhzhu/myWorks_vips/zbot_rl_mjlab
./install.sh                 # 只检查官方环境，不创建虚拟环境
./smoke.sh --num-envs 4
./visualize.sh                 # standing pose, joint axes, base and locomotion frames
./train.sh --num-envs 1024 --iterations 15000
./run.sh 32                  # 自动加载最新 checkpoint
./run.sh 32 --checkpoint logs/rsl_rl/zbot_g1_velocity/<run>/model_1499.pt
./evaluate.sh --steps 600
```

`train.sh` 使用 TensorBoard 日志，不连接 WandB。训练日志和 checkpoint 保存在 `logs/rsl_rl/zbot_g1_velocity/`。

默认使用 `assets/zbot_6dof/robot_geometry_inertia.xml`，保留 USD 刚体质量，按网格几何计算惯量。速度跟踪使用重力参考的 locomotion frame，线速度和角速度使用 EMA 滤波。项目仍处于训练调试阶段；奖励增长不代表已验证速度跟踪成功。可用 `scripts/measure_commands.py` 实测固定指令下的速度。

主要文件：

- `src/zbot_rl_mjlab/`：任务、奖励、动作、PPO runner 和 CLI。
- `assets/zbot_6dof/`：Zbot MJCF 与网格资产。
- `scripts/python.sh`：调用官方 MJLab Python 环境。
- `train.sh`、`run.sh`、`evaluate.sh`、`smoke.sh`：训练、回放、评估和 smoke test。
