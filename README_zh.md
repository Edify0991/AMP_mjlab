# AMP_mjlab

[English README](README.md)

部署集成代码位于 [ccrpRepo/wbc_fsm](https://github.com/ccrpRepo/wbc_fsm) 项目中的 `MJAmp State`。

基于 mjlab + rsl_rl 的 G1 AMP 运动控制项目。

本项目的核心特点是：

- 使用同一个 policy 同时学习 locomotion（走/跑）与 recovery（跌倒恢复）
- 通过 AMP 判别器约束动作风格与运动先验
- 在训练与导出链路中保持一致，支持直接导出 ONNX policy

## 核心思路

传统做法常把“走跑策略”和“恢复策略”分开训练并做切换；本项目将两类能力放入一个策略中统一学习。

实现要点：

- 运动数据分组：
	- Walk/Run 数据目录：`src/assets/motions/g1/amp/WalkandRun`
	- Recovery 数据目录：`src/assets/motions/g1/amp/Recovery`
- 延迟重置机制（Delayed Termination）：
	- 一部分环境在触发终止后不立即 reset，而是给定恢复窗口
	- 该子集环境优先从 Recovery 片段采样 reset 状态
- 统一 AMP 训练：
	- 单一 actor-critic + 单一 AMP discriminator
	- 在同一训练过程中学习速度跟踪、抗扰动与恢复能力

这样可以减少策略切换带来的状态不连续问题，得到更一致的行为。

## 环境要求

- Linux
- Python 3.11（建议）
- 已可用的 MuJoCo / GPU 驱动环境

## 快速开始

### 1. 安装仓库

```bash
conda activate mjlab
cd AMP_mjlab
python -m pip install -e .
```

### 2. 应用 mjlab 补丁（可选）

如果不打这个补丁，则需要在代码中去掉 `history_ordering` 配置。

补丁作用说明：

- 增加了历史观测的展开方式选项，可选择按时间维(`time`)或按观测项(`term`)展开。
- mjlab 默认仅支持按 `term` 展开。

补丁文件：

- `mjlab_patch/mjlab/managers/observation_manager.py`

示例覆盖命令：

```bash
cp mjlab_patch/mjlab/managers/observation_manager.py \
	/home/crp/miniconda3/envs/mjlab/lib/python3.11/site-packages/mjlab/managers/observation_manager.py
```

### 3. 查看可用任务

```bash
python scripts/list_envs.py --keyword AMP
```

主要任务：

- `Unitree-G1-AMP-Rough`
- `Unitree-G1-AMP-Flat`

## 训练


```bash
python scripts/train.py Unitree-G1-AMP-Flat --env.scene.num-envs=4096
```


日志默认在：

- `logs/rsl_rl/g1_amp_locomotion/<time_stamp_run>/`

### 训练 Jingchu01 AMP

Jingchu01 任务名：

```bash
python scripts/list_envs.py --keyword Jingchu01
```

当前注册了两个任务：

- `Jingchu01-AMP-Flat`：平地训练，建议先从这里开始调通。
- `Jingchu01-AMP-Rough`：粗糙地形训练，适合在平地策略稳定后继续训练或从头训练。

训练前请确认运动数据已经转换到：

- `src/assets/motions/jingchu01/amp/WalkandRun`
- `src/assets/motions/jingchu01/amp/Recovery`

单卡平地训练：

```bash
WARP_CACHE_PATH=/tmp/warp_cache python scripts/train.py Jingchu01-AMP-Flat \
	--env.scene.num-envs=4096
```

先做小规模冒烟测试：

```bash
WARP_CACHE_PATH=/tmp/warp_cache python scripts/train.py Jingchu01-AMP-Flat \
	--env.scene.num-envs=64 \
	--agent.max-iterations=10 \
	--agent.amp-num-preload-transitions=5000
```

粗糙地形训练：

```bash
WARP_CACHE_PATH=/tmp/warp_cache python scripts/train.py Jingchu01-AMP-Rough \
	--env.scene.num-envs=4096
```

多 GPU 训练：

```bash
WARP_CACHE_PATH=/tmp/warp_cache python scripts/train.py Jingchu01-AMP-Flat \
	--gpu-ids all \
	--env.scene.num-envs=8192
```

如果只想使用指定 GPU，例如 0 和 1：

```bash
WARP_CACHE_PATH=/tmp/warp_cache python scripts/train.py Jingchu01-AMP-Flat \
	--gpu-ids "[0,1]" \
	--env.scene.num-envs=8192
```

断点续训：

```bash
WARP_CACHE_PATH=/tmp/warp_cache python scripts/train.py Jingchu01-AMP-Flat \
	--env.scene.num-envs=4096 \
	--agent.resume True \
	--agent.load-run <run_dir> \
	--agent.load-checkpoint model_<iter>.pt
```

Jingchu01 日志默认在：

- `logs/rsl_rl/jingchu01_amp_locomotion/<time_stamp_run>/`

查看 TensorBoard：

```bash
tensorboard --logdir logs/rsl_rl/jingchu01_amp_locomotion
```

## 训练曲线说明（重要）

- 在约 `2w` 轮（约 20k iterations）附近，策略通常会突然学会“跌倒后恢复”行为。
- 对应地，`logs` 中多个指标会出现明显突变（阶跃式变化），这是正常现象，不一定是训练异常。

![训练日志突变示例](logs.png)

## 评估与可视化

使用已训练权重回放：

```bash
python scripts/play.py Unitree-G1-AMP-Rough \
	--checkpoint-file logs/rsl_rl/g1_amp_locomotion/<run_dir>/model_<iter>.pt 
```

说明：训练与回放阶段都支持 ONNX 导出（默认开启）。

Jingchu01 策略回放：

```bash
WARP_CACHE_PATH=/tmp/warp_cache python scripts/play.py Jingchu01-AMP-Flat \
	--checkpoint-file logs/rsl_rl/jingchu01_amp_locomotion/<run_dir>/model_<iter>.pt \
	--num-envs 1 \
	--device cuda:0 \
	--viewer native
```

无显示器或远程环境可以使用 Viser：

```bash
WARP_CACHE_PATH=/tmp/warp_cache python scripts/play.py Jingchu01-AMP-Flat \
	--checkpoint-file logs/rsl_rl/jingchu01_amp_locomotion/<run_dir>/model_<iter>.pt \
	--num-envs 1 \
	--device cuda:0 \
	--viewer viser
```

回放时默认导出 ONNX，路径位于：

- `logs/rsl_rl/jingchu01_amp_locomotion/<run_dir>/export/`

## 运动数据准备

仓库提供 CSV 到 NPZ 的转换脚本：

```bash
python scripts/csv_to_npz.py --help
```

推荐目录组织：

- 原始 CSV：`motion_data_csv/amp`
- 转换后 NPZ：`src/assets/motions/g1/amp/WalkandRun` 与 `src/assets/motions/g1/amp/Recovery`

只要上述目录中存在可用 NPZ，训练配置会自动加载。

### Jingchu01 PKL 到 NPZ

Jingchu01 当前使用 MJCF 模型：

- 训练模型文件：`/home/user/wmd/jingchu01/JC01-7DOF-URDF/JC01-URDF-18所/jingchu01.xml`
- 场景检查文件：`/home/user/wmd/jingchu01/JC01-7DOF-URDF/JC01-URDF-18所/scene_jingchu01.xml`
- 关节顺序文件：`src/assets/robots/jingchu01/gmr_pkl_joint_names_28.txt`
- 训练环境入口：`src.tasks.amp_loco.config.jingchu01.env_cfgs:jingchu01_amp_flat_env_cfg`

完整转换示例：

```bash
WARP_CACHE_PATH=/tmp/warp_cache python scripts/csv_to_npz.py \
	--input-file motion_data_csv/amp/jingchu01/walk1_subject5.pkl \
	--output-dir src/assets/motions/jingchu01/amp/WalkandRun \
	--output-name walk1_subject5.npz \
	--output-fps 50 \
	--device cuda:0 \
	--input-quat-format xyzw \
	--joint-names-file src/assets/robots/jingchu01/gmr_pkl_joint_names_28.txt \
	--env-cfg-entry-point src.tasks.amp_loco.config.jingchu01.env_cfgs:jingchu01_amp_flat_env_cfg
```

如果只是快速检查前几帧，可以加 `--line-range "(1,120)"`。

### 转换时渲染

离屏渲染并保存 mp4：

```bash
WARP_CACHE_PATH=/tmp/warp_cache python scripts/csv_to_npz.py \
	--input-file motion_data_csv/amp/jingchu01/walk1_subject5.pkl \
	--output-dir src/assets/motions/jingchu01/amp/WalkandRun \
	--output-name walk1_subject5.npz \
	--output-fps 50 \
	--device cuda:0 \
	--input-quat-format xyzw \
	--joint-names-file src/assets/robots/jingchu01/gmr_pkl_joint_names_28.txt \
	--env-cfg-entry-point src.tasks.amp_loco.config.jingchu01.env_cfgs:jingchu01_amp_flat_env_cfg \
	--render True \
	--render-backend offscreen \
	--video-output src/assets/motions/jingchu01/amp/WalkandRun/walk1_subject5.mp4
```

打开 MuJoCo 窗口实时回放：

```bash
WARP_CACHE_PATH=/tmp/warp_cache python scripts/csv_to_npz.py \
	--input-file motion_data_csv/amp/jingchu01/walk1_subject5.pkl \
	--output-dir src/assets/motions/jingchu01/amp/WalkandRun \
	--output-name walk1_subject5.npz \
	--output-fps 50 \
	--device cuda:0 \
	--input-quat-format xyzw \
	--joint-names-file src/assets/robots/jingchu01/gmr_pkl_joint_names_28.txt \
	--env-cfg-entry-point src.tasks.amp_loco.config.jingchu01.env_cfgs:jingchu01_amp_flat_env_cfg \
	--render True \
	--render-backend window \
	--window-realtime True \
	--window-realtime-scale 1.0
```

## 目录说明

- `src/tasks/amp_loco`：AMP locomotion/recovery 任务实现
- `src/tasks/amp_loco/config/g1`：G1 任务注册、环境与 RL 配置
- `src/tasks/amp_loco/mdp`：奖励、观测、事件、终止逻辑
- `scripts/train.py`：训练入口
- `scripts/play.py`：回放入口
- `scripts/csv_to_npz.py`：动作数据转换工具
- `mjlab_patch`：依赖的 mjlab 本地补丁

## 项目亮点总结

- 单一策略统一覆盖走跑与跌倒恢复
- AMP + 速度任务联合优化，兼顾风格与任务性能
- 延迟重置与 recovery 采样机制，显式强化恢复能力
- 训练到部署链路完整，支持 ONNX 导出

## 致谢

- 感谢 [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) 项目的开源工作与启发。
- 感谢 [Open-X-Humanoid/TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab)，本项目在 rsl_rl 的 AMP 部分参考了该实现。

WARP_CACHE_PATH=/tmp/warp_cache python scripts/train.py Jingchu01-AMP-Flat \
  --gpu-ids "[0,1]" \
  --env.scene.num-envs=4096
