# MuJoCo Minimal RECAP

一个低维、确定性 BC → 自主 rollout → Value → Advantage → 条件 Gaussian 策略研究原型。Gaussian 策略是简化实现，并非原始 Flow Matching VLA 复现。

## Linux 服务器安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
# 若需要指定 CUDA wheel，请按服务器驱动选择对应 PyTorch 安装源。
pytest -q
recap all --config configs/smoke.yaml --run runs/smoke
```

纯状态训练无需 OpenGL。离屏图像渲染可设置 `MUJOCO_GL=egl`（需要服务器 EGL 驱动），或使用系统支持的 OSMesa。`ObstacleReach2D(config, render_mode='human')` 使用 MuJoCo viewer，`rgb_array` 返回图像。

## 完整实验

```bash
recap generate --run runs/seed42
recap train --model bc --run runs/seed42
recap rollout --run runs/seed42
recap train --model sft --run runs/seed42
recap train --model value --run runs/seed42
recap train --model recap --run runs/seed42
recap evaluate --run runs/seed42
recap plot --run runs/seed42
# 中断训练恢复（相同配置）
recap train --model recap --run runs/seed42 --resume
```

`recap all` 顺序完成上述步骤。默认 200 条训练专家轨迹、400 条自主轨迹，另有独立 validation 和六个测试集合。生成成本可能较高，先运行 smoke。生成器达到尝试预算仍不足时会报错，不能把规划失败解释为场景不可解。已有场景 manifest 不允许覆盖；不同实验配置使用不同目录。多训练种子修改 YAML seed 并使用各自目录；如需跨种子严格共享测试集，应在实验前显式准备共享 manifest，目前没有跨运行聚合命令。

## 模块与数据

- `recap/env.py`：2×0.5m 胶囊连杆，位置执行器，20Hz 控制，逐物理步全连杆解析碰撞检测。胶囊端帽覆盖关节及末端；不包含机器人自碰撞任务。
- `recap/planning.py`：两分支 IK、有限关节网格 A*、插值验证、捷径平滑、实际控制器专家执行与场景验收。
- `recap/models.py`：6×128 SiLU MLP、201-bin 插值标签、标量 Value 消融、有界 Gaussian NLL。
- `recap/cli.py`：数据、训练、断点恢复、固定场景配对评测和静态绘图。

每条 NPZ 独立保存 episode，禁止跨 episode 拼接 TD 窗口；场景参数保存在 `scenes.json`，通过 scene_id 关联。`data/<split>/expert` 中的测试专家只用于可解性证明。train 和 validation 的自主数据仅在各自场景中采集。所有评测策略共享同一测试列表，报告保存逐场景结果、Wilson 成功率区间和 RECAP 相对 BC 增益。

输入为明确列举的 **11D**，原规格中的 12D 不额外虚构状态。条件策略使用三态 one-hot（低/高/null），因此网络输入 14D。速度除以 10，其余状态使用米、sin/cos；目标与障碍物在同一世界坐标系。动作单位 rad，默认 ±0.05。

200 步任务最差回报为 -349，因此统一使用 horizon+149 归一化，避免原文针对 100 步的 249 尺度造成越界。成功终止奖励 0，其余普通步 -1，碰撞、约束和超时终止 -150。gamma=1，所有终止（含超时）取消 bootstrap。默认 TD horizon=10。quantile 筛选只表示排名，不能称为正优势。

RECAP 使用 null NLL + alpha×高优势 NLL，另以 0.1 权重学习低条件，条件 dropout 可配置。Gaussian 标签在边界做 1e-6 数值裁剪，以避免 atanh 无穷；评测使用 tanh(mean)。BC/SFT 使用以动作范围归一化的 MSE。SFT 从 BC checkpoint 初始化，保留专家并加入成功 rollout；RECAP 从随机 Gaussian 网络训练。Value 使用专家和混合 rollout，validation 按场景独立；模型保存最佳 validation checkpoint 和最近训练状态。

## 验证状态与限制

代码已在当前开发环境执行 Python 语法编译检查；遵照服务器运行要求，没有安装本地研究依赖，也没有执行 MuJoCo/PyTorch 测试或训练。服务器 `pytest` 和 smoke 必须通过后再启动完整实验；不能将此仓库视为已经通过动力学验收或达到任何成功率。

生成器按几何与执行指标分类 easy/medium/hard，并按配置的配额验收。专家采用逐路点限幅跟踪与捷径平滑，没有高级时间参数化。规划插值为有限分辨率验证，不是数学意义的连续碰撞证明。当前图表包括 loss 和单轨迹 Value/Return/Advantage、机械臂路径；尚未实现难度匹配子集、跨种子聚合、优势 oracle/标签准确率、排序相关性报告、成功率训练曲线、多轮 RECAP、专家纠偏和仿真状态分支接口。上述功能不应报告为已完成。没有为达到预期成功率修改标签或筛选测试结果。

目标和障碍物中心采用覆盖四个象限的环带采样。角度在 [-π, π) 内均匀采样，半径为 sqrt(U(r_min², r_max²))，使候选点按面积均匀分布。默认障碍物中心距原点 0.20～0.80 m，各集合共用该分布。目标距离和障碍物中心距离可在 YAML 的 `sampling` 中调整。

| 数据集 | 默认场景数 | 目标距离（m） | 障碍物半径（m） | 用途 |
|---|---:|---|---|---|
| train | 200 | 0.40～0.70 | 0.05～0.09 | 训练 |
| validation | 30 | 0.40～0.70 | 0.05～0.09 | 独立验证与 checkpoint 选择 |
| id | 50 | 0.40～0.70 | 0.05～0.09 | 同分布测试 |
| inner_goal_ood | 50 | 0.20～0.35 | 0.05～0.09 | 近基座目标泛化 |
| outer_goal_ood | 50 | 0.75～0.90 | 0.05～0.09 | 远目标泛化 |
| size_ood | 50 | 0.40～0.70 | 0.10～0.13 | 障碍物尺寸泛化 |
| inner_composition_ood | 50 | 0.20～0.35 | 0.10～0.13 | 内圈目标与大障碍物组合 |
| outer_composition_ood | 50 | 0.75～0.90 | 0.10～0.13 | 外圈目标与大障碍物组合 |

测试集合用于评测及专家可解性验证。最终保留的场景经过初始碰撞、规划和专家执行成功筛选，因此实际角度分布需要检查 `scenes.json`。新实验使用新目录生成数据，例如 `recap generate --run runs/annulus42`。旧配置中的 `goal_ood`、`composition_ood` 保留原有角度采样语义，旧运行可继续按其 manifest 评测。

各数据划分独立执行顶层 `difficulty_ratio` 配比，默认 easy 为 0.60、medium 为 0.30、hard 为 0.10，三者之和为 1。整数配额使用最大余数法，相同余数按 easy、medium、hard 顺序分配。例如 200 条对应 120/60/20，50 条对应 30/15/5，4 条对应 3/1/0，2 条对应 1/1/0。

默认 `planner.proposal: guided` 按剩余配额优先构造 hard、medium、easy 候选。medium 的障碍物放在关节直连运动时连杆扫过的区域。hard 还会从目标 IK 解附近采样起始关节姿态，以提高阻断直连后产生较大绕行比的机会。候选先检查起点、目标姿态间隙及障碍物中心距离，再进入 A* 和专家执行。目标仍从各集合对应的环带采样，障碍物位置和 hard 起点使用上述条件分布。实际难度继续由规划绕行比和执行间隙决定。`difficulty.json` 中的 `proposed` 记录各目标难度的候选次数，`generation.json` 中的 `construction_failed` 记录几何构造失败次数。`planner.proposal: random` 可使用独立随机候选。省略该字段的旧配置沿用 random。

easy 表示初始关节姿态到所选 IK 解的关节空间直连满足安全间隙。其余场景中，规划绕行比达到 `planner.hard_ratio` 且实际专家轨迹最小间隙低于 `planner.hard_clearance` 时标为 hard，其余标为 medium。配额已满的难度停止收录，分类指标继续按原定义计算。达到 `planner.max_attempts` 后仍有缺额时生成失败，错误信息列出各难度缺额。`difficulty.json` 保存各集合的目标配额、已验收数量和尝试次数，失败时也会写入当前统计。旧配置省略 `difficulty_ratio` 时沿用无配额采样。
