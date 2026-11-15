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

`generate --resume` 检查当前运行目录，跳过已完成的数据划分，并补齐缺少或未完成的集合。`--splits` 可指定集合，例如仅补 composition：

```bash
recap generate --run runs/seed42 --config runs/seed42/config.json --resume --splits inner_composition_ood outer_composition_ood
```

每完成一个集合立即原子保存 `scenes.json`，并在新专家 NPZ 中嵌入该场景的 `scene_json`。旧版生成中断后若仅保存了 NPZ，首次续生成会从保存的观测和控制指令恢复场景，再核对专家动作序列与难度配额。恢复的场景标记 `recovered_from_expert`，目标和障碍参数精度来自原始 float32 观测。后续运行直接读取恢复后的清单。已有完整集合的 NPZ 保持原始内容。选择部分集合时，场景种子仍按完整数据划分顺序计算。

BC 训练完成后可独立评测成功率：

```bash
recap evaluate --model bc --run runs/seed42 --config runs/seed42/config.json
# 先评测 ID 集合
recap evaluate --model bc --run runs/seed42 --config runs/seed42/config.json --splits id
```

单策略评测输出每个数据集、每种场景难度和所选场景总体的成功率，报告写入 `evaluation.bc.json`，轨迹写入 `data/<split>/bc`。已有 `value.pt` 时同时计算 Value 误差。省略 `--model` 时评测 BC、SFT、RECAP 并写入 `evaluation.json`，同时保存各自独立报告。

评测某个更新步的 checkpoint：

```bash
recap evaluate --model bc --run runs/seed42 --checkpoint bc.step00002000.pt
# 只评测 ID
recap evaluate --model bc --run runs/seed42 --checkpoint bc.step00002000.pt --splits id
```

指定 `--checkpoint` 时只在终端报告结果，不保存 JSON 或轨迹，不修改 `bc.pt` 和已有评测结果。不指定时仍评测默认的 `bc.pt` 并记录。文件名会依次在运行目录及其 `checkpoints` 子目录查找，也可传入相对运行目录的路径或绝对路径。使用 `--checkpoint` 必须明确 `--model`，同样支持 SFT、RECAP。`evaluate` 省略 `--config` 时优先读取该运行保存的 `config.json`；快照与运行的环境配置不一致时会拒绝评测，防止误用新旧控制设置。

环境观测、保存的 `observation` / `next_observation` 和 BC、SFT、Value、RECAP 全部使用 **9D 观测**，彻底移除两维关节速度。RECAP 追加三维条件后网络输入为 12D。旧的 `train.policy_use_velocity` 设置已移除。MuJoCo 仍正常模拟物理速度与惯性，速度不暴露给模型或保存到观测中。由于相同姿态可能具有不同物理速度，9D 观测不再包含完整动力学状态，Value 预测也受此信息限制。

默认训练使用 **AdamW、学习率 0.001、weight_decay=0.0001、8000 个优化器更新步**。`train.max_steps` 非空时优先于 `epochs`；一个 step 指一个 batch 的参数更新。默认每 `train.validation_steps: 500` 步验证一次，第 1 步和最后一步也验证。BC 对全部 validation 场景执行闭环评测，优先按成功率、成功率相同时按 validation MSE 保存 `bc.pt`；SFT、Value 和 RECAP 按各自 validation loss 选模。测试集合不参与选择。

`train.checkpoint_steps: 500` 控制额外快照间隔，写入 `checkpoints/bc.step00000500.pt` 等文件；验证或快照时更新 `bc.last.pt`。快照包含优化器、随机状态和当前数据乱序位置，`--resume` 可从最近保存点精确续训。`bc.history.json` 保存 step、epoch、loss、验证成功率及各难度指标。将 `max_steps` 设为 `null` 或省略时仍使用 epoch 训练；此模式由 `validation_rollout_interval` 控制 BC 闭环验证间隔，默认 10 轮，设为 0 时仅按 MSE 选模。smoke 使用 2 轮训练。

旧 11D NPZ 在训练读取时会自动移除速度列；要使磁盘上的数据也变为 9D，可复制转换到新目录，无需重新生成。迁移覆盖 `data` 下的所有 NPZ，保留动作、奖励、姿态、场景和逐帧对应关系；复制原运行配置和场景/生成清单，移除过时的速度开关。原目录保持不变，新目录不复制旧权重、训练历史或评测报告。旧的 11D 权重及断点需要重新训练，加载或续训时会明确报错。

```bash
recap migrate --run runs/seed42 --output-run runs/seed42-9d --train-config configs/default.yaml
recap train --model bc --run runs/seed42-9d --config runs/seed42-9d/config.json
recap evaluate --model bc --run runs/seed42-9d --config runs/seed42-9d/config.json --splits id
# 用同一份数据创建学习率更小的独立实验
recap migrate --run runs/seed42 --output-run runs/seed42-9d-lr1e4 --train-config configs/default.yaml --lr 0.0001
recap train --model bc --run runs/seed42-9d-lr1e4 --config runs/seed42-9d-lr1e4/config.json
```

`migrate` 读取源目录的 `config.json`；`--train-config` 只合并指定 YAML 的 train 段，保留源数据的环境、场景、种子和配比。`--lr` 可再覆盖学习率。不提供覆盖选项时沿用源训练配置。目标目录必须尚不存在；旧运行保存的配置不会自动变成新默认值。9D checkpoint 可用 `--resume` 续训，续训保留原优化器设置。后续实验应使用新 BC 重新采集 rollout，再训练 SFT、Value 和 RECAP。

固定数据、seed、AdamW 和权重衰减，对比三种学习率，各训练 10000 步并每 500 步验证及保存快照：

```bash
python scripts/sweep_bc.py --run runs/seed42 --output runs/bc-sweep-10k --lrs 0.0003 0.001 0.003 --steps 10000 --interval 500
```

脚本生成各组独立运行目录、`sweep.json` 和 `learning_curves.png`。依据 validation 闭环成功率选出配置与 checkpoint，再仅对选定模型执行 ID 测试；单次 seed 的最佳结果只代表本次比较范围。脚本默认使用可用 GPU，`--device cpu --threads 1` 可指定 CPU。

本地 seed42 的三组 10000 步对照已完成：lr=0.001 在第 8000 步取得最佳 validation 成功率 73.3%，选定后 ID 成功率为 66.0%。完整设置、曲线和难度指标见 [实验记录](docs/experiments/bc_adamw_10k_seed42.md)。

后续 [20000 步对照](docs/experiments/bc_adamw_20k_seed42.md) 固定 batch size=256，比较 lr=0.001 与 0.0005：两组最佳 validation 均为 73.3%，按较低 MSE 选中 lr=0.0005 的第 14000 步 checkpoint，ID 为 34/50（68%）。训练更久未持续改善闭环成功率，仍应保留中间最优模型。

`recap all` 顺序完成上述步骤。默认 200 条训练专家轨迹、400 条自主轨迹，另有独立 validation 和六个测试集合。生成成本可能较高，先运行 smoke。生成器达到尝试预算仍不足时会报错，不能把规划失败解释为场景不可解。已有场景 manifest 不允许覆盖；不同实验配置使用不同目录。多训练种子修改 YAML seed 并使用各自目录；如需跨种子严格共享测试集，应在实验前显式准备共享 manifest，目前没有跨运行聚合命令。

## 平滑专家与近目标纠偏

新配置使用 `planner.expert_controller: spline`：对整条关节路点路径联合求解分段七次最小 snap 样条，约束内部路点的位置、速度、加速度和 jerk 连续，起终点速度、加速度和 jerk 为零。内部路点可带非零速度通过。默认每关节参考速度上限为 0.35 rad/s、加速度上限为 2 rad/s²、jerk 上限为 10 rad/s³；按三种导数约束给每段分配时间，再统一拉长时长满足限值。这些是参考轨迹限值，实际运动仍由 MuJoCo 积分。

整条参考曲线转换为 Bézier 控制点后递归检查关节限位和两根连杆的完整间隙，并用控制点包络与连杆位移上界给出保守安全判定。弯曲路径不安全时逐级缩小内部路点导数，必要时退回路点处零速度、零加速度、零 jerk 的 C³ 路径；无法通过检查则拒收。实际执行仍逐物理步检查碰撞。NPZ 的 `expert_spline_blend` 记录导数缩放系数，0 表示使用停稳通过的安全回退；`expert_spline_duration` 记录参考时长。旧配置仍支持 `quintic`，缺少控制器字段时沿用 waypoint。

参考时长加成功保持时间若超过任务预算，会在执行前拒收，记录为 `spline_too_long`，避免慢速轨迹反复仿真到超时。采样范围和难度配额保持原配置；收录的专家仍必须通过实际执行验收。

默认控制频率为 **50 Hz**：物理步长仍为 0.005 s，每 4 个物理步更新一次动作。动作目标偏移上限降为 ±0.04 rad。`env.control_interpolation: c3_filter` 使用四级连续滤波，在每个物理步精确推进控制目标，并保留跨控制周期的导数状态，构成 C³ 连续的理想控制信号；MuJoCo 以 200 Hz 采样执行。滤波每级时间常数 `control_filter_tau: 0.003` s，总低频延迟约 12 ms，专家与所有学习策略共用这一执行规则。实际运动的高阶导数还受采样、动作限幅和动力学影响。

任务 horizon 与专家上限均为 **500 控制步（10 秒）**；成功保持 13 步约为 0.26 秒。TD horizon 调为 25 步，仍约为 0.5 秒。BC 的学习率 0.001、batch size 256、8000 个参数更新步保持不变。旧运行缺少插值字段时仍使用原 hold 模式；要使用新频率、动作范围和专家，需以新配置在新目录重新生成并训练。

新生成时若专家实际经过另一个 IK 解到达目标，会安全截短原路径并重新执行，再按最终路径和实际间隙计算难度。原有 easy/medium/hard 配额仍执行。需要全新数据时，用新目录直接生成：

```bash
recap generate --run runs/seed42-c3 --config configs/default.yaml
recap train --model bc --run runs/seed42-c3 --config runs/seed42-c3/config.json
recap evaluate --model bc --run runs/seed42-c3 --config runs/seed42-c3/config.json
```

BC 新增 `train.goal_anchor_samples: 16`：仅在 train 场景的安全 IK 解附近，以每关节 ±0.12 rad 扰动补充局部纠偏标签，同时加入精确目标处的零动作标签。所有样本都通过关节限位和直连避障检查；它们只参与 BC 动作监督，不伪造 episode、回报或 Value 标签。设为 0 可关闭。验证/测试不添加控制器，仍直接使用策略输出。`bc.goal_anchors.json` 记录补充样本数量。

默认成功距离从 0.03 m 放宽到 **0.035 m**，连续保持 13 步，成功结束不要求末端精确重合目标中心。评测另外报告 `strict_success_rate`（已记录轨迹中按 3 cm 连续保持同样步数的成功率）和 `tolerance_only_success_count`（仅因放宽范围而通过的数量）；前者不等于在旧判定下重新执行一次策略。`near_goal_stall_count` 独立统计范围外的静止超时：最终距离在成功阈值外两倍阈值内，且最后 20 帧相邻关节位移均小于 1e-5 rad。

新轨迹保存逐帧 `distance`、`goal_hold_steps`，以及物理子步 `control_commands`、`substep_qpos`、`executed_substeps`。`applied_command` 保留限幅后的目标命令，实际送入 MuJoCo 的目标在 `control_commands` 中；碰撞提前终止时只读取每帧前 `executed_substeps` 个子步，余下槽位为末值填充。`control_dt` 和 `physics_timestep` 记录物理时间单位，评测的 `completion_seconds` 可用于不同控制频率之间的时间比较。模型观测仍为 9D。

若需保留既有场景，可运行 `python scripts/rebuild_experts.py --run runs/seed42-8k --output runs/seed42-smooth-replay --config configs/default.yaml`。该工具在新目录重新执行专家，冻结场景几何和原难度分组，另存 `expert_replay_metrics` 与 `expert_replay.json`，不把重放后的指标冒充原始采样难度。迁移数据的 `recap migrate` 不会重新执行或平滑已有动作。

此前的 20 Hz、3 cm、分段 quintic 历史对照中，530 条平滑专家均执行成功，峰值 jerk 的中位数降低约 77%；平滑数据加近目标监督的 8000 步 BC 对照 ID 为 64%，仅在旧专家上增加近目标监督时为 72%。这些数据不属于当前 50 Hz、C³、3.5 cm 配置，详见 [历史检查记录](docs/experiments/expert_smooth_goal_stall_seed42.md)。

当前实现已通过 149 项测试、含 hard 配额的从零生成完整 smoke，以及 step/epoch 两种训练模式检查。在同一批 24 个固定场景、相同 200 Hz 物理采样和 3.5 cm 成功标准下，旧 20 Hz quintic 为 24/24 成功，新 50 Hz 慢速 C³ 为 20/24；4 个失败均为 10 秒预算内未完成。在两组都成功的 20 个场景中，每条轨迹峰值物理 jerk 的中位数为 10245→463 rad/s³，子步控制目标跳变为 0.03139→0.00287 rad，中位执行时间为 4.225→6.69 秒。该对照同时改变速度、动作上限、样条和控制频率，不能将效果单独归因于某一项，也不代表 BC 策略成功率。详情见 [验证数据](docs/experiments/c3_50hz_verification.json) 和 [对照图](docs/experiments/c3_50hz_comparison.png)。

## 模块与数据

- `recap/env.py`：2×0.5m 胶囊连杆，位置执行器，默认 50Hz 控制、200Hz 物理步，逐物理步全连杆解析碰撞检测。胶囊端帽覆盖关节及末端；不包含机器人自碰撞任务。
- `recap/planning.py`：两分支 IK、有限关节网格 A*、插值验证、捷径平滑、实际控制器专家执行与场景验收。
- `recap/models.py`：6×128 SiLU MLP、201-bin 插值标签、标量 Value 消融、有界 Gaussian NLL。
- `recap/controllers.py`、`recap/splines.py`：全局 C³ 样条、整曲线安全检查，兼容旧路点及 quintic 专家。
- `recap/control_filter.py`：专家和学习策略共用的 C³ 子步控制滤波。
- `recap/goal_supervision.py`：只用于 BC 训练的安全近目标纠偏样本。
- `recap/cli.py`：数据、训练、断点恢复、固定场景配对评测和静态绘图。

每条 NPZ 独立保存 episode，禁止跨 episode 拼接 TD 窗口；场景参数保存在 `scenes.json`，通过 scene_id 关联。`data/<split>/expert` 中的测试专家只用于可解性证明。train 和 validation 的自主数据仅在各自场景中采集。所有评测策略共享同一测试列表，报告保存逐场景结果、Wilson 成功率区间和 RECAP 相对 BC 增益。

观测明确为 **9D**：两关节的 sin（2D）、cos（2D）、目标坐标（2D）、障碍物中心（2D）及半径（1D）。条件策略使用三态 one-hot（低/高/null），因此网络输入 12D。目标与障碍物在同一世界坐标系，位置和半径使用米；动作单位 rad，默认 ±0.04。

默认 500 步任务最差回报为 -649，统一使用 horizon+149 归一化；旧 200 步运行仍自动使用 349。成功终止奖励 0，其余普通步 -1，碰撞、约束和超时终止 -150；提高控制频率后每物理秒包含更多普通步，回报不应与旧运行直接比较。gamma=1，所有终止（含超时）取消 bootstrap。默认 TD horizon=25。quantile 筛选只表示排名，不能称为正优势。

RECAP 使用 null NLL + alpha×高优势 NLL，另以 0.1 权重学习低条件，条件 dropout 可配置。Gaussian 标签在边界做 1e-6 数值裁剪，以避免 atanh 无穷；评测使用 tanh(mean)。BC/SFT 使用以动作范围归一化的 MSE。SFT 从 BC checkpoint 初始化，保留专家并加入成功 rollout；RECAP 从随机 Gaussian 网络训练。Value 使用专家和混合 rollout，validation 按场景独立；模型保存最佳验证 checkpoint 和最近训练状态。

## 验证状态与限制

当前开发环境已使用真实 PyTorch 与 MuJoCo 3.3.7 执行完整测试，以及现有专家数据上的 BC 对照训练与闭环评测。本地依赖装在临时隔离目录中。单次、单种子的结果不能视为稳定成功率保证；服务器仍应执行 `pytest -q` 和 smoke 检查。

生成器按几何与执行指标分类 easy/medium/hard，并按配置的配额验收。新专家为全局 C³ 样条；旧配置保留 quintic 和逐路点跟踪。初始直线规划仍使用有限分辨率检查，最终样条使用保守的整曲线间隙检查，实际动力学跟踪继续依赖逐物理步执行验收。当前图表包括 loss 和单轨迹 Value/Return/Advantage、机械臂路径；BC 历史保存定期验证成功率，sweep 脚本可绘制曲线。尚未实现难度匹配子集、跨种子聚合、优势 oracle/标签准确率、排序相关性报告、多轮 RECAP、在线专家纠偏和仿真状态分支接口。BC 的近目标几何监督不等于在线专家纠偏。按用户要求将成功范围从 3 cm 调到 3.5 cm，并独立报告轨迹内 3 cm 指标；新旧环境结果不能直接视作模型提升。

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

各数据划分独立执行顶层 `difficulty_ratio` 配比，三者之和为 1。整数配额使用最大余数法，相同余数按 easy、medium、hard 顺序分配。以 easy=0.60、medium=0.30、hard=0.10 为例，200 条对应 120/60/20，50 条对应 30/15/5，4 条对应 3/1/0，2 条对应 1/1/0。实际配比以所用 YAML 为准。

默认 `planner.proposal: guided` 按各难度剩余配额加权选择目标难度，让每个有缺额的难度都有构造机会。medium 起点的每个关节与目标 IK 解相差 0.60～1.10 rad，障碍物放在关节直连运动时连杆扫过区域的边缘，以构造较浅的阻挡和较短的绕行。hard 起点以每关节 0.30～0.70 rad 为基准，按障碍物半径、连杆半径和安全间隙缩放偏移，以适应内圈大障碍物所需的起终姿态间隔，并检查两根连杆上的多组内部阻挡位置。越过关节限位时翻转偏移方向。候选先检查起点、目标姿态间隙及障碍物中心距离，再进入路径规划和专家执行。目标仍从各集合对应的环带采样，障碍物位置和 medium/hard 起点使用上述条件分布。实际难度继续由规划绕行比和执行间隙决定。`difficulty.json` 中的 `proposed` 记录各目标难度的候选次数，`rejections` 记录当前数据划分的淘汰原因。`generation.json` 记录全程累计结果。`planner.proposal: random` 可使用独立随机候选。省略该字段的旧配置沿用 random。

easy 表示初始关节姿态到所选 IK 解的关节空间直连满足安全间隙。其余场景中，规划绕行比达到 `planner.hard_ratio` 且实际专家轨迹最小间隙低于 `planner.hard_clearance` 时标为 hard，其余标为 medium。配额已满的难度停止收录，分类指标继续按原定义计算。达到 `planner.max_attempts` 后仍有缺额时生成失败，错误信息列出各难度缺额。`difficulty.json` 保存各集合的目标配额、已验收数量和尝试次数，失败时也会写入当前统计。旧配置省略 `difficulty_ratio` 时沿用无配额采样。

medium 构造会检查两根连杆上的有限组浅阻挡位置，并将越过关节限位的起点偏移方向翻转。随后优先检查最多六条单路点局部绕行路径，每段都满足原有插值安全检查且绕行比低于 hard 门槛时，直接交给专家执行。局部路径不可用时再使用 A*。medium 定向候选中直连受阻且规划绕行比达到 hard 门槛的路径会提前淘汰，计入 `generation.json` 的 `medium_detour_too_long`，从而集中生成小绕行的 medium 样本。实际难度分类仍按原有规则计算。`difficulty.json` 的 `local_detours` 记录通过几何检查的局部路径数量，其中包含后续专家执行可能失败的路径。
