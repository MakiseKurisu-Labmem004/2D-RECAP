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

默认训练使用 **AdamW、学习率 0.001、weight_decay=0.0001、2000 个优化器更新步**。默认 BC 近目标监督样本数为 `goal_anchor_samples: 64`，采用下文 50 Hz、C³ 对照选出的配置。默认模板中的 `train.max_steps` 为 2000，SFT、Value 和 RECAP 新训练也会采用该默认值；各模型可独立覆盖和保存训练预算，本轮预算实验仅验证了 BC。`train.max_steps` 非空时优先于 `epochs`；一个 step 指一个 batch 的参数更新。默认每 `train.validation_steps: 500` 步验证一次，第 1 步和最后一步也验证。BC 对全部 validation 场景执行闭环评测，优先按成功率、成功率相同时按 validation MSE 保存 `bc.pt`；SFT、Value 和 RECAP 按各自 validation loss 选模。测试集合不参与选择。

同一个运行目录现在可以复用专家数据和评测场景，并独立调整各模型的训练参数：

```bash
# 读取当前默认训练参数，在已有数据上重新训练 BC
recap train --model bc --run runs/seed42
# 只覆盖学习率
recap train --model bc --run runs/seed42 --lr 0.0005
# 只合并指定文件的 train 段（可以只有 train: {...}）
recap train --model bc --run runs/seed42 --train-config configs/bc_best.yaml
# 从该模型自己的最近 checkpoint 续训，沿用其中的训练参数
recap train --model bc --run runs/seed42 --resume
```

`config.json` 固定数据生成时的环境、场景划分及生成参数；训练不会重写它。旧文件内的 `train` 段仅是生成时的历史快照，当前生效参数分别记录在 `training.bc.json`、`training.sft.json`、`training.value.json`、`training.recap.json`，并完整存入对应 checkpoint。新训练默认使用当前 `configs/default.yaml` 的训练参数及 seed/device，数据和环境继续读取原目录；`--config` 支持完整配置或只有训练参数的 YAML/JSON，显式指定不同环境会报错。`--train-config` 只合并训练字段。

修改学习率、步数、网络和近目标监督数量无需复制或重新生成专家数据。普通 `train` 从头训练，覆盖同名模型、历史和快照；其他步数的旧快照仍携带各自的结构与参数，显式评测时按该快照加载。不会自动搬移或归档文件。专家 NPZ、`scenes.json`、数据生成配置保持不变。评测、Rollout、绘图及默认续生成直接使用保存的数据环境，不会因全局训练默认值改变而拒绝运行。

`data.spec.json` 记录固定环境、规范化后的观测字段及动作定义；旧目录首次成功准备训练时会补齐此元数据。训练还检查轨迹动作维度/范围和可用的控制频率、成功阈值等元数据。所有模型加载都校验环境和观测维度；默认 `bc.pt` 与指定快照使用相同的兼容性规则。改变动力学、控制频率、动作范围或观测/动作定义时，需要兼容的数据。

`--resume` 恢复模型、优化器及随机状态，不能同时改变学习率、网络、batch size 或监督定义；应去掉 `--resume`，在同一数据上从头训练。可通过仅含 `train: {max_steps: 4000}` 的文件配合 `--resume --train-config 文件名` 延长预算。自主 rollout 可作为离线数据继续使用，新采集文件会记录来源 BC checkpoint 的 SHA-256 和训练步数。

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

`migrate` 读取源目录的 `config.json`；`--train-config` 只合并指定 YAML 的 train 段，保留源数据的环境、场景、种子和配比。`--lr` 可再覆盖学习率。不提供覆盖选项时沿用源训练配置。目标目录必须尚不存在；需要独立数据副本时可用 migrate，单纯调整训练参数直接在原目录训练即可。9D checkpoint 可用 `--resume` 续训，续训保留原优化器设置。后续实验应使用新 BC 重新采集 rollout，再训练 SFT、Value 和 RECAP。

固定数据、seed、AdamW 和权重衰减，对比三种学习率，各训练 10000 步并每 500 步验证及保存快照：

```bash
python scripts/sweep_bc.py --run runs/seed42 --output runs/bc-sweep-10k --lrs 0.0003 0.001 0.003 --steps 10000 --interval 500
```

脚本生成各组独立运行目录、`sweep.json` 和 `learning_curves.png`。依据 validation 闭环成功率选出配置与 checkpoint，再仅对选定模型执行 ID 测试；单次 seed 的最佳结果只代表本次比较范围。脚本默认使用可用 GPU，`--device cpu --threads 1` 可指定 CPU。

本地 seed42 的三组 10000 步对照已完成：lr=0.001 在第 8000 步取得最佳 validation 成功率 73.3%，选定后 ID 成功率为 66.0%。完整设置、曲线和难度指标见 [实验记录](docs/experiments/bc_adamw_10k_seed42.md)。

后续 [20000 步对照](docs/experiments/bc_adamw_20k_seed42.md) 固定 batch size=256，比较 lr=0.001 与 0.0005：两组最佳 validation 均为 73.3%，按较低 MSE 选中 lr=0.0005 的第 14000 步 checkpoint，ID 为 34/50（68%）。训练更久未持续改善闭环成功率，仍应保留中间最优模型。

`recap all` 顺序完成上述步骤。默认 200 条训练专家轨迹、400 条自主轨迹，另有独立 validation 和六个测试集合。生成成本可能较高，先运行 smoke。生成器达到尝试预算仍不足时会报错，不能把规划失败解释为场景不可解。已有场景 manifest 不允许覆盖；改变数据生成配置使用不同目录，仅改变训练配置可复用原目录。多训练种子修改 YAML seed 并使用各自目录；如需跨种子严格共享测试集，应在实验前显式准备共享 manifest；BC 搜索脚本支持冻结同一份场景后聚合多个训练种子。

## 平滑专家与近目标纠偏

新配置使用 `planner.expert_controller: spline`：对整条关节路点路径联合求解分段七次最小 snap 样条，约束内部路点的位置、速度、加速度和 jerk 连续，起终点速度、加速度和 jerk 为零。内部路点可带非零速度通过。默认每关节参考速度上限为 0.35 rad/s、加速度上限为 2 rad/s²、jerk 上限为 10 rad/s³；按三种导数约束给每段分配时间，再统一拉长时长满足限值。这些是参考轨迹限值，实际运动仍由 MuJoCo 积分。

整条参考曲线转换为 Bézier 控制点后递归检查关节限位和两根连杆的完整间隙，并用控制点包络与连杆位移上界给出保守安全判定。弯曲路径不安全时逐级缩小内部路点导数，必要时退回路点处零速度、零加速度、零 jerk 的 C³ 路径；无法通过检查则拒收。实际执行仍逐物理步检查碰撞。NPZ 的 `expert_spline_blend` 记录导数缩放系数，0 表示使用停稳通过的安全回退；`expert_spline_duration` 记录参考时长。旧配置仍支持 `quintic`，缺少控制器字段时沿用 waypoint。

参考时长加成功保持时间若超过任务预算，会在执行前拒收，记录为 `spline_too_long`，避免慢速轨迹反复仿真到超时。采样范围和难度配额保持原配置；收录的专家仍必须通过实际执行验收。

默认控制频率为 **50 Hz**：物理步长仍为 0.005 s，每 4 个物理步更新一次动作。动作目标偏移上限降为 ±0.04 rad。`env.control_interpolation: c3_filter` 使用四级连续滤波，在每个物理步精确推进控制目标，并保留跨控制周期的导数状态，构成 C³ 连续的理想控制信号；MuJoCo 以 200 Hz 采样执行。滤波每级时间常数 `control_filter_tau: 0.003` s，总低频延迟约 12 ms，专家与所有学习策略共用这一执行规则。实际运动的高阶导数还受采样、动作限幅和动力学影响。

任务 horizon 与专家上限均为 **500 控制步（10 秒）**；成功保持 13 步约为 0.26 秒。TD horizon 调为 25 步，仍约为 0.5 秒。BC 默认学习率为 0.001、batch size 为 256，参数更新步数按本轮实验结果设为 2000。旧运行缺少插值字段时仍使用原 hold 模式；要使用新频率、动作范围和专家，需以新配置在新目录重新生成并训练。

新生成时若专家实际经过另一个 IK 解到达目标，会安全截短原路径并重新执行，再按最终路径和实际间隙计算难度。原有 easy/medium/hard 配额仍执行。需要全新数据时，用新目录直接生成：

```bash
recap generate --run runs/seed42-c3 --config configs/default.yaml
recap train --model bc --run runs/seed42-c3 --config runs/seed42-c3/config.json
recap evaluate --model bc --run runs/seed42-c3 --config runs/seed42-c3/config.json
```

BC 默认使用 `train.goal_anchor_samples: 64`：仅在 train 场景的安全 IK 解附近，以每关节 ±0.12 rad 扰动补充局部纠偏标签，同时加入精确目标处的零动作标签。所有样本都通过关节限位和直连避障检查；它们只参与 BC 动作监督，不伪造 episode、回报或 Value 标签。设为 0 可关闭。验证/测试不添加控制器，仍直接使用策略输出。`bc.goal_anchors.json` 记录补充样本数量。

默认成功距离从 0.03 m 放宽到 **0.035 m**，连续保持 13 步，成功结束不要求末端精确重合目标中心。评测另外报告 `strict_success_rate`（已记录轨迹中按 3 cm 连续保持同样步数的成功率）和 `tolerance_only_success_count`（仅因放宽范围而通过的数量）；前者不等于在旧判定下重新执行一次策略。`near_goal_stall_count` 独立统计范围外的静止超时：最终距离在成功阈值外两倍阈值内，且最后 20 帧相邻关节位移均小于 1e-5 rad。

新轨迹保存逐帧 `distance`、`goal_hold_steps`，以及物理子步 `control_commands`、`substep_qpos`、`executed_substeps`。`applied_command` 保留限幅后的目标命令，实际送入 MuJoCo 的目标在 `control_commands` 中；碰撞提前终止时只读取每帧前 `executed_substeps` 个子步，余下槽位为末值填充。`control_dt` 和 `physics_timestep` 记录物理时间单位，评测的 `completion_seconds` 可用于不同控制频率之间的时间比较。模型观测仍为 9D。

若需保留既有场景，可运行 `python scripts/rebuild_experts.py --run runs/seed42-8k --output runs/seed42-smooth-replay --config configs/default.yaml`。该工具在新目录重新执行专家，冻结场景几何和原难度分组，另存 `expert_replay_metrics` 与 `expert_replay.json`，不把重放后的指标冒充原始采样难度。迁移数据的 `recap migrate` 不会重新执行或平滑已有动作。

此前的 20 Hz、3 cm、分段 quintic 历史对照中，530 条平滑专家均执行成功，峰值 jerk 的中位数降低约 77%；平滑数据加近目标监督的 8000 步 BC 对照 ID 为 64%，仅在旧专家上增加近目标监督时为 72%。这些数据不属于当前 50 Hz、C³、3.5 cm 配置，详见 [历史检查记录](docs/experiments/expert_smooth_goal_stall_seed42.md)。

此前实现已通过 161 项测试、含 hard 配额的从零生成完整 smoke，以及 step/epoch 两种训练模式检查。在同一批 24 个固定场景、相同 200 Hz 物理采样和 3.5 cm 成功标准下，旧 20 Hz quintic 为 24/24 成功，新 50 Hz 慢速 C³ 为 20/24；4 个失败均为 10 秒预算内未完成。在两组都成功的 20 个场景中，每条轨迹峰值物理 jerk 的中位数为 10245→463 rad/s³，子步控制目标跳变为 0.03139→0.00287 rad，中位执行时间为 4.225→6.69 秒。该对照同时改变速度、动作上限、样条和控制频率，不能将效果单独归因于某一项，也不代表 BC 策略成功率。详情见 [验证数据](docs/experiments/c3_50hz_verification.json) 和 [对照图](docs/experiments/c3_50hz_comparison.png)。

## 当前 BC 配置搜索（2026-11-15）

本轮使用 `runs/seed42-smooth` 中完整的 **50 Hz、C³ spline、9D** 数据，与上述 20 Hz 历史实验分开统计。固定 200 个 train 场景、30 个 validation 场景和六类各 50 个测试场景；难度配比为 70%/20%/10%。已核验 530 个场景 ID 和几何互不重复、专家全部成功，轨迹元数据与当前环境一致。成功标准始终为 3.5 cm、保持 13 步，未通过修改环境提高分数。

首轮固定 seed42、AdamW、weight decay=1e-4、batch size=256，各训练 8000 步，每 500 步用完整 validation 的闭环成功率选择 checkpoint；成功率相同再比较动作 MSE。

| 学习率 | 网络（隐藏层×宽度） | goal_anchor_samples | 最佳步数 | validation |
|---|---|---:|---:|---:|
| 0.001 | 6×128 | 16（原默认） | 2000 | 21/30，70.0% |
| 0.0005 | 6×128 | 16 | 5000 | 22/30，73.3% |
| 0.0003 | 6×128 | 16 | 7000 | 20/30，66.7% |
| 0.001 | 3×128 | 16 | 4500 | 19/30，63.3% |
| 0.0005 | 3×128 | 16 | 6000 | 17/30，56.7% |
| 0.001 | 3×256 | 16 | 2000 | 17/30，56.7% |
| 0.0005 | 3×256 | 16 | 4500 | 17/30，56.7% |
| 0.001 | 6×128 | 64 | 2000 | 24/30，80.0% |

前两名扩展到 20000 步，并固定同一份场景用训练种子 42、43、44 复验。`lr=0.001, goal_anchor_samples=64` 的三个最佳 validation 为 80.0%、83.3%、73.3%，平均 **78.9%**，且最佳 checkpoint 都在第 **2000** 步；`lr=0.0005, goal_anchor_samples=16` 的平均为 73.3%，最佳步数分别为 5000、4000、4500。依据三种子平均 validation 成功率锁定前者后，再开启测试集；这里的“最佳”只指本轮比较范围，30 个固定验证场景和三个训练种子不能代表所有数据种子的稳定排序。

| 测试集合 | 默认配置，seed42 | 推荐配置，seed42 | 推荐配置，三训练种子平均 |
|---|---:|---:|---:|
| id | 29/50，58% | 34/50，68% | 64.0% |
| inner_goal_ood | 9/50，18% | 20/50，40% | 41.3% |
| outer_goal_ood | 7/50，14% | 13/50，26% | 28.0% |
| size_ood | 22/50，44% | 30/50，60% | 58.7% |
| inner_composition_ood | 6/50，12% | 14/50，28% | 32.0% |
| outer_composition_ood | 12/50，24% | 21/50，42% | 38.7% |
| 全部测试 | 85/300，28.3% | 132/300，44.0% | 43.8% |

推荐配置三个种子的测试总体为 44.0%、45.0%、42.3%，均值 43.8%、样本标准差 1.3 个百分点；这是相同 300 个场景的三次训练种子复验。提升仍主要来自 easy，三个种子合计 medium 为 12/180、hard 为 3/90 次成功，不能据此称为已解决困难绕障或 OOD 泛化。近目标静止超时仍存在：seed42 默认配置为 10/300 次，推荐配置为 17/300 次；本轮按整体成功率选择，没有将停滞指标作为优化目标。

可复用配置见 [configs/bc_best.yaml](configs/bc_best.yaml)：AdamW、lr=0.001、weight decay=1e-4、batch size=256、6×128、goal_anchor_samples=64、max_steps=2000、每 500 步验证。已额外从头训练 2000 步，逐参数核对与本轮 seed42 最佳 checkpoint 完全一致。该预设用于 BC；其余模型需要各自选择训练预算。`configs/default.yaml` 已对齐此配置；各模型可在训练时独立覆盖预算。

```bash
# 复用已生成的当前数据，在独立目录训练推荐 BC
recap migrate --run runs/seed42-smooth --output-run runs/seed42-bc-best --train-config configs/bc_best.yaml
recap train --model bc --run runs/seed42-bc-best --config runs/seed42-bc-best/config.json
recap evaluate --model bc --run runs/seed42-bc-best
```

完整数据审计、参数、分难度结果和复现记录见 [实验结果 JSON](docs/experiments/bc_c3_search_20261115.json)，验证曲线见 [对照图](docs/experiments/bc_c3_search_20261115_curves.png)。本地权重和逐场景记录位于 `runs/experiments/bc_c3_search_20261115`，该目录按仓库既有规则被 Git 忽略。实验在 CPU、PyTorch 2.11.0 和 MuJoCo 3.3.7 上完成；不同设备和数值实现可能产生差异。

新增搜索脚本将源数据复制到独立运行目录，只复制 train/validation 专家标签，完整测试几何保持一致。源模型和源运行文件不修改。首轮、复验、选择记录与逐场景测试指标保存到输出目录；相同命令可恢复训练，并保留已经完成的首轮选择记录。

```bash
# CPU 并行执行本轮对照
python scripts/search_bc.py --run runs/seed42-smooth --output runs/experiments/bc-c3-search --plan configs/bc_search.json --steps 8000 --extend-steps 20000 --seeds 42 43 44 --finalists 2 --workers 4 --threads 1 --device cpu
# 单 GPU：使用 --workers 1 --device cuda；不同时启动多个训练进程占用同一 GPU
```

## 目标任务持续改进：两组接管对照

`recap improve` 将避障到达视为一个任务，默认把 train、ID 和各 OOD 的已有场景合成一个统一实践池，集合名仅保留为来源元数据；validation 保留作训练监控。默认不额外划分目标 holdout。在一个运行目录内支持多轮实践，两组都持续使用原 `data/train/expert` 和已有 `data/train/rollout`。原 validation 数据只用于验证。组间区别是新增数据是否包含少量专家接管：

| 模式 | 每轮训练数据 |
|---|---|
| `autonomous` | 原训练集 + 累积自主 Rollout |
| `corrections` | 原训练集 + 累积自主 Rollout + 累积局部专家接管片段 |
| `both` | 从同一个 BC checkpoint 分别运行上述两组 |

```bash
# 对 ID 目标任务做两组实验，原数据和初始 BC 保留在原处
recap improve --run runs/seed42 --mode both --experiment pooled-v1 --rounds 3 --episodes 200 --steps 1000 --lr 0.0001 --expert-episodes 10
# 单独运行一组，使用不同实验名称
recap improve --run runs/seed42 --mode autonomous --experiment autonomous-v1
recap improve --run runs/seed42 --mode corrections --experiment corrections-v1
```

`--steps 1000` 表示每轮 Value 和 RECAP 分别执行 1000 次优化器更新。每组从同一个 BC 初始化；RECAP 的均值分支复制 BC 权重，新增条件输入初始为零，避免随机重建一个已经表现不错的策略。后续轮次从上一轮训练后的策略继续微调，Value 也随其更新。每轮的原训练数据与累积新经验共同参与优化，保留成功经验以减少遗忘。自主采样在统一实践池中均匀、有放回随机抽取，每个场景概率相同，与历史成功或失败无关。同一轮两组使用相同 seed 和场景抽样序列。专家接管组仍只从本轮实际采到的失败轨迹中提供有限纠正。

接管组先完成与自主组相同预算的均匀随机采集，再从实际失败记录中选择至多 `--expert-episodes 10` 次接管。默认回放到 `max(0,失败轨迹长度-25)`，重放真实动作恢复当时的物理和滤波状态。随后由规划专家持续控制直到满足成功距离和保持步数要求，不再交还策略，也不设单次或每轮专家执行步数上限。`--lookback` 可调整接管起点。若无法规划、发生碰撞/约束失败，或专家参考结束后仍在目标外静止，则记录接管失败；不会把失败数据当作成功纠正。

这是对已知失败进行事后接管的仿真近似，不是部署时预知失败的控制器；只使用当前状态、目标和几何规划，不读取目标场景原有的专家轨迹。从接管起到成功结束的全部帧均由专家执行，`intervention_mask` 全为真，全部强制标为高优势，默认纠正帧权重为 `--correction-weight 5`。原训练集和 validation 的 `expert/` 中所有示范帧也强制标高；此前的自主轨迹仍按估计 Advantage 标记。

自主执行和正式评测仍遵守原任务 horizon。专家示教可延长到成功，数据记录 `task_success` 和 `expert_completion_step`：超过原时限才恢复成功，不计为按时完成任务。完整接管轨迹用于 RECAP 策略纠偏；Value 只使用原时限内的前缀，若超时则在原截止处赋予超时终止与失败奖励，避免改变原任务的价值定义。每轮分别报告实际专家动作数、专家恢复次数、时限内辅助成功次数和重放步数。原训练数据继续参与两组训练。

默认 `--splits all --holdout-fraction 0`：合并所有非 validation 的现有场景，一起采集 Rollout、混合原训练集进行训练，不重新生成几何场景。默认数据规模下为 train 200 + 六类各 50，共 500 个实践场景；30 个 validation 场景仅作监控。`--episodes 200` 表示从统一池均匀有放回采样 200 次，不保证每个场景每轮都采到；回归检查覆盖整个实践池。原 ID/OOD 标签仅记录数据来源，池内不按这些标签分别训练。结果表示已练习场景上的改进。若另需独立泛化检查，可显式设置 `--holdout-fraction 0.2`；`--splits id` 或 `--scene-ids id-2000042-22` 仍可用于限定调试范围。原 `scenes.json` 和专家文件不修改；共同监督数据仍按两组约定使用原训练集，其他来源只提供实践场景，并不整批加入其专家答案。

每轮训练结束后，直接使用本轮完整训练预算后的 `recap.last.pt` 作为下一轮策略，不设成功率接受门槛，也不根据退步回退策略。无专家辅助的实践池和原 validation 评测仅记录指标、修复及退步场景。Value 仍按验证 loss 选择本轮 `value.pt`。显式启用 holdout 时，仅在开始和最后报告；默认其为空。

每轮原始自主轨迹保存在 `data/adaptation/rollout/`，完整专家接管段保存在 `data/adaptation/corrections/`。这两类本轮新生成的 NPZ 在训练后也会写入当前轮的 Value、Return、Advantage、标签及 Value 模型哈希，直接读取即可查看；旧轮次原文件保留其当时的标注。RECAP 计算 Advantage 后，为截至该轮的全部新增轨迹保存紧凑标注到 `data/adaptation/annotations/rollout/` 和 `annotations/corrections/`；`annotations.json` 索引将标注关联到首次采集的原始 NPZ，避免逐轮复制整份旧轨迹。每个标注文件包含 `value_prediction`、`mc_return`、`td_advantage`、`advantage_indicator`、`intervention_mask`、帧索引、原始轨迹相对路径、当前 Value checkpoint 哈希、n_step、阈值和回报缩放。不同轮次会为同一条历史轨迹保存不同 Value 下的标注。

Value/Advantage 的索引 t 对应动作前 `observation[t]` 及 `action[t]`；原轨迹 `qpos[t]` 是执行该动作后的姿态，元数据 `value_alignment` 明确标记这一对齐方式。原始专家数据继续共享，固定高优势规则记录在训练诊断中。每轮无辅助评测 NPZ 另存到 `data/evaluation/practice/` 与 `data/evaluation/validation/`，其中直接包含逐帧 Value、Return、Advantage 和标签；评测只用于记录，不影响下一轮策略选择。显式 `recap evaluate --checkpoint ...` 的只报告、不保存行为保持不变。

结果保存在原目录的独立子树，原 BC 权重、专家数据和普通训练产物不覆盖：

```text
runs/seed42/improvement/pooled-v1/
├── protocol.json             # 公共初始权重哈希、训练来源、统一实践池与可选留出场景
├── baseline.json
├── initial.bc.pt            # 固定初始模型快照，两组共用
├── comparison.json           # 两组最终策略、无辅助成功率、回归与专家预算
├── autonomous/
│   ├── summary.json
│   └── round-001/            # Value/RECAP、training.*、本轮数据和 result.json
└── corrections/
    ├── summary.json
    └── round-001/
        └── data/adaptation/
            ├── rollout/      # 纯自主轨迹
            └── corrections/  # 接管起至专家成功的完整片段，intervention_mask 全为真
```

各组 `summary.json` 的 `active_policy` 与 `active_policy_kind` 指向最后一轮的 `recap.last.pt`，每轮都更新，不执行策略接受/拒绝。Value 按验证 loss 选模。实验名称已存在时会拒绝覆盖，另取 `--experiment` 名称可重新对照。`--train-config` 可覆盖其他训练字段，但网络须与初始 BC 一致，步数、学习率和纠正权重以 improve 的命令行参数为准。这些默认值是实验预算，尚未构成长训练成功率保证。

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

成功终止奖励 0，其余普通步 -1，碰撞、约束和超时终止奖励为 `-env.horizon`。默认 500 步任务失败终止奖励 -500，最差总回报 -999，统一使用 `2*horizon-1` 归一化。因此时限内最慢成功的回报也高于第一步失败。Value/RECAP 训练会在内存中将旧轨迹的失败奖励改为新定义，原始 NPZ 与 BC 可以复用；旧 Value/RECAP 训练断点不能续训，需要重新训练，新 checkpoint 记录 `train.reward_version`。旧 Value 不用于新奖励下的 Advantage 标注，前端补算历史实验仍保留其旧模型的奖励尺度。提高控制频率后每物理秒包含更多普通步，回报不应与旧运行直接比较。gamma=1，所有终止（含超时）取消 bootstrap。默认 TD horizon=25。quantile 筛选只表示排名，不能称为正优势。

RECAP 中原始专家示范帧与专家实际接管帧强制使用高优势条件，自主帧按估计 Advantage 和阈值标记；Value 的回报监督与数值 Advantage 保持原定义。`diagnostics.json` 的 `selected_fraction` 表示最终标签中高优势帧占比，`estimated_selected_fraction` 保留按数值 Advantage 阈值筛选的占比。RECAP 使用 null NLL + alpha×高优势 NLL，另以 0.1 权重学习低条件，条件 dropout 可配置。Gaussian 标签在边界做 1e-6 数值裁剪，以避免 atanh 无穷；评测使用 tanh(mean)。BC/SFT 使用以动作范围归一化的 MSE。SFT 从 BC checkpoint 初始化，保留专家并加入成功 rollout；RECAP 从随机 Gaussian 网络训练。Value 使用专家和混合 rollout，validation 按场景独立；模型保存最佳验证 checkpoint 和最近训练状态。

## 验证状态与限制

当前开发环境已使用真实 PyTorch 与 MuJoCo 3.3.7 执行完整测试，以及现有专家数据上的 BC 对照训练与闭环评测。本地依赖装在临时隔离目录中。单次、单种子的结果不能视为稳定成功率保证；服务器仍应执行 `pytest -q` 和 smoke 检查。

生成器按几何与执行指标分类 easy/medium/hard，并按配置的配额验收。新专家为全局 C³ 样条；旧配置保留 quintic 和逐路点跟踪。初始直线规划仍使用有限分辨率检查，最终样条使用保守的整曲线间隙检查，实际动力学跟踪继续依赖逐物理步执行验收。当前图表包括 loss 和单轨迹 Value/Return/Advantage、机械臂路径；BC 历史保存定期验证成功率，sweep 脚本可绘制曲线。尚未实现难度匹配子集、跨数据种子聚合、优势 oracle/标签准确率、排序相关性报告、多轮 RECAP、在线专家纠偏和仿真状态分支接口。BC 的近目标几何监督不等于在线专家纠偏。按用户要求将成功范围从 3 cm 调到 3.5 cm，并独立报告轨迹内 3 cm 指标；新旧环境结果不能直接视作模型提升。

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
