# 专家平滑与近目标停滞检查

检查对象为本地 `runs/seed42-8k`。发现两个独立问题：旧专家直接切换路点并限幅动作，路点转向时会发生明显动作跳变；部分 BC 超时轨迹则在目标外产生零动作平衡点。例如 ID 场景 `id-2000042-665`、`id-2000042-689` 最终误差约为 3.21、3.23 cm，最后动作约为 1e-8 rad。成功标准为距离小于 3 cm 并连续保持 5 步。

## 平滑执行对照

在原有 530 个固定场景上重新执行平滑专家，全部成功。参考轨迹使用五次时间标定，每条关节路径线段在起止处的参考速度和加速度为零；不修改 MuJoCo 动力学或成功条件。重放时保留原场景及难度分组，新执行指标另存。

| 每条轨迹指标的中位数 | 原专家 | 平滑专家 |
|---|---:|---:|
| 控制步数 | 61 | 95 |
| 峰值关节 jerk 范数（rad/s³，离散估计） | 100.18 | 23.27 |
| 相邻动作跳变范数（rad） | 0.02620 | 0.00683 |

训练集的实际转移数从 11995 增加到 18224。参考速度、加速度上限不等同于物理实际速度、加速度的严格约束；实际轨迹仍通过 MuJoCo 执行及逐物理步碰撞检测验收。

## 8000 步 BC 对照

均使用 seed42、9D 观测、AdamW、lr=0.001、weight_decay=1e-4、batch size 256。近目标监督仅由 train 场景构造，通过局部 IK 和避障检查补充动作标签；测试策略没有额外控制器。

| 训练数据/设置 | ID 成功数 | 近目标停滞数 |
|---|---:|---:|
| 原专家 | 33/50 | 2 |
| 原专家 + 近目标监督 | 36/50 | 0 |
| 平滑专家 + 近目标监督 | 32/50 | 0 |

这里的“近目标停滞”专指超时、最终距离在 3～6 cm 内，且最后 20 帧相邻关节位移小于 1e-5 rad；不代表所有超时或停滞都已解决。平滑专家加监督的完整对照没有超过原来的 ID 成功率。它的轨迹更长，同样 8000 次更新对应的数据遍历次数更少，是否需要重新调参尚待验证；本次仍保留用户指定的 8000 步默认值。

## 全新生成检查

已用新采样运行完整 smoke 流程：train 和 inner_composition_ood 各 10 个场景、均包含 hard 配额，其余 smoke 数据集按配置生成，随后完成 BC、rollout、SFT、Value、RECAP、评测和绘图。134 项测试通过。该 smoke 用于验证流程，不作为新采样数据的成功率基准。

服务器从新目录生成完整数据集：

```bash
recap generate --run runs/seed42-smooth --config configs/default.yaml
recap train --model bc --run runs/seed42-smooth --config runs/seed42-smooth/config.json
recap evaluate --model bc --run runs/seed42-smooth --config runs/seed42-smooth/config.json
```

五次时间标定参考：[Modern Robotics：Point-to-Point Trajectories](https://modernrobotics.northwestern.edu/nu-gm-book-resource/9-1-and-9-2-point-to-point-trajectories-part-2-of-2/)。位置执行器参考：[MuJoCo XML Reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)。
