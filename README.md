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

`recap all` 顺序完成上述步骤。默认 200 条训练专家轨迹、400 条自主轨迹，另有独立 validation 和四个测试集合。生成成本可能较高，先运行 smoke。生成器达到尝试预算仍不足时会报错，不能把规划失败解释为场景不可解。已有场景 manifest 不允许覆盖；不同实验配置使用不同目录。多训练种子修改 YAML seed 并使用各自目录；如需跨种子严格共享测试集，应在实验前显式准备共享 manifest，目前没有跨运行聚合命令。

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

当前按几何指标分类 easy/medium/hard，但未实现按难度配额采样，因此默认生成不保证 medium 为主；正式研究前需检查 `scenes.json` 中的统计。专家采用逐路点限幅跟踪与捷径平滑，没有高级时间参数化。规划插值为有限分辨率验证，不是数学意义的连续碰撞证明。当前图表包括 loss 和单轨迹 Value/Return/Advantage、机械臂路径；尚未实现难度匹配子集、跨种子聚合、优势 oracle/标签准确率、排序相关性报告、成功率训练曲线、多轮 RECAP、专家纠偏和仿真状态分支接口。上述功能不应报告为已完成。没有为达到预期成功率修改标签或筛选测试结果。
