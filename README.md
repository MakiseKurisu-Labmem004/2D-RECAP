# MuJoCo Minimal RECAP

纯数值、CPU 可运行的平面接触抓取研究工程。所有实验由 YAML 合并配置驱动；没有视觉分支、镜像 OOD、物体附着、传送或专家兜底。RECAP 在这里指优势条件监督更新的最小研究实现，不声称复现某篇论文的全部算法。

## Linux 服务器安装与入口

建议 Python 3.12，使用 uv 管理服务器环境。在仓库根目录执行（需先安装 uv 并加入 PATH）：

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --torch-backend cpu -r requirements.txt
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -m pytest -q
.venv/bin/python -m minimal_recap.run --config configs/experiments/smoke.yaml
```

16 vCPU 服务器采集 **200 条成功专家轨迹**：

```bash
.venv/bin/python -m minimal_recap.run --config configs/experiments/server_cpu.yaml
```

该配置只执行 physics → expert → freeze，生成专家数据、长度统计和奖励 contract。也可直接 `bash scripts/server.sh`，一次完成 uv 环境准备、测试与这次采集。结果写入启动时打印的 `runs/<run>` 目录。SSH 长任务建议在 tmux 中运行。

中断后保持配置不变，使用实际目录恢复：

```bash
.venv/bin/python -m minimal_recap.run --config configs/experiments/server_cpu.yaml --resume runs/<run>
```

使用同一批 200 条示范继续训练与评估：

```bash
.venv/bin/python -m minimal_recap.run --config configs/experiments/server_cpu.yaml --resume runs/<run> --stages bc rollout value label update next_rollout evaluate diagnose
```

Slurm 脚本申请 16 vCPU / 32 GB / 12 小时，默认同样只采集 200 条成功轨迹。先在允许联网的环境按上面的 uv 命令准备依赖，再在项目根目录执行 `sbatch scripts/recap.slurm`；分区和账户按集群要求设置。当前采集逐回合串行执行，MLP 仍使用单线程，申请 16 vCPU 不代表自动并行或 16 倍加速。

数值训练和评估无需 DISPLAY、OpenGL 或 Xvfb。只有 GIF/图像回放需要渲染上下文；NVIDIA/EGL 可用时运行 `MUJOCO_GL=egl python -m minimal_recap.replay ...`，CPU 软件渲染可使用 `MUJOCO_GL=osmesa`，需系统安装对应 Mesa 库。原生交互 viewer 需要显示器，不用于服务器训练。

默认小 MLP 用 CPU 单线程，通常更经济；GPU 可用 `configs/experiments/server_cuda.yaml`。按服务器驱动安装匹配的官方 PyTorch CUDA wheel，再运行该配置；不要假设任意 CUDA wheel 都能匹配本机驱动。上述 uv 安装命令和 server.sh 明确使用 CPU wheel；CUDA 路径提供支持，本次本地测试仅验证 CPU。GPU 运行请在所在集群添加相应 GPU 分区和资源指令。

多训练种子及汇总：

```bash
python scripts/run_seeds.py --config configs/experiments/recap.yaml --seeds 1 2 3 --stages physics expert freeze bc rollout value label compare diagnose
python -m minimal_recap.aggregate --runs runs/<seed1> runs/<seed2> runs/<seed3> --output runs/summary.json
```

该脚本共用专家场景、rollout 场景与测试场景，只改变训练随机种子；独立初始化/采样使训练结果有随机变化。也可自行创建 YAML 覆盖文件明确设置所有 seed。长任务建议使用 Slurm 或 tmux，`--resume` 恢复指定 run。没有后台隐式任务。

依赖版本见 requirements.txt；实验同时记录实际版本。同一软件/硬件环境可重放，不保证跨 MuJoCo 版本或 CPU/CUDA 逐位一致。Windows 可使用 `.venv\Scripts\python` 运行相同模块。

正式配置：

```bash
python -m minimal_recap.run --config configs/experiments/recap.yaml
python -m minimal_recap.run --config configs/experiments/recap.yaml --stages physics expert freeze bc rollout value label compare
python -m minimal_recap.run --config configs/experiments/online.yaml
```

每种更新方法使用相同数量的优化步；批次从各自合格训练集有放回采样。固定数据比较优先运行 `compare`，它从同一 BC 权重初始化 aggregate_bc、success_bc、conditioned 与 unconditioned，共用原始 rollout、优化轮数、每轮 updates_per_epoch、batch、ID checkpoint 选择和测试种子。success_bc 的成功过滤是有意的基线差异；若成功轨迹不足以划分训练验证集，显式报错，不能伪造结果。RECAP 仍保留全部失败轨迹。初始 BC 没有额外交互，更新方法共享相同的额外交互预算；报告中应列出这个差异。

`--stages` 支持 physics、expert、freeze、bc、rollout、value、label、update、compare、online、evaluate、diagnose、next_rollout。完整流水线按配置执行。已完成阶段通过 done 标记复用。每个 run 都有独立时间戳目录；不会覆盖其他实验。

```bash
python -m minimal_recap.run --config configs/experiments/recap.yaml --resume runs/<run> --stages value label update evaluate diagnose
python -m minimal_recap.run --config configs/experiments/aggregate_bc.yaml --reuse runs/<recap-run> --stages value label update evaluate
```

Resume 要求解析后的配置完全相同，调整奖励、预算或分布需要新版本新目录。reuse 校验物理、模型、奖励、数据、种子与执行设置一致。训练在每个 epoch 保存模型、优化器和 RNG；中断后恢复到上一个完整 epoch。采集按完整 episode 原子保存；捕获取消/异常时保存 aborted 与最后完整状态，重启继续。OS 强杀或断电可能丢失当前未保存 episode，此时从该 episode 的同一种子重跑，不把残片加入 MC。不要修改已产生数据的 run。

## 物理与任务定义

x/z 两个滑动关节控制固定朝下夹爪；双指各一个滑动关节，通过 equality joint 耦合为一个开度，有限力 position servo 驱动。物块只保留 x/z 与绕 y 转动，locked_rotation 配置移除转动关节。真实重力、摩擦和碰撞决定抓取，所有专家动作通过同一个 3 维接口执行。

归一化动作 `[dx,dz,opening]` 先裁剪到 [-1,1]。前两维乘 max_delta，从**当前实际**夹爪位置生成目标，再裁剪工作区；第三维线性映射到 [0,max_opening] 的物理开度。20 Hz / 0.0025 s 对应每控制步 20 个物理步，不可整除时拒绝配置。位置目标伺服有限力，实际位移可能小于目标或因惯性超过单步目标差。

4 cm 物块、8 cm 开度、4 cm 墙宽、6–10 cm 训练墙高、±30 cm 工作区、35 cm 最大高度、1 cm 最大目标增量均在配置中。`physics` 是必须先跑的接触抓起、运输越墙、目标释放与稳定判定门禁；它还检查两级墙高与远远布局的专家可行性。有限样本通过不等于所有随机场景的形式证明。

成功要求物体完整水平包络进入目标范围、接触桌面、线/角速度足够小、夹爪打开且与物体脱离，并保持配置步数。可恢复空抓、掉落不终止；物块水平越工作区/落出桌面才失败。专家无任务步数限制、不使用 TimeLimit；即使超过 200 步也保留其成功轨迹。physics 的 1200 步诊断保护抛程序异常，不作为任务失败。采集 max_attempts 是批次失败次数上限，不是单条成功轨迹的时限。

BC/RECAP 与评估使用独立配置预算。正式比较前查看 expert_lengths.json 与独立 ID 验证，再统一预算，并**新建实验版本**冻结。预算在任务里视为 terminal（terminated=true），外部中断是 aborted，不产生 terminal MC 标签。success 同步发生时优先。

## 奖励、价值与优势

普通步 -1；成功终止步 0；失败/任务超时终止步 -C_fail，终止奖励覆盖普通奖励，gamma=1。专家成功长度 L 的 Return 为 -(L-1)。先收集专家，再取 M=max(专家最大成功长度, rollout预算, evaluation预算)；默认 C_fail=M，严格检查 C_fail>M-1。最快失败 -C_fail 严格低于保证范围最慢成功 -(M-1)。最坏策略回报下界为 -(最大策略预算-1+C_fail)，价值分布支持默认取 [该下界,0]，所有标签越界直接报错。contract.json 固定该组定义，新增超范围数据需要新版本、统一重算；不可单独更改某种方法奖励。

独立策略与价值网络均为 **6 个隐藏层 × 128**，默认 ReLU，输出层另算。策略 tanh 三维输出；价值 softmax 201 bins，取支持点期望。MC 标签在邻近 bins 线性插值，可关闭。归一化只由训练轨迹计算；策略更新沿用初始 BC 统计量以保持初始化完全一致。价值仅使用有相同任务预算的完整 BC/RECAP rollout，成功与失败均包含。专家默认只用于 BC 与专家保留监督项。

`A[t] = sum(r[t:t+k]) + V(next_obs[t+k-1]) - V(obs[t])`，k=min(n,剩余片段长度)，到 task terminal 后 bootstrap=0，n 默认10。条件为 A>threshold 的二元标签；推理固定高优势条件。condition_weight 控制高条件监督损失权重，expert_ratio 以损失混合比例实现专家保留（每个 minibatch 独立重采样专家）。unconditioned 是无条件全数据监督路径；aggregate_bc 与它数值等价，是用于核对的别名。支持 all_rounds 与 latest_round 混合，checkpoint 记录训练/验证 episode ID 与混合方式。

V 是该轮行为数据下的未来回报条件期望。行为策略包括 checkpoint、动作处理和探索分布。网络不输入剩余时间，而有限预算改变同一几何状态的未来回报，因此这是部分可观测、数据分布相关的估计；不是精确的时间依赖 Markov V，也不是几何对齐分数。错位但可恢复的状态不必低值。混合多轮策略时 V 对应混合行为分布，不能声称是单一当前策略的精确 V。

## 数据、诊断与泛化

[数据格式](docs/data_format.md)。expert 只取成功用于 BC；BC 冻结后在独立训练种子 rollout，固定回合数、无专家、无在线更新，成功失败完整保留。none 和 gaussian 两种探索设置；动作记录的是裁剪后真正执行值。正式评估单独配置 exploration，默认无噪声。

训练布局：near=12–16 cm、far=20–24 cm，物体在墙左、目标在墙右；near_near、near_far、far_near 训练，far_far 为组合 OOD。墙高 OOD 11–12 和13–14 cm 保持 ID 布局组合，每套只变一个因素，不把 OOD 结果用于 checkpoint 选择。

恢复池通过实际专家控制轨迹施加偏移空抓或运输张爪产生，保存全物理状态、控制器状态、已耗步数与剩余预算。所有策略恢复相同状态，同一总预算，剩余预算不进网络。池里只存尚未终止状态。recovery 成功率按 empty/drop 分别报告。

诊断从实际抓取前状态出发，通过动作移动到水平偏移；每个偏移用同一参考 BC 续跑，比较预测 V 与实际 Return。direct/correct/wrong 使用相同长度前缀（默认10步，前4步保持打开并执行零/正确/错误移动，后6步闭合），然后共同参考策略续跑；比较实际 Return 与前缀估计优势。确定性只计一次；随机参考必须配置至少2次续跑。默认 trace 展示同一 BC 行为轨迹的 Value、MC Return、n-step Advantage，输出 offsets.png、trajectory.png、diagnostics.json 与原始 trace.npz。

评估输出成功率及 Wilson 区间、首次抓取成功率、重抓次数、成功完成时间、平均 Return、恢复成功率。诊断输出价值 MAE 与优势成对排序准确率。first_grasp 以首次闭合尝试是否造成经接触验证的抬升定义；不是仅看开度。恢复池成功率是主要恢复指标，普通 rollout 的 drops 可能包含正常释放，不能直接把 drops 当意外掉落标签。

可视化回放：

```bash
python -m minimal_recap.replay --run runs/<run> --episode runs/<run>/expert/000000.npz
python -m minimal_recap.replay --run runs/<run> --episode runs/<run>/expert/000000.npz --output replay.gif --stride 2
```

实际恢复池与诊断均保存 JSON；数值输入策略完全不使用渲染。批量结果要以多个**训练种子**重复（例如 seeds/training=1,2,3 的覆盖配置），共用评估种子。不能把确定性重复 rollout 或同一训练种子的多个测试场景当成多个训练种子。smoke 仅检查工程链路，不用于科学结论。物理门禁、实际 smoke 结果和测试状态见 [验证记录](docs/validation.md)。

## 设计假设与解释边界

本工程刻意限制为平面接触、已知数值状态、桌面单块物体与固定朝向夹爪。闭环专家依赖环境可读接触，但学习网络只看声明数值特征。默认 YAML 是低成本起点，不保证两个 epoch BC 会抓取，也不预设 RECAP 优于 BC 或所有 OOD 可泛化。正式实验需要足够示范、ID 验证确定预算、多个训练种子以及固定数据集比较后再迭代。所有成功率必须引用具体 run，未运行实验不得写成结论。

实现依据：MuJoCo 的 [position servo 与 XML](https://mujoco.readthedocs.io/en/stable/XMLreference.html)、Gymnasium 的 [自定义环境接口](https://gymnasium.farama.org/introduction/create_custom_env/)；本项目将任务内预算耗尽建模为任务终止，明确区分外部采集截断。

导出 MP4（可选视频依赖；服务器渲染需要 EGL 或 OSMesa）：

```bash
uv pip install "imageio>=2.34" "imageio-ffmpeg>=0.5"
MUJOCO_GL=egl python -m minimal_recap.replay --run runs/gpu_seed1 --episode runs/gpu_seed1/expert/000000.npz --output runs/gpu_seed1/expert_rollout.mp4 --title "Expert rollout"
```

MP4 使用原始控制频率，H.264 编码；图像从保存的 MuJoCo 物理状态渲染，回放不会影响数据或策略。
