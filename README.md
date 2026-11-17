# MuJoCo Minimal RECAP · 3R

当前项目只实现一个任务：三关节机械臂从台座边缘侧抓水平长方体，抬起、旋转约90°，在桌面目标区竖直放置并松爪撤离。采用数值输入、真实接触、有限力矩和可恢复仿真状态。

动作4维 `[Δx, Δz, Δθ, 目标开度]`，默认观测40维。20 Hz控制、0.0025 s物理步长。夹爪总开度参考限速6 cm/s、限加速度0.4 m/s²。专家与学习策略共用动作接口和底层控制。数据/奖励版本为 `arm3r_upright_smooth_v1`。

- [当前任务与控制](docs/task.md)
- [详细研究设计及待实现事项](docs/arm3r_design.md)
- [数据格式](docs/data_format.md)
- [验证记录](docs/validation.md)

## Linux GPU服务器与uv

在Git仓库根目录执行；以下使用Python 3.12，PyTorch后端由uv检测服务器选择。先完成GPU驱动配置。

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --torch-backend auto -r requirements.txt
uv pip install --python .venv/bin/python --no-deps -e .
uv pip install --python .venv/bin/python 'imageio>=2.34' 'imageio-ffmpeg>=0.5'
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()"
python -m pytest -q
```

标准MuJoCo物理采集在CPU进行，PyTorch训练可用GPU。训练无需显示器或OpenGL；仅导出视频需要EGL等渲染后端。未实测本项目在服务器上的CUDA吞吐量，同环境可复现，不保证跨硬件逐位一致。

## 配置

所有当前参数集中在 `configs/base.yaml`，实验覆盖文件位于 `configs/experiments/`。

| 配置 | 用途 |
|---|---|
| arm3r.yaml | 当前3R任务，CPU训练 |
| arm3r_cuda.yaml | 同一任务，GPU训练 |
| smoke.yaml | 6条专家、6回合rollout、各4次优化的工程冒烟测试 |
| fixed_compare.yaml | 固定数据的基线比较 |
| online.yaml | 3轮在线迭代 |
| aggregate_bc.yaml / success_bc.yaml | 聚合BC / 成功轨迹BC |
| bc_only.yaml / unconditioned.yaml | 初始BC / 无条件监督更新 |
| n1.yaml / hard_bins.yaml / no_expert.yaml / noisy.yaml | n步、分布标签、专家保留、探索消融 |
| recap.yaml | 默认RECAP实验入口 |

正式默认预算：100条成功专家、每轮200回合策略数据、策略与评估300步；BC、Value和每轮更新均40 epoch×50更新，batch 256。更新额外采样256条专家，`expert_ratio=0.25`是损失权重。以上是试运行起点，正式多种子比较需先完成ID预算审计。

```bash
python -u -m minimal_recap.run --config configs/experiments/smoke.yaml
```

## 分步执行

下面各阶段使用同一个run目录与完整配置；新实验换目录。`-u`使日志及时输出，`-m`运行Python模块。`--resume`支持新建指定目录，也支持复用已完成阶段。

```bash
python -u -m minimal_recap.run --config configs/experiments/arm3r_cuda.yaml --resume runs/arm3r_gpu_seed1 --stages physics
python -u -m minimal_recap.run --config configs/experiments/arm3r_cuda.yaml --resume runs/arm3r_gpu_seed1 --stages expert
python -u -m minimal_recap.run --config configs/experiments/arm3r_cuda.yaml --resume runs/arm3r_gpu_seed1 --stages freeze
python -u -m minimal_recap.run --config configs/experiments/arm3r_cuda.yaml --resume runs/arm3r_gpu_seed1 --stages bc
python -u -m minimal_recap.run --config configs/experiments/arm3r_cuda.yaml --resume runs/arm3r_gpu_seed1 --stages rollout
python -u -m minimal_recap.run --config configs/experiments/arm3r_cuda.yaml --resume runs/arm3r_gpu_seed1 --stages value
python -u -m minimal_recap.run --config configs/experiments/arm3r_cuda.yaml --resume runs/arm3r_gpu_seed1 --stages label
python -u -m minimal_recap.run --config configs/experiments/arm3r_cuda.yaml --resume runs/arm3r_gpu_seed1 --stages update
python -u -m minimal_recap.run --config configs/experiments/arm3r_cuda.yaml --resume runs/arm3r_gpu_seed1 --stages next_rollout
python -u -m minimal_recap.run --config configs/experiments/arm3r_cuda.yaml --resume runs/arm3r_gpu_seed1 --stages evaluate
python -u -m minimal_recap.run --config configs/experiments/arm3r_cuda.yaml --resume runs/arm3r_gpu_seed1 --stages diagnose
python -u -m minimal_recap.run --config configs/experiments/arm3r_cuda.yaml --resume runs/arm3r_gpu_seed1 --stages compare
```

`compare`要求策略rollout中有足够成功轨迹供成功BC划分训练/验证；不足时显式报错，不用专家冒充策略成功。固定数据比较优先于正式在线实验。用CUDA运行其他实验，可创建只覆盖`model.device: cuda`的YAML。

每次运行保存resolved.yaml、manifest.json、代码哈希、随机种子、数据/奖励contract和阶段标记。修改配置或奖励需要新run。专家没有任务时限；程序异常记aborted，已有快照可续采。当前不承诺SIGKILL或断电后恢复尚未落盘的回合。

| 阶段 | 主要产物，位于run目录内 |
|---|---|
| physics | physics.json |
| expert | expert/*.npz |
| freeze | contract.json、expert_lengths.json |
| bc | checkpoints/bc.pt及训练日志 |
| rollout | rollouts/round_00/*.npz、behavior.json |
| value | checkpoints/value_00.pt |
| label | labels/round_00/*.npz、manifest.json |
| update | checkpoints/conditioned_00.pt |
| next_rollout / online | rollouts/round_01/及后续轮次 |
| evaluate / compare | evaluation/<method>/、metrics.json、recovery_pool.json |
| diagnose | diagnostics/diagnostics.json、trajectory.png、offsets.png、trace.npz |

## 视频、仪表盘与多种子

重新采集当前闭环专家并录制真实MuJoCo视频，输出目录必须为空：

```bash
MUJOCO_GL=egl python -u -m minimal_recap.demo --config configs/experiments/arm3r.yaml --seed 10000 --output runs/arm3r_demo
MUJOCO_GL=egl python -m minimal_recap.replay --run runs/arm3r_gpu_seed1 --episode runs/arm3r_gpu_seed1/expert/000000.npz --output runs/arm3r_gpu_seed1/expert.mp4
python -m minimal_recap.dashboard
```

仪表盘在本机`127.0.0.1:8765`只读显示实验、3R数值示意回放、训练曲线与Value/Return/Advantage。示意图不代替MuJoCo视频。服务器访问可通过SSH端口转发。

```bash
python scripts/run_seeds.py --config configs/experiments/arm3r_cuda.yaml --seeds 1 2 3 --stages physics expert freeze bc rollout value label compare diagnose
python -m minimal_recap.aggregate --runs runs/seed1 runs/seed2 runs/seed3 --output runs/summary.json
```

多种子脚本会打印真实run目录，汇总时替换示例目录。默认共享场景种子，只改变训练随机性。可选CPU部署脚本`server.sh`与`recap.slurm`也使用当前3R任务。

当前只验证了物理任务、软件闭环与小预算冒烟流程，未完成正式多种子性能实验，也未证明RECAP优于BC。恢复池的物理有效性不代表所有掉落状态都可恢复；完整组合OOD及更严格协议见设计文档的待实现条目。runs与虚拟环境不纳入Git，不提供源码包。
