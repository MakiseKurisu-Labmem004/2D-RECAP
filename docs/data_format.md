# 数据格式（v1）

每条轨迹一个 npz（allow_pickle=False），写入临时文件后原子替换。不同来源目录分开，metadata.source 再显式标识 expert、bc、recap_N、evaluation、recovery、diagnostic。完整回合划分 train/validation，训练 checkpoint 保留 episode ID。

| 字段 | 形状/语义 |
|---|---|
| observation | T×D，float32；不含时间、阶段、未来结果 |
| action | T×3，执行前最终归一化裁剪动作 |
| reward | T；成功专家可用，失败专家审计占位0，禁止进入 value 数据 |
| next_observation | T×D |
| terminated | T；成功、不可恢复失败、任务预算终止 |
| truncated | T；正常任务轨迹均false；外部截断不进 MC |
| metadata | JSON unicode scalar：episode_id、source、seed、scene、status、reason、success、checkpoint、control_hz、探索设置、数据/奖励版本与 contract（policy rollout） |
| states | JSON，T+1 条 mjSTATE_INTEGRATION 快照，包含 ctrl、warmstart、仿真时间等，场景、环境计数器、RNG、预算 |
| controller_states | JSON，专家 T+1 条阶段/计数/试抬起点；策略为空，随机行为状态在 aborted metadata.behavior_rng |

默认 D=22，顺序由 observation.features 决定：

1. gripper（6）：x,z,vx,vz,opening,opening_velocity；
2. object（7）：x,z,vx,vz,sin(angle),cos(angle),angular_velocity；
3. goal（2）：x,z；
4. wall（3）：中心x、宽、高；
5. relative（4）：gripper-object x/z、object-goal x/z。

单位：米、秒、弧度。场景参数仅包含布局与几何；管理计数器存在快照/日志，永不拼接到 observation。

status=complete 只有完整终止回合；status=aborted 是程序异常或人工取消，reason 记录异常类型/内容，不是普通任务失败。MC 函数拒绝 aborted、非终止或任何 truncated 数据。程序外部截断选择保存状态续采，没有实现将截断残片冒充 MC 的路径。

专家失败没有冻结的 C_fail，因此 reward 中失败终止占位0仅用于审计；初始 BC 与后续专家保留项只读成功专家 action，价值训练明确拒绝专家来源。contract 冻结后每轮 rollout 都保存相同定义。修改上界/预算/奖励后用新 dataset/reward version 创建新实验，不能在原 run 改 contract。

labels/round_N 每条 npz 保存 Value、MC Return、n-step Advantage，manifest 关联价值 checkpoint 与 n。训练与测试快照只可在同配置、同 MuJoCo 版本恢复。torch checkpoint 是可信本地文件，勿加载不可信外部 pt。
