# 数据格式：arm3r_upright_stable_target_no_relative_v2

每条轨迹一个NPZ，使用allow_pickle=False读取，原子替换写入。按完整轨迹划分训练/验证，来源由metadata.source标识专家、BC、RECAP轮次、评估或恢复诊断。

| 字段 | 形状/语义 |
|---|---|
| observation / next_observation | T×34，float32 |
| action | T×4，加噪并裁剪后提交给环境的归一化命令 |
| reward | T；终止奖励覆盖普通步 |
| terminated | T；成功、不可恢复失败或任务期限终止 |
| truncated | T；外部截断不得直接作为完整MC |
| metadata | JSON：episode_id、source、seed、scene、status、reason、success、checkpoint、控制频率、探索设置、版本与contract |
| states | T+1条完整积分状态、场景、管理计数器、随机数状态、控制器状态与预算 |
| controller_states | 专家T+1条状态机状态；策略的随机行为恢复信息另存metadata |

默认观测顺序：gripper 9、object 7、goal 4、pedestal 5、geometry 3、arm 6。实际顺序由observation.features决定。单位为米、秒、弧度，姿态含sin/cos；不包含时间、阶段或未来结果。

`action[-1]`是绝对开度请求，不是实测开度。`states[i].gripper_motion`保存target、reference（均m）和velocity（m/s）；`arm_controller`保存joint_targets与ik_scale。MuJoCo的mjSTATE_INTEGRATION还包含ctrl、warmstart、仿真时间等。它们用于续跑，不追加到网络输入。缺少必要控制状态时拒绝恢复。

status=complete只表示完整任务终止；程序异常/取消保存为aborted。完整MC拒绝aborted、非终止或truncated数据。作业强杀前尚未落盘的回合不保证可恢复。

专家失败尚无冻结的C_fail，其终止奖励占位0仅用于审计。BC和专家保留项只读成功专家动作，默认价值训练拒绝专家来源。价值使用预算一致的完整策略rollout，包含成功、失败和任务超时。

`behavior.json`记录checkpoint哈希、探索规则、动作处理、开度平滑参数、阻尼与冻结状态。每轮rollout共享冻结的contract。修改场景分布/预算/奖励/动力学需要新版本与新run，禁止混用不同定义。

`labels/round_N/*.npz`保存Value、MC Return、n步Advantage；manifest关联价值checkpoint与n。快照只在对应配置和MuJoCo版本下恢复。PyTorch checkpoint仅加载可信本地文件。

当前默认观测删除了夹爪相对物体、物体相对目标和目标姿态误差共6维。普通BC/Value输入34维，条件策略另外拼接1维条件。数据版本已更新，40维模型权重和归一化统计不能直接用于新输入；使用新run重新采集和训练。历史文件按其自身resolved配置解析，不自动改写。奖励版本为arm3r_upright_stable_target_v1。

当前成功定义为桌面目标区内稳定竖直，不要求张爪或撤离。成功时刻与旧定义不同，即使每步奖励数值相同，MC回报和优势标签也不同；必须按新的数据/奖励版本采集并冻结contract，不与旧判定的数据混用。

v2将台座高度12–19 cm、悬伸3.0–4.6 cm、目标x=12–20 cm作为统一ID范围，独立均匀采样，所有组合均可用于训练。评估只产生id分组与恢复诊断；数据版本变更用于区分旧采样协议，奖励和成功规则沿用stable_target_v1。旧轨迹、checkpoint及恢复池不自动混入新run，需重新采集并冻结contract。

训练日志`checkpoints/*.pt.history.json`每条记录包含epoch、gradient_step、train_loss和validation_loss。gradient_step是已完成的优化器更新次数，train_loss是当前epoch更新块内小批量loss的均值；validation_loss来自留出的完整轨迹。只有实际执行策略验证的记录才有id_validation及可选的validation_checkpoint（相对checkpoints目录）。不将未验证的步填成0成功率。

`initial_bc.yaml`每50步记录一次loss并保存续跑状态，每2500步验证100个固定ID场景并保存`bc_steps/step_NNNNNN.pt`，各模型元数据包含gradient_step与对应id_validation。`bc.pt`为验证选出的最佳模型，`bc.pt.training.pt`为最新续跑状态。`plots/bc/`导出loss.png、success_rate.png和curves.csv；图中成功率为百分比，JSON/CSV中为0–1。最终独立测试另存metrics.json，不混入用于选模型的验证曲线。
