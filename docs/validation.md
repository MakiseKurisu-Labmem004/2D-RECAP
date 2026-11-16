# 验证记录

本地验证日期：2026-11-16。平台：Windows、Python 3.12、CPU；服务器入口已提供，Linux/CUDA/Slurm 尚未实际运行。安装初期最新版二进制被 Windows 应用控制阻止，固定到 requirements.txt 中稳定版本后正常运行。

## 已实际验证

- pytest：11 passed。包含真实 MuJoCo 接触抓取（自由转动/锁定转动）、归一化动作与快照重放、成功和任务预算同一步时成功优先、任务 timeout 奖励、aborted 续采、MC/n-step 终止逻辑、201-bin 插值与越界拒绝、奖励顺序、四种监督训练与训练恢复。
- 物理门禁：ID、far_far、11–12 cm、13–14 cm 各12条专家轨迹，48/48成功。三个 smoke seed 重复的是同一套物理场景，不能计作144个独立场景。
- 六条专家成功示范长度：129、141、140、133、139、143步；全部完整保留。M=200、C_fail=200、value支持[-399,0]。
- 三个训练种子1/2/3均完成：physics → expert → freeze → BC → 冻结BC rollout → value → label → conditioned update → 新策略 rollout → ID/recovery评估 → 偏移及优势排序诊断。
- 每个训练种子仅6条示范、BC/value/recap各2 epoch、每epoch仅2次优化、每种策略2条ID测试、每类恢复2个状态。这些极小预算只用于验证数据链路。
- 三个种子的初始BC与conditioned策略ID成功率均为0，空抓/掉落恢复成功率也为0。没有观察到策略性能改善，不能据此推断正式训练的效果。
- 已生成专家回放GIF，检查了运输中的方块双指接触和越墙画面。Matplotlib输出偏移比较图和Value/Return/Advantage曲线。
- 汇总程序已对三个不同训练种子运行，结果见 smoke_summary.json。固定数据的四方法训练由集成测试验证；正式固定数据比较尚未运行，因为 smoke BC 没有采集到成功轨迹，success-only BC 无法定义有效训练/验证划分。

## 待在服务器完成的实验

1. 用正式示范数量与训练预算运行BC，查看专家长度统计和独立ID验证；如需改变共同任务预算，建立新实验版本后重新冻结奖励。
2. 在同一冻结BC数据集运行compare，保证success-only基线有足够成功轨迹；不得为这一基线偷偷补专家rollout。
3. 检查Value held-out误差、偏移续跑回报、等长前缀优势排序，不预设几何错位的价值顺序。
4. 固定数据比较后运行online覆盖配置，再按相同场景/剩余预算比较恢复和OOD。
5. 多训练种子统计，保留各次run配置、包版本、源代码hash、checkpoint、数据及图表。OOD不能参与checkpoint选择。

## 解释边界

物理门禁通过只证明这些有限场景的闭环专家可行；不证明学习策略已经学会任务。服务器脚本和CUDA路径属于已实现、未在服务器验证的功能。当前实际结果是工程冒烟测试，尚无RECAP优于BC、抓取纠错有效或OOD泛化有效的实验证据。
