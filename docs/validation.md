# 统一 ID 分布验证

日期：2026-11-17。本地环境：Windows、Python 3.12、MuJoCo 3.3.7，物理检查与单元测试使用CPU。默认实验训练配置仍为CUDA。

数据版本：arm3r_upright_stable_target_no_relative_v2。奖励版本：arm3r_upright_stable_target_v1。成功规则：environment.success_rule=stable_target。

## 当前实际检查

- 统一范围：台座高12–19 cm、悬伸3.0–4.6 cm、目标x=12–20 cm，三个参数独立均匀采样，全部组合均可用于训练。评估仅保留id；恢复池单列诊断。
- 30项相关测试通过，命令为`python -m pytest -q tests/test_upright.py tests/test_arm3r.py tests/test_success.py tests/test_contracts.py tests/test_gripper_motion.py`。覆盖真实抓取、旋转和稳定放置，简化成功规则、连续保持计数、关节/力矩/平滑器、状态恢复、MC/优势和三种更新方法。
- 其中8个物理测试覆盖高度、悬伸量、目标位置上下界的全部交叉组合；均成功，并在成功后额外开爪观察40步验证稳定性。这些是边界物理检查，不是额外评估分组。
- 14份YAML均解析为统一范围、suites=[id]、34维无相对观测和CUDA默认。旧独立套件名或非空scenario_overrides会明确报错。
- 每个阶段抽取64个场景种子检查采样覆盖：expert、rollout、validation、evaluation均包含高台座、短悬伸及二者组合；这是reset采样检查，不是256条任务成功率实验。
- 使用正式arm3r配置执行physics阶段，12/12条专家成功，长度198–210步。记录位于runs/arm3r_all_id_check/physics.json，同目录保存resolved.yaml、manifest.json和阶段记录。
- 仪表盘移除预置外推选项，JavaScript语法检查通过。现有历史run仍按自己的记录显示，不修改历史指标。

成功只表示桌面目标区内稳定竖直，允许夹爪仍接触或夹持；不代表已经独立释放。之前较窄采样范围的物理检查保存在runs/arm3r_stable_target_check/，它属于数据v1，不作为当前统一分布的正式结果。

## 解释边界

这些是有限场景的物理检查和小规模训练路径测试，不能证明连续范围内所有状态都可解。尚未完成统一分布下正式多训练种子的BC、Value、RECAP实验或服务器CUDA性能验证，未证明RECAP优于BC或所有掉落状态都可恢复。场景分布已变更，需新建run重新采专家并冻结contract后执行正式比较；奖励规则没有变化，不改写历史数据或标签。

## initial_bc长训练入口与曲线验证

新增configs/experiments/initial_bc.yaml：50,000次梯度更新、batch 256，每2,500步验证100个固定ID场景，共20个评估点。本机PyTorch为2.7.1+cpu，尚未执行正式CUDA的50,000步实验。

- 24项相关测试通过，命令为`python -m pytest -q tests/test_training_schedule.py tests/test_contracts.py tests/test_upright.py`。覆盖训练中断后续跑、评估间隔不漂移、历史不重复、最佳与最终checkpoint区分、未验证处保持缺失值，以及MC/优势和各训练路径。
- 用相同随机种子比较中断续跑与连续训练，最终模型所有参数和缓冲区逐位相同；已完成训练再次调用不追加评估点。
- 真实小预算BC流程保存在runs/initial_bc_curve_check/：6条成功专家、12次更新，在第6/12步各评估2个ID场景，均为0/2成功。这仅验证记录和导出，不用于判断BC性能。
- plots/bc/loss.png、success_rate.png和curves.csv已生成并检查；独立training_plots命令可从日志重新生成。正式配置的20个验证步为2500、5000、…、50000，loss日志每50步记录一次。
- 仪表盘可按gradient_step显示训练横轴，并兼容仅含epoch的旧日志。JavaScript语法检查通过。
