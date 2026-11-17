# 当前3R任务验证记录

日期：2026-11-17。环境：Windows、Python 3.12、MuJoCo 3.3.7、CPU。版本arm3r_upright_smooth_v1。

## 实际验证

- 当前27项回归测试均通过，覆盖接触抓取与稳定释放、关节与力矩限制、IK、开度平滑、状态恢复、任务超时/成功优先、aborted、MC与n步优势、分布bins、训练及checkpoint恢复、仪表盘数据和本地HTTP边界。HTTP测试需允许访问本机回环地址。
- 15个YAML配置全部能解析，并指向同一个3R任务、四维动作接口。
- 现有237步专家视频轨迹用清理后的代码逐步重算，40维观测最大差异为0，成功结果一致。
- 当前物理抽检ID、短悬伸、高台座各12/12专家成功，长度234–244、235–246、238–247步。报告：runs/arm3r_gripper_smoothing_check/physics.json。
- 当前视频：runs/arm3r_smooth_gripper_demo/rollout.mp4，237步仿真（11.85秒）、238帧、20fps。无掉落，一次抓取成功。
- 完整3R冒烟流程已跑通：physics、expert、freeze、bc、rollout、value、label、update、next_rollout、evaluate、diagnose。目录runs/arm3r_smoke_validation。
- 此冒烟运行采集6条专家，长度237、235、237、244、236、242步；C_fail=300、支持[-599,0]。每个网络只训练4次优化、每种策略仅2个ID场景，是工程验证。

- initial_bc: ID 0%（2回合）；空抓恢复 0%，掉落恢复 0%。
- conditioned: ID 0%（2回合）；空抓恢复 0%，掉落恢复 0%。

仪表盘已使用40维特征映射、四维动作、台座和旋转长方体及3R关节链示意。前端JavaScript语法检查通过。

## 尚未验证

没有完成正式多种子BC/RECAP性能实验、完整组合OOD、所有边界布局或全部掉落池的可恢复性验证。没有证明RECAP优于BC。Linux/CUDA训练吞吐量尚未实测。当前的物理成功率和极小预算训练成绩不能互相替代。

正式实验前按详细设计完成预算审计、数据划分和公平比较；只使用ID验证选择checkpoint，OOD不参与选择。
