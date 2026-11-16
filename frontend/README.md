# RECAP 实验观测台

在仓库根目录运行（Python 3.10+，无需安装前端或研究依赖）：

```powershell
python frontend/serve.py
```

打开 http://127.0.0.1:8042 。按 Ctrl+C 停止。

支持场景空间分布、数据划分和难度筛选、步数分布、紧凑侧栏轨迹导航、搜索及步数排序、关节状态回放、点击即时切换、上下方向键浏览、时间轴拖动、播放速度、CSV 和单条轨迹 JSON 下载，以及配置查看。

回放采用保存的 qpos，包含初始 q0；两根连杆各 0.5 m。控制步长来自 timestep × substeps。距离曲线由末端坐标计算，间隙曲线来自 NPZ 的 clearance（每个控制步内的最小间隙）。专家成功不能作为策略成功率。页面不虚构训练曲线、Value、Advantage 或策略评测。

```powershell
python frontend/serve.py --run runs/seed42 --port 8042
```

服务仅监听本机，实验文件只读。新增或替换轨迹后，点击页面“刷新目录数据”即可重新建立索引，无需重启服务。划分与数量来自 scenes.json 和实际 NPZ，难度配额来自 difficulty.json，生成统计完整读取 generation.json。前端不依赖 CDN。NPZ 使用标准库读取非 Fortran 排序的数值 / Unicode 数组，无需 NumPy。

默认展示 seed42。支持 expert / BC 来源筛选、同场景专家与 BC 轨迹切换；“训练与评测”页从 *.history.json 和 evaluation*.json 读取曲线和分难度结果。checkpoint 只列计数，不读取模型权重。迁移观测字段来自 migration.json。

seed42 的 data.spec.json、training.*.json 与 *.goal_anchors.json 会同步读取；评测区分任务成功和严格成功，并显示完成时长、仅宽容阈值成功及近目标停滞。回放计时使用记录的 executed_substeps × physics_timestep（缺少时使用 control_dt），保持真实控制步对应关系。

支持 improvement/<实验>/<分支>/round-*：轨迹集合下拉框切换基础数据、自主练习及专家纠正各轮；改进实验页显示基线、候选/生效策略、gate 接纳、数据预算与 Value/RECAP 训练记录。pooled_practice 是练习池评测，不作为独立测试表现。纠正片段优先使用 initial_qpos，场景参数优先使用嵌入的 scene_json。

逐帧 Value / Advantage：支持原始 NPZ 标注和 frontend/.cache/value 下的展示缓存。使用训练环境（需 torch、numpy）执行 `python frontend/annotate_values.py --run runs/seed42` 可从每轮 value.pt 补算对应新采集轨迹的 V、n-step TD Advantage 与 MC Return；只写展示缓存，不改原始 NPZ。记录轨迹与权重 SHA256，任一改变则拒绝使用旧缓存。标注 t 对齐 observation[t]/action[t]，回放末帧 T 没有动作标注；基础 BC/Expert 不自动借用其他轮次 critic。
