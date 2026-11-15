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
