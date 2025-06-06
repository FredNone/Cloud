# 多机器人任务分配优化算法比较

本项目实现了四种不同的优化算法来解决多机器人任务分配问题：
1. 混合PSO-GWO算法 (Hybrid Particle Swarm Optimization - Grey Wolf Optimizer)
2. PSO算法 (Particle Swarm Optimization)
3. GWO算法 (Grey Wolf Optimizer)
4. SAE算法 (Simulated Annealing Evolution)

## 问题描述

给定一组任务和机器人，每个任务有其复杂度，每个机器人有其能力值。目标是找到最优的任务分配方案，使得：
1. 最小化任务到机器人的总距离
2. 最小化任务复杂度与机器人能力的不匹配度

## 项目结构

```
.
├── main.py              # 主程序
├── config.py            # 配置文件
├── requirements.txt     # 依赖包
├── models/
│   └── environment.py   # 环境模型
├── algorithms/
│   ├── hybrid_pso_gwo.py
│   ├── pso.py
│   ├── gwo.py
│   └── sae.py
└── utils/
    └── visualization.py # 可视化工具
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行方法

```bash
python main.py
```

## 输出结果

程序会生成两个图形文件：
1. convergence.png - 显示各算法的收敛过程
2. task_allocation.png - 显示最优算法（Hybrid PSO-GWO）的任务分配结果

## 参数配置

可以在config.py中修改以下参数：
- 仿真参数：任务数量、机器人数量、工作空间大小等
- 算法参数：种群大小、最大迭代次数、算法特定参数等 