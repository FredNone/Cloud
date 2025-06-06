# 仿真参数
SIMULATION_CONFIG = {
    'num_tasks': 50,          # 任务数量
    'num_robots': 10,         # 机器人数量
    'workspace_size': 100,    # 工作空间大小
    'task_complexity': {      # 任务复杂度范围
        'min': 1,
        'max': 10
    },
    'robot_capabilities': {   # 机器人能力范围
        'min': 1,
        'max': 10
    },
    'network': {             # 网络参数
        'bandwidth': 1000,    # Mbps
        'latency': 1,        # ms
        'packet_size': 1024,  # bytes
        'topology': 'mesh'    # 网络拓扑类型
    },
    'communication': {       # 通信参数
        'overhead': 0.1,     # 通信开销系数
        'max_connections': 5  # 每个机器的最大连接数
    }
}

# 算法参数
ALGORITHM_CONFIG = {
    'population_size': 50,    # 种群大小
    'max_iterations': 100,    # 最大迭代次数
    'w': 0.7,                # 惯性权重 (PSO)
    'c1': 2.0,               # 个体学习因子 (PSO)
    'c2': 2.0,               # 社会学习因子 (PSO)
    'a': 2.0,                # GWO参数
    'load_balance': {        # 负载均衡参数
        'bandwidth_weight': 0.3,    # 带宽权重
        'capacity_weight': 0.4,     # 处理能力权重
        'connection_weight': 0.3    # 连接数权重
    }
}